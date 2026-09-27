"""Claude API로, 1차 컷(무음/필러/완전반복)이 끝난 최종 자막을 번호가 붙은 줄
단위로 검토해서, 문맥상 여전히 남아있는 재시도(retake)를 찾는다.

처음엔 Claude에게 원본 전체 대본을 보여주고 임의의 시작/끝 시각을 직접
만들어내게 했었는데, 그러면 경계를 잘못 그려서 71초짜리 구간을 통째로
"재시도"라고 잘못 잡아버리는 등 위험한 결과가 실제로 나왔다(실측 확인됨).
그래서 방식을 바꿨다: 이미 1차로 컷하고 자막까지 만든 "번호가 붙은" 최종
대본을 보여주고, 그 중 "삭제할 줄 번호"만 고르게 한다. 번호에 대응하는
정확한 원본 시각은 우리가 이미 알고 있으므로(subtitles.py의 orig_spans_out),
경계 하락(잘못된 시각 지어내기)이 원천적으로 불가능하다. 그래도 혹시
모를 판단 실수에 대비해 연속 줄 수 제한과 최대 길이 안전장치를 둔다."""
from __future__ import annotations

import json
from typing import Callable, List, Optional, Tuple

import anthropic

from .subtitles import Cue
from .utils import PipelineError

Span = Tuple[float, float]
LogFn = Callable[[str], None]


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


SYSTEM_PROMPT = """당신은 부동산 경매 나레이션 영상의 편집자입니다. 아래는 무음/필러/
완전반복 컷이 1차로 끝난 최종 자막입니다 (번호. [시각] 내용 형식입니다).

이 대본을 처음부터 끝까지 읽으면서, 문맥상 여전히 남아있는 재시도(retake)를
찾아주세요: 바로 이어지는 두 줄(또는 몇 줄)이 같은 사실/주장을 전달하고
있다면 - 표현이나 단어가 달라도 - 재시도입니다. 그 중 **먼저 나온 시도의
줄 번호만** 삭제 대상으로 표시하세요 (마지막 시도는 남겨야 합니다).

예시: "3. 이사비협의와 명도기간은 처음부터 개선할 수 있습니다." 다음
"4. 이사비협의와 명도기간은 처음부터 개선해 넣으셔야 합니다."가 왔다면,
3번을 삭제 대상으로 표시하세요 (4번은 남김).

규칙:
- 삭제 대상은 반드시 **제공된 목록에 있는 번호만** 쓰세요. 새로운 시각이나
  존재하지 않는 번호를 지어내지 마세요.
- 서로 다른 정보(다른 숫자, 다른 대상)를 말하는 줄은 절대 삭제 대상에 넣지
  마세요 - 문장 구조가 비슷해도 내용이 다르면 재시도가 아닙니다.
- 한 번에 삭제 대상으로 묶는 연속 번호는 되도록 짧게(보통 1~3줄) 잡으세요.
  많은 줄을 통째로 삭제 대상으로 묶지 마세요 - 그 자체가 판단이 잘못됐다는
  신호입니다.
- 확신이 없으면 포함시키지 마세요.
- 삭제할 게 없으면 빈 배열을 제출하세요.

반드시 mark_lines_to_remove 도구를 호출해서 결과를 제출하세요. 다른 설명
텍스트는 필요 없습니다."""

_REMOVE_TOOL = {
    "name": "mark_lines_to_remove",
    "description": "재시도로 판단해 삭제해야 할 자막 줄 번호 묶음들을 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "removals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_numbers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "삭제할 연속된 줄 번호들 (예: [3] 또는 [3,4])",
                        },
                        "reason": {"type": "string", "description": "삭제 이유(한국어 한 줄)"},
                    },
                    "required": ["line_numbers"],
                },
            },
        },
        "required": ["removals"],
    },
}


def _build_numbered_script(cues: List[Cue]) -> str:
    lines = [f"{i}. [{_fmt_ts(start)}-{_fmt_ts(end)}] {text}" for i, (start, end, text) in enumerate(cues, start=1)]
    return "\n".join(lines)


def _parse_removals(tool_use) -> Optional[list]:
    """highlight.py의 관용적 파싱과 같은 패턴 - 정상 배열이면 그대로, 문자열로
    이중 인코딩됐으면 풀어서 시도한다."""
    raw = tool_use.input.get("removals")
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    result = parsed.get("removals") if isinstance(parsed, dict) else parsed
    return result if isinstance(result, list) else None


def review_cut_script(
    cues: List[Cue],
    orig_spans: List[Span],
    api_key: str,
    model: str,
    max_tokens: int,
    log: LogFn = print,
    max_consecutive_lines: int = 4,
    max_removal_duration: float = 15.0,
) -> List[Span]:
    """1차 컷이 끝난 cues를 번호로 검토해서, 문맥상 남은 재시도 구간을 원본
    시각(Span) 목록으로 돌려준다. len(cues) == len(orig_spans)여야 한다.
    실패해도(API 오류 등) 파이프라인 전체를 죽이지 않고 빈 리스트를 돌려준다 -
    이 기능은 1차 컷(무음/필러/완전반복) 위에 얹는 추가 안전망이기 때문."""
    if not cues:
        return []
    if len(cues) != len(orig_spans):
        raise PipelineError("cues와 orig_spans의 길이가 다릅니다 (내부 오류).")
    if not api_key:
        log("      Claude API 키가 없어 최종 검토를 건너뜁니다 (secrets.yaml 확인).")
        return []

    script_text = _build_numbered_script(cues)
    log(f"      Claude로 편집된 대본 최종 검토 중... (자막 {len(cues)}줄)")
    client = anthropic.Anthropic(api_key=api_key)

    max_attempts = 3
    raw_removals = None
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                tools=[_REMOVE_TOOL],
                tool_choice={"type": "tool", "name": "mark_lines_to_remove"},
                messages=[{"role": "user", "content": script_text}],
            )
        except anthropic.APIError as e:
            log(f"      Claude API 호출 실패, 최종 검토를 건너뜁니다: {e}")
            return []

        if response.stop_reason == "max_tokens":
            last_error = "응답이 max_tokens 제한에 걸려 중간에 끊김"
        else:
            tool_use = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
            if tool_use is None:
                last_error = "응답에서 tool_use를 찾지 못함"
            else:
                raw_removals = _parse_removals(tool_use)
                if raw_removals is None:
                    last_error = f"removals를 해석하지 못함: {tool_use.input}"

        if raw_removals is not None:
            break
        if attempt < max_attempts:
            log(f"      Claude 응답이 이상해서 다시 시도합니다 ({attempt}/{max_attempts}회차): {last_error}")

    if raw_removals is None:
        log(f"      최종 검토 {max_attempts}번 실패, 이 단계는 건너뜁니다: {last_error}")
        return []

    n = len(cues)
    spans: List[Span] = []
    for item in raw_removals:
        if not isinstance(item, dict):
            continue
        try:
            line_numbers = sorted({int(x) for x in item.get("line_numbers", [])})
        except (TypeError, ValueError):
            continue
        line_numbers = [ln for ln in line_numbers if 1 <= ln <= n]
        if not line_numbers:
            continue
        # 연속된 번호인지 확인 (건너뛴 번호가 섞여 있으면 존재하지 않는 조합을
        # 지어냈을 가능성이 있어 신뢰하지 않는다).
        if line_numbers[-1] - line_numbers[0] + 1 != len(line_numbers):
            log(f"      [건너뜀] 연속되지 않은 줄 번호 조합이라 무시: {line_numbers}")
            continue
        if len(line_numbers) > max_consecutive_lines:
            log(f"      [건너뜀] 한 번에 {len(line_numbers)}줄이나 삭제하려는 건 의심스러워서 무시: {line_numbers}")
            continue

        first_idx, last_idx = line_numbers[0] - 1, line_numbers[-1] - 1
        orig_start = orig_spans[first_idx][0]
        orig_end = orig_spans[last_idx][1]
        if orig_end - orig_start > max_removal_duration:
            log(f"      [건너뜀] 삭제 구간이 {orig_end - orig_start:.1f}초로 너무 길어서 무시 (안전장치)")
            continue

        reason = str(item.get("reason", "")).strip()
        display_text = " / ".join(cues[i][2] for i in range(first_idx, last_idx + 1))
        log(f"      [{_fmt_ts(orig_start)}-{_fmt_ts(orig_end)}] {reason or display_text}")
        spans.append((orig_start, orig_end))

    return spans
