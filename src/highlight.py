"""Claude API로 완성된 롱폼 영상의 대본을 읽고, 1분 이내 숏폼에 쓸 하이라이트
구간(들)을 자동으로 고른다."""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Tuple

import anthropic

from .transcribe import Word
from .utils import PipelineError

Span = Tuple[float, float]
LogFn = Callable[[str], None]

_SENTENCE_END_CHARS = ".!?…"
_PAUSE_BREAK = 0.8   # 이보다 긴 침묵이면 대본 줄을 새로 나눈다
_MAX_LINE_CHARS = 60

SYSTEM_PROMPT = """당신은 유튜브 쇼츠 편집자입니다. 이미 완성된 롱폼 영상의 대본을
타임스탬프와 함께 받습니다. 이 영상에서 가장 임팩트 있고 몰입감 있는 부분을 골라
60초를 넘지 않는 세로형 쇼츠 영상을 구성하는 것이 목표입니다.

규칙:
- 1~6개의 구간(clip)을 고르세요. 각 구간은 반드시 원본 영상의 시간 순서(오름차순)를
  지켜야 하고, 서로 겹치면 안 됩니다. (구간 순서를 뒤섞거나 재배열하지 마세요.)
- 각 구간의 start/end는 반드시 제공된 대본 줄의 타임스탬프 중에서 골라 그대로
  사용하세요 (임의의 시각을 지어내지 마세요).
- 전체 구간 길이의 합은 {target_duration}초를 넘으면 안 되지만, 너무 짧게 고르지
  마세요. 목표는 {target_duration}초에 최대한 가깝게(최소 {target_duration_min}초 이상)
  채우는 것입니다. 영상이 충분히 길다면 하이라이트가 될 만한 구간을 4~6개까지 적극적으로
  찾아서 담으세요. 구간이 2~3개뿐이라 목표 시간에 한참 못 미치는 것은 실패입니다.
- 우선순위: 놀라운 결과/숫자(가격, 낙찰가, 수익률 등)의 공개, 반전, 결정적 정보,
  감정적으로 몰입되는 순간, 실용적인 팁. 도입부는 시청자의 시선을 붙잡을 수 있는
  한 문장이면 좋습니다.
- 모든 구간은 반드시 문장이 끝나는 지점(마침표 등으로 끝나는 대본 줄)에서
  시작하고 끝내세요. 문장 중간에서 시작하거나 끝나면 안 됩니다.
- 특히 각 구간의 끝은 "~습니다.", "~있어요.", "~됐어요." 처럼 자연스럽게 맺는
  문장이어야 합니다. "그래서", "근데"처럼 뒤에 말이 더 이어질 것 같은 단어에서
  끝나면 안 됩니다.
- 각 구간마다 왜 골랐는지 한 줄 이유(reason)를 한국어로 적으세요.

반드시 select_highlight_clips 도구를 호출해서 결과를 제출하세요. 다른 설명 텍스트는 필요 없습니다.
"""

_CLIPS_TOOL = {
    "name": "select_highlight_clips",
    "description": "쇼츠에 사용할 하이라이트 구간 목록을 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number", "description": "구간 시작 시각(초)"},
                        "end": {"type": "number", "description": "구간 끝 시각(초)"},
                        "reason": {"type": "string", "description": "이 구간을 고른 이유(한국어 한 줄)"},
                    },
                    "required": ["start", "end"],
                },
            },
        },
        "required": ["clips"],
    },
}


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _build_transcript_lines(words: List[Word]) -> List[Tuple[float, float, str]]:
    """단어들을 문장/침묵 경계 기준으로 대본 줄 단위로 묶는다 (Claude에게 보여줄 용도)."""
    lines: List[Tuple[float, float, str]] = []
    cur: List[Word] = []

    def flush() -> None:
        if not cur:
            return
        text = "".join(w.text for w in cur).strip()
        if text:
            lines.append((cur[0].start, cur[-1].end, text))
        cur.clear()

    prev_end = None
    for w in words:
        if prev_end is not None and w.start - prev_end >= _PAUSE_BREAK:
            flush()
        cur.append(w)
        prev_end = w.end

        stripped = w.text.rstrip()
        cur_len = sum(len(x.text) for x in cur)
        if (stripped and stripped[-1] in _SENTENCE_END_CHARS) or cur_len >= _MAX_LINE_CHARS:
            flush()

    flush()
    return lines


def _build_transcript_text(lines: List[Tuple[float, float, str]]) -> str:
    return "\n".join(f"[{_fmt_ts(s)}-{_fmt_ts(e)}] {text}" for s, e, text in lines)


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise PipelineError(f"Claude 응답에서 JSON을 찾지 못했습니다:\n{text[:500]}")
        return json.loads(match.group(0))


def _is_sentence_end(word: Word) -> bool:
    stripped = word.text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_END_CHARS


def _snap_to_words(
    start: float, end: float, words: List[Word], total_duration: float,
    pad_before: float = 0.0, pad_after: float = 0.0,
) -> Span:
    """구간 경계를 실제 단어 시작/끝 시각에 맞춘다 (단어 중간이 잘리지 않도록).
    한국어 ASR은 단어의 '끝'(어미) 시각을 실제보다 이르게 추정하는 경향이 있어서
    (cutdetect.py의 padding_before/padding_after와 동일한 이유), 구간의 끝은
    pad_before만큼 더 늦게, 시작은 pad_after만큼 더 이르게 여유를 준다."""
    covering = [w for w in words if w.end > start and w.start < end]
    if not covering:
        return start, end
    new_start = max(0.0, covering[0].start - pad_after)
    new_end = min(total_duration, covering[-1].end + pad_before)
    return new_start, new_end


def _cap_before_next_word_midpoint(candidate_end: float, sentence_word_end: float, words: List[Word]) -> float:
    """pad_before를 더한 끝이 다음 단어의 '중심(midpoint)'을 넘지 않게 캡을 씌운다.
    자막/오디오 소속 판정은 midpoint 기준이라(subtitles.py 참고), 이렇게만 해도
    다음 단어 전체가 자막에 통째로 딸려 나오는 일은 절대 없다. 다음 단어의 '시작'
    지점에서 캡을 씌우던 버전(2026-07-25 최초 시도)은 문장 끝 60개 중 78%에서
    패딩이 사실상 0으로 깎여 말끝이 잘리는 문제를 만들었고, 반대로 캡을 아예
    없앤 버전은 다음 단어 전체가 자막 맨 끝에 딸려 나와("이 물건은" 같은 미완성
    문장으로 영상이 끝나 보이는 문제)를 만들었다. midpoint 캡은 그 중간 지점 -
    말끝 보호용 여유는 최대한 살리면서, 다음 단어가 자막에 끼어드는 것만은
    확실히 막는다."""
    next_word = next((w for w in words if w.start > sentence_word_end), None)
    if next_word is not None:
        next_midpoint = (next_word.start + next_word.end) / 2
        candidate_end = min(candidate_end, next_midpoint - 0.01)
    return candidate_end


def _snap_end_to_sentence(start: float, end: float, words: List[Word], pad_before: float,
                           search_forward: float = 10.0) -> float:
    """구간의 끝이 문장 중간에서 뚝 끊기지 않도록, 마침표 등으로 문장이 끝나는
    가장 가까운 단어의 끝으로 맞춘다. 반드시 구간 시작(start) 이후 범위에서만
    찾는다 - 그렇지 않으면 영상 앞부분의 무관한 문장 경계를 잘못 끌어와서
    구간을 엉뚱하게 뭉개버릴 수 있다. 구간 안에 없으면 end 이후 몇 초 안에서
    찾는다 (짧게 조금 늘려서라도 문장을 완결시키는 게 낫다)."""
    before = [w for w in words if start <= w.end <= end and _is_sentence_end(w)]
    if before:
        w = before[-1]
        return _cap_before_next_word_midpoint(w.end + pad_before, w.end, words)
    after = [w for w in words if end < w.end <= end + search_forward and _is_sentence_end(w)]
    if after:
        w = after[0]
        return _cap_before_next_word_midpoint(w.end + pad_before, w.end, words)
    return end  # 근처에 문장이 끝나는 지점이 없으면 원래 끝을 유지 (포기)


def _snap_start_to_sentence(start: float, end: float, words: List[Word], pad_after: float) -> float:
    """구간의 시작이 이전 문장의 중간이 아니라, 새 문장이 막 시작하는 지점이
    되도록 앞으로 당긴다 (직전 단어가 문장을 끝맺은 경우에만 시작점으로 인정).
    단, 그 지점이 end를 넘어서면(=구간 안에 문장 경계가 아예 없으면) 구간 전체를
    집어삼키게 되므로, 그럴 땐 원래 시작을 그대로 유지한다."""
    idx = next((i for i, w in enumerate(words) if w.start >= start), None)
    if idx is None:
        return start
    while idx < len(words) and words[idx].start < end and idx > 0 and not _is_sentence_end(words[idx - 1]):
        idx += 1
    if idx >= len(words) or words[idx].start >= end:
        return start  # 구간 안에서 문장 시작점을 못 찾으면 원래 시작을 유지 (포기)
    return max(0.0, words[idx].start - pad_after)


def _fit_end_to_budget(start: float, limit: float, words: List[Word], pad_before: float) -> Optional[float]:
    """예산 초과로 구간을 줄여야 할 때, limit을 넘지 않으면서 문장을 완결할 수
    있는 지점을 찾는다. 예산 안에 문장을 끝맺을 단어가 하나도 없으면, 아무
    단어에서나 어중간하게 자르지 말고 아예 포기한다(None) - 특히 영상의
    마지막 구간에서 "~습니다" 없이 뚝 끊긴 채로 끝나는 것보다는, 이 구간을
    통째로 빼고 그 앞 구간에서 깔끔하게 끝나는 게 낫다."""
    candidates = [w for w in words if w.start >= start and w.end + pad_before <= limit]
    sentence_end_candidates = [w for w in candidates if _is_sentence_end(w)]
    if not sentence_end_candidates:
        return None
    last_word = max(sentence_end_candidates, key=lambda w: w.end)
    return _cap_before_next_word_midpoint(last_word.end + pad_before, last_word.end, words)


_MIN_CLIP_DURATION = 1.5  # 이보다 짧은 구간은 실제 하이라이트로 보기 어려움 (Claude가 가끔 보내는
                          # 의미없는 placeholder성 짧은 구간 제거용)


def _clean_clips(raw_clips: List[dict], words: List[Word], total_duration: float,
                  target_duration: float, pad_before: float, pad_after: float,
                  log: LogFn) -> List[Tuple[float, float, str]]:
    clips: List[Tuple[float, float, str]] = []
    for c in raw_clips:
        try:
            start = float(c["start"])
            end = float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        # Claude가 가끔 진짜 이유 대신 "placeholder" 같은 더미 값을 보낼 때가
        # 있다 - 실제로 두 번 관측된 패턴이라 방어적으로 걸러낸다.
        if str(c.get("reason", "")).strip().lower() == "placeholder":
            continue
        start = max(0.0, min(start, total_duration))
        end = max(0.0, min(end, total_duration))
        if end - start < _MIN_CLIP_DURATION:
            continue
        start, end = _snap_to_words(start, end, words, total_duration, pad_before, pad_after)
        if end - start < _MIN_CLIP_DURATION:
            continue
        # 대사가 문장 중간에서 끊기지 않도록, 시작은 새 문장이 막 시작하는 지점으로
        # 당기고 끝은 문장이 끝나는(마침표 등) 지점으로 맞춘다. (반드시 이 구간
        # 범위 안에서만 찾아야 영상의 다른 부분과 뒤섞이지 않는다.)
        start = _snap_start_to_sentence(start, end, words, pad_after)
        end = _snap_end_to_sentence(start, end, words, pad_before)
        end = min(end, total_duration)
        if end - start < _MIN_CLIP_DURATION:
            continue
        clips.append((start, end, str(c.get("reason", "")).strip()))

    clips.sort(key=lambda c: c[0])

    cleaned: List[Tuple[float, float, str]] = []
    cursor = 0.0
    for start, end, reason in clips:
        if start < cursor:
            # 앞 구간과 겹쳐서 시작을 뒤로 밀어야 한다면, 그냥 숫자로 자르지 말고
            # 그 안에서 다시 문장 시작점을 찾는다 (안 그러면 문장 중간부터 시작될 수 있음).
            # pad_after만큼 살짝 당겨주는 여유 때문에 다시 cursor보다 앞으로 갈 수
            # 있으니, 마지막으로 한 번 더 cursor 밑으로 못 내려가게 고정한다.
            start = max(_snap_start_to_sentence(cursor, end, words, pad_after), cursor)
        if end - start < _MIN_CLIP_DURATION:
            continue
        cleaned.append((start, end, reason))
        cursor = end

    budget = target_duration + 5.0
    result: List[Tuple[float, float, str]] = []
    used = 0.0
    for start, end, reason in cleaned:
        remaining = budget - used
        if remaining <= 0.5:
            break
        limit = start + remaining
        if end > limit:
            end = _fit_end_to_budget(start, limit, words, pad_before)
            if end is None or end - start < _MIN_CLIP_DURATION:
                continue
        result.append((start, end, reason))
        used += end - start

    if not result:
        raise PipelineError("하이라이트 구간을 고르지 못했습니다. 영상 길이가 너무 짧거나 대본이 비어있을 수 있습니다.")

    for start, end, reason in result:
        log(f"      [{_fmt_ts(start)}-{_fmt_ts(end)}] {reason}" if reason else f"      [{_fmt_ts(start)}-{_fmt_ts(end)}]")

    return result


_ARITH_FIELD_RE = re.compile(r'("(?:start|end)":\s*)(-?\d+\.?\d*)\s*([+\-])\s*(-?\d+\.?\d*)')


def _repair_inline_arithmetic(text: str) -> str:
    """아주 가끔 Claude가 JSON 문자열 안의 숫자 자리에 "0.01+1.39" 같은 계산식을
    그대로 남겨서 JSON 파싱이 깨질 때가 있다. start/end 필드 안의 단순 덧셈/뺄셈만
    안전하게 계산해서 숫자로 치환한다 (임의 코드 실행 없이 정규식 범위 안에서만)."""

    def _eval(m: "re.Match[str]") -> str:
        prefix, a, op, b = m.groups()
        value = float(a) + float(b) if op == "+" else float(a) - float(b)
        return f"{prefix}{value}"

    return _ARITH_FIELD_RE.sub(_eval, text)


def _parse_tool_clips(tool_use) -> Optional[list]:
    """tool_use.input에서 clips 배열을 최대한 관대하게 뽑아낸다. 정상 배열이면
    그대로, 문자열로 이중 인코딩됐으면 풀어서, 그 안에 계산식이 남아있으면
    고쳐서 시도한다. 그래도 안 되면 None을 돌려줘서 호출부가 재시도하게 한다."""
    raw_clips = tool_use.input.get("clips")
    if isinstance(raw_clips, list):
        return raw_clips or None
    if not isinstance(raw_clips, str):
        return None

    for text in (raw_clips, _repair_inline_arithmetic(raw_clips)):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        result = parsed.get("clips") if isinstance(parsed, dict) else parsed
        if isinstance(result, list) and result:
            return result
    return None


def select_highlight_spans(
    words: List[Word],
    total_duration: float,
    api_key: str,
    model: str,
    target_duration: float,
    max_tokens: int,
    pad_before: float = 0.0,
    pad_after: float = 0.0,
    log: LogFn = print,
) -> List[Span]:
    if not api_key:
        raise PipelineError(
            "Claude API 키가 설정되지 않았습니다. secrets.yaml 파일을 열어 anthropic_api_key 값을 입력하세요."
        )
    if not words:
        raise PipelineError("대본이 비어있어 하이라이트를 고를 수 없습니다.")

    lines = _build_transcript_lines(words)
    transcript_text = _build_transcript_text(lines)

    log(f"      Claude로 하이라이트 구간 선정 중... (대본 {len(lines)}줄)")
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = SYSTEM_PROMPT.format(
        target_duration=int(target_duration), target_duration_min=int(target_duration * 0.8)
    )

    max_attempts = 3
    raw_clips = None
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=[_CLIPS_TOOL],
                tool_choice={"type": "tool", "name": "select_highlight_clips"},
                messages=[{"role": "user", "content": transcript_text}],
            )
        except anthropic.APIError as e:
            raise PipelineError(f"Claude API 호출 실패: {e}") from e

        if response.stop_reason == "max_tokens":
            last_error = "응답이 max_tokens 제한에 걸려 중간에 끊김"
        else:
            tool_use = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
            if tool_use is None:
                last_error = "응답에서 tool_use를 찾지 못함"
            else:
                raw_clips = _parse_tool_clips(tool_use)
                if raw_clips is None:
                    last_error = f"clips를 해석하지 못함: {tool_use.input}"

        if raw_clips is not None:
            break
        if attempt < max_attempts:
            log(f"      Claude 응답이 이상해서 다시 시도합니다 ({attempt}/{max_attempts}회차): {last_error}")

    if raw_clips is None:
        raise PipelineError(
            f"Claude 응답에서 유효한 하이라이트 구간을 {max_attempts}번 시도해도 받지 못했습니다: {last_error}"
        )

    clips = _clean_clips(raw_clips, words, total_duration, target_duration, pad_before, pad_after, log)
    total = sum(e - s for s, e, _ in clips)
    log(f"      하이라이트 {len(clips)}개 구간 선정 완료 (합계 {total:.1f}초)")

    return [(s, e) for s, e, _ in clips]
