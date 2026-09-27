"""컷 이후 새 타임라인 기준으로 자막(SRT)을 생성한다."""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from .transcribe import Word

Span = Tuple[float, float]
Cue = Tuple[float, float, str]

_SENTENCE_END_CHARS = ".!?…"

# 이 어미/조사로 끝나는 단어 뒤는 의미 단위가 끊기는 자연스러운 지점으로 본다.
# 주의: "만"은 조사(-만큼만)로도, "1,683만"처럼 숫자 단위로도 쓰여서 목록에서 뺐다 -
# 이 프로젝트(경매 매물 시세 설명)는 숫자/금액이 아주 많이 나오는데, 단위로 쓰인
# "만"을 조사로 오인해서 가격 표현 중간에서 끊는 문제가 있었다.
_BREAK_SUFFIXES = (
    # 조사
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "에게", "께",
    "로", "으로", "와", "과", "도", "부터", "까지", "이나", "라도",
    "처럼", "같이", "만큼",
    # 연결어미
    "고", "며", "면서", "는데", "은데", "지만", "서", "니까", "으니까",
    "라서", "어서", "아서", "게", "도록",
)

# 콤마/숫자로 시작하는 단어는 직전 단어의 숫자 표현이 이어지는 것으로 본다
# (예: "11억" 다음의 "1,683만", "3,800원입니다." - 가격 하나를 자막 중간에서
# 쪼개지 않기 위해 이런 경우엔 하드캡을 조금 더 넉넉하게 허용한다).
_NUMBER_CONTINUATION_RE = re.compile(r"^[\d,]")


def _is_number_continuation(word_text: str) -> bool:
    stripped = word_text.strip()
    return bool(stripped) and bool(_NUMBER_CONTINUATION_RE.match(stripped))


def _ends_with_break_suffix(word_text: str) -> bool:
    stripped = word_text.strip()
    if not stripped:
        return False
    return stripped.endswith(_BREAK_SUFFIXES)


def build_remap(keep_spans: List[Span]) -> Tuple[Callable[[float], Optional[float]], float]:
    """원본 시각 -> 컷 이후 새 타임라인 시각으로 변환하는 함수를 만든다.
    구간 밖(컷된 부분)의 시각이 들어오면 None을 반환한다."""
    offsets = _build_offsets(keep_spans)
    cum = offsets[-1][1] - offsets[-1][0] + offsets[-1][2] if offsets else 0.0

    def remap(t: float) -> Optional[float]:
        span = _find_span(offsets, t)
        if span is None:
            return None
        s, e, off = span
        return off + (t - s)

    return remap, cum


def _build_offsets(keep_spans: List[Span]) -> List[Tuple[float, float, float]]:
    offsets = []
    cum = 0.0
    for s, e in keep_spans:
        offsets.append((s, e, cum))
        cum += e - s
    return offsets


def _find_span(offsets: List[Tuple[float, float, float]], t: float) -> Optional[Tuple[float, float, float]]:
    for s, e, off in offsets:
        if s <= t <= e:
            return (s, e, off)
    return None


def _map_word(offsets: List[Tuple[float, float, float]], w: Word) -> Optional[Tuple[float, float]]:
    """단어 중심 시각이 속한 구간을 기준으로 시작/끝을 매핑한다.
    (패딩으로 인해 컷 구간 가장자리에 아주 작은 유지 조각이 남을 수 있어,
    단어의 시작/끝 시각만 보면 컷된 단어인데도 걸치는 것처럼 보일 수 있다 -
    그래서 항상 단어의 중심 시각으로 소속 구간을 정하고, 시작/끝은 그 구간 안으로 clamp한다.)"""
    mid = (w.start + w.end) / 2
    span = _find_span(offsets, mid)
    if span is None:
        return None
    s, e, off = span
    start = off + (max(s, min(w.start, e)) - s)
    end = off + (max(s, min(w.end, e)) - s)
    return start, end


def group_into_lines(
    words: List[Word],
    keep_spans: List[Span],
    max_chars_per_line: int,
    max_lines: int,
    max_cue_duration: float = 6.0,
    break_margin: int = 6,
    hard_break_at_span: bool = False,
    orig_spans_out: Optional[List[Span]] = None,
) -> List[Cue]:
    """컷에 살아남은 단어들을 자막 줄 단위로 묶는다.
    max_chars_per_line*max_lines(목표 길이)를 넘기면, 그 이후로는 조사/어미로
    끝나는 첫 지점(의미 단위 경계)에서 끊는다. break_margin을 넘어서면
    적당한 경계가 없어도 강제로 끊는다.

    hard_break_at_span=True면 keep_spans 경계가 바뀔 때마다 문장부호와
    무관하게 항상 자막을 끊는다. 메인 자동편집(무음/필러 컷)은 이웃한
    구간끼리 같은 문장이 자연스럽게 이어지는 경우가 많아서 기존 동작(길이/
    문장부호 기준)을 그대로 둬야 하지만, 숏폼 하이라이트처럼 구간끼리
    서로 멀리 떨어진 별개 내용일 때는 이 옵션을 켜야 한다 - 안 그러면 한
    구간의 말 끝과 다음 구간의 말 시작이 한 자막 줄로 이어붙어 뜻이 안 통하는
    문장("보여집니다 있다고 보시면" 같은)이 만들어질 수 있다.

    orig_spans_out: 넘겨주면, 각 자막 줄에 대응하는 "원본(컷 전) 타임라인" 구간을
    cues와 같은 순서로 여기에 채워 넣는다 (disfluency.py가 컷 이후 결과를
    번호로 검토하고, 그 번호를 다시 원본 시각으로 되돌릴 때 씀)."""
    offsets = _build_offsets(keep_spans)
    soft_target = max_chars_per_line * max_lines
    hard_cap = soft_target + break_margin
    cues: List[Cue] = []
    cur_words: List[Word] = []
    cur_starts: List[float] = []
    cur_ends: List[float] = []
    prev_span: Optional[Tuple[float, float, float]] = None

    def flush():
        if not cur_words:
            return
        # 단어 텍스트가 원본의 공백 유무를 그대로 담고 있으므로 그냥 이어붙인다
        # (콤마/조사 앞에 " ".join으로 인한 불필요한 공백이 생기지 않도록).
        text = "".join(w.text for w in cur_words).strip()
        cues.append((cur_starts[0], cur_ends[-1], text))
        if orig_spans_out is not None:
            orig_spans_out.append((cur_words[0].start, cur_words[-1].end))
        cur_words.clear()
        cur_starts.clear()
        cur_ends.clear()

    for w in words:
        mid = (w.start + w.end) / 2
        span = _find_span(offsets, mid)
        if span is None:
            continue  # 컷된 단어
        if hard_break_at_span and prev_span is not None and span != prev_span:
            flush()
        prev_span = span

        mapped = _map_word(offsets, w)
        if mapped is None:
            continue  # 컷된 단어
        new_start, new_end = mapped

        # 이 단어를 더했을 때 하드캡을 넘기면, 더하기 전에 먼저 끊는다.
        # (다 더한 뒤에 검사하면 긴 단어 하나 때문에 하드캡을 훌쩍 넘어버릴 수 있음 -
        # 실제로 이 문제로 22자짜리 자막이 나온 적이 있었음.)
        # 다만 이 단어가 직전 단어의 숫자 표현을 잇는 중이면(콤마/숫자로 시작),
        # "11억" / "1,683만" / "3,800원" 같은 가격 표현이 중간에서 잘리지 않도록
        # 하드캡을 조금 더 넉넉하게 봐준다.
        effective_hard_cap = hard_cap + 10 if _is_number_continuation(w.text) else hard_cap
        prospective_len = sum(len(x.text) for x in cur_words) + len(w.text)
        if cur_words and prospective_len > effective_hard_cap:
            flush()

        cur_words.append(w)
        cur_starts.append(new_start)
        cur_ends.append(new_end)

        cur_len = sum(len(x.text) for x in cur_words)
        duration = cur_ends[-1] - cur_starts[0]
        stripped = w.text.rstrip()
        is_sentence_end = bool(stripped) and stripped[-1] in _SENTENCE_END_CHARS

        if is_sentence_end:
            flush()
        elif duration > max_cue_duration:
            flush()
        elif cur_len >= soft_target and _ends_with_break_suffix(w.text):
            flush()

    flush()
    return cues


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(cues: List[Cue], path: str) -> None:
    lines = []
    for i, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp(start)} --> {_format_timestamp(end)}")
        lines.append(text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_subtitles(
    words: List[Word], keep_spans: List[Span], subtitle_cfg: dict, out_path: str,
    hard_break_at_span: bool = False,
    orig_spans_out: Optional[List[Span]] = None,
) -> List[Cue]:
    """자막 생성 전체 과정을 실행하고 참고용 SRT를 저장한 뒤, 자막 큐 목록을 반환."""
    cues = group_into_lines(
        words,
        keep_spans,
        subtitle_cfg.get("max_chars_per_line", 16),
        subtitle_cfg.get("max_lines", 2),
        break_margin=subtitle_cfg.get("break_margin", 6),
        hard_break_at_span=hard_break_at_span,
        orig_spans_out=orig_spans_out,
    )
    write_srt(cues, out_path)
    return cues
