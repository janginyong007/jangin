"""무음 구간 / 필러워드 / 반복 발화(버벅임)를 감지해서 컷 구간을 계산한다."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, List, Optional, Tuple

from .transcribe import Word

if TYPE_CHECKING:
    from .audio_energy import Energy

Span = Tuple[float, float]

_PUNCT_RE = re.compile(r"[.,!?…~\"'　\s]+")


def normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text).lower()


# 반복 감지 전용: 조사가 붙었는지 여부로 같은 말인데 다른 토큰이 되는 걸 막는다.
# 예: 다시 말할 때 뒤에 숫자가 더 붙으면 "보증금은"(한 토큰) -> "보증금"+"2억원은"
# (두 토큰)으로 위스퍼가 다르게 쪼개서, 완전 반복인데도 단어가 다르게 보였다.
_TRAILING_PARTICLES = ("은", "는", "이", "가", "을", "를", "의", "에", "도", "만", "과", "와", "로")


def _repeat_key(text: str) -> str:
    norm = normalize(text)
    for p in _TRAILING_PARTICLES:
        if len(norm) > len(p) and norm.endswith(p):
            return norm[: -len(p)]
    return norm


_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_DIGITS_RE = re.compile(r"\d+")


def _jamo(text: str) -> str:
    """한글을 자모로 풀어쓴다 ("감정가" -> "ㄱㅏㅁㅈㅓㅇㄱㅏ").
    발음이 꼬인 테이크는 위스퍼 인식 결과도 한두 글자 달라지기 쉬운데
    ("감종가" vs "감정가"), 음절 단위로는 완전히 다른 글자여도 자모
    단위로는 대부분 같아서 '비슷한 반복'을 잡을 수 있다."""
    out = []
    for ch in text:
        c = ord(ch) - 0xAC00
        if 0 <= c < 11172:
            out.append(_CHO[c // 588])
            out.append(_JUNG[(c % 588) // 28])
            if c % 28:
                out.append(_JONG[c % 28])
        else:
            out.append(ch)
    return "".join(out)


def _is_fuzzy_repeat(first: str, second: str, threshold: float, min_syllables: int) -> bool:
    """first(먼저 말한 것)가 second의 앞부분을 비슷하게 다시 말한 것인지.
    숫자가 다르면 절대 반복으로 보지 않는다 - "1억 원입니다" / "2억 원입니다"는
    자모로는 거의 같지만 전혀 다른 정보다."""
    if len(first) < min_syllables or not second:
        return False
    if _DIGITS_RE.findall(first) != _DIGITS_RE.findall(second[: len(first) + 2]):
        return False
    ja, jb = _jamo(first), _jamo(second)
    n = len(ja)
    best = 0.0
    for m in {n, int(n * 0.85), int(n * 1.15)}:  # 두 번째 테이크 길이가 조금 달라도 허용
        if m <= 0:
            continue
        sm = SequenceMatcher(None, ja, jb[:m], autojunk=False)
        if sm.quick_ratio() * 100 < threshold:
            continue
        best = max(best, sm.ratio() * 100)
    return best >= threshold


def _spoken_syllables(text: str) -> float:
    """말로 읽을 때 대략 몇 음절인지. 숫자는 한 자리당 약 1.5음절로 본다
    ("2013년" -> 이천십삼년, 약 7음절)."""
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    digits = sum(len(d) for d in _DIGITS_RE.findall(text))
    return hangul + digits * 1.5


def _plausible_rate(ws: List[Word], max_rate: float) -> bool:
    """이 단어들을 그 시간 안에 사람이 실제로 말할 수 있는 속도인지.
    위스퍼가 한 번 말한 걸 여러 번 받아적는 오류를 내면("2013년 2013년 2013년"을
    1초 안에) 가짜 반복들이 비현실적으로 짧은 시간에 몰린다. 그걸 진짜
    반복으로 믿고 자르면 실제로 말한 부분이 잘려나간다."""
    dur = ws[-1].end - ws[0].start
    syl = _spoken_syllables("".join(w.text for w in ws))
    return dur > 0 and syl / dur <= max_rate


def _word_similar(ja: str, jb: str, threshold: float) -> bool:
    if not ja or not jb or ja[0] != jb[0]:
        return False
    if ja == jb:
        return True
    return SequenceMatcher(None, ja, jb, autojunk=False).ratio() * 100 >= threshold


def find_silence_gaps(words: List[Word], total_duration: float, min_gap: float) -> List[Span]:
    spans: List[Span] = []
    if not words:
        return [(0.0, total_duration)] if total_duration >= min_gap else []

    if words[0].start >= min_gap:
        spans.append((0.0, words[0].start))

    for a, b in zip(words, words[1:]):
        gap = b.start - a.end
        if gap >= min_gap:
            spans.append((a.end, b.start))

    tail = total_duration - words[-1].end
    if tail >= min_gap:
        spans.append((words[-1].end, total_duration))

    return spans


def find_filler_words(words: List[Word], filler_words: List[str], max_duration: float) -> List[Span]:
    filler_set = {normalize(w) for w in filler_words}
    spans: List[Span] = []
    for w in words:
        if normalize(w.text) in filler_set and (w.end - w.start) <= max_duration:
            spans.append((w.start, w.end))
    return spans


def find_repeated_phrases(
    words: List[Word],
    window: float,
    max_ngram: int = 12,
    max_gap_words: int = 4,
    fuzzy_threshold: float = 0.0,
    fuzzy_min_syllables: int = 4,
    gap_min_words: int = 3,
    gap_min_syllables: int = 6,
    max_syllable_rate: float = 10.0,
) -> List[Span]:
    """동일 어절(구)이 짧은 시간 안에 반복되면 앞쪽(먼저 말한 쪽)을 컷 후보로 표시.

    두 반복 사이에 최대 max_gap_words개의 다른 단어가 끼어있어도 같은 반복으로
    인정한다 - 발음이 마음에 안 들어 다시 말할 때 "모텔 임대료는" 같은 짧은
    도입구를 새로 붙이고 나머지는 그대로 반복하는 경우가 실제로 있었다
    (예: "29억 원대에 낙찰받는다고 계산하면 2.5%대입니다." 뒤에 "모텔 임대료는"을
    붙여 같은 문장을 다시 반복). 그 사이에 낀 단어는 컷하지 않고 남긴다 - 새로
    추가된 맥락일 수 있으므로. 단어 1개짜리 반복(간투사 중복 등)은 우연의 일치를
    피하기 위해 간격 없이 바로 붙어있을 때만 잡는다.

    비교는 조사를 뗀 형태(_repeat_key)로 한다 - 다시 말할 때 뒤에 말이 더
    붙으면 위스퍼가 같은 단어를 다르게 쪼갠다 (예: "보증금은"(한 토큰)을 다시
    말하면서 숫자를 덧붙이면 "보증금"+"2억원은"(두 토큰)이 됨) - 이러면 조사
    포함 완전 일치로는 절대 안 걸린다.

    fuzzy_threshold > 0이면 완전히 같지 않아도, 자모 단위 유사도가 이 값
    이상이면 반복으로 본다 - 발음이 꼬여서 다시 말한 경우 첫 테이크의 인식
    결과가 한두 글자 틀리기 때문 ("감종가는" / "감정가는"). 우연히 비슷한 말을
    잡지 않도록 2단어 이상 + fuzzy_min_syllables 음절 이상일 때만 적용하고,
    숫자가 다르면 제외한다."""
    spans: List[Span] = []
    n = len(words)
    norm = [_repeat_key(w.text) for w in words]
    jamo = [_jamo(t) for t in norm]
    i = 0
    while i < n:
        matched = False
        max_len = min(max_ngram, n - i)
        for glen in range(max_len, 0, -1):
            j = i + glen
            if j > n:
                continue
            ngram = norm[i:j]
            if not any(ngram):
                continue
            # 사이에 다른 말이 끼어있는 반복은 짧은 구절이면 인정하지 않는다.
            # "이 건물은 튼튼합니다. 이 건물은 1990년에…"처럼 같은 말로 시작하는
            # 서로 다른 문장에서 앞 문장의 "이 건물은"만 잘려나가는 일을 막기 위함.
            allow_gap = (glen >= 2 and glen >= gap_min_words
                         and len("".join(ngram)) >= gap_min_syllables)
            gap_range = range(0, max_gap_words + 1) if allow_gap else range(0, 1)
            for gap in gap_range:
                k = j + gap
                m = k + glen
                if m > n:
                    continue
                time_gap = words[k].start - words[j - 1].end
                if time_gap > window:
                    continue
                same = norm[k:m] == ngram
                # 첫 단어끼리도 비슷해야 한다. 안 그러면 전체 유사도가 높다는 이유로
                # 반복 바로 앞의 다른 말("그래서 이 물건은… / 이 물건은…"의 "그래서")까지
                # 같이 잘려나간다.
                if (not same and fuzzy_threshold > 0 and glen >= 2
                        and _word_similar(jamo[i], jamo[k], fuzzy_threshold)):
                    # 다시 말할 때 위스퍼가 단어를 다르게 쪼갤 수 있으니 한 단어 더 붙여서 비교
                    same = _is_fuzzy_repeat(
                        "".join(ngram), "".join(norm[k:min(n, m + 1)]),
                        fuzzy_threshold, fuzzy_min_syllables,
                    )
                if same and max_syllable_rate > 0 and not (
                        _plausible_rate(words[i:j], max_syllable_rate)
                        and _plausible_rate(words[k:m], max_syllable_rate)):
                    same = False  # 사람이 낼 수 없는 속도 -> 인식 오류로 보고 자르지 않음
                if same:
                    spans.append((words[i].start, words[j - 1].end))
                    i = k
                    matched = True
                    break
            if matched:
                break
        if not matched:
            i += 1
    return spans


def merge_spans(spans: List[Span]) -> List[Span]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def shrink_spans(spans: List[Span], pad_before: float, pad_after: float) -> List[Span]:
    """컷 구간을 양쪽에서 줄여서(=유지 구간을 늘려서) 말이 안 잘리게 한다.
    pad_before: 컷 시작 쪽 여유 (직전 단어의 '어미'가 잘리지 않도록 보호)
    pad_after: 컷 끝나는 쪽 여유 (다음 단어의 초반이 잘리지 않도록 보호)"""
    result = []
    for s, e in spans:
        ns, ne = s + pad_before, e - pad_after
        if ne > ns:
            result.append((ns, ne))
    return result


def compute_keep_spans(cut_spans: List[Span], total_duration: float, min_keep_duration: float) -> List[Span]:
    cuts = merge_spans(cut_spans)
    keep: List[Span] = []
    cursor = 0.0
    for cs, ce in cuts:
        cs = max(0.0, min(cs, total_duration))
        ce = max(0.0, min(ce, total_duration))
        if cs > cursor:
            keep.append((cursor, cs))
        cursor = max(cursor, ce)
    if cursor < total_duration:
        keep.append((cursor, total_duration))
    return [(s, e) for s, e in keep if e - s >= min_keep_duration]


def detect_cuts(
    words: List[Word],
    total_duration: float,
    cfg: dict,
    ai_spans: Optional[List[Span]] = None,
    energy: Optional["Energy"] = None,
) -> Tuple[List[Span], dict]:
    """전체 컷 감지 파이프라인. keep_spans와 통계(dict)를 반환.
    ai_spans: disfluency.py가 Claude로 찾아낸 추가 컷 구간 (말더듬/재시도 등,
    사전 매칭이나 완전 반복 감지로는 못 잡는 것들). 없으면 룰 기반 결과만 쓴다.
    energy: audio_energy.load_energy() 결과. 주면 컷 경계를 실제 소리 크기로
    보정해서 말끝/첫소리가 잘리지 않게 한다. 없으면 기존처럼 위스퍼 시각만 쓴다."""
    silence_spans = find_silence_gaps(words, total_duration, cfg["silence_min_gap"])
    filler_spans = find_filler_words(words, cfg["filler_words"], cfg["filler_max_duration"])
    repeat_spans = find_repeated_phrases(
        words,
        cfg["repeat_window"],
        max_ngram=cfg.get("repeat_max_ngram", 12),
        max_gap_words=cfg.get("repeat_max_gap_words", 4),
        fuzzy_threshold=cfg.get("repeat_fuzzy_threshold", 80),
        fuzzy_min_syllables=cfg.get("repeat_fuzzy_min_syllables", 4),
        gap_min_words=cfg.get("repeat_gap_min_words", 3),
        gap_min_syllables=cfg.get("repeat_gap_min_syllables", 6),
        max_syllable_rate=cfg.get("repeat_max_syllable_rate", 10.0),
    )
    ai_spans = ai_spans or []

    # 무음 컷은 앞뒤에 진짜 침묵(여유 공간)이 있어서 padding_before/after를
    # 넉넉하게(기본 0.6/0.15초) 줘도 안전하다 - 그 여유는 어차피 침묵을 갚아먹을
    # 뿐이다. 하지만 필러/반복/AI 컷은 바로 옆에 실제 말이 붙어있는 경우가 많아서
    # 같은 큰 padding을 쓰면 "자를 구간의 맨 앞/뒤 단어"가 그대로 살아남는
    # 문제가 생긴다 (실제로 "이 사이에서 세입자의 보증금은"을 컷했는데
    # padding_before=0.6이 그 구간의 첫 단어 "이"(0.4초)보다 커서, "이"만
    # 살아남아 다음 문장 앞에 "이 이 사이에서..."처럼 겹쳐 보이는 버그가 있었음).
    # 그래서 필러/반복/AI 컷에는 훨씬 작은 padding만 준다.
    silence_padded = shrink_spans(merge_spans(silence_spans), cfg["padding_before"], cfg["padding_after"])
    content_padded = shrink_spans(
        merge_spans(filler_spans + repeat_spans + ai_spans),
        cfg.get("content_padding_before", 0.05),
        cfg.get("content_padding_after", 0.05),
    )
    if energy is not None and cfg.get("energy_refine", True):
        from .audio_energy import refine_silence_spans, snap_content_spans

        # 위스퍼가 말끝을 일찍 끝났다고 잡아도, 실제 소리가 사그라든 뒤에만 컷을
        # 시작한다. padding과 '둘 중 더 안전한 쪽'을 쓰는 것이라 컷이 늘어나지는 않는다.
        silence_padded = refine_silence_spans(
            silence_padded, energy,
            tail_margin=cfg.get("energy_tail_margin", 0.12),
            head_margin=cfg.get("energy_head_margin", 0.08),
        )
        content_padded = snap_content_spans(content_padded, energy)
    merged = merge_spans(silence_padded + content_padded)
    keep_spans = compute_keep_spans(merged, total_duration, cfg["min_keep_duration"])

    total_cut = total_duration - sum(e - s for s, e in keep_spans)
    stats = {
        "silence_spans": len(silence_spans),
        "filler_spans": len(filler_spans),
        "repeat_spans": len(repeat_spans),
        "ai_spans": len(ai_spans),
        "merged_cut_spans": len(merged),
        "keep_spans": len(keep_spans),
        "total_duration": total_duration,
        "total_cut_seconds": total_cut,
        "kept_duration": total_duration - total_cut,
        # 리포트용: 말이 들어있는 컷(필러/반복/AI)이 각각 무엇을 잘랐는지
        "cut_details": _cut_details(words, filler_spans, repeat_spans, ai_spans),
    }
    return keep_spans, stats


def _words_in(words: List[Word], s: float, e: float) -> str:
    return "".join(w.text if w.text.startswith(" ") else " " + w.text
                   for w in words if s - 0.01 <= (w.start + w.end) / 2 <= e + 0.01).strip()


def _cut_details(words, filler_spans, repeat_spans, ai_spans) -> List[dict]:
    rows = []
    for kind, spans in (("필러워드", filler_spans), ("반복발화", repeat_spans), ("AI검토", ai_spans)):
        for s, e in spans:
            rows.append({
                "kind": kind, "start": s, "end": e,
                "text": _words_in(words, s, e),
                "after": _words_in(words, e + 0.01, e + 4.0),  # 바로 뒤에 이어지는 말 (살아남은 쪽)
            })
    return sorted(rows, key=lambda r: r["start"])
