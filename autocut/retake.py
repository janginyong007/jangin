"""반복 발화(다시 말하기) 검출.

유튜버가 발음이 꼬이거나 억양이 이상하면 같은 문장을 다시 말한다.
패턴은 크게 세 가지:

  1) 문장 통째로 다시      "이 물건은 감정가가 삼억입니다." (쉼) "이 물건은 감정가가 삼억입니다."
  2) 말하다 멈추고 처음부터  "이 물건은 감정가가 삼..." (쉼) "이 물건은 감정가가 삼억입니다."
  3) 쉼 없이 바로 고쳐 말하기 "이 물건은 감종, 감정가가 삼억입니다"

원칙: 항상 '뒤에 말한 테이크'를 살리고 앞의 것을 지운다.

비교는 한글을 자모(ㄱ ㅏ ㅁ ...)로 풀어서 한다.
발음이 꼬인 테이크는 인식 결과도 한두 글자 틀리기 쉬운데('감종가' vs '감정가'),
음절 단위보다 자모 단위가 이런 차이에 훨씬 관대하다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .transcribe import Utterance

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def normalize(text: str) -> str:
    """공백/문장부호 제거."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


def jamo(text: str) -> str:
    out = []
    for ch in normalize(text):
        c = ord(ch) - 0xAC00
        if 0 <= c < 11172:
            out.append(_CHO[c // 588])
            out.append(_JUNG[(c % 588) // 28])
            if c % 28:
                out.append(_JONG[c % 28])
        else:
            out.append(ch)
    return "".join(out)


def prefix_similarity(a: str, b: str) -> float:
    """a가 b의 '앞부분'과 얼마나 같은지 (0~100). 자모 단위."""
    ja, jb = jamo(a), jamo(b)
    if not ja or not jb:
        return 0.0
    # b 앞부분을 a 길이 근처(±15%)로 잘라 가장 잘 맞는 값
    best = 0.0
    n = len(ja)
    for m in {n, int(n * 0.85), int(n * 1.15)}:
        if m <= 0:
            continue
        best = max(best, fuzz.ratio(ja, jb[:m]))
    return best


def syllables(text: str) -> int:
    return len(normalize(text))


@dataclass
class Removal:
    start: float
    end: float
    reason: str
    text: str
    score: float


@dataclass
class RetakeConfig:
    threshold: float = 78.0        # 자모 유사도 기준 (낮출수록 공격적)
    min_syllables: int = 3         # 이보다 짧은 조각은 반복 판정 안 함 (오검출 방지)
    lookahead: int = 3             # 뒤로 몇 개 발화까지 비교할지
    max_gap: float = 15.0          # 이 시간(초) 안에 다시 말한 것만 반복으로 봄
    tail_max_gap: float = 5.0      # 문장 '뒷부분'만 겹칠 때는 더 짧은 간격만 허용
    tail_min_syllables: int = 5
    inline: bool = True            # 3) 쉼 없이 고쳐 말하기 검출
    inline_window: int = 8         # 한 번에 반복되는 최대 단어 수


def _removal_from_word(u: Utterance, k: int) -> float:
    """u.words[k]부터 지울 때의 시작 시각."""
    if k == 0:
        return u.start
    return (u.words[k - 1].end + u.words[k].start) / 2


def find_cross_retakes(utts: list[Utterance], cfg: RetakeConfig) -> list[Removal]:
    """발화 A 뒤에 A를 다시 말한 발화 B가 오면 A(또는 A의 뒷부분)를 제거."""
    removals = []
    for i, a in enumerate(utts):
        if not a.words or syllables(a.text) < cfg.min_syllables:
            continue
        best = None  # (k, score, j)
        for j in range(i + 1, min(len(utts), i + 1 + cfg.lookahead)):
            b = utts[j]
            gap = b.start - a.end
            if gap > cfg.max_gap:
                break
            if not b.words:
                continue
            # k=0: A 전체가 B의 앞부분 (패턴 1, 2)
            # k>0: A의 뒷부분만 B의 앞부분과 같음 → 문장 중간부터 다시 말함
            for k in range(len(a.words)):
                tail = " ".join(w.text for w in a.words[k:])
                n = syllables(tail)
                if k > 0 and (gap > cfg.tail_max_gap or n < cfg.tail_min_syllables):
                    break
                if n < cfg.min_syllables:
                    break
                sc = prefix_similarity(tail, b.text)
                if sc >= cfg.threshold:
                    if best is None or k < best[0]:
                        best = (k, sc, j)
                    break
            if best is not None and best[0] == 0:
                break
        if best:
            k, sc, j = best
            tail = " ".join(w.text for w in a.words[k:])
            kind = "문장 다시 말함" if k == 0 else "문장 중간부터 다시 말함"
            removals.append(Removal(_removal_from_word(a, k), a.end,
                                    f"{kind} (→ {utts[j].start:.1f}s 테이크 사용)",
                                    tail, sc))
    return removals


def find_inline_retakes(utts: list[Utterance], cfg: RetakeConfig) -> list[Removal]:
    """한 발화 안에서 쉼 없이 고쳐 말한 부분: '감종, 감정가가' → 앞의 '감종' 제거."""
    removals = []
    for u in utts:
        ws = u.words
        b = 1
        while b < len(ws):
            hit = None
            for a in range(max(0, b - cfg.inline_window), b):
                x = " ".join(w.text for w in ws[a:b])
                if syllables(x) < 2:
                    continue
                y = " ".join(w.text for w in ws[b:b + (b - a) + 2])
                sc = prefix_similarity(x, y)
                # 짧은 조각일수록 우연히 비슷할 수 있으니 기준을 올림
                need = cfg.threshold + (8 if syllables(x) < 4 else 0)
                if sc >= need:
                    hit = (a, sc, x)
                    break  # 가장 긴 반복(가장 앞의 a) 채택
            if hit:
                a, sc, x = hit
                s = _removal_from_word(u, a)
                e = (ws[b - 1].end + ws[b].start) / 2
                removals.append(Removal(s, e, "바로 고쳐 말함", x, sc))
            b += 1
    return _merge_overlaps(removals)


def _merge_overlaps(rs: list[Removal]) -> list[Removal]:
    rs = sorted(rs, key=lambda r: r.start)
    out: list[Removal] = []
    for r in rs:
        if out and r.start <= out[-1].end:
            p = out[-1]
            p.end = max(p.end, r.end)
            p.text = p.text + " / " + r.text
        else:
            out.append(r)
    return out


def find_retakes(utts: list[Utterance], cfg: RetakeConfig | None = None) -> list[Removal]:
    cfg = cfg or RetakeConfig()
    rs = find_cross_retakes(utts, cfg)
    if cfg.inline:
        rs += find_inline_retakes(utts, cfg)
    return sorted(rs, key=lambda r: r.start)
