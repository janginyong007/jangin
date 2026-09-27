"""실제 음량(에너지)으로 컷 경계를 보정한다 - 말끝 잘림 방지.

위스퍼 단어 타임스탬프는 "~습니다", "~거든요" 처럼 약하게 끝나는 어미를
실제보다 일찍 끝난 것으로 잡는 경향이 있다. 그 시각을 기준으로 무음 컷을
시작하면 padding을 줘도 말끝이 잘리는 경우가 생긴다 (말끝이 padding보다
길게 늘어지는 경우). 그래서 컷 경계를 실제 소리 크기로 한 번 더 확인한다:

- 무음 컷의 시작: 위스퍼 단어 끝에서부터 앞으로 훑어서, 소리가 '진짜로'
  배경 소음 수준까지 떨어진 지점 이후로만 컷을 시작한다.
- 무음 컷의 끝: 다음 단어 시작에서 거꾸로 훑어서, 소리가 올라오기
  시작한 지점(자음 첫소리) 이전에서 컷을 끝낸다.
- 반복/필러/AI 컷처럼 말 사이에 걸친 컷은 경계 근처에서 가장 조용한
  지점으로 옮겨서, 앞 단어 꼬리나 뒷 단어 첫소리를 먹지 않게 한다.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Span = Tuple[float, float]

HOP = 0.01  # 10ms 프레임
WIN = 0.03


@dataclass
class Energy:
    db: np.ndarray        # 프레임별 dBFS
    noise_floor: float
    speech_off: float     # 이 아래면 '말이 끝남'으로 본다
    speech_on: float = 0.0  # 이 이상이면 확실히 '말하는 중'

    def frame(self, t: float) -> int:
        return int(min(max(t, 0.0) / HOP, len(self.db) - 1))

    def time(self, i: int) -> float:
        return i * HOP


def load_energy(wav_path: str, off_margin: float = 7.0) -> Energy:
    """transcribe.extract_audio()가 만든 16kHz mono wav에서 에너지 계산."""
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x /= float(np.iinfo(dtype).max)
    return energy_from_samples(x, sr, off_margin)


def energy_from_samples(x: np.ndarray, sr: int, off_margin: float = 7.0) -> Energy:
    hop, win = int(sr * HOP), int(sr * WIN)
    n = max(1, 1 + (len(x) - win) // hop)
    # 프레임별 RMS (메모리 절약을 위해 누적합으로 계산)
    sq = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(n) * hop
    ends = np.minimum(starts + win, len(x))
    rms = np.sqrt((sq[ends] - sq[starts]) / np.maximum(ends - starts, 1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 95))
    # 말끝(약한 어미)도 '말하는 중'으로 보도록 소음보다 조금만 높게 잡는다.
    # 단, 녹음이 작아도 동작하도록 peak에서 너무 가깝지 않게 제한.
    off = min(floor + off_margin, peak - 15.0)
    on = max(off + 3.0, min(floor + 14.0, peak - 12.0))
    return Energy(db=db, noise_floor=floor, speech_off=off, speech_on=on)


def speech_end_after(en: Energy, t: float, limit: float, hang: float = 0.15) -> float:
    """t부터 앞으로 훑어 소리가 speech_off 아래로 hang초 이상 유지되기 시작하는 시각."""
    i, lim = en.frame(t), en.frame(limit)
    need = max(1, int(hang / HOP))
    quiet = 0
    while i < lim:
        if en.db[i] < en.speech_off:
            quiet += 1
            if quiet >= need:
                return en.time(i - quiet + 1)
        else:
            quiet = 0
        i += 1
    return en.time(lim - quiet) if quiet else limit


def speech_start_before(en: Energy, t: float, limit: float) -> float:
    """t부터 거꾸로 훑어 소리가 speech_off 이상으로 올라오기 시작한 시각 (첫소리 보호)."""
    i, lim = en.frame(t), en.frame(limit)
    while i > lim and en.db[i - 1] >= en.speech_off:
        i -= 1
    return en.time(i)


def quietest_point(en: Energy, t0: float, t1: float) -> float:
    a, b = en.frame(t0), en.frame(t1)
    if b - a < 2:
        return (t0 + t1) / 2
    seg = en.db[a:b]
    low = np.where(seg <= seg.min() + 1.5)[0]
    return en.time(a + int(low[len(low) // 2]))


def refine_silence_spans(spans: List[Span], en: Energy, tail_margin: float = 0.12,
                         head_margin: float = 0.08) -> List[Span]:
    """무음 컷 구간을 실제 소리 기준으로 줄인다 (늘리지는 않음 - 안전 방향).

    tail_margin: 말이 실제로 끝난 뒤 추가로 남길 여유
    head_margin: 다음 말이 실제로 시작하기 전 추가로 남길 여유"""
    out = []
    for s, e in spans:
        real_end = speech_end_after(en, s, e)
        real_start = speech_start_before(en, e, real_end)
        # 소리가 위스퍼 시각보다 실제로 더 이어졌을 때만 경계를 옮긴다
        # (이미 조용한 곳이면 기존 padding 결과를 그대로 둔다)
        ns = max(s, real_end + tail_margin) if real_end > s else s
        ne = min(e, real_start - head_margin) if real_start < e else e
        if ne > ns:
            out.extend(_split_around_speech(en, ns, ne, tail_margin, head_margin))
    return out


def loud_regions(en: Energy, t0: float, t1: float, min_len: float = 0.25) -> List[Span]:
    """[t0, t1] 안에서 확실히 말소리로 보이는 구간들 (speech_on 이상이 min_len초 이상)."""
    a, b = en.frame(t0), en.frame(t1)
    regions = []
    i = a
    while i < b:
        if en.db[i] >= en.speech_on:
            j = i
            while j < b and en.db[j] >= en.speech_off:
                j += 1
            # 앞쪽 약한 첫소리까지 포함
            k = i
            while k > a and en.db[k - 1] >= en.speech_off:
                k -= 1
            if (j - k) * HOP >= min_len:
                regions.append((en.time(k), en.time(j)))
            i = j
        else:
            i += 1
    return regions


def _split_around_speech(en: Energy, s: float, e: float, tail_margin: float,
                         head_margin: float) -> List[Span]:
    """무음 컷 안에 실제 말소리가 있으면 그 부분은 남기고 컷을 쪼갠다.
    위스퍼가 말을 통째로 빼먹고 받아적으면(가끔 있음) 그 구간이 '단어 사이
    무음'처럼 보여서 필요한 말이 잘려나가는데, 이를 막는 안전장치."""
    pieces = []
    cur = s
    for rs, re_ in loud_regions(en, s, e):
        if rs - head_margin > cur:
            pieces.append((cur, rs - head_margin))
        cur = max(cur, re_ + tail_margin)
    if e > cur:
        pieces.append((cur, e))
    return [(a, b) for a, b in pieces if b - a > 0.05]


def snap_content_spans(spans: List[Span], en: Energy, window: float = 0.12) -> List[Span]:
    """반복/필러/AI 컷 경계를 근처의 가장 조용한 지점(단어 사이 틈)으로 옮긴다."""
    out = []
    for s, e in spans:
        ns = quietest_point(en, s - window, s + window * 0.5)
        ne = quietest_point(en, e - window * 0.5, e + window)
        if ne > ns:
            out.append((ns, ne))
    return out
