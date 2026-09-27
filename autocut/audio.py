"""오디오 추출 / 에너지 분석 / 발화 구간 검출.

컷 경계는 Whisper 타임스탬프가 아니라 '실제 음량(에너지)'으로 정한다.
Whisper 단어 타임스탬프는 문장 끝(~습니다, ~거든요)의 약한 어미를
일찍 끝난 것으로 잡는 경향이 있어서, 그대로 자르면 말끝이 잘린다.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

SR = 16000          # 분석용 샘플레이트
HOP = 0.01          # 에너지 프레임 간격 (10ms)
WIN = 0.03          # 에너지 프레임 길이 (30ms)


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """ffmpeg로 영상/음성에서 mono float32 PCM 추출."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", path,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def energy_db(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """10ms 간격 RMS 에너지(dBFS)."""
    hop, win = int(sr * HOP), int(sr * WIN)
    n = max(1, 1 + (len(audio) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    idx = np.minimum(idx, len(audio) - 1)
    rms = np.sqrt(np.mean(audio[idx] ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


@dataclass
class Thresholds:
    noise_floor: float
    speech_on: float    # 이 이상이면 '말하는 중' 시작
    speech_off: float   # 이 아래로 내려가야 '말 끝남' (히스테리시스, 더 낮음)


def estimate_thresholds(db: np.ndarray, on_margin: float = 14.0,
                        off_margin: float = 7.0) -> Thresholds:
    """배경 소음 수준을 추정해 임계값을 자동 설정.

    말끝을 살리기 위해 '끝' 판정 임계값(off)을 시작 임계값(on)보다 낮게 둔다.
    """
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 95))
    # 녹음 레벨이 낮아도 동작하도록 on 임계값이 peak에 너무 붙지 않게 제한
    on = min(floor + on_margin, peak - 12.0)
    off = min(floor + off_margin, on - 3.0)
    return Thresholds(floor, on, off)


def speech_regions(db: np.ndarray, th: Thresholds, min_silence: float = 0.35,
                   min_speech: float = 0.12, hangover: float = 0.20):
    """에너지 기반 발화 구간 [(start_s, end_s), ...].

    - 시작: speech_on 초과
    - 끝: speech_off 아래로 떨어진 뒤 hangover 동안 다시 안 올라오면 종료
    - min_silence보다 짧은 쉼은 같은 구간으로 합침 (문장 중간 호흡에서 자르지 않음)
    """
    n = len(db)
    regions = []
    i = 0
    hang = int(hangover / HOP)
    while i < n:
        if db[i] < th.speech_on:
            i += 1
            continue
        # 시작점: on 초과 지점에서 off 아래가 될 때까지 뒤로 (자음 시작부 보호)
        s = i
        while s > 0 and db[s - 1] >= th.speech_off:
            s -= 1
        # 끝점: off 아래로 hang 프레임 이상 유지되는 곳
        j = i
        below = 0
        while j < n:
            if db[j] < th.speech_off:
                below += 1
                if below > hang:
                    break
            else:
                below = 0
            j += 1
        e = j - below
        regions.append([s * HOP, e * HOP + WIN])
        i = j
    # 짧은 쉼 병합
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] < min_silence:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_speech]


def quietest_point(db: np.ndarray, t0: float, t1: float) -> float:
    """[t0, t1] 사이에서 가장 조용한 시점 — 컷 위치를 여기로 맞추면 '툭' 끊기는 느낌이 줄어든다."""
    a, b = int(max(0, t0) / HOP), int(max(0, t1) / HOP)
    b = min(b, len(db))
    if b - a < 2:
        return (t0 + t1) / 2
    seg = db[a:b]
    # 최소값 근처(1.5dB 이내) 중 가운데 지점
    low = np.where(seg <= seg.min() + 1.5)[0]
    return (a + low[len(low) // 2]) * HOP
