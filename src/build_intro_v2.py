"""부동산장인 채널 인트로 v2 — 업로드된 실사 클립 5개(법원/도보/입찰봉투/판사봉/로고)를
이어붙여 7.5초짜리 인트로를 CapCut 초안으로 만든다.

각 클립은 AI로 생성된 영상으로, 이미 하나의 이야기로 이어지도록 되어 있다:
법원 건물 -> 서류 들고 걸어가는 사람 -> 입찰봉투 -> 판사봉 타격(불꽃) -> 금빛 파티클이
모여 "부동산장인" 로고로 완성.

앞의 세 장면(법원/도보/입찰봉투)은 맥락 설명이라 빠르게 넘기고, 감정적 정점인
판사봉 타격과 로고 완성 장면에 전체 시간의 대부분을 배정해 "감동"이 실릴 시간을 준다.
효과음은 무겁지 않되(서브베이스 없음) 타격엔 몸집 있는 펀치感을, 로고 직전엔 기대감을
쌓는 라이저를, 로고가 완전히 맺히는 순간엔 밝은 차임을 배치한다.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import yaml

import pycapcut as cc
from pycapcut import trange

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SFX_DIR = ASSETS_DIR / "sfx"

SOURCE_DIR = Path(r"E:\Down")
CLIPS = {
    "courthouse": SOURCE_DIR / "Use_the_uploaded_courthouse_im.mp4",
    "portrait": SOURCE_DIR / "Use_the_uploaded_full_body_por.mp4",
    "bidding": SOURCE_DIR / "Use_the_uploaded_auction_biddi.mp4",
    "gavel": SOURCE_DIR / "Create_an_ultra_cinematic_open.mp4",
    "logo": SOURCE_DIR / "Use_the_uploaded_logo_as_the_r.mp4",
}

WIDTH, HEIGHT = 1920, 1080
FPS = 30
SR = 48000

# (key, source_start, source_end, global_start, global_end)
SEGMENTS = [
    ("courthouse", 0.0, 0.5, 0.0, 0.5),
    ("portrait", 0.0, 0.5, 0.5, 1.0),
    ("bidding", 0.0, 0.5, 1.0, 1.5),
    ("gavel", 1.5, 3.5, 1.5, 3.5),   # 스윙 전체 + 타격(실제 "쾅" 소리는 원본 3.25s 지점)
    ("logo", 3.6, 7.6, 3.5, 7.5),    # 파티클 소용돌이 -> "부동산장인" 로고 완성 -> 정지 여운
]

CUT_POINTS = [0.5, 1.0, 1.5]   # 맥락 장면 빠른 컷 3곳 - 밝은 스윙
IMPACT_AT = 3.25               # 판사봉이 실제로 내려치는 순간 (원본 오디오 피크와 동일)
RISER_AT = 3.35                # 타격 직후 ~ 로고 소용돌이로 넘어가는 기대감 라이저
RESOLVE_CHIME_AT = 6.1         # 로고가 완전히 맺히는 음악적 정점


def _s(seconds: float) -> str:
    return f"{seconds}s"


def _write_wav(path: Path, stereo: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def make_swish(path: Path, dur: float = 0.12) -> None:
    """저음 없는 밝은 스윙 효과음 (1.5k~5.5kHz 대역만 남긴 노이즈)."""
    n = int(SR * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    noise = np.random.randn(n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(freqs < 1500) | (freqs > 5500)] = 0.0
    filtered = np.fft.irfft(spec, n)
    env = np.sin(np.pi * t / dur) ** 0.7
    sig = filtered * env
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.5
    _write_wav(path, np.stack([sig, sig], axis=1))


def make_chime(path: Path, dur: float = 0.7) -> None:
    """밝은 3화음 차임벨 (저음 없이 상승감 있는 소리) - 로고 완성 순간용."""
    n = int(SR * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    sig = np.zeros(n)
    for f in (1046.5, 1318.5, 1568.0):  # C6-E6-G6
        sig += np.sin(2 * np.pi * f * t)
    env = np.exp(-t * 4.0)
    sig = sig * env
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.6
    _write_wav(path, np.stack([sig, sig], axis=1))


def make_impact(path: Path, dur: float = 0.28) -> None:
    """판사봉 타격용 펀치感 있는 히트 - 서브베이스 없이 몸집(200Hz대)+브라이트 클릭."""
    n = int(SR * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)

    body_freq = 220.0 * np.exp(-t * 7.0)
    body_phase = 2 * np.pi * np.cumsum(body_freq) / SR
    body = np.sin(body_phase) * np.exp(-t * 16.0)

    noise = np.random.randn(n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(freqs < 2000) | (freqs > 8000)] = 0.0
    click = np.fft.irfft(spec, n) * np.exp(-t * 35.0)

    sig = body * 0.8 + click * 0.6
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.75
    _write_wav(path, np.stack([sig, sig], axis=1))


def make_riser(path: Path, dur: float = 0.6) -> None:
    """타격 -> 로고 소용돌이로 넘어가는 기대감 라이저 (저음 없이 점점 커지는 브라이트 노이즈)."""
    n = int(SR * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    noise = np.random.randn(n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(freqs < 1000) | (freqs > 6000)] = 0.0
    filtered = np.fft.irfft(spec, n)
    env = (t / dur) ** 1.5
    sig = filtered * env
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.4
    _write_wav(path, np.stack([sig, sig], axis=1))


def build_intro_draft(draft_name: str = "부동산장인_인트로_영상조합") -> str:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    draft_folder_path = cfg["capcut"]["draft_folder"]

    SFX_DIR.mkdir(parents=True, exist_ok=True)
    swish_path = SFX_DIR / "swish.wav"
    chime_path = SFX_DIR / "chime.wav"
    impact_path = SFX_DIR / "impact.wav"
    riser_path = SFX_DIR / "riser.wav"
    make_swish(swish_path)
    make_chime(chime_path)
    make_impact(impact_path)
    make_riser(riser_path)

    draft_folder = cc.DraftFolder(draft_folder_path)
    script = draft_folder.create_draft(draft_name, WIDTH, HEIGHT, fps=FPS, allow_replace=True)

    script.add_track(cc.TrackType.video)
    for key, s0, s1, g0, g1 in SEGMENTS:
        material = cc.VideoMaterial(str(CLIPS[key]))
        seg = cc.VideoSegment(
            material,
            target_timerange=trange(_s(g0), _s(g1 - g0)),
            source_timerange=trange(_s(s0), _s(s1 - s0)),
        )
        script.add_segment(seg)

    # 라이저가 임팩트음과 겹치므로(기대감을 미리 깔기 위해) 트랙을 분리한다.
    script.add_track(cc.TrackType.audio, track_name="효과음")
    script.add_track(cc.TrackType.audio, track_name="효과음_라이저")
    swish_material = cc.AudioMaterial(str(swish_path))
    chime_material = cc.AudioMaterial(str(chime_path))
    impact_material = cc.AudioMaterial(str(impact_path))
    riser_material = cc.AudioMaterial(str(riser_path))

    for cp in CUT_POINTS:
        start = max(0.0, cp - 0.05)
        seg = cc.AudioSegment(swish_material, target_timerange=trange(_s(start), _s(0.12)), volume=0.5)
        script.add_segment(seg, track_name="효과음")

    impact_seg = cc.AudioSegment(impact_material, target_timerange=trange(_s(IMPACT_AT), _s(0.28)), volume=0.85)
    script.add_segment(impact_seg, track_name="효과음")

    chime_seg = cc.AudioSegment(chime_material, target_timerange=trange(_s(RESOLVE_CHIME_AT), _s(0.7)), volume=0.6)
    script.add_segment(chime_seg, track_name="효과음")

    riser_seg = cc.AudioSegment(riser_material, target_timerange=trange(_s(RISER_AT), _s(0.6)), volume=0.5)
    script.add_segment(riser_seg, track_name="효과음_라이저")

    script.save()
    return draft_name


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    name = build_intro_draft()
    print(f"완료: '{name}' 초안이 CapCut 프로젝트 목록에 생성되었습니다.")
