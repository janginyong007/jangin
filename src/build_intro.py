"""부동산장인 채널 인트로 영상을 CapCut 초안(draft)으로 생성한다.

빨강/오렌지 그라디언트 배경(살짝 줌인되는 움직임) 위에
채널명과 슬로건 텍스트가 순서대로 등장하는 4초짜리 인트로를 만든다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import pycapcut as cc
from pycapcut import trange

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION = 4.0

CHANNEL_NAME = "부동산장인"
SLOGAN = "오늘의 알짜 매물, 지금 확인하세요"


def _s(seconds: float) -> str:
    return f"{seconds}s"


def make_background(path: Path, width: int = WIDTH, height: int = HEIGHT) -> None:
    """대각선 빨강->오렌지 그라디언트 + 중앙 상단 은은한 글로우 + 가장자리 비네트."""
    c1 = np.array([100, 8, 8], dtype=np.float64)      # 진한 빨강 (좌상단)
    c2 = np.array([255, 120, 0], dtype=np.float64)     # 강렬한 오렌지 (우하단)

    xs = np.linspace(0.0, 1.0, width)
    ys = np.linspace(0.0, 1.0, height)
    gx, gy = np.meshgrid(xs, ys)
    t = np.clip((gx + gy) / 2.0, 0.0, 1.0)

    img = (c1[None, None, :] * (1 - t[:, :, None]) + c2[None, None, :] * t[:, :, None])

    # 텍스트가 들어갈 중앙 부근에 은은한 밝은 글로우
    cx, cy = width * 0.5, height * 0.42
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    dist = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 1.3) ** 2)
    glow = np.clip(1.0 - dist / (width * 0.55), 0.0, 1.0) ** 2
    img = img + glow[:, :, None] * np.array([60, 40, 20])[None, None, :]

    # 가장자리 비네트 (어둡게)
    vx = np.abs(xx - width / 2) / (width / 2)
    vy = np.abs(yy - height / 2) / (height / 2)
    vignette = 1.0 - 0.35 * np.clip(np.sqrt(vx ** 2 + vy ** 2) - 0.4, 0.0, 1.0)
    img = img * vignette[:, :, None]

    img = np.clip(img, 0, 255).astype(np.uint8)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(path)


def build_intro_draft(draft_name: str = "부동산장인_인트로") -> str:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    draft_folder_path = cfg["capcut"]["draft_folder"]

    bg_path = ASSETS_DIR / "intro_bg.png"
    make_background(bg_path)

    draft_folder = cc.DraftFolder(draft_folder_path)
    script = draft_folder.create_draft(draft_name, WIDTH, HEIGHT, fps=FPS, allow_replace=True)

    # 배경: 은은하게 줌인되는 그라디언트
    script.add_track(cc.TrackType.video)
    bg_material = cc.VideoMaterial(str(bg_path))
    bg_segment = cc.VideoSegment(
        bg_material,
        target_timerange=trange(_s(0), _s(DURATION)),
    )
    bg_segment.add_keyframe(cc.KeyframeProperty.uniform_scale, _s(0), 1.0)
    bg_segment.add_keyframe(cc.KeyframeProperty.uniform_scale, _s(DURATION), 1.15)
    script.add_segment(bg_segment)

    # 채널명: 강렬하게 등장, 인트로 내내 유지
    script.add_track(cc.TrackType.text, track_name="채널명")
    name_style = cc.TextStyle(
        size=18.0, bold=True, color=(1.0, 0.85, 0.25), align=1,
    )
    name_border = cc.TextBorder(color=(0.0, 0.0, 0.0), width=90.0)
    name_segment = cc.TextSegment(
        CHANNEL_NAME,
        trange(_s(0), _s(DURATION)),
        style=name_style,
        border=name_border,
        clip_settings=cc.ClipSettings(transform_y=0.15),
    )
    name_segment.add_animation(cc.TextIntro.闪光出击, duration=_s(0.6))
    script.add_segment(name_segment, track_name="채널명")

    # 슬로건: 채널명이 자리잡은 뒤 살짝 늦게 페이드인
    script.add_track(cc.TrackType.text, track_name="슬로건")
    slogan_style = cc.TextStyle(
        size=8.0, bold=True, color=(1.0, 1.0, 1.0), align=1,
    )
    slogan_border = cc.TextBorder(color=(0.0, 0.0, 0.0), width=60.0)
    slogan_start = 1.3
    slogan_segment = cc.TextSegment(
        SLOGAN,
        trange(_s(slogan_start), _s(DURATION - slogan_start)),
        style=slogan_style,
        border=slogan_border,
        clip_settings=cc.ClipSettings(transform_y=-0.15),
    )
    slogan_segment.add_animation(cc.TextIntro.渐显, duration=_s(0.5))
    script.add_segment(slogan_segment, track_name="슬로건")

    script.save()
    return draft_name


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    name = build_intro_draft()
    print(f"완료: '{name}' 초안이 CapCut 프로젝트 목록에 생성되었습니다.")
