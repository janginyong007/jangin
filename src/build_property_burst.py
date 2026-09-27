"""판사봉 타격 순간의 '금빛 스파크'를 '건물/아파트/토지/상가 아이콘이 사방으로
튀어나가는' 연출로 대체(가중)하는 오버레이를 만들어 기존 인트로 영상에 합성한다.

절차:
  1. 4종 부동산 아이콘(아파트/단독주택/토지/상가)을 골드 실루엣으로 그린다.
  2. 타격 지점(impact point)에서 사방으로 날아가는 파티클 애니메이션을
     투명 배경 PNG 시퀀스로 렌더링한다.
  3. 원본 스파크를 살짝 눌러주는 어두운 라디얼 마스크 + 아이콘 오버레이를
     원본 영상 위에 합성해 최종 mp4로 내보낸다.
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FFMPEG = (
    r"C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python311\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
)

SCRATCH = Path(
    r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator---\ca8ca881-e1dd-4925-957f-d2668595b7dd"
    r"\scratchpad\property_burst"
)
FRAMES_DIR = SCRATCH / "frames"

SRC_VIDEO = Path(r"C:\Users\Administrator\Downloads\부동산장인_인트로_0717.mp4")
OUT_VIDEO = Path(r"C:\Users\Administrator\Downloads\부동산장인_인트로_0717_건물버스트.mp4")

W, H = 3840, 2160
FPS = 30
TOTAL_DURATION = 5.2
IMPACT_T = 1.00          # 실제 타격("쾅") 시각 (초)
IMPACT_X, IMPACT_Y = 1456, 1866  # 타격 지점 픽셀 좌표 (밝기 분석으로 확인)
# 원본 영상은 타격 후 약 0.19초 만에 로고 소용돌이 장면으로 컷 전환된다(1.19s 부근).
# 컷 전에 전부 끝나야 "타격 순간"과 정확히 맞아떨어진다 - 다음 장면으로 새어나가지 않게 한다.
CUT_T = 1.18
BURST_DURATION = 0.14    # 컷 전에 확산이 끝나도록 아주 짧게
ACTIVE_END_T = CUT_T     # 이 시점 이후 프레임은 완전히 비운다
N_ICONS = 90             # 성긴 몇 개가 아니라 금가루처럼 촘촘하게
GOLD = (255, 205, 110)
GOLD_DARK = (150, 100, 20)

# 원본 금빛 스파크를 눌러줘서 건물 아이콘이 스파크를 "대체"하는 것처럼 보이게 한다.
MASK_CENTER = (IMPACT_X, IMPACT_Y + 30)
MASK_RADIUS = 780
MASK_MAX_ALPHA = 175


def _icon_canvas() -> Image.Image:
    return Image.new("RGBA", (400, 400), (0, 0, 0, 0))


def make_apartment() -> Image.Image:
    img = _icon_canvas()
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([120, 40, 280, 380], radius=10, fill=GOLD)
    for row in range(7):
        for col in range(3):
            x0 = 138 + col * 42
            y0 = 70 + row * 42
            d.rectangle([x0, y0, x0 + 26, y0 + 26], fill=(0, 0, 0, 0))
    return img


def make_house() -> Image.Image:
    img = _icon_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([90, 190, 310, 370], fill=GOLD)
    d.polygon([(70, 190), (200, 60), (330, 190)], fill=GOLD)
    d.rectangle([175, 270, 225, 370], fill=(0, 0, 0, 0))
    d.rectangle([115, 230, 155, 265], fill=(0, 0, 0, 0))
    d.rectangle([245, 230, 285, 265], fill=(0, 0, 0, 0))
    return img


def make_land() -> Image.Image:
    img = _icon_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([70, 70, 330, 330], outline=GOLD, width=22)
    d.line([70, 200, 330, 200], fill=GOLD, width=14)
    d.line([200, 70, 200, 330], fill=GOLD, width=14)
    for cx, cy in [(70, 70), (330, 70), (70, 330), (330, 330)]:
        d.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=GOLD)
    return img


def make_shop() -> Image.Image:
    img = _icon_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([80, 160, 320, 360], fill=GOLD)
    for i in range(6):
        x0 = 70 + i * 42
        pts = [(x0, 110), (x0 + 42, 110), (x0 + 52, 160), (x0 - 10, 160)]
        d.polygon(pts, fill=GOLD if i % 2 == 0 else GOLD_DARK)
    d.rectangle([180, 260, 220, 360], fill=(0, 0, 0, 0))
    d.rectangle([105, 200, 165, 245], fill=(0, 0, 0, 0))
    d.rectangle([235, 200, 295, 245], fill=(0, 0, 0, 0))
    return img


def build_icons() -> list[Image.Image]:
    makers = [make_apartment, make_house, make_land, make_shop]
    icons = []
    for maker in makers:
        icon = maker()
        glow = icon.filter(ImageFilter.GaussianBlur(6))
        composed = Image.alpha_composite(Image.new("RGBA", icon.size, (0, 0, 0, 0)), glow)
        composed = Image.alpha_composite(composed, icon)
        icons.append(composed)
    return icons


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def make_spark_mask() -> Image.Image:
    """원본 스파크를 눌러주는 부드러운 어두운 라디얼 마스크."""
    size = MASK_RADIUS * 3
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(1.0 - dist / MASK_RADIUS, 0.0, 1.0) ** 1.6 * MASK_MAX_ALPHA
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., 3] = alpha.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def render_frames(icons: list[Image.Image]) -> int:
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)

    total_frames = int(round(TOTAL_DURATION * FPS))
    impact_frame = int(round(IMPACT_T * FPS))
    active_end_frame = int(round(ACTIVE_END_T * FPS))
    mask = make_spark_mask()

    random.seed(7)
    particles = []
    for i in range(N_ICONS):
        angle = math.radians(random.uniform(0, 360))  # 사방으로 - 금가루처럼 전방위 확산
        is_hero = random.random() < 0.18  # 일부만 크게 - "이게 건물이구나" 인지되도록
        if is_hero:
            base_scale = random.uniform(0.22, 0.34)
            speed = random.uniform(2600, 4200)
        else:
            base_scale = random.uniform(0.05, 0.12)
            speed = random.uniform(3800, 7200)
        icon = random.choice(icons)
        spin = random.uniform(-200, 200)
        delay = random.uniform(0.0, 0.03)
        particles.append(dict(angle=angle, speed=speed, icon=icon, base_scale=base_scale, spin=spin, delay=delay))

    blank = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    blank_path = FRAMES_DIR / "blank.png"
    blank.save(blank_path)

    for f in range(total_frames):
        if f < impact_frame or f >= active_end_frame:
            shutil.copy(blank_path, FRAMES_DIR / f"frame_{f:04d}.png")
            continue

        local_t = (f - impact_frame) / FPS
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        mx = int(MASK_CENTER[0] - mask.width / 2)
        my = int(MASK_CENTER[1] - mask.height / 2)
        canvas.alpha_composite(mask, (mx, my))

        for p in particles:
            t = local_t - p["delay"]
            if t < 0:
                continue
            progress = min(t / BURST_DURATION, 1.0)
            eased = ease_out_cubic(progress)
            dist = p["speed"] * eased
            x = IMPACT_X + math.cos(p["angle"]) * dist
            y = IMPACT_Y + math.sin(p["angle"]) * dist

            grow = min(progress / 0.35, 1.0)
            scale = p["base_scale"] * (0.4 + 0.6 * grow)
            if scale <= 0.005:
                continue

            icon = p["icon"]
            size = max(1, int(icon.width * scale))
            resized = icon.resize((size, size), Image.LANCZOS)
            angle_deg = p["spin"] * progress
            rotated = resized.rotate(angle_deg, expand=True)

            px = int(x - rotated.width / 2)
            py = int(y - rotated.height / 2)
            canvas.alpha_composite(rotated, (px, py))

        canvas.save(FRAMES_DIR / f"frame_{f:04d}.png")

    return total_frames


def composite_final() -> None:
    filter_complex = "[0:v][1:v]overlay=x=0:y=0:format=auto[vout]"

    cmd = [
        FFMPEG, "-y",
        "-i", str(SRC_VIDEO),
        "-framerate", str(FPS), "-i", str(FRAMES_DIR / "frame_%04d.png"),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "0:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(TOTAL_DURATION),
        str(OUT_VIDEO),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    icons = build_icons()
    n = render_frames(icons)
    print(f"프레임 {n}개 렌더링 완료")
    composite_final()
    print(f"완료: {OUT_VIDEO}")
