"""무음/버벅임 컷이 적용된 오디오를 하나의 mp3 파일로 렌더링한다 (오디오 단독 입력용).
캡컷 프로젝트로 만들면 기존에 작업 중인 다른 프로젝트/영상과 합치기 어려우므로,
바로 끌어다 쓸 수 있는 평범한 mp3 파일로 내보낸다."""
from __future__ import annotations

import os
import subprocess
from typing import List, Tuple

import imageio_ffmpeg

Span = Tuple[float, float]


def render_audio(source_path: str, keep_spans: List[Span], out_path: str, declick_fade: float = 0.02) -> str:
    """declick_fade: 각 컷 조각의 시작/끝에 넣는 짧은 페이드(초). 하드컷을 그냥
    이어붙이면 이어지는 지점마다 "딸깍"거리는 클릭 노이즈가 생기는데, 사람이
    못 느낄 정도로 아주 짧은 페이드만 넣어도 이게 없어진다."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    filters = []
    labels = []
    for i, (start, end) in enumerate(keep_spans):
        duration = end - start
        if duration <= 0:
            continue
        fade = min(declick_fade, duration / 2 - 0.001) if duration > 0.002 else 0.0
        fade_part = f",afade=t=in:st=0:d={fade},afade=t=out:st={duration - fade}:d={fade}" if fade > 0 else ""
        filters.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS{fade_part}[a{i}]")
        labels.append(f"[a{i}]")

    if not labels:
        raise RuntimeError("렌더링할 오디오 구간이 없습니다.")

    concat = "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]"
    filter_complex = ";".join(filters + [concat])

    cmd = [
        ffmpeg_exe, "-y", "-i", source_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-codec:a", "libmp3lame", "-q:a", "2",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"오디오 렌더링 실패:\n{result.stderr}")
    return out_path
