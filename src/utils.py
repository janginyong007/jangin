"""공용 유틸: 미디어 메타데이터 조회, 설정 로드."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import yaml
from pymediainfo import MediaInfo

_ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')


class PipelineError(Exception):
    """파이프라인 실행 중 사용자에게 그대로 보여줘도 되는 오류."""


@dataclass
class MediaInfoResult:
    width: int
    height: int
    fps: float
    duration: float  # seconds
    is_audio_only: bool = False


def probe_video(path: str) -> MediaInfoResult:
    """영상 또는 (영상 트랙이 없는) 오디오 전용 파일의 메타데이터를 읽는다."""
    mi = MediaInfo.parse(path)
    video_track = next((t for t in mi.tracks if t.track_type == "Video"), None)
    audio_track = next((t for t in mi.tracks if t.track_type == "Audio"), None)
    general_track = next((t for t in mi.tracks if t.track_type == "General"), None)

    if video_track is None and audio_track is None:
        raise ValueError(f"영상/오디오 트랙을 찾을 수 없습니다: {path}")

    if video_track is not None:
        width = int(video_track.width)
        height = int(video_track.height)
        fps = float(video_track.frame_rate)
        duration_ms = video_track.duration or (general_track.duration if general_track else None)
        is_audio_only = False
    else:
        width = height = 0
        fps = 30.0
        duration_ms = audio_track.duration or (general_track.duration if general_track else None)
        is_audio_only = True

    if duration_ms is None:
        raise ValueError(f"길이를 읽을 수 없습니다: {path}")
    duration = float(duration_ms) / 1000.0

    return MediaInfoResult(width=width, height=height, fps=fps, duration=duration, is_audio_only=is_audio_only)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_secrets(secrets_path: str | None = None) -> dict:
    secrets_path = secrets_path or os.path.join(project_root(), "secrets.yaml")
    if not os.path.exists(secrets_path):
        return {}
    with open(secrets_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data or {}


def get_anthropic_api_key(secrets: dict) -> str:
    key = (secrets.get("anthropic_api_key") or "").strip()
    return key


def sanitize_draft_name(name: str) -> str:
    """캡컷 draft 폴더 이름으로 쓸 수 없는 문자(\\ / : * ? " < > |)를 제거한다.
    사용자가 "?"로 끝나는 후킹용 제목 같은 걸 draft 이름 칸에 그대로 입력하면
    윈도우가 폴더 생성을 거부하므로(WinError 123), 미리 걸러낸다."""
    cleaned = _ILLEGAL_FILENAME_CHARS.sub(" ", name).strip()
    cleaned = cleaned.rstrip(". ")  # 윈도우는 폴더 이름이 마침표/공백으로 끝나는 것도 허용하지 않음
    return cleaned or "이름없음"
