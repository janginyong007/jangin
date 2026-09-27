"""완성된 롱폼 영상 -> 1분 이내 세로 숏폼 자동 생성 파이프라인.

CLI 사용법:
    python -m src.shorts_pipeline "<롱폼영상경로>" [--name "draft 이름"] [--config config.yaml]

run_shorts_pipeline()은 CLI와 웹서버(webapp.py)가 함께 쓰는 핵심 함수다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from .build_draft import build_shorts_draft
from .highlight import select_highlight_spans
from .subtitles import build_subtitles
from .transcribe import transcribe
from .utils import (
    PipelineError,
    get_anthropic_api_key,
    load_config,
    load_secrets,
    probe_video,
    project_root,
    sanitize_draft_name,
)

LogFn = Callable[[str], None]


def run_shorts_pipeline(
    media_path: str,
    draft_name: str,
    cfg: dict,
    cache_dir: str,
    log: LogFn = print,
    title: str | None = None,
) -> dict:
    """완성된 롱폼 영상 하나를 끝까지 처리해서 숏폼 캡컷 draft를 만든다."""
    media_path = os.path.abspath(media_path)
    if not os.path.exists(media_path):
        raise PipelineError(f"파일을 찾을 수 없습니다: {media_path}")

    secrets = load_secrets()
    api_key = get_anthropic_api_key(secrets)

    log(f"[1/5] 파일 정보 확인: {media_path}")
    media = probe_video(media_path)
    if media.is_audio_only:
        raise PipelineError("숏폼 만들기는 영상 파일에만 사용할 수 있습니다 (오디오 전용 파일은 지원하지 않습니다).")
    log(f"      해상도 {media.width}x{media.height}, {media.fps:.2f}fps, 길이 {media.duration:.1f}초")

    log("[2/5] 오디오 추출 및 음성 인식 (시간이 걸릴 수 있습니다)...")
    t0 = time.time()
    words = transcribe(media_path, cache_dir, cfg["whisper"])
    log(f"      완료 ({time.time() - t0:.1f}초, 단어 {len(words)}개 인식)")

    log("[3/5] AI로 하이라이트 구간 선정...")
    shorts_cfg = cfg["shorts"]
    cut_cfg = cfg["cut"]
    clip_spans = select_highlight_spans(
        words,
        media.duration,
        api_key,
        shorts_cfg.get("claude_model", "claude-sonnet-5"),
        shorts_cfg.get("target_duration", 55),
        shorts_cfg.get("claude_max_tokens", 4096),
        pad_before=cut_cfg.get("padding_before", 0.6),
        pad_after=cut_cfg.get("padding_after", 0.15),
        log=log,
    )

    cues = []
    srt_path = None
    if shorts_cfg.get("captions_enabled", False):
        log("[4/5] 자막 생성...")
        srt_path = os.path.join(
            cache_dir, f"{os.path.splitext(os.path.basename(media_path))[0]}_숏폼.srt"
        )
        cues = build_subtitles(words, clip_spans, cfg["subtitle"], srt_path, hard_break_at_span=True)
        log(f"      자막 {len(cues)}줄 생성 (참고용 SRT: {srt_path})")
    else:
        log("[4/5] 자막 생성 건너뜀 (본영상에 이미 자막이 박혀있다고 설정됨 - shorts.captions_enabled)")

    log("[5/5] 캡컷 숏폼 draft 조립 및 저장 (세로 캔버스)...")
    build_shorts_draft(
        media_path,
        clip_spans,
        cues,
        media,
        draft_name,
        cfg["capcut"],
        cfg["subtitle"],
        shorts_cfg,
        title=title,
    )

    total = sum(e - s for s, e in clip_spans)
    log("")
    log(f"완료! CapCut을 열어 '{draft_name}' 프로젝트를 확인하세요. (총 {total:.1f}초, {len(clip_spans)}개 구간)")
    log("(캡컷이 이미 실행 중이었다면, 최근 프로젝트 목록을 새로고침하거나 재시작 후 확인하세요.)")

    return {
        "draft_name": draft_name,
        "srt_path": srt_path,
        "clip_count": len(clip_spans),
        "total_duration": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="완성된 롱폼 영상에서 1분 이내 세로 숏폼 캡컷 draft를 자동 생성합니다.")
    parser.add_argument("video", help="완성된 롱폼 영상 파일 경로")
    parser.add_argument("--name", help="캡컷에 생성할 draft 이름 (기본: 파일명 + _숏폼)")
    parser.add_argument("--title", help="숏폼 상단에 표시할 제목 (보통 본영상 제목)")
    parser.add_argument("--config", default=None, help="config.yaml 경로 (기본: 프로젝트 루트의 config.yaml)")
    args = parser.parse_args()

    root = project_root()
    config_path = args.config or os.path.join(root, "config.yaml")
    cfg = load_config(config_path)
    cache_dir = os.path.join(root, "cache")

    media_path = os.path.abspath(args.video)
    draft_name = sanitize_draft_name(args.name or (os.path.splitext(os.path.basename(media_path))[0] + "_숏폼"))

    try:
        run_shorts_pipeline(media_path, draft_name, cfg, cache_dir, title=args.title)
    except PipelineError as e:
        print(f"[오류] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
