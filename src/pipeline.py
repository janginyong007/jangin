"""캡컷 자동 편집 파이프라인: 무음/버벅임 구간 컷 + 자막 자동 생성.

CLI 사용법:
    python -m src.pipeline "<원본영상경로>" [--name "draft 이름"] [--config config.yaml]

run_pipeline()은 CLI와 웹서버(webapp.py)가 함께 쓰는 핵심 함수다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Callable, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from .audio_energy import load_energy
from .build_draft import build_draft
from .cutdetect import collapse_phantom_repeats, detect_cuts
from .disfluency import review_cut_script
from .render_audio import render_audio
from .subtitles import build_subtitles
from .transcribe import extract_audio, extract_audio_hq, transcribe
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

# 업데이트가 실제로 적용됐는지 로그 첫 줄에서 바로 확인할 수 있게 표시한다.
VERSION = "2026-09-27 수정판 6 (작은 목소리 재시도 잡기)"


def run_pipeline(
    media_path: str,
    draft_name: str,
    cfg: dict,
    cache_dir: str,
    log: LogFn = print,
    cut_video: bool = True,
    apply_cuts: bool = True,
) -> dict:
    """영상/오디오 하나를 끝까지 처리해서 캡컷 draft를 만든다. 진행 상황은 log()로 보고한다.
    apply_cuts=False면 무음/필러/반복발화 컷을 아예 건너뛰고, 원본 길이 그대로에
    자막만 정확히 맞춰서 붙인다 (다른 도구로 이미 컷 편집을 끝내고, 자막 싱크만
    캡컷에서 다시 잡고 싶을 때 쓰는 모드)."""
    media_path = os.path.abspath(media_path)
    if not os.path.exists(media_path):
        raise PipelineError(f"파일을 찾을 수 없습니다: {media_path}")

    log(f"[프로그램 버전] {VERSION}")
    log(f"[1/5] 파일 정보 확인: {media_path}")
    media = probe_video(media_path)
    if media.is_audio_only:
        log(f"      오디오 전용 파일, 길이 {media.duration:.1f}초")
    else:
        log(f"      해상도 {media.width}x{media.height}, {media.fps:.2f}fps, 길이 {media.duration:.1f}초")

    log("[2/5] 오디오 추출 및 음성 인식 (시간이 걸릴 수 있습니다)...")
    t0 = time.time()
    words = transcribe(media_path, cache_dir, cfg["whisper"])
    log(f"      완료 ({time.time() - t0:.1f}초, 단어 {len(words)}개 인식)")
    n_before = len(words)
    words = collapse_phantom_repeats(words, cfg.get("cut", {}).get("repeat_max_syllable_rate", 10.0))
    if len(words) < n_before:
        log(f"      음성 인식이 중복으로 적은 단어 {n_before - len(words)}개 정리 (예: '2013년 2013년')")

    if apply_cuts:
        log("[3/5] 무음/필러/반복발화 컷 구간 감지...")
        cut_cfg = cfg["cut"]
        energy = None
        if cut_cfg.get("energy_refine", True):
            # 실제 음량으로 컷 경계를 보정 (말끝 잘림 방지). 실패해도 기존 방식으로 진행.
            try:
                energy = load_energy(extract_audio(media_path, cache_dir))
            except Exception as e:  # noqa: BLE001
                log(f"      (음량 분석 실패, 기존 방식으로 진행: {e})")
        keep_spans, stats = detect_cuts(words, media.duration, cut_cfg, energy=energy)
        log(
            f"      무음 {stats['silence_spans']}건, 필러워드 {stats['filler_spans']}건, "
            f"반복발화 {stats['repeat_spans']}건 -> 병합 후 {stats['merged_cut_spans']}개 구간 컷"
        )
        log(
            f"      원본 {stats['total_duration']:.1f}초 -> 1차 편집본 {stats['kept_duration']:.1f}초 "
            f"({stats['total_cut_seconds']:.1f}초 삭제, 유지 구간 {stats['keep_spans']}개)"
        )

        if cut_cfg.get("ai_disfluency", False):
            # 1차 컷 결과를 번호가 붙은 자막으로 만들어 Claude에게 검토시킨다.
            # (원본 전체를 주고 임의의 시각을 만들게 하면 경계를 잘못 그릴 위험이
            # 커서 - 71초짜리 구간을 통째로 잘못 잡은 사례가 실제로 있었음 -
            # 이미 정해진 자막 "줄 번호"만 고르게 하는 게 훨씬 안전하다.)
            orig_spans: list = []
            review_cues = build_subtitles(
                words, keep_spans, cfg["subtitle"],
                os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(media_path))[0]}_1차검토.srt"),
                orig_spans_out=orig_spans,
            )
            secrets = load_secrets()
            api_key = get_anthropic_api_key(secrets)
            extra_spans = review_cut_script(
                review_cues,
                orig_spans,
                api_key,
                cut_cfg.get("claude_model", "claude-sonnet-5"),
                cut_cfg.get("claude_max_tokens", 4096),
                log=log,
            )
            if extra_spans:
                keep_spans, stats = detect_cuts(
                    words, media.duration, cut_cfg, ai_spans=extra_spans, energy=energy
                )
                if stats.get("ai_spans_rejected"):
                    log(f"      (AI 제안 중 {stats['ai_spans_rejected']}개는 뒤에 다시 말한 게 안 보여서 자르지 않음)")
                log(
                    f"      최종 검토로 {stats['ai_spans']}개 구간 추가 컷 -> "
                    f"편집본 {stats['kept_duration']:.1f}초 (유지 구간 {stats['keep_spans']}개)"
                )
    else:
        log("[3/5] 컷 없이 원본 그대로 유지 (자막만 생성하는 모드)")
        keep_spans = [(0.0, media.duration)]
        stats = {
            "silence_spans": 0,
            "filler_spans": 0,
            "repeat_spans": 0,
            "ai_spans": 0,
            "merged_cut_spans": 0,
            "keep_spans": 1,
            "total_duration": media.duration,
            "total_cut_seconds": 0.0,
            "kept_duration": media.duration,
        }

    if apply_cuts:
        report_path = write_cut_report(media_path, draft_name, stats, words=words, keep_spans=keep_spans)
        log(f"      컷 리포트(무엇을 왜 잘랐는지): {report_path}")

    if not keep_spans:
        raise PipelineError("컷 감지 결과 남는 구간이 없습니다. config.yaml의 cut 임계값을 조정하세요.")

    log("[4/5] 자막 생성...")
    srt_path = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(media_path))[0]}.srt")
    cues = build_subtitles(words, keep_spans, cfg["subtitle"], srt_path)
    log(f"      자막 {len(cues)}줄 생성 (참고용 SRT: {srt_path})")

    if not apply_cuts:
        # 컷 없이 자막만 생성하는 모드 - 오디오/영상이 이미 다른 도구(브루 등)에서
        # 편집을 마친 상태라고 가정하므로, 원본 타임라인 그대로에 맞춘 SRT만 내보낸다.
        # 캡컷 draft를 새로 만들지 않는 이유: 이미 작업 중인 캡컷 프로젝트가 따로
        # 있을 테니, 그 프로젝트에 "자막 가져오기"로 이 SRT를 직접 불러오면 된다.
        log("[5/5] 자막(SRT) 파일 내보내는 중...")
        output_dir = os.path.dirname(media_path)
        os.makedirs(output_dir, exist_ok=True)
        srt_output = os.path.join(output_dir, f"{draft_name}.srt")
        shutil.copyfile(srt_path, srt_output)

        log("")
        log(f"완료! 자막 파일: {srt_output}")
        log("캡컷에서 작업 중인 프로젝트를 열고, 상단 메뉴의 '텍스트 > 자막 가져오기'로 이 SRT를 불러오면 됩니다.")

        return {
            "draft_name": None,
            "srt_output": srt_output,
            "stats": stats,
        }

    if media.is_audio_only:
        # 오디오만 넣은 경우엔 캡컷 프로젝트로 만들지 않는다 - 이미 작업 중인 다른
        # 캡컷 프로젝트(영상)에 합치기 어려워지므로, 바로 끌어다 쓸 mp3+자막으로 내보낸다.
        log("[5/5] 편집된 오디오를 mp3로 내보내는 중...")
        output_dir = os.path.dirname(media_path)
        os.makedirs(output_dir, exist_ok=True)
        audio_output = os.path.join(output_dir, f"{draft_name}.mp3")
        render_audio(media_path, keep_spans, audio_output)
        srt_output = os.path.join(output_dir, f"{draft_name}.srt")
        shutil.copyfile(srt_path, srt_output)

        log("")
        log(f"완료! 편집된 오디오: {audio_output}")
        log(f"자막 파일: {srt_output}")
        log("이 mp3를 캡컷에서 기존 영상 타임라인에 직접 끌어다 넣으시면 됩니다.")

        return {
            "draft_name": None,
            "audio_output": audio_output,
            "srt_output": srt_output,
            "stats": stats,
        }

    # 오디오는 항상 별도의 고음질 트랙으로 뽑아서(클릭 노이즈 방지용 페이드를
    # 넣기 위해 - build_draft.py 참고) 쓴다. cut_video는 화면을 자를지만 결정한다.
    log("      오디오 트랙용 고음질 오디오 추출 중...")
    hq_audio_path = extract_audio_hq(media_path, cache_dir)

    log("[5/5] 캡컷 draft 조립 및 저장...")
    build_draft(
        media_path,
        keep_spans,
        cues,
        media,
        draft_name,
        cfg["capcut"],
        cfg["subtitle"],
        cut_video=cut_video,
        hq_audio_path=hq_audio_path,
    )

    log("")
    log(f"완료! CapCut을 열어 '{draft_name}' 프로젝트를 확인하세요.")
    log("(캡컷이 이미 실행 중이었다면, 최근 프로젝트 목록을 새로고침하거나 재시작 후 확인하세요.)")

    return {"draft_name": draft_name, "srt_path": srt_path, "stats": stats}


def _fmt_ts(t: float) -> str:
    m, sec = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{sec:05.2f}"


def _edited_time(t: float, keep_spans) -> Optional[float]:
    """원본 시각 t가 편집본에서 몇 초인지 (잘려나간 곳이면 None)."""
    acc = 0.0
    for s, e in keep_spans:
        if s <= t <= e:
            return acc + (t - s)
        acc += e - s
    return None


def write_cut_report(media_path: str, draft_name: str, stats: dict, words=None, keep_spans=None) -> str:
    """잘라낸 말(필러/반복/AI검토)을 원본 시각과 함께 적은 txt. 필요한 말이 잘렸을 때
    어떤 규칙 때문인지 바로 알 수 있게 원본 영상 옆에 저장한다.
    words/keep_spans를 주면 전체 받아쓰기(잘린 말 표시, 편집본 시각 포함)도 덧붙인다 -
    반복을 못 잡았을 때 위스퍼가 그 부분을 어떻게 받아적었는지 확인하는 용도."""
    path = os.path.join(os.path.dirname(media_path), f"{draft_name}_컷리포트.txt")
    lines = [
        f"컷 리포트 - {os.path.basename(media_path)}",
        "시각은 '원본 영상' 기준입니다. 무음 컷은 목록에서 뺐습니다(말이 없는 구간).",
        "필요한 말이 잘렸다면 아래에서 그 줄을 찾아 [종류]를 확인하세요.",
        "",
    ]
    for r in stats.get("cut_details", []):
        lines.append(f"[{r['kind']}] {_fmt_ts(r['start'])} ~ {_fmt_ts(r['end'])}  잘라낸 말: \"{r['text']}\"")
        if r["after"]:
            lines.append(f"            바로 뒤 이어지는 말: \"{r['after']}\"")
    if not stats.get("cut_details"):
        lines.append("(말이 들어있는 컷 없음)")
    if stats.get("untranscribed"):
        lines += ["", "음성 인식이 받아적지 못했지만 말소리가 있어서 자르지 않고 남긴 곳:",
                  "  (여기서 반복이 들리면 이 시각을 알려주세요)"]
        for s_, e_ in stats["untranscribed"]:
            lines.append(f"[받아쓰기 없음] {_fmt_ts(s_)} ~ {_fmt_ts(e_)}")

    if words and keep_spans:
        lines += [
            "",
            "=" * 60,
            "전체 받아쓰기 (음성 인식이 받아적은 그대로)",
            "  [원본 시각 | 편집본 시각]  ⟦ ⟧ 안의 말은 잘려나간 말입니다.",
            "  편집본을 들으면서 이상한 곳이 있으면 '편집본 시각'으로 찾으세요.",
            "=" * 60,
        ]
        line_words: list = []

        def flush():
            if not line_words:
                return
            t0 = line_words[0].start
            ed = next((x for x in (_edited_time((w.start + w.end) / 2, keep_spans) for w in line_words)
                       if x is not None), None)
            ed_s = _fmt_ts(ed) if ed is not None else "  (잘림) "
            text, in_cut = [], False
            for w in line_words:
                cut = _edited_time((w.start + w.end) / 2, keep_spans) is None
                if cut and not in_cut:
                    text.append("⟦")
                if not cut and in_cut:
                    text.append("⟧")
                in_cut = cut
                text.append(w.text.strip())
            if in_cut:
                text.append("⟧")
            joined = " ".join(text).replace("⟦ ", "⟦").replace(" ⟧", "⟧")
            lines.append(f"[{_fmt_ts(t0)} | {ed_s}] {joined}")
            line_words.clear()

        for i, w in enumerate(words):
            # 0.5초 이상 쉬거나 한 줄이 너무 길어지면 줄바꿈
            if line_words and (w.start - line_words[-1].end >= 0.5 or len(line_words) >= 14):
                flush()
            line_words.append(w)
        flush()

    with open(path, "w", encoding="utf-8-sig") as f:  # 메모장에서 한글 안 깨지게 BOM 포함
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="무음/버벅임 구간을 컷하고 자막을 단 캡컷 draft를 자동 생성합니다.")
    parser.add_argument("video", help="원본 영상/오디오 파일 경로")
    parser.add_argument("--name", help="캡컷에 생성할 draft 이름 (기본: 파일명 + _자동편집)")
    parser.add_argument("--config", default=None, help="config.yaml 경로 (기본: 프로젝트 루트의 config.yaml)")
    parser.add_argument(
        "--audio-only-cut",
        action="store_true",
        help="영상은 자르지 않고 오디오만 편집 (드론/이동 촬영본처럼 컷하면 화면이 끊겨 보이는 경우)",
    )
    parser.add_argument(
        "--subtitles-only",
        action="store_true",
        help="컷 없이 원본 그대로 두고 자막만 생성 (다른 도구로 이미 컷 편집을 끝낸 경우)",
    )
    args = parser.parse_args()

    root = project_root()
    config_path = args.config or os.path.join(root, "config.yaml")
    cfg = load_config(config_path)
    cache_dir = os.path.join(root, "cache")

    media_path = os.path.abspath(args.video)
    draft_name = sanitize_draft_name(args.name or (os.path.splitext(os.path.basename(media_path))[0] + "_자동편집"))

    try:
        run_pipeline(
            media_path,
            draft_name,
            cfg,
            cache_dir,
            cut_video=not args.audio_only_cut,
            apply_cuts=not args.subtitles_only,
        )
    except PipelineError as e:
        print(f"[오류] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
