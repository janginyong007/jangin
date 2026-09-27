"""부동산장인 음성 자동 컷 편집기.

사용법:
  python -m autocut 원본.mp4                   # 분석 + 컷 + 렌더링 (원본_cut.mp4)
  python -m autocut 원본.mp4 --no-render       # 분석만 (cutlist.json, 리포트)
  python -m autocut --render 원본_cutlist.json  # 수정한 컷 목록으로 다시 렌더링
"""
from __future__ import annotations

import argparse
import os
import sys

from . import audio as A
from .retake import RetakeConfig, find_retakes
from .timeline import (CutConfig, finalize, load_cutlist, render, save_cutlist,
                       speech_keeps, subtract, write_report)


def main(argv=None):
    p = argparse.ArgumentParser(prog="autocut", description="음성 기준 자동 컷 편집 (무음 + 반복 발화 제거)")
    p.add_argument("input", nargs="?", help="원본 영상/음성 파일")
    p.add_argument("--render", metavar="CUTLIST", help="cutlist.json으로 렌더링만 다시 실행")
    p.add_argument("-o", "--output", help="출력 파일 경로")
    p.add_argument("--no-render", action="store_true", help="분석/리포트만 만들고 렌더링은 생략")
    p.add_argument("--audio-only", action="store_true", help="음성 파일(wav)만 출력")
    # 인식
    p.add_argument("--model", default="large-v3", help="Whisper 모델 (large-v3 / medium / small)")
    p.add_argument("--device", default="auto", help="auto / cuda / cpu")
    p.add_argument("--no-retake", action="store_true", help="반복 발화 검출 끄기 (무음 컷만)")
    # 컷 감도
    p.add_argument("--pre-pad", type=float, default=0.12, help="말 시작 전 여유 초 (기본 0.12)")
    p.add_argument("--post-pad", type=float, default=0.35, help="말 끝난 뒤 여유 초 (기본 0.35, 말끝 잘리면 올리기)")
    p.add_argument("--min-silence", type=float, default=0.35, help="이보다 짧은 쉼은 자르지 않음 (기본 0.35)")
    p.add_argument("--hangover", type=float, default=0.20, help="소리가 작아진 뒤 이만큼 더 기다렸다 끝으로 판정 (기본 0.20)")
    p.add_argument("--off-margin", type=float, default=7.0, help="말끝 판정 임계값 = 소음+N dB (낮출수록 말끝을 더 살림)")
    p.add_argument("--retake-threshold", type=float, default=78.0, help="반복 판정 유사도 (기본 78, 낮출수록 더 많이 잡음)")
    p.add_argument("--no-inline", action="store_true", help="쉼 없이 바로 고쳐 말한 것 검출 끄기")
    args = p.parse_args(argv)

    cut_cfg = CutConfig(pre_pad=args.pre_pad, post_pad=args.post_pad)

    if args.render:
        source, keeps = load_cutlist(args.render)
        out = args.output or _default_out(source, args.audio_only)
        print(f"렌더링: {len(keeps)}개 조각 → {out}")
        render(source, keeps, out, cut_cfg, audio_only=args.audio_only)
        return 0

    if not args.input:
        p.error("입력 파일을 지정하세요")
    src = args.input
    base = os.path.splitext(src)[0]

    print("1/4 오디오 분석 중...")
    audio = A.load_audio(src)
    duration = len(audio) / A.SR
    db = A.energy_db(audio)
    th = A.estimate_thresholds(db, off_margin=args.off_margin)
    regions = A.speech_regions(db, th, min_silence=args.min_silence, hangover=args.hangover)
    print(f"    소음 {th.noise_floor:.1f}dB / 시작 {th.speech_on:.1f}dB / 끝 {th.speech_off:.1f}dB, 발화 구간 {len(regions)}개")

    utts, removals = [], []
    if not args.no_retake:
        from .transcribe import load_model, split_long, transcribe_regions
        print(f"2/4 음성 인식 중 (모델 {args.model})...")
        model = load_model(args.model, device=args.device)
        chunks = split_long(regions, db)
        utts = transcribe_regions(model, audio, chunks,
                                  progress=lambda i, n: print(f"\r    {i}/{n}", end="", flush=True))
        print()
        print("3/4 반복 발화 검출 중...")
        cfg = RetakeConfig(threshold=args.retake_threshold, inline=not args.no_inline)
        removals = find_retakes(utts, cfg)
        print(f"    {len(removals)}건 발견")
    else:
        print("2-3/4 반복 발화 검출 생략")

    keeps = speech_keeps(regions, db, duration, cut_cfg)
    keeps = finalize(subtract(keeps, removals, db), cut_cfg)

    cutlist = base + "_cutlist.json"
    report = base + "_report.md"
    save_cutlist(cutlist, src, keeps, removals, utts)
    write_report(report, duration, keeps, removals, utts)
    print(f"    컷 목록: {cutlist}\n    리포트: {report}")

    if args.no_render:
        return 0
    out = args.output or _default_out(src, args.audio_only)
    print(f"4/4 렌더링 → {out}")
    render(src, keeps, out, cut_cfg, audio_only=args.audio_only)
    print("완료!")
    return 0


def _default_out(src, audio_only):
    base = os.path.splitext(src)[0]
    return base + ("_cut.wav" if audio_only else "_cut.mp4")


if __name__ == "__main__":
    sys.exit(main())
