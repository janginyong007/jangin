"""컷 목록 만들기 + 렌더링.

말끝 잘림 방지 장치:
  1) 발화 끝은 에너지 히스테리시스로 판정 (audio.speech_regions)
  2) 뒤 여유(post_pad)를 앞 여유(pre_pad)보다 길게 — 한국어는 어미가 약하게 끝남
  3) 여유를 더해도 쉼이 짧으면 자르지 않음
  4) 컷 지점은 쉼 구간 중 '가장 조용한 곳'으로 이동
  5) 이어붙인 곳마다 아주 짧은 페이드로 '틱' 소리 제거
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

from .audio import quietest_point
from .retake import Removal


@dataclass
class CutConfig:
    pre_pad: float = 0.12      # 말 시작 전 여유 (초)
    post_pad: float = 0.35     # 말 끝난 뒤 여유 (초) ← 말끝 잘림 방지 핵심
    min_keep: float = 0.25     # 이보다 짧게 남는 조각은 버림
    min_cut: float = 0.20      # 여유를 뺀 뒤 잘라낼 쉼이 이보다 짧으면 자르지 않음 (툭툭 튀는 편집 방지)
    fade: float = 0.012        # 이어붙인 곳 페이드 (초)


def speech_keeps(regions, db: np.ndarray, duration: float, cfg: CutConfig):
    """발화 구간 + 여유 → 남길 구간 목록."""
    keeps = []
    for i, (s, e) in enumerate(regions):
        prev_e = regions[i - 1][1] if i > 0 else 0.0
        next_s = regions[i + 1][0] if i + 1 < len(regions) else duration
        ks = max(prev_e, s - cfg.pre_pad, 0.0)
        ke = min(next_s, e + cfg.post_pad, duration)
        # 쉼이 짧아 잘라낼 게 거의 없으면 자르지 않음 (양쪽이 같은 지점을 가리켜 병합됨)
        if i > 0 and s - cfg.pre_pad <= prev_e + cfg.post_pad + cfg.min_cut:
            ks = quietest_point(db, prev_e, s)
        if i + 1 < len(regions) and e + cfg.post_pad + cfg.min_cut >= next_s - cfg.pre_pad:
            ke = quietest_point(db, e, next_s)
        keeps.append([ks, ke])
    return _merge(keeps)


def _merge(iv, tol: float = 1e-3):
    iv = sorted([list(x) for x in iv if x[1] > x[0]])
    out = []
    for s, e in iv:
        if out and s <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def subtract(keeps, removals: list[Removal], db: np.ndarray, snap: float = 0.08):
    """남길 구간에서 반복 발화 구간을 뺌. 경계는 근처의 조용한 지점으로 스냅."""
    out = [list(k) for k in keeps]
    for r in removals:
        rs = quietest_point(db, r.start - snap, r.start + snap)
        re_ = quietest_point(db, r.end - snap, r.end + snap)
        nxt = []
        for s, e in out:
            if re_ <= s or rs >= e:
                nxt.append([s, e])
                continue
            if rs > s:
                nxt.append([s, rs])
            if re_ < e:
                nxt.append([re_, e])
        out = nxt
    return out


def finalize(keeps, cfg: CutConfig):
    return [[round(s, 3), round(e, 3)] for s, e in _merge(keeps) if e - s >= cfg.min_keep]


def save_cutlist(path, source, keeps, removals: list[Removal], utts):
    data = {
        "source": os.path.abspath(source),
        "keeps": [{"start": s, "end": e, "keep": True} for s, e in keeps],
        "removed_retakes": [asdict(r) for r in removals],
        "transcript": [{"start": u.start, "end": u.end, "text": u.text} for u in utts],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cutlist(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    keeps = [[k["start"], k["end"]] for k in data["keeps"] if k.get("keep", True)]
    return data["source"], keeps


def _ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def write_report(path, duration, keeps, removals: list[Removal], utts):
    kept = sum(e - s for s, e in keeps)
    lines = [
        "# 자동 컷 편집 리포트",
        "",
        f"- 원본 길이: {_ts(duration)}",
        f"- 편집 후 길이: {_ts(kept)}  (−{_ts(duration - kept)}, {100 * (1 - kept / max(duration, 1e-9)):.0f}% 단축)",
        f"- 남긴 조각 수: {len(keeps)}",
        f"- 반복 발화 제거: {len(removals)}건",
        "",
        "## 제거한 반복 발화 (원본 시간 기준) — 틀린 게 있으면 cutlist.json에서 되살리세요",
        "",
    ]
    for r in removals:
        lines.append(f"- [{_ts(r.start)} ~ {_ts(r.end)}] {r.reason}  유사도 {r.score:.0f}")
        lines.append(f"  - 지운 말: \"{r.text}\"")
    lines += ["", "## 전체 받아쓰기 (원본 시간 기준)", ""]
    for u in utts:
        lines.append(f"- [{_ts(u.start)}] {u.text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def render(source: str, keeps, out_path: str, cfg: CutConfig, audio_only: bool = False,
           crf: int = 18, preset: str = "veryfast"):
    """ffmpeg로 남길 구간만 이어붙여 출력."""
    has_video = not audio_only and _has_video(source)
    parts, labels = [], []
    f = cfg.fade
    for i, (s, e) in enumerate(keeps):
        d = e - s
        if has_video:
            parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];")
        parts.append(
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:d={f}:curve=qsin,afade=t=out:st={max(0.0, d - f):.3f}:d={f}:curve=qsin[a{i}];")
        labels.append(f"[v{i}][a{i}]" if has_video else f"[a{i}]")
    n = len(keeps)
    if has_video:
        parts.append("".join(labels) + f"concat=n={n}:v=1:a=1[vout][aout]")
    else:
        parts.append("".join(labels) + f"concat=n={n}:v=0:a=1[aout]")
    script = out_path + ".filter.txt"
    with open(script, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    cmd = ["ffmpeg", "-y", "-nostdin", "-v", "error", "-stats", "-i", source,
           "-filter_complex_script", script]
    if has_video:
        cmd += ["-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-crf", str(crf),
                "-preset", preset, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-map", "[aout]"]
    cmd.append(out_path)
    subprocess.run(cmd, check=True)
    os.remove(script)


def _has_video(path: str) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return "video" in r.stdout
