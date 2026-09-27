"""로컬 재현: 캐시된 받아쓰기 + 원본 음성으로 컷/자막/리포트/렌더 (AI검토 제외)."""
import json, os, sys, yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.transcribe import Word
from src.cutdetect import collapse_phantom_repeats, detect_cuts
from src.audio_energy import load_energy
from src.subtitles import build_subtitles
from src.render_audio import render_audio
from src.pipeline import write_cut_report

W = os.environ.get("AUTOCUT_WORK", os.path.dirname(os.path.abspath(__file__)))
mp3 = sys.argv[1]
cfg = yaml.safe_load(open(os.path.join(W, "config.yaml"), encoding="utf-8"))
ws = collapse_phantom_repeats([Word(**w) for w in json.load(open(os.path.join(W, "transcript.json")))])
en = load_energy(os.path.join(W, "a16.wav"))
keeps, stats = detect_cuts(ws, len(en.db) * 0.01, cfg["cut"], energy=en)
out = os.path.join(W, "out"); os.makedirs(out, exist_ok=True)
ws = stats.get("subtitle_words", ws)
cues = build_subtitles(ws, keeps, cfg["subtitle"], os.path.join(out, "0927_자동편집.srt"))
write_cut_report(os.path.join(out, "0927.MP3"), "0927_자동편집", stats, words=ws, keep_spans=keeps)
render_audio(mp3, keeps, os.path.join(out, "0927_자동편집.mp3"))
print("keeps", len(keeps), "kept", round(stats["kept_duration"], 1), "cuts", len(stats["cut_details"]))
json.dump({"keeps": keeps}, open(os.path.join(W, "keeps.json"), "w"))
