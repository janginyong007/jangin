"""오디오 추출 + faster-whisper 단어 단위 전사 (결과는 캐싱)."""
from __future__ import annotations

import hashlib
import json
import os
import site
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import List

import imageio_ffmpeg


def _register_nvidia_dll_dirs() -> None:
    """pip로 설치한 nvidia-cublas-cu12/nvidia-cudnn-cu12의 DLL 폴더를
    Windows DLL 검색 경로에 등록한다 (CTranslate2가 GPU 추론 시점에 로드하기 때문에
    faster-whisper import 전에 미리 등록해둬야 한다)."""
    if sys.platform != "win32":
        return
    bases = set(site.getsitepackages())
    try:
        bases.add(site.getusersitepackages())
    except Exception:
        pass
    for base in bases:
        for pkg in ("cublas", "cudnn"):
            bin_dir = os.path.join(base, "nvidia", pkg, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except (OSError, AttributeError):
                    pass


_register_nvidia_dll_dirs()

from faster_whisper import WhisperModel


@dataclass
class Word:
    text: str
    start: float
    end: float


def extract_audio(video_path: str, cache_dir: str) -> str:
    """영상에서 16kHz mono wav 오디오를 추출한다. 이미 있으면 재사용."""
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(cache_dir, f"{base}.wav")
    if os.path.exists(wav_path):
        return wav_path

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"오디오 추출 실패:\n{result.stderr}")
    return wav_path


def extract_audio_hq(video_path: str, cache_dir: str) -> str:
    """캡컷 오디오 트랙에 실제로 쓸 고음질(48kHz 스테레오) wav를 추출한다.
    (음성인식용 extract_audio()는 16kHz mono라 최종 결과물 음질로 쓰기엔 부족함)"""
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(cache_dir, f"{base}_hq.wav")
    if os.path.exists(wav_path):
        return wav_path

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"고음질 오디오 추출 실패:\n{result.stderr}")
    return wav_path


_LONG_WORD_THRESHOLD = 1.5  # 이보다 긴 "한 단어" 구간은 버벅임을 뭉갰을 가능성이 있어 재검사
# (실측: 정상 단어는 거의 다 1.4초 이내. 1.8초짜리 "넣은" 사례처럼 경계선에
# 걸치는 경우도 놓치지 않도록 2.0에서 낮췄다.)


def _extract_subclip(source_wav: str, start: float, end: float, out_path: str) -> None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y", "-ss", f"{max(0.0, start)}", "-to", f"{end}", "-i", source_wav, out_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"구간 재추출 실패:\n{result.stderr}")


def _refine_long_words(words: List[Word], wav_path: str, model: "WhisperModel", language: str, cache_dir: str) -> List[Word]:
    """단어 하나의 구간이 비정상적으로 길면(정상은 대부분 1.5초 이내), 위스퍼가
    버벅임/재시도를 자연스러운 문장으로 뭉개서 실제로는 여러 번 반복된 말을
    하나의 단어처럼 잘못 묶었을 가능성이 있다 (실제로 "나들목역" 한 단어에
    3.74초가 잡힌 사례가 있었는데, 그 구간엔 "남대전 나들목"을 반복해서
    버벅인 게 있었다 - 텍스트에는 반복이 전혀 안 보여서 기존 반복감지로는
    절대 못 잡는 경우였음). 그 구간만 따로 잘라서 다시 전사해보고, 여러
    단어로 더 잘게 쪼개지면(=진짜 버벅임이었을 가능성) 교체한다."""
    suspicious = [w for w in words if (w.end - w.start) > _LONG_WORD_THRESHOLD]
    if not suspicious:
        return words

    print(f"[transcribe] 비정상적으로 긴 단어 {len(suspicious)}개 발견, 구간 재검사 중...")
    refined: List[Word] = []
    for w in words:
        if (w.end - w.start) <= _LONG_WORD_THRESHOLD:
            refined.append(w)
            continue

        pad = 0.3
        sub_start = max(0.0, w.start - pad)
        sub_end = w.end + pad
        sub_path = os.path.join(cache_dir, f"_recheck_{sub_start:.2f}_{sub_end:.2f}.wav")
        sub_words: List[Word] = []
        try:
            _extract_subclip(wav_path, sub_start, sub_end, sub_path)
            sub_segments, _info = model.transcribe(
                sub_path, language=language, word_timestamps=True, vad_filter=False
            )
            for seg in sub_segments:
                for sw in seg.words or []:
                    text = sw.word.strip()
                    if not text:
                        continue
                    sub_words.append(Word(text=sw.word.rstrip(), start=sub_start + sw.start, end=sub_start + sw.end))
        except Exception as e:  # noqa: BLE001 - 재검사가 실패해도 원래 단어를 그대로 쓰면 되므로 죽지 않는다
            print(f"[transcribe] 구간 재검사 실패({w.text.strip()}): {e}")
        finally:
            if os.path.exists(sub_path):
                os.remove(sub_path)

        if len(sub_words) >= 2:
            print(f"[transcribe]   '{w.text.strip()}'({w.end - w.start:.1f}초) -> {len(sub_words)}개 단어로 재검사됨")
            refined.extend(sub_words)
        else:
            refined.append(w)

    return refined


def _cache_path(cache_dir: str, video_path: str, model_name: str) -> str:
    key = f"{os.path.abspath(video_path)}|{model_name}|{os.path.getmtime(video_path)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"transcript_{digest}.json")


def _load_model(model_name: str, whisper_cfg: dict) -> WhisperModel:
    device = whisper_cfg.get("device", "auto")
    gpu_compute = whisper_cfg.get("compute_type_gpu", "float16")
    cpu_compute = whisper_cfg.get("compute_type_cpu", "int8")

    if device == "cpu":
        print(f"[transcribe] CPU로 {model_name} 모델 로드 (compute_type={cpu_compute})")
        return WhisperModel(model_name, device="cpu", compute_type=cpu_compute)

    if device == "cuda":
        print(f"[transcribe] GPU(cuda)로 {model_name} 모델 로드 (compute_type={gpu_compute})")
        return WhisperModel(model_name, device="cuda", compute_type=gpu_compute)

    # auto: GPU 시도 후 실패하면 CPU로 폴백
    try:
        model = WhisperModel(model_name, device="cuda", compute_type=gpu_compute)
        print(f"[transcribe] GPU(cuda)로 {model_name} 모델 로드 (compute_type={gpu_compute})")
        return model
    except Exception as e:
        print(f"[transcribe] GPU 로드 실패({e}), CPU로 폴백 (compute_type={cpu_compute})")
        return WhisperModel(model_name, device="cpu", compute_type=cpu_compute)


def transcribe(video_path: str, cache_dir: str, whisper_cfg: dict) -> List[Word]:
    """영상을 전사해서 단어 단위 타임스탬프 리스트를 반환한다. 결과는 캐싱됨."""
    os.makedirs(cache_dir, exist_ok=True)
    model_name = whisper_cfg.get("model", "large-v3")
    cache_file = _cache_path(cache_dir, video_path, model_name)

    if os.path.exists(cache_file):
        print(f"[transcribe] 캐시된 전사 결과 사용: {cache_file}")
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        return [Word(**w) for w in data]

    wav_path = extract_audio(video_path, cache_dir)
    model = _load_model(model_name, whisper_cfg)

    language = whisper_cfg.get("language", "ko")
    segments, info = model.transcribe(
        wav_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    words: List[Word] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if not w.word.strip():
                continue
            # 원본의 앞쪽 공백 유무를 그대로 보존한다 (자막에서 이어붙일 때
            # "1 ,683" 처럼 콤마/조사 앞에 불필요한 공백이 생기지 않도록).
            text = w.word.rstrip()
            words.append(Word(text=text, start=w.start, end=w.end))

    words = _refine_long_words(words, wav_path, model, language, cache_dir)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump([asdict(w) for w in words], f, ensure_ascii=False, indent=2)

    print(f"[transcribe] 전사 완료: 단어 {len(words)}개, 캐시 저장 {cache_file}")
    return words
