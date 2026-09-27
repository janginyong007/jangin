"""발화 구간별 음성 인식 (faster-whisper).

핵심: 전체 파일을 한 번에 인식하지 않고 '발화 구간마다 따로' 인식한다.
Whisper는 앞 문맥을 보고 말을 매끄럽게 정리하는 경향이 있어서
'이 물건은... 이 물건은 감정가가' 같은 반복을 한 번으로 합쳐버린다.
(그래서 기존 방식에서 반복 발화를 못 잡았던 것)
구간별로 끊고 condition_on_previous_text=False로 인식하면 각 테이크가 따로 남는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import SR

# 말더듬/반복을 그대로 받아적도록 유도하는 프롬프트 (Whisper는 프롬프트 문체를 따라감)
VERBATIM_PROMPT = "음, 그, 이 물건은, 이 물건은 감정가가, 감정가가 삼억, 어, 그러니까 그러니까 말씀드리면"

MAX_CHUNK = 25.0  # Whisper 30초 창 안에 들어가도록


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Utterance:
    start: float            # 에너지 기준 구간 시작 (초)
    end: float              # 에너지 기준 구간 끝 (초)
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


def load_model(name: str = "large-v3", device: str = "auto", compute_type: str = "auto"):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device=device, compute_type=compute_type)


def split_long(regions, db, max_len: float = MAX_CHUNK):
    """25초 넘는 구간은 가장 조용한 지점에서 나눔."""
    from .audio import quietest_point
    out = []
    for s, e in regions:
        while e - s > max_len:
            cut = quietest_point(db, s + max_len * 0.5, s + max_len)
            out.append((s, cut))
            s = cut
        out.append((s, e))
    return out


def transcribe_regions(model, audio: np.ndarray, regions, language: str = "ko",
                       beam_size: int = 5, prompt: str | None = VERBATIM_PROMPT,
                       progress=None) -> list[Utterance]:
    utts = []
    for i, (s, e) in enumerate(regions):
        # 앞뒤로 조금 여유를 줘서 인식 (첫 자음/말끝 인식률 향상)
        cs, ce = max(0.0, s - 0.1), min(len(audio) / SR, e + 0.2)
        chunk = audio[int(cs * SR):int(ce * SR)]
        segs, _ = model.transcribe(
            chunk, language=language, beam_size=beam_size,
            word_timestamps=True, vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt, temperature=0.0,
        )
        words = []
        for seg in segs:
            for w in seg.words or []:
                t = w.word.strip()
                if not t:
                    continue
                words.append(Word(t, cs + w.start, cs + w.end, w.probability))
        utts.append(Utterance(s, e, words))
        if progress:
            progress(i + 1, len(regions))
    return utts
