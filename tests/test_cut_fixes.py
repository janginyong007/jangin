import numpy as np

from src.audio_energy import energy_from_samples, refine_silence_spans, snap_content_spans
from src.cutdetect import _is_fuzzy_repeat, detect_cuts, find_repeated_phrases
from src.transcribe import Word

SR = 16000
CFG = {
    "silence_min_gap": 0.5, "filler_words": ["음", "어"], "filler_max_duration": 0.6,
    "repeat_window": 5.0, "padding_before": 0.15, "padding_after": 0.1, "min_keep_duration": 0.2,
}


def words_of(text, start=0.0, dur=0.4, gap=0.05):
    out, t = [], start
    for w in text.split():
        out.append(Word(w, t, t + dur))
        t += dur + gap
    return out


# ---------- 반복 발화 ----------

def test_exact_repeat_still_works():
    ws = words_of("이 물건은 감정가가") + words_of("이 물건은 감정가가 삼억입니다", start=2.5)
    spans = find_repeated_phrases(ws, 5.0)
    assert spans == [(ws[0].start, ws[2].end)]


def test_mispronounced_repeat_is_caught_only_with_fuzzy():
    ws = words_of("이 물건의 감종가는") + words_of("이 물건의 감정가는 삼억입니다", start=2.5)
    # 기존 방식: 똑같은 '이 물건'까지만 컷하고 발음 꼬인 '감종가는'이 남음
    assert find_repeated_phrases(ws, 5.0) == [(ws[0].start, ws[1].end)]
    spans = find_repeated_phrases(ws, 5.0, fuzzy_threshold=80)
    assert spans == [(ws[0].start, ws[2].end)]             # 개선: 앞 테이크 전체 컷


def test_different_numbers_are_not_repeats():
    assert not _is_fuzzy_repeat("보증금1억원", "보증금2억원입니다", 80, 4)
    ws = words_of("보증금은 1억 원이고") + words_of("월세는 2억 원입니다", start=2.0)
    assert find_repeated_phrases(ws, 5.0, fuzzy_threshold=80) == []


def test_different_sentences_not_repeats():
    ws = words_of("이 물건은 역세권입니다") + words_of("이 물건은 학군이 좋습니다", start=2.0)
    # '이 물건은'(2단어, 3음절)은 fuzzy 최소 음절(4) 미만이라 fuzzy로는 안 잡힘.
    # 완전 일치 2단어 반복은 기존 동작 그대로 (원래 프로그램 규칙 유지)
    spans_old = find_repeated_phrases(ws, 5.0)
    spans_new = find_repeated_phrases(ws, 5.0, fuzzy_threshold=80)
    assert spans_old == spans_new


# ---------- 말끝 잘림 ----------

def _speech(segments, total):
    """큰 소리 뒤에 점점 작아지는 약한 말끝(어미)을 흉내낸 신호."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.001, int(total * SR)).astype(np.float32)
    for s, loud, tail in segments:
        t = np.arange(int(loud * SR)) / SR
        x[int(s * SR):int(s * SR) + len(t)] += 0.3 * np.sin(2 * np.pi * 220 * t)
        tt = np.arange(int(tail * SR)) / SR
        env = np.linspace(1, 0, len(tt)) ** 2 * 0.06
        o = int((s + loud) * SR)
        x[o:o + len(tt)] += env * np.sin(2 * np.pi * 220 * tt)
    return x


def test_soft_ending_protected():
    # 실제 말: 0.5~1.5초 + 약한 말끝 0.5초(~2.0초), 다음 말 4.0초 시작
    x = _speech([(0.5, 1.0, 0.5), (4.0, 1.0, 0.3)], 6.0)
    en = energy_from_samples(x, SR)
    # 위스퍼가 말끝을 1.5초에 끝났다고 잘못 잡은 상황
    ws = [Word("감정가는", 0.5, 1.0), Word("삼억입니다", 1.0, 1.5), Word("다음", 4.0, 5.0)]
    keep_old, _ = detect_cuts(ws, 6.0, CFG)
    keep_new, _ = detect_cuts(ws, 6.0, CFG, energy=en)
    assert keep_old[0][1] < 1.8                    # 기존: 말끝이 잘림
    assert keep_new[0][1] >= 1.85                  # 개선: 말끝까지 살림
    assert keep_new[1][0] <= 4.0                   # 다음 말 첫소리도 보호
    assert len(keep_new) == 2                      # 무음은 여전히 컷됨


def test_refine_never_grows_cut():
    x = _speech([(0.5, 1.0, 0.1), (3.0, 1.0, 0.1)], 5.0)
    en = energy_from_samples(x, SR)
    spans = [(1.8, 2.8)]
    out = refine_silence_spans(spans, en)
    assert out and out[0][0] >= 1.8 and out[0][1] <= 2.8


def test_snap_moves_to_quiet_gap():
    x = _speech([(0.5, 1.0, 0.0), (1.7, 1.0, 0.0)], 3.0)   # 1.5~1.7초 사이 틈
    en = energy_from_samples(x, SR)
    (s, e), = snap_content_spans([(1.62, 2.5)], en)
    assert 1.5 <= s <= 1.7


def test_untranscribed_speech_is_not_cut_as_silence():
    # 0.5~1.5초 말, 2.5~3.5초에도 말이 있는데 위스퍼가 빼먹음, 5.0초 다음 말
    x = _speech([(0.5, 1.0, 0.1), (2.5, 1.0, 0.1), (5.0, 1.0, 0.1)], 7.0)
    en = energy_from_samples(x, SR)
    ws = [Word("감정가는", 0.5, 1.5), Word("다음", 5.0, 6.0)]
    keep_old, _ = detect_cuts(ws, 7.0, CFG)
    keep_new, _ = detect_cuts(ws, 7.0, CFG, energy=en)
    covered = lambda keeps, t: any(s <= t <= e for s, e in keeps)
    assert not covered(keep_old, 3.0)      # 기존: 빼먹은 말이 통째로 잘림
    assert covered(keep_new, 2.55) and covered(keep_new, 3.55)  # 개선: 살아남음
    assert not covered(keep_new, 4.3)      # 진짜 무음은 여전히 컷


def test_cut_report(tmp_path):
    from src.pipeline import write_cut_report
    ws = words_of("음") + words_of("이 물건은 감정가가", start=1.0) + words_of("이 물건은 감정가가 삼억", start=3.0)
    _, stats = detect_cuts(ws, 6.0, dict(CFG, repeat_window=5.0))
    kinds = [r["kind"] for r in stats["cut_details"]]
    assert "필러워드" in kinds and "반복발화" in kinds
    p = write_cut_report(str(tmp_path / "a.mp4"), "테스트", stats)
    txt = open(p, encoding="utf-8-sig").read()
    assert "잘라낸 말: \"이 물건은 감정가가\"" in txt


def test_fuzzy_does_not_eat_preceding_word():
    ws = words_of("그래서 이 물건은 감종가가") + words_of("이 물건은 감정가가 삼억입니다", start=2.5)
    spans = find_repeated_phrases(ws, 5.0, fuzzy_threshold=80)
    assert spans and spans[0][0] >= ws[1].start   # '그래서'는 살아남아야 함
