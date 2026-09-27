import numpy as np

from autocut import audio as A
from autocut.retake import RetakeConfig, find_retakes, jamo, prefix_similarity
from autocut.timeline import CutConfig, finalize, speech_keeps, subtract
from autocut.transcribe import Utterance, Word


def utt(start, text, dur_per_word=0.4, gap=0.05):
    words, t = [], start
    for w in text.split():
        words.append(Word(w, t, t + dur_per_word))
        t += dur_per_word + gap
    return Utterance(start, t, words)


def test_jamo():
    assert jamo("감정가") == "ㄱㅏㅁㅈㅓㅇㄱㅏ"
    # 발음 꼬인 인식 결과도 높은 유사도
    assert prefix_similarity("이 물건은 감종가가", "이 물건은 감정가가 삼억입니다") > 80


def test_full_retake_keeps_last():
    utts = [utt(0, "이 물건은 감정가가 삼억입니다"), utt(4, "이 물건은 감정가가 삼억입니다")]
    rs = find_retakes(utts)
    assert len(rs) == 1 and rs[0].start == 0 and rs[0].end == utts[0].end


def test_false_start():
    utts = [utt(0, "이 물건은 감정가가"), utt(3, "이 물건은 감정가가 삼억 이천만원입니다")]
    rs = find_retakes(utts)
    assert len(rs) == 1 and rs[0].end == utts[0].end


def test_three_takes():
    utts = [utt(0, "낙찰가율은 팔십"), utt(2, "낙찰가율은 팔십 퍼"), utt(4, "낙찰가율은 팔십 퍼센트입니다")]
    rs = find_retakes(utts)
    assert len(rs) == 2


def test_restart_mid_sentence():
    a = utt(0, "이 아파트는 역세권이고 초등학교가 바로 앞에")
    b = utt(4, "초등학교가 바로 앞에 있습니다")
    rs = find_retakes([a, b])
    assert len(rs) == 1
    assert rs[0].start > a.start  # 앞부분 '이 아파트는 역세권이고'는 살림
    assert "초등학교가" in rs[0].text and "역세권" not in rs[0].text


def test_inline_correction():
    u = utt(0, "이 물건의 감종가는 감정가는 삼억입니다")
    rs = find_retakes([u])
    assert len(rs) == 1 and rs[0].text == "감종가는"


def test_no_false_positive_on_different_sentences():
    utts = [utt(0, "이 물건은 역세권입니다"), utt(3, "이 물건은 학군이 아주 좋습니다"),
            utt(6, "그래서 저는 입찰을 추천드립니다")]
    assert find_retakes(utts) == []


def _tone_with_tail(sr, segments, total):
    """발화 흉내: 큰 소리 뒤에 점점 작아지는 말끝(어미)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.001, int(total * sr)).astype(np.float32)
    for s, loud, tail in segments:
        t = np.arange(int(loud * sr)) / sr
        x[int(s * sr):int(s * sr) + len(t)] += 0.3 * np.sin(2 * np.pi * 220 * t)
        tt = np.arange(int(tail * sr)) / sr
        env = np.linspace(1, 0, len(tt)) ** 2 * 0.06  # 약한 말끝 (-25dB 수준)
        o = int((s + loud) * sr)
        x[o:o + len(tt)] += env * np.sin(2 * np.pi * 220 * tt)
    return x


def test_soft_ending_not_cut():
    sr = A.SR
    # 1초 발화 + 0.4초 약한 말끝, 2초 쉼, 다시 발화
    x = _tone_with_tail(sr, [(0.5, 1.0, 0.4), (4.0, 1.0, 0.4)], 6.0)
    db = A.energy_db(x)
    th = A.estimate_thresholds(db)
    regions = A.speech_regions(db, th)
    assert len(regions) == 2
    # 말끝 대부분(0.5+1.0+0.4=1.9s 중 1.75s 이상)이 구간 안에 들어가야 함
    assert regions[0][1] >= 1.75
    keeps = finalize(speech_keeps(regions, db, 6.0, CutConfig()), CutConfig())
    assert keeps[0][1] >= 1.9


def test_short_pause_not_cut():
    sr = A.SR
    x = _tone_with_tail(sr, [(0.5, 1.0, 0.1), (1.85, 1.0, 0.1)], 4.0)
    db = A.energy_db(x)
    regions = A.speech_regions(db, A.estimate_thresholds(db))
    assert len(regions) == 1


def test_subtract_removal():
    from autocut.retake import Removal
    db = np.full(1000, -60.0)
    keeps = [[0.0, 5.0]]
    out = subtract(keeps, [Removal(1.0, 2.0, "", "", 90)], db)
    assert len(out) == 2 and abs(out[0][1] - 1.0) < 0.1 and abs(out[1][0] - 2.0) < 0.1


def test_tiny_gap_not_cut():
    sr = A.SR
    x = _tone_with_tail(sr, [(0.5, 1.0, 0.1), (2.2, 1.0, 0.1)], 4.0)  # 쉼 약 0.6초
    db = A.energy_db(x)
    regions = A.speech_regions(db, A.estimate_thresholds(db))
    assert len(regions) == 2
    keeps = finalize(speech_keeps(regions, db, 4.0, CutConfig()), CutConfig())
    assert len(keeps) == 1


def test_pipeline_with_fake_model(tmp_path, monkeypatch):
    """Whisper 없이 전체 파이프라인: 앞 테이크가 잘려나가는지."""
    import subprocess, json
    from autocut import __main__ as M, transcribe as T
    sr = A.SR
    x = _tone_with_tail(sr, [(0.5, 1.5, 0.3), (3.5, 1.5, 0.3), (6.5, 1.0, 0.3)], 8.5)
    raw = tmp_path / "a.raw"; x.astype("<f4").tofile(raw)
    wav = tmp_path / "in.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1",
                    "-i", str(raw), str(wav)], check=True)
    texts = ["이 물건은 감종가가 삼억", "이 물건은 감정가가 삼억입니다", "입찰을 추천드립니다"]

    def fake_transcribe(model, audio, regions, **kw):
        return [utt(s, texts[i], dur_per_word=(e - s) / 5) for i, (s, e) in enumerate(regions)]
    monkeypatch.setattr(T, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(T, "transcribe_regions", fake_transcribe)
    assert M.main([str(wav), "--audio-only"]) == 0
    data = json.load(open(tmp_path / "in_cutlist.json", encoding="utf-8"))
    assert len(data["removed_retakes"]) == 1
    assert data["keeps"][0]["start"] > 2.0  # 첫 테이크(0.5~2.3s)는 제거됨
    assert (tmp_path / "in_cut.wav").exists()
