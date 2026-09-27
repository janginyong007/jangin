# jangin — 부동산장인 음성 자동 컷 편집기

녹화 원본에서 **무음 구간**과 **다시 말한 부분(반복 발화)** 을 자동으로 잘라
컷편집이 끝난 영상을 만들어 줍니다. 결과 영상을 캡컷에 불러와 자막·효과 작업만 하면 됩니다.

## 기존 문제와 해결 방법

| 문제 | 원인 | 해결 |
|---|---|---|
| 두 번 반복해서 말한 것을 못 잡음 | Whisper는 긴 음성을 한 번에 인식하면 앞 문맥을 보고 반복을 **한 번으로 합쳐서** 받아적음 → 텍스트에 반복이 안 남음 | 쉼 단위로 **구간마다 따로 인식** (`condition_on_previous_text=False`), 말더듬 프롬프트로 받아쓰기 유도 |
| 발음 꼬인 테이크는 글자가 달라서 비교 실패 | "감종가" vs "감정가" 처럼 인식 결과가 달라짐 | 한글을 **자모(ㄱㅏㅁ…) 단위로 풀어서** 유사도 비교 |
| 컷이 너무 빨라 말끝이 잘림 | Whisper 단어 타임스탬프는 약한 어미(~습니다)를 일찍 끝난 걸로 잡음 | 컷 경계는 **실제 음량**으로 판정: 시작/끝 임계값을 다르게(히스테리시스), 뒤 여유 0.35초, 쉼 중 가장 조용한 곳에서 자르기, 이음매 페이드 |

반복 발화 3가지 패턴을 모두 잡습니다. **항상 마지막에 말한 테이크를 살립니다.**

1. 문장 통째로 다시 — "이 물건은 삼억입니다. / 이 물건은 삼억입니다."
2. 말하다 멈추고 처음부터 — "이 물건은 감정가가… / 이 물건은 감정가가 삼억입니다."
3. 쉼 없이 바로 고쳐 말하기 — "감종가는 감정가는 삼억입니다" → "감종가는" 제거

## 설치 (한 번만)

1. Python 3.10+ 와 ffmpeg 설치 (윈도우: `winget install Gyan.FFmpeg`)
2. `pip install -r requirements.txt`
3. NVIDIA 그래픽카드가 있으면 훨씬 빠릅니다 (없으면 `--model medium` 추천)

## 사용법

```bash
# 1) 분석 + 컷 + 렌더링 → 원본_cut.mp4, 원본_cutlist.json, 원본_report.md
python -m autocut 원본.mp4

# 2) 리포트(원본_report.md)에서 지운 부분 확인
#    잘못 지운 조각이 있으면 원본_cutlist.json 에서 "keep": false/true 를 고치거나
#    구간 시간을 수정한 뒤 다시 렌더링
python -m autocut --render 원본_cutlist.json
```

### 감도 조절

| 증상 | 옵션 |
|---|---|
| 아직도 말끝이 잘림 | `--post-pad 0.5` 또는 `--off-margin 5` |
| 쉼이 너무 길게 남음 | `--post-pad 0.25 --pre-pad 0.08` |
| 문장 중간 숨쉬는 곳까지 잘림 | `--min-silence 0.5` |
| 반복을 덜 잡음 | `--retake-threshold 72` |
| 반복 아닌 걸 지움 | `--retake-threshold 85` 또는 `--no-inline` |
| 무음만 자르고 싶음 | `--no-retake` |
| 느림 / GPU 없음 | `--model medium` 또는 `--model small` |

## 참고한 오픈소스

- [auto-editor](https://github.com/WyattBlue/auto-editor) — 음량 기반 무음 컷, margin(여유) 개념
- [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped), [WhisperX](https://github.com/m-bain/whisperX), [CrisperWhisper](https://github.com/openai/whisper/discussions/2341) — Whisper가 반복/말더듬을 지우는 문제와 단어 타임스탬프 부정확성
- [pyCapCut](https://github.com/GuanYixuan/pyCapCut) — 캡컷 초안(draft) 직접 생성 (향후 연동 후보)

## 테스트

```bash
pip install pytest && python -m pytest -q tests
```
