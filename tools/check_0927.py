"""사용자가 알려준 지점들 회귀 검사. keeps.json(원본 시각 기준 남길 구간)을 읽어 판정."""
import json, os
W = os.environ.get("AUTOCUT_WORK", os.path.dirname(os.path.abspath(__file__)))
k = json.load(open(os.path.join(W, "keeps.json")))["keeps"]
kept = lambda t: any(s <= t <= e for s, e in k)
def all_kept(a, b): return all(kept(a + (b - a) * i / 10) for i in range(11))
def all_cut(a, b): return not any(kept(a + (b - a) * i / 10) for i in range(11))
CASES = [
    ("1:40 고속도로 첫 테이크 잘림", all_cut(100.9, 102.4)),
    ("1:40 지척이라 남김", all_kept(99.7, 100.35)),
    ("1:43 고속도로 재시도 남김(첫음절 고 포함)", all_kept(103.36, 105.0)),
    ("1:20 2013년 남김", all_kept(80.7, 81.5)),
    ("2:19 취사시설 1차 테이크 잘림", all_cut(140.45, 145.4)),
    ("2:19 취사시설 조각/2차 잘림", all_cut(146.9, 148.4) and all_cut(149.7, 152.1)),
    ("2:33 취사시설 최종 테이크 남김", all_kept(153.6, 158.4)),
    ("2:54 ...300만원입니다 끝말 남김", all_kept(173.5, 174.48)),
    ("4:21 보증금 첫 테이크+끅 잘림", all_cut(261.85, 263.12)),
    ("4:23 보증금 재시도 남김", all_kept(263.45, 264.3)),
    ("4:27 다시 첫 테이크 잘림", all_cut(267.76, 268.9)),
    ("4:30 다시 재시도 남김", all_kept(270.46, 271.2)),
    ("4:44 2026 타경 남김", all_kept(283.85, 284.78)),
    ("4:45 첫 121호입니다 잘림", all_cut(284.9, 286.45)),
    ("4:52 7월에 재시도 남김", all_kept(292.2, 292.75)),
    ("4:55 두번째 법원은 남김", all_kept(295.63, 296.05)),
    ("4:55 첫 법원은 잘림", all_cut(295.0, 295.25)),
]
bad = 0
for name, ok in CASES:
    print(("OK  " if ok else "FAIL"), name); bad += (not ok)
print(f"{len(CASES)-bad}/{len(CASES)} 통과")
