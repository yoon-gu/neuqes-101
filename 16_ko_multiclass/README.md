# 16_ko_multiclass — 한국어 BERT Multi-class (KLUE-YNAT 7분류)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/16_ko_multiclass/16_ko_multiclass.ipynb)

## 한 줄 목표
Ch 15(한국어 binary, NSMC) 에서 *task 차원만* K=2 → K=7 로 일반화. 모델·토크나이저·hyperparams 는 그대로 두고 KLUE-YNAT 뉴스 헤드라인 7카테고리 분류로 확장.

## 다루는 핵심 개념
- KLUE-YNAT (Yonhap News Topic) — 한국어 뉴스 헤드라인 + 7카테고리 (IT과학·경제·사회·생활문화·세계·스포츠·정치)
- `num_labels=7`, `problem_type="single_label_classification"` — Ch 12 와 같은 multi-class 셋업, 모델만 한국어
- multi-class 평가 metric: macro precision/recall/F1, multi-class AUC (`multi_class="ovr"`)
- 7×7 혼동 행렬 (`seaborn.heatmap`) — 카테고리별 혼동 패턴 진단 (정치 ↔ 경제 같은 자연스러운 혼동)
- top-1 확률 분포로 calibration 진단 (correct vs wrong)
- 자신있는 / 망설이는 / 자신있게 틀린 헤드라인 샘플 단위 해석
- Random baseline loss = $\log 7 \approx 1.946$

## Loss
**`CrossEntropyLoss`** — Ch 15 그대로. K가 2 → 7 로 늘었을 뿐.

## 데이터
KLUE-YNAT (`load_dataset("klue", "ynat")`) — 5K train / 1K eval. 클래스 분포는 약간 불균형 (5K-8K 범위).

## 환경
Google Colab **T4 GPU 필수**. 약 3분.

## 변화 추적

| Ch | 모델 | 데이터 | Output | Activation | Loss |
|---|---|---|---|---|---|
| 12 | DistilBERT | Yelp 5클래스 (영어) | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |
| 15 | klue/bert-base | NSMC binary (한국어) | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| **16** | klue/bert-base | **KLUE-YNAT 7분류** | **`Linear(H, 7)`** | softmax | `CrossEntropyLoss` |
| 17 (다음) | klue/bert-base | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | sigmoid (per-label) | `BCEWithLogitsLoss` (per-label) |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[17_ko_multilabel](../17_ko_multilabel/) — 같은 데이터, *task 만* single-label → multi-label. Ch 13의 한국어 버전.
