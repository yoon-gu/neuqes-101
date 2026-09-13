# 17_ko_multilabel — 한국어 BERT Multi-label (KLUE-YNAT 합성 multi-label)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/17_ko_multilabel/17_ko_multilabel.ipynb)

## 한 줄 목표
Ch 16(한국어 multi-class, KLUE-YNAT 7분류)에서 *task 차원만* single-label → multi-label 로 전환. 모델·토크나이저·hyperparams 는 그대로 두고, KLUE-YNAT 헤드라인 *두 개를 결합* 해 두 카테고리가 동시 활성된 합성 multi-label 데이터로 학습. Ch 13(영어 multi-label)의 한국어 버전.

## 다루는 핵심 개념
- KLUE-YNAT single-label 두 샘플을 `[SEP]` 로 결합 → multi-hot 라벨 union (합성 multi-label)
- `num_labels=7` 그대로, `problem_type="multi_label_classification"` 한 줄 전환 → `BCEWithLogitsLoss` 자동 매핑
- 라벨은 multi-hot float 7차원 벡터 `[0, 1, 0, 0, 0, 1, 0]` 형식
- per-label sigmoid 확률 + threshold 0.5 (그리고 threshold sweep 으로 micro/macro F1 변화)
- multi-label 평가: hamming loss + micro/macro F1 + per-category F1 + macro AUC
- **softmax 는 multi-label 에 *수학적으로* 못 쓴다** — 합=1 강제가 동시 활성과 충돌 (경제+스포츠 예시)
- 카테고리별 sigmoid 확률 KDE (7 패널) + 카테고리 간 공동 활성 (co-occurrence) heatmap
- 무작위 결합 합성의 한계 — 자연스러운 카테고리 상관이 약함

## Loss
**`BCEWithLogitsLoss` per-label** — Ch 13 과 같은 식. K=7 개 binary BCE 의 평균. Ch 16 의 `CrossEntropyLoss` 에서 전환.

## 데이터
KLUE-YNAT (`load_dataset("klue/klue", "ynat")`) 두 헤드라인 결합으로 합성 — 5K train / 1K eval. seed 고정(42). 평균 활성 라벨 -1.86개 (두 번 뽑아 가끔 충돌).

## 환경
Google Colab **T4 GPU 필수**. 약 3분.

## 변화 추적

| Ch | 모델 | 데이터 | Output | Activation | Loss |
|---|---|---|---|---|---|
| 13 | DistilBERT | Yelp + 항목 합성 (영어) | `Linear(H, 5)` | sigmoid (per-label) | `BCEWithLogitsLoss` |
| 15 | klue/bert-base | NSMC binary (한국어) | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| 16 | klue/bert-base | KLUE-YNAT 7분류 | `Linear(H, 7)` | softmax | `CrossEntropyLoss` |
| **17** | klue/bert-base | **KLUE-YNAT 합성 multi-label** | `Linear(H, 7)` (그대로) | **per-label sigmoid** | **`BCEWithLogitsLoss`** |
| 18 (다음) | klue/bert-base + 보조 헤드 | 합성 multi-label + 라벨 개수 | 메인 + 보조 | 메인 sigmoid | BCE + λ·MSE |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[18_ko_auxiliary](../18_ko_auxiliary/) — 메인 task 는 Ch 17 과 *완전히 동일*, 활성 라벨 개수 회귀 보조 헤드를 더해 multi-task 학습. Ch 14 의 한국어 버전.
