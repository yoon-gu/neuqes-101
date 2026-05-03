# 13_bert_multilabel — BERT Multi-label (Yelp 측면 키워드)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/13_bert_multilabel/13_bert_multilabel.ipynb)

## 한 줄 목표
Ch 12(BERT 5클래스 분류)와 *모델 아키텍처가 완전히 동일* (`Linear(H, 5)` 분류 헤드)하지만 **task의 의미** 가 다릅니다 — 5개 라벨이 *서로 배타적인 클래스* 가 아니라 *각각 독립 활성될 수 있는 태그*. Ch 6(sklearn OvR LogReg)의 BERT 버전이고, Ch 10에서 본 `num_labels=1 + multi_label_classification` 트릭을 K=5 로 확장한 형태.

## 다루는 핵심 개념
- `num_labels=5` + `problem_type="multi_label_classification"` — Ch 12의 single-label 셋업에서 problem_type 한 줄만 변경
- 라벨은 multi-hot float 벡터 `[1, 0, 1, 0, 1]` 형식 (`BCEWithLogitsLoss` 가 요구하는 shape `(B, K)`)
- per-label sigmoid 확률 + threshold 0.5
- multi-label 평가: hamming loss + micro/macro F1 + per-label F1 + macro AUC
- **softmax는 multi-label에 *수학적으로* 못 쓴다** — 합=1 강제가 동시 활성과 충돌
- 모델이 라벨 간 상관을 학습하는 메커니즘 — *공유 BERT 본체* 를 통한 *간접* 결합 (loss엔 결합 항 없음)
- 측면 합성 라벨의 본질적 한계 (키워드 매칭의 얕음)

## Loss
**`BCEWithLogitsLoss` per-label** — Ch 6 sklearn OvR과 같은 식. K개 binary BCE의 평균.

## 데이터
Yelp 5,000건 + Ch 6의 측면 키워드 사전 (food/service/price/ambiance/location)으로 multi-hot 5차원 라벨 합성.

## 환경
Google Colab **T4 GPU 필수**. 약 12분 (BERT ~10분 + sklearn 비교 ~30초).

**Self-contained**: 다른 챕터 / 다른 세션 결과에 의존하지 않습니다. 5장 sklearn baseline은 inline 학습.

## 변화 추적

| Ch | 모델 | 데이터 | Output | Activation | Loss |
|---|---|---|---|---|---|
| 6 | OvR(LogReg) | TF-IDF + 측면 합성 | (5차원) | sigmoid (각각) | BCE per-label |
| 10 | DistilBERT | Yelp 이진화 | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| 12 | DistilBERT | Yelp 5클래스 | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |
| **13** | DistilBERT | **Yelp + 측면 합성** | **`Linear(H, 5)`** (그대로) | **per-label sigmoid** | **BCE per-label** |
| 14 (다음) | DistilBERT + 보조 헤드 | 측면 + 별점 | 메인 + 보조 | 메인 sigmoid | BCE + λ·MSE |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[14_auxiliary_loss](../14_auxiliary_loss/) — 메인 task는 Ch 13과 *완전히 동일*, 별점 회귀 보조 헤드를 더해 multi-task 학습.
