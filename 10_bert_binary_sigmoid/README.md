# 10_bert_binary_sigmoid — BERT Binary 방식 A (sigmoid + BCE)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/10_bert_binary_sigmoid/10_bert_binary_sigmoid.ipynb)

## 한 줄 목표
Ch 4(sklearn binary on softmax)에서 본 *두 방식 동등성* 의 BERT 버전을 시작합니다. 이번 챕터는 **방식 A** — `num_labels=1` + `sigmoid` + `BCEWithLogitsLoss` 패턴을 BERT로 학습합니다. 다음 Ch 11에서 같은 데이터로 **방식 B** (softmax + CE)를 학습한 뒤 두 결과를 비교합니다.

## 다루는 핵심 개념
- `num_labels=1` + `problem_type="multi_label_classification"` 트릭 — BCEWithLogitsLoss를 자동 매핑하기 위한 형식
- 라벨을 `[0.0]` 또는 `[1.0]` 길이 1짜리 multi-hot 벡터로 둠 (logits shape `(B, 1)` 과 맞추려고)
- `compute_metrics` 에서 logit → sigmoid → threshold 0.5 → 0/1 예측 만드는 패턴
- 정답 0과 1 그룹의 sigmoid 확률 분포 시각화
- accuracy / precision / recall / F1 / AUC 5종 평가
- Ch 11과의 비교를 위해 확률 예측 + metric을 디스크에 저장

## Loss
**`BCEWithLogitsLoss`** — Ch 3 sklearn LogReg와 같은 식. 다른 점은 logit 출처가 BERT 768-dim hidden state (TF-IDF 단어 빈도 대신).

## 데이터
Yelp 5,000건 → 별점 3 제외 + 이진화 (4-5 → 1.0, 1-2 → 0.0). Ch 3·4와 동일.

## 환경
Google Colab **T4 GPU 필수**. 약 10분 (모델 다운로드 + 2 에폭 + 평가).

## 변화 추적

| Ch | 모델 | 데이터 | Output | Activation | Loss |
|---|---|---|---|---|---|
| 9 | DistilBERT 파인튜닝 | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| **10** | DistilBERT 파인튜닝 | **Yelp 이진화** | (1차원) | **sigmoid** | **`BCEWithLogitsLoss`** |
| 11 (다음) | 같음 | 같음 | (2차원) | softmax | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[11_bert_binary_softmax](../11_bert_binary_softmax/) — 같은 데이터·모델·골격에서 *방식 B*(softmax+CE)로 학습. 이번 챕터의 저장 결과와 직접 비교.
