# 04_sklearn_multiclass — sigmoid가 softmax로

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/04_sklearn_multiclass/04_sklearn_multiclass.ipynb)

## 한 줄 목표
sigmoid → **softmax** 일반화. 출력 차원이 1에서 K로 늘고 loss는 `CrossEntropyLoss`로 바뀝니다. K=2 특수 경우(Section A)에서 sigmoid+BCE와 **수학적으로 동등** 함을 먼저 확인한 뒤, K=5로 확장합니다.

## 다루는 핵심 개념

### Section A — Binary 재방문
- Ch 3과 똑같은 binary 문제를 `LogisticRegression(multi_class="multinomial")` 로 다시 풀기
- 두 방식(sigmoid+BCE / softmax+CE)의 정확도와 `predict_proba` 비교
- 수학적 동등성: $\sigma(z) = \text{softmax}([0, z])_1 = \sigma(z_1 - z_0)$ — softmax+2차원은 sigmoid+1차원의 리파라미터화
- 직접 계산으로 두 방식 P(y=1)이 거의 일치 검증

### Section B — 5클래스 확장
- Yelp 별점 1-5를 5개 독립 클래스로 분류 (라벨 0-4)
- `predict_proba` shape `(N, 5)`, 행 합 = 1
- per-class precision/recall/F1
- confusion matrix가 **대각선 근처에 몰림** — 모델이 자연스럽게 ordinal 구조를 학습한 흔적
- multinomial vs OvR 비교 (Ch 5 multi-label로 가는 다리)

## Loss
**Cross Entropy** — 정답 클래스 확률의 -log. K=2일 때 BCE와 동등. 수치 예시 표(K=5, 정답=2): 정답에 집중 → 0.223 / 균등 → 1.609 / 틀린 곳 집중 → 2.996.

## 데이터
- Section A: Yelp 이진화 (Ch 3과 동일)
- Section B: Yelp 5,000건 별점 1-5 → 라벨 0-4 (5클래스)

## 환경
Google Colab CPU 런타임으로 충분 (GPU 불필요). 약 5-10분.

## 변화 추적

| Ch | 모델 | 데이터 | Activation | Loss |
|---|---|---|---|---|
| 3 | LogReg | Yelp 이진화 | sigmoid | `BCEWithLogitsLoss` |
| **4** | LogReg(multinomial) | Yelp 5클래스 | **softmax** | **`CrossEntropyLoss`** |

전체 18챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[05_sklearn_multilabel](../05_sklearn_multilabel/) — softmax의 합=1 제약을 풀고 multi-label로 확장. 한 샘플에 여러 라벨이 동시에 붙는 경우, K개 독립 sigmoid로 각 라벨을 따로 학습합니다.
