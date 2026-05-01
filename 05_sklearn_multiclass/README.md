# 05_sklearn_multiclass — K=5로 진짜 일반화

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/05_sklearn_multiclass/05_sklearn_multiclass.ipynb)

## 한 줄 목표
Ch 4의 multinomial LogReg를 K=2에서 K=5로 그대로 확장합니다. softmax/CE는 K가 무엇이든 같은 식이라 코드 변화는 거의 없고, 클래스 수만 늘어납니다.

## 다루는 핵심 개념
- 같은 `LogisticRegression(multi_class="multinomial")` 로 5클래스 분류
- `predict_proba` shape `(N, 5)`, 행 합 = 1
- per-class precision/recall/F1 (`classification_report`)
- 5×5 confusion matrix가 **대각선 근처에 몰림** — ordinal 데이터의 자연스러운 흔적
- 회귀(Ch 2) vs 5클래스 분류(Ch 5) 비교 — 같은 데이터, 다른 관점
- multinomial vs OvR 비교 — `OneVsRestClassifier`로 K개 독립 binary 모델을 직접 노출, 한 샘플의 (multinomial / OvR raw / OvR 정규화 후) 출력을 나란히 비교 (Ch 6 multi-label로 가는 다리)

## Loss 수치 예시 (K=5, 정답 클래스 = 2)
| 예측 분포 | 정답 확률 | 손실 |
|---|---|---|
| 정답 집중 `[0.05, 0.05, 0.80, 0.05, 0.05]` | 0.80 | 0.223 |
| 균등 `[0.20, 0.20, 0.20, 0.20, 0.20]` | 0.20 | 1.609 |
| 틀린 곳 집중 `[0.05, 0.05, 0.05, 0.05, 0.80]` | 0.05 | **2.996** |

baseline = $\log K = \log 5 \approx 1.609$ — 학습된 모델은 이보다 작아야 정상.

## 데이터
Yelp 5,000건, 별점 1-5 → 라벨 0-4 (5클래스).

## 환경
Google Colab CPU 런타임으로 충분 (GPU 불필요). 약 5-10분.

## 변화 추적

| Ch | 모델 | 데이터 | Output | Loss |
|---|---|---|---|---|
| 4 | `LogisticRegression(multi_class="multinomial")` | Yelp 이진화 | (2차원) | `CrossEntropyLoss` |
| **5** | `LogisticRegression(multi_class="multinomial")` | **Yelp 5클래스** | **(5차원)** | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[06_sklearn_multilabel](../06_sklearn_multilabel/) — softmax의 합=1 제약을 풀고 multi-label로 확장. K개 독립 sigmoid가 각 라벨을 따로 학습.
