# 04_softmax_binary — sigmoid와 softmax의 동등성 (Binary Classification)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/04_softmax_binary/04_softmax_binary.ipynb)

## 한 줄 목표
Ch 3과 **완전히 같은 binary 데이터** 를 출력 차원 2로 늘리고 softmax + CrossEntropy로 다시 풀어 두 방식이 수학적으로 동등함을 식과 코드로 직접 확인합니다. 이 직관은 Ch 10 BERT binary로 그대로 옮겨갑니다.

## 다루는 핵심 개념
- 첫 등장 Loss: **`CrossEntropyLoss`** — K=2 수치 예시가 Ch 3의 BCE 표와 완전히 같다는 점이 동등성의 첫 단서
- 두 방식 비교: 같은 데이터에 sigmoid+BCE / softmax+CE 학습 → predict_proba 거의 일치
- **수학적 동등성**: $\sigma(z) = \text{softmax}([z_0, z_1])_1 = \sigma(z_1 - z_0)$, K=2 CE → BCE
- 동등성 코드 시연: 임의의 logit 쌍에서 `softmax([z_0,z_1])_1 == sigmoid(z_1-z_0)` 직접 확인 (max 차이 ~1e-16)
- sklearn 동작 관찰: 모던 `LogisticRegression()` 도 K=2 binary 데이터에서는 `coef_.shape` 가 `(1, V)` — sklearn 이 K=2 multinomial 을 binary form 으로 자동 collapse 하기 때문. 진짜 (2, V) 두 logit head 는 Ch 10 PyTorch 에서 등장

## 손실 수치 예시 (K=2, 정답 y=1)
| 예측 분포 | 정답 확률 | 손실 |
|---|---|---|
| `[0.1, 0.9]` | 0.9 | 0.105 |
| `[0.5, 0.5]` | 0.5 | 0.693 |
| `[0.9, 0.1]` | 0.1 | **2.303** |

→ Ch 3 BCE 표와 완전히 동일.

## 데이터
Yelp 5,000건 → 별점 3 제외, 4-5 → 1, 1-2 → 0 (Ch 3과 정확히 같은 가공).

## 환경
Google Colab CPU 런타임으로 충분 (GPU 불필요). 약 5-10분.

## 변화 추적

| Ch | 모델 | 데이터 | Output | Activation | Loss |
|---|---|---|---|---|---|
| 3 | `LogisticRegression()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| **4** | `LogisticRegression()` (multinomial 자동) | Yelp 이진화 | **(2차원)** | **softmax** | **`CrossEntropyLoss`** |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[05_sklearn_multiclass](../05_sklearn_multiclass/) — 같은 multinomial LogReg를 K=5로 그대로 확장. 코드 변화 거의 없음, 데이터만 5클래스로.
