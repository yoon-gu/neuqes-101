**목표**: 가장 단순한 형태로 모델을 만나봅니다. 활성화 함수도 없고, 출력 범위 제한도 없는 — **그냥 숫자를 뱉는 회귀** 부터 시작합니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분 (학습은 즉시, 데이터 로딩이 대부분)


## 학습 흐름

1. 🚀 **실습**: 별점 1-5를 `LinearRegression`으로 회귀
2. 🔬 **해부**: `MSELoss` 수식과 의미 — "모델 출력은 그냥 숫자다"
3. 🛠️ **변형**: 별점을 [0, 1]로 정규화 후 회귀, 그리고 그 한계

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 샘플 | — | — | — |
| **2 ← 여기** | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` (sklearn: squared error) |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 1)

| 축 | Ch 1 | Ch 2 |
|---|---|---|
| 모델 | 없음 (변환만) | **`LinearRegression`** ← 추가 |
| Loss | 없음 | **`MSELoss`** ← 첫 등장 |
| 토크나이저 | TF-IDF | TF-IDF (그대로) |
| 데이터 | Yelp 5,000 | Yelp 5,000 (그대로) |

이번 챕터에서 바뀌는 것은 **모델 + Loss** 두 가지입니다 — 그러나 둘은 한 묶음으로 처음 등장하는 거라, 학습자 관점에선 사실상 "모델이 처음 생겼다"는 한 가지 변화입니다. 다음 챕터부터는 한 번에 하나씩 변합니다.

**왜 회귀부터?** 모델이 뱉는 출력은 본질적으로 그냥 숫자입니다. 분류·다중라벨·multi-task 같은 화려한 형태들은 모두 "그 숫자를 어떻게 가공해서 어떤 loss를 매길 것인가"의 변주일 뿐입니다. 가장 단순한 형태 — 활성화 함수도 없는 1차원 출력 — 부터 시작합니다.

## Loss 함수의 변화 — `MSELoss` 등장

이번 챕터의 loss는 **Mean Squared Error** 입니다.

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat y_i)^2$$

- $y_i$: 정답 (실제 별점)
- $\hat y_i$: 모델 예측

**숫자로 감 잡기** (정답이 별점 5인 한 샘플 기준, $N=1$이라 평균은 그대로):

| 정답 $y$ | 예측 $\hat y$ | 오차 $|y - \hat y|$ | 손실 $(y - \hat y)^2$ |
|---|---|---|---|
| 5 | 4 | 1 | **1** |
| 5 | 3 | 2 | **4** |
| 5 | 1 | 4 | **16** |

오차가 2배가 되면 손실은 4배, 4배가 되면 16배로 **비선형으로** 증폭됩니다. 이 제곱 항이 MSE의 성격을 결정합니다 — "조금씩 자주 틀리는 것" 보다 "어쩌다 크게 틀리는 것"을 모델이 더 강하게 회피합니다.

PyTorch에서는 `nn.MSELoss`, sklearn에서는 같은 개념이 `LinearRegression`에 내장돼 있고 평가 함수로는 `mean_squared_error`로 따로 부릅니다.

```python
# PyTorch (Ch 8 이후 등장)
criterion = nn.MSELoss()
loss = criterion(pred, target)

# sklearn (이번 챕터)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model.fit(X, y)                            # 내부적으로 MSE 최소화
mean_squared_error(y, model.predict(X))    # 평가
```

## 토크나이저 노트

이번 챕터의 토크나이저는 **Ch 1과 동일한 `TfidfVectorizer`** 입니다. 변하는 건 그 위에 모델이 붙는다는 것뿐이라, 같은 입력 벡터에서 출발해 모델만 차이를 만들도록 합니다.

같은 문장 `"I love using Hugging Face!"` 의 토큰화 결과는 Ch 1과 똑같습니다 — 소문자, 단일 문자 제거, OOV 무시. 모델 입장에서 입력은 길이 V짜리 sparse 벡터 한 개고, 그걸 받아 **숫자 한 개** 를 뱉으면 됩니다.

> **다음 챕터(Ch 3)**: 같은 TF-IDF 그대로. 변하는 건 출력에 sigmoid가 붙고 loss가 `BCEWithLogitsLoss`로 바뀌는 것입니다.

## 이 장의 구성

- [02-1. 실습: 별점 1-5를 그대로 회귀하기](02-sklearn_regression-practice.md)
- [02-2. 해부: "출력은 그냥 숫자다"](02-sklearn_regression-anatomy.md)
- [02-3. 변형: 별점을 [0, 1]로 정규화](02-sklearn_regression-variation.md)
- [02-4. 정리와 FAQ](02-sklearn_regression-wrapup.md)
