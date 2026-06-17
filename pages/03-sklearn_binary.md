**목표**: Ch 2의 회귀에서 한 단계 더 — 출력 직전에 **sigmoid** 가 붙어 [0, 1]을 강제하고, loss는 **BCE** 로 바뀝니다. 라벨도 정수 0/1로 이진화됩니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분 (학습은 즉시)


## 학습 흐름

1. 🚀 **실습**: Yelp 별점을 이진화(4-5 → 1, 1-2 → 0)하고 `LogisticRegression`으로 학습
2. 🔬 **해부**: BCE 수식 + sigmoid가 어떻게 logit을 확률로 바꾸는지 직접 재현
3. 🛠️ **변형**: 임계값(threshold) 0.5를 다른 값으로 옮기면 precision/recall이 어떻게 움직이나

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 샘플 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| **3 ← 여기** | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 (4-5→1, 1-2→0, 3 제외) | (1차원) | **sigmoid** | **`BCEWithLogitsLoss`** (sklearn: log loss) |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 2)

| 축 | Ch 2 | Ch 3 |
|---|---|---|
| 모델 | `LinearRegression` | **`LogisticRegression`** (출력에 sigmoid 내장) |
| Activation | 없음 | **sigmoid** |
| Loss | `MSELoss` | **`BCEWithLogitsLoss`** |
| 라벨 | float (1-5) | **int (0/1)** |
| 데이터 | Yelp 별점 1-5 | **Yelp 이진화** (3 제외) |
| 토크나이저 | TF-IDF | TF-IDF (그대로) |

겉보기엔 다섯 곳이 바뀌었지만, 한 가지 결정 — **"회귀 → 이진 분류"** — 이 자동으로 끌어오는 결과입니다. 분류 패러다임은 (출력 형태 = sigmoid, loss = BCE, 라벨 = 0/1, 데이터 = 이진화)가 한 묶음으로 따라옵니다.

**왜 이렇게 묶이나?** 회귀에서 본 한계 — "출력이 [0, 1]을 못 지킨다" — 를 모델 단계에서 강제로 해결하는 게 sigmoid입니다. 그러면 출력은 자연스럽게 "1일 확률"로 해석되고, 그 확률에 어울리는 loss가 BCE이고, 라벨도 0/1 정수로 단순해집니다.

## Loss 함수의 변화 — `BCEWithLogitsLoss` 등장

**Binary Cross Entropy** 는 모델이 뱉은 확률 $\hat p_i$ 와 정답 $y_i \in \{0, 1\}$ 사이의 차이를 잽니다.

$$L = -\frac{1}{N}\sum_{i=1}^{N}\left[\,y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\,\right]$$

핵심: $y_i = 1$이면 첫 항 $-\log \hat p_i$만 살고, $y_i = 0$이면 둘째 항 $-\log(1 - \hat p_i)$만 살아남습니다. 즉 한 샘플당 항상 한 항만 작동합니다.

**숫자로 감 잡기** (정답이 $y = 1$인 샘플 한 개, $N = 1$):

| 정답 $y$ | 예측 확률 $\hat p$ | 손실 $-\log \hat p$ |
|---|---|---|
| 1 | 0.9 | 0.105 |
| 1 | 0.5 | 0.693 |
| 1 | 0.1 | **2.303** |
| 1 | 0.01 | **4.605** |

확률이 정답에서 멀어질수록 — 특히 0에 가까워질수록 — 손실이 **로그 스케일로 폭증** 합니다. 자신 있게 틀린 예측을 강하게 처벌하는 게 BCE의 성격입니다.

**`BCEWithLogits`의 "Logits" 의미**: 모델 마지막 단의 raw 점수(logit) $z = w^\top x + b$를 sigmoid에 넣기 *전* 의 값을 의미합니다. PyTorch의 `BCEWithLogitsLoss`는 logit을 받아 내부에서 sigmoid + BCE를 한 번에 계산하는데, 따로 sigmoid를 통과시킨 뒤 BCE를 적용하는 것보다 수치적으로 안정적입니다.

```python
# PyTorch (Ch 9 이후)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets.float())   # logits: 활성화 전 raw 점수

# sklearn (이번 챕터)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)                              # 내부에서 sigmoid + log loss
```

## 토크나이저 노트

이번 챕터의 토크나이저는 **Ch 1·2와 동일한 `TfidfVectorizer`** 입니다. 입력 표현은 그대로고, 변화는 모델 출력단·loss·라벨에서만 일어납니다.

> **다음 챕터(Ch 4)**: 같은 TF-IDF 그대로. 변하는 건 출력이 5차원으로 늘어나고 sigmoid가 softmax로 바뀌는 것뿐.

## 이 장의 구성

- [03-1. 실습: `LogisticRegression`으로 이진 분류](03-sklearn_binary-practice.md)
- [03-2. 해부: sigmoid는 logit을 어떻게 확률로 바꾸나](03-sklearn_binary-anatomy.md)
- [03-3. 변형: 임계값(threshold)을 옮기면](03-sklearn_binary-variation.md)
- [03-4. 정리와 FAQ](03-sklearn_binary-wrapup.md)
