> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/02_sklearn_regression/02_sklearn_regression.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q datasets scikit-learn pandas matplotlib
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["axes.unicode_minus"] = False

dataset = load_dataset("Yelp/yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
print(f"Sample count: {len(df)}")
```

**▶ 실행 결과**

```text
Sample count: 5000
```

원시 별점(0-4)을 회귀 라벨(1-5)로 바꾸고, 텍스트를 TF-IDF 벡터로 변환해 학습/평가 셋을 만듭니다. 라벨을 `.astype(float)` 로 두는 것이 핵심입니다 — 회귀의 정답은 연속값이라 정수가 아닌 float 으로 다뤄야 MSE가 자연스럽게 매겨집니다.

```python
# 별점은 0-4로 저장돼 있으니 1-5로 변환
df["star"] = df["label"] + 1
```

**위 코드 읽기.** `df["label"]` 은 Yelp 원본이 0부터 시작하도록 저장한 별점입니다. 여기에 1을 더해 사람이 읽는 1-5 척도로 되돌립니다. 이 값이 곧 회귀가 맞춰야 할 정답이 됩니다.

```python
# train / test split
X_text_train, X_text_test, y_train, y_test = train_test_split(
    df["text"], df["star"].astype(float),
    test_size=0.2, random_state=42,
)
```

**위 코드 읽기.** 라벨에 `.astype(float)` 를 붙여 별점을 연속값으로 넘기는 점이 이 셀의 가장 중요한 부분입니다. 정수 라벨을 그대로 두면 분류처럼 보일 수 있지만, 회귀는 "5점에 가까운 4.7" 같은 거리를 다루므로 float 으로 두어야 합니다. `random_state=42` 로 분할을 고정해 매 실행 결과가 같게 합니다.

```python
# TF-IDF (Ch 1과 같은 설정)
tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
```

**위 코드 읽기.** `fit_transform` 은 train 텍스트로 어휘와 IDF를 학습하면서 동시에 벡터화하고, test에는 `transform` 만 써서 train에서 배운 어휘만 적용합니다. test로 어휘를 다시 학습하면 정보 누설이 되므로, 이 비대칭이 의도된 설계입니다.

**▶ 실행 결과**

```text
X_train: (4000, 10000), y_train: (4000,)
X_test:  (1000, 10000), y_test:  (1000,)
```

**결과 해석**

5,000 샘플이 4,000/1,000으로 나뉘고, 각 문서가 10,000차원 TF-IDF 벡터로 표현됩니다. 모델은 이 sparse 벡터 한 개를 받아 별점 숫자 하나를 예측합니다.

`LinearRegression`은 가중치 벡터 $w$와 편향 $b$를 학습해 다음을 출력합니다.

$$\hat y = w^\top x + b$$

활성화 함수 없음, 출력 범위 제한 없음. 정답 $y$와의 MSE를 최소화하도록 $w, b$를 푸는 게 학습의 전부입니다 (sklearn은 정규방정식으로 한 번에 풉니다).

```python
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"Test  MSE: {mean_squared_error(y_test,  y_pred_test):.4f}")
print(f"Test  MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"Test  R²:  {r2_score(y_test, y_pred_test):.4f}")
```

**▶ 실행 결과**

```text
Train MSE: 0.0000
Test  MSE: 1.5565
Test  MAE: 0.9952
Test  R²:  0.2139
```

**결과 해석**

Train MSE가 0인데 Test MSE는 1.56인 큰 격차는, 10,000개 feature가 4,000개 샘플을 거의 외워버린 과적합 신호입니다. Test MAE 0.99는 평균적으로 별점을 약 1점 빗나간다는 뜻이고, R² 0.21은 모델이 별점 변동의 21% 정도만 설명한다는 의미입니다.

예측값이 정답 범위인 1-5를 얼마나 벗어나는지 직접 확인하고, 분포를 히스토그램으로 그립니다. 활성화 함수가 없는 회귀 출력이 경계 밖으로 새어 나가는지 눈으로 보는 것이 핵심입니다.

```python
# 예측값이 1-5 범위를 얼마나 벗어나는지 확인
print(f"Pred range: [{y_pred_test.min():.2f}, {y_pred_test.max():.2f}]")
print(f"True range: [{y_test.min():.0f}, {y_test.max():.0f}]")

plt.hist(y_pred_test, bins=40, alpha=0.6, label="predicted")
plt.hist(y_test, bins=5, alpha=0.6, label="actual")
plt.axvline(1, color="red", linestyle="--", linewidth=1, label="1 / 5 boundary")
plt.axvline(5, color="red", linestyle="--", linewidth=1)
plt.xlabel("Star (1-5)")
plt.ylabel("Count")
plt.legend()
plt.title("Prediction distribution: actual vs predicted")
plt.show()
```

**▶ 실행 결과**

```text
Pred range: [-1.55, 7.15]
True range: [1, 5]
```

**결과 해석**

정답은 1-5에 갇혀 있지만 예측은 -1.55에서 7.15까지 퍼집니다. 활성화 함수도 범위 제약도 없는 가중합이라 음수도 5 초과도 그대로 나오는 것으로, 다음 절에서 해부할 "출력은 그냥 숫자다"의 직접 증거입니다.

![output](../assets/02-sklearn_regression-out1.png)
