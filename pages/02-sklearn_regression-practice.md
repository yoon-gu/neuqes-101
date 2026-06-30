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

# matplotlib 한글 폰트 (Colab — NanumGothic). plot 의 한국어가 □ 로 깨지지 않게.
import matplotlib.font_manager as fm, subprocess, os
_fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(_fp):
    subprocess.run("apt-get -qq -y install fonts-nanum", shell=True)
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "NanumGothic"
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

Yelp 데이터의 별점은 0-4로 저장돼 있어, 사람이 읽는 1-5 척도로 옮긴 뒤 회귀 라벨로 씁니다. 이때 라벨을 `astype(float)`로 실수형으로 만드는 것이 핵심입니다 — 회귀는 정수 클래스가 아니라 연속값을 예측하기 때문입니다. 같은 TF-IDF 설정(Ch 1과 동일)으로 텍스트를 sparse 벡터로 바꿔 입력을 만듭니다.

```python
# 별점은 0-4로 저장돼 있으니 1-5로 변환
df["star"] = df["label"] + 1

# train / test split
X_text_train, X_text_test, y_train, y_test = train_test_split(
    df["text"], df["star"].astype(float),
    test_size=0.2, random_state=42,
)
```

**위 코드 읽기** `df["star"].astype(float)` 가 회귀 라벨을 만드는 자리입니다. 별점에 1을 더해 1-5로 옮기고, 정수가 아니라 float 으로 둬 연속값 회귀의 정답으로 삼습니다. `random_state=42` 로 분할을 고정해 재현성을 확보합니다.

```python
# TF-IDF (Ch 1과 같은 설정)
tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
```

**위 코드 읽기** `fit_transform` 은 훈련 데이터로만 어휘를 학습하고, 평가 데이터에는 `transform` 만 적용해 같은 어휘로 변환합니다 — 평가 텍스트가 어휘 학습에 새지 않게 하는 표준 방식입니다.

**▶ 실행 결과**

```text
X_train: (4000, 10000), y_train: (4000,)
X_test:  (1000, 10000), y_test:  (1000,)
```

**결과 해석**

5,000 샘플이 4,000/1,000으로 나뉘고, 각 텍스트가 10,000차원 TF-IDF 벡터가 됐습니다. 이제 입력은 길이 10,000짜리 sparse 벡터고, 모델은 여기서 별점 숫자 하나를 예측하면 됩니다.

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

Train MSE가 0에 가까운 건 10,000차원 feature 가 4,000 샘플을 거의 완벽히 외운 과적합 신호입니다. Test MSE 1.56(별점 오차 약 ±1점)·R² 0.21로, 단순 선형 회귀가 일반화에는 한계가 있음을 보여줍니다.

예측값이 별점 척도 [1, 5]를 얼마나 벗어나는지 직접 확인합니다. 활성화 함수가 없는 만큼 음수나 5 초과가 나와도 자연스러운데, 분포 히스토그램으로 그 이탈 정도를 눈으로 봅니다.

```python
# 예측값이 1-5 범위를 얼마나 벗어나는지 확인
print(f"Pred range: [{y_pred_test.min():.2f}, {y_pred_test.max():.2f}]")
print(f"True range: [{y_test.min():.0f}, {y_test.max():.0f}]")

plt.hist(y_pred_test, bins=40, alpha=0.6, label="예측")
plt.hist(y_test, bins=5, alpha=0.6, label="실제")
plt.axvline(1, color="red", linestyle="--", linewidth=1, label="1 / 5 경계")
plt.axvline(5, color="red", linestyle="--", linewidth=1)
plt.xlabel("별점 (1-5)")
plt.ylabel("개수")
plt.legend()
plt.title("예측 분포: 실제 vs 예측")
plt.show()
```

**▶ 실행 결과**

```text
Pred range: [-1.55, 7.15]
True range: [1, 5]
```

![output](../assets/02-sklearn_regression-out1-1.png)

**결과 해석**

실제 별점은 [1, 5]에 갇혀 있지만 예측은 -1.55에서 7.15까지 퍼집니다. 가중합을 그대로 출력하는 한 범위 제약이 없다는 회귀의 본질이 그대로 드러납니다.
