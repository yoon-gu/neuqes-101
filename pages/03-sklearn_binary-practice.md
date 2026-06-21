> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/03_sklearn_binary/03_sklearn_binary.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    log_loss, precision_recall_fscore_support,
)

plt.rcParams["axes.unicode_minus"] = False

dataset = load_dataset("Yelp/yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
df["star"] = df["label"] + 1   # 0-4 → 1-5
print(f"Total samples: {len(df)}")
print(df["star"].value_counts().sort_index())
```

**▶ 실행 결과**

```text
Total samples: 5000
star
1    1017
2    1027
3     960
4    1021
5     975
Name: count, dtype: int64
```

별점 1-5 회귀 라벨을 0/1 이진 라벨로 바꾸는 단계입니다. 중립에 가까운 3점은 아예 제외하고, 4-5점을 positive(1), 1-2점을 negative(0)로 매핑합니다. 양쪽 클래스 비율이 비슷한지(positive rate)를 함께 확인해 둡니다.

```python
# 별점 3은 애매하므로 제외, 4-5 → 1 (positive), 1-2 → 0 (negative)
df_bin = df[df["star"] != 3].copy()
df_bin["y"] = (df_bin["star"] >= 4).astype(int)

print(f"Binarized samples: {len(df_bin)}  (3-star excluded)")
print(f"Class distribution:\n{df_bin['y'].value_counts().sort_index()}")
print(f"Positive rate: {df_bin['y'].mean():.1%}")
```

**▶ 실행 결과**

```text
Binarized samples: 4040  (3-star excluded)
Class distribution:
y
0    2044
1    1996
Name: count, dtype: int64
Positive rate: 49.4%
```

학습/평가 분할 후 텍스트를 TF-IDF 벡터로 바꿉니다. `stratify=df_bin["y"]`로 분할 후에도 두 클래스 비율이 유지되도록 하고, `tfidf.fit_transform`은 학습 데이터로만 어휘를 학습한 뒤 평가 데이터에는 `transform`만 적용해 정보 누수를 막습니다.

```python
X_text_train, X_text_test, y_train, y_test = train_test_split(
    df_bin["text"], df_bin["y"],
    test_size=0.2, random_state=42, stratify=df_bin["y"],
)

tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
```

**▶ 실행 결과**

```text
X_train: (3232, 10000), y_train: (3232,)
X_test:  (808, 10000), y_test:  (808,)
```

`LogisticRegression`이 내부에서 하는 일은 두 단계입니다.

1. logit 계산: $z = w^\top x + b$ (Ch 2와 똑같은 선형 결합)
2. sigmoid로 확률 변환: $\hat p = \sigma(z) = \dfrac{1}{1 + e^{-z}}$

학습은 BCE를 최소화하는 $w, b$를 찾는 것이고, 예측은 $\hat p \geq 0.5$를 기준으로 0/1로 자릅니다.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**▶ 실행 결과**

```text
Test accuracy: 0.8639
```

**결과 해석**

TF-IDF + `LogisticRegression`만으로 평가 정확도 약 86%를 얻었습니다. 클래스가 거의 균형(positive 49.4%)이라 이 accuracy는 그대로 신뢰할 만한 지표입니다.

```python
# predict_proba는 [P(y=0), P(y=1)] 형태로 두 확률을 모두 줌
y_proba = model.predict_proba(X_test)
print(f"y_proba shape: {y_proba.shape}  (per sample: [P(0), P(1)])")
print(f"\nFirst 5 predicted probabilities:")
print(pd.DataFrame(y_proba[:5], columns=["P(neg)", "P(pos)"]).round(4))
```

**위 코드 읽기.** `predict()`가 0/1 라벨만 주는 것과 달리 `predict_proba`는 샘플마다 `[P(y=0), P(y=1)]` 두 확률을 함께 돌려줍니다. 그래서 출력 shape가 `(N, 2)`가 되고, positive 확률만 쓰고 싶을 때는 `y_proba[:, 1]`로 두 번째 열을 꺼냅니다.

```python
# 두 확률을 합치면 항상 1
print(f"\nRow sums (should be 1): {y_proba.sum(axis=1)[:5]}")
```

**위 코드 읽기.** 두 열은 같은 사건의 여집합이므로 `P(neg) + P(pos)`가 항상 1입니다. `sum(axis=1)`로 행 합이 모두 1임을 확인하면, sigmoid가 만든 확률이 제대로 정규화되어 있다는 점검이 됩니다.

**▶ 실행 결과**

```text
y_proba shape: (808, 2)  (per sample: [P(0), P(1)])

First 5 predicted probabilities:
   P(neg)  P(pos)
0  0.5523  0.4477
1  0.8999  0.1001
2  0.8409  0.1591
3  0.3347  0.6653
4  0.2420  0.7580

Row sums (should be 1): [1. 1. 1. 1. 1.]
```

**결과 해석**

샘플마다 `P(neg)`와 `P(pos)`가 나오고, 0번 샘플처럼 0.5523/0.4477로 애매한 경우와 4번 샘플처럼 0.2420/0.7580으로 확신이 큰 경우가 섞여 있습니다. 행 합이 모두 정확히 1이라 두 확률이 정상적으로 정규화되었음을 알 수 있습니다.
