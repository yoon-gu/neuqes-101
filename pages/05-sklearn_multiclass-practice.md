## 환경 준비

```python
!pip install -q datasets scikit-learn pandas matplotlib
```

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams["axes.unicode_minus"] = False

dataset = load_dataset("Yelp/yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
print(f"Total samples: {len(df)}")
print("Class distribution (label 0-4 = star 1-5):")
print(df["label"].value_counts().sort_index())
```

**▶ 실행 결과**

```text
Total samples: 5000
Class distribution (label 0-4 = star 1-5):
label
0    1017
1    1027
2     960
3    1021
4     975
Name: count, dtype: int64
```

```python
# 5-class 데이터 (라벨이 이미 0-4 — 별점 1-5)
y_5class = df["label"]

X_text_train, X_text_test, y_train, y_test = train_test_split(
    df["text"], y_5class, test_size=0.2, random_state=42, stratify=y_5class,
)

tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, y_train distribution: {pd.Series(y_train).value_counts().sort_index().tolist()}")
```

**▶ 실행 결과**

```text
X_train: (4000, 10000), y_train distribution: [813, 822, 768, 817, 780]
```

코드는 Ch 4 와 동일한 `LogisticRegression()` 한 줄. sklearn 은 *학습 데이터의 클래스 개수* 를 보고, K=5 multi-class 라 자동으로 multinomial(softmax+CE)을 적용합니다.

```python
# K=5 multi-class 데이터 → sklearn 이 자동으로 multinomial(softmax+CE) 학습
model_5 = LogisticRegression(max_iter=1000)
model_5.fit(X_train, y_train)

y_pred = model_5.predict(X_test)
acc = accuracy_score(y_test, y_pred)
baseline = 1 / 5

print(f"Test accuracy: {acc:.4f}")
print(f"baseline (uniform guess): {baseline:.4f}")
print(f"Improvement over baseline: {acc - baseline:+.4f}")
```

**▶ 실행 결과**

```text
Test accuracy: 0.5080
baseline (uniform guess): 0.2000
Improvement over baseline: +0.3080
```

**결과 해석**

5클래스 정확도가 50.8%로 랜덤 추측(20%)의 2.5배입니다. Ch 3의 binary 86%보다 낮은 건 틀릴 수 있는 경우의 수가 2개에서 5개로 늘었기 때문이며, 같은 TF-IDF·모델에서 태스크 난이도만으로 점수가 갈린다는 걸 보여줍니다.

```python
proba_5 = model_5.predict_proba(X_test)
print(f"predict_proba shape: {proba_5.shape}  (N, K=5)")
print(f"Row sums (should be 1): {proba_5.sum(axis=1)[:5].round(4)}")
print(f"\nFirst 3 sample probability distributions:")
print(pd.DataFrame(proba_5[:3], columns=[f"P({i+1}★)" for i in range(5)]).round(3))
```

**▶ 실행 결과**

```text
predict_proba shape: (1000, 5)  (N, K=5)
Row sums (should be 1): [1. 1. 1. 1. 1.]

First 3 sample probability distributions:
   P(1★)  P(2★)  P(3★)  P(4★)  P(5★)
0  0.348  0.301  0.171  0.084  0.096
1  0.137  0.333  0.352  0.139  0.040
2  0.615  0.230  0.052  0.042  0.060
```

**결과 해석**

각 행이 5개 별점에 대한 확률이고 합이 정확히 1입니다. softmax가 클래스끼리 확률을 나눠 갖게 만들기 때문이며, 세 번째 샘플처럼 1★에 0.615가 몰리면 모델이 그만큼 확신한다는 뜻입니다.
