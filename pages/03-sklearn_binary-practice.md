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

**결과 해석**

3점을 빼고 4-5점을 positive, 1-2점을 negative로 묶으니 양쪽이 49.4% 대 50.6%로 거의 반반입니다. 균형 잡힌 데이터라 뒤에서 정확도(accuracy)를 성능 지표로 그대로 신뢰할 수 있습니다.

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

같은 TF-IDF·5,000건인데 정확도가 86.4%까지 오릅니다 — Ch 2의 5단계 회귀(R² 0.22)보다 쉬운 건 문제를 "몇 점"에서 "좋다/나쁘다" 둘로 줄였기 때문입니다. 모델과 특징을 그대로 둔 채 태스크만 단순화해도 성능이 크게 달라진다는 점을 보여줍니다.

```python
# predict_proba는 [P(y=0), P(y=1)] 형태로 두 확률을 모두 줌
y_proba = model.predict_proba(X_test)
print(f"y_proba shape: {y_proba.shape}  (per sample: [P(0), P(1)])")
print(f"\nFirst 5 predicted probabilities:")
print(pd.DataFrame(y_proba[:5], columns=["P(neg)", "P(pos)"]).round(4))

# 두 확률을 합치면 항상 1
print(f"\nRow sums (should be 1): {y_proba.sum(axis=1)[:5]}")
```

**▶ 실행 결과**

```text
y_proba shape: (808, 2)  (per sample: [P(0), P(1)])

First 5 predicted probabilities:
   P(neg)  P(pos)
0  0.5515  0.4485
1  0.9001  0.0999
2  0.8362  0.1638
3  0.3339  0.6661
4  0.2422  0.7578

Row sums (should be 1): [1. 1. 1. 1. 1.]
```
