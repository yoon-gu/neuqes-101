> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/04_softmax_binary/04_softmax_binary.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

plt.rcParams["axes.unicode_minus"] = False

dataset = load_dataset("Yelp/yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
df["star"] = df["label"] + 1   # 0-4 → 1-5
```

```python
# Ch 3과 동일한 이진화: 별점 3 제외, 4-5 → 1, 1-2 → 0
df_bin = df[df["star"] != 3].copy()
df_bin["y"] = (df_bin["star"] >= 4).astype(int)

X_text_train, X_text_test, y_train, y_test = train_test_split(
    df_bin["text"], df_bin["y"], test_size=0.2, random_state=42, stratify=df_bin["y"],
)

tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, positive rate: {y_train.mean():.1%}")
```

**▶ 실행 결과**

```text
X_train: (3232, 10000), positive rate: 49.4%
```

| 방식 | sklearn 인자 | 출력 차원 | 활성화 | loss |
|---|---|---|---|---|
| A (Ch 3 그대로) | `LogisticRegression()` | 1 | sigmoid | BCE |
| B (이번 챕터) | `LogisticRegression()` (multinomial 자동) | 2 | softmax | CE |

```python
# 방식 A — 1차원 출력 + sigmoid + BCE (sklearn binary 의 표준 학습 형태)
model_a = LogisticRegression(max_iter=1000)
model_a.fit(X_train, y_train)

# 방식 B — 2차원 출력 + softmax + CE 의도. binary(K=2) 데이터에선 sklearn 이
# 내부적으로 1차원 형태로 collapse 해서 방식 A 와 같은 결과를 줍니다 (FAQ Q4).
# 여기선 두 방식을 *명시적으로* 같이 학습한 뒤 predict_proba 일치를 확인.
model_b = LogisticRegression(max_iter=1000)
model_b.fit(X_train, y_train)

acc_a = accuracy_score(y_test, model_a.predict(X_test))
acc_b = accuracy_score(y_test, model_b.predict(X_test))

print(f"Method A (sigmoid + BCE) accuracy: {acc_a:.4f}")
print(f"Method B (softmax + CE)  accuracy: {acc_b:.4f}")
print(f"Diff: {abs(acc_a - acc_b):.4f}")
```

**▶ 실행 결과**

```text
Method A (sigmoid + BCE) accuracy: 0.8639
Method B (softmax + CE)  accuracy: 0.8639
Diff: 0.0000
```

```python
proba_a = model_a.predict_proba(X_test)   # (N, 2)
proba_b = model_b.predict_proba(X_test)   # (N, 2)

print(f"Method A predict_proba shape: {proba_a.shape}")
print(f"Method B predict_proba shape: {proba_b.shape}")
print(f"  (sklearn returns [P(0), P(1)] for both — sigmoid output expanded to two columns)")

p_a, p_b = proba_a[:, 1], proba_b[:, 1]
print(f"\nFirst 5 P(y=1):")
print(f"Method A: {p_a[:5].round(4)}")
print(f"Method B: {p_b[:5].round(4)}")
print(f"\nMax diff:  {np.abs(p_a - p_b).max():.4f}")
print(f"Mean diff: {np.abs(p_a - p_b).mean():.4f}")
```

**▶ 실행 결과**

```text
Method A predict_proba shape: (808, 2)
Method B predict_proba shape: (808, 2)
  (sklearn returns [P(0), P(1)] for both — sigmoid output expanded to two columns)

First 5 P(y=1):
Method A: [0.4477 0.1001 0.1591 0.6653 0.758 ]
Method B: [0.4477 0.1001 0.1591 0.6653 0.758 ]

Max diff:  0.0000
Mean diff: 0.0000
```
