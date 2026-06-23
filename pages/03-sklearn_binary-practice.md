> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/03_sklearn_binary/03_sklearn_binary.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q datasets scikit-learn pandas matplotlib
```

필요한 라이브러리를 불러온 뒤 Yelp 리뷰 5,000개를 내려받아 별점(1-5) 분포를 확인합니다. 이진화하기 전 원본 라벨이 어떻게 분포돼 있는지 먼저 봐 두면 다음 단계에서 어떤 별점을 어느 클래스로 묶을지 감을 잡기 쉽습니다.

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

회귀였던 별점을 이진 분류 라벨로 바꾸는 단계입니다. 별점 3은 긍정·부정이 모호하므로 통째로 빼고, 4-5점은 1(positive), 1-2점은 0(negative)으로 묶습니다.

```python
# 별점 3은 애매하므로 제외, 4-5 → 1 (positive), 1-2 → 0 (negative)
df_bin = df[df["star"] != 3].copy()
df_bin["y"] = (df_bin["star"] >= 4).astype(int)
```

**위 코드 읽기** — `df["star"] != 3`으로 중립 리뷰를 먼저 걸러낸 뒤, `(df_bin["star"] >= 4)`가 돌려주는 True/False를 `.astype(int)`로 0/1 정수 라벨 `y`로 만듭니다. Ch 2의 연속값 라벨이 여기서 정수 두 값으로 압축됩니다.

```python
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

3점을 빼니 5,000개가 4,040개로 줄고, 긍정 비율이 49.4%로 두 클래스가 거의 반반입니다. 클래스 불균형이 거의 없으니 뒤에서 accuracy를 봐도 의미가 있습니다.

train/test로 나누고 텍스트를 TF-IDF 벡터로 변환합니다. `stratify=df_bin["y"]`로 두 클래스 비율을 train·test에 똑같이 유지하고, TF-IDF는 train으로만 `fit`한 뒤 test에는 `transform`만 적용해 정보 누설을 막습니다.

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

TF-IDF + `LogisticRegression` 조합만으로 테스트 정확도 86.4%를 얻습니다. 클래스가 반반인 데이터라 무작위 추측이 50%인 점을 감안하면 선형 모델이 긍·부정 단어 패턴을 충분히 잡아낸 셈입니다.

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
0  0.5523  0.4477
1  0.8999  0.1001
2  0.8409  0.1591
3  0.3347  0.6653
4  0.2420  0.7580

Row sums (should be 1): [1. 1. 1. 1. 1.]
```

**결과 해석**

`predict_proba`는 샘플마다 `[P(neg), P(pos)]` 두 확률을 주고, 두 값의 합은 항상 1입니다. 0번 샘플처럼 0.55 대 0.45로 애매한 경우도 있고 1번처럼 0.90으로 부정에 확신하는 경우도 있어, 단순 0/1 예측보다 모델의 확신 정도까지 읽을 수 있습니다.
