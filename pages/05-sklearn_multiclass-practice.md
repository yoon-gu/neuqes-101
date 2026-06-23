> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/05_sklearn_multiclass/05_sklearn_multiclass.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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

**결과 해석**

5,000개 샘플이 별점 5개 클래스(0-4)에 대략 1,000개씩 고르게 분포합니다. Ch 3·4처럼 이진화하지 않고 별점 5단계를 그대로 5클래스로 쓰는 것이 이번 챕터의 유일한 변화입니다.

별점 5단계(라벨 0-4)를 그대로 5클래스 타깃으로 두고, train/test로 나눈 뒤 텍스트를 TF-IDF로 벡터화합니다. `stratify`로 분할해 train·test 양쪽의 클래스 비율을 원본과 같게 유지합니다.

```python
# 5-class 데이터 (라벨이 이미 0-4 — 별점 1-5)
y_5class = df["label"]

X_text_train, X_text_test, y_train, y_test = train_test_split(
    df["text"], y_5class, test_size=0.2, random_state=42, stratify=y_5class,
)
```

**위 코드 읽기** `stratify=y_5class` 가 핵심으로, 분할 시 5개 클래스 비율을 train·test 양쪽에 똑같이 유지합니다. 어느 한쪽에 특정 별점이 몰리면 평가가 왜곡되므로, 클래스가 늘어난 multi-class에서 더 중요해집니다.

```python
tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, y_train distribution: {pd.Series(y_train).value_counts().sort_index().tolist()}")
```

**위 코드 읽기** `fit_transform` 은 train 으로만 어휘를 학습하고, test 에는 `transform` 만 적용해 train 어휘를 그대로 씌웁니다(test 정보 누출 방지). TF-IDF·max_features 모두 Ch 1-4와 동일 — 입력 표현은 그대로 두고 클래스 수만 늘렸습니다.

**▶ 실행 결과**

```text
X_train: (4000, 10000), y_train distribution: [813, 822, 768, 817, 780]
```

**결과 해석**

train 4,000개가 10,000차원 TF-IDF 벡터로 바뀌었고, 5개 클래스가 800개 안팎으로 고르게 들어갔습니다. `stratify` 덕분에 클래스 분포가 원본 비율을 그대로 따릅니다.

코드는 Ch 4 와 동일한 `LogisticRegression()` 한 줄. sklearn 은 *학습 데이터의 클래스 개수* 를 보고, K=5 multi-class 라 자동으로 multinomial(softmax+CE)을 적용합니다.

```python
# K=5 multi-class 데이터 → sklearn 이 자동으로 multinomial(softmax+CE) 학습
model_5 = LogisticRegression(max_iter=1000)
model_5.fit(X_train, y_train)
```

**위 코드 읽기** `LogisticRegression(max_iter=1000)` 한 줄은 Ch 4와 글자까지 같습니다. 달라진 건 `y_train` 에 클래스가 5개 들어 있다는 것뿐이고, 모던 sklearn 은 이를 보고 자동으로 multinomial(softmax + CE)로 학습합니다.

```python
y_pred = model_5.predict(X_test)
acc = accuracy_score(y_test, y_pred)
baseline = 1 / 5

print(f"Test accuracy: {acc:.4f}")
print(f"baseline (uniform guess): {baseline:.4f}")
print(f"Improvement over baseline: {acc - baseline:+.4f}")
```

**위 코드 읽기** `baseline = 1 / 5` 는 5개 클래스를 아무 정보 없이 균등 추측할 때의 정확도(0.2)입니다. 이진(Ch 3·4)의 baseline 0.5보다 훨씬 낮아 — 클래스가 늘면 무작위로 맞히기가 그만큼 어려워집니다.

**▶ 실행 결과**

```text
Test accuracy: 0.5110
baseline (uniform guess): 0.2000
Improvement over baseline: +0.3110
```

**결과 해석**

정확도 0.5110으로, 무작위 추측(0.2)을 31%p 웃돕니다. 5클래스에 절반을 맞히는 건 별점 판별이 1·2점, 4·5점처럼 인접 클래스끼리 헷갈리기 쉬운 어려운 문제임을 감안하면 견고한 신호입니다.

`predict_proba` 로 각 샘플의 5개 클래스 확률 분포를 확인합니다. softmax 한 번을 거치므로 한 행(한 샘플)의 5개 확률은 합이 정확히 1이 되어야 — multi-class 의 상호배타 가정이 출력에 그대로 드러납니다.

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
0  0.357  0.307  0.163  0.081  0.092
1  0.136  0.339  0.348  0.138  0.040
2  0.614  0.233  0.052  0.042  0.059
```

**결과 해석**

`(1000, 5)` 모양에서 보듯 샘플마다 5개 클래스 확률이 나오고, 각 행 합이 정확히 1입니다. 분포는 인접 별점에 퍼져 있어 — 예컨대 첫 샘플은 1★(0.357)과 2★(0.307)에 확신이 갈리는데, 이는 모델이 인접 클래스를 헷갈리는 ordinal 성질을 그대로 보여줍니다.
