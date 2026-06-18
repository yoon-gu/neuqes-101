> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/06_sklearn_multilabel/06_sklearn_multilabel.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q datasets scikit-learn pandas matplotlib
```

필요한 라이브러리를 불러온 뒤 Yelp 리뷰 데이터를 내려받습니다. 전체를 다 쓰면 학습이 길어지므로 seed를 고정해 무작위로 5,000건만 추려 이후 실습 내내 같은 표본을 씁니다.

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score, classification_report,
)

plt.rcParams["axes.unicode_minus"] = False

dataset = load_dataset("Yelp/yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
print(f"Total samples: {len(df)}")
```

**▶ 실행 결과**

```text
Total samples: 5000
```

multi-label의 BCE가 라벨마다 어떻게 매겨지는지 한 샘플로 직접 분해해 봅니다. 라벨이 3개 이상 켜진 test 샘플을 골라, 5개 항목별로 정답(0/1)·예측 확률·각각의 BCE 항을 한 줄씩 출력하고 그 평균을 손실로 냅니다. 활성 라벨은 확률이 높을수록, 비활성 라벨은 낮을수록 loss가 작아지는 흐름을 눈으로 확인하세요.

```python
# 여러 라벨이 활성된 test 샘플 하나 고르기 (분해가 잘 보이도록)
multi_active = np.where(Y_test.sum(axis=1) >= 3)[0]
sample_idx = int(multi_active[0]) if len(multi_active) > 0 else 0

y_true = Y_test[sample_idx]
p_pred = proba_ml[sample_idx]
text = X_text_test.iloc[sample_idx]

print("Review preview (200 chars):")
print(f"{text[:200]}...")
print(f"Active labels: {y_true.sum()}\n")

print(f"{'aspect':>10}  {'y_true':>6}  {'pred p':>10}  {'term':>20}  {'loss':>10}")
print("-" * 64)
total_loss = 0.0
for k, aspect in enumerate(ASPECTS):
    y_k, p_k = int(y_true[k]), float(p_pred[k])
    if y_k == 1:
        loss_k = -np.log(max(p_k, 1e-12))
        formula = f"-log({p_k:.4f})"
    else:
        loss_k = -np.log(max(1 - p_k, 1e-12))
        formula = f"-log(1-{p_k:.4f})"
    total_loss += loss_k
    print(f"{aspect:>10}  {y_k:>6d}  {p_k:>10.4f}  {formula:>20}  {loss_k:>10.4f}")
print("-" * 64)
print(f"{'sum':>10}  {'':>6}  {'':>10}  {'':>20}  {total_loss:>10.4f}")
print(f"{'mean BCE':>10}  {'':>6}  {'':>10}  {'/ 5':>20}  {total_loss/5:>10.4f}")
```

**▶ 실행 결과**

```text
Review preview (200 chars):
Okay, so I just went to Vegas for the first time for my 21st birthday. Basically, I LOVE VEGAS! The clubs we went to were absolutely amazing …(뒤 63자 생략)
Active labels: 3

    aspect  y_true      pred p                  term        loss
----------------------------------------------------------------
      food       0      0.4466        -log(1-0.4466)      0.5917
   service       1      0.9027          -log(0.9027)      0.1024
     price       1      0.7366          -log(0.7366)      0.3057
  ambiance       1      0.5771          -log(0.5771)      0.5497
  location       0      0.2620        -log(1-0.2620)      0.3038
----------------------------------------------------------------
       sum                                                1.8533
  mean BCE                                       / 5      0.3707
```

**결과 해석**

활성 라벨(service·price·ambiance)은 확률이 높을수록, 비활성 라벨(food·location)은 낮을수록 loss가 작아집니다. 5개 라벨의 BCE를 따로 구해 평균낸 0.3707이 이 샘플의 손실이며, multi-label은 이렇게 라벨마다 독립적으로 BCE를 매긴다는 게 핵심입니다.

Yelp 데이터에는 multi-label 정답이 없으므로 **5개 항목(aspect)** 별 키워드 사전을 만들어 매칭합니다.

| 항목 | 의미 | 키워드 예시 |
|---|---|---|
| `food` | 음식의 맛/메뉴 | food, meal, dish, taste, delicious, ... |
| `service` | 서비스/응대 | service, staff, waiter, friendly, rude, ... |
| `price` | 가격/가성비 | price, cheap, expensive, value, worth, ... |
| `ambiance` | 분위기/인테리어 | atmosphere, decor, music, vibe, cozy, ... |
| `location` | 위치/주차 | location, parking, area, neighborhood, ... |

각 리뷰 텍스트에 대해 *어떤 키워드라도* 등장하면 해당 항목을 1로 활성화 — 5차원 multi-hot 벡터가 됩니다. **이 합성의 한계** 는 챕터 끝에서 솔직히 짚습니다.

```python
ASPECT_KEYWORDS = {
    "food": ["food", "meal", "dish", "taste", "delicious", "flavor", "menu",
             "cuisine", "tasty", "yummy", "spicy", "sweet", "salty", "fresh"],
    "service": ["service", "staff", "waiter", "waitress", "server", "friendly",
                "rude", "attentive", "host", "helpful", "polite", "manager"],
    "price": ["price", "cheap", "expensive", "value", "worth", "cost",
              "money", "afford", "overpriced", "pricey", "deal", "bargain"],
    "ambiance": ["atmosphere", "ambiance", "decor", "music", "vibe", "cozy",
                 "noisy", "quiet", "lighting", "interior", "comfortable", "loud"],
    "location": ["location", "parking", "area", "neighborhood", "access",
                 "downtown", "convenient", "spot"],
}
ASPECTS = list(ASPECT_KEYWORDS.keys())
K = len(ASPECTS)

def extract_aspects(text: str) -> list[int]:
    text_lower = text.lower()
    return [
        int(any(re.search(rf"\b{re.escape(kw)}\b", text_lower) for kw in keywords))
        for keywords in ASPECT_KEYWORDS.values()
    ]

# 5,000건에 적용
df["aspects"] = df["text"].apply(extract_aspects)
Y = np.array(df["aspects"].tolist())   # (N, 5) multi-hot
print(f"Y shape: {Y.shape}  (n_samples, n_aspects)")
print(f"First 3 multi-hot labels:\n{Y[:3]}")
```

**▶ 실행 결과**

```text
Y shape: (5000, 5)  (n_samples, n_aspects)
First 3 multi-hot labels:
[[0 1 0 0 1]
 [0 0 0 0 1]
 [0 0 0 1 0]]
```

합성한 multi-hot 라벨이 데이터에 어떻게 분포하는지 살펴봅니다. 항목별로 켜진 비율을 구하고, 샘플당 활성 라벨 개수의 평균과 분포를 함께 출력합니다. 라벨마다 빈도가 얼마나 다른지, 한 리뷰가 여러 항목을 동시에 말하는 경우가 실제로 흔한지를 확인하세요.

```python
print("Per-aspect activation rate (over all 5,000 samples):")
for k, aspect in enumerate(ASPECTS):
    print(f"  {aspect:>9}: {Y[:, k].mean():.1%}  ({Y[:, k].sum()} samples)")

n_labels_per_sample = Y.sum(axis=1)
print(f"\nMean active labels per sample: {n_labels_per_sample.mean():.2f}")
print(f"Active label distribution:")
for n in range(K + 1):
    count = (n_labels_per_sample == n).sum()
    print(f"  {n} labels: {count} samples  ({count/len(Y):.1%})")
```

**▶ 실행 결과**

```text
Per-aspect activation rate (over all 5,000 samples):
       food: 55.6%  (2778 samples)
    service: 49.6%  (2480 samples)
      price: 29.4%  (1472 samples)
   ambiance: 18.1%  (905 samples)
   location: 21.9%  (1095 samples)

Mean active labels per sample: 1.75
Active label distribution:
  0 labels: 741 samples  (14.8%)
  1 labels: 1464 samples  (29.3%)
  2 labels: 1506 samples  (30.1%)
  3 labels: 957 samples  (19.1%)
  4 labels: 277 samples  (5.5%)
  5 labels: 55 samples  (1.1%)
```

**결과 해석**

food·service가 절반 안팎으로 흔하고 ambiance는 18%로 드물어 라벨 빈도가 크게 다릅니다. 샘플당 평균 1.75개 라벨이 켜지고 2개 이상인 경우가 절반을 넘으니, 한 리뷰가 여러 항목을 동시에 말한다는 multi-label 가정이 데이터에 실제로 나타납니다.

데이터를 train/test로 8:2로 나눈 뒤, 텍스트를 TF-IDF 벡터로 바꿔 모델 입력을 준비합니다. 벡터화 기준은 train에서만 학습(`fit_transform`)하고 test에는 그대로 적용(`transform`)해 정보 누출을 막습니다. 출력되는 shape에서 Y가 `(N, 5)` 형태의 multi-hot임을 확인하세요.

```python
X_text_train, X_text_test, Y_train, Y_test = train_test_split(
    df["text"], Y, test_size=0.2, random_state=42,
)

tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"X_test:  {X_test.shape}, Y_test:  {Y_test.shape}")
```

**▶ 실행 결과**

```text
X_train: (4000, 10000), Y_train: (4000, 5)
X_test:  (1000, 10000), Y_test:  (1000, 5)
```

## 실습 2: `OneVsRestClassifier` (이번엔 argmax 없이)

Ch 5에서 multi-class의 *대안* 으로 본 `OneVsRestClassifier`가 이번엔 *기본 도구* 입니다. 코드 차이는 단 하나 — **`Y_train` shape이 `(N,)` → `(N, K)`** 로 바뀌는 것. sklearn이 multi-hot Y를 보고 multi-label 모드로 자동 전환해 K개 binary 분류기가 독립적으로 학습됩니다.

가장 중요한 사용 방식 변화는 **argmax가 사라진다는 것**.

- **Ch 5 OvR (multi-class)**: 5개 sigmoid 출력 중 가장 큰 것 *하나만* 정답으로 골랐음.
- **Ch 6 OvR (multi-label)**: 5개 sigmoid 출력을 *각자* 임계값 0.5와 비교해 0/1 결정. 동시 활성 가능.

`predict_proba`도 차이가 있습니다 — Ch 5에서는 sklearn이 후처리로 합=1로 정규화해줬지만, multi-label 모드에서는 *정규화하지 않고* 각 라벨의 P(label_k = 1)을 그대로 반환합니다 (각 라벨이 독립이니까).

```python
# 먼저 wrapper 없이 그냥 LogisticRegression() 에 multi-hot Y를 넣어보기
bare_model = LogisticRegression(max_iter=1000)
try:
    bare_model.fit(X_train, Y_train)
    print(f"Succeeded? coef_ shape: {bare_model.coef_.shape}")
except ValueError as e:
    print(f"Failed: {type(e).__name__}")
    print(f"   Message: {e}")
```

**▶ 실행 결과**

```text
Failed: ValueError
   Message: y should be a 1d array, got an array of shape (4000, 5) instead.
```

**왜 실패했나?** sklearn `LogisticRegression` 은 *1D Y* (각 샘플당 한 클래스 인덱스)만 받습니다. 우리의 `Y_train.shape == (N, 5)` 는 "한 샘플에 여러 라벨"이라는 의미인데 *형식 자체* 가 호환되지 않아요. fit이 첫 줄에서 죽습니다.

`OneVsRestClassifier` 의 역할은 단순합니다.

1. 2D Y를 K개의 1D 컬럼으로 **쪼개고**,
2. 각 컬럼마다 `LogisticRegression` 을 **별도로** 학습 (총 K개 모델),
3. 결과를 `.estimators_` 리스트에 보관해 `predict` 시 모두 적용.

알고리즘은 동일한 LogReg지만 **fit 시점의 Y 형식 처리** 가 결정적 차이입니다 — bare 호출은 죽고, wrapper 호출은 K개 모델로 분할 학습됩니다.

```python
# 위 실패와 대비: wrapper 한 줄로 K개 LogReg가 자동 분할 학습됨
model_ml = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_ml.fit(X_train, Y_train)

print(f"OvR fit succeeded!")
print(f"  Number of binary classifiers: {len(model_ml.estimators_)}")
print(f"  Each estimator type:          {type(model_ml.estimators_[0]).__name__}")
print(f"  Each estimator coef shape:    {model_ml.estimators_[0].coef_.shape}  (1, V — one binary per label)")
print(f"\nLabel for each classifier:")
for k, aspect in enumerate(ASPECTS):
    n_pos = Y_train[:, k].sum()
    print(f"  estimator[{k}] = '{aspect}': {n_pos} positives ({n_pos/len(Y_train):.1%})")

# 예측 + 확률
Y_pred = model_ml.predict(X_test)         # (N, K) multi-hot 0/1 (threshold 0.5 자동 적용)
proba_ml = model_ml.predict_proba(X_test) # (N, K) per-label probability (정규화 X)

print(f"\nY_pred shape: {Y_pred.shape}")
print(f"proba shape:  {proba_ml.shape}")
print(f"\nFirst 3 sample predicted probabilities (per-label):")
print(pd.DataFrame(proba_ml[:3], columns=ASPECTS).round(4))
```

**▶ 실행 결과**

```text
OvR fit succeeded!
  Number of binary classifiers: 5
  Each estimator type:          LogisticRegression
  Each estimator coef shape:    (1, 10000)  (1, V — one binary per label)

Label for each classifier:
  estimator[0] = 'food': 2214 positives (55.4%)
  estimator[1] = 'service': 1961 positives (49.0%)
  estimator[2] = 'price': 1196 positives (29.9%)
  estimator[3] = 'ambiance': 720 positives (18.0%)
  estimator[4] = 'location': 892 positives (22.3%)

Y_pred shape: (1000, 5)
proba shape:  (1000, 5)

First 3 sample predicted probabilities (per-label):
     food  service   price  ambiance  location
0  0.3002   0.3082  0.7970    0.1701    0.1122
1  0.3607   0.3612  0.4585    0.1855    0.1662
2  0.2121   0.2894  0.1212    0.1387    0.0692
```
