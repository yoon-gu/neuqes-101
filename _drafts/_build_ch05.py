"""Build 05_sklearn_multiclass/05_sklearn_multiclass.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "05_sklearn_multiclass" / "05_sklearn_multiclass.ipynb"

cells = []
_counter = 0


def _cid():
    global _counter
    _counter += 1
    return f"cell{_counter:03d}"


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": text,
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ----- 1. Title -----
md(r"""# Chapter 5. sklearn Multi-class — K=5로 진짜 일반화

**목표**: Ch 4에서 본 softmax+CE를 K=2에서 K=5로 그대로 확장합니다. 모델·loss·코드 변화는 거의 없습니다 — 데이터의 클래스 수만 늘어납니다. 회귀(Ch 2)와 5클래스 분류가 같은 데이터를 어떻게 다르게 해석하는지도 확인합니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분

---

## 학습 흐름

1. 🚀 **실습**: 별점 1-5를 5개 독립 클래스로 보고 multinomial LogReg로 분류
2. 🔬 **해부**: 5×5 confusion matrix가 대각선 근처에 몰리는 ordinal 흔적 관찰
3. 🛠️ **변형**: `multi_class="multinomial"` vs `"ovr"` 비교 — Ch 6 multi-label로 가는 다리""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression(multi_class="multinomial")` | `TfidfVectorizer()` | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` |
| **5 ← 여기** | `LogisticRegression(multi_class="multinomial")` | `TfidfVectorizer()` | Yelp 5클래스 (별점 0-4) | **(5차원)** | softmax | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 4)

| 축 | Ch 4 | Ch 5 |
|---|---|---|
| 데이터 | Yelp 이진화 (K=2) | **Yelp 5클래스 (K=5)** |
| Output Head | (2차원) | **(5차원)** |
| 활성화·Loss | softmax + CE | softmax + CE (그대로) |
| 모델·토크나이저 | LogReg(multinomial) + TF-IDF | 그대로 |

**한 가지 변화** — K=2가 K=5로 늘어난 것. softmax/CE는 K가 무엇이든 같은 식이라 코드 변화도 거의 없습니다. 이게 softmax/CE의 진짜 가치 — sigmoid+BCE는 K=2 전용이지만 softmax+CE는 자연스럽게 다중 클래스로 확장됩니다.""")

# ----- 4. Loss 노트 (작은 수치 표) -----
md(r"""## 📐 Loss 노트 — 같은 CE, K=5 수치 예시

Loss는 Ch 4와 동일한 `CrossEntropyLoss`. K가 늘어나도 식은 그대로:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i,\, y_i}$$

**숫자로 감 잡기** (K=5, 정답 클래스 = 2 인 한 샘플):

| 예측 분포 $[\hat p_0, \hat p_1, \hat p_2, \hat p_3, \hat p_4]$ | 정답 확률 $\hat p_2$ | 손실 $-\log \hat p_2$ |
|---|---|---|
| **정답에 집중**: `[0.05, 0.05, 0.80, 0.05, 0.05]` | 0.80 | 0.223 |
| **균등(uniform)**: `[0.20, 0.20, 0.20, 0.20, 0.20]` | 0.20 | 1.609 |
| **틀린 클래스에 집중**: `[0.05, 0.05, 0.05, 0.05, 0.80]` | 0.05 | **2.996** |

**baseline = $\log K = \log 5 \approx 1.609$**: 모델이 아무 정보 없이 균등 추측만 할 때의 손실. 학습된 모델은 이보다 작아야 하고, baseline을 초과하면 "정답이 *아닌* 곳에 자신 있다"는 신호 — gradient가 모델을 강하게 끌어당깁니다.""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

**Ch 1-4와 동일한 `TfidfVectorizer`**. 입력 표현, 모델, loss 모두 그대로 — 변하는 건 데이터의 클래스 수와 출력 차원뿐.

> **다음 챕터(Ch 6)**: 같은 TF-IDF. 변화는 활성화 함수가 softmax(합=1)에서 **K개 독립 sigmoid**(라벨 간 독립)로 일반화되어 multi-label로 확장.""")

# ----- 6. install -----
code(r"""!pip install -q datasets scikit-learn pandas matplotlib""")

# ----- 7. import + load -----
code(r"""import warnings
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

dataset = load_dataset("yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
print(f"Total samples: {len(df)}")
print("Class distribution (label 0-4 = star 1-5):")
print(df["label"].value_counts().sort_index())""")

# ----- 8. 5-class 데이터 -----
code(r"""# 5-class 데이터 (라벨이 이미 0-4 — 별점 1-5)
y_5class = df["label"]

X_text_train, X_text_test, y_train, y_test = train_test_split(
    df["text"], y_5class, test_size=0.2, random_state=42, stratify=y_5class,
)

tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, y_train distribution: {pd.Series(y_train).value_counts().sort_index().tolist()}")""")

# ----- 9. 실습 도입 -----
md(r"""## 🚀 실습: 5클래스 분류

코드는 Ch 4에서 `multi_class="multinomial"` 그대로 — sklearn은 K가 클래스 개수에 자동으로 맞춥니다.""")

# ----- 10. fit -----
code(r"""model_5 = LogisticRegression(multi_class="multinomial", max_iter=1000)
model_5.fit(X_train, y_train)

y_pred = model_5.predict(X_test)
acc = accuracy_score(y_test, y_pred)
baseline = 1 / 5

print(f"Test accuracy: {acc:.4f}")
print(f"baseline (uniform guess): {baseline:.4f}")
print(f"Improvement over baseline: {acc - baseline:+.4f}")""")

# ----- 11. predict_proba shape -----
code(r"""proba_5 = model_5.predict_proba(X_test)
print(f"predict_proba shape: {proba_5.shape}  (N, K=5)")
print(f"Row sums (should be 1): {proba_5.sum(axis=1)[:5].round(4)}")
print(f"\nFirst 3 sample probability distributions:")
print(pd.DataFrame(proba_5[:3], columns=[f"P({i+1}★)" for i in range(5)]).round(3))""")

# ----- 12. 해부 -----
md(r"""## 🔬 해부: 평가 지표 한꺼번에 보기

5클래스에서는 클래스마다 precision/recall/F1이 따로 정의됩니다. `classification_report`가 한 번에 정리.""")

# ----- 13. classification_report -----
code(r"""print(classification_report(y_test, y_pred, target_names=[f"{i+1}★" for i in range(5)]))""")

# ----- 14. confusion matrix -----
code(r"""cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"true {i+1}★" for i in range(5)],
    columns=[f"pred {i+1}★" for i in range(5)],
)
print(cm_df)""")

# ----- 15. ordinal 관찰 + 시각화 -----
md(r"""**관찰**: confusion matrix의 오답이 **대각선 근처에 몰립니다**.

- 4점을 5점으로 헷갈리는 경우는 많아도 4점을 1점으로 헷갈리는 경우는 거의 없습니다.
- 분류 모델은 클래스 간 거리(별점 1과 별점 2가 가깝다는 정보)를 *명시적으로* 모릅니다 — 5개 독립 클래스로 처리할 뿐입니다.
- 그런데도 오답이 대각선 근처에 모이는 이유: **데이터 자체가 ordinal**. 1점 리뷰의 단어 분포와 2점 리뷰의 단어 분포가 비슷해서, 모델이 자연스럽게 "비슷한 클래스끼리 헷갈림" 패턴을 학습.

**회귀(Ch 2)와 비교** — 같은 데이터, 다른 관점:

| 관점 | 가정 | 손실 | 4점을 5점으로 vs 1점으로 |
|---|---|---|---|
| 회귀 (Ch 2) | 별점 사이 **거리** 가 의미 있음 | MSE | 1 vs 16 (다른 페널티) |
| 분류 (Ch 5) | 5개 **독립 클래스** | CE | 둘 다 그냥 "틀림" (같은 페널티) |

회귀는 ordinal 정보를 loss에 반영하고, 분류는 클래스별 패턴(예: 1점 리뷰 욕설 vs 5점 리뷰 칭찬)에 더 집중합니다. 둘 중 어느 게 나은지는 케바케.""")

# ----- 16. multinomial vs OvR (개념) -----
md(r"""## 🛠️ 변형: multinomial vs OvR

`multi_class="multinomial"` 은 한 모델이 K개 logit을 **동시에** 학습합니다. softmax 한 번이라 합 = 1이 강제 — "K개 클래스 중 정확히 하나"라는 *상호배타* 가정.

또 다른 방식 **OvR (One-vs-Rest)** 은 K개의 *독립* binary 분류기. 각 분류기는 "이 클래스 vs 나머지 모든 클래스"만 학습합니다.

### 두 방식의 구조 비교

**multinomial (softmax)**:

```
[입력 x] ──→ Linear(V → K) ──→ logits [z_1, ..., z_K]
                            ──→ softmax 한 번
                            ──→ [p_1, ..., p_K]   (합 = 1, 클래스끼리 경쟁)
```

**OvR** (이 챕터의 K=5 예시):

```
             ┌──→ 분류기 1:  "1★ vs 나머지"  ──→ sigmoid ──→ P_1
             ├──→ 분류기 2:  "2★ vs 나머지"  ──→ sigmoid ──→ P_2
[입력 x] ──→ ├──→ 분류기 3:  "3★ vs 나머지"  ──→ sigmoid ──→ P_3
             ├──→ 분류기 4:  "4★ vs 나머지"  ──→ sigmoid ──→ P_4
             └──→ 분류기 5:  "5★ vs 나머지"  ──→ sigmoid ──→ P_5

   각 P_k 는 다른 P_j 와 무관한 독립 sigmoid 출력 (raw 합 ≠ 1)
   예측: argmax(P_k) — sklearn이 표시할 땐 행을 정규화해 합 1로 보여줌
```

핵심 차이는 **클래스가 서로 경쟁하느냐** 입니다. multinomial은 한 logit이 커지면 다른 logit의 softmax 확률이 자동으로 줄어듭니다 (분모 공유). OvR의 각 sigmoid는 다른 클래스 학습과 독립적이라 P_k 가 모두 동시에 0.8이 될 수도, 모두 0.1이 될 수도 있습니다.""")

# ----- 17. OvR 학습 + 한 샘플 비교 -----
code(r"""# OvR은 sklearn.multiclass.OneVsRestClassifier로 만듭니다.
# 내부적으로 K개 binary LogisticRegression이 따로 학습되어 model_ovr.estimators_ 에 들어갑니다.
model_ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_ovr.fit(X_train, y_train)

print(f"OvR estimators count: {len(model_ovr.estimators_)}")
print(f"Each estimator is a separate LogisticRegression for 'class k vs rest'")
print(f"  estimator 0 coef_ shape: {model_ovr.estimators_[0].coef_.shape}  (1, V)")

# 5개 binary 모델의 coef를 (5, V)로 쌓아 한 번에 행렬 곱
ovr_coef = np.vstack([est.coef_[0] for est in model_ovr.estimators_])         # (5, V)
ovr_intercept = np.array([est.intercept_[0] for est in model_ovr.estimators_]) # (5,)

ovr_logits_all = np.asarray(X_test @ ovr_coef.T) + ovr_intercept
ovr_sigmoid_all = 1.0 / (1.0 + np.exp(-ovr_logits_all))    # (N, 5) 독립 sigmoid

# 한 test 샘플을 골라 두 방식의 K=5 출력을 나란히 비교
sample_idx = 0
sample_text = X_text_test.iloc[sample_idx]
true_label = y_test.iloc[sample_idx]

p_multi = proba_5[sample_idx]                                  # multinomial softmax (합 = 1)
p_ovr_raw = ovr_sigmoid_all[sample_idx]                        # OvR 5개 독립 sigmoid (정규화 전)
p_ovr_norm = model_ovr.predict_proba(X_test)[sample_idx]       # OvR 정규화 후 (sklearn 표시용)

print("\nReview preview (200 chars):")
print(f"{sample_text[:200]}...")
print(f"True star:    {true_label + 1}★\n")

print(f"{'class':>8}  {'multinomial':>14}  {'OvR raw':>10}  {'OvR normalized':>16}")
print("-" * 56)
for k in range(5):
    print(f"  {k+1}★    {p_multi[k]:>14.4f}  {p_ovr_raw[k]:>10.4f}  {p_ovr_norm[k]:>16.4f}")
print("-" * 56)
print(f"  sum    {p_multi.sum():>14.4f}  {p_ovr_raw.sum():>10.4f}  {p_ovr_norm.sum():>16.4f}")""")

# ----- 18. 해석 + multi-label 떡밥 -----
md(r"""**관찰**

- **multinomial 열**: 깨끗한 분포, 합 = 1. "이 문서는 K개 별점 중 어느 *한* 별점일 확률"을 나타냄.
- **OvR raw 열**: 5개 sigmoid가 서로 독립적으로 작동한 결과. 합이 1이 아닙니다 (보통 1보다 크거나 작음).
- **OvR 정규화 후 열**: sklearn이 raw 값을 행 합으로 나눠 합=1을 만들어준 것. 표시용일 뿐 모델의 본래 출력은 아닙니다.

**왜 이 차이가 중요한가** (Ch 6 떡밥)

- multinomial의 합=1 제약은 "이 문서의 별점은 정확히 *하나* 다"라는 데이터 가정에 잘 맞습니다.
- 그러나 한 문서가 여러 라벨을 가질 수 있는 *multi-label* 문제에서는 이 가정이 깨집니다 — 영화는 "로맨스 + 코미디"일 수 있고, 식당 리뷰는 "음식 + 서비스 + 가격"을 동시에 다룰 수 있습니다.
- multi-label은 **OvR의 사고방식을 그대로** 가져갑니다: K개 독립 sigmoid를 별도로 학습하고, **정규화하지 않습니다**. 각 라벨이 독립적으로 0/1을 결정하는 것이 그 모델의 본래 모습.
- 그래서 OvR을 multi-class에서 미리 만나두는 게 다음 챕터의 다리가 됩니다.""")

# ----- 19. 전체 정확도 + raw 합 분포 -----
code(r"""# 전체 test set 정확도 비교
acc_ovr = accuracy_score(y_test, model_ovr.predict(X_test))
print(f"multinomial accuracy: {acc:.4f}")
print(f"OvR accuracy:         {acc_ovr:.4f}")
print(f"Diff: {abs(acc - acc_ovr):.4f}")

# OvR raw 확률 행 합 분포 (정규화 전)
raw_sums = ovr_sigmoid_all.sum(axis=1)
print(f"\nOvR raw row sum distribution (pre-normalization):")
print(f"  min:  {raw_sums.min():.3f}")
print(f"  max:  {raw_sums.max():.3f}")
print(f"  mean: {raw_sums.mean():.3f}")
print(f"  → 5 independent sigmoids; rows do not sum to exactly 1")""")

# ----- 18. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `sklearn.multiclass.OneVsRestClassifier` | K개 독립 binary 분류기를 묶어 estimators_ 로 노출 | Ch 6 multi-label의 핵심 도구로 그대로 재등장 |
| `sklearn.metrics.confusion_matrix` | 다중 클래스도 K×K로 일반화 | 분류 챕터마다 사용 |
| `sklearn.metrics.classification_report` | per-class precision/recall/F1 한 번에 | — |""")

# ----- 19. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. K=5 균등 분포일 때 CE 손실값은 얼마인가요? 학습된 모델은 이보다 작아야 하는 이유는?
2. 5×5 confusion matrix에서 오답이 대각선 근처에 몰리는 현상은 모델의 어떤 가정과 데이터의 어떤 성질 사이에서 나오나요?
3. multinomial과 OvR의 핵심 차이를 한 문장으로 요약해보세요.
4. 같은 별점 데이터를 회귀(Ch 2)로 풀 때와 5클래스 분류(Ch 5)로 풀 때 어떤 종류의 실수에 더 큰 페널티를 주나요?""")

# ----- 20. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) multinomial과 OvR은 어떻게 다른가요?

핵심 차이는 **클래스 간 의존성**.

- **multinomial (softmax)**: 한 모델이 K개 logit을 동시에 학습. softmax → 확률 합 = 1, 클래스 *상호배타*.
- **OvR (One-vs-Rest)**: K개 *독립* binary 분류기 (`OneVsRestClassifier`로 명시적으로 구성). 클래스 0의 logit은 다른 클래스 학습에 영향 없음. 정규화 전 확률 합이 1이 아닐 수 있음.

| 상황 | 적합한 방식 |
|---|---|
| 한 샘플에 정확히 한 라벨 (별점, 뉴스 카테고리) | multinomial |
| 한 샘플에 여러 라벨 가능 (영화 장르: 로맨스+코미디) | OvR (Ch 6) |
| 클래스 수백 개 + 빠른 학습 필요 | OvR (binary들이 병렬 학습 쉬움) |

### Q2. (실무) 클래스 수가 100개를 넘어가면 학습이 느려지는데 어떻게 하나요?

세 가지 흔한 처리법.

1. **Hierarchical classification**: 큰 그룹 → 세부 분류. 예: 의류 → 상의/하의 → 셔츠/티셔츠.
2. **희귀 클래스 묶기**: 빈도 1-2회짜리는 "기타"로. long tail 80%는 어차피 못 배움.
3. **계산 트릭**: hierarchical softmax, sampled softmax (학습 시 일부 클래스만 negative). PyTorch 모델에서 자주, sklearn 기본엔 없음.

K=5 정도는 위 트릭이 필요 없는 작은 규모.

### Q3. (실무) 클래스 불균형이 심한데 `class_weight`를 어떻게 적용하나요?

```python
model = LogisticRegression(
    multi_class="multinomial",
    class_weight="balanced",   # 빈도의 역수로 자동 가중치
    max_iter=1000,
)
```

`balanced`는 빈도 적은 클래스에 더 큰 가중치를 줍니다. 또는 `class_weight={0: 0.5, 1: 1.0, ...}` 으로 직접 지정 가능. 효과가 미미하면 데이터 레벨 처리(SMOTE 등) 또는 임계값 후처리.

### Q4. (실무) confusion matrix를 시각화하는 추천 방법은?

`seaborn.heatmap`이 깔끔합니다.

```python
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"{i+1}★" for i in range(5)],
            yticklabels=[f"{i+1}★" for i in range(5)])
plt.xlabel("Predicted"); plt.ylabel("True")
```

**행 정규화**(recall 관점)도 자주 봅니다.

```python
cm_norm = cm / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
```

### Q5. (이론) macro F1과 weighted F1 중 어떤 걸 봐야 하나요?

- **macro F1**: 클래스별 F1을 단순 평균. 클래스 불균형이 있으면 소수 클래스 성능을 가리지 않음.
- **weighted F1**: 클래스 빈도로 가중 평균. 다수 클래스 성능을 더 반영.

**언제 무엇:**
- 모든 클래스가 똑같이 중요 → macro F1.
- 다수 클래스 위주로 평가하고 싶음 → weighted F1.
- 보고에는 보통 둘 다 함께 적습니다.

### Q6. (이론) baseline = $\log K$의 의미는 무엇이고 왜 챙겨봐야 하나요?

CE 식에 균등 분포 $\hat p_k = 1/K$ 를 대입하면:

$$L = -\log(1/K) = \log K$$

**의미**: 모델이 *아무것도 안 학습한 상태* 의 손실. 학습 손실이 baseline보다 크면 "정답이 *아닌* 곳에 자신 있다"는 뜻 — 모델이 잘못된 방향으로 가고 있습니다.

K=5에서 baseline ≈ 1.609. 학습 도중 loss가 이 값 근처에서 정체되면 "모델이 데이터에서 신호를 못 잡고 있다"는 진단.""")

# ----- 21. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

같은 5클래스 데이터에서 회귀 모델(Ch 2 방식)과 분류 모델(이번 챕터)이 4★를 3★로 예측하는 비율과 4★를 1★로 예측하는 비율의 비를 각각 계산해보세요. 회귀가 더 큰 실수를 더 강하게 회피하는지 확인할 수 있습니다.

```python
from sklearn.linear_model import LinearRegression

# 회귀 모델 (Ch 2 그대로)
model_reg = LinearRegression()
model_reg.fit(X_train, y_train.astype(float))
pred_reg = np.clip(np.round(model_reg.predict(X_test)), 0, 4).astype(int)

# 분류 모델 (이번 챕터)
pred_clf = y_pred

for name, pred in [("회귀 후 round/clip", pred_reg), ("분류", pred_clf)]:
    mask4 = (y_test == 3)   # 라벨 3 = 4★
    n_4to3 = ((pred == 2) & mask4).sum()  # 4★ → 3★ 오답
    n_4to1 = ((pred == 0) & mask4).sum()  # 4★ → 1★ 오답
    print(f"{name}: 4★→3★ {n_4to3}건, 4★→1★ {n_4to1}건")
```

힌트: 회귀는 큰 실수일수록 손실이 제곱으로 커지므로 4★을 1★로 예측하는 큰 실수가 더 드물어야 합니다. 분류는 둘을 동등하게 처벌하므로 그런 경향이 약할 수 있습니다.""")

# ----- 22. next -----
md(r"""## 다음 챕터 예고

**Chapter 6. sklearn Multi-label — softmax 합=1 제약을 푼다**

- 한 샘플에 *여러* 라벨이 동시에 붙는 multi-label 문제로 확장
- 새 데이터: Yelp 리뷰 + **측면(aspect) 키워드 합성** (food/service/price/ambiance/location 5개)
- softmax 한 개 대신 **5개 독립 sigmoid** — 각 라벨이 다른 라벨에 영향받지 않음
- Loss는 CrossEntropyLoss에서 **per-label `BCEWithLogitsLoss`** 로
- `OneVsRestClassifier(LogisticRegression())` + `MultiLabelBinarizer`로 구현""")


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT.relative_to(REPO)}  ({len(cells)} cells)")
