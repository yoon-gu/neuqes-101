"""Build 03_sklearn_binary/03_sklearn_binary.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "03_sklearn_binary" / "03_sklearn_binary.ipynb"

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
md(r"""# Chapter 3. sklearn Binary — 출력에 sigmoid가 붙다

**목표**: Ch 2의 회귀에서 한 단계 더 — 출력 직전에 **sigmoid** 가 붙어 [0, 1]을 강제하고, loss는 **BCE** 로 바뀝니다. 라벨도 정수 0/1로 이진화됩니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분 (학습은 즉시)

---

## 학습 흐름

1. 🚀 **실습**: Yelp 별점을 이진화(4-5 → 1, 1-2 → 0)하고 `LogisticRegression`으로 학습
2. 🔬 **해부**: BCE 수식 + sigmoid가 어떻게 logit을 확률로 바꾸는지 직접 재현
3. 🛠️ **변형**: 임계값(threshold) 0.5를 다른 값으로 옮기면 precision/recall이 어떻게 움직이나""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 샘플 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| **3 ← 여기** | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 (4-5→1, 1-2→0, 3 제외) | (1차원) | **sigmoid** | **`BCEWithLogitsLoss`** (sklearn: log loss) |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 2)

| 축 | Ch 2 | Ch 3 |
|---|---|---|
| 모델 | `LinearRegression` | **`LogisticRegression`** (출력에 sigmoid 내장) |
| Activation | 없음 | **sigmoid** |
| Loss | `MSELoss` | **`BCEWithLogitsLoss`** |
| 라벨 | float (1-5) | **int (0/1)** |
| 데이터 | Yelp 별점 1-5 | **Yelp 이진화** (3 제외) |
| 토크나이저 | TF-IDF | TF-IDF (그대로) |

겉보기엔 다섯 곳이 바뀌었지만, 한 가지 결정 — **"회귀 → 이진 분류"** — 이 자동으로 끌어오는 결과입니다. 분류 패러다임은 (출력 형태 = sigmoid, loss = BCE, 라벨 = 0/1, 데이터 = 이진화)가 한 묶음으로 따라옵니다.

**왜 이렇게 묶이나?** 회귀에서 본 한계 — "출력이 [0, 1]을 못 지킨다" — 를 모델 단계에서 강제로 해결하는 게 sigmoid입니다. 그러면 출력은 자연스럽게 "1일 확률"로 해석되고, 그 확률에 어울리는 loss가 BCE이고, 라벨도 0/1 정수로 단순해집니다.""")

# ----- 4. Loss 변화 -----
md(r"""## 📐 Loss 함수의 변화 — `BCEWithLogitsLoss` 등장

**Binary Cross Entropy** 는 모델이 뱉은 확률 $\hat p_i$ 와 정답 $y_i \in \{0, 1\}$ 사이의 차이를 잽니다.

$$L = -\frac{1}{N}\sum_{i=1}^{N}\left[\,y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\,\right]$$

핵심: $y_i = 1$이면 첫 항 $-\log \hat p_i$만 살고, $y_i = 0$이면 둘째 항 $-\log(1 - \hat p_i)$만 살아남습니다. 즉 한 샘플당 항상 한 항만 작동합니다.

**숫자로 감 잡기** (정답이 $y = 1$인 샘플 한 개, $N = 1$):

| 정답 $y$ | 예측 확률 $\hat p$ | 손실 $-\log \hat p$ |
|---|---|---|
| 1 | 0.9 | 0.105 |
| 1 | 0.5 | 0.693 |
| 1 | 0.1 | **2.303** |
| 1 | 0.01 | **4.605** |

확률이 정답에서 멀어질수록 — 특히 0에 가까워질수록 — 손실이 **로그 스케일로 폭증** 합니다. 자신 있게 틀린 예측을 강하게 처벌하는 게 BCE의 성격입니다.

**`BCEWithLogits`의 "Logits" 의미**: 모델 마지막 단의 raw 점수(logit) $z = w^\top x + b$를 sigmoid에 넣기 *전* 의 값을 의미합니다. PyTorch의 `BCEWithLogitsLoss`는 logit을 받아 내부에서 sigmoid + BCE를 한 번에 계산하는데, 따로 sigmoid를 통과시킨 뒤 BCE를 적용하는 것보다 수치적으로 안정적입니다.

```python
# PyTorch (Ch 9 이후)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets.float())   # logits: 활성화 전 raw 점수

# sklearn (이번 챕터)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)                              # 내부에서 sigmoid + log loss
```""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

이번 챕터의 토크나이저는 **Ch 1·2와 동일한 `TfidfVectorizer`** 입니다. 입력 표현은 그대로고, 변화는 모델 출력단·loss·라벨에서만 일어납니다.

> **다음 챕터(Ch 4)**: 같은 TF-IDF 그대로. 변하는 건 출력이 5차원으로 늘어나고 sigmoid가 softmax로 바뀌는 것뿐.""")

# ----- 6. install -----
code(r"""!pip install -q datasets scikit-learn pandas matplotlib""")

# ----- 7. import + load -----
code(r"""import numpy as np
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

dataset = load_dataset("yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
df["star"] = df["label"] + 1   # 0-4 → 1-5
print(f"Total samples: {len(df)}")
print(df["star"].value_counts().sort_index())""")

# ----- 8. 이진화 -----
code(r"""# 별점 3은 애매하므로 제외, 4-5 → 1 (positive), 1-2 → 0 (negative)
df_bin = df[df["star"] != 3].copy()
df_bin["y"] = (df_bin["star"] >= 4).astype(int)

print(f"Binarized samples: {len(df_bin)}  (3-star excluded)")
print(f"Class distribution:\n{df_bin['y'].value_counts().sort_index()}")
print(f"Positive rate: {df_bin['y'].mean():.1%}")""")

# ----- 9. TF-IDF + split -----
code(r"""X_text_train, X_text_test, y_train, y_test = train_test_split(
    df_bin["text"], df_bin["y"],
    test_size=0.2, random_state=42, stratify=df_bin["y"],
)

tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")""")

# ----- 10. 실습 도입 -----
md(r"""## 🚀 실습: `LogisticRegression`으로 이진 분류

`LogisticRegression`이 내부에서 하는 일은 두 단계입니다.

1. logit 계산: $z = w^\top x + b$ (Ch 2와 똑같은 선형 결합)
2. sigmoid로 확률 변환: $\hat p = \sigma(z) = \dfrac{1}{1 + e^{-z}}$

학습은 BCE를 최소화하는 $w, b$를 찾는 것이고, 예측은 $\hat p \geq 0.5$를 기준으로 0/1로 자릅니다.""")

# ----- 11. fit + accuracy -----
code(r"""model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")""")

# ----- 12. predict_proba -----
code(r"""# predict_proba는 [P(y=0), P(y=1)] 형태로 두 확률을 모두 줌
y_proba = model.predict_proba(X_test)
print(f"y_proba shape: {y_proba.shape}  (per sample: [P(0), P(1)])")
print(f"\nFirst 5 predicted probabilities:")
print(pd.DataFrame(y_proba[:5], columns=["P(neg)", "P(pos)"]).round(4))

# 두 확률을 합치면 항상 1
print(f"\nRow sums (should be 1): {y_proba.sum(axis=1)[:5]}")""")

# ----- 13. 해부 -----
md(r"""## 🔬 해부: sigmoid는 logit을 어떻게 확률로 바꾸나

`LogisticRegression`의 `coef_`와 `intercept_`는 Ch 2 `LinearRegression`과 동일한 선형 가중치입니다. 차이는 그 위에 sigmoid가 한 번 더 붙는다는 것뿐.

직접 재현해 봅시다 — sklearn의 `predict_proba`가 정말 sigmoid에 logit을 넣은 결과인지 확인합니다.""")

# ----- 14. sigmoid 직접 재현 -----
code(r"""# 모델의 logit 직접 계산: z = X · w + b
# (sparse 행렬이라 .toarray() 안 쓰고 sparse dot product 사용)
logits = X_test @ model.coef_.T + model.intercept_   # shape: (N, 1)
logits = logits.flatten()

# sigmoid 적용
proba_manual = 1 / (1 + np.exp(-logits))
proba_sklearn = y_proba[:, 1]   # P(y=1)

# 둘이 같은가?
diff = np.abs(proba_manual - proba_sklearn).max()
print(f"Max diff (manual vs sklearn): {diff:.2e}")

print(f"\nManual first 5:  {proba_manual[:5].round(4)}")
print(f"sklearn first 5: {proba_sklearn[:5].round(4)}")""")

# ----- 15. log_loss 직접 재현 -----
code(r"""# BCE(log loss)도 직접 계산 가능
# 정답이 1이면 -log(p), 0이면 -log(1-p)
y_test_arr = y_test.values
p = proba_sklearn
manual_bce = -(y_test_arr * np.log(p) + (1 - y_test_arr) * np.log(1 - p)).mean()
sklearn_bce = log_loss(y_test, y_proba)

print(f"Manual BCE:  {manual_bce:.6f}")
print(f"sklearn BCE: {sklearn_bce:.6f}")
print(f"Diff:        {abs(manual_bce - sklearn_bce):.2e}")""")

# ----- 16. classification_report -----
code(r"""print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

cm = confusion_matrix(y_test, y_pred)
print(f"\nconfusion matrix:\n{cm}")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}\n  FN={cm[1,0]}, TP={cm[1,1]}")""")

# ----- 17. 변형 도입 -----
md(r"""## 🛠️ 변형: 임계값(threshold)을 옮기면

`predict()`는 기본적으로 $\hat p \geq 0.5$를 기준으로 0/1을 자릅니다. 이 임계값을 다른 값으로 옮기면 precision과 recall이 정반대로 움직입니다.

- 임계값 ↑ (예: 0.7): "확실해야만 positive" → **precision 상승**, recall 하락
- 임계값 ↓ (예: 0.3): "조금만 의심돼도 positive" → **recall 상승**, precision 하락

스팸 필터(precision 중시), 암 진단(recall 중시) 같은 도메인 요구에 따라 임계값을 조정합니다.""")

# ----- 18. threshold sweep -----
code(r"""thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
proba_pos = y_proba[:, 1]

rows = []
for t in thresholds:
    y_pred_t = (proba_pos >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_t, average="binary", zero_division=0
    )
    acc = accuracy_score(y_test, y_pred_t)
    rows.append({"threshold": t, "accuracy": acc, "precision": p, "recall": r, "f1": f1})

df_t = pd.DataFrame(rows).round(4)
print(df_t.to_string(index=False))""")

# ----- 19. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `sklearn.linear_model.LogisticRegression` | sigmoid + BCE 내장 이진/다중 분류 | Ch 4에서 multi-class로 확장 |
| `sklearn.metrics.classification_report` | accuracy/precision/recall/F1 한 번에 | 분류 챕터마다 계속 사용 |
| `sklearn.metrics.confusion_matrix` | 혼동 행렬 | 다중 분류에서도 활용 |
| `sklearn.metrics.log_loss` | BCE 평가 | Ch 9 이후 BERT binary에서도 |
| `sklearn.metrics.precision_recall_fscore_support` | 임계값별 지표 추적 | — |""")

# ----- 20. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. `LogisticRegression`이 내부에서 하는 두 단계(logit 계산, sigmoid)를 식으로 적어보세요.
2. BCE에서 정답이 $y = 1$이면 어느 항이 살아남고, 그 항이 예측 확률에 따라 어떻게 변하나요?
3. `predict_proba`의 출력 shape가 $(N, 2)$인 이유는? 두 열의 합은 항상 얼마여야 하나요?
4. 임계값을 0.5에서 0.7로 올리면 precision과 recall은 어떤 방향으로 움직이고, 그 이유는 무엇인가요?""")

# ----- 21. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) 왜 binary classification에서 MSE를 쓰면 안 되나요?

작동은 하지만 두 가지 문제가 있습니다.

1. **sigmoid + MSE는 비볼록(non-convex)**: 손실면이 매끈한 그릇 모양이 아니라 평탄한 구간이 생깁니다. local minima에 빠질 수 있고 학습이 불안정합니다. 반면 sigmoid + BCE는 볼록(convex)이라 전역 최적해로 수렴이 보장됩니다.

2. **gradient 소실(saturation)**: sigmoid 양 극단(출력이 0이나 1 근처)에서 도함수가 거의 0이 됩니다. MSE의 gradient는 그 도함수에 곱해지므로 함께 0으로 죽습니다. BCE는 sigmoid와 결합되었을 때 gradient가 단순히 $\hat p - y$로 떨어져 saturation이 사라집니다.

참고로 통계학에선 0/1 라벨을 그대로 `LinearRegression`으로 푸는 **Linear Probability Model(LPM)** 이 있긴 합니다. 단순 분석엔 쓰지만 출력이 [0, 1]을 안 지키고 BCE보다 학습 안정성이 떨어져 ML에선 거의 안 씁니다.

### Q2. (이론) sigmoid 대신 다른 활성화 함수(tanh, softmax)를 쓰면 어떻게 되나요?

- **tanh**: 출력 범위가 $[-1, 1]$입니다. 라벨을 -1/+1로 매핑하고 hinge loss 등을 쓰면 SVM과 비슷한 모델이 됩니다. 수학적으로는 $\tanh(z) = 2\sigma(2z) - 1$이라 sigmoid의 단순 변환이지만 라벨 컨벤션이 다릅니다.
- **softmax**: 다중 클래스용 일반화. binary에 softmax를 쓰려면 출력 차원을 2로 늘리고 라벨도 one-hot으로 바꿔야 합니다 → 이게 정확히 Ch 9에서 다룰 "방식 B"입니다 (방식 A=sigmoid 1차원, 방식 B=softmax 2차원이 수학적으로 동등).
- **ReLU/identity**: 출력이 [0, 1] 보장이 안 됩니다 — 음수도 1 초과도 가능 → BCE의 $\log$가 정의되지 않습니다.

### Q3. (실무) 클래스 불균형이 있으면 어떻게 하나요?

세 가지 접근이 있습니다.

1. **`class_weight='balanced'`**: sklearn에 한 줄로 적용. 소수 클래스의 손실에 더 큰 가중치.

```python
model = LogisticRegression(class_weight="balanced", max_iter=1000)
```

2. **임계값 조정**: 확률 분포가 한쪽으로 쏠려 있어 0.5 임계값이 의미 없을 때, 위 셀의 threshold sweep으로 F1이 최대인 점을 찾습니다.

3. **데이터 레벨 처리**: SMOTE(소수 클래스 합성), undersampling(다수 클래스 줄이기). `imbalanced-learn` 라이브러리.

### Q4. (실무) 임계값을 0.5 외에 다른 값으로 바꾸려면?

`predict_proba`로 확률을 얻은 뒤 직접 자르면 됩니다.

```python
proba_pos = model.predict_proba(X_test)[:, 1]
y_pred_custom = (proba_pos >= 0.3).astype(int)
```

최적 임계값은 보통 검증 데이터에서 F1이 최대가 되는 지점, 또는 ROC 곡선의 Youden's J 통계량(`tpr - fpr`이 최대인 지점)으로 잡습니다.

```python
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(y_test, proba_pos)
best_thr = thr[(tpr - fpr).argmax()]
print(f"Youden's J 기준 최적 임계값: {best_thr:.3f}")
```

### Q5. (이론) accuracy, precision, recall, F1 중 뭘 봐야 하나요?

데이터 분포와 도메인 비용에 따라 다릅니다.

- **accuracy** (전체 맞춘 비율): 클래스가 균형 잡혔을 때만 의미 있음. 95% 음성/5% 양성 데이터에서 모두 음성이라 찍어도 95%.
- **precision** (positive 예측 중 진짜 positive 비율): **오탐 비용** 이 클 때. 스팸 필터, 광고 추천.
- **recall** (실제 positive 중 잡아낸 비율): **놓치는 비용** 이 클 때. 암 진단, 사기 탐지.
- **F1** (precision·recall의 조화 평균): 둘 다 중요할 때의 균형 지표. 어느 한 쪽이 0이면 F1도 0.

이 챕터의 Yelp 이진화는 클래스 불균형이 크지 않아(긍정 ≈ 60%) accuracy도 의미 있지만, 실무에선 항상 classification_report 전체를 보는 습관이 안전합니다.

### Q6. (실무) sklearn `LogisticRegression`은 정규화(L2)가 기본인데 끄려면?

`penalty=None`(0.22 이상) 또는 `C` 값을 매우 크게 설정합니다.

```python
LogisticRegression(penalty=None, max_iter=1000)        # 정규화 없음
LogisticRegression(C=1e10, max_iter=1000)              # 거의 정규화 없음 (이전 버전 호환)
```

기본값은 `C=1.0`(L2 정규화)이고, 텍스트 분류처럼 feature가 많은 경우 정규화가 있어야 일반화 성능이 안정적입니다. 실무에선 거의 끄지 않습니다.""")

# ----- 22. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

다음 코드를 돌려보고 결과를 예측해보세요. 0/1 라벨에 `LinearRegression`을 그대로 적용하면 무슨 일이 일어날까요?

```python
from sklearn.linear_model import LinearRegression

lpm = LinearRegression()
lpm.fit(X_train, y_train)
pred_lpm = lpm.predict(X_test)

print(f"LPM 예측 범위: [{pred_lpm.min():.3f}, {pred_lpm.max():.3f}]")
print(f"0 미만 예측: {(pred_lpm < 0).sum()}개")
print(f"1 초과 예측: {(pred_lpm > 1).sum()}개")

# 임계값 0.5로 잘라서 정확도 비교
acc_lpm = accuracy_score(y_test, (pred_lpm >= 0.5).astype(int))
print(f"\nLPM accuracy (threshold 0.5): {acc_lpm:.4f}")
print(f"LogReg accuracy:               {accuracy_score(y_test, y_pred):.4f}")
```

힌트: LPM은 출력이 [0, 1]을 안 지키지만, 임계값 0.5로 자르면 분류 성능 자체는 LogReg와 비슷할 수도 있습니다. 다만 "출력이 확률"이라는 해석을 잃습니다.""")

# ----- 23. next -----
md(r"""## 다음 챕터 예고

**Chapter 4. sklearn Multi-class — sigmoid가 softmax로**

- 별점 1-5 를 5개 독립 클래스로 보고 `LogisticRegression()` 한 줄로 분류 (sklearn 이 multi-class 데이터에 자동으로 multinomial+softmax 적용)
- 출력이 1차원에서 **5차원** 으로 늘어나고, sigmoid 대신 **softmax** 가 붙음 (합 = 1 강제)
- Loss 는 BCE 에서 **`CrossEntropyLoss`** (sklearn: multinomial log loss) 로 일반화
- multinomial(softmax+CE) vs OvR(One-vs-Rest, K개 독립 binary) 비교 — OvR 이 Ch 6 multi-label 로 가는 다리""")


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
