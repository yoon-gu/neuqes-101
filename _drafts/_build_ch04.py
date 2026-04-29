"""Build 04_softmax_binary/04_softmax_binary.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "04_softmax_binary" / "04_softmax_binary.ipynb"

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
md(r"""# Chapter 4. Binary on softmax — CrossEntropy 등장 + sigmoid+BCE와의 동등성

**목표**: Ch 3과 **완전히 같은 binary 데이터** 를 출력 차원 2로 늘리고 softmax + CrossEntropy로 다시 풀어봅니다. 두 방식이 수학적으로 동등하다는 것을 식과 코드로 직접 확인합니다 — 이 직관은 Ch 10에서 BERT binary로 옮길 때 곧장 재활용됩니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분

---

## 학습 흐름

1. 🚀 **실습**: 같은 Yelp 이진화 데이터에 두 방식을 학습 — sigmoid+BCE(Ch 3 그대로) vs softmax+CE(이번 챕터)
2. 🔬 **해부**: $\sigma(z) = \text{softmax}([z_0, z_1])_1 = \sigma(z_1 - z_0)$ — 동등성을 식으로 보이고 코드로 검증
3. 🛠️ **변형**: 두 모델의 coefficient 자유도 차이 — softmax+2차원에는 잉여 자유도가 있다""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer` | Yelp 5,000 | — | — | — |
| 2 | LinearReg | TF-IDF | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | LogReg | TF-IDF | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| **4 ← 여기** | LogReg(multinomial) | TF-IDF | Yelp 이진화 (Ch 3과 동일) | **(2차원)** | **softmax** | **`CrossEntropyLoss`** |

전체 19챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 3)

| 축 | Ch 3 | Ch 4 |
|---|---|---|
| Output Head | (1차원) | **(2차원)** |
| Activation | sigmoid | **softmax** |
| Loss | `BCEWithLogitsLoss` | **`CrossEntropyLoss`** |
| 라벨 | int (0/1) | int (0/1) (그대로) |
| 데이터 | Yelp 이진화 | Yelp 이진화 (그대로) |
| 토크나이저 | TF-IDF | TF-IDF (그대로) |

**왜 같은 데이터에 같은 task인데 따로 챕터로 다루나?** 두 방식은 출력 차원·activation·loss가 모두 바뀌어 보이지만 *수학적으로 동등* 합니다. 이 동등성을 가장 단순한 sklearn 환경에서 미리 체험해두면, Ch 10에서 BERT binary가 두 방식 중 어느 쪽을 골라도 같다는 사실을 자연스럽게 받아들일 수 있습니다.

또 K=2를 통과하면 같은 식이 K=5(다음 챕터)로 자연스럽게 일반화됩니다 — softmax/CE는 K가 무엇이든 작동합니다.""")

# ----- 4. Loss 변화 -----
md(r"""## 📐 Loss 함수의 변화 — `CrossEntropyLoss` 등장

**Cross Entropy** 는 모델 예측 분포 $\hat{\mathbf{p}}$ 와 정답 one-hot $\mathbf{y}$ 사이의 차이를 잽니다.

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log \hat p_{ik}$$

원-핫 정답이라 정답 클래스 항만 살아남습니다.

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i,\, y_i}$$

**숫자로 감 잡기** (K=2, 정답 $y = 1$인 한 샘플):

| 정답 $y$ | 예측 분포 $[\hat p_0, \hat p_1]$ | 정답 확률 $\hat p_1$ | 손실 $-\log \hat p_1$ |
|---|---|---|---|
| 1 | `[0.1, 0.9]` | 0.9 | 0.105 |
| 1 | `[0.5, 0.5]` | 0.5 | 0.693 |
| 1 | `[0.9, 0.1]` | 0.1 | **2.303** |

**Ch 3 BCE 표와 비교해보면** 손실값이 *완전히 같습니다* (0.105 / 0.693 / 2.303). K=2에서 BCE와 CE가 동등하다는 첫 단서.

```python
# PyTorch (Ch 10 이후, 방식 B)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)   # logits: (N, K), targets: (N,) 정수 인덱스

# sklearn (이번 챕터)
LogisticRegression(multi_class="multinomial", max_iter=1000)
```""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

이번 챕터의 토크나이저는 **Ch 1-3과 동일한 `TfidfVectorizer`** 입니다. 모델·loss·데이터까지 Ch 3과 같고, 변하는 건 출력 차원과 활성화 함수뿐.

> **다음 챕터(Ch 5)**: 같은 TF-IDF, 같은 multinomial LogReg. 변하는 건 데이터가 binary에서 5클래스로, K가 2에서 5로.""")

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

plt.rcParams["axes.unicode_minus"] = False

dataset = load_dataset("yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
df["star"] = df["label"] + 1   # 0-4 → 1-5""")

# ----- 8. binary 데이터 -----
code(r"""# Ch 3과 동일한 이진화: 별점 3 제외, 4-5 → 1, 1-2 → 0
df_bin = df[df["star"] != 3].copy()
df_bin["y"] = (df_bin["star"] >= 4).astype(int)

X_text_train, X_text_test, y_train, y_test = train_test_split(
    df_bin["text"], df_bin["y"], test_size=0.2, random_state=42, stratify=df_bin["y"],
)

tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, 긍정 비율: {y_train.mean():.1%}")""")

# ----- 9. 두 방식 학습 -----
md(r"""## 🚀 실습: 두 방식을 나란히 학습

| 방식 | sklearn 인자 | 출력 차원 | 활성화 | loss |
|---|---|---|---|---|
| A (Ch 3 그대로) | `LogisticRegression()` | 1 | sigmoid | BCE |
| B (이번 챕터) | `LogisticRegression(multi_class="multinomial")` | 2 | softmax | CE |""")

# ----- 10. fit -----
code(r"""model_a = LogisticRegression(max_iter=1000)                           # 방식 A
model_a.fit(X_train, y_train)

model_b = LogisticRegression(multi_class="multinomial", max_iter=1000)  # 방식 B
model_b.fit(X_train, y_train)

acc_a = accuracy_score(y_test, model_a.predict(X_test))
acc_b = accuracy_score(y_test, model_b.predict(X_test))

print(f"방식 A (sigmoid + BCE)  accuracy: {acc_a:.4f}")
print(f"방식 B (softmax + CE)   accuracy: {acc_b:.4f}")
print(f"차이: {abs(acc_a - acc_b):.4f}")""")

# ----- 11. predict_proba 비교 -----
code(r"""proba_a = model_a.predict_proba(X_test)   # (N, 2)
proba_b = model_b.predict_proba(X_test)   # (N, 2)

print(f"방식 A predict_proba shape: {proba_a.shape}")
print(f"방식 B predict_proba shape: {proba_b.shape}")
print(f"  (방식 A도 sklearn이 내부적으로 [P(0), P(1)]을 내줌 — sigmoid 결과를 두 열로 펼친 것)")

p_a, p_b = proba_a[:, 1], proba_b[:, 1]
print(f"\n앞 5개 P(y=1):")
print(f"방식 A: {p_a[:5].round(4)}")
print(f"방식 B: {p_b[:5].round(4)}")
print(f"\n전체 max 차이: {np.abs(p_a - p_b).max():.4f}")
print(f"전체 mean 차이: {np.abs(p_a - p_b).mean():.4f}")""")

# ----- 12. 동등성 도출 -----
md(r"""## 🔬 해부: 수학적 동등성

방식 B는 *개념적으로* logit을 두 개 ($z_0, z_1$) 학습합니다. softmax의 두 번째 성분을 풀어보면:

$$\text{softmax}([z_0, z_1])_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \frac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)$$

즉 두 logit에서 **의미 있는 정보는 $z_1 - z_0$ 뿐** 입니다 — 두 logit에 같은 상수를 더해도 softmax 결과가 안 바뀌니까요 ($e^{z+c}/\sum e^{z+c} = e^{z}/\sum e^{z}$). softmax+2는 sigmoid+1의 **리파라미터화** 일 뿐입니다.

CE 쪽도 K=2에서 BCE와 같은 식이 됩니다 (one-hot이라 $y_1 = y$, $y_0 = 1-y$ 대입):

$$\text{CE} = -[y_1 \log \hat p_1 + y_0 \log \hat p_0] = -[y \log \hat p_1 + (1-y)\log(1 - \hat p_1)] = \text{BCE}$$

확률·loss가 같으니 학습된 결정 경계도, gradient도 같습니다.

먼저 식이 정말 일치하는지 임의의 logit 쌍으로 직접 확인합니다.""")

# ----- 13. 임의 logit으로 동등성 시연 -----
code(r"""# 임의의 logit 쌍 4개를 만들어 softmax([z0,z1])_1 == sigmoid(z1 - z0) 인지 확인
z0_arr = np.array([-2.0, 0.0, 1.5, 3.0])
z1_arr = np.array([ 1.0, 0.5, -0.5, 2.0])

softmax_p1  = np.exp(z1_arr) / (np.exp(z0_arr) + np.exp(z1_arr))
sigmoid_diff = 1.0 / (1.0 + np.exp(-(z1_arr - z0_arr)))

print(f"{'z_0':>6} {'z_1':>6}    {'softmax([z0,z1])_1':>22}    {'sigmoid(z1-z0)':>16}")
print("-" * 60)
for i in range(len(z0_arr)):
    print(f"{z0_arr[i]:>6.1f} {z1_arr[i]:>6.1f}    {softmax_p1[i]:>22.8f}    {sigmoid_diff[i]:>16.8f}")

print(f"\nmax 차이: {np.abs(softmax_p1 - sigmoid_diff).max():.2e}  (수치적 오차 수준)")""")

# ----- 14. 변형: sklearn coef shape 관찰 -----
md(r"""## 🛠️ 변형: sklearn은 왜 K=2 multinomial에서 `(2, V)` coef를 안 만드나?

위 동등성 덕분에 K=2에서 두 logit 중 하나는 잉여입니다. sklearn은 이 사실을 알고 **K=2 multinomial을 자동으로 binary form으로 collapse** 시킵니다 — `coef_` 를 `(2, V)` 가 아니라 `(1, V)` 로만 저장합니다. 두 방식이 그래서 사실상 같은 모델이 되어 `predict_proba`도 거의 일치하는 거였죠.

직접 두 모델의 `coef_` 모양을 확인합니다.""")

# ----- 15. coef shape 비교 코드 -----
code(r"""print(f"방식 A coef_ shape:      {model_a.coef_.shape}")
print(f"방식 B coef_ shape:      {model_b.coef_.shape}")
print(f"방식 A intercept_ shape: {model_a.intercept_.shape}")
print(f"방식 B intercept_ shape: {model_b.intercept_.shape}")
print()
print("→ 둘 다 (1, V) — sklearn이 K=2 multinomial을 binary form으로 collapse")
print()
print(f"두 coef_ 의 max 차이:      {np.abs(model_a.coef_ - model_b.coef_).max():.2e}")
print(f"두 intercept_ 의 max 차이: {np.abs(model_a.intercept_ - model_b.intercept_).max():.2e}")
print()
print("(미세한 차이는 solver 수렴 기준의 차이일 뿐, 본질적으로 같은 모델)")
print()
print("진짜 (2, V) 두 logit head는 PyTorch가 '직접' 만들어주는 Ch 10 BERT binary에서 등장합니다.")""")

# ----- 16. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `LogisticRegression(multi_class="multinomial")` | softmax + CE 다중 분류 (이번 챕터엔 K=2) | Ch 5에서 K=5로 확장, Ch 11 BERT multi-class에서 같은 패러다임 |
| `sklearn.metrics.log_loss` | CE/BCE 평가 함수 (multi-class 호환) | — |""")

# ----- 17. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. softmax 함수 정의를 적고, $\text{softmax}([z_0, z_1])_1$ 이 $\sigma(z_1 - z_0)$ 와 같음을 증명해보세요.
2. Cross Entropy를 K=2에 적용하면 정확히 BCE가 되는 과정을 식으로 보일 수 있나요? ($y_1 = y$, $y_0 = 1-y$ 대입)
3. 방식 B의 두 coefficient 벡터 사이에 어떤 관계가 학습되는 경향이 있나요? 그 이유는?
4. 같은 binary 데이터에 두 방식의 accuracy가 거의 같다면, 실무에서 어느 쪽을 택해야 하나요?""")

# ----- 18. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) K=2일 때 sigmoid+BCE와 softmax+CE가 정확히 동등하다는 걸 식으로 어떻게 보이나요?

두 가지를 따로 보여야 합니다.

**확률 동등성:**

$$\text{softmax}([z_0, z_1])_1 = \frac{e^{z_1}}{e^{z_0}+e^{z_1}} = \frac{1}{1+e^{-(z_1-z_0)}} = \sigma(z_1 - z_0)$$

방식 A의 logit을 $z = z_1 - z_0$ 로 두면 두 모델의 P(y=1)이 일치합니다.

**Loss 동등성** ($K=2$, one-hot 정답에서 $y_1 = y$, $y_0 = 1-y$):

$$\text{CE} = -\sum_{k=0}^{1} y_k \log \hat p_k = -[y \log \hat p_1 + (1-y) \log \hat p_0] = -[y \log \hat p_1 + (1-y) \log(1 - \hat p_1)] = \text{BCE}$$

확률과 loss가 모두 같으니 학습된 결정 경계도 같고, gradient도 같습니다.

### Q2. (실무) 실제로 둘 중 어느 방식이 더 널리 쓰이나요?

두 가지 관행이 공존합니다.

- **sigmoid+BCE (방식 A, num_labels=1)**: sklearn 기본, 통계학·의학 분야 표준. 출력 1개라 "확률 하나"라는 해석이 단순.
- **softmax+CE (방식 B, num_labels=2)**: BERT/PyTorch 기본, 딥러닝 표준. 다중 클래스로 일반화하기 자연스럽고 라이브러리 코드가 단순(같은 헤드/같은 loss로 K가 2이든 N이든 호환).

이 커리큘럼의 BERT 챕터(Ch 9-13)는 방식 B가 기본이라 이번 챕터에서 미리 익숙해지는 게 의미 있습니다. Ch 10에서 두 방식을 BERT로 다시 비교합니다.

### Q3. (이론) softmax 합=1 제약은 어디서 오나요?

수식 자체가 정규화를 강제합니다.

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

분모가 모든 클래스 $e^{z_j}$의 합이라 분자들이 그 합을 정확히 분할 — 합이 1.

**왜 이런 구조?** 모델 출력을 "확률 분포"로 해석하고 싶어서입니다. 확률 분포는 정의상 $\sum_k p_k = 1$ 이고 각 항이 [0, 1]. softmax는 임의 실수 logit 벡터를 그런 분포로 보내는 가장 자연스러운 변환 중 하나(지수 함수 = 단조증가 + 양수 보장 → 정규화).

### Q4. (실무) sklearn에서 binary에 `multi_class="multinomial"` 을 줘도 `coef_.shape`가 `(1, V)` 인 이유는?

위 본문에서 확인했듯 K=2 softmax는 두 logit 중 하나가 잉여(redundant)입니다 — $z_1 - z_0$ 만 의미가 있어요. sklearn은 이걸 알고 **K=2 multinomial을 자동으로 binary form으로 collapse** 시킵니다. 그래서 `coef_` 가 `(2, V)` 가 아니라 `(1, V)`, `intercept_` 도 `(1,)` 로 저장됩니다.

```python
LogisticRegression(multi_class="multinomial").fit(X, y_binary).coef_.shape  # (1, V)
LogisticRegression(multi_class="multinomial").fit(X, y_3class).coef_.shape  # (3, V) — K≥3에선 (K, V)
```

그래서 방식 A와 방식 B가 sklearn 안에서는 사실상 같은 모델이고, predict_proba도 미세한 수치 오차 빼고 일치합니다. 진짜 *두 별개의 logit head* 가 살아 있는 형태는 프레임워크가 collapse하지 않는 환경 — PyTorch에서 `nn.Linear(H, 2)` 를 직접 만들 때 — 비로소 등장합니다 (Ch 10).

### Q5. (이론) sklearn `multi_class` 인자의 의미와 자동 선택 기준은?

- `"multinomial"`: 모든 클래스를 한 번에 학습 (softmax). 이번 챕터.
- `"ovr"` (One-vs-Rest): K개 독립 binary 분류기. Ch 6 multi-label로 가는 다리.
- `"auto"` (기본값): solver와 데이터에 맞춰 자동 — 클래스 2개면 binary, 3개 이상이면 multinomial(`lbfgs`, `newton-cg`, `sag`, `saga` solver) 또는 OvR(`liblinear`).

sklearn 1.5부터 `multi_class` 자체가 deprecated이고 multinomial이 기본 동작이 됩니다. 호환성을 위해 명시할 때만 `multi_class="multinomial"` 처럼 적습니다.

### Q6. (실무) Hugging Face `Trainer`도 두 방식이 가능한가요?

가능하고, `AutoModelForSequenceClassification` 인자 한 줄 차이입니다.

```python
# 방식 A: num_labels=1, BCE 자동 적용
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1,
    problem_type="single_label_classification",  # 또는 자동
)

# 방식 B: num_labels=2, CE 자동 적용 (BERT 표준)
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)
```

이 커리큘럼의 Ch 10이 두 방식을 BERT로 다시 비교하는 챕터입니다. 이번 챕터에서 익힌 동등성이 그 비교의 출발점이 됩니다.""")

# ----- 19. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

방식 A와 B의 정답 분포 자체는 같았는데, 정규화(`C`) 강도를 줄이면 결과가 갈릴까요?

```python
for C in [100, 1, 0.01]:
    a = LogisticRegression(C=C, max_iter=2000).fit(X_train, y_train)
    b = LogisticRegression(C=C, multi_class="multinomial", max_iter=2000).fit(X_train, y_train)
    pa = a.predict_proba(X_test)[:, 1]
    pb = b.predict_proba(X_test)[:, 1]
    print(f"C={C}: max |P_a - P_b| = {np.abs(pa - pb).max():.4f}")
```

힌트: 정규화가 매우 약해지면 (`C` 크게) 두 방식의 잉여 자유도가 다른 식으로 정해질 수 있어 차이가 약간 늘어날 수 있습니다. 하지만 결정 경계(predict 결과)는 거의 같게 유지됩니다.""")

# ----- 20. next -----
md(r"""## 다음 챕터 예고

**Chapter 5. sklearn Multi-class — K=5로 진짜 일반화**

- 같은 multinomial LogReg를 별점 1-5(5클래스)로 그대로 확장
- 수식·코드 변화 거의 없음 — softmax/CE는 K가 무엇이든 같은 형태
- 5×5 confusion matrix가 대각선 근처에 몰리는 ordinal 흔적 관찰
- multinomial vs OvR 비교 (Ch 6 multi-label로 가는 다리)""")


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
