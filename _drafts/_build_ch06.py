"""Build 06_sklearn_multilabel/06_sklearn_multilabel.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "06_sklearn_multilabel" / "06_sklearn_multilabel.ipynb"

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
md(r"""# Chapter 6. sklearn Multi-label — softmax 합=1 제약을 푼다

**목표**: 한 샘플에 *여러* 라벨이 동시에 붙는 multi-label 문제로 확장합니다. softmax의 클래스 *상호배타* 가정이 깨지고, K개 sigmoid가 라벨마다 독립적으로 작동합니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분

---

## 학습 흐름

1. 🚀 **실습**: Yelp 리뷰에 측면(aspect) 키워드를 매칭해 5개 라벨(food/service/price/ambiance/location) multi-hot 합성 → `OneVsRestClassifier`로 학습
2. 📐 **Loss 분해**: 학습된 모델의 실제 예측으로 BCE 5개를 직접 합산해 본다 — multinomial CE를 못 쓰는 이유를 숫자로
3. 🔬 **해부**: multi-label 평가 지표 — subset accuracy, hamming loss, micro/macro F1
4. 🛠️ **변형**: 임계값(threshold)을 옮기면 micro/macro F1이 어떻게 움직이나
5. ⚠️ **합성의 한계** — 키워드 매칭으로 만든 라벨이 실제 라벨링과 어떻게 다른지 솔직히 짚기""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer` | Yelp 5,000 | — | — | — |
| 2 | LinearReg | TF-IDF | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | LogReg | TF-IDF | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | LogReg(multinomial) | TF-IDF | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` |
| 5 | LogReg(multinomial) | TF-IDF | Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| **6 ← 여기** | OneVsRest LogReg | TF-IDF | Yelp + 측면 키워드 합성 | (5차원) | **sigmoid (각각 독립)** | **`BCEWithLogitsLoss` per-label** |

전체 19챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 5)

가장 큰 변화는 **데이터 가정** 이고, 그게 모델 선택을 강제합니다.

| 축 | Ch 5 (multi-class) | Ch 6 (multi-label) |
|---|---|---|
| 데이터 가정 | 클래스 *상호배타* (한 샘플 = 한 라벨) | **라벨 *독립* (한 샘플에 여러 라벨 가능)** |
| 라벨 형식 | int 한 개 (0-4) | **multi-hot 벡터** (예: `[1, 0, 1, 0, 1]`) |
| 모델 패러다임 | **multinomial 기본** + OvR 대안 (Ch 5 후반에서 두 방식 비교) | **OvR 만** (multinomial은 데이터 가정과 충돌) |
| Activation | softmax 한 번 (합 = 1 강제) | **per-label sigmoid** (라벨끼리 독립) |
| Loss | `CrossEntropyLoss` | **per-label `BCEWithLogitsLoss` 평균** |
| OvR 사용 방식 | K개 sigmoid → **argmax로 한 라벨 선택** | K개 sigmoid **그대로** (argmax 없음, 각자 임계값 0.5와 비교) |
| 데이터 | 별점 5클래스 | **Yelp + 측면 키워드 합성** (5개 측면) |
| 토크나이저 | TF-IDF | TF-IDF (그대로) |

### 왜 OvR이 multi-label의 자연스러운 선택인가

Ch 5에선 두 패러다임이 모두 가능했습니다.

- **multinomial**: softmax 한 번으로 K개 logit을 묶어 합=1 강제. "K개 중 정확히 하나"라는 데이터 가정과 정합.
- **OvR (대안)**: K개 *독립* sigmoid + argmax 후처리. 각 binary 모델은 독립이지만 마지막에 강제로 하나만 고르므로 결과는 상호배타.

Ch 6의 multi-label은 이 가정 자체를 깹니다 — 한 리뷰가 "음식 + 서비스 + 가격"을 동시에 다룰 수 있어요. 그러면:

- **multinomial은 부적합**: softmax가 합=1을 강제하므로 'food=0.9, service=0.85 동시 활성' 같은 분포를 *표현할 수가 없습니다*. P(food)=0.9면 나머지 합이 0.1로 강제돼 동시 활성이 수학적으로 불가능.
- **OvR은 자연스럽게 들어맞음**: K개 sigmoid가 *각자* 0/1을 결정 → 어떤 조합이든 표현 가능. argmax 후처리 단계만 빼면 그대로 multi-label.

요약: Ch 5에서 *대안* 이었던 OvR이 Ch 6에서는 *유일한 자연스러운 선택* 이 됩니다. 알고리즘(`OneVsRestClassifier`)은 그대로, 사용 방식만 "argmax로 한 라벨 선택" → "K개 출력 그대로" 로 바뀝니다.""")

# ----- 4. Loss 변화 -----
md(r"""## 📐 Loss 함수의 변화 — `BCEWithLogitsLoss` per-label

K개 라벨에 대해 BCE를 각각 계산하고 평균을 냅니다.

$$L = \frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K}\bigl[-y_{ik}\log \hat p_{ik} - (1 - y_{ik})\log(1 - \hat p_{ik})\bigr]$$

각 (샘플, 라벨) 쌍이 독립적으로 손실에 기여합니다 — 한 라벨에서 틀렸다고 다른 라벨의 손실이 변하지 않습니다. CE의 클래스 경쟁 구조와 정반대.

**숫자로 감 잡기** (K=5, 정답 multi-hot $\mathbf{y} = [1, 0, 1, 0, 1]$):

| 시나리오 | 예측 확률 $\hat{\mathbf{p}}$ | 라벨별 손실 | 평균 BCE |
|---|---|---|---|
| 잘 맞춤 | `[0.9, 0.1, 0.8, 0.2, 0.6]` | 0.105 / 0.105 / 0.223 / 0.223 / 0.511 | **0.233** |
| 균등 (baseline) | `[0.5, 0.5, 0.5, 0.5, 0.5]` | 0.693 × 5 | **0.693** |
| 정반대로 자신감 | `[0.1, 0.9, 0.1, 0.9, 0.1]` | 2.303 × 5 | **2.303** |

baseline = $\log 2 = 0.693$ — 모든 라벨에 0.5를 줄 때 (BCE에서 K=2 분포의 균등 추측과 같은 값). 학습된 모델은 이 값보다 작아야 정상.

```python
# PyTorch (Ch 12 이후, multi-label)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets.float())   # logits: (N, K), targets: (N, K) multi-hot

# sklearn (이번 챕터)
from sklearn.multiclass import OneVsRestClassifier
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X, Y_multilabel)   # Y_multilabel shape: (N, K) 0/1
```""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

이번 챕터의 토크나이저는 **Ch 1-5와 동일한 `TfidfVectorizer`**. 입력 표현은 그대로고, 변화는 라벨 구조와 출력 헤드의 형태에 있습니다.

> **다음 챕터(Ch 7)** — Phase 1 시작: 사전학습된 **DistilBERT WordPiece** 가 처음 등장합니다. TF-IDF의 단어 단위 어휘 학습과 어떻게 다른지 비교 시작.""")

# ----- 6. install -----
code(r"""!pip install -q datasets scikit-learn pandas matplotlib""")

# ----- 7. import + load -----
code(r"""import warnings
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

dataset = load_dataset("yelp_review_full")
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()
print(f"전체 샘플 수: {len(df)}")""")

# ----- 8. 합성 도입 -----
md(r"""## 🚀 실습 1: 측면 라벨 합성

Yelp 데이터에는 multi-label 정답이 없으므로 **5개 측면(aspect)** 별 키워드 사전을 만들어 매칭합니다.

| 측면 | 의미 | 키워드 예시 |
|---|---|---|
| `food` | 음식의 맛/메뉴 | food, meal, dish, taste, delicious, ... |
| `service` | 서비스/응대 | service, staff, waiter, friendly, rude, ... |
| `price` | 가격/가성비 | price, cheap, expensive, value, worth, ... |
| `ambiance` | 분위기/인테리어 | atmosphere, decor, music, vibe, cozy, ... |
| `location` | 위치/주차 | location, parking, area, neighborhood, ... |

각 리뷰 텍스트에 대해 *어떤 키워드라도* 등장하면 해당 측면을 1로 활성화 — 5차원 multi-hot 벡터가 됩니다. **이 합성의 한계** 는 챕터 끝에서 솔직히 짚습니다.""")

# ----- 9. 합성 코드 -----
code(r"""ASPECT_KEYWORDS = {
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
print(f"Y shape: {Y.shape}  (샘플 수, 측면 수)")
print(f"앞 3개 샘플의 multi-hot 라벨:\n{Y[:3]}")""")

# ----- 10. 합성 통계 -----
code(r"""print("측면별 활성 비율 (전체 5,000건 기준):")
for k, aspect in enumerate(ASPECTS):
    print(f"  {aspect:>9}: {Y[:, k].mean():.1%}  ({Y[:, k].sum()}건)")

n_labels_per_sample = Y.sum(axis=1)
print(f"\n샘플당 평균 활성 라벨 수: {n_labels_per_sample.mean():.2f}")
print(f"활성 라벨 분포:")
for n in range(K + 1):
    count = (n_labels_per_sample == n).sum()
    print(f"  {n}개: {count}건  ({count/len(Y):.1%})")""")

# ----- 11. TF-IDF + split -----
code(r"""X_text_train, X_text_test, Y_train, Y_test = train_test_split(
    df["text"], Y, test_size=0.2, random_state=42,
)

tfidf = TfidfVectorizer(max_features=10000)
X_train = tfidf.fit_transform(X_text_train)
X_test = tfidf.transform(X_text_test)

print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"X_test:  {X_test.shape}, Y_test:  {Y_test.shape}")""")

# ----- 12. 학습 도입 -----
md(r"""## 🚀 실습 2: `OneVsRestClassifier` (이번엔 argmax 없이)

Ch 5에서 multi-class의 *대안* 으로 본 `OneVsRestClassifier`가 이번엔 *기본 도구* 입니다. 코드 차이는 단 하나 — **`Y_train` shape이 `(N,)` → `(N, K)`** 로 바뀌는 것. sklearn이 multi-hot Y를 보고 multi-label 모드로 자동 전환해 K개 binary 분류기가 독립적으로 학습됩니다.

가장 중요한 사용 방식 변화는 **argmax가 사라진다는 것**.

- **Ch 5 OvR (multi-class)**: 5개 sigmoid 출력 중 가장 큰 것 *하나만* 정답으로 골랐음.
- **Ch 6 OvR (multi-label)**: 5개 sigmoid 출력을 *각자* 임계값 0.5와 비교해 0/1 결정. 동시 활성 가능.

`predict_proba`도 차이가 있습니다 — Ch 5에서는 sklearn이 후처리로 합=1로 정규화해줬지만, multi-label 모드에서는 *정규화하지 않고* 각 라벨의 P(label_k = 1)을 그대로 반환합니다 (각 라벨이 독립이니까).""")

# ----- 13a. bare LogReg 시도 -----
code(r"""# 먼저 wrapper 없이 그냥 LogisticRegression() 에 multi-hot Y를 넣어보기
bare_model = LogisticRegression(max_iter=1000)
try:
    bare_model.fit(X_train, Y_train)
    print(f"성공? coef_ shape: {bare_model.coef_.shape}")
except ValueError as e:
    print(f"❌ 실패: {type(e).__name__}")
    print(f"   메시지: {e}")""")

# ----- 13b. 해설 -----
md(r"""**왜 실패했나?** sklearn `LogisticRegression` 은 *1D Y* (각 샘플당 한 클래스 인덱스)만 받습니다. 우리의 `Y_train.shape == (N, 5)` 는 "한 샘플에 여러 라벨"이라는 의미인데 *형식 자체* 가 호환되지 않아요. fit이 첫 줄에서 죽습니다.

`OneVsRestClassifier` 의 역할은 단순합니다.

1. 2D Y를 K개의 1D 컬럼으로 **쪼개고**,
2. 각 컬럼마다 `LogisticRegression` 을 **별도로** 학습 (총 K개 모델),
3. 결과를 `.estimators_` 리스트에 보관해 `predict` 시 모두 적용.

알고리즘은 동일한 LogReg지만 **fit 시점의 Y 형식 처리** 가 결정적 차이입니다 — bare 호출은 죽고, wrapper 호출은 K개 모델로 분할 학습됩니다.""")

# ----- 13c. OvR 학습 -----
code(r"""# 위 실패와 대비: wrapper 한 줄로 K개 LogReg가 자동 분할 학습됨
model_ml = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_ml.fit(X_train, Y_train)

print(f"OvR fit 성공!")
print(f"  학습된 binary 분류기 수: {len(model_ml.estimators_)}")
print(f"  각 estimator 타입:       {type(model_ml.estimators_[0]).__name__}")
print(f"  각 estimator coef shape: {model_ml.estimators_[0].coef_.shape}  (1, V — 한 라벨용 binary)")
print(f"\n각 분류기가 학습한 라벨:")
for k, aspect in enumerate(ASPECTS):
    n_pos = Y_train[:, k].sum()
    print(f"  estimator[{k}] = '{aspect}': positive {n_pos}건 ({n_pos/len(Y_train):.1%})")

# 예측 + 확률
Y_pred = model_ml.predict(X_test)         # (N, K) multi-hot 0/1 (threshold 0.5 자동 적용)
proba_ml = model_ml.predict_proba(X_test) # (N, K) per-label probability (정규화 X)

print(f"\nY_pred shape: {Y_pred.shape}")
print(f"proba shape:  {proba_ml.shape}")
print(f"\n앞 3개 샘플의 예측 확률 (per-label):")
print(pd.DataFrame(proba_ml[:3], columns=ASPECTS).round(4))""")

# ----- 14a. Loss 분해 도입 -----
md(r"""## 📐 Loss 한 단계 더: 학습된 모델의 실제 예측으로 BCE 분해

방금 fit한 `model_ml`이 한 샘플에 대해 어떤 손실을 만들어내는지 직접 분해합니다. 변경점 표에서 본 "**per-label BCE 평균**" 이 단순한 수식이 아니라 **실제 5개 숫자의 산수** 라는 걸 확인합니다 — 그리고 그 위에서 "왜 multinomial CE는 여기 못 쓰나"를 진짜 값으로 짚습니다.""")

# ----- 14b. Loss 분해 코드 -----
code(r"""# 여러 라벨이 활성된 test 샘플 하나 고르기 (분해가 잘 보이도록)
multi_active = np.where(Y_test.sum(axis=1) >= 3)[0]
sample_idx = int(multi_active[0]) if len(multi_active) > 0 else 0

y_true = Y_test[sample_idx]
p_pred = proba_ml[sample_idx]
text = X_text_test.iloc[sample_idx]

print(f"리뷰 (앞 200자): {text[:200]}...")
print(f"활성 라벨 수: {y_true.sum()}\n")

print(f"{'측면':>10}  {'정답 y':>6}  {'예측 p':>10}  {'기여 항':>20}  {'손실':>10}")
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
print(f"{'합':>10}  {'':>6}  {'':>10}  {'':>20}  {total_loss:>10.4f}")
print(f"{'평균 BCE':>10}  {'':>6}  {'':>10}  {'÷ 5':>20}  {total_loss/5:>10.4f}")""")

# ----- 14c. Loss 분해 해석 + multinomial 못 쓰는 이유 -----
md(r"""**관찰**

- 5개 라벨이 *각자 독립적으로* 손실을 기여합니다 — 한 라벨에서 잘 맞춰도 다른 라벨에서 못 맞추면 그 영향이 그대로 평균에 더해집니다.
- 정답이 1인 라벨은 $-\log(p)$ — 예측 확률이 1에 가까울수록 손실 0에 수렴.
- 정답이 0인 라벨은 $-\log(1-p)$ — 예측 확률이 0에 가까울수록 손실 0에 수렴.
- 같은 sigmoid 출력에 대해 정답이 0이냐 1이냐에 따라 **정반대 방향** 으로 페널티가 커집니다 (대칭 구조).

### 같은 샘플을 multinomial CE로 풀려고 하면

위 샘플의 정답은 multi-hot — 여러 라벨이 동시에 1입니다. 만약 multinomial CE를 *억지로* 적용하려면 다음 두 가지 *임의 결정* 이 필요합니다.

1. **5개 활성 라벨 중 *하나만* 정답으로 골라야 함**: argmax? 첫 활성? 어느 기준이든 *임의*.
2. **그러면 나머지 활성 라벨들은 *틀린* 클래스로 학습됨**: 모델이 그 라벨에 강한 확률을 줄수록 손실이 *커짐*.

결과: 모델이 "동시 활성 패턴을 *피하려고*" 학습됩니다 — 실제 정답에서는 동시 활성이 정답인데도. 이건 *데이터 가정과 정반대 방향* 으로 학습 신호가 작동하는 셈입니다.

per-label BCE는 위 표처럼 5개 손실을 *독립적으로* 합산하므로 각 라벨이 자기 정답에만 책임을 집니다. **multi-label 데이터의 본래 구조와 정합한 유일한 선택** 인 이유가 이 산수에 있습니다.""")

# ----- 15. 해부 도입 -----
md(r"""## 🔬 해부: multi-label 평가 지표

multi-class에서 자주 쓰던 accuracy는 multi-label에서 의미가 미묘하게 다릅니다.

- **subset accuracy** (`accuracy_score`): "K개 라벨이 *전부* 일치한 샘플 비율" — 라벨 하나만 틀려도 0점. 매우 엄격.
- **hamming loss**: 평균 라벨별 오답 비율. 5개 중 1개 틀리면 0.2 기여. 가장 직관적.
- **micro F1**: 모든 라벨의 TP/FP/FN를 한 풀에 모아 계산 — 빈도 큰 라벨이 영향력 큼.
- **macro F1**: 라벨별 F1을 단순 평균 — 모든 라벨 동등 가중.""")

# ----- 15. 평가 지표 코드 -----
code(r"""print(f"Subset accuracy (모두 일치): {accuracy_score(Y_test, Y_pred):.4f}")
print(f"Hamming loss (라벨별 평균 오답): {hamming_loss(Y_test, Y_pred):.4f}")
print(f"micro F1: {f1_score(Y_test, Y_pred, average='micro', zero_division=0):.4f}")
print(f"macro F1: {f1_score(Y_test, Y_pred, average='macro', zero_division=0):.4f}")""")

# ----- 16. classification_report -----
code(r"""# 측면별 precision/recall/F1
print(classification_report(
    Y_test, Y_pred,
    target_names=ASPECTS,
    zero_division=0,
))""")

# ----- 17. 변형 도입 -----
md(r"""## 🛠️ 변형: 라벨별 임계값(threshold) 변경

multi-label에서는 K개 sigmoid 출력에 대해 임계값 0.5로 자르는 게 기본입니다. 이 임계값을 옮기면 모든 라벨이 함께 반응합니다 — 임계값을 낮추면 더 많은 라벨이 활성되어 recall은 오르고 precision은 내려갑니다.

(고급 트릭: 라벨마다 *별도* 임계값을 정해 검증 F1을 최대화할 수도 있음. 여기서는 하나로 통일.)""")

# ----- 18. threshold sweep -----
code(r"""thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
rows = []
for t in thresholds:
    Y_pred_t = (proba_ml >= t).astype(int)
    rows.append({
        "threshold": t,
        "subset_acc": accuracy_score(Y_test, Y_pred_t),
        "hamming": hamming_loss(Y_test, Y_pred_t),
        "micro_F1": f1_score(Y_test, Y_pred_t, average="micro", zero_division=0),
        "macro_F1": f1_score(Y_test, Y_pred_t, average="macro", zero_division=0),
    })
df_t = pd.DataFrame(rows).round(4)
print(df_t.to_string(index=False))""")

# ----- 19. 합성의 한계 -----
md(r"""## ⚠️ 합성의 한계 — 솔직한 한계 짚기

이 챕터의 학습 결과가 너무 좋아 보일 수 있습니다 (subset accuracy, micro F1 모두 매우 높음). 그 이유는 **모델이 *키워드 매칭 규칙* 자체를 학습** 하기 때문입니다 — 우리가 정한 사전을 다시 거꾸로 풀어내고 있을 뿐, 진짜 측면 추출 능력을 입증한 게 아닙니다.

실제 multi-label 문제에서 부딪히는 것들:

1. **부정·반어 무시** — `"this place is not noisy at all"` 의 'noisy'를 ambiance 활성으로 잡는 게 우리 사전의 한계. 사람이 읽으면 ambiance가 *아닌* 데도.
2. **사전 협소** — 'food'에 'sushi', 'ramen', 'pasta' 같은 구체 음식명이 빠져 있으면 그 리뷰는 food=0이 되어 버림.
3. **정답이 노이지** — 우리 라벨 자체가 진짜 정답이 아닌 휴리스틱이라, 모델 성능을 이 정답에 비교하는 건 결국 "모델이 휴리스틱을 얼마나 따라 했나"를 잴 뿐.
4. **빈 라벨**: 모든 측면이 0인 샘플도 있음 (`{n_labels_per_sample == 0).sum()` 건). 실제 multi-label 데이터에선 보통 최소 한 라벨은 보장.

**그럼 왜 합성을 쓰나?** — 학습 코드의 *형태* 와 *평가 지표 해석* 을 익히는 게 이 챕터의 목적이기 때문입니다. Ch 12 BERT multi-label에서 **같은 합성 라벨** 을 그대로 사용하므로 비교가 깔끔하게 됩니다. 진짜 multi-label 데이터(예: GoEmotions, Reuters)는 라벨이 사람 손으로 만들어져 있어 비용이 큽니다.""")

# ----- 20. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `sklearn.multiclass.OneVsRestClassifier` | K개 독립 binary 분류기를 묶고 multi-label 모드 자동 감지 | Ch 12 BERT multi-label에서 같은 패러다임을 BERT로 |
| `sklearn.metrics.hamming_loss` | 라벨별 평균 오답 비율 | Ch 12에서도 평가 지표로 |
| `sklearn.metrics.f1_score(average="micro" / "macro")` | multi-label F1 집계 방식 | — |
| `sklearn.metrics.classification_report` | 라벨별 precision/recall/F1 한 번에 | — |""")

# ----- 21. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. multi-class와 multi-label은 라벨 구조와 활성화 함수에서 어떻게 다른가요?
2. multi-label에서 subset accuracy가 너무 엄격한 지표가 되는 이유는?
3. multi-label baseline BCE는 왜 $\log 2 = 0.693$ 인가요?
4. 임계값을 0.5에서 0.3으로 낮추면 micro F1과 macro F1은 어느 방향으로 움직이는 게 일반적이고, 그 이유는 무엇인가요?""")

# ----- 22. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) Multi-class와 Multi-label은 정확히 어떻게 다른가요?

| 항목 | Multi-class | Multi-label |
|---|---|---|
| 한 샘플의 라벨 수 | 정확히 1개 | 0개 이상 (K개 가능) |
| 라벨 구조 | 정수 인덱스 (예: 3) | multi-hot 벡터 (예: [1, 0, 1, 0, 1]) |
| 활성화 | softmax (합 = 1) | per-label sigmoid (라벨끼리 독립) |
| Loss | CrossEntropy | per-label BCE 평균 |
| 가정 | 클래스 *상호배타* | 라벨 *독립* |
| 예시 | 별점 1-5 중 하나, 뉴스 7카테고리 | 영화 장르(로맨스+코미디), 영화 태그 |

### Q2. (이론) micro F1과 macro F1 중 뭘 봐야 하나요?

데이터 분포에 따라 다릅니다.

- **micro F1**: 라벨 빈도와 무관하게 *전체 예측 풀* 의 정확도를 봄. 빈도 큰 라벨이 점수를 끌고감 — 우리 데이터에서 food가 매우 자주 활성된다면 food 성능이 좋으면 micro F1도 좋아 보임.
- **macro F1**: 라벨별 F1을 단순 평균. *드문 라벨* 도 동등하게 평가 — location 같은 빈도 작은 라벨에서 못하면 점수가 깎임.

**언제 무엇:**
- 라벨 빈도가 비슷 + 모든 라벨 동등 중요 → 둘이 비슷, 그냥 macro 보면 됨.
- 빈도 차이가 크고 *드문 라벨도 잘 잡고 싶다* → macro F1 (소수 클래스 보호).
- 전체 시스템의 평균적 정확도가 핵심 → micro F1.

실무에선 둘 다 보고 차이를 해석하는 게 표준입니다.

### Q3. (실무) 모든 라벨이 0인 샘플은 어떻게 처리하나요?

세 가지 접근.

1. **그대로 둔다**: BCE가 처리 가능. 각 라벨이 0이라는 신호를 학습. 가장 흔한 방식.
2. **버린다**: "라벨이 없으면 학습 신호가 없다"는 가정. 합성 데이터에서 빈 라벨 비율이 너무 크면 고려.
3. **"기타" 라벨 추가**: K+1번째 라벨을 만들어 "어느 측면도 안 맞음"을 명시적으로 표현. 데이터셋 설계 결정.

```python
# 옵션 (b): 빈 라벨 샘플 제거
mask_nonempty = Y.sum(axis=1) > 0
Y_clean = Y[mask_nonempty]
df_clean = df[mask_nonempty]
```

이번 챕터는 (a) 그대로 두기 — 합성 데이터에서 "어느 키워드도 없는 짧은 리뷰"는 자연스러운 부분이라.

### Q4. (실무) 키워드 매칭이 너무 단순한 것 같은데 실무에선 어떻게 하나요?

세 가지 접근이 일반적.

1. **사람 라벨링**: 가장 정확하지만 비용 큼. crowdsourcing, 도메인 전문가, 사내 라벨러.
2. **약지도 학습 (weak supervision)**: 우리 챕터처럼 휴리스틱 라벨로 *부트스트랩* → 학습된 모델로 *재라벨링* → 사람이 검수. Snorkel 같은 프레임워크.
3. **거대 모델 보조**: GPT-4/Claude 같은 LLM에게 라벨 지시문을 줘서 자동 태깅. 비용은 사람 라벨링보다 훨씬 저렴, 정확도는 도메인에 따라 다름.

이번 합성은 (2)의 가장 단순한 형태입니다 — 한 번 매칭하고 끝. 실무에선 매칭 결과를 한 번 학습한 모델로 다시 라벨을 매겨 사전을 보강하는 사이클을 돌립니다.

### Q5. (이론) hamming loss는 정확히 무엇을 측정하나요?

평균 라벨별 오답 비율입니다.

$$\text{Hamming loss} = \frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K} \mathbb{1}[\hat y_{ik} \neq y_{ik}]$$

직관: K개 라벨 중 평균적으로 몇 개가 틀렸나. K=5에서 hamming loss 0.1 = 평균 0.5개 라벨 틀림 = 한 샘플의 5개 라벨 중 평균 0.5개 잘못됨.

**Subset accuracy와의 관계**: subset accuracy는 "5개 라벨 *전부* 맞아야 1점"이라 매우 보수적, hamming은 *부분 점수* 를 줌. 일반적으로 hamming이 더 너그럽게 나옵니다.

### Q6. (실무) 임계값을 라벨마다 다르게 줄 수 있나요?

네, 자주 합니다. 라벨 빈도가 매우 다를 때 (예: food는 활성률 60%, location은 10%) 한 임계값으로는 둘 다 잘 잡기 어렵습니다.

```python
from sklearn.metrics import f1_score

# 라벨별 임계값 sweep으로 F1 최대화
best_thresholds = []
for k in range(K):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 17):
        f1 = f1_score(Y_test[:, k], (proba_ml[:, k] >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    best_thresholds.append(best_t)

print(f"라벨별 최적 임계값: {dict(zip(ASPECTS, best_thresholds))}")
```

⚠️ 주의: 위 코드는 *test set* 으로 임계값을 정해 보여준 것 — 실무에선 *별도 검증 데이터셋* 으로 임계값을 정하고 test에는 적용만 해야 합니다 (안 그러면 test에 누설).""")

# ----- 23. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

같은 데이터를 multi-class로 잘못 풀면 어떻게 될까요? 측면이 가장 강한 것 *하나* 만 정답으로 골라보고 (argmax) multinomial LogReg를 학습합니다.

```python
# 측면이 하나라도 활성된 샘플만 사용
mask_nonempty = Y.sum(axis=1) > 0
y_pseudo_class = Y[mask_nonempty].argmax(axis=1)   # 0-4 사이 정수, "가장 강한 측면"
texts = df["text"][mask_nonempty]

X_pseudo = tfidf.transform(texts)
model_pseudo = LogisticRegression(multi_class="multinomial", max_iter=1000)
model_pseudo.fit(X_pseudo[:4000], y_pseudo_class[:4000])
acc_pseudo = (model_pseudo.predict(X_pseudo[4000:]) == y_pseudo_class[4000:]).mean()
print(f"강제로 multi-class화 한 경우 accuracy: {acc_pseudo:.4f}")
```

힌트: 한 리뷰에 여러 측면이 동시에 있을 때 *하나만* 정답으로 골라 학습하면 정보가 사라집니다. 정답이 임의 선택이라 모델이 어느 라벨을 골라야 할지 모호해지고, multi-label 결과보다 정보가 적은 모델이 됩니다.""")

# ----- Phase 0 → 1 미리보기 -----
md(r"""## 🔮 Phase 0 마무리 — sklearn vs HuggingFace 미리보기

이 챕터로 sklearn 시대가 끝납니다. 다음 챕터(Ch 7)부터 등장하는 `transformers` (Hugging Face)는 *loss를 최소화한다* 는 목적은 같지만 **그 방식과 철학** 이 다릅니다. 큰 그림을 미리 잡아두면 Phase 1의 학습 코드가 낯설지 않습니다.

### 핵심 차이 한 문장

> **sklearn은 *수학 문제를 풀어준다*.**
> **HuggingFace는 *수학 문제를 푸는 과정* 을 우리가 통제한다.**

이 챕터에서 쓴 `LogisticRegression(max_iter=1000)` 은 lbfgs solver가 알아서 BCE를 최소화하는 가중치를 찾아 돌려줬습니다 — 우리는 `fit()` 한 줄과 결과만 봤어요.

HF의 `Trainer`는 학습 *과정* 을 명시합니다 — 학습률, 배치 크기, 에폭 수, 스케줄러, 평가 빈도. loss는 매 step마다 계산되어 backprop으로 가중치를 *조금씩* 옮깁니다. 같은 데이터로 학습해도 random seed가 바뀌면 결과가 미세하게 달라지는 이유.

### 한 표로 정리

| 축 | sklearn (Phase 0) | HuggingFace / PyTorch (Phase 1+) |
|---|---|---|
| **최적화 방식** | 수렴 보장 solver (lbfgs 등) 한 번 호출 → 전역 최적해 | 미니배치 SGD/Adam — 학습자가 epoch·step 통제 |
| **언제 끝나나** | 수렴 기준(`tol`) 도달 시 자동 | 사용자 지정 epoch 수 (멈출 시점 직접 결정) |
| **결정성** | convex 문제라 같은 입력엔 같은 출력 | non-convex — random seed·batch 순서에 따라 매번 미세 차이 |
| **에폭/배치 개념** | 보통 없음 — 전체 데이터 한 번에 | **핵심** — `num_train_epochs`, `batch_size` 명시 필수 |
| **loss를 직접 보나** | 거의 안 봄 (fit 후 평가만) | 매 step마다 loss 출력 + 곡선 추적 (학습이 망가지면 즉시 보임) |
| **하드웨어** | CPU, 단일 스레드 위주 | **GPU 필수** (fp16, gradient accumulation 등) |
| **loss 함수 지정** | 모델 클래스에 내장 (LogReg = log loss) | `problem_type` 자동 매핑 또는 `compute_loss` 오버라이드 |
| **모델 크기** | 수만 ~ 수십만 파라미터 | 사전학습 BERT — 6천만 ~ 수억 파라미터 |
| **학습 시간 (Yelp 5,000)** | 1초 미만 | T4 GPU에서 2-5분 |

### 코드 형태 미리보기 (Ch 9 BERT 회귀에서 본격 등장)

```python
# Phase 0 (sklearn — 우리가 한 형태)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)   # 한 줄. 수렴 기준까지 알아서 풀어줌.

# Phase 1 이후 (HuggingFace — 다음부터)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2,
)
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,             # 학습 반복 횟수
    per_device_train_batch_size=16, # 미니배치 크기
    learning_rate=2e-5,             # 학습률
    fp16=True,                      # T4에서 GPU 효율
    logging_steps=20,               # loss 곡선 출력
)
trainer = Trainer(model=model, args=args, train_dataset=..., ...)
trainer.train()   # 매 step마다 loss → backward → optimizer step
```

각 인자가 무엇을 하는지는 Ch 9에서 본격적으로 펼쳐 봅니다. 지금은 "fit 한 줄이 수십 개 인자로 펼쳐진다" 는 감만 가지면 됩니다.

### 변하지 않는 것

Loss 자체 — `BCEWithLogitsLoss`, `CrossEntropyLoss`, `MSELoss` — 는 sklearn에서 익힌 그대로 Phase 1+에서도 등장합니다. **달라지는 건 *어떻게 최소화하느냐* 의 도구뿐**. Phase 0의 직관(BCE 수치 표, OvR fit 분해, softmax 동등성 등)이 Phase 1의 BERT 학습에서도 그대로 살아 있습니다.""")

# ----- 마지막. next -----
md(r"""## 다음 챕터 예고

**Phase 1 시작 — Chapter 7. BERT 첫 만남 (`pipeline`)**

- sklearn 시대 끝, **`transformers` 라이브러리** 가 처음 등장
- `pipeline("sentiment-analysis")` 한 줄로 사전학습된 DistilBERT 추론
- `pipeline` 안에서 일어나는 3단계 (tokenizer / model / post-processing) 직접 풀어보기
- **첫 WordPiece 등장** — 같은 문장이 TF-IDF와 어떻게 다른 단위로 쪼개지는지 비교
- 학습 없음 (추론만) — `Trainer`는 Ch 9에서 본격 등장""")


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
