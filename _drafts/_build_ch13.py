"""Build 13_bert_multilabel/13_bert_multilabel.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "13_bert_multilabel" / "13_bert_multilabel.ipynb"

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
md(r"""# Chapter 13. BERT Multi-label — Yelp 측면(aspect) 키워드

**목표**: Ch 12(BERT 5클래스 분류) 셋업과 출력 헤드 크기를 *완전히 동일* 하게 둡니다 (`num_labels=5` 그대로). 변하는 건 **task의 의미** 입니다 — 5개 라벨이 *서로 배타적인 클래스* 가 아니라 *각각 독립적으로 활성될 수 있는 태그* 입니다 ("food" 와 "service" 가 동시에 1).

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 12분 (BERT 학습 ~10분 + sklearn 비교 ~30초 + 평가/시각화)

---

## 학습 흐름

1. 🚀 **실습**: Ch 6에서 만들었던 *측면 키워드 합성 라벨* (food/service/price/ambiance/location)을 그대로 BERT로 학습. `num_labels=5` + `problem_type="multi_label_classification"` 로 BCE per-label 자동 매핑.
2. 🔬 **해부**: 라벨별 sigmoid 확률 분포 (5 패널 KDE) + 라벨 간 공동 활성 패턴 (correlation heatmap).
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 Ch 6의 sklearn `OneVsRestClassifier(LogisticRegression)` baseline 재현 → 라벨별 metric 비교.

---

> 📒 **사전 학습 자료**: Ch 6 (sklearn multi-label, OvR), Ch 10 (BERT `num_labels=1` + `multi_label_classification` 트릭 — Ch 13은 그 트릭을 K=5 로 확장한 형태), Ch 12 (BERT multi-class). 이번 챕터는 self-contained.""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 6 | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 측면 키워드 합성 | (5차원) | sigmoid (각각) | `BCEWithLogitsLoss` per-label |
| 10 | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp 이진화 | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| 12 | DistilBERT 파인튜닝 | 같음 | Yelp 5클래스 | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |
| **13 ← 여기** | DistilBERT 파인튜닝 | 같음 | **Yelp + 측면 키워드 합성** | **`Linear(H, 5)`** | **sigmoid (per-label)** | **`BCEWithLogitsLoss` (per-label)** |
| 14 (다음) | DistilBERT 파인튜닝 | 같음 | Yelp + 측면 + 별점 보조 | `Linear(H, 5)` 메인 + `Linear(H, 1)` 보조 | sigmoid + 없음 | `BCE(per-label) + λ·MSE` |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 12)

| 축 | Ch 12 (multi-class) | Ch 13 (multi-label) |
|---|---|---|
| **Task** | 5클래스 *single-label* (서로 배타적) | **5라벨 *multi-label*** (동시 활성 가능) ← 본질적 변화 |
| `num_labels` | 5 | **5** (그대로!) |
| `problem_type` | `"single_label_classification"` | **`"multi_label_classification"`** ← BCE 자동 매핑 |
| Activation | softmax (합=1 강제) | **per-label sigmoid** (각각 독립 0-1) |
| Loss | `CrossEntropyLoss` | **`BCEWithLogitsLoss`** per-label |
| 라벨 형식 | int 스칼라 (0-4) | **multi-hot float `[1, 0, 1, 0, 1]`** |
| 모델 출력 의미 | logits[k] = 클래스 k의 *상대적* 점수 | logits[k] = 라벨 k의 *독립적* 활성 점수 |
| 평가 metric | accuracy + macro F1 + AUC OvR | per-label precision/recall/F1 + micro/macro F1 + per-label AUC |

> **결정적 인사이트**: 모델 *아키텍처는 동일* — `Linear(H, 5)` 헤드. 변하는 건 *해석* 과 그에 따른 loss/activation입니다. 같은 5차원 출력을 *softmax로 모아서 한 클래스 고르기* 와 *5개 sigmoid로 각자 0/1 결정하기* 두 가지로 다르게 쓰는 것.

### 왜 multi-label은 softmax로 풀 수 없는가

softmax는 출력의 *합 = 1* 을 강제합니다 ($\sum_k \mathrm{softmax}(z)_k = 1$). 이는 *서로 배타적* 클래스에 자연스럽지만 multi-label과 충돌합니다.

리뷰가 "food=1 (음식 언급) 그리고 service=1 (서비스 언급)" 일 때:
- **softmax 모델은 표현 불가**: P(food)=0.9 면 나머지 4 라벨 합이 0.1 로 강제 → 'service=0.85 동시 활성' 이 *수학적으로 불가능*.
- **per-label sigmoid 모델은 표현 가능**: 각 라벨이 독립이라 P(food)=0.9 와 P(service)=0.85 가 동시에 자연스러움.

즉 task가 *진짜 multi-label* 이라면 loss/activation 선택이 강제됩니다 (Ch 6 에서 본 동일한 논리, BERT로 옮겨옴).""")

# ----- 4. Loss 노트 -----
md(r"""## 📐 Loss 노트 — `BCEWithLogitsLoss` per-label (Ch 6 그대로, BERT 맥락)

K개 라벨 각각에 *독립적* BCE를 적용한 뒤 평균:

$$L = \frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K}\left[ y_{i,k} \log \sigma(z_{i,k}) + (1-y_{i,k}) \log(1-\sigma(z_{i,k})) \right]$$

각 $z_{i,k}$ 는 *독립 logit* — 라벨 k가 *얼마나 활성될지* 의 점수, 다른 라벨과 무관. PyTorch `BCEWithLogitsLoss` 가 5개 위치를 한 번에 처리하지만 수식적으론 K개의 binary BCE 평균.

**숫자로 감 잡기 (K=5, 정답 multi-hot $\mathbf{y} = [1, 0, 1, 0, 1]$)** — logits 별 손실 분해:

| 라벨 | $y_k$ | logit $z_k$ | $\sigma(z_k)$ | 정답일 때 손실 |
|---|---|---|---|---|
| food | 1 | 3.0 | 0.953 | $-\log 0.953 = 0.048$ |
| service | 0 | -2.0 | 0.119 | $-\log(1-0.119) = 0.127$ |
| price | 1 | 0.5 | 0.622 | $-\log 0.622 = 0.474$ |
| ambiance | 0 | 1.5 | 0.818 | $-\log(1-0.818) = 1.704$ ← 자신 있게 틀림 |
| location | 1 | -0.5 | 0.378 | $-\log 0.378 = 0.974$ |

평균 loss = $(0.048 + 0.127 + 0.474 + 1.704 + 0.974) / 5 \approx 0.665$.

**핵심 직관 — 라벨 사이엔 직접 신호가 없음**: BCE per-label은 라벨 k의 logit이 라벨 j의 정답에서 *직접* 학습 신호를 받지 않습니다. 모델이 라벨 간 상관을 학습하는 건 *공유 BERT 본체* (모든 라벨이 같은 768-dim CLS 표현에서 옴) 덕분이지 loss 자체에는 라벨 간 결합 항이 없습니다. 이 점이 multi-class softmax와 결정적 차이.""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

Ch 12와 동일 — `distilbert-base-uncased` WordPiece, `max_length=128`. 토크나이저는 라벨 *형식* 에 무관하므로 single-label 이든 multi-label 이든 변화 없음.

> **다음 챕터(Ch 14)**: 토크나이저 그대로. 변하는 건 *모델에 보조 헤드* 가 추가되고 *loss에 보조 항* 이 가중합으로 더해지는 점.""")

# ----- 6. install + import -----
code(r"""!pip install -q transformers datasets""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report,
    roc_auc_score, hamming_loss,
)
# Ch 6 sklearn baseline 비교용
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

plt.rcParams["axes.unicode_minus"] = False

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CPU runtime — training will be very slow. Switch to T4 recommended.")""")

# ----- 7. nvidia-smi baseline -----
md(r"""**baseline VRAM**:""")

code(r"""!nvidia-smi""")

# ----- 8. 데이터 + 측면 라벨 합성 -----
md(r"""## 1. 🚀 데이터 — Yelp + 측면(aspect) 합성 라벨 (Ch 6과 동일)

Yelp 리뷰엔 multi-label 정답이 없습니다. Ch 6에서처럼 5개 측면(aspect)별 키워드 사전을 만들어 텍스트에서 매칭 — 어떤 키워드라도 등장하면 해당 측면을 1로 활성. 5차원 multi-hot 벡터가 합성됩니다.

| 측면 | 의미 | 키워드 예시 |
|---|---|---|
| `food` | 음식의 맛/메뉴 | food, meal, dish, taste, delicious, ... |
| `service` | 서비스/응대 | service, staff, waiter, friendly, rude, ... |
| `price` | 가격/가성비 | price, cheap, expensive, value, worth, ... |
| `ambiance` | 분위기/인테리어 | atmosphere, decor, music, vibe, cozy, ... |
| `location` | 위치/주차 | location, parking, area, neighborhood, ... |

> **합성의 한계** — 키워드 매칭은 *언급한 측면* 만 잡고 *언급한 측면이 긍정인지 부정인지* 는 알 수 없습니다. 또 *키워드 없이* 측면이 표현된 경우(예: "10 minutes wait" → service)도 놓칩니다. 이 한계는 챕터 끝에서 솔직히 짚습니다.""")

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


def extract_aspects(text: str) -> list[float]:
    text_lower = text.lower()
    return [
        float(any(re.search(rf"\b{re.escape(kw)}\b", text_lower) for kw in keywords))
        for keywords in ASPECT_KEYWORDS.values()
    ]

print(f"K (number of aspects): {K}")
print(f"aspects: {ASPECTS}")""")

code(r"""tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("yelp_review_full")
train_full = ds["train"].shuffle(seed=42).select(range(5000))
eval_full  = ds["test"].shuffle(seed=42).select(range(1000))

# 측면 라벨 합성 — 각 텍스트에 multi-hot 5차원 벡터 부착
def attach_aspects(batch):
    batch["aspects"] = [extract_aspects(t) for t in batch["text"]]
    return batch

train_full = train_full.map(attach_aspects, batched=True)
eval_full  = eval_full.map(attach_aspects,  batched=True)

print(f"train: {len(train_full)}, eval: {len(eval_full)}")
print(f"\nFirst sample:")
print(f"  text: {train_full[0]['text'][:150]}...")
print(f"  aspects (multi-hot): {train_full[0]['aspects']}")
print(f"  active aspects: {[a for a, v in zip(ASPECTS, train_full[0]['aspects']) if v > 0]}")""")

code(r"""# 측면별 활성률
Y_train = np.array(train_full["aspects"])
Y_eval  = np.array(eval_full["aspects"])

print("Per-aspect activation rate (train):")
for k, aspect in enumerate(ASPECTS):
    rate = Y_train[:, k].mean()
    print(f"  {aspect:>9}: {rate:.1%}  ({int(Y_train[:, k].sum())} / {len(Y_train)})")

n_active = Y_train.sum(axis=1)
print(f"\nMean active labels per sample: {n_active.mean():.2f}")
print(f"Active label distribution (train):")
for n in range(K + 1):
    cnt = (n_active == n).sum()
    print(f"  {n} labels active: {cnt} samples ({cnt/len(Y_train):.1%})")""")

# ----- 9. 토큰화 -----
md(r"""**Ch 12와의 한 줄 차이**: `out["labels"] = [int(l) for l in batch["label"]]` → `out["labels"] = [list(map(float, a)) for a in batch["aspects"]]`. 라벨이 *int 스칼라* 가 아니라 *길이 5 multi-hot float 벡터*. 이 형식 + `problem_type="multi_label_classification"` 두 가지가 BCE per-label 자동 매핑의 트리거.""")

code(r"""def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # multi-hot 5차원 float 벡터 (BCEWithLogitsLoss가 받는 형식)
    out["labels"] = [list(map(float, a)) for a in batch["aspects"]]
    return out

train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(["text", "label", "aspects"])
eval_tok  = eval_full.map(tokenize_fn,  batched=True).remove_columns(["text", "label", "aspects"])

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (length-5 multi-hot float vector)")""")

# ----- 10. 모델 로드 -----
md(r"""## 2. 모델 로드 — `num_labels=5` + `multi_label_classification`

Ch 12와 *모델 아키텍처는 동일* (`Linear(H, 5)` 분류 헤드). 변하는 한 가지 — `problem_type="multi_label_classification"` — 가 자동 매핑되는 loss를 BCE per-label 로 바꿉니다.""")

code(r"""model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=K,
    problem_type="multi_label_classification",   # ← BCEWithLogitsLoss per-label 자동 매핑
    id2label={i: a for i, a in enumerate(ASPECTS)},
    label2id={a: i for i, a in enumerate(ASPECTS)},
)

def param_summary(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

total, trainable = param_summary(model)
print(f"Parameters:           {total:>13,}  ({total/1e6:.1f} M)")
print(f"Trainable parameters: {trainable:>13,}  ({trainable/total:.1%})")
print(f"Classifier:           {model.classifier}")
print(f"problem_type:         {model.config.problem_type}")
print(f"id2label:             {model.config.id2label}")""")

md(r"""**Ch 12와 파라미터 수가 *완전히 동일*** — 차이는 `problem_type` 한 줄뿐. 같은 모델이 *어떻게 해석되고 어떤 loss로 학습되는가* 만 바뀝니다.""")

code(r"""!nvidia-smi""")

# ----- 11. 학습 -----
md(r"""## 3. 학습 — Ch 12와 동일한 hyperparams

Ch 12와 *완전히 같은* learning rate, batch size, epoch 수, seed. 평가 metric만 multi-label용으로 새로 짭니다.""")

code(r"""def compute_metrics(eval_pred):
    logits, labels = eval_pred                      # logits: (N, K), labels: (N, K) float
    probs = 1.0 / (1.0 + np.exp(-logits))           # per-label sigmoid
    preds = (probs >= 0.5).astype(int)              # threshold 0.5

    out = {}
    # Hamming loss — 전체 라벨 위치 중 틀린 비율 (낮을수록 좋음)
    out["hamming_loss"] = float(hamming_loss(labels, preds))

    # Micro F1 — 전체 라벨을 한꺼번에 (TP/FP/FN 합산 후 F1)
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0,
    )
    out["micro_f1"] = float(f1_mi)
    out["micro_precision"] = float(p_mi)
    out["micro_recall"]    = float(r_mi)

    # Macro F1 — 라벨별 F1을 평균 (각 라벨에 동일 가중치)
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0,
    )
    out["macro_f1"] = float(f1_ma)
    out["macro_precision"] = float(p_ma)
    out["macro_recall"]    = float(r_ma)

    # Per-label AUC (One-vs-Rest 자체)
    try:
        out["macro_auc"] = float(roc_auc_score(labels, probs, average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    return out""")

code(r"""training_args = TrainingArguments(
    output_dir="./ch13_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=True,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
print(f"\nTraining done — mean train loss: {train_result.training_loss:.4f}")""")

code(r"""!nvidia-smi""")

# ----- 12. 평가 -----
md(r"""## 4. 🔬 평가 — 라벨별 sigmoid 확률 + 활성 패턴

Ch 10의 sigmoid+BCE 평가 패턴을 *5번 반복* 한 셈입니다 — 각 라벨에 대해 독립적으로 확률 분포·정확도·F1을 계산.""")

code(r"""# 평가 metric
eval_metrics = trainer.evaluate()
print("BERT multi-label evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")""")

code(r"""# logits → per-label sigmoid → multi-hot 예측
preds_output = trainer.predict(eval_tok)
logits = preds_output.predictions                   # (N, 5)
labels = preds_output.label_ids.astype(int)         # (N, 5) multi-hot
probs  = 1.0 / (1.0 + np.exp(-logits))              # (N, 5) per-label prob
preds  = (probs >= 0.5).astype(int)                 # (N, 5) multi-hot prediction

print(f"logits shape: {logits.shape}")
print(f"prob ranges per label:")
for k, a in enumerate(ASPECTS):
    print(f"  {a:>9}: [{probs[:, k].min():.4f}, {probs[:, k].max():.4f}]  "
          f"true rate={labels[:, k].mean():.1%}, pred rate={preds[:, k].mean():.1%}")""")

code(r"""# Per-label classification report
print(classification_report(
    labels, preds,
    target_names=ASPECTS,
    digits=4, zero_division=0,
))""")

# ----- 12a. 메인 시각화: per-label sigmoid prob KDE (5 facets) -----
md(r"""### 4-1. 메인 그림 — 라벨별 sigmoid 확률 KDE (5 패널)

Ch 10에서 봤던 *확률 공간 KDE* 를 5개 라벨에 대해 *각각* 그립니다. 라벨이 *독립* 이라는 multi-label의 본질이 시각적으로 드러나는 그림입니다 — 라벨마다 학습 난이도와 분리도가 *다를 수* 있습니다.""")

code(r"""sns.set_theme(style="whitegrid", context="talk")

# Long-form DataFrame 만들기
records = []
for k, a in enumerate(ASPECTS):
    for i in range(len(probs)):
        records.append({"aspect": a, "prob": probs[i, k], "label": int(labels[i, k])})
df_long = pd.DataFrame(records)

g = sns.FacetGrid(
    df_long, col="aspect", col_wrap=3, height=3.2, aspect=1.4,
    sharex=True, sharey=False,
)
g.map_dataframe(
    sns.kdeplot, x="prob", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette={0: "#5B8DEF", 1: "#F47272"}, clip=(0, 1),
)
for ax in g.axes.flat:
    ax.axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("sigmoid prob")
g.add_legend(title="label")
g.fig.suptitle("Per-label sigmoid probability distribution by ground truth", y=1.03)
plt.tight_layout()
plt.show()""")

md(r"""**해석**

- **잘 학습된 라벨** (예: food): label=0 곡선은 0 근처, label=1 곡선은 1 근처에 있고 둘이 거의 만나지 않음. *분리가 깨끗*.
- **활성률이 낮은 라벨** (예: location): label=1 샘플이 적어 곡선이 노이즈가 큼. 그래도 분리는 보여야 함.
- **두 곡선이 0.5 근처에서 크게 겹치면** → 그 라벨은 모델이 잘 못 분리. 키워드 매칭이 *얕아서* 진짜 신호를 못 잡았거나, 학습 데이터가 부족한 상태.""")

# ----- 12b. 보조: 라벨 간 공동 활성 -----
md(r"""### 4-2. 보조 그림 — 라벨 간 공동 활성 패턴

Multi-label 의 핵심 질문 중 하나: *어떤 라벨 쌍이 같이 등장하는가?* 모델이 라벨 *간 상관* 을 학습 데이터에서 흡수했는지 확인합니다.

`true co-occurrence` (실제 데이터의 라벨 동시 등장 빈도)와 `predicted co-occurrence` (모델 예측의 동시 등장 빈도)를 나란히 그려 두 행렬이 비슷하면 모델이 라벨 구조를 잘 잡고 있는 것.""")

code(r"""def cooccurrence_matrix(Y):
    # Y: (N, K) multi-hot. Returns (K, K) where M[i,j] = P(label_j=1 | label_i=1).
    Y = Y.astype(float)
    K_ = Y.shape[1]
    M = np.zeros((K_, K_))
    for i in range(K_):
        row_i = Y[:, i]
        n_i = row_i.sum()
        if n_i == 0:
            continue
        for j in range(K_):
            M[i, j] = (row_i * Y[:, j]).sum() / n_i
    return M

cooc_true = cooccurrence_matrix(labels)
cooc_pred = cooccurrence_matrix(preds)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, M, title in [
    (axes[0], cooc_true, "True co-occurrence  P(j | i)"),
    (axes[1], cooc_pred, "Predicted co-occurrence  P(j | i)"),
]:
    sns.heatmap(
        M, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=ASPECTS, yticklabels=ASPECTS,
        cbar_kws={"label": "conditional probability"}, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("label j")
    ax.set_ylabel("given label i")
plt.tight_layout()
plt.show()""")

md(r"""**해석**

- **대각선 = 1.0** — 자기 자신과는 항상 같이 등장 (정의상).
- **off-diagonal cell M[i, j]** = "라벨 i가 활성된 샘플 중 라벨 j 도 활성된 비율". 비대칭 행렬.
- food와 service가 같이 자주 등장하면 두 모델 모두 0.5+ 값. *true* 와 *predicted* 가 거의 비슷한 패턴이면 모델이 라벨 구조를 잘 학습했다는 뜻.
- **predicted cell이 true cell 보다 일관되게 높으면** → 모델이 라벨을 *너무 많이* 활성하는 경향 (over-prediction). threshold를 0.5보다 높게 (예: 0.6) 두면 calibration 개선.""")

# ----- 13. sklearn baseline 비교 (CLIMAX) -----
md(r"""## 5. 🛠️ 클라이맥스 — Ch 6 sklearn `OneVsRestClassifier(LogisticRegression)` 와 비교

Ch 6의 sklearn 셋업을 *이 노트북 안에서* 다시 학습해 라벨별로 BERT와 비교합니다. **multi-label에서도 BERT의 67M이 sklearn 대비 어디서 이기는가?**""")

code(r"""# Ch 6 셋업 재현 — TF-IDF + OneVsRestClassifier(LogisticRegression)
texts_train = list(train_full["text"])
texts_eval  = list(eval_full["text"])
Y_train_bin = np.array(train_full["aspects"]).astype(int)
Y_eval_bin  = np.array(eval_full["aspects"]).astype(int)

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train = vec.fit_transform(texts_train)
X_eval  = vec.transform(texts_eval)

clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, n_jobs=-1))
clf.fit(X_train, Y_train_bin)

probs_sk = clf.predict_proba(X_eval)        # (N, 5)
preds_sk = (probs_sk >= 0.5).astype(int)    # (N, 5)

p_mi_sk, r_mi_sk, f1_mi_sk, _ = precision_recall_fscore_support(
    Y_eval_bin, preds_sk, average="micro", zero_division=0,
)
p_ma_sk, r_ma_sk, f1_ma_sk, _ = precision_recall_fscore_support(
    Y_eval_bin, preds_sk, average="macro", zero_division=0,
)
auc_sk = float(roc_auc_score(Y_eval_bin, probs_sk, average="macro"))

print(f"sklearn TF-IDF + OvR LogReg:")
print(f"  vocabulary size:    {len(vec.vocabulary_):,}")
print(f"  micro F1:           {f1_mi_sk:.4f}")
print(f"  macro F1:           {f1_ma_sk:.4f}")
print(f"  macro AUC:          {auc_sk:.4f}")
print(f"  hamming loss:       {hamming_loss(Y_eval_bin, preds_sk):.4f}")""")

md(r"""### 5-1. 두 모델의 metric 비교""")

code(r"""metrics_bert = {
    k.replace("eval_", ""): v for k, v in eval_metrics.items()
    if k.startswith("eval_") and isinstance(v, float)
}
metrics_sk = {
    "hamming_loss":    float(hamming_loss(Y_eval_bin, preds_sk)),
    "micro_f1":        float(f1_mi_sk),
    "micro_precision": float(p_mi_sk),
    "micro_recall":    float(r_mi_sk),
    "macro_f1":        float(f1_ma_sk),
    "macro_precision": float(p_ma_sk),
    "macro_recall":    float(r_ma_sk),
    "macro_auc":       auc_sk,
}

common = [k for k in metrics_bert if k in metrics_sk]
cmp = pd.DataFrame({
    "metric":             common,
    "sklearn (OvR)":      [metrics_sk[k]   for k in common],
    "BERT (this chapter)":[metrics_bert[k] for k in common],
})
cmp["BERT - sklearn"] = cmp["BERT (this chapter)"] - cmp["sklearn (OvR)"]
print(cmp.round(4).to_string(index=False))""")

md(r"""### 5-2. 라벨별 F1 비교 — 어디서 BERT가 이기나""")

code(r"""def per_label_f1(Y_true, Y_pred):
    f1s = []
    for k in range(K):
        _, _, f1, _ = precision_recall_fscore_support(
            Y_true[:, k], Y_pred[:, k], average="binary", zero_division=0,
        )
        f1s.append(float(f1))
    return f1s

f1_bert = per_label_f1(labels, preds)
f1_sk   = per_label_f1(Y_eval_bin, preds_sk)

label_cmp = pd.DataFrame({
    "aspect":     ASPECTS,
    "sklearn F1": f1_sk,
    "BERT F1":    f1_bert,
})
label_cmp["BERT - sklearn"] = label_cmp["BERT F1"] - label_cmp["sklearn F1"]
print(label_cmp.round(4).to_string(index=False))

# 막대 그래프
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(K)
width = 0.38
ax.bar(x_pos - width/2, f1_sk,   width, label="sklearn (OvR)",     color="#5B8DEF")
ax.bar(x_pos + width/2, f1_bert, width, label="BERT (this chapter)", color="#F47272")
ax.set_xticks(x_pos)
ax.set_xticklabels(ASPECTS)
ax.set_ylim(0, 1)
ax.set_ylabel("Per-label F1")
ax.set_title("Per-label F1 — sklearn OvR vs BERT")
ax.legend()
plt.tight_layout()
plt.show()""")

md(r"""**해석**

- 키워드 매칭으로 합성한 라벨은 *키워드 단어가 본질 신호* 라 sklearn TF-IDF가 의외로 강합니다 — 라벨 정의 자체가 단어 빈도와 일치하기 때문.
- BERT가 *큰 폭으로* 이기는 라벨이 있다면 → 그 라벨의 *합성 룰이 키워드만으로 안 잡히는 신호* 를 BERT가 추가로 학습한 것 (예: ambiance에서 "lighting was perfect" 같은 묘사).
- BERT가 *지는 라벨* 도 있을 수 있음 — 키워드가 *결정적* 인 라벨에서 sklearn은 *완벽* 한 매칭, BERT는 *근사* 라 약간의 noise가 들어감.

**합성 라벨의 본질적 한계** — 이 비교는 *키워드 매칭으로 만든 라벨* 위에서의 비교. 실제 사람-annotated multi-label 데이터에선 BERT 격차가 훨씬 큼 (단어 빈도로 안 잡히는 미묘한 측면 인식이 BERT의 강점).""")

# ----- 14. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModelForSequenceClassification(num_labels=5, problem_type="multi_label_classification")` | Ch 12 셋업에서 problem_type만 변경 → BCE per-label 자동 매핑 | Ch 14에서 메인 헤드로 그대로 |
| `sklearn.metrics.hamming_loss` | 전체 (sample × label) 위치에서 틀린 비율 | multi-label 챕터마다 |
| `precision_recall_fscore_support(..., average="micro"/"macro")` | multi-label용 F1 — micro는 라벨 합산, macro는 라벨 평균 | Ch 14·17·18 |
| `roc_auc_score(..., average="macro")` | 라벨별 AUC를 평균 | Ch 17·18 |
| `seaborn.FacetGrid + map_dataframe` | 5개 라벨에 같은 KDE를 facet으로 | 라벨이 많은 시각화에 재등장 |
| `OneVsRestClassifier(LogisticRegression())` | sklearn multi-label baseline | 비교용 |""")

# ----- 15. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. multi-label 문제를 *softmax + CrossEntropyLoss* 로 풀려고 하면 무엇이 잘못되나요? 수식으로 한 줄 설명할 수 있나요?
2. `num_labels=5` 가 Ch 12와 Ch 13에서 *같은 숫자* 인데 *모델이 학습하는 의미* 는 어떻게 다른가요?
3. Macro F1과 Micro F1의 차이는 무엇이고, 어느 한쪽이 *훨씬* 낮으면 무엇을 의심해야 하나요?
4. 라벨별 *공동 활성 행렬* 에서 모델 예측이 실제보다 일관되게 높은 행을 보였다면 어떤 처치가 필요한가요?""")

# ----- 16. FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) Multi-label에서 threshold 0.5는 항상 옳은가요?

아닙니다. 0.5는 *기본값* 일 뿐 라벨마다 최적 threshold가 다를 수 있습니다.

- **클래스 불균형**: 라벨 활성률이 5% 인 라벨에선 0.5가 너무 보수적 — 모델이 거의 안 활성. threshold를 0.2-0.3으로 낮추면 recall이 크게 올라감.
- **F1 최적 threshold 탐색**: validation set에서 *라벨별로* 0.1-0.9 grid search → F1 최대 지점 선택.

```python
def best_threshold(probs_k, labels_k):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.1, 0.91, 0.05):
        preds_k = (probs_k >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels_k, preds_k, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1

# 라벨별로 따로
thresholds = [best_threshold(probs[:, k], labels[:, k])[0] for k in range(K)]
```

운영 환경에선 *라벨별 threshold* 를 저장해 두고 추론 시 적용.

### Q2. (이론) Multi-class를 multi-label로 풀어도 되나요? (single-label 데이터를 multi-hot 형식으로 변환)

기술적으론 됩니다. 단점이 큽니다.

| | multi-class (CE) | single-label을 multi-label (BCE) 로 |
|---|---|---|
| 라벨 간 *경쟁* 학습 | softmax가 자동으로 강제 | BCE per-label은 라벨 간 직접 신호 없음 |
| confidence 의미 | 항상 합 = 1, 명확 | 5개 라벨의 sigmoid 확률, *합 ≠ 1* |
| 추론 후처리 | `argmax` 한 줄 | `argmax(probs)` 또는 `np.where(probs > th)` |
| 학습 수렴 속도 | 빠름 (라벨 경쟁 신호 있음) | 약간 느림 |

multi-class가 *명확히 single-label* 이면 CE가 정답. multi-label로 푸는 건 *어떤 이유로 두 task를 통합해야 할 때* (예: Ch 14 보조 헤드처럼).

### Q3. (실무) Class weights를 multi-label BCE에 적용하려면?

`pos_weight` 파라미터 — `BCEWithLogitsLoss(pos_weight=...)` 가 라벨별 양성 가중치를 받습니다.

```python
import torch
from torch import nn

# 라벨별 양성 비율 → pos_weight = (negative count / positive count)
pos_count = Y_train_bin.sum(axis=0)
neg_count = len(Y_train_bin) - pos_count
pos_weight = torch.tensor(neg_count / np.maximum(pos_count, 1), dtype=torch.float).to("cuda")
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

`Trainer.compute_loss` 를 오버라이드해서 위 `loss_fn` 으로 바꾸면 됩니다. 활성률 5%인 라벨은 pos_weight ≈ 19 → 양성 샘플의 손실이 19배 가중되어 모델이 양성 예측을 더 자주 하도록 강제.

### Q4. (이론) 모델이 라벨 *간* 상관을 학습하는 메커니즘은? (loss엔 라벨 결합 항이 없는데)

핵심은 **공유 BERT 본체** 입니다. 5개 라벨의 logit이 같은 768-dim CLS hidden state $h$ 에서 *별도의 5개 가중치 행* 을 통해 나옵니다.

$$z_k = w_k^\top h + b_k, \quad k = 1, \ldots, K$$

학습 중:

- 라벨 1이 활성된 샘플에서 $w_1$ 이 $h$ 의 특정 차원을 강조하도록 학습됩니다.
- 같은 샘플에서 라벨 2도 활성됐다면 $w_2$ 도 *비슷한 차원* 을 강조하게 됩니다.
- 결과적으로 라벨 1과 2가 같이 활성되는 *입력 패턴* 에 대해 두 logit이 동시에 커지는 *간접* 결합이 생깁니다.

Loss에는 결합 항이 없지만 *gradient가 BERT 본체를 거쳐 흐를 때* 결합이 자연스럽게 학습됩니다. 이게 BERT 같은 *공유 표현* 모델이 multi-label에서 sklearn OvR 보다 강한 이유 — sklearn은 K개 LogReg가 *완전 분리* 학습되므로 이런 간접 결합이 없습니다.

### Q5. (실무) 라벨이 100개 이상인 multi-label은 BERT로 어떻게 푸나요?

**일반적 패턴**:

1. **라벨이 ~50개 이하**: `num_labels=K` + per-label sigmoid + BCE 그대로. 본 챕터 패턴.
2. **라벨이 100-1000개**: 헤드의 weight matrix가 커짐 (768·K). 메모리 압박. 해결책:
   - **Hierarchical labels**: 라벨을 트리로 구조화해 *상위 → 하위* 단계적 분류 (예: "음식 > 한식 > 김치찌개")
   - **Knowledge distillation**: 큰 multi-label 모델에서 작은 모델로 distill
3. **라벨이 1000+ 개**: extreme multi-label (XML). 별도 분야 — `XML-CNN`, `BERT-XMC` 등 특화 모델 사용.

영화 장르 분류 (~30개), 기사 토픽 (~50개) 정도면 본 챕터 패턴으로 충분.

### Q6. (실무) Multi-label에서 *클래스가 추가되면* 모델을 처음부터 다시 학습해야 하나요?

기본적으론 그렇습니다. 분류 헤드의 weight shape이 `(K, 768)` 이라 K가 바뀌면 헤드가 호환 안 됨. 단, BERT 본체는 그대로 재사용 가능.

```python
# 새 라벨 1개 추가 (K=5 → K=6)
old_model = AutoModelForSequenceClassification.from_pretrained("./ch13_output")
new_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=6, problem_type="multi_label_classification",
)
# 기존 5라벨 weight를 새 모델에 복사
new_model.classifier.weight.data[:5] = old_model.classifier.weight.data
new_model.classifier.bias.data[:5]   = old_model.classifier.bias.data
# 6번째 라벨은 random init 그대로 → fine-tune 시작
```

이러면 기존 5라벨은 *학습된 상태로 시작*, 6번째만 *처음부터 학습* . 실무에서 자주 쓰는 incremental learning 패턴.""")

# ----- 17. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# multi-label 모델에 int 스칼라 라벨 (Ch 12 형식) 을 넣어보기
def tokenize_wrong(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # multi-hot vector 대신 첫 번째 활성 라벨의 인덱스만 (single-label 형식)
    out["labels"] = [
        next((i for i, v in enumerate(a) if v > 0), 0)
        for a in batch["aspects"]
    ]
    return out
```

힌트: `BCEWithLogitsLoss` 는 *logits 와 같은 shape의 float 텐서* 를 라벨로 받는데, 위 코드는 *(B,) int* 를 넘깁니다. shape mismatch + dtype mismatch 두 가지 에러가 동시에 날 수 있어 메시지가 길어집니다.""")

# ----- 18. next -----
md(r"""## 다음 챕터 예고

**Chapter 14. BERT Auxiliary Loss — 측면 분류 + 별점 보조 회귀**

- 메인 task: Ch 13의 multi-label 측면 분류 (`num_labels=5` + BCE per-label) — *완전히 동일*
- 추가: *보조 헤드* `Linear(H, 1)` 로 별점 점수 회귀 (별점 정규화 0-1)
- 손실: `L = BCE_per_label(메인) + λ · MSE(보조)` 가중합 (λ는 hyperparameter)
- `Trainer.compute_loss` 오버라이드로 두 헤드를 동시 학습
- 보조 task가 메인 task의 정확도를 *얼마나 끌어올리는지* 측정 (auxiliary loss의 정통 활용)

> **변하는 축**: 메인 task와 모델 본체는 그대로, *Loss에 보조 항이 추가* 됩니다 — Loss 축의 마지막 단계 ("BCE per-label → +Auxiliary").""")


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
