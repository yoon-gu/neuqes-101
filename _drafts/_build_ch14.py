"""Build 14_auxiliary_loss/14_auxiliary_loss.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "14_auxiliary_loss" / "14_auxiliary_loss.ipynb"

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
md(r"""# Chapter 14. BERT Auxiliary Loss — 측면 분류 + 별점 보조 회귀 (Phase 1 클라이맥스)

**목표**: Ch 13의 multi-label 측면 분류를 *메인 task* 로 그대로 두고, **별점 회귀 보조 헤드** 를 추가합니다. 손실은 가중합 형태:

$$L = L_\text{main}(\text{측면 BCE per-label}) + \lambda \cdot L_\text{aux}(\text{별점 MSE})$$

핵심 질문은 *"보조 task가 메인 task의 정확도를 끌어올리는가?"* — 같은 BERT 본체를 두 task가 *공유* 학습하면서, 별점이라는 *연속적이고 일관성 있는 신호* 가 측면 분류 표현에 도움이 되는지 직접 측정합니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 22분 (보조 ON 학습 ~9분 + 보조 OFF 비교용 학습 ~9분 + 평가/시각화)

---

## 학습 흐름

1. 🚀 **실습**: Ch 13의 데이터 + *별점 정규화 회귀 라벨* 추가. `AutoModelForSequenceClassification` (`num_labels=5`, multi-label) 에 `aux_head = Linear(H, 1)` 한 줄 추가, `Trainer.compute_loss` 오버라이드.
2. 🔬 **해부**: 메인 metric (per-label F1, hamming, AUC) + 보조 metric (RMSE, Pearson r) 동시 측정.
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 **λ=0 baseline** (= Ch 13 재현)을 학습한 뒤 λ=1 결과와 비교 — *보조 loss가 메인 task에 도움이 됐는가?* 라벨별 F1 차이로 시각화.

---

> 📒 **사전 학습 자료**: Ch 9 (BERT 회귀 — MSELoss), Ch 13 (BERT multi-label — BCE per-label). 이번 챕터는 둘을 한 모델 안에 같이 넣습니다.

> ⚠️ **이번 챕터에 *처음* 등장**: `Trainer.compute_loss` 오버라이드 — 자동 매핑이 다루지 못하는 *복합 loss* 를 위해 필요한 패턴 (CLAUDE.md 의 "자동 매핑을 못 쓸 때만 오버라이드" 규약).""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 9 | DistilBERT 파인튜닝 | WordPiece | Yelp 별점 | `Linear(H, 1)` | 없음 | `MSELoss` |
| 13 | DistilBERT 파인튜닝 | WordPiece | Yelp + 측면 합성 | `Linear(H, 5)` | sigmoid (각각) | `BCEWithLogitsLoss` |
| **14 ← 여기** | DistilBERT + **보조 헤드** | WordPiece | Yelp + 측면 + **별점** | **메인(5) + 보조(1)** | 메인 sigmoid + 보조 없음 | **`BCE per-label + λ·MSE`** |
| 15 (다음 Phase 2) | klue/bert-base | WordPiece (한국어) | NSMC | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 13)

| 축 | Ch 13 (multi-label) | Ch 14 (multi-label + auxiliary) |
|---|---|---|
| Task (메인) | 5라벨 multi-label | (그대로) |
| `num_labels` | 5 | (그대로) |
| `problem_type` | `multi_label_classification` | (그대로 — 자동 매핑은 메인 loss 만 처리) |
| 메인 활성화 / loss | per-label sigmoid / BCE | (그대로) |
| **보조 head** | 없음 | **새로 추가**: `Linear(H, 1)` linear regressor |
| **보조 라벨** | 없음 | **새로 추가**: 별점 정규화 0-1 (`label / 4`) |
| **보조 loss** | — | **`MSELoss`** (Ch 9 그대로) |
| **결합 loss** | `outputs.loss` 자동 | **`L_main + λ·L_aux`** 직접 계산 |
| `Trainer.compute_loss` | 자동 (오버라이드 X) | **오버라이드 필수** |
| 데이터 콜레이터 | `DataCollatorWithPadding` 자동 | **커스텀** — `aux_labels` 도 같이 batching |
| 학습 hyperparams | (epoch=2, lr=2e-5, …) | (그대로) |

> **변하는 축 — Loss 축 끝**: 메인 task와 모델 본체는 *완전히 동일*, *Loss에 보조 항이 가중합으로 추가* 됩니다 (CLAUDE.md "Loss 축: ... → BCE per-label → +Auxiliary"). 실무에서 *학습 데이터에 추가 신호가 있을 때* 이를 활용하는 정통 multi-task learning 패턴.

### 왜 보조 task가 메인 task에 도움이 되는가 (직관)

- BERT 본체(67M 파라미터)는 *공유* — 메인 분류 헤드와 보조 회귀 헤드가 같은 768-dim CLS 표현을 입력으로 받음.
- 보조 task가 *메인 task와 부분적으로 상관* 이 있는 신호라면 (예: 긍정 측면 ↔ 높은 별점), 보조 학습이 BERT 본체를 *더 일반적인* 표현으로 끌고 갑니다.
- 결과적으로 메인 task만 학습할 때보다 *덜 과적합* 되고 *측면 분류에도 도움* 이 됩니다.
- 단, 보조 task가 메인과 *상관 없거나 반대 신호* 면 오히려 학습을 *방해* 합니다. λ 선택이 중요한 이유.""")

# ----- 4. Loss 노트 -----
md(r"""## 📐 Loss 노트 — Combined loss `L = L_main + λ · L_aux`

$$L = \underbrace{\frac{1}{N \cdot K}\sum_{i,k}\text{BCE}(z_{i,k}^{main}, y_{i,k}^{main})}_{L_\text{main}: \text{측면 BCE per-label}} + \lambda \cdot \underbrace{\frac{1}{N}\sum_{i}(z_{i}^{aux} - y_{i}^{aux})^2}_{L_\text{aux}: \text{별점 MSE}}$$

- $z^\text{main} \in \mathbb{R}^5$ — 측면 logit 5개, sigmoid 후 BCE.
- $z^\text{aux} \in \mathbb{R}$ — 별점 회귀 logit (활성화 없음, 직접 MSE).
- $\lambda$ — 보조 loss의 가중치 (hyperparameter, 보통 0.1-10 범위 탐색).

**λ 선택 가이드 — 숫자로 감 잡기**

| λ | 의미 | 효과 (예상) |
|---|---|---|
| 0.0 | 보조 loss 무시 | Ch 13과 *완전히 동일* (이번 챕터 §5 baseline) |
| 0.1 | 보조 *약하게* 반영 | 메인이 압도적, 보조는 살짝 정규화 효과 |
| 1.0 | **보조와 메인 *동등* 가중** | 표준 시작점. 두 task가 균형 잡혀 학습 |
| 5.0 | 보조 우세 | 보조 task가 학습 방향을 지배 — 메인 task 성능 떨어질 수 있음 |
| 10+ | 보조 절대 우세 | 메인 task 신호가 거의 묻힘. 권장 안 됨 |

이번 챕터에선 **λ=1** 로 학습하고 λ=0 baseline과 비교. 실무에선 validation set에서 λ를 grid search (0.1, 0.3, 1, 3, 10).

**숫자로 감 잡기 (단일 샘플)** — 측면 multi-hot $\mathbf{y}^\text{main} = [1, 0, 1, 0, 1]$, 별점 정규화 $y^\text{aux} = 0.75$ (4★/4):

| 단계 | 값 |
|---|---|
| $L_\text{main}$ (Ch 13의 BCE per-label 평균) | 0.45 |
| $L_\text{aux}$ ($(z^\text{aux} - 0.75)^2$, 가정 $z^\text{aux} = 0.55$) | $(0.55 - 0.75)^2 = 0.04$ |
| $L$ (λ=1) | $0.45 + 1 \cdot 0.04 = 0.49$ |
| $L$ (λ=10) | $0.45 + 10 \cdot 0.04 = 0.85$ ← 보조 항이 메인보다 큼 |

λ가 너무 크면 *메인 신호 자체가 보조에 묻힙니다*. 반대로 너무 작으면 *보조 신호가 학습에 영향 안 줌*. λ=1이 균형의 기본 출발점.""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

Ch 13과 완전히 동일 — `distilbert-base-uncased` WordPiece, `max_length=128`. 토크나이저는 *라벨에 무관* 합니다. 보조 라벨이 추가됐다고 토큰화가 바뀌지는 않음 — 별점 정수가 *float 값으로 한 칸 더* 추가될 뿐.

> **다음 챕터(Ch 15)부터 Phase 2**: 같은 셋업을 한국어 BERT(`klue/bert-base`)에서 다시. 토크나이저가 *영어 WordPiece → 한국어 WordPiece* 로 바뀌는 게 변화의 본질.""")

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
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report,
    roc_auc_score, hamming_loss, mean_squared_error, r2_score,
)

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

# ----- 8. 데이터 + 측면 + 별점 -----
md(r"""## 1. 🚀 데이터 — Yelp + 측면 (Ch 13) + 별점 정규화

Ch 13의 측면 합성 라벨을 그대로 쓰고, *별점 정규화* 보조 라벨을 추가합니다.

- 메인 라벨 $\mathbf{y}^\text{main} \in \{0, 1\}^5$ — 측면 multi-hot.
- 보조 라벨 $y^\text{aux} = (\text{label} + 1 - 1) / 4 = \text{label} / 4 \in [0, 1]$ — 별점을 0-1로 정규화 (1★ → 0.0, 5★ → 1.0).""")

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


print(f"K (aspects): {K}, aspects: {ASPECTS}")""")

code(r"""tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("yelp_review_full")
train_full = ds["train"].shuffle(seed=42).select(range(5000))
eval_full  = ds["test"].shuffle(seed=42).select(range(1000))


def attach_aspects_and_aux(batch):
    batch["aspects"] = [extract_aspects(t) for t in batch["text"]]
    # label: 0-4 (Yelp 기본) → aux float [0, 1]
    batch["aux_score"] = [float(l) / 4.0 for l in batch["label"]]
    return batch


train_full = train_full.map(attach_aspects_and_aux, batched=True)
eval_full  = eval_full.map(attach_aspects_and_aux,  batched=True)

print(f"train: {len(train_full)}, eval: {len(eval_full)}")
print(f"\nFirst sample:")
print(f"  text: {train_full[0]['text'][:120]}...")
print(f"  aspects (multi-hot): {train_full[0]['aspects']}")
print(f"  aux_score (star/4): {train_full[0]['aux_score']:.2f}  (star = {train_full[0]['label'] + 1})")""")

code(r"""# 보조 라벨 분포 (별점 정규화)
import numpy as np
aux_train = np.array(train_full["aux_score"])
print(f"aux score range: [{aux_train.min():.2f}, {aux_train.max():.2f}]")
print(f"aux score mean: {aux_train.mean():.3f}, std: {aux_train.std():.3f}")
print(f"\nAux score distribution (train):")
for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
    cnt = (np.isclose(aux_train, v)).sum()
    star = int(v * 4) + 1
    print(f"  {v:.2f}  (star {star}): {cnt} samples ({cnt/len(aux_train):.1%})")""")

# ----- 9. 토큰화 (두 라벨 같이) -----
md(r"""## 2. 토큰화 — 메인 multi-hot + 보조 float 같이 부착

`tokenize_fn` 이 두 라벨을 모두 attach. 메인은 `labels` (multi-hot float), 보조는 `aux_labels` (float scalar).""")

code(r"""def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"]     = [list(map(float, a)) for a in batch["aspects"]]   # multi-hot 5차원
    out["aux_labels"] = [float(s) for s in batch["aux_score"]]            # float scalar
    return out


train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(
    ["text", "label", "aspects", "aux_score"]
)
eval_tok  = eval_full.map(tokenize_fn,  batched=True).remove_columns(
    ["text", "label", "aspects", "aux_score"]
)

print(train_tok)
print(f"\nFirst sample labels: {train_tok[0]['labels']}")
print(f"First sample aux_labels: {train_tok[0]['aux_labels']:.2f}")""")

# ----- 10. 커스텀 Collator -----
md(r"""## 3. 커스텀 Data Collator — `aux_labels` 도 batch에 같이 담기

기본 `DataCollatorWithPadding` 은 input_ids·attention_mask·labels 만 알고 있어 *추가 라벨* 은 통과시키지 못합니다. 한 줄짜리 wrapper로 `aux_labels` 를 텐서로 만들어 batch에 추가합니다.""")

code(r"""class AuxCollator:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        # 1. aux_labels 분리
        aux = torch.tensor([f.pop("aux_labels") for f in features], dtype=torch.float)
        # 2. 나머지(input_ids/attention_mask/labels)는 표준 padding
        batch = self.base(features)
        # 3. labels 가 multi-hot float 이므로 dtype 보정
        batch["labels"] = batch["labels"].float()
        # 4. aux 추가
        batch["aux_labels"] = aux
        return batch


collator = AuxCollator(tokenizer)
# 동작 확인 — 첫 4개 샘플로 batch 만들어 shape 보기
sample_features = [dict(train_tok[i]) for i in range(4)]
batch = collator(sample_features)
print("Batch keys:", list(batch.keys()))
for k, v in batch.items():
    print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")""")

# ----- 11. 모델 + aux head -----
md(r"""## 4. 모델 셋업 — Ch 13 모델 + 보조 헤드 한 줄 추가

`AutoModelForSequenceClassification` (Ch 13과 *완전히 동일*) 을 로드한 뒤 `model.aux_head = nn.Linear(...)` 한 줄로 보조 헤드를 *모델 객체에 attach*. 이후 `Trainer.compute_loss` 가 메인 출력 + 보조 헤드를 동시에 사용해 결합 loss 를 계산합니다.""")

code(r"""def make_model():
    m = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=K,
        problem_type="multi_label_classification",
        id2label={i: a for i, a in enumerate(ASPECTS)},
        label2id={a: i for i, a in enumerate(ASPECTS)},
    )
    # 보조 헤드: CLS hidden (768-dim) → scalar
    H = m.config.dim   # distilbert hidden size
    m.aux_head = nn.Linear(H, 1)
    # 보조 헤드 가중치도 같은 device로 옮겨야 함 (model.to() 가 알아서 처리)
    return m


model = make_model()

def param_summary(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    aux_only  = sum(p.numel() for n, p in m.named_parameters() if n.startswith("aux_head"))
    return total, trainable, aux_only


total, trainable, aux_only = param_summary(model)
print(f"Parameters:           {total:>13,}  ({total/1e6:.1f} M)")
print(f"Trainable parameters: {trainable:>13,}  ({trainable/total:.1%})")
print(f"Aux head parameters:  {aux_only:>13,}  ({aux_only/total:.4%})")
print(f"Main classifier:      {model.classifier}")
print(f"Aux head:             {model.aux_head}")""")

md(r"""**보조 헤드는 ~770개 파라미터** — 768→1 Linear의 weight + bias. 전체 67M 의 *0.001%*. 이 *미세한 추가 자유도* 만으로 멀티태스크 학습이 동작합니다.""")

# ----- 12. 커스텀 Trainer -----
md(r"""## 5. 커스텀 Trainer — `compute_loss` 오버라이드

핵심 로직 (코드 한 줄로 요약):

```python
loss = outputs.loss + λ · MSE(aux_head(CLS), aux_labels)
```

- `outputs.loss` 는 `problem_type="multi_label_classification"` 자동 매핑으로 이미 BCE per-label 평균이 계산됨.
- 보조 loss는 우리가 *직접 계산* — `output_hidden_states=True` 로 받은 마지막 layer의 CLS 표현을 `aux_head` 에 통과.""")

code(r"""from transformers.modeling_outputs import SequenceClassifierOutput


class AuxTrainer(Trainer):
    def __init__(self, *args, lambda_aux: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_aux = lambda_aux

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        aux_labels = inputs.pop("aux_labels")
        # output_hidden_states=True 로 BERT 마지막 layer hidden 까지 받기
        outputs = model(**inputs, output_hidden_states=True)
        main_loss = outputs.loss   # BCE per-label (자동 매핑)

        # 마지막 layer CLS hidden → aux_head → scalar
        cls = outputs.hidden_states[-1][:, 0, :]   # (B, 768)
        aux_logits = model.aux_head(cls).squeeze(-1)   # (B,)
        aux_loss = F.mse_loss(aux_logits, aux_labels.float())

        loss = main_loss + self.lambda_aux * aux_loss

        if return_outputs:
            # 평가 단계에서 Trainer 가 outputs.hidden_states/attentions 를 prediction 로 모아
            # tuple 로 반환하거나 메모리 폭주를 일으키는 걸 방지 — logits 만 가진 깔끔한
            # SequenceClassifierOutput 으로 교체해서 돌려줌.
            clean = SequenceClassifierOutput(loss=loss, logits=outputs.logits)
            return (loss, clean)
        return loss


print("AuxTrainer 정의 완료 — Trainer 의 compute_loss 만 교체.")""")

# ----- 13. compute_metrics -----
md(r"""**평가용 metric 함수** — 메인 (Ch 13과 동일) + 보조 (RMSE, R², Pearson r). 보조 logit 추출은 `Trainer.predict()` 가 메인 logits 만 반환하기 때문에 별도 단계로 빼서 처리.""")

code(r"""def compute_metrics_main(eval_pred):
    # 메인 task 평가 — Ch 13과 동일
    logits, labels = eval_pred
    # 방어적 처리: Trainer 가 hidden_states 까지 collected 하면 logits 가 tuple 이 됨.
    # AuxTrainer.compute_loss 가 clean output 으로 막지만 안전장치로 한 번 더.
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    out = {"hamming_loss": float(hamming_loss(labels, preds))}
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0,
    )
    out["micro_f1"] = float(f1_mi)
    out["micro_precision"] = float(p_mi)
    out["micro_recall"]    = float(r_mi)
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0,
    )
    out["macro_f1"] = float(f1_ma)
    out["macro_precision"] = float(p_ma)
    out["macro_recall"]    = float(r_ma)
    try:
        out["macro_auc"] = float(roc_auc_score(labels, probs, average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    return out""")

# ----- 14. 학습 (λ=1) -----
md(r"""## 6. 학습 — λ=1 (보조 ON)

Ch 13과 동일한 hyperparams. `AuxTrainer` + `lambda_aux=1.0`.""")

code(r"""LAMBDA_AUX = 1.0

training_args = TrainingArguments(
    output_dir="./ch14_aux_output",
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
    remove_unused_columns=False,   # ← aux_labels 가 model.forward 시그니처에 없어 자동 제거되는 걸 방지
)

trainer_aux = AuxTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics_main,
    lambda_aux=LAMBDA_AUX,
)

train_result_aux = trainer_aux.train()
print(f"\nWith-aux training done — mean train loss: {train_result_aux.training_loss:.4f}")""")

md(r"""**중요: `remove_unused_columns=False`** — Trainer는 기본으로 *model.forward 시그니처에 없는 컬럼* 을 제거합니다. `aux_labels` 는 모델 시그니처에 없어 자동 제거되면 우리 `compute_loss` 가 받을 수 없습니다. 이 옵션을 꺼야 함.""")

code(r"""!nvidia-smi""")

# ----- 15. 평가 (메인 + 보조) -----
md(r"""## 7. 🔬 평가 — 메인 task + 보조 task

메인 metric 은 자동으로 계산됨 (`compute_metrics`). 보조 metric (RMSE, R², Pearson r) 은 별도 forward로 보조 logits 를 추출해 측정.""")

code(r"""# 메인 metric
eval_metrics_aux = trainer_aux.evaluate()
print("With-aux (λ=1) — main task metrics:")
for k, v in eval_metrics_aux.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")""")

code(r"""# 보조 metric — eval 전체에 대해 수동 forward (작아서 빠름)
@torch.no_grad()
def aux_predictions(trainer, dataset, batch_size=64):
    trainer.model.eval()
    device = trainer.model.device
    aux_preds, aux_true = [], []
    for i in range(0, len(dataset), batch_size):
        batch_features = [dict(dataset[j]) for j in range(i, min(i + batch_size, len(dataset)))]
        batch = trainer.data_collator(batch_features)
        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        aux_lbl = batch_on_device.pop("aux_labels").cpu().numpy()
        out = trainer.model(**{k: v for k, v in batch_on_device.items() if k != "labels"},
                            output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0, :]
        aux_logits = trainer.model.aux_head(cls).squeeze(-1).cpu().numpy()
        aux_preds.extend(aux_logits.tolist())
        aux_true.extend(aux_lbl.tolist())
    return np.array(aux_preds), np.array(aux_true)


aux_preds_aux, aux_true = aux_predictions(trainer_aux, eval_tok)
rmse_aux = float(np.sqrt(mean_squared_error(aux_true, aux_preds_aux)))
r2_aux   = float(r2_score(aux_true, aux_preds_aux))
pear_aux = float(np.corrcoef(aux_true, aux_preds_aux)[0, 1])

print("\nWith-aux (λ=1) — aux task metrics (star regression, normalized 0-1):")
print(f"  RMSE:    {rmse_aux:.4f}")
print(f"  R^2:     {r2_aux:.4f}")
print(f"  Pearson: {pear_aux:.4f}")""")

# ----- 16. 메인 logit + 메인 평가 데이터 보존 -----
code(r"""# 메인 task per-sample 예측 (다음 비교 단계에서 사용)
preds_output_aux = trainer_aux.predict(eval_tok)
logits_aux = preds_output_aux.predictions
if isinstance(logits_aux, tuple):
    logits_aux = logits_aux[0]
labels_eval = preds_output_aux.label_ids.astype(int)
probs_aux = 1.0 / (1.0 + np.exp(-logits_aux))
preds_main_aux = (probs_aux >= 0.5).astype(int)

print(f"Main logits shape: {logits_aux.shape}")
print(f"Eval samples:      {len(labels_eval)}")""")

# ----- 17. §5 클라이맥스 — λ=0 baseline -----
md(r"""## 8. 🛠️ 클라이맥스 — *λ=0 baseline* 학습 (= Ch 13 재현)

같은 코드를 `lambda_aux=0.0` 으로 한 번 더 돌립니다. 그러면 보조 loss 의 gradient 가 0이 되어 메인 task만 학습되는 상태 = **Ch 13과 정확히 동일한 학습 결과**. (보조 헤드는 학습되긴 하지만 메인 학습엔 영향 없음.)

> 의도적으로 *Ch 13 노트북을 따로 돌리지 않고* 이 셀에서 baseline을 다시 만듭니다 — 비교가 *같은 노트북·같은 환경* 안에서 self-contained 하도록.""")

code(r"""# 새 모델 인스턴스 — λ=0 학습용
model_no_aux = make_model()

training_args_no_aux = TrainingArguments(
    output_dir="./ch14_baseline_output",
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
    remove_unused_columns=False,
)

trainer_no_aux = AuxTrainer(
    model=model_no_aux,
    args=training_args_no_aux,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics_main,
    lambda_aux=0.0,    # ← 보조 loss 무시
)

train_result_no_aux = trainer_no_aux.train()
print(f"\nNo-aux (λ=0) baseline training done — mean train loss: {train_result_no_aux.training_loss:.4f}")""")

code(r"""# baseline 메인 metric
eval_metrics_no_aux = trainer_no_aux.evaluate()
print("No-aux (λ=0) baseline — main task metrics:")
for k, v in eval_metrics_no_aux.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")

# baseline 메인 per-sample 예측
preds_output_no_aux = trainer_no_aux.predict(eval_tok)
logits_no_aux = preds_output_no_aux.predictions
if isinstance(logits_no_aux, tuple):
    logits_no_aux = logits_no_aux[0]
probs_no_aux = 1.0 / (1.0 + np.exp(-logits_no_aux))
preds_main_no_aux = (probs_no_aux >= 0.5).astype(int)""")

# ----- 18. 비교 시각화 -----
md(r"""### 8-1. 메인 metric 비교 — λ=0 baseline vs λ=1 aux""")

code(r"""m_aux    = {k.replace("eval_", ""): v for k, v in eval_metrics_aux.items()
            if k.startswith("eval_") and isinstance(v, float)}
m_no_aux = {k.replace("eval_", ""): v for k, v in eval_metrics_no_aux.items()
            if k.startswith("eval_") and isinstance(v, float)}

common = [k for k in m_aux if k in m_no_aux]
cmp = pd.DataFrame({
    "metric":             common,
    "no aux (lambda=0)":  [m_no_aux[k] for k in common],
    "with aux (lambda=1)":[m_aux[k]    for k in common],
})
cmp["delta (aux - no_aux)"] = cmp["with aux (lambda=1)"] - cmp["no aux (lambda=0)"]
print(cmp.round(4).to_string(index=False))""")

md(r"""**해석 가이드**

- `delta` > 0 — 보조 loss 가 메인 task 에 *도움* 이 됨 (멀티태스크의 정통 효과).
- `delta` < 0 — 보조 loss 가 메인 task 를 *방해* 함 (λ가 너무 큼 / 보조 task 가 메인과 상관 약함).
- `delta` ≈ 0 — 별 영향 없음 (보조 신호가 메인 표현에 의미 없는 추가).

별점은 측면 분포와 *부분적으로* 상관 (긍정 측면 → 높은 별점) 이라 *작은 양의 delta* 가 자연스러운 결과. 0.5%p 정도면 노이즈일 수 있고, 1-2%p 면 의미 있는 효과.""")

md(r"""### 8-2. 라벨별 F1 비교 — 어느 측면이 보조 loss로 가장 도움받았나""")

code(r"""def per_label_f1(Y_true, Y_pred):
    f1s = []
    for k in range(K):
        _, _, f1, _ = precision_recall_fscore_support(
            Y_true[:, k], Y_pred[:, k], average="binary", zero_division=0,
        )
        f1s.append(float(f1))
    return f1s


f1_no_aux = per_label_f1(labels_eval, preds_main_no_aux)
f1_aux    = per_label_f1(labels_eval, preds_main_aux)

label_cmp = pd.DataFrame({
    "aspect":              ASPECTS,
    "no aux F1":           f1_no_aux,
    "with aux F1":         f1_aux,
    "delta (aux - no_aux)": np.array(f1_aux) - np.array(f1_no_aux),
})
print(label_cmp.round(4).to_string(index=False))

# 막대 그래프
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(K)
width = 0.38
ax.bar(x_pos - width/2, f1_no_aux, width, label="no aux (lambda=0)",  color="#5B8DEF")
ax.bar(x_pos + width/2, f1_aux,    width, label="with aux (lambda=1)",color="#F47272")
ax.set_xticks(x_pos)
ax.set_xticklabels(ASPECTS)
ax.set_ylim(0, 1)
ax.set_ylabel("Per-label F1")
ax.set_title("Per-label F1 — auxiliary loss effect")
ax.legend()
plt.tight_layout()
plt.show()""")

md(r"""**해석**

- **별점과 상관이 강한 측면** (food, service): 보조 별점 회귀 학습이 *긍정/부정 신호* 를 잘 잡으면 도움이 됩니다. 작은 양의 delta 기대.
- **별점과 상관이 약한 측면** (location, price): 별점 신호가 *직접적 도움* 이 안 됨. delta가 0 근처거나 약간 음수일 수 있음.
- **분산이 큰 라벨** — eval 표본이 적어 F1 자체가 노이즈가 큼. delta 도 의미 해석 조심.""")

# ----- 19. 보조 task 자체 평가 -----
md(r"""### 8-3. 보조 task 자체는 얼마나 잘 학습됐나

별점 회귀가 잘 됐다는 건 BERT 본체가 *별점 신호도 효율적으로 인코딩* 하고 있다는 뜻 — 메인 task 표현에도 그 신호가 들어가 있을 가능성.""")

code(r"""fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(
    x=aux_true, y=aux_preds_aux, ax=ax,
    color="#F47272", alpha=0.55, s=35,
)
ax.plot([0, 1], [0, 1], color="black", lw=1.3, ls="--", alpha=0.7,
        label="y = x (perfect)")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel("True normalized star  (0=1*, 1=5*)")
ax.set_ylabel("Predicted by aux head")
ax.set_title(f"Aux task — star regression (RMSE={rmse_aux:.3f}, r={pear_aux:.3f})")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()""")

md(r"""**해석**

- 점들이 $y=x$ 직선 *주변에 몰려 있으면* 보조 head 가 별점 신호를 잘 학습한 것.
- 0/0.25/0.5/0.75/1.0 5개 *세로 띠* 가 나타남 (정답이 5개 별점 정수에서만 나오므로). 각 띠 안에서 예측이 분산되는 정도가 *그 별점에서의 모델 불확실성*.
- 점들이 직선과 멀어지면 보조 head 학습 부족 — λ를 더 크게 두거나 학습량을 늘려야 함.""")

# ----- 20. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `Trainer.compute_loss` 오버라이드 | 자동 매핑이 못 다루는 *복합 loss* 직접 계산 | Ch 18 한국어 auxiliary 에서 다시 |
| `model.aux_head = nn.Linear(...)` 한 줄 추가 | 표준 BERT 모델에 보조 헤드 동적 부착 | Ch 18 |
| `output_hidden_states=True` | 마지막 layer hidden 까지 받아 보조 헤드 입력으로 사용 | 보조 헤드 패턴마다 |
| `remove_unused_columns=False` | model.forward 시그니처에 없는 컬럼(aux_labels)을 자동 제거 안 함 | custom collator 패턴마다 |
| 커스텀 `DataCollator` | input_ids 외 *추가 라벨* 도 batch에 같이 담기 | Ch 18 |
| `r2_score`, `np.corrcoef` | 회귀 task 보조 metric (R², Pearson r) | 회귀 결합 task 마다 |""")

# ----- 21. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. λ를 0.1에서 10까지 grid search한다고 할 때 *작은 λ* 와 *큰 λ* 가 각각 어떤 학습 양상을 만드는지 한 줄로 설명하세요.
2. `remove_unused_columns=False` 를 빠뜨리면 어떤 에러가 나나요? `aux_labels` 가 어디로 사라지는지 추적해 보세요.
3. 보조 헤드의 파라미터가 ~770개에 불과한데도 모델이 *공유 본체* 학습에 영향을 주는 이유는?
4. 메인 metric의 delta가 *음수* 로 나왔다면 무엇을 의심해야 하고 어떻게 처치하나요?""")

# ----- 22. FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) λ는 어떻게 정하나요? 그냥 1.0으로 두면 되나요?

1.0은 *시작점* 일 뿐 거의 항상 grid search가 필요합니다.

```python
# 권장 grid search (validation set 위에서 수행)
for lam in [0.1, 0.3, 1.0, 3.0, 10.0]:
    trainer = AuxTrainer(..., lambda_aux=lam)
    trainer.train()
    metrics = trainer.evaluate()
    print(f"lambda={lam}: macro_f1={metrics['eval_macro_f1']:.4f}")
```

흔한 패턴:
- 메인 task가 분류, 보조가 회귀 → λ=0.1-1.0 (회귀 MSE는 보통 분류 BCE 보다 작아 균형이 자연스러움)
- 메인이 복잡하고 보조가 *미세* 보조 → λ=0.01-0.1
- 메인과 보조가 비슷한 중요도 → λ=1.0

또 다른 패턴: **uncertainty weighting** — λ를 *학습 가능한 파라미터* 로 두고 모델이 직접 결정하게 함 (Kendall et al. 2018).

### Q2. (이론) `outputs.loss` 는 BCE per-label 평균인데 왜 *우리가 따로* 계산 안 하나요?

`AutoModelForSequenceClassification` 의 forward 가 `problem_type="multi_label_classification"` 와 `labels` 를 보고 *자동으로* `BCEWithLogitsLoss(logits, labels)` 를 계산해 `outputs.loss` 에 담기 때문입니다.

```python
# transformers 내부 (개념적)
class DistilBertForSequenceClassification:
    def forward(self, ..., labels=None):
        logits = self.classifier(self.pre_classifier(hidden))
        loss = None
        if labels is not None:
            if self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits, labels)
            elif self.config.problem_type == "regression":
                loss = F.mse_loss(logits.squeeze(), labels.float())
        return SequenceClassifierOutput(loss=loss, logits=logits, ...)
```

우리는 *메인 loss는 자동으로 받고, 보조 loss만 직접 계산해 더하면* 됩니다. 깔끔.

### Q3. (실무) 보조 헤드의 가중치는 사전학습이 안 되어 있는데 그래도 잘 학습되나요?

네. 이유:

1. *작은 분류 헤드* 는 보통 사전학습 없이 random init 부터 시작해도 fine-tune 동안 충분히 빠르게 학습됩니다 (몇백 step 안에).
2. 보조 헤드 위에 가중치가 ~770개라 데이터에 *과적합* 할 위험도 적습니다.
3. BERT 본체(67M)는 이미 표상이 풍부해 작은 head가 *읽어내기만* 하면 됨.

대조적으로 *BERT 본체를 random init 부터* 학습하려면 수백 GB 데이터가 필요합니다. 사전학습 + 작은 head fine-tune 패턴이 강력한 이유.

### Q4. (이론) 보조 task가 메인과 *반대 방향* 신호면 학습이 망가지나요?

네, 정확히 그렇습니다. 예시:

- 메인: 측면 분류 (긍정 측면 vs 부정 측면)
- 보조: *반대로* 라벨링된 별점 (1★=좋음, 5★=나쁨 — 라벨링 실수 시나리오)

이 경우 두 task의 gradient가 BERT 본체에서 *반대 방향* 으로 끌어당겨 학습이 *발산* 하거나 *느려집니다*. 진단 신호:

- 두 task의 train loss 가 *둘 다 정체* (서로 상쇄)
- λ 를 키울수록 메인 metric 이 *떨어짐*

해결: 보조 task가 메인과 같은 방향 신호인지 *간단한 sklearn baseline* 부터 확인. 별점이 측면 분류와 양의 상관이면 보조로 쓸 만함.

### Q5. (실무) 보조 task로 어떤 신호를 쓰는 게 좋나요?

좋은 보조 task의 조건:

| 좋은 조건 | 예시 |
|---|---|
| *메인과 양의 상관* | 측면 분류 ↔ 별점, 감성 분류 ↔ 이모지 사용 |
| *학습 데이터가 *공짜로* 있음* | 메타데이터(별점, 작성일, 길이) — 라벨링 비용 0 |
| *연속적이고 안정적인 신호* | float regression, 순서형 점수 |
| *메인보다 *덜 복잡* 한 task* | 회귀 < 분류, 단어 vs 문장 |

피해야 할 보조:

- 메인보다 *복잡한* task (예: 분류에 *생성* 보조) — 학습 비용 폭증
- *드물게 등장하는* 라벨 — 학습 신호 부족
- *중복* 라벨 (메인과 거의 같은 정보) — 추가 정보 없음

### Q6. (실무) 보조 task가 *test 환경에서는 정답이 없는* 라벨이면 어떻게 활용하나요?

학습 시에만 보조 라벨 사용, 추론 시에는 메인 head만 사용 — *학습 정규화* 용. 이게 auxiliary loss 의 *정통 사용 패턴* 입니다.

```python
# 학습 시: 별점이 학습 데이터에 있음 → 보조 학습
trainer.train()   # main + aux 둘 다 학습

# 추론 시: 별점 없음 → 메인 head만 사용
with torch.no_grad():
    out = model(input_ids=..., output_hidden_states=False)
    main_probs = torch.sigmoid(out.logits)   # 측면 예측만
    # aux_head 는 *호출 안 함*
```

훈련 데이터에만 있는 *부가 신호* (운영 시엔 사라지는 메타데이터)를 학습 시 활용하는 좋은 트릭.""")

# ----- 23. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# 1. remove_unused_columns 를 True (기본값) 로 두기
training_args = TrainingArguments(
    ...,
    remove_unused_columns=True,   # ← 잘못 (default 가 True 임)
)
trainer = AuxTrainer(...)
trainer.train()
```

힌트: Trainer 가 `aux_labels` 를 *자동 제거* 한 뒤 우리 `compute_loss` 안에서 `inputs.pop("aux_labels")` 가 KeyError. `aux_labels` 가 어디서 사라지는지 추적해 보면 학습 inputs 가 model.forward 시그니처와 맞춰지는 구조를 이해하게 됩니다.""")

# ----- 24. next -----
md(r"""## 다음 챕터 예고 — Phase 2 시작

**Chapter 15. BERT 한국어 Binary — NSMC**

- Phase 1 영어(DistilBERT) → Phase 2 한국어(klue/bert-base) 전환
- 데이터: 네이버 영화 리뷰 (NSMC) 이진 분류 (긍정/부정)
- 셋업: `num_labels=2` + `problem_type="single_label_classification"` (Ch 11과 같은 표준 binary 셋업)
- 변하는 축: *언어 + 데이터 + 토크나이저* (영어 WordPiece → 한국어 WordPiece). 모델 크기는 비슷, 셋업도 같음.
- 회귀 챕터는 *생략* (영어 Ch 9 에서 다뤘으므로). Phase 2는 Binary 부터 시작.

> **Phase 1 마무리** — Ch 7-14를 통해 BERT 분류·회귀·multi-label·auxiliary loss의 기본 5가지를 다 익혔습니다. Phase 2는 같은 패턴을 *한국어 데이터* 위에서 압축적으로 재방문 — 토크나이저가 어떻게 다른지, 한국어 특유의 학습 어려움이 무엇인지 확인.""")


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
