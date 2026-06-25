> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/14_auxiliary_loss/14_auxiliary_loss.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q transformers datasets
```

```python
import warnings
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

# matplotlib 한글 폰트 (Colab — NanumGothic). plot 의 한국어가 □ 로 깨지지 않게.
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, subprocess, os
_fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(_fp):
    subprocess.run("apt-get -qq -y install fonts-nanum", shell=True)
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CPU runtime — training will be very slow. Switch to T4 recommended.")
```

**▶ 실행 결과**

```text
PyTorch:        2.11.0+cu128
CUDA available: True
GPU:             Tesla T4
```

**baseline VRAM**:

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 24 04:13:57 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   48C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## 데이터 — Yelp + 항목 (Ch 13) + 별점 보조 라벨

Ch 13의 항목 합성 라벨을 그대로 쓰고, **별점 보조 회귀 라벨** 을 추가합니다. 별점은 1-5 정수지만 회귀 헤드와 MSE를 자연스럽게 쓰기 위해 *0-1 스케일* 로 변환만 해 둡니다 (학습 정규화 효과를 위한 데이터 가공이 아니라, 그냥 단위만 맞추는 작업).

- 메인 라벨 $\mathbf{y}^\text{main} \in \{0, 1\}^5$ — 항목 multi-hot.
- 보조 라벨 $y^\text{aux} = \text{label} / 4 \in [0, 1]$ — 1★ → 0.0, 5★ → 1.0.

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


def extract_aspects(text: str) -> list[float]:
    text_lower = text.lower()
    return [
        float(any(re.search(rf"\b{re.escape(kw)}\b", text_lower) for kw in keywords))
        for keywords in ASPECT_KEYWORDS.values()
    ]


print(f"K (aspects): {K}, aspects: {ASPECTS}")
```

**▶ 실행 결과**

```text
K (aspects): 5, aspects: ['food', 'service', 'price', 'ambiance', 'location']
```

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("Yelp/yelp_review_full")
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
print(f"  aux_score (star/4): {train_full[0]['aux_score']:.2f}  (star = {train_full[0]['label'] + 1})")
```

**▶ 실행 결과**

```text
train: 5000, eval: 1000

First sample:
  text: I stalk this truck.  I've been to industrial parks where I pretend to be a tech worker standing in line, strip mall park...
  aspects (multi-hot): [0.0, 1.0, 0.0, 0.0, 1.0]
  aux_score (star/4): 1.00  (star = 5)
```

```python
# 보조 라벨 분포 (별점 0-1 스케일)
import numpy as np
aux_train = np.array(train_full["aux_score"])
print(f"aux score range: [{aux_train.min():.2f}, {aux_train.max():.2f}]")
print(f"aux score mean: {aux_train.mean():.3f}, std: {aux_train.std():.3f}")
print(f"\nAux score distribution (train):")
for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
    cnt = (np.isclose(aux_train, v)).sum()
    star = int(v * 4) + 1
    print(f"  {v:.2f}  (star {star}): {cnt} samples ({cnt/len(aux_train):.1%})")
```

**▶ 실행 결과**

```text
aux score range: [0.00, 1.00]
aux score mean: 0.495, std: 0.354

Aux score distribution (train):
  0.00  (star 1): 1017 samples (20.3%)
  0.25  (star 2): 1027 samples (20.5%)
  0.50  (star 3): 960 samples (19.2%)
  0.75  (star 4): 1021 samples (20.4%)
  1.00  (star 5): 975 samples (19.5%)
```

## 토큰화 — 메인 multi-hot + 보조 float 같이 부착

`tokenize_fn` 이 두 라벨을 모두 attach. 메인은 `labels` (multi-hot float), 보조는 `aux_labels` (float scalar).

```python
def tokenize_fn(batch):
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
print(f"First sample aux_labels: {train_tok[0]['aux_labels']:.2f}")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'aux_labels'],
    num_rows: 5000
})

First sample labels: [0.0, 1.0, 0.0, 0.0, 1.0]
First sample aux_labels: 1.00
```

## 커스텀 Data Collator — `aux_labels` 도 batch에 같이 담기

기본 `DataCollatorWithPadding` 은 input_ids·attention_mask·labels 만 알고 있어 *추가 라벨* 은 통과시키지 못합니다. 한 줄짜리 wrapper로 `aux_labels` 를 텐서로 만들어 batch에 추가합니다.

```python
class AuxCollator:
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
    print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
```

**▶ 실행 결과**

```text
Batch keys: ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'aux_labels']
  input_ids: shape=(4, 128), dtype=torch.int64
  token_type_ids: shape=(4, 128), dtype=torch.int64
  attention_mask: shape=(4, 128), dtype=torch.int64
  labels: shape=(4, 5), dtype=torch.float32
  aux_labels: shape=(4,), dtype=torch.float32
```

## 모델 셋업 — Ch 13 모델 + 보조 헤드 한 줄 추가

`AutoModelForSequenceClassification` (Ch 13과 *완전히 동일*) 을 로드한 뒤 `model.aux_head = nn.Linear(...)` 한 줄로 보조 헤드를 *모델 객체에 attach*. 이후 `Trainer.compute_loss` 가 메인 출력 + 보조 헤드를 동시에 사용해 결합 loss 를 계산합니다.

```python
def make_model():
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


torch.manual_seed(42); np.random.seed(42)   # baseline 과 동일 초기화 — λ 만 변수가 되도록
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
print(f"Aux head:             {model.aux_head}")
```

**▶ 실행 결과**

```text
[transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
Key                     | Status     | 
------------------------+------------+-
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
classifier.weight       | MISSING    | 
pre_classifier.bias     | MISSING    | 
pre_classifier.weight   | MISSING    | 
classifier.bias         | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:              66,958,086  (67.0 M)
Trainable parameters:    66,958,086  (100.0%)
Aux head parameters:            769  (0.0011%)
Main classifier:      Linear(in_features=768, out_features=5, bias=True)
Aux head:             Linear(in_features=768, out_features=1, bias=True)
```

**보조 헤드는 ~770개 파라미터** — 768→1 Linear의 weight + bias. 전체 67M 의 *0.001%*. 이 *미세한 추가 자유도* 만으로 멀티태스크 학습이 동작합니다.

## 커스텀 Trainer — `compute_loss` 오버라이드

핵심 로직 (코드 한 줄로 요약):

```python
loss = outputs.loss + λ · MSE(aux_head(CLS), aux_labels)
```

- `outputs.loss` 는 `problem_type="multi_label_classification"` 자동 매핑으로 이미 BCE per-label 평균이 계산됨.
- 보조 loss는 우리가 *직접 계산* — `output_hidden_states=True` 로 받은 마지막 layer의 CLS 표현을 `aux_head` 에 통과.

```python
from transformers.modeling_outputs import SequenceClassifierOutput


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


print("AuxTrainer 정의 완료 — Trainer 의 compute_loss 만 교체.")
```

**▶ 실행 결과**

```text
AuxTrainer 정의 완료 — Trainer 의 compute_loss 만 교체.
```

**평가용 metric 함수** — 메인 (Ch 13과 동일) + 보조 (RMSE, R², Pearson r). 보조 logit 추출은 `Trainer.predict()` 가 메인 logits 만 반환하기 때문에 별도 단계로 빼서 처리.

```python
def compute_metrics_main(eval_pred):
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
    return out
```

## 학습 — λ=0.05 (sweet spot, 보조 ON)

Ch 13과 동일한 hyperparams. `AuxTrainer` + `lambda_aux=0.05`. 이 값은 부록 `14_auxiliary_loss_lambda_sweep` 의 λ 스윕에서 **메인 F1 을 가장 끌어올린 지점** 입니다 (λ 를 키우면 §9 곡선처럼 메인이 무너집니다).

```python
LAMBDA_AUX = 0.05

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
print(f"\nWith-aux training done — mean train loss: {train_result_aux.training_loss:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
With-aux training done — mean train loss: 0.4046
```

**중요: `remove_unused_columns=False`** — Trainer는 기본으로 *model.forward 시그니처에 없는 컬럼* 을 제거합니다. `aux_labels` 는 모델 시그니처에 없어 자동 제거되면 우리 `compute_loss` 가 받을 수 없습니다. 이 옵션을 꺼야 함.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 24 04:15:10 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   67C    P0             35W /   70W |    1619MiB /  15360MiB |     70%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1097      C   /usr/bin/python3                       1616MiB |
+-----------------------------------------------------------------------------------------+
```
