> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/11_bert_binary_softmax/11_bert_binary_softmax.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q transformers datasets
```

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score,
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
Mon Jun 22 03:44:42 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   43C    P8             11W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 데이터 — Yelp 이진화 (Ch 10과 정확히 동일)

같은 seed, 같은 5,000/1,000 샘플, 같은 별점 3 제외 + 4-5 → 1, 1-2 → 0 룰. **마지막 비교가 의미를 가지려면 데이터가 동일해야 합니다.**

Ch 10과 동일한 Yelp 데이터를 같은 seed로 불러와 별점 3을 제외하고 이진화합니다. 단 하나 달라지는 곳은 라벨을 `[float(b)]` 벡터가 아니라 `int` 스칼라로 둔다는 점입니다 — 방식 B의 `CrossEntropyLoss`가 정수 인덱스 라벨을 받기 때문입니다.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("Yelp/yelp_review_full")
train_full = ds["train"].shuffle(seed=42).select(range(5000))
eval_full  = ds["test"].shuffle(seed=42).select(range(1000))

# 별점 3 제외 + 이진화
def add_binary(batch):
    bins = []
    for lbl in batch["label"]:
        star = lbl + 1
        bins.append(1 if star >= 4 else 0)   # ← Ch 10과 다른 점: int (스칼라)
    batch["binary"] = bins
    return batch

train_bin = train_full.filter(lambda x: (x["label"] + 1) != 3).map(add_binary, batched=True)
eval_bin  = eval_full.filter(lambda x:  (x["label"] + 1) != 3).map(add_binary, batched=True)

print(f"train (after excluding 3-star): {len(train_bin)}")
print(f"eval  (after excluding 3-star): {len(eval_bin)}")
print(f"train positive rate: {sum(train_bin['binary']) / len(train_bin):.1%}")
```

**▶ 실행 결과**

```text
train (after excluding 3-star): 4040
eval  (after excluding 3-star): 804
train positive rate: 49.4%
```

**Ch 10과의 한 줄 차이**: `out["labels"] = [[float(b)] for b in batch["binary"]]` → `out["labels"] = [int(b) for b in batch["binary"]]`. 라벨이 *길이 1짜리 float 리스트* 가 아니라 *int 스칼라* 입니다.

토큰화하면서 라벨을 데이터셋에 심습니다. 방식 B의 핵심은 `out["labels"]`를 `int` 스칼라로 둔다는 점 — Ch 10(방식 A)이 `[float(b)]` 길이-1 벡터를 넘긴 것과 대비됩니다. 이 한 줄이 `Trainer`가 자동으로 `CrossEntropyLoss`를 고르도록 만드는 데이터 측 신호입니다.

```python
def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(b) for b in batch["binary"]]   # ← int 스칼라 (Ch 10은 [float(b)])
    return out

train_tok = train_bin.map(tokenize_fn, batched=True).remove_columns(["text", "label", "binary"])
eval_tok  = eval_bin.map(tokenize_fn,  batched=True).remove_columns(["text", "label", "binary"])

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (int scalar)")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 4040
})

First sample label: 1  (int scalar)
```

## 모델 로드 — 방식 B 셋업

`num_labels=2` + `problem_type="single_label_classification"` (BERT 분류의 *기본값*).

방식 B의 모델 셋업입니다. `num_labels=2`로 출력 헤드를 2차원으로 두고, `problem_type="single_label_classification"`을 명시해 `Trainer`가 `CrossEntropyLoss`를 쓰도록 합니다 — BERT 분류의 표준 셋업입니다.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    problem_type="single_label_classification",   # ← CrossEntropyLoss 자동 매핑 (BERT 분류 표준)
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)
```

**위 코드 읽기** — `num_labels=2`라 분류 헤드가 `Linear(768→2)`로 잡혀 logit이 `(B, 2)` 형태로 나옵니다 (방식 A는 `Linear(768→1)`). `problem_type="single_label_classification"`이 softmax + `CrossEntropyLoss` 경로를 고르는 스위치이고, `id2label` / `label2id`는 추론 시 사람이 읽는 라벨 이름을 config에 등록해 둡니다.

```python
def param_summary(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

total, trainable = param_summary(model)
print(f"Parameters:           {total:>13,}  ({total/1e6:.1f} M)")
print(f"Trainable parameters: {trainable:>13,}  ({trainable/total:.1%})")
print(f"Classifier:           {model.classifier}")
print(f"problem_type:         {model.config.problem_type}")
print(f"id2label:             {model.config.id2label}")
```

**▶ 실행 결과**

```text
[transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
Key                     | Status     | 
------------------------+------------+-
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
classifier.bias         | MISSING    | 
pre_classifier.bias     | MISSING    | 
pre_classifier.weight   | MISSING    | 
classifier.weight       | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:              66,955,010  (67.0 M)
Trainable parameters:    66,955,010  (100.0%)
Classifier:           Linear(in_features=768, out_features=2, bias=True)
problem_type:         single_label_classification
id2label:             {0: 'negative', 1: 'positive'}
```

**파라미터 수 비교 — 방식 A vs 방식 B**

| 부분 | 방식 A (`num_labels=1`) | 방식 B (`num_labels=2`) |
|---|---|---|
| DistilBERT body | 66,362,880 | 66,362,880 |
| pre_classifier (`Linear(768→768)`) | 590,592 | 590,592 |
| classifier (`Linear(768→K)`) | **769** (=768+1) | **1,538** (=768·2+2) |
| 합계 | 66,954,241 | 66,955,010 |

방식 B의 분류 헤드 파라미터가 정확히 *2배* 입니다. 차이는 **769개** — 전체 67M 중 0.001%. 이 미세한 자유도 차이가 두 방식의 *최종 확률* 을 거의 같게, *학습된 가중치* 는 미묘하게 다르게 만듭니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:45:15 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   43C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 학습 — Ch 10과 동일한 hyperparams

Ch 10과 *완전히 같은* learning rate, batch size, epoch 수, seed. **변하는 건 모델 출력 shape와 loss 종류뿐**.

평가 지표 함수입니다. 방식 B는 logit이 `(B, 2)`라, 안정 softmax(최댓값을 빼고 정규화)로 두 클래스 확률을 구한 뒤 클래스 1의 확률만 뽑아 AUC에, `argmax`로 0/1 예측을 떨어뜨려 accuracy·F1에 씁니다.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits.shape = (B, 2)  → softmax → 클래스 1 확률
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))   # 안정화
    probs_full = exp / exp.sum(axis=1, keepdims=True)          # (B, 2)
    probs = probs_full[:, 1]                                   # (B,) 클래스 1 확률
    preds = probs_full.argmax(axis=1)                          # 0/1 예측

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(labels, probs)),
    }
```

Ch 10과 *완전히 동일한* hyperparams(lr=2e-5, batch 16, 2 epoch, seed=42, fp16)로 `Trainer`를 구성해 학습합니다. 마지막 비교가 의미를 가지려면 데이터·모델 본체·학습 설정이 모두 같고 loss 종류와 출력 shape만 달라야 합니다.

```python
training_args = TrainingArguments(
    output_dir="./ch11_output",
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
print(f"\nTraining done — mean train loss: {train_result.training_loss:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Training done — mean train loss: 0.2582
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:45:47 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   57C    P0             65W /   70W |    1579MiB /  15360MiB |     56%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A             629      C   /usr/bin/python3                       1576MiB |
+-----------------------------------------------------------------------------------------+
```
