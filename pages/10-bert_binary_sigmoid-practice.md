> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/10_bert_binary_sigmoid/10_bert_binary_sigmoid.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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
Mon Jun 22 03:42:33 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   49C    P8             14W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 데이터 — Yelp 이진화 (Ch 3·4와 동일)

별점 4-5는 `1.0` (긍정), 1-2는 `0.0` (부정), 3은 제외. 라벨을 *float 1차원 multi-hot 벡터* (`[0.0]` 또는 `[1.0]`) 형태로 둡니다 — 이게 BCE를 자동 적용시키는 핵심 형식.

토크나이저를 불러오고 Yelp 리뷰에서 학습 5,000개·평가 1,000개를 뽑습니다. 이어서 별점 3(중립)을 제외하고 4-5는 `1.0`, 1-2는 `0.0` 으로 이진화해 방식 A가 다룰 binary 라벨을 만듭니다.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("Yelp/yelp_review_full")
train_full = ds["train"].shuffle(seed=42).select(range(5000))
eval_full  = ds["test"].shuffle(seed=42).select(range(1000))

# 별점 3 제외 + 이진화
def to_binary(example):
    star = example["label"] + 1   # 0-4 → 1-5
    if star == 3:
        return False
    return True

def add_binary(batch):
    bins = []
    for lbl in batch["label"]:
        star = lbl + 1
        bins.append(1.0 if star >= 4 else 0.0)
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

```python
def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # 라벨을 길이 1짜리 multi-hot 벡터로 — Trainer가 BCEWithLogitsLoss 자동 적용
    out["labels"] = [[float(b)] for b in batch["binary"]]
    return out
```

**위 코드 읽기** — 방식 A의 핵심은 라벨 형식입니다. `out["labels"]` 에 scalar가 아니라 `[float(b)]` 로 한 번 감싼 *길이 1짜리 float 벡터* 를 넣습니다. 이렇게 두면 batching 시 shape가 `(batch, 1)` 이 되어 `(batch, 1)` logits와 맞아떨어지고, `Trainer` 가 `multi_label_classification` 으로 인식해 `BCEWithLogitsLoss` 를 자동 적용합니다.

```python
train_tok = train_bin.map(tokenize_fn, batched=True).remove_columns(["text", "label", "binary"])
eval_tok  = eval_bin.map(tokenize_fn,  batched=True).remove_columns(["text", "label", "binary"])

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (length-1 float vector)")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 4040
})

First sample label: [1.0]  (length-1 float vector)
```

## 모델 로드 — 방식 A 셋업

`num_labels=1` + `problem_type="multi_label_classification"` 이 핵심.

방식 A의 모델 셋업입니다. `num_labels=1` 로 출력 헤드를 1차원 logit으로 두고, `problem_type="multi_label_classification"` 으로 지정해 `Trainer` 가 BCE를 자동으로 고르게 합니다. 새로 붙는 분류 헤드는 무작위 초기화되므로 파인튜닝으로 학습됩니다.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1,
    problem_type="multi_label_classification",   # ← BCEWithLogitsLoss 자동 매핑
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
```

**▶ 실행 결과**

```text
[transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
Key                     | Status     | 
------------------------+------------+-
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
pre_classifier.weight   | MISSING    | 
pre_classifier.bias     | MISSING    | 
classifier.weight       | MISSING    | 
classifier.bias         | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:              66,954,241  (67.0 M)
Trainable parameters:    66,954,241  (100.0%)
Classifier:           Linear(in_features=768, out_features=1, bias=True)
problem_type:         multi_label_classification
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:42:58 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   49C    P8             14W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 학습 — Ch 9 골격 그대로

`compute_metrics` 만 binary 분류용으로 새로 짭니다 — sigmoid + threshold 0.5 로 0/1 예측을 만들고 accuracy/F1/AUC 계산.

평가 지표 함수입니다. 모델이 내놓는 건 logit이므로, 여기서 직접 sigmoid를 통과시켜 확률로 바꾼 뒤 0.5 임계값으로 0/1 예측을 만듭니다. accuracy·precision·recall·F1과 함께, 임계값과 무관하게 분리도를 보는 AUC도 계산합니다.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.flatten()         # (B, 1) → (B,)
    labels = labels.flatten().astype(int)

    # logit → 확률 (sigmoid 직접 적용)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(labels, probs)),
    }
```

학습 설정과 `Trainer` 를 구성해 2 에폭을 돌립니다. T4에서는 `fp16=True` 가 필수이며(bf16 미지원), 골격은 Ch 9 회귀와 동일하고 loss만 BCE로 바뀝니다.

```python
training_args = TrainingArguments(
    output_dir="./ch10_output",
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
Epoch  Training Loss  Validation Loss  Accuracy  Precision  Recall    F1        Auc
1      0.244964       0.263394         0.905473  0.885117   0.913747  0.899204  0.966201
2      0.159652       0.275900         0.898010  0.898072   0.878706  0.888283  0.967991
Training done — mean train loss: 0.2617
```

**결과 해석**

평균 train loss가 0.26 수준으로 내려와 BCE가 정상적으로 줄어들었음을 보여줍니다. 회귀(MSE)와 달리 loss 값 자체는 확률 오분류의 로그 페널티라 단위가 다릅니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:43:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   64C    P0             70W /   70W |    1579MiB /  15360MiB |     77%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A             673      C   /usr/bin/python3                       1576MiB |
+-----------------------------------------------------------------------------------------+
```
