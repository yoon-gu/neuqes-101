> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/13_bert_multilabel/13_bert_multilabel.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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
Mon Jun 22 03:49:55 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   46C    P8             11W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 데이터 — Yelp + 항목(aspect) 합성 라벨 (Ch 6과 동일)

Yelp 리뷰엔 multi-label 정답이 없습니다. Ch 6에서처럼 5개 항목(aspect)별 키워드 사전을 만들어 텍스트에서 매칭 — 어떤 키워드라도 등장하면 해당 항목을 1로 활성. 5차원 multi-hot 벡터가 합성됩니다.

| 항목 | 의미 | 키워드 예시 |
|---|---|---|
| `food` | 음식의 맛/메뉴 | food, meal, dish, taste, delicious, ... |
| `service` | 서비스/응대 | service, staff, waiter, friendly, rude, ... |
| `price` | 가격/가성비 | price, cheap, expensive, value, worth, ... |
| `ambiance` | 분위기/인테리어 | atmosphere, decor, music, vibe, cozy, ... |
| `location` | 위치/주차 | location, parking, area, neighborhood, ... |

> **합성의 한계** — 키워드 매칭은 *언급한 항목* 만 잡고 *언급한 항목이 긍정인지 부정인지* 는 알 수 없습니다. 또 *키워드 없이* 항목이 표현된 경우(예: "10 minutes wait" → service)도 놓칩니다. 이 한계는 챕터 끝에서 솔직히 짚습니다.

Ch 6과 동일하게 5개 항목(aspect)별 키워드 사전을 정의합니다. `extract_aspects`는 텍스트를 소문자화한 뒤 각 항목의 키워드가 단어 경계(`\b`)로 하나라도 매칭되면 1.0, 아니면 0.0을 채워 길이 5의 multi-hot float 벡터를 만듭니다. 정답이 없는 Yelp에 라벨을 합성하는 단계입니다.

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

print(f"K (number of aspects): {K}")
print(f"aspects: {ASPECTS}")
```

**▶ 실행 결과**

```text
K (number of aspects): 5
aspects: ['food', 'service', 'price', 'ambiance', 'location']
```

Yelp 리뷰 train 5,000건·eval 1,000건을 샘플링한 뒤, 각 텍스트에 `extract_aspects`로 만든 5차원 multi-hot 벡터를 `aspects` 컬럼으로 부착합니다. 첫 샘플의 활성 항목을 출력해 합성 라벨이 제대로 붙었는지 눈으로 확인합니다.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("Yelp/yelp_review_full")
train_full = ds["train"].shuffle(seed=42).select(range(5000))
eval_full  = ds["test"].shuffle(seed=42).select(range(1000))

# 항목 라벨 합성 — 각 텍스트에 multi-hot 5차원 벡터 부착
def attach_aspects(batch):
    batch["aspects"] = [extract_aspects(t) for t in batch["text"]]
    return batch

train_full = train_full.map(attach_aspects, batched=True)
eval_full  = eval_full.map(attach_aspects,  batched=True)

print(f"train: {len(train_full)}, eval: {len(eval_full)}")
print(f"\nFirst sample:")
print(f"  text: {train_full[0]['text'][:150]}...")
print(f"  aspects (multi-hot): {train_full[0]['aspects']}")
print(f"  active aspects: {[a for a, v in zip(ASPECTS, train_full[0]['aspects']) if v > 0]}")
```

**▶ 실행 결과**

```text
train: 5000, eval: 1000

First sample:
  text: I stalk this truck.  I've been to industrial parks where I pretend to be a tech worker standing in line, strip mall parking lots, an …(뒤 21자 생략)
  aspects (multi-hot): [0.0, 1.0, 0.0, 0.0, 1.0]
  active aspects: ['service', 'location']
```

항목별 활성률과 샘플당 평균 활성 라벨 수를 집계해 데이터의 불균형 구조를 파악합니다. 활성률이 낮은 항목일수록 학습이 어렵고, 한 샘플에 몇 개 라벨이 동시에 켜지는지가 multi-label의 핵심 성격입니다.

```python
# 항목별 활성률
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
    print(f"  {n} labels active: {cnt} samples ({cnt/len(Y_train):.1%})")
```

**▶ 실행 결과**

```text
Per-aspect activation rate (train):
       food: 55.6%  (2778 / 5000)
    service: 49.6%  (2480 / 5000)
      price: 29.4%  (1472 / 5000)
   ambiance: 18.1%  (905 / 5000)
   location: 21.9%  (1095 / 5000)

Mean active labels per sample: 1.75
Active label distribution (train):
  0 labels active: 741 samples (14.8%)
  1 labels active: 1464 samples (29.3%)
  2 labels active: 1506 samples (30.1%)
  3 labels active: 957 samples (19.1%)
  4 labels active: 277 samples (5.5%)
  5 labels active: 55 samples (1.1%)
```

**결과 해석**

food(55.6%)·service(49.6%)가 흔하고 ambiance(18.1%)·location(21.9%)·price(29.4%)는 드뭅니다. 샘플당 평균 1.75개 라벨이 활성되며 2개 활성이 30.1%로 가장 많아, 라벨이 서로 배타적이지 않은 전형적 multi-label 분포임이 드러납니다.

**Ch 12와의 한 줄 차이**: `out["labels"] = [int(l) for l in batch["label"]]` → `out["labels"] = [list(map(float, a)) for a in batch["aspects"]]`. 라벨이 *int 스칼라* 가 아니라 *길이 5 multi-hot float 벡터*. 이 형식 + `problem_type="multi_label_classification"` 두 가지가 BCE per-label 자동 매핑의 트리거.

```python
def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # multi-hot 5차원 float 벡터 (BCEWithLogitsLoss가 받는 형식)
    out["labels"] = [list(map(float, a)) for a in batch["aspects"]]
    return out
```

**위 코드 읽기** — 토큰화한 결과의 `out["labels"]`에 `aspects`를 `float`로 변환한 길이 5 multi-hot 벡터를 그대로 넣습니다. Ch 12의 `int` 스칼라 라벨과 달리 *벡터·float* 형식이어야 `BCEWithLogitsLoss`가 받을 수 있습니다 — 이 한 줄이 multi-label로 가는 핵심 변경점입니다.

```python
train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(["text", "label", "aspects"])
eval_tok  = eval_full.map(tokenize_fn,  batched=True).remove_columns(["text", "label", "aspects"])

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (length-5 multi-hot float vector)")
```

**위 코드 읽기** — `map`으로 전체에 적용한 뒤 `text`·`label`·`aspects` 원본 컬럼을 제거해 `input_ids`·`attention_mask`·`labels`만 남깁니다. 첫 샘플 라벨이 `[0.0, 1.0, 0.0, 0.0, 1.0]`처럼 길이 5 float 벡터로 찍히는지 확인합니다.

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample label: [0.0, 1.0, 0.0, 0.0, 1.0]  (length-5 multi-hot float vector)
```

## 모델 로드 — `num_labels=5` + `multi_label_classification`

Ch 12와 *모델 아키텍처는 동일* (`Linear(H, 5)` 분류 헤드). 변하는 한 가지 — `problem_type="multi_label_classification"` — 가 자동 매핑되는 loss를 BCE per-label 로 바꿉니다.

모델을 `num_labels=5`로 로드하되 `problem_type="multi_label_classification"`을 지정해 loss를 BCE per-label로 자동 매핑합니다. 헤드 구조(`Linear(768, 5)`)와 파라미터 수는 Ch 12와 동일하고, 바뀌는 것은 이 한 줄과 라벨 형식뿐임을 출력으로 확인합니다.

```python
model = AutoModelForSequenceClassification.from_pretrained(
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
print(f"id2label:             {model.config.id2label}")
```

**▶ 실행 결과**

```text
[transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
Key                     | Status     | 
------------------------+------------+-
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
classifier.weight       | MISSING    | 
classifier.bias         | MISSING    | 
pre_classifier.bias     | MISSING    | 
pre_classifier.weight   | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:              66,957,317  (67.0 M)
Trainable parameters:    66,957,317  (100.0%)
Classifier:           Linear(in_features=768, out_features=5, bias=True)
problem_type:         multi_label_classification
id2label:             {0: 'food', 1: 'service', 2: 'price', 3: 'ambiance', 4: 'location'}
```

**Ch 12와 파라미터 수가 *완전히 동일*** — 차이는 `problem_type` 한 줄뿐. 같은 모델이 *어떻게 해석되고 어떤 loss로 학습되는가* 만 바뀝니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:50:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   47C    P8             14W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 학습 — Ch 12와 동일한 hyperparams

Ch 12와 *완전히 같은* learning rate, batch size, epoch 수, seed. 평가 metric만 multi-label용으로 새로 짭니다.

multi-label용 평가 함수를 정의합니다. logits를 라벨마다 독립 sigmoid로 확률화하고 0.5 임계값으로 0/1을 정한 뒤, Hamming loss와 micro/macro F1, macro AUC를 계산합니다.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred                      # logits: (N, K), labels: (N, K) float
    probs = 1.0 / (1.0 + np.exp(-logits))           # per-label sigmoid
    preds = (probs >= 0.5).astype(int)              # threshold 0.5

    out = {}
    # Hamming loss — 전체 라벨 위치 중 틀린 비율 (낮을수록 좋음)
    out["hamming_loss"] = float(hamming_loss(labels, preds))
```

**위 코드 읽기** — softmax가 아니라 `1/(1+exp(-logits))`로 *라벨마다 독립* sigmoid를 적용하는 것이 multi-label의 본질입니다. `probs >= 0.5`로 multi-hot 예측을 만들고, `hamming_loss`는 전체 (샘플 × 라벨) 위치 중 틀린 비율을 잽니다.

```python
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
```

**위 코드 읽기** — `average="micro"`는 모든 라벨의 TP/FP/FN을 합산 후 F1을 내므로 흔한 라벨이 점수를 지배하고, `average="macro"`는 라벨별 F1을 *동일 가중치로 평균* 해 드문 라벨도 똑같이 반영합니다. 두 값의 차이가 클수록 라벨 불균형이 심하다는 신호입니다.

```python
    # Per-label AUC (One-vs-Rest 자체)
    try:
        out["macro_auc"] = float(roc_auc_score(labels, probs, average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    return out
```

**위 코드 읽기** — AUC는 임계값과 무관하게 *확률 자체의 분리력* 을 재는 지표라, 0.5 임계값으로 손해 본 F1과 별개로 모델이 양/음을 얼마나 잘 정렬하는지 보여줍니다.

Ch 12와 동일한 learning rate·batch size·epoch·seed로 `Trainer`를 구성하고 학습합니다. 바뀐 건 라벨 형식과 `problem_type`, 그리고 위에서 정의한 multi-label용 `compute_metrics`뿐입니다.

```python
training_args = TrainingArguments(
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
print(f"\nTraining done — mean train loss: {train_result.training_loss:.4f}")
```

**▶ 실행 결과**

```text
Epoch  Training Loss  Validation Loss  Hamming Loss  Micro F1  Micro Precision  Micro Recall  Macro F1  Macro Precision  Macro Recall  Macro Auc  Runtime   Samples Per Second  Steps Per Second
1      0.391352       0.375564         0.158200      0.730494  0.884488         0.622171      0.579077  0.712338         0.516885      0.858364   0.964200  1037.144000         33.189000
2      0.313462       0.320437         0.124600      0.797662  0.905605         0.712710      0.723947  0.915523         0.644669      0.899381   0.919700  1087.355000         34.795000
Training done — mean train loss: 0.4126
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:51:02 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   63C    P0             34W /   70W |    1577MiB /  15360MiB |     72%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1283      C   /usr/bin/python3                       1574MiB |
+-----------------------------------------------------------------------------------------+
```
