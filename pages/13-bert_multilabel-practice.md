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

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:39:27 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   40C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
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

Yelp 리뷰에는 항목 라벨이 없으므로, 항목별 키워드 사전을 정의해 라벨을 합성합니다. `extract_aspects`는 한 리뷰에서 각 항목의 키워드가 단어 경계로 등장하는지 검사해 5차원 multi-hot 벡터를 만듭니다. 라벨이 서로 배타적이지 않아 여러 항목이 동시에 켜질 수 있다는 점이 multi-label의 핵심입니다.

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

DistilBERT 토크나이저를 불러오고 Yelp 데이터에서 학습 5,000개·평가 1,000개를 추출합니다. 이어 `attach_aspects`로 모든 텍스트에 앞서 정의한 5차원 multi-hot 항목 벡터를 부착합니다. 첫 샘플의 활성 항목까지 출력해 합성 라벨이 의도대로 붙는지 눈으로 확인합니다.

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

합성한 라벨의 분포를 점검합니다. 항목별 활성률과 함께, 한 샘플에 평균 몇 개의 항목이 켜지는지·0개부터 5개까지 어떻게 퍼져 있는지를 집계합니다. 라벨별 빈도 차와 다중 활성 정도를 미리 보면 뒤의 평가 지표 해석이 쉬워집니다.

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

food/service는 절반 가까이 활성이지만 ambiance/location은 20% 안팎으로 라벨별 빈도 차가 큽니다. 한 샘플에 평균 1.75개 항목이 켜지고 0개부터 5개까지 분포가 퍼져 있어, 라벨마다 독립 sigmoid로 다루는 multi-label 설정이 자연스럽습니다.

텍스트를 토큰화하면서 multi-hot 항목 벡터를 `labels` 컬럼에 넣습니다. `BCEWithLogitsLoss`는 라벨을 float 텐서로 받으므로 정수 0/1이 아니라 실수로 변환합니다. 학습에 불필요한 원본 컬럼은 제거해 `input_ids`·`labels`만 남깁니다.

```python
def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # multi-hot 5차원 float 벡터 (BCEWithLogitsLoss가 받는 형식)
    out["labels"] = [list(map(float, a)) for a in batch["aspects"]]
    return out

train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(["text", "label", "aspects"])
eval_tok  = eval_full.map(tokenize_fn,  batched=True).remove_columns(["text", "label", "aspects"])

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (length-5 multi-hot float vector)")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample label: [0.0, 1.0, 0.0, 0.0, 1.0]  (length-5 multi-hot float vector)
```

DistilBERT에 5차원 분류 헤드를 얹어 모델을 만듭니다. `problem_type="multi_label_classification"`을 지정하면 `Trainer`가 라벨별 독립 sigmoid + `BCEWithLogitsLoss`를 자동으로 적용합니다. 헤드가 새로 초기화되며(분류기 5출력) 파라미터 수와 설정이 의도대로 잡혔는지 출력으로 확인합니다.

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
vocab_transform.bias    | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
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

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:40:03 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   42C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

평가용 지표 함수를 정의합니다. 라벨별 logit에 sigmoid를 씌워 확률로 바꾸고 0.5 임계값으로 multi-hot 예측을 만든 뒤, Hamming loss와 micro·macro F1을 계산합니다. micro는 전체 라벨을 합산하고 macro는 라벨별 F1을 동일 가중으로 평균하므로, 빈도가 다른 라벨에서 두 값이 갈리는 점을 눈여겨봅니다.

```python
def compute_metrics(eval_pred):
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
    return out
```

학습 설정을 구성하고 `Trainer`로 파인튜닝을 실행합니다. T4 제약에 맞춰 배치 16·2에폭·`fp16=True`로 두고, 앞서 만든 `compute_metrics`를 연결해 에폭마다 평가가 돌게 합니다. 끝나면 평균 학습 손실을 출력합니다.

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
<IPython.core.display.HTML object>
Training done — mean train loss: 0.3983
```

**결과 해석**

평균 학습 손실 0.3983은 5개 라벨 각각의 per-label BCEWithLogitsLoss를 모두 합산-평균한 값입니다. 단일 라벨 분류보다 항이 많아 절대값을 다른 챕터와 직접 비교하기는 어렵고, 실제 성능은 다음 평가 지표로 판단합니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:40:42 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   62C    P0             70W /   70W |    1577MiB /  15360MiB |     73%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           13525      C   /usr/bin/python3                       1574MiB |
+-----------------------------------------------------------------------------------------+
```

학습된 모델을 평가 셋에 돌려 앞서 정의한 모든 지표를 한 번에 계산합니다. 손실뿐 아니라 Hamming loss·micro/macro F1·AUC가 함께 출력되어 multi-label 성능을 여러 각도로 읽을 수 있습니다.

```python
# 평가 metric
eval_metrics = trainer.evaluate()
print("BERT multi-label evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
BERT multi-label evaluation:
               eval_loss: 0.2932
       eval_hamming_loss: 0.1020
           eval_micro_f1: 0.8398
    eval_micro_precision: 0.9151
       eval_micro_recall: 0.7760
           eval_macro_f1: 0.8019
    eval_macro_precision: 0.9259
       eval_macro_recall: 0.7229
          eval_macro_auc: 0.9155
            eval_runtime: 0.9146
  eval_samples_per_second: 1093.3460
   eval_steps_per_second: 34.9870
```

**결과 해석**

micro F1 0.8398에 비해 macro F1 0.8019이 낮은데, 빈도 낮은 라벨까지 동일 가중치로 평균하는 macro 쪽이 손해를 보기 때문입니다. precision(0.92)이 recall(0.78)보다 높아 0.5 임계값에서 모델이 보수적으로 라벨을 켜고 있으며, macro AUC 0.9155는 임계값과 무관하게 라벨별 순위 분리력 자체는 우수함을 보여줍니다.

평가 셋 전체의 logit을 뽑아 sigmoid 확률과 0.5 임계값 예측을 직접 만듭니다. 라벨별로 확률 범위와 실제 활성률·예측 활성률을 나란히 출력해, 어떤 항목에서 모델이 양성을 충분히 켜지 못하는지 확인합니다.

```python
# logits → per-label sigmoid → multi-hot 예측
preds_output = trainer.predict(eval_tok)
logits = preds_output.predictions                   # (N, 5)
labels = preds_output.label_ids.astype(int)         # (N, 5) multi-hot
probs  = 1.0 / (1.0 + np.exp(-logits))              # (N, 5) per-label prob
preds  = (probs >= 0.5).astype(int)                 # (N, 5) multi-hot prediction

print(f"logits shape: {logits.shape}")
print(f"prob ranges per label:")
for k, a in enumerate(ASPECTS):
    print(f"  {a:>9}: [{probs[:, k].min():.4f}, {probs[:, k].max():.4f}]  "
          f"true rate={labels[:, k].mean():.1%}, pred rate={preds[:, k].mean():.1%}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
logits shape: (1000, 5)
prob ranges per label:
       food: [0.0319, 0.9882]  true rate=55.2%, pred rate=55.1%
    service: [0.0494, 0.9794]  true rate=49.7%, pred rate=47.6%
      price: [0.0563, 0.8972]  true rate=30.4%, pred rate=17.4%
   ambiance: [0.0196, 0.9381]  true rate=16.8%, pred rate=10.2%
   location: [0.0316, 0.9322]  true rate=20.2%, pred rate=15.8%
```

**결과 해석**

food/service는 예측률이 실제율에 거의 맞지만, price(30.4%→17.4%)나 ambiance(16.8%→10.2%)처럼 빈도 낮은 라벨은 0.5 임계값에서 모델이 절반 가까이를 켜지 못합니다. 양성이 드문 라벨일수록 sigmoid 확률이 0.5를 넘기 어려워 recall이 떨어지는 multi-label의 전형적 패턴입니다.

라벨별 precision·recall·F1을 한 표로 정리합니다. 항목마다 성능을 따로 보면, 빈도 낮은 라벨에서 precision은 높은데 recall이 주저앉는 불균형이 드러납니다.

```python
# Per-label classification report
print(classification_report(
    labels, preds,
    target_names=ASPECTS,
    digits=4, zero_division=0,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

        food     0.9347    0.9330    0.9338       552
     service     0.8761    0.8390    0.8571       497
       price     0.9080    0.5197    0.6611       304
    ambiance     0.9804    0.5952    0.7407       168
    location     0.9304    0.7277    0.8167       202

   micro avg     0.9151    0.7760    0.8398      1723
   macro avg     0.9259    0.7229    0.8019      1723
weighted avg     0.9170    0.7760    0.8310      1723
 samples avg     0.7733    0.6906    0.7114      1723
```

**결과 해석**

빈도 낮은 라벨은 precision은 높지만(price 0.91, ambiance 0.98) recall이 0.52-0.60으로 주저앉아 F1을 깎아먹습니다. 즉 모델은 켠 라벨은 거의 맞히지만 켜야 할 것을 놓치는 쪽으로 치우쳐 있고, 이 라벨들의 임계값을 0.5보다 낮추면 recall과 F1을 끌어올릴 여지가 있습니다.

지표 대신 개별 사례를 직접 읽어 봅니다. 정답 항목이 가장 많은 샘플과 가장 적은 샘플을 골라, 원문과 함께 라벨별 정답·확률·예측을 한 줄씩 비교합니다. 어떤 항목에서 확률이 임계값 근처를 맴도는지 손으로 짚어 보는 단계입니다.

```python
# 평가 셋에서 항목 활성이 가장 *많은* 샘플 1개 + 가장 *적은* 샘플 1개 골라 읽어보기
n_active = labels.sum(axis=1)
idx_many = int(np.argmax(n_active))   # 정답 항목이 가장 많은 샘플
idx_few  = int(np.argmin(n_active))   # 정답 항목이 가장 적은 샘플

# eval_full 에서 원문 텍스트 가져오기 (eval_tok 와 같은 순서)
texts = list(eval_full["text"])

for label_kind, idx in [("many active labels", idx_many), ("few active labels", idx_few)]:
    print("=" * 78)
    print(f"sample #{idx}  ({label_kind})")
    print("=" * 78)
    print(f"text (truncated): {texts[idx][:320]}{'...' if len(texts[idx]) > 320 else ''}")
    print()
    print(f"{'aspect':>10}  {'true':>6}  {'prob':>8}  {'pred (>=0.5)':>14}  match")
    for k, a in enumerate(ASPECTS):
        t = int(labels[idx, k])
        p = float(probs[idx, k])
        pr = int(preds[idx, k])
        ok = "✓" if t == pr else "✗"
        print(f"  {a:>9}  {t:>6}  {p:>8.4f}  {pr:>14}    {ok}")

    # 사람이 읽는 한 줄 해석
    pred_active = [a for k, a in enumerate(ASPECTS) if preds[idx, k]]
    true_active = [a for k, a in enumerate(ASPECTS) if labels[idx, k]]
    print()
    print(f"  predicted: {pred_active}")
    print(f"  true:      {true_active}")
    print()
```

**▶ 실행 결과**

```text
==============================================================================
sample #29  (many active labels)
==============================================================================
text (truncated): It's hard to complain about this place given the price I got it for! \n**Warning** This is a long review, there is a lot t …(뒤 201자 생략)

    aspect    true      prob    pred (>=0.5)  match
       food       1    0.2553               0    ✗
    service       1    0.5081               1    ✓
      price       1    0.7802               1    ✓
   ambiance       1    0.2684               0    ✗
   location       1    0.7138               1    ✓

  predicted: ['service', 'price', 'location']
  true:      ['food', 'service', 'price', 'ambiance', 'location']

==============================================================================
sample #4  (few active labels)
==============================================================================
text (truncated): I don't quite get this place or why Asians love it, but it is very good :)

    aspect    true      prob    pred (>=0.5)  match
       food       0    0.0903               0    ✓
    service       0    0.0672               0    ✓
      price       0    0.1050               0    ✓
   ambiance       0    0.0460               0    ✓
   location       0    0.0612               0    ✓

  predicted: []
  true:      []
```

**결과 해석**

5개 라벨이 모두 켜진 어려운 샘플에서는 food/ambiance를 놓쳐 3개만 맞혔는데, 두 라벨의 확률이 0.26-0.27로 0.5에 못 미친 경계 사례입니다. 반대로 라벨이 하나도 없는 샘플은 모든 확률이 0.1 아래로 깔끔히 떨어져, 라벨별 독립 sigmoid가 각 항목을 따로 끄고 켜는 구조가 잘 드러납니다.

라벨별 sigmoid 확률 분포를 양성·음성으로 나눠 그립니다. 항목마다 작은 패널을 만들고 0.5 점선을 함께 표시해, 양성과 음성 봉우리가 임계값을 기준으로 얼마나 잘 갈라지는지 시각적으로 확인합니다.

```python
sns.set_theme(style="whitegrid", context="talk")

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
plt.show()
```

**▶ 실행 결과**

![output](../assets/13-bert_multilabel-out1.png)

**결과 해석**

라벨마다 양성(빨강)과 음성(파랑) 확률 분포가 0.5 점선을 기준으로 잘 갈라집니다. 다만 ambiance/location처럼 양성이 드문 라벨은 양성 봉우리가 점선 왼쪽까지 끌려와 있어, 0.5 임계값이 이들에게는 다소 높게 작동하며 recall 손실로 이어집니다.

라벨 간 동시발생 구조를 들여다봅니다. 한 항목이 켜졌을 때 다른 항목이 함께 켜질 조건부 확률 행렬을 실제 라벨과 예측 라벨 각각에 대해 계산해 히트맵으로 나란히 그립니다. 라벨별 독립 sigmoid가 상관을 직접 모델링하지 않는데도 예측이 실제 동시발생 패턴을 따라가는지 비교하는 것이 핵심입니다.

```python
def cooccurrence_matrix(Y):
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
plt.show()
```

**▶ 실행 결과**

![output](../assets/13-bert_multilabel-out2.png)

**결과 해석**

실제 라벨 간 동시발생 패턴을 예측 행렬이 대체로 따라가지만, 빈도 낮은 라벨은 적게 켜진 만큼 예측 쪽 조건부 확률이 옅게 나옵니다. 라벨별 독립 sigmoid는 항목 간 상관을 직접 모델링하지 않는데도 본문 표현이 공통 신호를 담아 동시발생 구조가 어느 정도 재현되는 점이 흥미롭습니다.

BERT와 견줄 베이스라인으로 Ch 6의 TF-IDF + OvR 로지스틱 회귀를 같은 데이터에 재현합니다. 라벨마다 독립 이진 분류기를 학습해(OneVsRest) multi-label을 처리하고, 동일한 0.5 임계값과 지표로 micro/macro F1·AUC·Hamming loss를 계산합니다. 같은 조건에서 BERT와 직접 비교하기 위한 기준선입니다.

```python
# Ch 6 셋업 재현 — TF-IDF + OneVsRestClassifier(LogisticRegression)
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
print(f"  hamming loss:       {hamming_loss(Y_eval_bin, preds_sk):.4f}")
```

**▶ 실행 결과**

```text
sklearn TF-IDF + OvR LogReg:
  vocabulary size:    20,000
  micro F1:           0.7634
  macro F1:           0.6141
  macro AUC:          0.9387
  hamming loss:       0.1426
```

**결과 해석**

Ch 6를 재현한 TF-IDF + OvR LogReg 베이스라인은 micro F1 0.7634, macro F1 0.6141로, BERT보다 특히 macro 쪽이 크게 뒤집니다. 흥미롭게도 macro AUC는 0.9387로 BERT(0.9155)보다 살짝 높아, 순위 분리력 자체는 비슷해도 0.5 임계값에서 양성을 켜는 능력에서 격차가 벌어집니다.

두 모델의 지표를 같은 표로 모아 차이를 한눈에 봅니다. 공통 지표마다 sklearn·BERT 값과 그 차이(BERT − sklearn)를 나란히 출력해, 격차가 precision에서 나는지 recall에서 나는지 가립니다.

```python
metrics_bert = {
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
print(cmp.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
         metric  sklearn (OvR)  BERT (this chapter)  BERT - sklearn
   hamming_loss         0.1426               0.1020         -0.0406
       micro_f1         0.7634               0.8398          0.0765
micro_precision         0.8915               0.9151          0.0237
   micro_recall         0.6674               0.7760          0.1085
       macro_f1         0.6141               0.8019          0.1878
macro_precision         0.9036               0.9259          0.0223
   macro_recall         0.5307               0.7229          0.1923
      macro_auc         0.9387               0.9155         -0.0231
```

**결과 해석**

BERT의 이득은 거의 전부 recall에서 나옵니다. micro recall +0.11, macro recall +0.19로 크게 앞서는 반면 precision 차이는 미미하고 AUC는 오히려 sklearn이 약간 높습니다. 문맥을 읽는 BERT가 베이스라인이 놓치던 양성 라벨을 더 많이 건져 올려 F1 격차를 만든다는 뜻입니다.

전체 평균이 아닌 라벨별 F1로 쪼개 두 모델을 비교합니다. 항목마다 sklearn과 BERT의 F1과 그 차이를 표로 내고 막대그래프로도 그려, 어느 항목에서 BERT의 이득이 집중되는지 또렷하게 드러냅니다.

```python
def per_label_f1(Y_true, Y_pred):
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
plt.show()
```

**▶ 실행 결과**

```text
  aspect  sklearn F1  BERT F1  BERT - sklearn
    food      0.9057   0.9338          0.0281
 service      0.8833   0.8571         -0.0262
   price      0.5271   0.6611          0.1340
ambiance      0.3333   0.7407          0.4074
location      0.4211   0.8167          0.3956
```

**결과 해석**

food/service처럼 빈도 높은 라벨은 두 모델이 비슷하지만, 빈도 낮은 ambiance(+0.41)와 location(+0.40)에서 BERT가 압도적으로 앞섭니다. 단어 일치에 의존하는 TF-IDF가 희소 라벨에서 표현이 부족한 반면, 사전학습 문맥 표현이 적은 양성 신호도 잡아내 격차가 가장 크게 벌어지는 지점입니다.

![output](../assets/13-bert_multilabel-out3.png)
