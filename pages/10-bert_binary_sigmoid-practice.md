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
Wed Jun 17 21:32:42 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   43C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
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

DistilBERT 토크나이저를 불러오고 Yelp 리뷰 데이터에서 학습 5,000개, 평가 1,000개를 뽑습니다. 이어 별점 3(중립)을 제외하고 4-5점을 1, 1-2점을 0으로 이진화합니다. 별점 3을 빼는 이유는 긍정과 부정의 경계가 모호한 샘플을 학습에서 미리 걸러 내기 위해서입니다.

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

텍스트를 토큰화하면서 라벨을 길이 1짜리 float 벡터로 만듭니다. `Trainer`가 `problem_type="multi_label_classification"`을 보고 `BCEWithLogitsLoss`를 자동으로 적용하려면 라벨이 `[1.0]` 같은 multi-hot 실수 텐서여야 하기 때문입니다. 토큰화가 끝나면 모델 입력에 필요 없는 원본 컬럼을 제거합니다.

```python
def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # 라벨을 길이 1짜리 multi-hot 벡터로 — Trainer가 BCEWithLogitsLoss 자동 적용
    out["labels"] = [[float(b)] for b in batch["binary"]]
    return out

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

DistilBERT에 출력 차원 1짜리 분류 헤드를 올려 모델을 만듭니다. `num_labels=1`에 `problem_type="multi_label_classification"`을 함께 지정해 sigmoid+BCE 방식(방식 A)을 쓰도록 합니다. 분류 헤드가 `out_features=1` Linear로 새로 초기화되는지, problem_type이 의도대로 잡혔는지 출력에서 확인해 보세요.

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
vocab_transform.weight  | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
vocab_layer_norm.bias   | UNEXPECTED | 
pre_classifier.weight   | MISSING    | 
classifier.bias         | MISSING    | 
pre_classifier.bias     | MISSING    | 
classifier.weight       | MISSING    | 

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
Wed Jun 17 21:33:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   44C    P8             14W /   70W |       3MiB /  15360MiB |      0%      Default |
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

평가 때 쓸 지표 계산 함수를 정의합니다. 모델이 내놓는 logit에 직접 sigmoid를 적용해 확률로 바꾸고, 0.5를 기준으로 0/1 예측을 만든 뒤 정확도·정밀도·재현율·F1·AUC를 계산합니다. AUC는 임계값과 무관하게 확률 순위 자체의 품질을 보는 지표라 함께 넣었습니다.

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

학습 설정을 정의하고 `Trainer`로 묶어 실제 파인튜닝을 돌립니다. T4에서 30분 안에 끝나도록 2 에폭, batch size 16, `fp16=True`로 잡았습니다. 끝나면 평균 학습 loss가 출력되며, 에폭마다 평가가 함께 수행됩니다.

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
<IPython.core.display.HTML object>
Training done — mean train loss: 0.2558
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:33:38 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   61C    P0             68W /   70W |    1579MiB /  15360MiB |     69%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            6252      C   /usr/bin/python3                       1576MiB |
+-----------------------------------------------------------------------------------------+
```

학습이 끝난 모델을 평가셋 전체에 대해 돌려 앞서 정의한 지표들을 한 번에 계산합니다. `eval_` 접두사가 붙은 float 값만 골라 보기 좋게 출력합니다.

```python
# 평가 metric
eval_metrics = trainer.evaluate()
print("BERT method A evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
BERT method A evaluation:
             eval_loss: 0.2819
         eval_accuracy: 0.9030
        eval_precision: 0.8970
           eval_recall: 0.8922
               eval_f1: 0.8946
              eval_auc: 0.9685
```

**결과 해석**

별점 3을 제외한 이진 분류에서 정확도 90.3%, AUC 0.97입니다. Ch 3의 sklearn(86%)보다 한 단계 높아, 같은 binary 태스크라도 sigmoid+BCE 헤드 아래 BERT 표현이 성능을 더 끌어올린다는 걸 보여줍니다.

평가셋의 raw logit을 직접 받아 sigmoid로 확률로 변환하고, logit과 확률의 범위·양성 예측 비율을 살펴봅니다. 첫 5개 샘플을 표로 뽑아 logit 부호와 확률, 예측이 어떻게 연결되는지 눈으로 확인합니다.

```python
# logit → 확률
preds_output = trainer.predict(eval_tok)
logits = preds_output.predictions.flatten()
probs  = 1.0 / (1.0 + np.exp(-logits))
labels = preds_output.label_ids.flatten().astype(int)

print(f"Logit range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"Prob range:  [{probs.min():.4f}, {probs.max():.4f}]")
print(f"Positive prediction rate (prob >= 0.5): {(probs >= 0.5).mean():.1%}")
print(f"\nFirst 5 samples:")
print(pd.DataFrame({
    "label": labels[:5],
    "logit": logits[:5].round(2),
    "prob":  probs[:5].round(4),
    "pred":  (probs[:5] >= 0.5).astype(int),
}).to_string(index=False))
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Logit range: [-4.57, 4.11]
Prob range:  [0.0102, 0.9838]
Positive prediction rate (prob >= 0.5): 45.9%

First 5 samples:
 label  logit   prob  pred
     1   3.77 0.9775     1
     0  -3.29 0.0360     0
     1   4.04 0.9827     1
     1   3.71 0.9761     1
     1   4.06 0.9830     1
```

**결과 해석**

logit이 [−4.57, 4.11], sigmoid를 통과한 확률이 [0.01, 0.98]에 퍼집니다. 첫 5개처럼 logit 부호(양수→1, 음수→0)가 그대로 예측을 가르고, |logit|이 클수록 확률이 0이나 1에 바싹 붙습니다.

실제 라벨별로 예측 확률의 분포를 KDE 곡선으로 그립니다. 0.5 결정 경계를 기준으로 두 라벨이 얼마나 잘 갈라지는지, 어디서 겹치는지 한눈에 보기 위한 그림입니다.

```python
# 메인: 확률 공간 KDE — seaborn으로 부드러운 분포 + 라벨별 hue
sns.set_theme(style="whitegrid", context="talk")

df = pd.DataFrame({"prob": probs, "logit": logits, "label": labels})
PAL = {0: "#5B8DEF", 1: "#F47272"}  # 파랑=negative, 빨강=positive

fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df, x="prob", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, clip=(0, 1), ax=ax,
)
ax.axvline(0.5, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_title("Method A — Probability Distribution by Actual Label")
ax.set_xlabel("Predicted probability  P(y=1)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/10-bert_binary_sigmoid-out1.png)

**결과 해석**

두 라벨의 확률 분포가 0.5 경계를 두고 양쪽 끝(0 근처·1 근처)으로 갈라집니다. 모델이 대부분의 샘플을 자신 있게 분류한다는 뜻이고, 가운데에서 겹치는 구간이 오분류가 나는 영역입니다.

같은 분포를 sigmoid 통과 전 logit 공간에서 다시 그립니다. z=0 경계를 기준으로 두 라벨이 좌우로 나뉘는 모습을 보면서, 확률 공간의 양극단 쏠림이 logit의 좌우 분리를 sigmoid가 눌러 만든 결과임을 비교해 보세요.

```python
# 보조: logit 공간 KDE — sigmoid를 통과하기 전 모습
fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df, x="logit", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, ax=ax,
)
ax.axvline(0.0, color="black", lw=1.2, ls="--", alpha=0.7,
           label="decision boundary z=0")
ax.set_title("Method A — Logit Distribution (pre-sigmoid)")
ax.set_xlabel("Logit  z")
ax.set_ylabel("Density")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/10-bert_binary_sigmoid-out2.png)

**결과 해석**

sigmoid 통과 전 logit 공간에서는 두 분포가 z=0 경계를 사이에 두고 좌우로 나뉩니다. 확률 공간의 0/1 양극단 쏠림이 사실은 logit의 좌우 분리를 sigmoid가 눌러 만든 모습임을 보여줍니다.

negative와 positive 각 클래스별로 정밀도·재현율·F1을 한 표로 정리합니다. 전체 정확도만으로는 가려지는 클래스별 불균형이 있는지 확인하기 위한 상세 리포트입니다.

```python
# 상세 분류 리포트
print(classification_report(
    labels, (probs >= 0.5).astype(int),
    target_names=["negative", "positive"],
    digits=4,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

    negative     0.9080    0.9122    0.9101       433
    positive     0.8970    0.8922    0.8946       371

    accuracy                         0.9030       804
   macro avg     0.9025    0.9022    0.9024       804
weighted avg     0.9030    0.9030    0.9030       804
```

방식 A의 예측 확률·라벨과 지표 요약을 파일로 저장합니다. 다음 장(Ch 11)에서 softmax+CE 방식(방식 B)의 결과와 직접 비교하려고 미리 디스크에 남겨 두는 것입니다.

```python
import json, os

os.makedirs("./shared_binary_results", exist_ok=True)

# numpy 배열을 그대로 저장
np.save("./shared_binary_results/method_a_probs.npy", probs)
np.save("./shared_binary_results/method_a_labels.npy", labels)

# metric 요약
method_a_summary = {
    "method": "A (sigmoid + BCE, num_labels=1)",
    "metrics": {
        k.replace("eval_", ""): v
        for k, v in eval_metrics.items()
        if k.startswith("eval_") and isinstance(v, float)
    },
}
with open("./shared_binary_results/method_a_summary.json", "w") as f:
    json.dump(method_a_summary, f, indent=2)

print("Saved: ./shared_binary_results/")
for f in sorted(os.listdir("./shared_binary_results")):
    size_kb = os.path.getsize(f"./shared_binary_results/{f}") / 1024
    print(f"  {f}  ({size_kb:.1f} KB)")
```

**▶ 실행 결과**

```text
Saved: ./shared_binary_results/
  method_a_labels.npy  (6.4 KB)
  method_a_probs.npy  (3.3 KB)
  method_a_summary.json  (0.3 KB)
```
