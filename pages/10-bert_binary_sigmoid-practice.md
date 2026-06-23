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
<IPython.core.display.HTML object>
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
             eval_loss: 0.2759
         eval_accuracy: 0.8980
        eval_precision: 0.8981
           eval_recall: 0.8787
               eval_f1: 0.8883
              eval_auc: 0.9680
```

**결과 해석**

평가 정확도 약 89.8%, F1 0.888, AUC 0.968로 방식 A가 binary 분류를 잘 학습했습니다. AUC가 0.97에 가깝다는 건 임계값을 어디에 두든 두 클래스가 확률적으로 잘 분리된다는 뜻입니다. Ch 11에서 학습할 방식 B(softmax+CE)와 비교할 기준선이 됩니다.

`Trainer.predict()` 로 평가셋의 raw logit을 받아 sigmoid로 확률을 만들고, 처음 5개 샘플의 logit·확률·예측을 정답과 나란히 봅니다.

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
Logit range: [-4.41, 4.26]
Prob range:  [0.0120, 0.9861]
Positive prediction rate (prob >= 0.5): 45.1%

First 5 samples:
 label  logit   prob  pred
     1   3.70 0.9758     1
     0  -3.38 0.0331     0
     1   4.18 0.9849     1
     1   3.87 0.9796     1
     1   4.21 0.9854     1
```

**결과 해석**

logit이 약 -4.4 ~ +4.3 범위로 뻗어, sigmoid 통과 후 확률은 0.01 ~ 0.99 양 끝에 압착됩니다. 처음 5개 모두 정답과 예측이 일치하며 확률도 0.03 또는 0.97처럼 자신감 있게 한쪽으로 쏠려 있습니다 — sigmoid가 큰 logit을 0/1로 포화시키는 성질이 그대로 드러납니다.

```python
# 메인: 확률 공간 KDE — seaborn으로 부드러운 분포 + 라벨별 hue
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})

df = pd.DataFrame({"prob": probs, "logit": logits, "label": labels})
PAL = {0: "#5B8DEF", 1: "#F47272"}  # 파랑=negative, 빨강=positive

fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df, x="prob", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, clip=(0, 1), ax=ax,
)
ax.axvline(0.5, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_title("방식 A — 실제 라벨별 확률 분포")
ax.set_xlabel("예측 확률  P(y=1)")
ax.set_ylabel("밀도")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/10-bert_binary_sigmoid-out1.png)

```python
# 보조: logit 공간 KDE — sigmoid를 통과하기 전 모습
fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df, x="logit", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, ax=ax,
)
ax.axvline(0.0, color="black", lw=1.2, ls="--", alpha=0.7,
           label="결정 경계 z=0")
ax.set_title("방식 A — logit 분포 (sigmoid 통과 전)")
ax.set_xlabel("logit  z")
ax.set_ylabel("밀도")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/10-bert_binary_sigmoid-out2.png)

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

    negative     0.8980    0.9145    0.9062       433
    positive     0.8981    0.8787    0.8883       371

    accuracy                         0.8980       804
   macro avg     0.8980    0.8966    0.8972       804
weighted avg     0.8980    0.8980    0.8979       804
```

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
