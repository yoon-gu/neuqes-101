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

```python
# 평가 metric
eval_metrics = trainer.evaluate()
print("BERT method B evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
BERT method B evaluation:
             eval_loss: 0.2656
         eval_accuracy: 0.9104
        eval_precision: 0.9030
           eval_recall: 0.9030
               eval_f1: 0.9030
              eval_auc: 0.9689
```

**결과 해석**

방식 B는 accuracy 0.9104, F1 0.9030, AUC 0.9689로 안정적으로 수렴합니다. 뒤에서 다시 학습하는 방식 A(accuracy 0.9055)와 거의 같은 자리에 떨어지는데, 바로 이 일치가 이 챕터의 핵심입니다.

방식 A와 비교하려면 2차원 logit을 1차원으로 환산해야 합니다. 여기서 핵심은 `logits = logits2[:, 1] - logits2[:, 0]`, 즉 $z = z_1 - z_0$ — 이렇게 두면 $\sigma(z) = \mathrm{softmax}(z_0, z_1)[1] = p_1$ 이 되어 방식 A의 1차원 logit과 정확히 같은 의미를 가집니다.

```python
# logits → softmax → 클래스 1 확률 + 1차원 logit z = z1 - z0
preds_output = trainer.predict(eval_tok)
logits2 = preds_output.predictions          # (B, 2)
labels  = preds_output.label_ids.astype(int)

# 안정 softmax
exp = np.exp(logits2 - logits2.max(axis=1, keepdims=True))
probs_full = exp / exp.sum(axis=1, keepdims=True)
probs = probs_full[:, 1]                    # (B,) 클래스 1 확률

# 방식 A와 동등성 비교를 위해 1차원 logit 만들기: z = z1 - z0
logits = logits2[:, 1] - logits2[:, 0]      # (B,)

print(f"logits2 (raw)  shape: {logits2.shape}")
print(f"logit z = z1-z0 range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"Prob range:            [{probs.min():.4f}, {probs.max():.4f}]")
print(f"Positive prediction rate (prob >= 0.5): {(probs >= 0.5).mean():.1%}")
print(f"\nFirst 5 samples:")
print(pd.DataFrame({
    "label":   labels[:5],
    "z0":      logits2[:5, 0].round(2),
    "z1":      logits2[:5, 1].round(2),
    "z=z1-z0": logits[:5].round(2),
    "prob_B":  probs[:5].round(4),
    "pred":    probs_full[:5].argmax(axis=1),
}).to_string(index=False))
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
logits2 (raw)  shape: (804, 2)
logit z = z1-z0 range: [-5.10, 4.88]
Prob range:            [0.0061, 0.9925]
Positive prediction rate (prob >= 0.5): 46.1%

First 5 samples:
 label    z0    z1  z=z1-z0  prob_B  pred
     1 -1.81  2.26     4.08  0.9833     1
     0  1.47 -1.55    -3.02  0.0466     0
     1 -2.05  2.60     4.65  0.9905     1
     1 -1.69  2.29     3.99  0.9818     1
     1 -2.13  2.64     4.77  0.9916     1
```

**결과 해석**

확률이 [0.0061, 0.9925]로 양 끝까지 벌어져 모델이 확신을 갖고 분류하고 있음을 보여줍니다. 첫 5개 샘플에서 `z=z1-z0`의 부호가 그대로 예측 클래스를 가르고(양수 → pred 1, 음수 → pred 0), 모두 정답 라벨과 일치합니다.

```python
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})

df = pd.DataFrame({"prob": probs, "logit": logits, "label": labels})
PAL = {0: "#5B8DEF", 1: "#F47272"}

fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df, x="prob", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, clip=(0, 1), ax=ax,
)
ax.axvline(0.5, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_title("방식 B — 실제 라벨별 확률 분포")
ax.set_xlabel("예측 확률  P(y=1) = softmax(logits)[1]")
ax.set_ylabel("밀도")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/11-bert_binary_softmax-out1.png)

```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df, x="logit", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, ax=ax,
)
ax.axvline(0.0, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_title("방식 B — logit 분포  (z = z1 − z0)")
ax.set_xlabel("logit  z = z1 − z0")
ax.set_ylabel("밀도")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/11-bert_binary_softmax-out2.png)

```python
# 상세 분류 리포트
print(classification_report(
    labels, probs_full.argmax(axis=1),
    target_names=["negative", "positive"],
    digits=4,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

    negative     0.9169    0.9169    0.9169       433
    positive     0.9030    0.9030    0.9030       371

    accuracy                         0.9104       804
   macro avg     0.9099    0.9099    0.9099       804
weighted avg     0.9104    0.9104    0.9104       804
```

이제 같은 노트북 안에서 방식 A를 다시 학습해 직접 비교합니다. 텍스트·attention_mask는 그대로 두고 라벨만 방식 A 형식(길이 1 multi-hot float `[0.0]`/`[1.0]`)으로 바꿉니다 — 샘플 순서가 동일해야 나중에 점 대 점 비교가 성립합니다.

```python
# 방식 A용 라벨 변환 — int 0/1 → 길이 1 multi-hot float [0.0]/[1.0]
def to_method_a_labels(batch):
    batch["labels"] = [[float(l)] for l in batch["labels"]]
    return batch

# 텍스트·attention_mask는 그대로, labels만 바꿔서 새 데이터셋
train_tok_A = train_tok.map(to_method_a_labels, batched=True)
eval_tok_A  = eval_tok.map(to_method_a_labels,  batched=True)

print(f"Method A first sample label: {train_tok_A[0]['labels']}  (length-1 float vector)")
print(f"Method B first sample label: {train_tok[0]['labels']}    (int scalar)")
```

**▶ 실행 결과**

```text
Method A first sample label: [1.0]  (length-1 float vector)
Method B first sample label: 1    (int scalar)
```

```python
# 방식 A 모델 — Ch 10과 동일 셋업
model_A = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1,
    problem_type="multi_label_classification",
)

def compute_metrics_A(eval_pred):
    logits, lbl = eval_pred
    logits = logits.flatten()
    lbl    = lbl.flatten().astype(int)
    p_hat  = 1.0 / (1.0 + np.exp(-logits))
    preds  = (p_hat >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(lbl, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(lbl, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(lbl, p_hat)),
    }

print(f"Method A classifier:    {model_A.classifier}")
print(f"Method A problem_type:  {model_A.config.problem_type}")
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
Method A classifier:    Linear(in_features=768, out_features=1, bias=True)
Method A problem_type:  multi_label_classification
```

```python
# 방식 A 학습 — Ch 10과 동일한 hyperparams (방식 B와도 동일)
training_args_A = TrainingArguments(
    output_dir="./ch11_method_a_output",
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

trainer_A = Trainer(
    model=model_A,
    args=training_args_A,
    train_dataset=train_tok_A,
    eval_dataset=eval_tok_A,
    processing_class=tokenizer,
    compute_metrics=compute_metrics_A,
)

train_result_A = trainer_A.train()
print(f"\nMethod A training done — train loss: {train_result_A.training_loss:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Method A training done — train loss: 0.2588
```

```python
# 방식 A 예측 추출
preds_A_out = trainer_A.predict(eval_tok_A)
logits_A    = preds_A_out.predictions.flatten()
probs_A     = 1.0 / (1.0 + np.exp(-logits_A))
labels_A    = preds_A_out.label_ids.flatten().astype(int)

# eval_tok과 eval_tok_A는 라벨 형식만 다르고 샘플 순서는 동일 → 라벨 일치해야 함
assert (labels_A == labels).all(), "Sample order mismatch — check eval_tok / eval_tok_A derivation"

eval_metrics_A = trainer_A.evaluate()
print("Method A evaluation:")
for k, v in eval_metrics_A.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
Method A evaluation:
             eval_loss: 0.2811
         eval_accuracy: 0.9055
        eval_precision: 0.9041
           eval_recall: 0.8895
               eval_f1: 0.8967
              eval_auc: 0.9663
```

```python
metrics_A = {k.replace("eval_", ""): v for k, v in eval_metrics_A.items()
             if k.startswith("eval_") and isinstance(v, float)}
metrics_B = {k.replace("eval_", ""): v for k, v in eval_metrics.items()
             if k.startswith("eval_") and isinstance(v, float)}

common = [k for k in metrics_A if k in metrics_B]
cmp = pd.DataFrame({
    "metric":                  common,
    "method A (sigmoid+BCE)":  [metrics_A[k] for k in common],
    "method B (softmax+CE)":   [metrics_B[k] for k in common],
})
cmp["|A-B|"] = (cmp["method A (sigmoid+BCE)"] - cmp["method B (softmax+CE)"]).abs()
print(cmp.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
   metric  method A (sigmoid+BCE)  method B (softmax+CE)  |A-B|
     loss                  0.2811                 0.2656 0.0155
 accuracy                  0.9055                 0.9104 0.0050
precision                  0.9041                 0.9030 0.0011
   recall                  0.8895                 0.9030 0.0135
       f1                  0.8967                 0.9030 0.0062
      auc                  0.9663                 0.9689 0.0026
```

**결과 해석**

모든 지표에서 두 방식의 차이가 0.02 미만입니다 (accuracy 0.0050, AUC 0.0026, precision 0.0011). 같은 데이터·같은 hyperparams로 학습했으니 남는 차이는 random init·dropout 같은 학습 경로 노이즈일 뿐 — 식으로 본 동등성이 BERT에서도 그대로 성립함을 수치로 확인합니다.

```python
df_cmp = pd.DataFrame({
    "prob_A": probs_A,
    "prob_B": probs,
    "label":  labels.astype(int),
})

fig, ax = plt.subplots(figsize=(7, 7))
sns.scatterplot(
    data=df_cmp, x="prob_A", y="prob_B", hue="label",
    palette={0: "#5B8DEF", 1: "#F47272"}, alpha=0.55, s=35, ax=ax,
)
ax.plot([0, 1], [0, 1], color="black", lw=1.3, ls="--", alpha=0.7,
        label="y = x (완전 일치)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("방식 A — P(y=1) = sigmoid(z_A)")
ax.set_ylabel("방식 B — P(y=1) = softmax(z_0, z_1)[1]")
ax.set_title("방식 A vs 방식 B — 샘플별 확률 일치도")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

corr = float(np.corrcoef(probs_A, probs)[0, 1])
mae  = float(np.abs(probs_A - probs).mean())
print(f"Pearson corr:        {corr:.4f}  (1.0 = perfect equivalence)")
print(f"Mean abs diff |A-B|: {mae:.4f}")
```

**▶ 실행 결과**

![output](../assets/11-bert_binary_softmax-out3.png)

```text
Pearson corr:        0.9883  (1.0 = perfect equivalence)
Mean abs diff |A-B|: 0.0239
```

**결과 해석**

샘플별 확률의 Pearson 상관이 0.9883, 평균 절대 차가 0.0239로 두 방식이 사실상 같은 함수를 학습했음을 보여줍니다. scatter의 점들이 $y=x$ 직선 주변에 흩어지되 체계적 치우침이 없으니, 차이는 학습 경로 노이즈일 뿐입니다.

```python
pred_A = (probs_A >= 0.5).astype(int)
pred_B = (probs   >= 0.5).astype(int)

agree         = (pred_A == pred_B).mean()
both_correct  = ((pred_A == labels) & (pred_B == labels)).mean()
only_A_right  = ((pred_A == labels) & (pred_B != labels)).mean()
only_B_right  = ((pred_A != labels) & (pred_B == labels)).mean()
both_wrong    = ((pred_A != labels) & (pred_B != labels)).mean()

print(f"Agreement rate (A vs B predictions): {agree:.1%}")
print()
print(f"Prediction quadrants:")
print(f"  both correct:           {both_correct:.1%}")
print(f"  only A correct (B wrong): {only_A_right:.1%}")
print(f"  only B correct (A wrong): {only_B_right:.1%}")
print(f"  both wrong:             {both_wrong:.1%}")
```

**▶ 실행 결과**

```text
Agreement rate (A vs B predictions): 97.8%

Prediction quadrants:
  both correct:           89.7%
  only A correct (B wrong): 0.9%
  only B correct (A wrong): 1.4%
  both wrong:             8.1%
```

**결과 해석**

threshold 0.5에서 두 방식의 예측이 97.8% 일치하고, 의견이 갈리는 경우는 합쳐서 2.3%(only A 0.9% + only B 1.4%)에 불과합니다. 실질적으로 같은 분류기로 봐도 무방하다는 결론입니다.
