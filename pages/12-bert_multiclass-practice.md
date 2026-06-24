> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/12_bert_multiclass/12_bert_multiclass.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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
    classification_report, roc_auc_score, confusion_matrix,
)
# Ch 5 sklearn baseline 비교용
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
Mon Jun 22 03:47:31 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   54C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("Yelp/yelp_review_full")
train_full = ds["train"].shuffle(seed=42).select(range(5000))
eval_full  = ds["test"].shuffle(seed=42).select(range(1000))

print(f"train: {len(train_full)} samples")
print(f"eval:  {len(eval_full)} samples")
print(f"\nClass distribution (train):")
for k in range(5):
    n = sum(1 for x in train_full["label"] if x == k)
    print(f"  star {k + 1} (label {k}): {n} ({n / len(train_full):.1%})")

# 첫 샘플 미리보기
print(f"\nFirst sample:")
print(f"  label: {train_full[0]['label']}  (star {train_full[0]['label'] + 1})")
print(f"  text:  {train_full[0]['text'][:200]}...")
```

**▶ 실행 결과**

```text
train: 5000 samples
eval:  1000 samples

Class distribution (train):
  star 1 (label 0): 1017 (20.3%)
  star 2 (label 1): 1027 (20.5%)
  star 3 (label 2): 960 (19.2%)
  star 4 (label 3): 1021 (20.4%)
  star 5 (label 4): 975 (19.5%)

First sample:
  label: 4  (star 5)
  text:  I stalk this truck.  I've been to industrial parks where I pretend to be a tech worker standing in line, strip mall parking lots, a …(뒤 72자 생략)
```

```python
def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(l) for l in batch["label"]]   # 0-4 int (Yelp의 라벨 그대로)
    return out

train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(["text", "label"])
eval_tok  = eval_full.map(tokenize_fn,  batched=True).remove_columns(["text", "label"])

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (int scalar in 0-4)")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample label: 4  (int scalar in 0-4)
```

```python
STAR_LABELS = {0: "1★", 1: "2★", 2: "3★", 3: "4★", 4: "5★"}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5,
    problem_type="single_label_classification",
    id2label=STAR_LABELS,
    label2id={v: k for k, v in STAR_LABELS.items()},
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
vocab_transform.weight  | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
pre_classifier.bias     | MISSING    | 
pre_classifier.weight   | MISSING    | 
classifier.bias         | MISSING    | 
classifier.weight       | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:              66,957,317  (67.0 M)
Trainable parameters:    66,957,317  (100.0%)
Classifier:           Linear(in_features=768, out_features=5, bias=True)
problem_type:         single_label_classification
id2label:             {0: '1★', 1: '2★', 2: '3★', 3: '4★', 4: '5★'}
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:47:57 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   55C    P8             15W /   70W |       3MiB /  15360MiB |      0%      Default |
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

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 안정 softmax (K=5)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_full = exp / exp.sum(axis=1, keepdims=True)   # (B, 5)
    preds = probs_full.argmax(axis=1)                   # (B,)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {
        "accuracy":        float(accuracy_score(labels, preds)),
        "macro_precision": float(p),
        "macro_recall":    float(r),
        "macro_f1":        float(f1),
    }
    # multi-class AUC: One-vs-Rest, 모든 라벨이 적어도 한 개 등장해야 계산 가능
    try:
        out["auc_ovr"] = float(roc_auc_score(labels, probs_full, multi_class="ovr"))
    except ValueError:
        out["auc_ovr"] = float("nan")
    return out
```

```python
training_args = TrainingArguments(
    output_dir="./ch12_output",
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
print(f"random baseline loss (K=5): {np.log(5):.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Training done — mean train loss: 1.0802
random baseline loss (K=5): 1.6094
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:48:37 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   75C    P0             64W /   70W |    1577MiB /  15360MiB |     73%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2000      C   /usr/bin/python3                       1574MiB |
+-----------------------------------------------------------------------------------------+
```

```python
# 평가 metric
eval_metrics = trainer.evaluate()
print("BERT 5-class evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
BERT 5-class evaluation:
               eval_loss: 1.0000
           eval_accuracy: 0.5580
    eval_macro_precision: 0.5555
       eval_macro_recall: 0.5595
           eval_macro_f1: 0.5561
            eval_auc_ovr: 0.8657
```

```python
# logits → softmax → argmax
preds_output = trainer.predict(eval_tok)
logits  = preds_output.predictions               # (B, 5)
labels  = preds_output.label_ids.astype(int)     # (B,)

exp = np.exp(logits - logits.max(axis=1, keepdims=True))
probs_full = exp / exp.sum(axis=1, keepdims=True)  # (B, 5)
preds = probs_full.argmax(axis=1)                  # (B,)

# top-1 확률 (모델이 선택한 클래스의 확률)
top1_prob = probs_full.max(axis=1)
correct = (preds == labels)

print(f"logits shape: {logits.shape}")
print(f"top-1 prob range: [{top1_prob.min():.4f}, {top1_prob.max():.4f}]")
print(f"top-1 prob mean: correct={top1_prob[correct].mean():.4f}, wrong={top1_prob[~correct].mean():.4f}")
print(f"\nFirst 5 samples:")
print(pd.DataFrame({
    "label (star-1)": labels[:5],
    "pred (star-1)":  preds[:5],
    "top-1 prob":     top1_prob[:5].round(4),
    "correct?":       correct[:5],
}).to_string(index=False))
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
logits shape: (1000, 5)
top-1 prob range: [0.2245, 0.8730]
top-1 prob mean: correct=0.6279, wrong=0.5414

First 5 samples:
 label (star-1)  pred (star-1)  top-1 prob  correct?
              2              4      0.4724     False
              4              3      0.4351     False
              1              0      0.7647     False
              4              4      0.4687      True
              3              3      0.5864      True
```

```python
sns.set_theme(style="white", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})

cm = confusion_matrix(labels, preds, labels=list(range(5)))
# 정답 라벨별 정규화 — 각 행 합이 1이 되어 *재현율* 을 직접 읽을 수 있음
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_norm, annot=cm, fmt="d",                       # 색은 비율, 숫자는 raw count
    cmap="Blues", vmin=0, vmax=1,
    xticklabels=[STAR_LABELS[k] for k in range(5)],
    yticklabels=[STAR_LABELS[k] for k in range(5)],
    cbar_kws={"label": "행 정규화 (재현율)"}, ax=ax,
)
ax.set_xlabel("예측 별점")
ax.set_ylabel("실제 별점")
ax.set_title("혼동 행렬 — 5-class Yelp")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/12-bert_multiclass-out1.png)

```python
df_top = pd.DataFrame({
    "top1_prob": top1_prob,
    "outcome":   np.where(correct, "correct", "wrong"),
})

fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df_top, x="top1_prob", hue="outcome",
    fill=True, common_norm=False, alpha=0.5,
    palette={"correct": "#5BD17F", "wrong": "#E55050"},
    clip=(1/5, 1.0), ax=ax,
)
ax.axvline(1/5, color="black", lw=1.0, ls=":", alpha=0.5)
ax.text(1/5, ax.get_ylim()[1]*0.95, "  균등분포 = 1/K", va="top", fontsize=10, alpha=0.6)
ax.set_title("top-1 확률 — 정답/오답으로 나눈 분포")
ax.set_xlabel("top-1 예측 확률  max_k P(y=k)")
ax.set_ylabel("밀도")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/12-bert_multiclass-out2.png)

```python
# 클래스별 분류 리포트 (precision/recall/F1 클래스 단위)
print(classification_report(
    labels, preds,
    target_names=[STAR_LABELS[k] for k in range(5)],
    digits=4, zero_division=0,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

          1★     0.6520    0.7409    0.6936       220
          2★     0.5056    0.4225    0.4604       213
          3★     0.5134    0.4898    0.5013       196
          4★     0.4651    0.4878    0.4762       205
          5★     0.6412    0.6566    0.6488       166

    accuracy                         0.5580      1000
   macro avg     0.5555    0.5595    0.5561      1000
weighted avg     0.5535    0.5580    0.5542      1000
```

```python
# Ch 5 셋업 재현 — TF-IDF + multinomial LogReg
texts_train  = list(train_full["text"])
labels_train = list(train_full["label"])
texts_eval   = list(eval_full["text"])
labels_eval  = list(eval_full["label"])

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train = vec.fit_transform(texts_train)
X_eval  = vec.transform(texts_eval)

clf = LogisticRegression(max_iter=2000, n_jobs=-1)   # 최신 sklearn은 multinomial이 default for multi-class
clf.fit(X_train, labels_train)

probs_sk = clf.predict_proba(X_eval)                 # (B, 5)
preds_sk = probs_sk.argmax(axis=1)                   # (B,)

acc_sk = float(accuracy_score(labels_eval, preds_sk))
ps, rs, f1s, _ = precision_recall_fscore_support(labels_eval, preds_sk, average="macro", zero_division=0)
auc_sk = float(roc_auc_score(labels_eval, probs_sk, multi_class="ovr"))

print(f"sklearn TF-IDF + LogReg:")
print(f"  vocabulary size:    {len(vec.vocabulary_):,}")
print(f"  trained parameters: {clf.coef_.size + clf.intercept_.size:,}  (~{clf.coef_.size/1e3:.0f} K)")
print(f"  accuracy:           {acc_sk:.4f}")
print(f"  macro F1:           {f1s:.4f}")
print(f"  AUC (OvR):          {auc_sk:.4f}")
```

**▶ 실행 결과**

```text
sklearn TF-IDF + LogReg:
  vocabulary size:    20,000
  trained parameters: 100,005  (~100 K)
  accuracy:           0.5420
  macro F1:           0.5380
  AUC (OvR):          0.8420
```

```python
metrics_bert = {
    k.replace("eval_", ""): v for k, v in eval_metrics.items()
    if k.startswith("eval_") and isinstance(v, float)
}
metrics_sk = {
    "accuracy":        acc_sk,
    "macro_precision": float(ps),
    "macro_recall":    float(rs),
    "macro_f1":        float(f1s),
    "auc_ovr":         auc_sk,
}

common = [k for k in metrics_bert if k in metrics_sk]
cmp = pd.DataFrame({
    "metric":             common,
    "sklearn (TF-IDF)":   [metrics_sk[k]   for k in common],
    "BERT":               [metrics_bert[k] for k in common],
})
cmp["BERT - sklearn"] = cmp["BERT"] - cmp["sklearn (TF-IDF)"]
print(cmp.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
         metric  sklearn (TF-IDF)   BERT  BERT - sklearn
       accuracy            0.5420 0.5580          0.0160
macro_precision            0.5377 0.5555          0.0177
   macro_recall            0.5410 0.5595          0.0185
       macro_f1            0.5380 0.5561          0.0181
        auc_ovr            0.8420 0.8657          0.0236
```

```python
cm_bert = confusion_matrix(labels, preds, labels=list(range(5)))
cm_sk   = confusion_matrix(labels_eval, preds_sk, labels=list(range(5)))

cm_bert_n = cm_bert / cm_bert.sum(axis=1, keepdims=True)
cm_sk_n   = cm_sk   / cm_sk.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, cm_n, cm_raw, title in [
    (axes[0], cm_sk_n,   cm_sk,   "sklearn TF-IDF + LogReg"),
    (axes[1], cm_bert_n, cm_bert, "BERT"),
]:
    sns.heatmap(
        cm_n, annot=cm_raw, fmt="d", cmap="Blues", vmin=0, vmax=1,
        xticklabels=[STAR_LABELS[k] for k in range(5)],
        yticklabels=[STAR_LABELS[k] for k in range(5)],
        cbar=False, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("예측 별점")
    ax.set_ylabel("실제 별점")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/12-bert_multiclass-out3.png)
