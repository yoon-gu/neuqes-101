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
Wed Jun 17 21:41:56 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   39C    P8             12W /   70W |       3MiB /  15360MiB |      0%      Default |
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

메인 multi-label 태스크의 정답을 만들기 위해, 리뷰 텍스트를 다섯 가지 측면(food, service, price, ambiance, location)으로 라벨링하는 규칙을 정의합니다. 각 측면마다 키워드 목록을 두고, `extract_aspects`가 문장에 해당 키워드가 하나라도 등장하면 1, 아니면 0을 매겨 5차원 multi-hot 벡터를 만듭니다. 단어 경계(`\b`)로 매칭해 부분 문자열 오탐을 막는 점을 눈여겨보세요.

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

토크나이저를 불러오고 Yelp 리뷰에서 학습 5,000개, 평가 1,000개를 샘플링합니다. `attach_aspects_and_aux`로 두 종류의 라벨을 한꺼번에 붙이는데, 메인 태스크용 5차원 측면 multi-hot과 보조 태스크용 별점 회귀값(0-4 별점을 4로 나눠 0-1 스케일로)을 함께 만듭니다. 하나의 입력에서 두 라벨이 동시에 나오는 구조가 이 장 보조 손실의 출발점입니다.

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

보조 회귀 라벨인 별점 점수가 어떻게 분포하는지 확인합니다. 범위와 평균, 표준편차를 출력하고 다섯 별점 구간별로 샘플 수와 비율을 세어, 특정 별점에 쏠리지 않고 고르게 퍼져 있는지 점검합니다.

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

**결과 해석**

별점 5개 구간이 모두 19-20% 대로 고르게 분포해 있어, 보조 회귀 태스크가 특정 별점에 치우치지 않고 균형 잡힌 신호를 제공합니다.

텍스트를 토큰화하면서 두 라벨을 모델이 받을 형태로 정리합니다. 메인 multi-label 정답은 `labels` 키에 5차원 float multi-hot으로, 보조 별점은 `aux_labels` 키에 float 스칼라로 담습니다. 토큰화 후에는 원본 텍스트와 중간 컬럼을 제거해 데이터셋을 가볍게 유지합니다.

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

표준 `DataCollatorWithPadding`은 `aux_labels` 같은 비표준 키를 어떻게 다룰지 모르므로, 이를 감싸는 collator를 직접 만듭니다. 먼저 `aux_labels`를 텐서로 빼낸 뒤 나머지를 표준 패딩에 넘기고, multi-hot `labels`를 float으로 보정한 다음 보조 라벨을 다시 붙여 배치를 완성합니다. 마지막에 네 개 샘플로 배치를 만들어 각 키의 shape과 dtype을 찍어 봅니다.

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

메인 분류 모델을 만들면서 보조 회귀용 헤드를 직접 덧붙입니다. `AutoModelForSequenceClassification`을 multi-label로 불러오면 5차원 메인 분류 헤드는 자동으로 붙고, 여기에 CLS hidden(768차원)을 스칼라로 내보내는 `aux_head` Linear를 수동으로 추가합니다. 출력의 파라미터 요약에서 보조 헤드가 전체의 0.001% 남짓에 불과한 가벼운 추가임을 눈여겨보세요.

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
vocab_transform.bias    | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
pre_classifier.weight   | MISSING    | 
pre_classifier.bias     | MISSING    | 
classifier.bias         | MISSING    | 
classifier.weight       | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:              66,958,086  (67.0 M)
Trainable parameters:    66,958,086  (100.0%)
Aux head parameters:            769  (0.0011%)
Main classifier:      Linear(in_features=768, out_features=5, bias=True)
Aux head:             Linear(in_features=768, out_features=1, bias=True)
```

이 장의 핵심으로, `Trainer`의 `compute_loss`만 오버라이드해 결합 손실을 구현합니다. 메인 BCE 손실은 자동 매핑으로 그대로 받고, `output_hidden_states=True`로 받은 마지막 layer의 CLS hidden을 `aux_head`에 통과시켜 보조 MSE 손실을 따로 계산합니다. 최종 손실은 `main_loss + λ * aux_loss`로 두 손실을 λ로 가중 합산하며, λ가 보조 태스크의 영향력을 조절하는 손잡이가 됩니다.

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


print("AuxTrainer defined — only compute_loss is overridden in Trainer.")
```

**▶ 실행 결과**

```text
AuxTrainer defined — only compute_loss is overridden in Trainer.
```

평가 단계에서 메인 multi-label 성능을 재는 함수를 정의합니다. 로짓에 sigmoid를 적용해 확률로 바꾼 뒤 0.5 임계값으로 예측을 정하고, hamming loss와 micro/macro 평균의 F1, precision, recall, 그리고 macro AUC를 한꺼번에 계산합니다. 보조 헤드는 여기서 평가하지 않고 메인 태스크에만 집중하는 점을 눈여겨보세요.

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

이제 λ=1로 결합 손실을 적용해 실제로 학습을 돌립니다. T4 제약에 맞춰 2 에폭, batch size 16, `fp16=True`로 설정하고, `aux_labels`가 `model.forward` 시그니처에 없어 자동 제거되지 않도록 `remove_unused_columns=False`를 켭니다. 앞서 정의한 `AuxTrainer`에 λ를 넘겨 보조 손실을 메인과 동등한 비중으로 섞은 결과를 확인합니다.

```python
LAMBDA_AUX = 1.0

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
With-aux training done — mean train loss: 0.5672
```

**결과 해석**

여기서의 평균 train loss 0.5672는 메인 BCE 손실과 λ=1로 더해진 보조 MSE 손실을 합친 결합 손실이므로, 메인 단독 학습 때보다 값이 더 큰 것이 자연스럽습니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:43:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   60C    P0             67W /   70W |    1619MiB /  15360MiB |     69%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2268      C   /usr/bin/python3                       1616MiB |
+-----------------------------------------------------------------------------------------+
```

학습이 끝난 λ=1 모델로 평가 셋에서 메인 multi-label 성능을 측정합니다. `evaluate`가 돌려준 지표 중 `eval_`로 시작하는 float 값만 추려 보기 좋게 출력합니다.

```python
# 메인 metric
eval_metrics_aux = trainer_aux.evaluate()
print("With-aux (λ=1) — main task metrics:")
for k, v in eval_metrics_aux.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
With-aux (λ=1) — main task metrics:
               eval_loss: 0.4758
       eval_hamming_loss: 0.1958
           eval_micro_f1: 0.6559
    eval_micro_precision: 0.8316
       eval_micro_recall: 0.5415
           eval_macro_f1: 0.4004
    eval_macro_precision: 0.7101
       eval_macro_recall: 0.3760
          eval_macro_auc: 0.8293
            eval_runtime: 0.9760
  eval_samples_per_second: 1024.5580
   eval_steps_per_second: 32.7860
```

**결과 해석**

λ=1로 보조 손실을 강하게 섞은 결과 메인 multi-label 성능이 micro F1 0.6559, macro F1 0.4004 수준에 그쳐, 보조 태스크가 메인 학습을 상당히 잠식했음을 보여줍니다. 곧 학습할 λ=0 베이스라인과 비교하면 그 차이가 분명해집니다.

이번에는 보조 회귀 헤드가 별점을 얼마나 잘 맞히는지 직접 측정합니다. `Trainer`의 자동 평가는 메인 헤드만 보므로, 평가 셋을 배치 단위로 수동 forward해 CLS hidden에서 `aux_head` 예측을 모은 뒤 RMSE, R^2, Pearson 상관을 계산합니다.

```python
# 보조 metric — eval 전체에 대해 수동 forward (작아서 빠름)
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

print("\nWith-aux (λ=1) — aux task metrics (star regression, 0-1 scale):")
print(f"  RMSE:    {rmse_aux:.4f}")
print(f"  R^2:     {r2_aux:.4f}")
print(f"  Pearson: {pear_aux:.4f}")
```

**▶ 실행 결과**

```text
With-aux (λ=1) — aux task metrics (star regression, 0-1 scale):
  RMSE:    0.2099
  R^2:     0.6383
  Pearson: 0.8026
```

**결과 해석**

보조 헤드는 별점을 R^2 0.64, Pearson 0.80으로 꽤 잘 회귀해, 0-1 스케일 RMSE가 0.21에 그칩니다. 메인 성능을 희생한 대가로 보조 태스크 자체는 신뢰할 만한 수준으로 학습됐습니다.

뒤에서 라벨별로 베이스라인과 비교하기 위해, λ=1 모델의 샘플별 메인 예측을 미리 뽑아 둡니다. `predict`로 받은 로짓에 sigmoid와 0.5 임계값을 적용해 multi-hot 예측을 만들고, 정답도 함께 정수형으로 보관합니다.

```python
# 메인 task per-sample 예측 (다음 비교 단계에서 사용)
preds_output_aux = trainer_aux.predict(eval_tok)
logits_aux = preds_output_aux.predictions
if isinstance(logits_aux, tuple):
    logits_aux = logits_aux[0]
labels_eval = preds_output_aux.label_ids.astype(int)
probs_aux = 1.0 / (1.0 + np.exp(-logits_aux))
preds_main_aux = (probs_aux >= 0.5).astype(int)

print(f"Main logits shape: {logits_aux.shape}")
print(f"Eval samples:      {len(labels_eval)}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Main logits shape: (1000, 5)
Eval samples:      1000
```

보조 손실의 효과를 분리해 보기 위해 λ=0 베이스라인을 학습합니다. 같은 데이터와 하이퍼파라미터, 같은 `AuxTrainer`를 쓰되 λ만 0으로 두어 보조 MSE 항을 완전히 끄므로, 메인 BCE 손실만으로 학습한 순수 비교군이 됩니다. 동등한 비교를 위해 모델도 새 인스턴스로 새로 초기화하는 점을 눈여겨보세요.

```python
# 새 모델 인스턴스 — λ=0 학습용
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
print(f"\nNo-aux (λ=0) baseline training done — mean train loss: {train_result_no_aux.training_loss:.4f}")
```

**▶ 실행 결과**

```text
[transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
Key                     | Status     | 
------------------------+------------+-
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
pre_classifier.weight   | MISSING    | 
pre_classifier.bias     | MISSING    | 
classifier.bias         | MISSING    | 
classifier.weight       | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
<IPython.core.display.HTML object>
No-aux (λ=0) baseline training done — mean train loss: 0.4075
```

**결과 해석**

λ=0 베이스라인의 train loss 0.4075는 보조 항이 빠진 순수 메인 BCE 손실이라, 결합 손실 0.5672보다 작은 것이 당연합니다. 둘은 손실 구성이 다르므로 절댓값이 아니라 메인 metric으로 비교해야 합니다.

베이스라인 모델로 메인 metric을 평가하고, 라벨별 비교에 쓸 샘플별 예측도 함께 뽑아 둡니다. 앞서 λ=1 모델에 적용한 것과 똑같이 sigmoid와 0.5 임계값을 거쳐 multi-hot 예측을 만들어, 두 모델을 같은 기준에서 비교할 수 있게 합니다.

```python
# baseline 메인 metric
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
preds_main_no_aux = (probs_no_aux >= 0.5).astype(int)
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
No-aux (λ=0) baseline — main task metrics:
               eval_loss: 0.3091
       eval_hamming_loss: 0.1138
           eval_micro_f1: 0.8163
    eval_micro_precision: 0.9199
       eval_micro_recall: 0.7336
           eval_macro_f1: 0.7577
    eval_macro_precision: 0.9205
       eval_macro_recall: 0.6737
          eval_macro_auc: 0.9081
            eval_runtime: 1.1311
  eval_samples_per_second: 884.1300
   eval_steps_per_second: 28.2920
<IPython.core.display.HTML object>
```

**결과 해석**

보조 손실 없이 학습한 베이스라인은 micro F1 0.8163, macro F1 0.7577로 λ=1 모델을 크게 앞섭니다. 여기서는 λ=1이 너무 커 보조 태스크가 메인을 짓눌렀고, 균형점을 찾으려면 λ를 더 작게 두어야 한다는 신호입니다.

두 모델의 메인 metric을 한 표로 모아 직접 비교합니다. 공통 지표마다 λ=0 베이스라인과 λ=1 값을 나란히 놓고 차이(`delta`)까지 계산해, 보조 손실이 각 지표를 얼마나 올리거나 내렸는지 한눈에 읽을 수 있게 합니다.

```python
m_aux    = {k.replace("eval_", ""): v for k, v in eval_metrics_aux.items()
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
print(cmp.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
            metric  no aux (lambda=0)  with aux (lambda=1)  delta (aux - no_aux)
              loss             0.3091               0.4758                0.1667
      hamming_loss             0.1138               0.1958                0.0820
          micro_f1             0.8163               0.6559               -0.1604
   micro_precision             0.9199               0.8316               -0.0884
      micro_recall             0.7336               0.5415               -0.1921
          macro_f1             0.7577               0.4004               -0.3573
   macro_precision             0.9205               0.7101               -0.2104
      macro_recall             0.6737               0.3760               -0.2977
         macro_auc             0.9081               0.8293               -0.0788
           runtime             1.1311               0.9760               -0.1551
samples_per_second           884.1300            1024.5580              140.4280
  steps_per_second            28.2920              32.7860                4.4940
```

**결과 해석**

모든 메인 metric에서 delta가 음수라, λ=1 보조 손실이 메인 성능을 일관되게 끌어내렸습니다. 특히 macro F1이 -0.3573으로 가장 크게 떨어져, 데이터가 적은 희소 라벨일수록 보조 항의 간섭에 취약함을 시사합니다.

평균 지표만으로는 가려진, 보조 손실의 영향이 라벨마다 어떻게 다른지 들여다봅니다. 다섯 측면 각각에 대해 베이스라인과 λ=1 모델의 F1을 따로 구해 표로 비교하고, 막대그래프로 나란히 그려 어느 라벨이 가장 크게 무너지는지 시각적으로 드러냅니다.

```python
def per_label_f1(Y_true, Y_pred):
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
plt.show()
```

**▶ 실행 결과**

```text
  aspect  no aux F1  with aux F1  delta (aux - no_aux)
    food     0.9306       0.8897               -0.0409
 service     0.8653       0.7833               -0.0820
   price     0.4612       0.1173               -0.3439
ambiance     0.7111       0.2116               -0.4995
location     0.8202       0.0000               -0.8202
```

![output](../assets/14-auxiliary_loss-out1.png)

**결과 해석**

food, service처럼 풍부한 라벨은 보조 손실에도 F1이 거의 유지되지만, price, ambiance, location 같은 희소 라벨은 크게 무너지고 특히 location은 F1이 0.8202에서 0으로 완전히 collapse합니다. 보조 태스크의 간섭이 약한 라벨에 집중된다는 점이 막대 그래프에서 한눈에 드러납니다.

마지막으로 보조 회귀 헤드의 예측을 실제 별점별로 violin 그래프로 그려, 회귀가 별점 순서를 제대로 잡았는지 확인합니다. 정답이 다섯 개 정수 별점에서만 나오므로 별점마다 예측 분포를 묶고, 각 별점의 목표값(0.0-1.0)을 점선 가이드로 함께 표시해 분포 중심이 가이드를 따라 오르는지 비교합니다.

```python
# True star 별로 예측값 분포를 violin 으로 — 정답이 5개 정수 라벨에서만 나오므로
# scatter 보다 분포가 훨씬 깔끔하게 보임
true_star = np.round(np.array(aux_true) * 4).astype(int) + 1   # 0-1 스케일을 1-5 별점으로
star_label = [f"{s}*" for s in true_star]
df_aux = pd.DataFrame({"True star": star_label, "Predicted (0-1 scale)": aux_preds_aux})
order = ["1*", "2*", "3*", "4*", "5*"]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
sns.violinplot(
    data=df_aux, x="True star", y="Predicted (0-1 scale)",
    order=order, inner="quart", cut=0,
    color="#F47272", alpha=0.6, ax=ax,
)
# 정답이 있는 위치를 점선 가이드로 표시 (1* -> 0.0, 5* -> 1.0)
for i, target in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    ax.hlines(target, i - 0.4, i + 0.4, color="black", lw=1.1, ls="--", alpha=0.7)
ax.set_ylim(-0.2, 1.2)
ax.set_title(f"Aux task — predicted vs true star  (RMSE={rmse_aux:.3f}, r={pear_aux:.3f})")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/14-auxiliary_loss-out2.png)

**결과 해석**

별점이 높아질수록 보조 헤드의 예측 분포 중심이 점선 가이드(0.0 -> 1.0)를 따라 단조롭게 올라가, 회귀가 별점 순서를 제대로 학습했음을 보여줍니다. RMSE 0.21, Pearson 0.80이라는 수치가 그림의 깔끔한 단조 상승으로 시각화됩니다.
