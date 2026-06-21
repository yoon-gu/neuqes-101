> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/18_ko_auxiliary/18_ko_auxiliary.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModel,
    Trainer, TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report,
    roc_auc_score, hamming_loss, mean_squared_error, r2_score,
)

plt.rcParams["axes.unicode_minus"] = False

# device 자동감지 — Colab(T4) 은 CUDA, 로컬 Mac 은 MPS, 그 외 CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device:         {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
elif DEVICE == "cpu":
    print("Warning: CPU runtime — training will be very slow. Switch to T4 recommended.")
```

**▶ 실행 결과**

```text
PyTorch:        2.11.0+cu128
CUDA available: True
Device:         cuda
GPU:             Tesla T4
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:51:43 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   40C    P8             11W /   70W |       3MiB /  15360MiB |      0%      Default |
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

KLUE-YNAT(연합뉴스 토픽 분류) 데이터를 내려받아 split과 크기, 7개 카테고리 라벨 이름을 확인합니다. 출력·플롯은 한글 폰트 깨짐을 피하려고 영문 라벨로 매핑해 두고, `title` 컬럼은 이후 코드와 맞추기 위해 `text`로 이름을 바꿉니다.

```python
ds = load_dataset("klue/klue", "ynat")
print(f"splits: {list(ds.keys())}")
print(f"sizes: {[(k, len(v)) for k, v in ds.items()]}")

LABEL_NAMES = ds["train"].features["label"].names   # KLUE-YNAT 원본 (한국어)
# 출력·플롯은 영문으로 (matplotlib 한글 폰트 깨짐·조판 문제 방지)
_KO2EN = {"IT과학": "IT/Science", "경제": "Economy", "사회": "Society", "생활문화": "Life&Culture", "세계": "World", "스포츠": "Sports", "정치": "Politics"}
LABEL_NAMES_EN = [_KO2EN.get(n, n) for n in LABEL_NAMES]
K = len(LABEL_NAMES)
print(f"label names ({K}): {LABEL_NAMES}")

# title 컬럼명을 'text' 로 통일
ds = ds.rename_column("title", "text")
print(f"\nfirst 2 raw samples:")
for ex in ds["train"].select(range(2)):
    print(f"  label={ex['label']} ({LABEL_NAMES_EN[ex['label']]:>12})  text={ex['text']!r}")
```

**▶ 실행 결과**

```text
splits: ['train', 'validation']
sizes: [('train', 45678), ('validation', 9107)]
label names (7): ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']

first 2 raw samples:
  label=3 (Life&Culture)  text='유튜브 내달 2일까지 크리에이터 지원 공간 운영'
  label=3 (Life&Culture)  text='어버이날 맑다가 흐려져…남부지방 옅은 황사'
```

single-label 데이터를 두 헤드라인씩 짝지어 multi-label로 합성하는 함수입니다. 텍스트는 ` [SEP] ` 로 이어 붙이고 라벨은 두 카테고리를 1로 켠 multi-hot으로 만드는데, 여기서 활성 카테고리 개수 `n_active`(1 또는 2)가 이 장의 보조 task 정답이 됩니다. 같은 카테고리끼리 짝지어지면 활성이 1개뿐이라는 점을 눈여겨봐 주세요.

```python
SEED = 42
N_TRAIN = 5000
N_EVAL  = 1000


def make_multilabel(source_split, n_samples, seed):
    '''single-label split 에서 두 샘플씩 결합해 multi-label 데이터셋 합성.

    - 2*n_samples 개 인덱스를 섞어 앞/뒤 절반을 짝으로 묶음
    - 텍스트는 " [SEP] " 로 이어붙임
    - multi-hot 라벨은 두 카테고리 위치를 1 로 (같은 카테고리면 1개만)
    - n_active 는 활성 카테고리 개수 (1 또는 2) — Ch 18 의 보조 task 정답
    '''
    rng = np.random.default_rng(seed)
    n_src = len(source_split)
    # 짝지을 인덱스 2*n_samples 개 (중복 허용 — 소스가 부족할 때 대비)
    # numpy.int64 로 datasets 컬럼을 인덱싱하면 TypeError → python int 로 캐스팅
    idx = rng.integers(0, n_src, size=2 * n_samples).tolist()
    idx_a, idx_b = idx[:n_samples], idx[n_samples:]

    # 컬럼을 미리 파이썬 list 로 (반복 인덱싱이 빠르고 타입 안전)
    src_text = list(source_split["text"])
    src_label = list(source_split["label"])

    texts, multi_hots, active_counts = [], [], []
    for a, b in zip(idx_a, idx_b):
        ca, cb = int(src_label[a]), int(src_label[b])
        combined = f"{src_text[a]} [SEP] {src_text[b]}"
        mh = [0.0] * K
        mh[ca] = 1.0
        mh[cb] = 1.0   # ca == cb 면 같은 위치 → 활성 1개
        texts.append(combined)
        multi_hots.append(mh)
        active_counts.append(int(sum(mh)))
    return Dataset.from_dict({
        "text": texts,
        "multi_hot": multi_hots,
        "n_active": active_counts,
    })


train_full = make_multilabel(ds["train"], N_TRAIN, seed=SEED)
eval_full  = make_multilabel(ds["validation"], N_EVAL, seed=SEED + 1)

print(f"synthetic train: {len(train_full)}")
print(f"synthetic eval:  {len(eval_full)}")
print(f"\nFirst synthetic sample:")
print(f"  text:      {train_full[0]['text']}")
print(f"  multi_hot: {train_full[0]['multi_hot']}")
print(f"  n_active:  {train_full[0]['n_active']}  ← Ch 18 aux label")
```

**▶ 실행 결과**

```text
synthetic train: 5000
synthetic eval:  1000

First synthetic sample:
  text:      신간 언어와 탱크를 응시하며·자본주의 리얼리즘 [SEP] 우리은행 주택금융공사와 도시재생 발굴·지원 업무협약
  multi_hot: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  n_active:  2  ← Ch 18 aux label
```

합성된 보조 라벨 `n_active`가 train/eval에서 어떻게 분포하는지 1·2 두 값의 비율과 평균을 출력합니다. 무작위 짝짓기라면 평균이 이론값 1.857에 가까워야 하므로, 데이터가 의도대로 만들어졌는지 가늠하는 점검입니다.

```python
# 보조 라벨 (n_active) 분포 — train/eval
n_active_train = np.array(train_full["n_active"])
n_active_eval  = np.array(eval_full["n_active"])

print("Aux label (n_active) distribution:")
print(f"{'value':>7}  {'train':>8}  {'eval':>8}")
for v in [1, 2]:
    n_tr = int((n_active_train == v).sum())
    n_ev = int((n_active_eval  == v).sum())
    print(f"  {v:>5}  {n_tr:>5} ({n_tr/len(n_active_train):>5.1%})  {n_ev:>5} ({n_ev/len(n_active_eval):>5.1%})")
print(f"\n  train mean: {n_active_train.mean():.3f}  (expected ~1.857 = 2*6/7 + 1*1/7)")
print(f"  eval  mean: {n_active_eval.mean():.3f}")
```

**▶ 실행 결과**

```text
Aux label (n_active) distribution:
  value     train      eval
      1    732 (14.6%)    242 (24.2%)
      2   4268 (85.4%)    758 (75.8%)

  train mean: 1.854  (expected ~1.857 = 2*6/7 + 1*1/7)
  eval  mean: 1.758
```

**결과 해석**

두 헤드라인을 무작위로 짝지으므로 보조 라벨 `n_active`는 대부분 2(train 85.4%)이고 1은 소수입니다. 평균 1.854가 이론값 1.857과 거의 일치해 합성 데이터가 의도대로 만들어졌음을 확인할 수 있습니다.

`klue/bert-base` 토크나이저를 불러와 결합 헤드라인의 토큰 길이를 먼저 살펴본 뒤 `tokenize_fn`으로 데이터를 인코딩합니다. 메인 라벨은 7차원 multi-hot float, 보조 라벨은 활성 개수를 담은 float 스칼라로 따로 만들어 두 헤드가 같은 배치에서 학습되도록 준비합니다.

```python
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 결합 헤드라인 토큰 길이 미리 보기
sample_lens = [len(tokenizer.encode(t)) for t in train_full["text"][:200]]
print(f"Token length (combined, sample 200): "
      f"mean={np.mean(sample_lens):.1f}, median={np.median(sample_lens):.0f}, max={max(sample_lens)}")


def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # 메인: multi-hot 7차원 float
    out["labels"]   = [list(map(float, mh)) for mh in batch["multi_hot"]]
    # 보조: float scalar (활성 개수)
    out["n_active"] = [float(n) for n in batch["n_active"]]
    return out


keep = ("input_ids", "attention_mask", "token_type_ids", "labels", "n_active")
train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(
    [c for c in train_full.column_names if c not in keep]
)
eval_tok = eval_full.map(tokenize_fn, batched=True).remove_columns(
    [c for c in eval_full.column_names if c not in keep]
)

print(train_tok)
print(f"\nFirst sample labels:    {train_tok[0]['labels']}  (length-7 multi-hot float)")
print(f"First sample n_active:  {train_tok[0]['n_active']}  (aux scalar)")
```

**▶ 실행 결과**

```text
Token length (combined, sample 200): mean=30.0, median=30, max=41
Dataset({
    features: ['n_active', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample labels:    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]  (length-7 multi-hot float)
First sample n_active:  2  (aux scalar)
```

배치를 만들 때 스칼라 보조 라벨 `n_active`는 표준 패딩과 섞이지 않으므로, 이를 먼저 떼어 낸 뒤 나머지를 `DataCollatorWithPadding`으로 패딩하고 마지막에 다시 붙이는 커스텀 collator입니다. 첫 4개 샘플로 배치를 만들어 각 키의 shape와 dtype이 의도대로 나오는지 확인합니다.

```python
class AuxCollator:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        # 1. n_active 분리
        n_act = torch.tensor([f.pop("n_active") for f in features], dtype=torch.float)
        # 2. 나머지(input_ids/attention_mask/labels)는 표준 padding
        batch = self.base(features)
        # 3. labels 가 multi-hot float 이므로 dtype 보정
        batch["labels"] = batch["labels"].float()
        # 4. 보조 라벨 추가
        batch["n_active"] = n_act
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
Batch keys: ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'n_active']
  input_ids: shape=(4, 32), dtype=torch.int64
  token_type_ids: shape=(4, 32), dtype=torch.int64
  attention_mask: shape=(4, 32), dtype=torch.int64
  labels: shape=(4, 7), dtype=torch.float32
  n_active: shape=(4,), dtype=torch.float32
```

KLUE-BERT 본체를 공유하면서 메인 헤드(multi-label 7차원)와 보조 헤드(활성 개수 회귀 1차원)를 함께 얹은 멀티태스크 모델입니다. forward 안에서 메인 BCE 손실과 보조 MSE 손실을 `l_main + lambda_aux * l_aux`로 결합하는 부분이 이 장의 핵심이니 눈여겨봐 주세요. 두 헤드가 본체에 비해 파라미터를 거의 더하지 않는다는 점도 함께 확인합니다.

```python
class KoBertMultiTask(nn.Module):
    '''KLUE-BERT 본체 공유 + 메인(multi-label 7) + 보조(count regression 1).

    forward 가 dict 반환 — Trainer 가 outputs.loss / outputs.logits 형태로 사용.
    '''
    def __init__(self, model_name: str, num_labels: int = 7):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        H = self.bert.config.hidden_size
        # 메인 헤드: multi-label 카테고리 logits
        self.cls_head   = nn.Linear(H, num_labels)
        # 보조 헤드: 활성 개수 회귀 (스칼라)
        self.count_head = nn.Linear(H, 1)
        # config 일부 — id2label 보존용
        self.config = self.bert.config

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, n_active=None, lambda_aux: float = 0.1):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.bert(**kwargs)
        # CLS hidden (B, H)
        cls = out.last_hidden_state[:, 0, :]

        main_logits = self.cls_head(cls)                # (B, K)
        count_pred  = self.count_head(cls).squeeze(-1)  # (B,)

        loss = None
        if labels is not None and n_active is not None:
            l_main = F.binary_cross_entropy_with_logits(main_logits, labels.float())
            l_aux  = F.mse_loss(count_pred, n_active.float())
            loss = l_main + lambda_aux * l_aux

        # Trainer 와 호환되도록 SequenceClassifierOutput 형태로 반환 (loss + logits)
        # count_pred 는 self.last_count_pred 에 보관 (eval 단계에서 따로 추출)
        self.last_count_pred = count_pred.detach()
        return SequenceClassifierOutput(loss=loss, logits=main_logits)


def make_model(model_name="klue/bert-base"):
    return KoBertMultiTask(model_name, num_labels=K)


model = make_model()


def param_summary(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    aux_only  = sum(p.numel() for n, p in m.named_parameters() if n.startswith("count_head"))
    main_only = sum(p.numel() for n, p in m.named_parameters() if n.startswith("cls_head"))
    return total, trainable, main_only, aux_only


total, trainable, main_only, aux_only = param_summary(model)
print(f"Parameters:           {total:>13,}  ({total/1e6:.1f} M)")
print(f"Trainable parameters: {trainable:>13,}  ({trainable/total:.1%})")
print(f"Main head params:     {main_only:>13,}  ({main_only/total:.4%})")
print(f"Aux  head params:     {aux_only:>13,}  ({aux_only/total:.4%})")
print(f"Main head: {model.cls_head}")
print(f"Aux  head: {model.count_head}")
```

**▶ 실행 결과**

```text
[transformers] BertModel LOAD REPORT from: klue/bert-base
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Parameters:             110,623,496  (110.6 M)
Trainable parameters:   110,623,496  (100.0%)
Main head params:             5,383  (0.0049%)
Aux  head params:               769  (0.0007%)
Main head: Linear(in_features=768, out_features=7, bias=True)
Aux  head: Linear(in_features=768, out_features=1, bias=True)
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:52:05 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   41C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

기본 `Trainer`의 `compute_loss`만 교체해 forward에 `lambda_aux`를 넘기는 커스텀 Trainer입니다. 손실 결합은 모델이 직접 하므로 여기서는 λ 값을 전달하기만 하면 되고, λ를 0으로 주면 보조 항을 끈 baseline으로도 그대로 재사용할 수 있습니다.

```python
class AuxTrainer(Trainer):
    def __init__(self, *args, lambda_aux: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_aux = lambda_aux

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # forward 에 lambda_aux 전달 — 모델이 combined loss 계산
        inputs = {**inputs, "lambda_aux": self.lambda_aux}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


print("AuxTrainer defined — only Trainer.compute_loss is overridden.")
```

**▶ 실행 결과**

```text
AuxTrainer defined — only Trainer.compute_loss is overridden.
```

메인 multi-label task를 평가하는 지표 함수로, logits에 sigmoid를 씌워 0.5 임계값으로 예측을 만든 뒤 hamming loss와 micro/macro F1·precision·recall, macro-AUC를 계산합니다. 보조 task는 여기서 다루지 않고 뒤에서 따로 평가합니다.

```python
def compute_metrics_main(eval_pred):
    # 메인 task 평가 — Ch 17과 동일
    logits, labels = eval_pred
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

λ=0.1로 보조 항을 켠 멀티태스크 모델을 실제로 학습합니다. T4 제약에 맞춰 2 에폭·batch 16·fp16으로 돌리고, `n_active` 컬럼이 자동 제거되지 않도록 `remove_unused_columns=False`를 둔 점을 눈여겨봐 주세요.

```python
LAMBDA_AUX = 0.1

training_args = TrainingArguments(
    output_dir="./ch18_aux_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=True,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED,
    remove_unused_columns=False,   # ← n_active 가 model.forward 시그니처에 있긴 하지만
                                   #    안전상 자동 제거를 끔
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
With-aux training done — mean train loss: 0.2477
```

**결과 해석**

결합 손실(main + 0.1·aux) 기준 평균 학습 loss가 0.2477까지 내려가 두 헤드가 함께 수렴했음을 보여줍니다. 이 값은 메인 BCE에 보조 MSE가 더해진 합이라 뒤의 baseline(λ=0) loss와 직접 비교하지는 않습니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:52:48 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   62C    P0             35W /   70W |    2189MiB /  15360MiB |     39%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2229      C   /usr/bin/python3                       2186MiB |
+-----------------------------------------------------------------------------------------+
```

학습된 멀티태스크 모델로 eval 셋의 메인 task 지표를 뽑아 봅니다. 여기서 나오는 micro-F1·macro-AUC 등이 뒤에서 보조 항 없는 baseline과 비교할 기준점이 됩니다.

```python
# 메인 metric
eval_metrics_aux = trainer_aux.evaluate()
print("With-aux (lambda=0.1) — main task metrics:")
for k, v in eval_metrics_aux.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
With-aux (lambda=0.1) — main task metrics:
               eval_loss: 0.2115
       eval_hamming_loss: 0.0737
           eval_micro_f1: 0.8521
    eval_micro_precision: 0.8590
       eval_micro_recall: 0.8453
           eval_macro_f1: 0.8470
    eval_macro_precision: 0.8392
       eval_macro_recall: 0.8577
          eval_macro_auc: 0.9628
            eval_runtime: 0.8829
  eval_samples_per_second: 1132.6520
   eval_steps_per_second: 36.2450
```

**결과 해석**

보조 헤드를 더한 메인 태스크는 micro-F1 0.8521, macro-AUC 0.9628로 한국어 multi-label을 안정적으로 풀어냅니다. 이 값들은 뒤에서 보조 항 없는 baseline과 비교할 기준점이 됩니다.

보조 헤드의 예측은 `Trainer.evaluate`가 자동으로 모아 주지 않으므로, eval 셋 전체를 직접 forward해 `last_count_pred`에 보관된 활성 개수 예측을 수집합니다. 이렇게 모은 예측으로 RMSE·R^2·Pearson을 계산해 보조 회귀가 얼마나 잘 맞는지 확인합니다.

```python
# 보조 metric — eval 전체에 대해 수동 forward
@torch.no_grad()
def aux_predictions(trainer, dataset, batch_size=64):
    trainer.model.eval()
    device = trainer.model.bert.device
    aux_preds, aux_true = [], []
    for i in range(0, len(dataset), batch_size):
        batch_features = [dict(dataset[j]) for j in range(i, min(i + batch_size, len(dataset)))]
        batch = trainer.data_collator(batch_features)
        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        n_act_true = batch_on_device.pop("n_active").cpu().numpy()
        # 메인 labels 도 잠시 제거 (forward 에서 loss 계산 안 하도록)
        batch_on_device.pop("labels", None)
        _ = trainer.model(**batch_on_device, labels=None, n_active=None)
        count_pred = trainer.model.last_count_pred.cpu().numpy()
        aux_preds.extend(count_pred.tolist())
        aux_true.extend(n_act_true.tolist())
    return np.array(aux_preds), np.array(aux_true)


aux_preds_aux, aux_true = aux_predictions(trainer_aux, eval_tok)
rmse_aux = float(np.sqrt(mean_squared_error(aux_true, aux_preds_aux)))
r2_aux   = float(r2_score(aux_true, aux_preds_aux))
pear_aux = float(np.corrcoef(aux_true, aux_preds_aux)[0, 1])

print("\nWith-aux (lambda=0.1) — aux task metrics (n_active regression):")
print(f"  RMSE:    {rmse_aux:.4f}")
print(f"  R^2:     {r2_aux:.4f}")
print(f"  Pearson: {pear_aux:.4f}")
print(f"\n  Aux pred range: [{aux_preds_aux.min():.3f}, {aux_preds_aux.max():.3f}]")
print(f"  Aux true range: [{aux_true.min():.1f}, {aux_true.max():.1f}]")
```

**▶ 실행 결과**

```text
With-aux (lambda=0.1) — aux task metrics (n_active regression):
  RMSE:    0.3980
  R^2:     0.1366
  Pearson: 0.4851

  Aux pred range: [1.002, 2.514]
  Aux true range: [1.0, 2.0]
```

**결과 해석**

보조 회귀는 Pearson 0.485, R^2 0.137로 활성 개수의 방향성은 어느 정도 잡지만 정확히 맞히지는 못합니다. 라벨이 1·2 두 값뿐이고 분포가 2로 크게 치우쳐 있어 학습 신호가 약한 점이 한계로 작용합니다.

뒤의 카테고리별 비교를 위해 메인 task의 per-sample 예측을 미리 만들어 둡니다. logits에 sigmoid를 씌워 확률로 바꾸고 0.5 임계값으로 multi-hot 예측을 만든 결과를 보관합니다.

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
Main logits shape: (1000, 7)
Eval samples:      1000
```

보조 항을 켠 모델의 카테고리별 precision·recall·F1을 `classification_report`로 한눈에 봅니다. 7개 토픽 각각에서 성능이 어떻게 갈리는지 확인하는 단계입니다.

```python
# Per-category classification report (with-aux)
print("Per-category report — with aux (lambda=0.1):")
print(classification_report(
    labels_eval, preds_main_aux,
    target_names=LABEL_NAMES_EN,
    digits=4, zero_division=0,
))
```

**▶ 실행 결과**

```text
Per-category report — with aux (lambda=0.1):
              precision    recall  f1-score   support

  IT/Science     0.7634    0.7937    0.7782       126
     Economy     0.8214    0.7731    0.7965       238
     Society     0.9186    0.8246    0.8690       684
Life&Culture     0.8378    0.8921    0.8641       278
       World     0.8742    0.8742    0.8742       159
      Sports     0.8810    0.9487    0.9136       117
    Politics     0.7778    0.8974    0.8333       156

   micro avg     0.8590    0.8453    0.8521      1758
   macro avg     0.8392    0.8577    0.8470      1758
weighted avg     0.8625    0.8453    0.8522      1758
 samples avg     0.8823    0.8625    0.8530      1758
```

같은 데이터·설정으로 λ=0 baseline을 새 모델 인스턴스에 학습합니다. 보조 항만 끄고 나머지를 그대로 두어, 뒤에서 보조 항의 효과를 공정하게 분리해 비교하기 위한 대조군입니다.

```python
# 새 모델 인스턴스 — λ=0 학습용
model_no_aux = make_model()

training_args_no_aux = TrainingArguments(
    output_dir="./ch18_baseline_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=True,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED,
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
print(f"\nNo-aux (lambda=0) baseline training done — mean train loss: {train_result_no_aux.training_loss:.4f}")
```

**▶ 실행 결과**

```text
[transformers] BertModel LOAD REPORT from: klue/bert-base
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
<IPython.core.display.HTML object>
No-aux (lambda=0) baseline training done — mean train loss: 0.2213
```

baseline 모델의 메인 task 지표와 per-sample 예측을 뽑습니다. 보조를 켠 모델과 같은 방식으로 평가해 두 결과를 곧바로 나란히 놓고 비교할 수 있게 합니다.

```python
# baseline 메인 metric
eval_metrics_no_aux = trainer_no_aux.evaluate()
print("No-aux (lambda=0) baseline — main task metrics:")
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
No-aux (lambda=0) baseline — main task metrics:
               eval_loss: 0.1867
       eval_hamming_loss: 0.0701
           eval_micro_f1: 0.8600
    eval_micro_precision: 0.8622
       eval_micro_recall: 0.8578
           eval_macro_f1: 0.8561
    eval_macro_precision: 0.8435
       eval_macro_recall: 0.8708
          eval_macro_auc: 0.9653
            eval_runtime: 0.7025
  eval_samples_per_second: 1423.4440
   eval_steps_per_second: 45.5500
<IPython.core.display.HTML object>
```

**결과 해석**

보조 항을 끈 baseline은 micro-F1 0.8600, macro-F1 0.8561로 보조를 더한 쪽(0.8521, 0.8470)보다 오히려 높습니다. 이 셋업에서는 보조 태스크가 메인 성능을 끌어올리지 못한다는 신호입니다.

두 모델의 공통 지표를 한 표로 모아 `delta (aux - no_aux)` 열로 보조 항이 각 지표를 얼마나 올리거나 내렸는지 한눈에 봅니다. delta의 부호와 크기가 이 비교의 핵심 결론입니다.

```python
m_aux    = {k.replace("eval_", ""): v for k, v in eval_metrics_aux.items()
            if k.startswith("eval_") and isinstance(v, float)}
m_no_aux = {k.replace("eval_", ""): v for k, v in eval_metrics_no_aux.items()
            if k.startswith("eval_") and isinstance(v, float)}

common = [k for k in m_aux if k in m_no_aux]
cmp = pd.DataFrame({
    "metric":               common,
    "no aux (lambda=0)":    [m_no_aux[k] for k in common],
    "with aux (lambda=0.1)":[m_aux[k]    for k in common],
})
cmp["delta (aux - no_aux)"] = cmp["with aux (lambda=0.1)"] - cmp["no aux (lambda=0)"]
print(cmp.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
            metric  no aux (lambda=0)  with aux (lambda=0.1)  delta (aux - no_aux)
              loss             0.1867                 0.2115                0.0248
      hamming_loss             0.0701                 0.0737                0.0036
          micro_f1             0.8600                 0.8521               -0.0079
   micro_precision             0.8622                 0.8590               -0.0032
      micro_recall             0.8578                 0.8453               -0.0125
          macro_f1             0.8561                 0.8470               -0.0091
   macro_precision             0.8435                 0.8392               -0.0043
      macro_recall             0.8708                 0.8577               -0.0131
         macro_auc             0.9653                 0.9628               -0.0025
           runtime             0.7025                 0.8829                0.1804
samples_per_second          1423.4440              1132.6520             -290.7920
  steps_per_second            45.5500                36.2450               -9.3050
```

**결과 해석**

micro-F1 -0.0079, macro-F1 -0.0091, macro-AUC -0.0025로 모든 메인 지표의 delta가 음수입니다. 영어 Ch 14와 마찬가지로 이 데이터에서는 λ=0.1의 보조 항이 도움이 되기보다 메인 성능을 소폭 깎습니다.

이번에는 전체 평균이 아니라 카테고리별 F1을 두 모델에 대해 따로 계산해 표와 막대그래프로 비교합니다. 보조 항의 영향이 특정 토픽에 몰리는지, 아니면 전반에 고르게 퍼지는지를 살펴보는 단계입니다.

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
    "category":              LABEL_NAMES_EN,
    "no aux F1":             f1_no_aux,
    "with aux F1":           f1_aux,
    "delta (aux - no_aux)":  np.array(f1_aux) - np.array(f1_no_aux),
})
print(label_cmp.round(4).to_string(index=False))

# 막대 그래프
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(11, 5))
x_pos = np.arange(K)
width = 0.38
ax.bar(x_pos - width/2, f1_no_aux, width, label="no aux (lambda=0)",    color="#5B8DEF")
ax.bar(x_pos + width/2, f1_aux,    width, label="with aux (lambda=0.1)", color="#F47272")
ax.set_xticks(x_pos)
ax.set_xticklabels(LABEL_NAMES_EN, rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Per-label F1")
ax.set_title("Per-category F1 — auxiliary loss effect (Korean multi-label)")
ax.legend()
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

```text
    category  no aux F1  with aux F1  delta (aux - no_aux)
  IT/Science     0.7739       0.7782                0.0043
     Economy     0.8213       0.7965               -0.0247
     Society     0.8736       0.8690               -0.0045
Life&Culture     0.8646       0.8641               -0.0005
       World     0.8869       0.8742               -0.0126
      Sports     0.9256       0.9136               -0.0120
    Politics     0.8466       0.8333               -0.0133
```

![output](../assets/18-ko_auxiliary-out1.png)

**결과 해석**

카테고리별로 보면 IT/Science만 +0.0043으로 미세하게 오르고 Economy(-0.0247), Politics(-0.0133) 등 나머지는 모두 떨어집니다. 보조 항의 영향이 특정 카테고리에 집중되기보다 전반적으로 약간의 손해로 나타납니다.

보조 회귀의 예측을 정답 `n_active`(1·2) 그룹별로 나눠 violin plot으로 그립니다. 점선은 정답 위치이므로, 예측 분포가 그 선 근처에 모이는지 또는 다수값 쪽으로 쏠리는지를 시각적으로 확인할 수 있습니다.

```python
# True n_active 별 예측 분포 — violin
df_aux = pd.DataFrame({
    "True n_active": [f"{int(v)}" for v in aux_true],
    "Predicted":     aux_preds_aux,
})
order = ["1", "2"]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.violinplot(
    data=df_aux, x="True n_active", y="Predicted",
    order=order, inner="quart", cut=0,
    color="#F47272", alpha=0.6, ax=ax,
)
# 정답 위치 점선 가이드
for i, target in enumerate([1.0, 2.0]):
    ax.hlines(target, i - 0.4, i + 0.4, color="black", lw=1.1, ls="--", alpha=0.7)
ax.set_ylim(0.0, 3.0)
ax.set_title(f"Aux task — predicted vs true n_active  (RMSE={rmse_aux:.3f}, r={pear_aux:.3f})")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/18-ko_auxiliary-out2.png)

**결과 해석**

True n_active=2 그룹의 예측 분포는 점선(정답 2.0) 근처에 모이지만, n_active=1 그룹은 위로 넓게 퍼져 1을 과대 예측하는 경향이 보입니다. 1이 소수라 보조 헤드가 다수값 2 쪽으로 끌려가는 분포 불균형이 그대로 드러납니다.
