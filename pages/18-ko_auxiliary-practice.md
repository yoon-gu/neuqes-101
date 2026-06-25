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

# matplotlib 한글 폰트 (Colab — NanumGothic). plot 의 한국어가 □ 로 깨지지 않게.
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, subprocess, os
_fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(_fp):
    subprocess.run("apt-get -qq -y install fonts-nanum", shell=True)
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "NanumGothic"
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

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 24 21:40:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   35C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 데이터 — KLUE-YNAT 합성 multi-label + 활성 개수 보조 라벨

Ch 17 의 `make_multilabel` 을 *그대로* 가져옵니다. 함수 안에서 이미 `n_active` (활성 라벨 개수) 컬럼이 만들어지고 있어 보조 라벨로 그대로 사용 가능 — *합성 과정의 자연스러운 부산물*.

| 라벨 | 카테고리 |
|---|---|
| 0 | IT과학 |
| 1 | 경제 |
| 2 | 사회 |
| 3 | 생활문화 |
| 4 | 세계 |
| 5 | 스포츠 |
| 6 | 정치 |

> **합성 규칙 (Ch 17 동일)** — 두 헤드라인 A, B 를 `" [SEP] "` 로 연결, multi-hot 라벨에서 $c_A, c_B$ 위치를 1 로. 우연히 $c_A = c_B$ 면 활성 개수 1, 다르면 2. 7카테고리에서 무작위 결합이므로 $P(c_A = c_B) = 1/7$ → 평균 `n_active` 약 $2 \cdot 6/7 + 1 \cdot 1/7 \approx 1.86$.

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

### 1-1. 합성 함수 — Ch 17 의 `make_multilabel` 재사용

`n_active` (활성 개수) 컬럼이 합성 시 만들어집니다. Ch 18 의 보조 task 정답이 바로 이 값.

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

## 토큰화 — 메인 multi-hot + 보조 `n_active` 같이 부착

Ch 14 의 `aux_labels` 패턴 그대로 — `tokenize_fn` 이 두 라벨을 모두 attach. 메인은 `labels` (multi-hot 7차원 float), 보조는 `n_active` (float scalar).

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

## 커스텀 Data Collator — `n_active` 도 batch 에 같이 담기

Ch 14 의 `AuxCollator` 패턴 그대로. 기본 `DataCollatorWithPadding` 은 `input_ids`/`attention_mask`/`labels` 만 알고 있어 *추가 라벨* 은 통과시키지 못합니다. wrapper 로 `n_active` 를 텐서로 만들어 batch 에 추가.

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

## 모델 — `AutoModel` 본체 + 메인 헤드 + 보조 헤드 직접 부착

Ch 14 는 `AutoModelForSequenceClassification` 의 자동 매핑을 *그대로* 쓰면서 `model.aux_head = nn.Linear(...)` 한 줄로 보조 헤드를 attach 했습니다. Ch 18 도 같은 패턴이 가능하지만, *두 헤드를 명시적으로 한 클래스에서 관리* 하는 패턴이 multi-task 의 정통 — 이번엔 **`nn.Module` 을 직접 정의** 해 두 헤드를 같은 곳에 둡니다.

두 패턴 모두 결과는 같습니다. 명시 정의가 *디버깅·확장* (e.g. 헤드를 더 추가하거나 layer-wise lr 차등) 에 유리.

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


torch.manual_seed(SEED); np.random.seed(SEED)   # baseline 과 동일 초기화 (λ 만 변수)
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
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Parameters:             110,623,496  (110.6 M)
Trainable parameters:   110,623,496  (100.0%)
Main head params:             5,383  (0.0049%)
Aux  head params:               769  (0.0007%)
Main head: Linear(in_features=768, out_features=7, bias=True)
Aux  head: Linear(in_features=768, out_features=1, bias=True)
```

**보조 헤드는 약 769개 파라미터** — 768→1 Linear 의 weight + bias. 전체 약 110M 의 *0.0007%*. 이 미세한 추가 자유도만으로 multi-task 학습이 동작합니다 (Ch 14 와 동일한 직관).

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 24 21:40:58 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   36C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 커스텀 Trainer — `compute_loss` 오버라이드

핵심 로직 한 줄:

```python
loss = l_main + λ · l_aux       # l_main: BCE per-label, l_aux: MSE on n_active
```

Ch 14 와의 차이 — Ch 14 는 `outputs.loss` (자동 매핑 메인 BCE) 를 그대로 받고 보조만 직접 계산. Ch 18 은 모델 forward 가 *이미* combined loss 를 계산해 반환하므로 `compute_loss` 는 forward 결과를 그대로 돌려주기만 하면 됩니다. λ 만 trainer 에서 model forward 로 넘김.

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


print("AuxTrainer 정의 완료 — Trainer 의 compute_loss 만 교체.")
```

**▶ 실행 결과**

```text
AuxTrainer 정의 완료 — Trainer 의 compute_loss 만 교체.
```

**평가용 metric 함수** — 메인 (Ch 17 과 동일) 만 자동 계산. 보조 metric (RMSE, R², Pearson r) 은 별도 forward 로 `count_pred` 를 추출해 측정 (eval 후 별도 단계).

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

## 학습 — λ=0.05 (sweet spot, 보조 ON)

Ch 17 과 동일한 hyperparams. `AuxTrainer` + `lambda_aux=0.05`. 이 값은 부록 `18_ko_auxiliary_lambda_sweep` 의 λ 스윕에서 **메인 F1 을 가장 끌어올린 지점** 입니다 (λ≥0.2 부터는 §10 처럼 메인이 무너집니다).

```python
LAMBDA_AUX = 0.05

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
With-aux training done — mean train loss: 0.2369
```

**중요: `remove_unused_columns=False`** — Trainer 는 기본으로 *model.forward 시그니처에 없는 컬럼* 을 제거합니다. `n_active` 는 KoBertMultiTask.forward 에 있어 자동 인식되지만, 모델 클래스를 바꿔 끼울 때 위험할 수 있어 명시적으로 끕니다 (Ch 14 와 같은 보호 패턴).

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 24 21:41:41 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   57C    P0             34W /   70W |    2189MiB /  15360MiB |     75%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           16123      C   /usr/bin/python3                       2186MiB |
+-----------------------------------------------------------------------------------------+
```
