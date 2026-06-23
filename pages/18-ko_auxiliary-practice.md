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

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 04:03:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   37C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
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

KLUE-YNAT 뉴스 분류 데이터를 불러옵니다. 원본 7개 카테고리 이름은 한국어이지만, 플롯·출력의 조판 안정성을 위해 영문 라벨도 함께 준비합니다. `title` 컬럼은 이후 코드에서 일관되게 쓰도록 `text` 로 이름을 바꿉니다.

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

single-label 데이터를 두 샘플씩 결합해 multi-label 데이터를 합성합니다. 핵심은 결합 과정에서 보조 라벨 `n_active`(활성 카테고리 개수, 1 또는 2)가 *공짜로* 함께 만들어진다는 점입니다. Ch 17 의 합성 함수를 그대로 가져오되 이 부산물을 보조 task 정답으로 활용합니다.

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

**결과 해석**

`n_active`=2 가 train 의 85.4% 로 압도적입니다 (7카테고리 무작위 결합이라 두 카테고리가 같을 확률은 1/7 뿐). 보조 task 가 *1 vs 2 이항에 가까운 매우 치우친 분포* 라 상수 평균만 예측해도 손실이 작게 유지된다는 점을 미리 염두에 둬야 합니다.

토큰화 단계에서 메인 라벨과 보조 라벨을 *함께* 부착합니다. 메인은 `labels`(multi-hot 7차원 float), 보조는 `n_active`(float scalar)로, 두 신호가 같은 샘플에 묶여 batch 까지 흘러가도록 준비합니다.

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

기본 `DataCollatorWithPadding` 은 `input_ids`/`attention_mask`/`labels` 만 다룰 줄 알아 `n_active` 같은 *추가 라벨* 은 통과시키지 못합니다. wrapper 를 만들어 `n_active` 를 먼저 빼내 텐서로 만들고, 나머지는 표준 padding 을 거친 뒤 batch 에 다시 합칩니다.

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

메인 task 와 보조 task 가 같은 BERT 본체를 공유하도록 커스텀 `nn.Module` 을 직접 정의합니다. 자동 매핑(`AutoModelForSequenceClassification`)을 못 쓰는 *복합 헤드* 구조라, 본체와 두 헤드를 한 클래스에 명시적으로 둬 디버깅·확장이 쉽도록 합니다.

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
```

**위 코드 읽기** — `self.bert` 는 분류 헤드 없는 *본체* 만 (`AutoModel`). 그 위에 `cls_head`(768→7, 메인 multi-label)와 `count_head`(768→1, 보조 회귀) *두 헤드* 를 나란히 둡니다. 두 헤드가 같은 `self.bert` 출력을 공유하는 것이 multi-task 학습의 핵심입니다.

```python
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
```

**위 코드 읽기** — `cls` 는 `[CLS]` 위치의 768차원 벡터로, 두 헤드가 *공유* 하는 표현입니다. 같은 `cls` 에서 `main_logits`(7차원)와 `count_pred`(스칼라)가 갈라져 나옵니다. `forward` 가 `lambda_aux` 를 *인자로* 받는 점도 주목 — 보조 가중치를 trainer 가 동적으로 주입할 수 있게 한 설계입니다.

```python
        loss = None
        if labels is not None and n_active is not None:
            l_main = F.binary_cross_entropy_with_logits(main_logits, labels.float())
            l_aux  = F.mse_loss(count_pred, n_active.float())
            loss = l_main + lambda_aux * l_aux
```

**위 코드 읽기** — 이 챕터의 심장입니다. 메인은 `binary_cross_entropy_with_logits`(per-label BCE), 보조는 `mse_loss`(활성 개수 회귀), 그리고 `loss = l_main + lambda_aux * l_aux` 로 *가중합* 합니다. `lambda_aux=0.1` 이면 보조 MSE 가 메인 BCE 의 1/10 비중으로 섞입니다 — Loss 축에 보조 항을 더하는 변화가 정확히 이 한 줄입니다.

```python
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
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 

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
Mon Jun 22 04:03:27 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   39C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

`Trainer` 의 기본 loss 계산을 교체합니다. 자동 매핑이 못 다루는 *복합 loss* 이므로 `compute_loss` 를 오버라이드해, trainer 에 저장된 λ 를 forward 로 흘려보내고 모델이 계산한 combined loss 를 그대로 받습니다.

```python
class AuxTrainer(Trainer):
    def __init__(self, *args, lambda_aux: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_aux = lambda_aux
```

**위 코드 읽기** — λ 를 trainer 인스턴스에 `self.lambda_aux` 로 보관합니다. 이렇게 두면 같은 trainer 클래스로 λ=0.1 학습과 λ=0 baseline 학습을 *인자만 바꿔* 두 번 돌릴 수 있습니다.

```python
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # forward 에 lambda_aux 전달 — 모델이 combined loss 계산
        inputs = {**inputs, "lambda_aux": self.lambda_aux}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


print("AuxTrainer 정의 완료 — Trainer 의 compute_loss 만 교체.")
```

**위 코드 읽기** — `inputs` 에 `lambda_aux` 를 끼워 넣어 forward 로 전달하는 것이 전부입니다. 실제 combined loss(`l_main + λ·l_aux`)는 *모델 forward 안에서* 이미 계산되므로, `compute_loss` 는 `outputs.loss` 를 그대로 돌려주기만 합니다. Ch 14 가 메인 loss 를 받아 보조를 *여기서* 더했던 것과 달리, Ch 18 은 결합을 모델 쪽으로 옮긴 구조입니다.

**▶ 실행 결과**

```text
AuxTrainer 정의 완료 — Trainer 의 compute_loss 만 교체.
```

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

보조 ON 상태(λ=0.1)로 학습합니다. Ch 17 과 모든 hyperparams 가 동일하고, `remove_unused_columns=False` 로 `n_active` 컬럼이 collator·forward 까지 안전하게 전달되도록 합니다.

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
With-aux training done — mean train loss: 0.2448
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 04:04:11 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   62C    P0             78W /   70W |    2189MiB /  15360MiB |     54%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2860      C   /usr/bin/python3                       2186MiB |
+-----------------------------------------------------------------------------------------+
```

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
               eval_loss: 0.2057
       eval_hamming_loss: 0.0714
           eval_micro_f1: 0.8562
    eval_micro_precision: 0.8661
       eval_micro_recall: 0.8464
           eval_macro_f1: 0.8507
    eval_macro_precision: 0.8459
       eval_macro_recall: 0.8579
          eval_macro_auc: 0.9643
            eval_runtime: 0.6873
  eval_samples_per_second: 1455.0020
   eval_steps_per_second: 46.5600
```

**결과 해석**

보조 ON 의 메인 micro-F1 은 0.8562, macro-F1 은 0.8507 입니다. 이 값만으로는 좋고 나쁨을 알 수 없고, 뒤에서 학습하는 λ=0 baseline 과 비교해야 보조 loss 의 효과를 판단할 수 있습니다.

보조 헤드의 예측(`count_pred`)은 `SequenceClassifierOutput` 표준 필드가 아니라 `model.last_count_pred` 에 보관됩니다. 보조 metric 을 측정하려면 eval 셋 전체를 수동 forward 해 이 값을 꺼내고, RMSE·R²·Pearson r 로 회귀 성능을 잽니다.

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
  RMSE:    0.4011
  R^2:     0.1228
  Pearson: 0.4766

  Aux pred range: [1.029, 2.705]
  Aux true range: [1.0, 2.0]
```

**결과 해석**

보조 회귀의 R² 는 0.123 으로 *약합니다*. Pearson r 0.477 로 입력 의존적 학습이 일어나긴 했지만, `n_active` 가 1/7·6/7 로 치우친 *너무 쉬운* 신호라 상수 평균 예측에서 크게 벗어나지 못한 모습입니다. 보조 task 자체가 이 정도로 약하면 메인 정규화 효과도 기대하기 어렵습니다.

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

  IT/Science     0.7559    0.7619    0.7589       126
     Economy     0.8604    0.8025    0.8304       238
     Society     0.9199    0.8231    0.8688       684
Life&Culture     0.8425    0.8849    0.8632       278
       World     0.8675    0.9057    0.8862       159
      Sports     0.8880    0.9487    0.9174       117
    Politics     0.7874    0.8782    0.8303       156

   micro avg     0.8661    0.8464    0.8562      1758
   macro avg     0.8459    0.8579    0.8507      1758
weighted avg     0.8692    0.8464    0.8562      1758
 samples avg     0.8875    0.8635    0.8561      1758
```

같은 코드를 `lambda_aux=0.0` 으로 한 번 더 돌려 baseline 을 만듭니다. 보조 loss 의 gradient 가 0 이 되어 메인 task 만 학습되는 상태 = Ch 17 과 동일한 결과입니다. 같은 노트북·같은 환경에서 비교가 self-contained 하도록 의도한 구성입니다.

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
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
<IPython.core.display.HTML object>
No-aux (lambda=0) baseline training done — mean train loss: 0.2213
```

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
            eval_runtime: 0.7098
  eval_samples_per_second: 1408.8240
   eval_steps_per_second: 45.0820
<IPython.core.display.HTML object>
```

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
              loss             0.1867                 0.2057                0.0191
      hamming_loss             0.0701                 0.0714                0.0013
          micro_f1             0.8600                 0.8562               -0.0038
   micro_precision             0.8622                 0.8661                0.0039
      micro_recall             0.8578                 0.8464               -0.0114
          macro_f1             0.8561                 0.8507               -0.0053
   macro_precision             0.8435                 0.8459                0.0024
      macro_recall             0.8708                 0.8579               -0.0129
         macro_auc             0.9653                 0.9643               -0.0010
           runtime             0.7098                 0.6873               -0.0225
samples_per_second          1408.8240              1455.0020               46.1780
  steps_per_second            45.0820                46.5600                1.4780
```

**결과 해석**

보조 loss 는 메인 task 를 *살짝 깎았습니다*: micro-F1 0.8600 → 0.8562 (Δ-0.0038), macro-F1 0.8561 → 0.8507 (Δ-0.0053) 로 delta 의 부호가 모두 *음(-)* 입니다. λ 를 0.1 로 낮춰 저하 폭은 줄였지만 방향은 여전히 마이너스 — 보조 신호가 메인에 도움이 되지 못한 셋업입니다.

카테고리별 F1 을 baseline 과 보조 ON 으로 나란히 계산해 막대 그래프로 비교합니다. 어느 카테고리가 보조 loss 로 도움받았는지(혹은 손해를 봤는지)를 한눈에 봅니다.

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
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
fig, ax = plt.subplots(figsize=(11, 5))
x_pos = np.arange(K)
width = 0.38
ax.bar(x_pos - width/2, f1_no_aux, width, label="aux 없음 (lambda=0)",    color="#5B8DEF")
ax.bar(x_pos + width/2, f1_aux,    width, label="aux 적용 (lambda=0.1)", color="#F47272")
ax.set_xticks(x_pos)
ax.set_xticklabels(LABEL_NAMES_EN, rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("라벨별 F1")
ax.set_title("카테고리별 F1 — 보조 loss 효과 (한국어 multi-label)")
ax.legend()
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

```text
    category  no aux F1  with aux F1  delta (aux - no_aux)
  IT/Science     0.7739       0.7589               -0.0151
     Economy     0.8213       0.8304                0.0092
     Society     0.8736       0.8688               -0.0047
Life&Culture     0.8646       0.8632               -0.0014
       World     0.8869       0.8862               -0.0007
      Sports     0.9256       0.9174               -0.0083
    Politics     0.8466       0.8303               -0.0163
```

**결과 해석**

카테고리별로 봐도 6/7 이 음수이고, 가장 크게 떨어진 곳은 Politics(-0.0163)·IT/Science(-0.0151) 입니다. 유일하게 오른 Economy(+0.0092)도 delta 가 quick 모드 노이즈 영역(±0.01) 안이라 보조 효과로 보기 어렵습니다.

![output](../assets/18-ko_auxiliary-out1.png)

보조 task 가 활성 개수를 얼마나 잘 구분하는지 violin 으로 봅니다. 실제 `n_active`=1, 2 각각에서 예측값 분포가 점선 가이드(1.0/2.0)에 모이면 보조 헤드가 잘 학습된 것입니다.

```python
# True n_active 별 예측 분포 — violin
df_aux = pd.DataFrame({
    "실제 n_active": [f"{int(v)}" for v in aux_true],
    "예측값":     aux_preds_aux,
})
order = ["1", "2"]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.violinplot(
    data=df_aux, x="실제 n_active", y="예측값",
    order=order, inner="quart", cut=0,
    color="#F47272", alpha=0.6, ax=ax,
)
# 정답 위치 점선 가이드
for i, target in enumerate([1.0, 2.0]):
    ax.hlines(target, i - 0.4, i + 0.4, color="black", lw=1.1, ls="--", alpha=0.7)
ax.set_ylim(0.0, 3.0)
ax.set_title(f"보조 task — 예측 n_active vs 실제 n_active  (RMSE={rmse_aux:.3f}, r={pear_aux:.3f})")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/18-ko_auxiliary-out2.png)
