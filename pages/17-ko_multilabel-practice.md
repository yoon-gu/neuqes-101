> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/17_ko_multilabel/17_ko_multilabel.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report,
    roc_auc_score, hamming_loss,
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
Wed Jun 17 21:49:36 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   40C    P8             12W /   70W |       3MiB /  15360MiB |      0%      Default |
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

KLUE-YNAT 뉴스 헤드라인 분류 데이터를 내려받아 split 크기와 라벨 종류를 확인합니다. 원본은 헤드라인 하나에 카테고리 하나가 붙은 single-label 데이터이므로, 7개 한국어 카테고리 이름을 영문으로 매핑해 두고 출력·플롯에서는 영문을 씁니다. `title` 컬럼은 `transformers` 표준에 맞춰 `text` 로 이름을 바꿉니다.

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

# title 컬럼명을 'text' 로 통일 (transformers 표준)
ds = ds.rename_column("title", "text")
print(f"\nfirst 2 raw samples:")
for ex in ds["train"].select(range(2)):
    print(f"  label={ex['label']} ({LABEL_NAMES_EN[ex['label']]:>8})  text={ex['text']!r}")
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

single-label 데이터로 multi-label 문제를 만들기 위해 두 헤드라인을 ` [SEP] ` 로 이어 붙여 합성 샘플을 생성합니다. 결합한 두 헤드라인의 카테고리를 모두 1로 켜서 multi-hot 라벨을 만들고, 두 카테고리가 같으면 활성 라벨이 1개가 됩니다. 학습 5,000개·평가 1,000개를 만들고 첫 샘플의 결합 텍스트와 multi-hot 라벨을 살펴봅니다.

```python
SEED = 42
N_TRAIN = 5000
N_EVAL  = 1000


def make_multilabel(source_split, n_samples, seed):
    '''single-label split 에서 두 샘플씩 결합해 multi-label 데이터셋 합성.

    - 2*n_samples 개 인덱스를 섞어 앞/뒤 절반을 짝으로 묶음
    - 텍스트는 " [SEP] " 로 이어붙임
    - multi-hot 라벨은 두 카테고리 위치를 1 로 (같은 카테고리면 1개만)
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
active0 = [LABEL_NAMES_EN[k] for k in range(K) if train_full[0]['multi_hot'][k] > 0]
print(f"  active categories: {active0}")
```

**▶ 실행 결과**

```text
synthetic train: 5000
synthetic eval:  1000

First synthetic sample:
  text:      신간 언어와 탱크를 응시하며·자본주의 리얼리즘 [SEP] 우리은행 주택금융공사와 도시재생 발굴·지원 업무협약
  multi_hot: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  active categories: ['Economy', 'Life&Culture']
```

합성 데이터가 어떤 분포인지 확인하기 위해 카테고리별 활성률과 샘플당 활성 라벨 개수를 집계합니다. 두 헤드라인을 무작위로 짝지으므로 대부분 샘플은 라벨 2개가 켜지고, 같은 카테고리끼리 뽑힌 경우에만 1개가 됩니다. 라벨 간 빈도 차이가 이후 카테고리별 성능 차이로 이어지므로 미리 눈여겨봅니다.

```python
# 카테고리별 활성률 + 활성 라벨 개수 분포
Y_train = np.array(train_full["multi_hot"])

print("Per-category activation rate (train):")
for k in range(K):
    rate = Y_train[:, k].mean()
    print(f"  {LABEL_NAMES_EN[k]:>9} (label {k}): {rate:.1%}  ({int(Y_train[:, k].sum())} / {len(Y_train)})")

n_active = Y_train.sum(axis=1)
print(f"\nMean active labels per sample: {n_active.mean():.2f}  (expected ~1.86: two draws, occasional collision)")
print(f"Active label distribution (train):")
for n in range(K + 1):
    cnt = int((n_active == n).sum())
    if cnt:
        print(f"  {n} labels active: {cnt} samples ({cnt/len(Y_train):.1%})")
```

**▶ 실행 결과**

```text
Per-category activation rate (train):
  IT/Science (label 0): 22.0%  (1100 / 5000)
    Economy (label 1): 25.2%  (1260 / 5000)
    Society (label 2): 21.8%  (1089 / 5000)
  Life&Culture (label 3): 23.3%  (1165 / 5000)
      World (label 4): 32.8%  (1638 / 5000)
     Sports (label 5): 31.0%  (1549 / 5000)
   Politics (label 6): 29.3%  (1467 / 5000)

Mean active labels per sample: 1.85  (expected ~1.86: two draws, occasional collision)
Active label distribution (train):
  1 labels active: 732 samples (14.6%)
  2 labels active: 4268 samples (85.4%)
```

**결과 해석**

카테고리별 활성률이 21.8%-32.8% 로 어느 한쪽으로 크게 치우치진 않지만 World/Sports 가 IT과학/사회보다 약 1.5배 자주 등장해, 빈도가 낮은 라벨일수록 학습 신호가 적어 recall 이 흔들리기 쉽습니다. 샘플당 평균 1.85개 라벨이 켜지고 85.4% 가 두 라벨을 함께 갖는 multi-label 구조라, 라벨마다 독립 sigmoid 로 푸는 이번 설정이 적절합니다.

`klue/bert-base` 토크나이저를 불러와 결합 헤드라인의 토큰 길이를 먼저 확인하고, `max_length=128` 로 잘라 토큰화합니다. multi-hot 라벨은 `BCEWithLogitsLoss` 가 요구하는 7차원 float 벡터로 `labels` 컬럼에 넣습니다. 모델 입력에 필요 없는 원본 컬럼은 제거합니다.

```python
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 결합 헤드라인 토큰 길이 미리 보기
sample_lens = [len(tokenizer.encode(t)) for t in train_full["text"][:200]]
print(f"Token length (combined, sample 200): "
      f"mean={np.mean(sample_lens):.1f}, median={np.median(sample_lens):.0f}, max={max(sample_lens)}")


def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # multi-hot 7차원 float 벡터 (BCEWithLogitsLoss 가 받는 형식)
    out["labels"] = [list(map(float, mh)) for mh in batch["multi_hot"]]
    return out

keep = ("input_ids", "attention_mask", "token_type_ids", "labels")
train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(
    [c for c in train_full.column_names if c not in keep]
)
eval_tok = eval_full.map(tokenize_fn, batched=True).remove_columns(
    [c for c in eval_full.column_names if c not in keep]
)

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (length-7 multi-hot float vector)")
```

**▶ 실행 결과**

```text
Token length (combined, sample 200): mean=30.0, median=30, max=41
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample label: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]  (length-7 multi-hot float vector)
```

`klue/bert-base` 위에 7개 출력을 가진 분류 헤드를 얹어 모델을 만듭니다. `problem_type="multi_label_classification"` 을 지정하면 `Trainer` 가 라벨마다 독립적인 `BCEWithLogitsLoss` 를 자동으로 적용합니다. 새로 초기화된 분류 헤드와 사전학습 가중치가 어떻게 로드되는지 리포트를 함께 확인합니다.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=K,
    problem_type="multi_label_classification",   # ← BCEWithLogitsLoss per-label 자동 매핑
    id2label={i: name for i, name in enumerate(LABEL_NAMES_EN)},
    label2id={name: i for i, name in enumerate(LABEL_NAMES_EN)},
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
[transformers] BertForSequenceClassification LOAD REPORT from: klue/bert-base
Key                                        | Status     | 
-------------------------------------------+------------+-
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
cls.predictions.transform.dense.bias       | UNEXPECTED | 
cls.seq_relationship.bias                  | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
classifier.bias                            | MISSING    | 
classifier.weight                          | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:             110,622,727  (110.6 M)
Trainable parameters:   110,622,727  (100.0%)
Classifier:           Linear(in_features=768, out_features=7, bias=True)
problem_type:         multi_label_classification
id2label:             {0: 'IT/Science', 1: 'Economy', 2: 'Society', 3: 'Life&Culture', 4: 'World', 5: 'Sports', 6: 'Politics'}
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:49:57 2026       
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

multi-label 평가에 쓸 지표 함수를 정의합니다. logits 에 라벨별 sigmoid 를 적용하고 0.5 임계값으로 multi-hot 예측을 만든 뒤, hamming loss·micro/macro F1·macro AUC 를 계산합니다. micro 는 전체 라벨을 합산해, macro 는 라벨마다 동일 가중치로 평균해 빈도 불균형의 영향을 다르게 비춥니다.

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

    # Macro F1 — 라벨별 F1 을 평균 (각 라벨에 동일 가중치)
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0,
    )
    out["macro_f1"] = float(f1_ma)
    out["macro_precision"] = float(p_ma)
    out["macro_recall"]    = float(r_ma)

    # Per-label AUC 의 macro 평균
    try:
        out["macro_auc"] = float(roc_auc_score(labels, probs, average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    return out
```

학습 설정을 정하고 `Trainer` 로 2 에폭 파인튜닝을 돌립니다. T4 16GB 안에서 끝나도록 batch_size 16·`max_length=128`·`fp16=True` 를 쓰고, 매 에폭마다 평가하도록 `eval_strategy="epoch"` 를 지정합니다. 학습이 끝나면 평균 train loss 를 확인합니다.

```python
training_args = TrainingArguments(
    output_dir="./ch17_output",
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
Training done — mean train loss: 0.2661
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:50:40 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   63C    P0             75W /   70W |    2209MiB /  15360MiB |     74%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3175      C   /usr/bin/python3                       2206MiB |
+-----------------------------------------------------------------------------------------+
```

학습이 끝난 모델을 평가 셋에서 돌려 앞서 정의한 지표들을 한 번에 확인합니다. eval loss 와 함께 hamming loss·micro/macro F1·macro AUC 가 출력되며, micro 와 macro 값을 비교해 라벨 빈도 불균형의 영향을 가늠합니다.

```python
eval_metrics = trainer.evaluate()
print("klue/bert-base KLUE-YNAT multi-label — evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
klue/bert-base KLUE-YNAT multi-label — evaluation:
               eval_loss: 0.2087
       eval_hamming_loss: 0.0724
           eval_micro_f1: 0.8538
    eval_micro_precision: 0.8660
       eval_micro_recall: 0.8419
           eval_macro_f1: 0.8520
    eval_macro_precision: 0.8465
       eval_macro_recall: 0.8596
          eval_macro_auc: 0.9633
            eval_runtime: 0.6860
  eval_samples_per_second: 1457.7020
   eval_steps_per_second: 46.6460
```

**결과 해석**

micro F1 0.8538 과 macro F1 0.8520 이 거의 같아, 라벨 빈도가 비교적 고른 이번 데이터에서는 두 평균이 비슷하게 나옵니다. macro AUC 0.9633 은 threshold 와 무관하게 모델의 순위 매김 자체가 우수함을 뜻하고, hamming loss 0.0724 는 전체 라벨 위치 중 7% 정도만 틀렸다는 의미입니다.

평가 셋 전체에 대한 logits 를 뽑아 sigmoid 확률과 0.5 임계값 예측을 구합니다. 카테고리별로 확률이 어디서부터 어디까지 분포하는지, 그리고 실제 활성률과 예측 활성률이 얼마나 일치하는지 비교합니다. 활성률 차이가 큰 카테고리는 recall 이 흔들리는 지점을 알려줍니다.

```python
# logits → per-label sigmoid → multi-hot 예측
preds_output = trainer.predict(eval_tok)
logits = preds_output.predictions                   # (N, 7)
labels = preds_output.label_ids.astype(int)         # (N, 7) multi-hot
probs  = 1.0 / (1.0 + np.exp(-logits))              # (N, 7) per-label prob
preds  = (probs >= 0.5).astype(int)                 # (N, 7) multi-hot prediction

print(f"logits shape: {logits.shape}")
print(f"prob ranges per category:")
for k in range(K):
    print(f"  {LABEL_NAMES_EN[k]:>9}: [{probs[:, k].min():.4f}, {probs[:, k].max():.4f}]  "
          f"true rate={labels[:, k].mean():.1%}, pred rate={preds[:, k].mean():.1%}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
logits shape: (1000, 7)
prob ranges per category:
  IT/Science: [0.0076, 0.9774]  true rate=12.6%, pred rate=13.0%
    Economy: [0.0142, 0.9856]  true rate=23.8%, pred rate=23.0%
    Society: [0.0364, 0.9659]  true rate=68.4%, pred rate=60.1%
  Life&Culture: [0.0114, 0.9887]  true rate=27.8%, pred rate=28.6%
      World: [0.0088, 0.9889]  true rate=15.9%, pred rate=16.7%
     Sports: [0.0091, 0.9888]  true rate=11.7%, pred rate=12.7%
   Politics: [0.0087, 0.9853]  true rate=15.6%, pred rate=16.8%
```

**결과 해석**

확률 범위가 모든 카테고리에서 0.01 부근부터 0.98 부근까지 양극단으로 벌어져, 모델이 확신을 갖고 예측하고 있음을 보여줍니다. 예측 활성률이 실제 활성률과 대체로 맞아떨어지지만 Society 만 실제 68.4% 대비 예측 60.1% 로 낮아, 가장 빈번한 카테고리에서 일부를 놓치고 있어 recall 저하로 이어집니다.

카테고리별 precision·recall·F1 을 표로 출력해 어느 라벨이 약한지 자세히 봅니다. support(실제 양성 개수)가 적은 라벨일수록 학습 신호가 부족해 F1 이 떨어지는 경향을 함께 확인합니다.

```python
# Per-category classification report
print(classification_report(
    labels, preds,
    target_names=LABEL_NAMES_EN,
    digits=4, zero_division=0,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

  IT/Science     0.7538    0.7778    0.7656       126
     Economy     0.8304    0.8025    0.8162       238
     Society     0.9151    0.8041    0.8560       684
Life&Culture     0.8706    0.8957    0.8830       278
       World     0.8623    0.9057    0.8834       159
      Sports     0.8898    0.9658    0.9262       117
    Politics     0.8036    0.8654    0.8333       156

   micro avg     0.8660    0.8419    0.8538      1758
   macro avg     0.8465    0.8596    0.8520      1758
weighted avg     0.8687    0.8419    0.8536      1758
 samples avg     0.8883    0.8585    0.8553      1758
```

**결과 해석**

가장 빈번한 Society(support 684)는 precision 0.9151 은 높지만 recall 0.8041 로 낮아, 확신이 설 때만 켜는 보수적 태도가 일부 정답을 놓치고 있습니다. 반대로 support 가 가장 적은 IT/Science(126)는 F1 0.7656 으로 최저인데, 학습 샘플이 적은 라벨일수록 성능이 떨어지는 빈도 불균형 효과가 드러납니다.

카테고리마다 정답(label=1)과 비정답(label=0)의 sigmoid 확률 분포를 KDE 곡선으로 그립니다. 0.5 점선을 기준으로 두 분포가 얼마나 잘 갈라지는지 보면 라벨별 독립 sigmoid 가 분리를 잘 학습했는지 한눈에 들어옵니다. 두 봉우리가 겹치는 카테고리일수록 임계값 선택에 민감합니다.

```python
sns.set_theme(style="whitegrid", context="talk")

# Long-form DataFrame
records = []
for k in range(K):
    name = LABEL_NAMES_EN[k]
    for i in range(len(probs)):
        records.append({"category": name, "prob": probs[i, k], "label": int(labels[i, k])})
df_long = pd.DataFrame(records)

g = sns.FacetGrid(
    df_long, col="category", col_wrap=4, height=2.8, aspect=1.3,
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
g.fig.suptitle("Per-category sigmoid probability distribution by ground truth", y=1.03)
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/17-ko_multilabel-out1.png)

**결과 해석**

각 카테고리에서 정답(label=1)과 비정답(label=0)의 확률 분포가 0.5 점선을 사이에 두고 양쪽으로 잘 갈라져, 라벨마다 독립 sigmoid 가 분리를 잘 학습했음을 보여줍니다. 두 봉우리가 겹치는 카테고리일수록 threshold 선택에 따라 precision-recall 트레이드오프가 민감하게 바뀝니다.

라벨 간 동시출현 패턴을 조건부 확률 P(j|i) 행렬로 만들어 실제와 예측을 나란히 히트맵으로 비교합니다. 두 무늬가 비슷할수록 모델이 결합 입력 안의 카테고리 공동 등장 구조까지 재현했다는 뜻입니다. 합성 데이터는 무작위로 짝지었으므로 특정 쌍에 쏠리지 않는지도 확인합니다.

```python
def cooccurrence_matrix(Y):
    # Y: (N, K) multi-hot. Returns (K, K) where M[i, j] = P(label_j=1 | label_i=1).
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

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, M, title in [
    (axes[0], cooc_true, "True co-occurrence  P(j | i)"),
    (axes[1], cooc_pred, "Predicted co-occurrence  P(j | i)"),
]:
    sns.heatmap(
        M, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=LABEL_NAMES_EN, yticklabels=LABEL_NAMES_EN,
        cbar_kws={"label": "conditional probability"}, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("category j")
    ax.set_ylabel("given category i")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/17-ko_multilabel-out2.png)

**결과 해석**

실제 동시출현 행렬과 예측 행렬의 무늬가 거의 같아, 모델이 두 헤드라인을 결합한 입력에서 카테고리 간 공동 등장 패턴까지 충실히 재현함을 보여줍니다. 합성 데이터는 두 헤드라인을 무작위로 짝지었기에 대각선(자기 자신) 외에는 특정 쌍에 강하게 쏠리지 않는 점도 확인됩니다.
