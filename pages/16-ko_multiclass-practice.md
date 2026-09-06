> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/16_ko_multiclass/16_ko_multiclass.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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

**baseline VRAM**:

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:58:31 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   45C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 데이터 — KLUE-YNAT (뉴스 헤드라인 7분류)

**KLUE** = Korean Language Understanding Evaluation 벤치마크. **YNAT** = Yonhap News Agency Topic. 연합뉴스 헤드라인 한 줄 + 7카테고리 라벨.

| 라벨 | 카테고리 |
|---|---|
| 0 | IT과학 |
| 1 | 경제 |
| 2 | 사회 |
| 3 | 생활문화 |
| 4 | 세계 |
| 5 | 스포츠 |
| 6 | 정치 |

`datasets.load_dataset("klue/klue", "ynat")` 로 정상 로드 (parquet 기반).

KLUE-YNAT 를 불러와 split·크기·7개 카테고리 라벨을 확인합니다. 원본 라벨은 한국어지만, plot 의 한글 깨짐을 피하려고 영문 매핑(`_KO2EN`)을 따로 만들어 출력·시각화에는 영문 이름을 씁니다. 클래스 분포도 함께 찍어 데이터가 얼마나 균형에 가까운지 봅니다.

```python
ds = load_dataset("klue/klue", "ynat")
print(f"splits: {list(ds.keys())}")
print(f"sizes: {[(k, len(v)) for k, v in ds.items()]}")
print(f"label names: {ds['train'].features['label'].names}")
```

**위 코드 읽기** — `load_dataset("klue/klue", "ynat")` 가 KLUE 벤치마크의 YNAT(연합뉴스 토픽) 서브셋을 내려받습니다. `features["label"].names` 가 `datasets.ClassLabel` 에 박혀 있는 7개 카테고리의 사람-읽는 이름이고, 뒤에서 `id2label` 매핑의 출처가 됩니다.

```python
# 클래스 분포
import collections
cnt = collections.Counter(ds["train"]["label"])
LABEL_NAMES = ds["train"].features["label"].names   # KLUE-YNAT 원본 (한국어)
# 출력·플롯은 영문으로 (matplotlib 한글 폰트 깨짐·조판 문제 방지)
_KO2EN = {"IT과학": "IT/Science", "경제": "Economy", "사회": "Society",
          "생활문화": "Life&Culture", "세계": "World", "스포츠": "Sports", "정치": "Politics"}
LABEL_NAMES_EN = [_KO2EN.get(n, n) for n in LABEL_NAMES]
```

**위 코드 읽기** — `LABEL_NAMES` 는 한국어 원본 이름(`'사회'`, `'스포츠'` 등)을 그대로 보존하고, `LABEL_NAMES_EN` 은 plot·콘솔 출력용 영문 이름입니다. 모델 라벨 인덱스(0-6)는 그대로 두고 *표시 이름만* 두 벌로 갈라 둔 셈입니다.

```python
print(f"\nClass distribution (train):")
for k in range(len(LABEL_NAMES_EN)):
    n = cnt[k]
    print(f"  {LABEL_NAMES_EN[k]:>13}  (label {k}): {n:>5}  ({n / len(ds['train']):.1%})")

print(f"\nfirst 3 samples:")
for ex in ds["train"].select(range(3)):
    print(f"  label={ex['label']} ({LABEL_NAMES_EN[ex['label']]:>13})  title={ex['title']!r}")
```

**▶ 실행 결과**

```text
splits: ['train', 'validation']
sizes: [('train', 45678), ('validation', 9107)]
label names: ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
Class distribution (train):
     IT/Science  (label 0):  5235  (11.5%)
        Economy  (label 1):  6118  (13.4%)
        Society  (label 2):  5133  (11.2%)
   Life&Culture  (label 3):  5751  (12.6%)
          World  (label 4):  8320  (18.2%)
         Sports  (label 5):  7742  (16.9%)
       Politics  (label 6):  7379  (16.2%)

first 3 samples:
  label=3 ( Life&Culture)  title='유튜브 내달 2일까지 크리에이터 지원 공간 운영'
  label=3 ( Life&Culture)  title='어버이날 맑다가 흐려져…남부지방 옅은 황사'
  label=2 (      Society)  title='내년부터 국가RD 평가 때 논문건수는 반영 않는다'
```

**결과 해석**

train 45,678건이 7개 카테고리에 11.2%~18.2% 로 퍼져 있어 *완벽 균형은 아니지만* 심한 불균형도 아닙니다 — World(18.2%)·Sports(16.9%)·Politics(16.2%) 가 다수, IT/Science(11.5%)·Society(11.2%) 가 소수입니다. 이 정도 분포면 accuracy 와 macro F1 을 함께 봐도 큰 괴리가 나지 않습니다.

T4 30분 룰에 맞춰 train 5,000 / eval 1,000 건만 샘플합니다. `title` 컬럼을 `transformers` 표준 `text` 로 이름만 바꿔 두고, `klue/bert-base` 토크나이저로 헤드라인의 토큰 길이 분포를 미리 확인합니다.

```python
# T4 30분 룰: 5K train / 1K eval (KLUE 의 validation split 에서 sample)
SEED = 42
train_full = ds["train"].shuffle(seed=SEED).select(range(5000))
eval_full  = ds["validation"].shuffle(seed=SEED).select(range(1000))

# title 컬럼명을 transformers 표준 'text' 로 통일
train_full = train_full.rename_column("title", "text")
eval_full  = eval_full.rename_column("title", "text")

print(f"sampled train: {len(train_full)}")
print(f"sampled eval:  {len(eval_full)}")

# 토큰 길이 분포 미리 보기
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
sample_lens = [len(tokenizer.encode(t)) for t in train_full["text"][:200]]
print(f"\nToken length (sample 200): mean={np.mean(sample_lens):.1f}, median={np.median(sample_lens):.0f}, max={max(sample_lens)}")
```

**▶ 실행 결과**

```text
sampled train: 5000
sampled eval:  1000
Token length (sample 200): mean=15.8, median=16, max=27
```

**결과 해석**

헤드라인 한 줄이라 토큰 길이가 평균 15.8, 최대 27 로 매우 짧습니다 — `max_length=128` 은 충분히 여유롭고, 짧은 시퀀스 덕에 학습이 빠릅니다.

## 토큰화 — Ch 15 패턴 그대로

라벨 형식만 binary int → 0-6 int. 한 줄 차이.

토큰화는 Ch 15 패턴 그대로입니다. 라벨만 binary 0/1 대신 0-6 정수로 들어가는데, single-label 분류라 `CrossEntropyLoss` 가 받는 그대로 *정수 인덱스* 면 됩니다(multi-hot float 아님). 모델 입력에 필요한 컬럼만 남기고 나머지는 제거합니다.

```python
def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(l) for l in batch["label"]]
    return out

train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(
    [c for c in train_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)
eval_tok  = eval_full.map(tokenize_fn,  batched=True).remove_columns(
    [c for c in eval_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (int 0-6)")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample label: 3  (int 0-6)
```

## 모델 로드 — `num_labels=7` 만 바뀜

Ch 15 셋업에서 K=2 → K=7 한 줄 변화.

Ch 15 의 모델 로드에서 `num_labels` 만 2 → 7 로 바뀝니다. `num_labels=len(LABEL_NAMES)` 로 실제 클래스 수를 직접 세어 넣고, `problem_type="single_label_classification"` 으로 softmax + `CrossEntropyLoss` 경로를 명시합니다. `id2label`/`label2id` 를 같이 넘겨 두면 나중에 `model.config` 가 라벨 이름을 기억합니다.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=len(LABEL_NAMES),
    problem_type="single_label_classification",
    id2label={i: name for i, name in enumerate(LABEL_NAMES_EN)},
    label2id={name: i for i, name in enumerate(LABEL_NAMES_EN)},
)
```

**위 코드 읽기** — `num_labels` 를 상수 7 로 박지 않고 `len(LABEL_NAMES)` 로 *데이터에서 센* 값을 쓰는 게 핵심입니다 — 분류 헤드 출력 차원과 라벨 범위가 어긋나면 학습 중 CUDA assert 가 나기 때문(삽질 코너 참고). `problem_type="single_label_classification"` 이 softmax+CE 경로를 고정합니다.

```python
def param_summary(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

total, trainable = param_summary(model)
print(f"Parameters:           {total:>13,}  ({total/1e6:.1f} M)")
print(f"Trainable parameters: {trainable:>13,}  ({trainable/total:.1%})")
print(f"Classifier:           {model.classifier}")
print(f"id2label:             {model.config.id2label}")
```

**▶ 실행 결과**

```text
[transformers] BertForSequenceClassification LOAD REPORT from: klue/bert-base
Key                                        | Status     | 
-------------------------------------------+------------+-
cls.seq_relationship.bias                  | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
cls.predictions.transform.dense.bias       | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
classifier.bias                            | MISSING    | 
classifier.weight                          | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:             110,622,727  (110.6 M)
Trainable parameters:   110,622,727  (100.0%)
Classifier:           Linear(in_features=768, out_features=7, bias=True)
id2label:             {0: 'IT/Science', 1: 'Economy', 2: 'Society', 3: 'Life&Culture', 4: 'World', 5: 'Sports', 6: 'Politics'}
```

**결과 해석**

LOAD REPORT 의 `cls.*` UNEXPECTED / `classifier.*` MISSING 은 정상입니다 — 사전학습된 MLM·NSP 헤드를 버리고 7-클래스 분류 헤드를 *새로 초기화* 했다는 뜻입니다. 분류기가 `Linear(768 → 7)` 로 잡혔고, 전체 파라미터는 110.6M 인데 헤드는 그중 5,383개뿐이라 K 가 늘어도 모델이 거의 무거워지지 않습니다.

**Ch 15 와의 파라미터 수 비교** — 7클래스로 늘어났는데도 모델은 *거의 안 무거워짐*:

| 부분 | Ch 15 (K=2) | Ch 16 (K=7) |
|---|---|---|
| BERT body (12 layer) | 110,617,344 | 110,617,344 |
| classifier `Linear(768, K)` | 1,538 | **5,383** |
| 합계 | 110,618,882 | **110,622,727** |

분류 헤드만 K 에 비례해 늘어나지만 BERT body 가 ~110M 이라 K 가 5 늘어도 전체 차이는 0.003%. **K 가 늘어났다고 학습이 *훨씬* 무거워지지는 않는다** — multi-class BERT 의 매력.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:58:51 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   46C    P8             14W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 학습 — Ch 15 와 동일한 hyperparams

`compute_metrics` 만 multi-class 용으로 (Ch 12 의 패턴 그대로).

multi-class 용 `compute_metrics` 입니다. logits 에 수치 안정 softmax 를 적용해 확률을 만들고 argmax 로 예측 클래스를 뽑습니다. accuracy 외에 *macro* 평균 precision/recall/F1 을 함께 계산해 소수 클래스 성능이 묻히지 않게 하고, multi-class AUC 는 One-vs-Rest 로 구합니다.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 안정 softmax (K=7)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_full = exp / exp.sum(axis=1, keepdims=True)
    preds = probs_full.argmax(axis=1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {
        "accuracy":        float(accuracy_score(labels, preds)),
        "macro_precision": float(p),
        "macro_recall":    float(r),
        "macro_f1":        float(f1),
    }
    # multi-class AUC: One-vs-Rest
    try:
        out["auc_ovr"] = float(roc_auc_score(labels, probs_full, multi_class="ovr"))
    except ValueError:
        out["auc_ovr"] = float("nan")
    return out
```

학습 hyperparams 는 Ch 15 와 *완전히 동일* 합니다 — 2 에폭, batch 16, lr 2e-5, `fp16=True`(T4 필수). 학습 후 평균 train loss 를 K=7 의 random baseline($\log 7 \approx 1.95$)과 나란히 찍어, 모델이 균등 추측 단계에서 얼마나 내려왔는지 한눈에 봅니다.

```python
training_args = TrainingArguments(
    output_dir="./ch16_output",
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
print(f"random baseline loss (K=7): {np.log(7):.4f}")
```

**▶ 실행 결과**

```text
Epoch  Training Loss  Validation Loss  Accuracy  Macro Precision  Macro Recall  Macro F1  Auc Ovr
1      0.441529       0.465393         0.856000  0.838248         0.881824      0.856753  0.978641
2      0.286288       0.402871         0.856000  0.849702         0.873584      0.860287  0.982976
Training done — mean train loss: 0.4626
random baseline loss (K=7): 1.9459
```

**결과 해석**

평균 train loss 0.4626 은 random baseline 1.9459 의 약 1/4 수준으로, 모델이 균등 추측에서 충분히 멀어졌음을 보여줍니다. 단 2 에폭·5K 샘플로도 한국어 헤드라인의 카테고리 신호를 잘 잡았다는 신호입니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:59:33 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   65C    P0             72W /   70W |    2195MiB /  15360MiB |     64%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A             875      C   /usr/bin/python3                       2192MiB |
+-----------------------------------------------------------------------------------------+
```
