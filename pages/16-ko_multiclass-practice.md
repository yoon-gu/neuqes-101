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
<IPython.core.display.HTML object>
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

검증셋 1,000건으로 평가 지표를 한 번에 출력합니다.

```python
eval_metrics = trainer.evaluate()
print("klue/bert-base KLUE-YNAT — evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
klue/bert-base KLUE-YNAT — evaluation:
             eval_loss: 0.4029
         eval_accuracy: 0.8560
  eval_macro_precision: 0.8497
     eval_macro_recall: 0.8736
         eval_macro_f1: 0.8603
          eval_auc_ovr: 0.9830
```

**결과 해석**

accuracy 0.856, macro F1 0.860 으로 7클래스 분류치고 견고합니다. accuracy 와 macro F1 이 거의 같다는 건 *클래스 편향이 작다* 는 뜻 — 다수·소수 클래스 모두 고르게 맞혔습니다. OvR AUC 0.983 은 모델이 정답 클래스에 다른 클래스보다 높은 확률을 부여하는 *순위 능력* 이 매우 좋음을 보여줍니다.

예측 logits 를 직접 받아 softmax 확률·예측 클래스·정답 여부를 손으로 계산합니다. top-1 확률(가장 높은 클래스의 확률)을 맞은 샘플과 틀린 샘플로 나눠 평균을 비교하면, 모델 자신감과 정답 여부가 얼마나 연동되는지 가늠할 수 있습니다.

```python
preds_output = trainer.predict(eval_tok)
logits = preds_output.predictions
labels = preds_output.label_ids.astype(int)

exp = np.exp(logits - logits.max(axis=1, keepdims=True))
probs_full = exp / exp.sum(axis=1, keepdims=True)
preds = probs_full.argmax(axis=1)

top1_prob = probs_full.max(axis=1)
correct = (preds == labels)

print(f"logits shape:    {logits.shape}")
print(f"top-1 prob range: [{top1_prob.min():.4f}, {top1_prob.max():.4f}]")
print(f"top-1 prob mean: correct={top1_prob[correct].mean():.4f}, wrong={top1_prob[~correct].mean():.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
logits shape:    (1000, 7)
top-1 prob range: [0.2669, 0.9909]
top-1 prob mean: correct=0.9033, wrong=0.7097
```

**결과 해석**

맞은 샘플의 평균 top-1 확률(0.903)이 틀린 샘플(0.710)보다 확실히 높아, 모델 자신감이 정답 여부를 어느 정도 반영합니다. 다만 틀린 샘플 평균도 0.71 로 꽤 높아 *자신 있게 틀리는* 케이스가 존재함을 시사하며, 이는 뒤의 KDE·샘플 해석에서 확인됩니다.

카테고리별 precision/recall/F1 을 표로 출력해 어느 클래스가 강하고 약한지 한눈에 봅니다.

```python
# 클래스별 분류 리포트
print(classification_report(
    labels, preds,
    target_names=LABEL_NAMES_EN,
    digits=4, zero_division=0,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

  IT/Science     0.7857    0.9483    0.8594        58
     Economy     0.8210    0.8471    0.8339       157
     Society     0.8830    0.8300    0.8557       400
Life&Culture     0.8129    0.8630    0.8372       146
       World     0.9053    0.8866    0.8958        97
      Sports     0.9459    0.9459    0.9459        74
    Politics     0.7941    0.7941    0.7941        68

    accuracy                         0.8560      1000
   macro avg     0.8497    0.8736    0.8603      1000
weighted avg     0.8582    0.8560    0.8562      1000
```

**결과 해석**

Sports(F1 0.946)·World(0.896) 가 가장 강하고, Politics(0.794)·Economy(0.834) 가 상대적으로 약합니다 — 정치·경제는 정책·국제 이슈가 겹쳐 경계가 모호한 카테고리라는 통념과 일치합니다. IT/Science 는 recall 0.948 로 잘 잡지만 precision 0.786 으로, *IT가 아닌 글을 IT로 오인* 하는 경우가 더 있음을 보여줍니다.

7×7 혼동 행렬을 그립니다. 색은 행 정규화(recall), 숫자는 원본 카운트라, 대각선이 진할수록 그 카테고리를 잘 맞힌 것이고 비대각 셀이 *어떤 카테고리끼리 헷갈리는지* 알려줍니다.

```python
sns.set_theme(style="white", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_NAMES))))
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(
    cm_norm, annot=cm, fmt="d",
    cmap="Blues", vmin=0, vmax=1,
    xticklabels=LABEL_NAMES_EN,
    yticklabels=LABEL_NAMES_EN,
    cbar_kws={"label": "행 정규화 (recall)"}, ax=ax,
)
ax.set_xlabel("예측 카테고리")
ax.set_ylabel("실제 카테고리")
ax.set_title("혼동 행렬 — KLUE-YNAT (7개 카테고리)")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/16-ko_multiclass-out1.png)

top-1 확률 분포를 정답/오답으로 나눠 KDE 로 겹쳐 그립니다. K=7 이라 균등분포 기준선(1/7 ≈ 0.143)도 함께 표시해, 모델이 *압도적으로 확신* 하는 영역과 *판단을 못 하는* 영역을 구분합니다.

```python
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
df_top = pd.DataFrame({
    "top1_prob": top1_prob,
    "outcome":   np.where(correct, "맞음", "틀림"),
})

fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df_top, x="top1_prob", hue="outcome",
    fill=True, common_norm=False, alpha=0.5,
    palette={"맞음": "#5BD17F", "틀림": "#E55050"},
    clip=(1/7, 1.0), ax=ax,
)
ax.axvline(1/7, color="black", lw=1.0, ls=":", alpha=0.5)
ax.text(1/7, ax.get_ylim()[1]*0.95, "  균등분포 = 1/K", va="top", fontsize=10, alpha=0.6)
ax.set_title("top-1 확률 — 정답 여부별 분포 (K=7)")
ax.set_xlabel("top-1 예측 확률  max_k P(y=k)")
ax.set_ylabel("밀도")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/16-ko_multiclass-out2.png)

세 종류의 대표 샘플 — 가장 확신한 것, 가장 망설인 것(확률이 2/K 근처), 가장 확신하며 틀린 것 — 을 골라 헤드라인 원문과 top-3 분포를 직접 읽습니다. 모델이 어떤 신호로 분류하고 어디서 무너지는지 감각을 잡는 단계입니다.

```python
texts = list(eval_full["text"])

idx_top    = int(np.argmax(top1_prob))
idx_unc    = int(np.argmin(np.abs(top1_prob - 1/len(LABEL_NAMES) * 2)))   # 1/7 의 2배 근처 (거의 모름)
wrong_mask = ~correct
idx_wrong  = int(np.argmax(top1_prob * wrong_mask)) if wrong_mask.any() else -1

samples = [
    ("most confident overall", idx_top),
    ("most uncertain (~2/K)", idx_unc),
    ("most confident WRONG",   idx_wrong),
]

for label_str, idx in samples:
    if idx < 0:
        continue
    print("=" * 78)
    print(f"sample #{idx}  ({label_str})")
    print("=" * 78)
    print(f"text:        {texts[idx]}")
    print(f"true label:  {labels[idx]}  ({LABEL_NAMES[labels[idx]]})")
    print(f"prediction:  {preds[idx]}  ({LABEL_NAMES[preds[idx]]})  match: {'✓' if correct[idx] else '✗'}")
    print(f"top-1 prob:  {top1_prob[idx]:.4f}")
    # top-3 클래스 모두 보기
    top3 = np.argsort(probs_full[idx])[::-1][:3]
    print(f"top-3 distribution:")
    for k in top3:
        print(f"  {LABEL_NAMES[k]:>8}: {probs_full[idx, k]:.4f}")
    print()
```

**▶ 실행 결과**

```text
==============================================================================
sample #131  (most confident overall)
==============================================================================
text:        美 시리아 북동부에 다국적 감시군 추진…미군 400명 잔류
true label:  4  (세계)
prediction:  4  (세계)  match: ✓
top-1 prob:  0.9909
top-3 distribution:
        세계: 0.9909
        정치: 0.0024
      IT과학: 0.0014

==============================================================================
sample #929  (most uncertain (~2/K))
==============================================================================
text:        수능 작년보다 쉬워…1등급 컷 국어·수학가 92점·수학나 88점
true label:  2  (사회)
prediction:  3  (생활문화)  match: ✗
top-1 prob:  0.2918
top-3 distribution:
      생활문화: 0.2918
        사회: 0.2758
      IT과학: 0.1989

==============================================================================
sample #57  (most confident WRONG)
==============================================================================
text:        朴대통령 스캐퍼로티 연합사령관에 보국훈장 통일장 수여
true label:  4  (세계)
prediction:  6  (정치)  match: ✗
top-1 prob:  0.9853
top-3 distribution:
        정치: 0.9853
        사회: 0.0055
        세계: 0.0034
```

**결과 해석**

가장 확신한 샘플(#131, "美 시리아 북동부…미군 잔류")은 *세계*에 0.99 를 몰아주며 명확히 맞혔습니다. 망설인 샘플(#929, 수능 등급컷)은 생활문화 0.29 vs 사회 0.28 로 거의 동률이라 *사회*(정답)를 근소하게 놓쳤는데, 헤드라인 자체가 두 카테고리에 걸친 경계 사례입니다. 가장 위험한 건 #57 — 박대통령이 연합사령관에게 훈장 수여라는 헤드라인을 *정치*로 0.99 확신하며 틀렸습니다(정답 세계). 인물·훈장 신호가 강해 자신 있게 오답을 낸, 라벨 경계가 모호한 전형적 사례입니다.
