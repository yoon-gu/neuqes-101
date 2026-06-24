> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/09_bert_regression.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

**baseline VRAM** — 모델 로드 전:

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Sun Jun 21 22:51:03 2026       
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

## 데이터 준비

Ch 8에서 익힌 `datasets` + 토크나이저 패턴을 그대로 적용합니다. 차이는 라벨을 *float* 형으로 바꾼다는 점입니다 — 회귀이므로 정답이 정수 클래스가 아닌 실수입니다.

별점 1-5를 그대로 학습 라벨로 사용합니다 (`label` 필드는 0-4로 저장돼 있어 +1).

Ch 8에서 익힌 토크나이저·`datasets` 패턴을 그대로 가져옵니다. T4에서 30분 안에 끝나도록 학습 4,000건·평가 1,000건만 잘라 씁니다.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("Yelp/yelp_review_full")

# train 4,000 + eval 1,000 — T4 30분 안에 학습 끝나도록 작게
train_ds = ds["train"].shuffle(seed=42).select(range(4000))
eval_ds  = ds["test"].shuffle(seed=42).select(range(1000))
```

**위 코드 읽기** — `shuffle(seed=42)` 로 섞은 뒤 `select(range(...))` 로 앞쪽 일부만 추립니다. 같은 seed라 매번 동일한 부분집합이 잡혀 재현이 됩니다.

```python

def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # label(0-4) → 별점(1-5) float 으로 변환. Trainer는 'labels' 컬럼을 사용
    out["labels"] = [float(lbl) + 1.0 for lbl in batch["label"]]
```

**위 코드 읽기** — 회귀의 핵심 한 줄입니다. 0-4로 저장된 `label` 에 `+1.0` 을 더해 별점 1-5로 되돌리되, `float(...)` 로 *실수* 라벨을 만듭니다. `Trainer` 는 `labels` 라는 컬럼명을 정답으로 인식하고, 라벨이 float이면 `problem_type="regression"` 과 맞물려 `MSELoss` 로 흘러갑니다. `max_length=128` 로 자른 건 attention 비용과 학습 시간을 T4 한도 안에 두기 위함입니다.

```python
    return out

train_tok = train_ds.map(tokenize_fn, batched=True).remove_columns(["text", "label"])
eval_tok  = eval_ds.map(tokenize_fn,  batched=True).remove_columns(["text", "label"])

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (float)")
```

**위 코드 읽기** — `map(batched=True)` 로 전체를 토큰화한 뒤, 더 이상 필요 없는 원문 `text` 와 정수 `label` 컬럼을 `remove_columns` 로 제거합니다. 남는 컬럼은 `input_ids`·`attention_mask` 등 모델 입력과 `labels` 뿐입니다.

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 4000
})

First sample label: 5.0  (float)
```

## 모델 로드 — `num_labels=1`, `problem_type="regression"`

Ch 7에서는 사전학습된 분류 헤드(`distilbert-base-uncased-finetuned-sst-2-english`, num_labels=2)를 그대로 썼습니다. 이번엔 본체 모델만 받고 **분류 헤드를 새로** 만듭니다 — `num_labels=1` 이라 출력 차원이 1, `problem_type="regression"` 이라 `Trainer` 가 자동으로 MSELoss 사용.

본체만 받고 회귀 헤드를 새로 붙이는 단계입니다. 두 인자가 핵심입니다.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1,
    problem_type="regression",
)
print(f"Parameters:    {sum(p.numel() for p in model.parameters()):,}")
print(f"Classifier:    {model.classifier}")
print(f"problem_type:  {model.config.problem_type}")
```

**위 코드 읽기** — `num_labels=1` 이라 분류 헤드가 `Linear(768, 1)` 로 만들어져 출력이 스칼라 한 개입니다. `problem_type="regression"` 을 명시하면 `Trainer` 가 별다른 설정 없이 `MSELoss` 를 골라 씁니다. 출력의 `MISSING` 표시(`classifier.weight` 등)는 새 헤드가 *랜덤 초기화* 됐다는 정상 신호로, 이 부분이 학습으로 채워집니다.

**▶ 실행 결과**

```text
[transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
Key                     | Status     | 
------------------------+------------+-
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
classifier.bias         | MISSING    | 
pre_classifier.bias     | MISSING    | 
pre_classifier.weight   | MISSING    | 
classifier.weight       | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:    66,954,241
Classifier:    Linear(in_features=768, out_features=1, bias=True)
problem_type:  regression
```

**경고 메시지를 보셨을 겁니다** — `Some weights of DistilBertForSequenceClassification were not initialized ...`. 분류 헤드(`Linear(768, 1)`)가 새로 만들어지면서 *랜덤 초기화* 됐다는 알림입니다. 이 부분이 학습으로 채워지고, BERT 본체는 사전학습 가중치를 미세 조정합니다 (transfer learning의 본 모습).

### 학습되는 파라미터 vs 동결된 파라미터

`from_pretrained()` 직후엔 *모든* 파라미터가 학습 대상입니다 (`requires_grad=True`). 그러나 데이터가 작거나 빠른 학습이 필요하면 BERT 본체를 *동결(freeze)* 하고 분류 헤드만 학습하기도 합니다. 학습 시작 전에 *전체 vs 학습되는 파라미터* 를 한 번 확인하는 게 좋은 습관입니다.

```python
def param_summary(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    frozen    = total - trainable
    return total, trainable, frozen

total, trainable, frozen = param_summary(model)
print(f"Total parameters:     {total:>13,}  ({total/1e6:.1f} M)")
print(f"Trainable parameters: {trainable:>13,}  ({trainable/1e6:.1f} M, {trainable/total:.1%})")
print(f"Frozen parameters:    {frozen:>13,}  ({frozen/1e6:.1f} M, {frozen/total:.1%})")
print(f"\nDefault — all layers are trainable")
```

**▶ 실행 결과**

```text
Total parameters:        66,954,241  (67.0 M)
Trainable parameters:    66,954,241  (67.0 M, 100.0%)
Frozen parameters:                0  (0.0 M, 0.0%)

Default — all layers are trainable
```

### 시연: BERT 본체 동결 패턴

본 학습은 *모든 파라미터* 를 학습하지만, 동결 패턴이 어떻게 적용되는지 *별도 모델 인스턴스* 로 한 번 보여드립니다 (이 시연 모델은 학습에 사용하지 않습니다).

```python
# 시연용 — 같은 모델을 한 번 더 만들고 BERT 본체를 동결
demo_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1, problem_type="regression",
)

# distilbert 본체의 모든 파라미터에 requires_grad=False 설정
for p in demo_model.distilbert.parameters():
    p.requires_grad = False

# 분류 헤드는 학습 대상으로 그대로 둠 (default가 True)

t, tr, fr = param_summary(demo_model)
print(f"After freezing BERT body:")
print(f"  Total:       {t:>13,}")
print(f"  Trainable:   {tr:>13,}  ({tr/t:.1%})  ← classification head only")
print(f"  Frozen:      {fr:>13,}  ({fr/t:.1%})  ← BERT body")
print(f"\nOnly {tr:,} classification head parameters update — fast, low memory.")
print(f"Tradeoff: BERT body cannot adapt to the task — usually train the body too if data is sufficient.")
print(f"\n(This demo model is not used for the actual training — del demo_model frees memory.)")

import gc
del demo_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**▶ 실행 결과**

```text
[transformers] DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased
Key                     | Status     | 
------------------------+------------+-
vocab_layer_norm.bias   | UNEXPECTED | 
vocab_projector.bias    | UNEXPECTED | 
vocab_transform.weight  | UNEXPECTED | 
vocab_layer_norm.weight | UNEXPECTED | 
vocab_transform.bias    | UNEXPECTED | 
classifier.bias         | MISSING    | 
pre_classifier.bias     | MISSING    | 
pre_classifier.weight   | MISSING    | 
classifier.weight       | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
After freezing BERT body:
  Total:          66,954,241
  Trainable:         591,361  (0.9%)  ← classification head only
  Frozen:         66,362,880  (99.1%)  ← BERT body

Only 591,361 classification head parameters update — fast, low memory.
Tradeoff: BERT body cannot adapt to the task — usually train the body too if data is sufficient.

(This demo model is not used for the actual training — del demo_model frees memory.)
```

**언제 동결을 쓰나**

- **분류 헤드만 학습 (모든 본체 동결)**: 데이터 매우 작음 (수백 건), 빠른 baseline 필요.
- **하위 N개 layer 동결**: 일반 언어 표현은 BERT 그대로, 상위 layer만 task 적응.
- **모든 파라미터 학습 (default)**: 데이터 충분 (수천 건+), 본체도 task에 맞게 적응.

이번 챕터는 4,000건이라 default(전체 학습)이 가장 좋은 선택입니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Sun Jun 21 22:51:27 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   37C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

모델 가중치(약 67M 파라미터, fp32 약 255 MB)가 GPU에 올라간 상태입니다. 학습이 시작되면 *옵티마이저 모멘텀(2배) + gradient(1배)* 가 추가되어 VRAM이 더 늘어납니다.

## `TrainingArguments` + `Trainer`

Ch 6 끝에서 미리 본 코드 형태가 이제 실제로 등장합니다. `TrainingArguments` 한 객체에 학습 하이퍼파라미터를 모두 모으고, `Trainer` 가 학습 루프·평가·로그·체크포인트를 자동화합니다.

학습 하이퍼파라미터를 한 객체에 모읍니다. 각 값이 T4 30분 제약을 지키는 기본 안전대입니다.

```python
training_args = TrainingArguments(
    output_dir="./ch09_output",
    num_train_epochs=2,                 # 2 에폭이면 T4에서 5-8분
    per_device_train_batch_size=16,     # T4 16GB에 안전
    per_device_eval_batch_size=32,
    learning_rate=2e-5,                 # BERT 파인튜닝 표준
    fp16=True,                          # T4 GPU 효율 (bf16은 T4 미지원)
    eval_strategy="epoch",              # 에폭마다 평가
    logging_steps=50,                   # 50 step마다 loss 출력
    save_strategy="no",                 # 체크포인트 저장 안 함 (디스크·VRAM 절약)
    report_to="none",                   # wandb 등 외부 로깅 비활성
    seed=42,
)

print(f"Total training steps: {len(train_tok) // training_args.per_device_train_batch_size * training_args.num_train_epochs}")
```

**위 코드 읽기** — `fp16=True` 는 T4에서 필수 선택입니다(T4는 bf16 미지원). `eval_strategy="epoch"` 으로 에폭마다 평가가 돌고, `save_strategy="no"` 라 체크포인트를 남기지 않아 디스크·VRAM을 아낍니다. 전체 step 수는 `4000 / 16 × 2 = 500` 으로 출력에서 확인됩니다.

**▶ 실행 결과**

```text
Total training steps: 500
```

평가 때 출력할 지표를 직접 정의합니다. `Trainer` 가 넘겨주는 `(preds, labels)` 를 받아 sklearn 헬퍼로 계산합니다.

```python
# 평가 지표를 직접 정의 — sklearn 헬퍼 그대로 활용
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.flatten()
    return {
        "mse": float(mean_squared_error(labels, preds)),
        "mae": float(mean_absolute_error(labels, preds)),
        "r2":  float(r2_score(labels, preds)),
    }
```

**위 코드 읽기** — 회귀이므로 `preds` 가 `(N, 1)` 형태라 `flatten()` 으로 1차원으로 폅니다. 반환한 dict의 키(`mse`·`mae`·`r2`)는 평가 로그에서 `eval_` 접두사가 붙어 `eval_mse` 처럼 출력됩니다.

이제 모델·인자·데이터·지표를 `Trainer` 하나로 묶고 `train()` 한 줄로 학습을 돌립니다.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tokenizer,         # ← 이 한 줄이 DataCollatorWithPadding 을 자동 생성
                                        # (transformers 4.46+ 의 새 인자명. 그 이전엔 tokenizer=tokenizer)
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
print(f"\nTraining done — mean train loss: {train_result.training_loss:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Training done — mean train loss: 1.1182
```

**결과 해석**

500 step 전체의 평균 train loss가 1.1182로 끝났습니다. 이는 MSE 단위(별점² 오차)라, 초반 step의 큰 loss까지 평균에 섞인 값입니다. 학습이 실제로 줄었는지는 아래 eval 지표(`eval_mse` 0.65)로 확인하는 편이 정확합니다.

학습이 진행되는 동안 step별 loss와 에폭별 평가 metric이 출력됩니다. **핵심 관찰**:

- `loss` 가 처음 수 step에서 큰 값(흔히 0.3-0.5)이었다가 학습이 진행되면 줄어들어야 정상입니다.
- 에폭 끝에서 출력되는 `eval_mse`, `eval_mae`, `eval_r2` 가 우리가 정의한 평가 지표입니다.
- `loss` 가 줄어들지 않거나 nan으로 가면 학습률을 낮추거나(`5e-6`), `fp16=False` 로 시도해 봅니다.

> 📒 **부록 노트북 두 편**
>
> 1. [`appendix_experiment_tracking.ipynb`](./appendix_experiment_tracking.ipynb) — `report_to` 인자로 **wandb · trackio · MLflow** 같은 experiment tracker를 붙이는 패턴. 학습 곡선·평가 metric을 dashboard에서 보고 여러 run을 한 화면에 비교. ([Colab으로](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/appendix_experiment_tracking.ipynb))
>
> 2. [`appendix_hpo.ipynb`](./appendix_hpo.ipynb) — **하이퍼파라미터 최적화(HPO)의 어려움**. `TrainingArguments` 인자 정리, HPO가 어려운 5가지 이유, `Trainer.hyperparameter_search` + Optuna 직접 시도, wandb sweeps · MLflow autolog 통합. ([Colab으로](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/appendix_hpo.ipynb))

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Sun Jun 21 22:51:58 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   53C    P0             64W /   70W |    1573MiB /  15360MiB |     77%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            9461      C   /usr/bin/python3                       1570MiB |
+-----------------------------------------------------------------------------------------+
```

학습 후 VRAM 상태입니다. 학습 *중* 에는 옵티마이저 모멘텀과 gradient가 추가되어 더 큰 VRAM을 잠시 쓰지만, 학습이 끝나면 일부가 해제됩니다 (단, PyTorch 캐시 할당자가 다음 사용을 위해 일부 메모리를 보유).

**학습 시 VRAM 구성 (fp16 기준)**:

| 구성 요소 | 크기 (DistilBERT 67M 기준) |
|---|---|
| 모델 가중치 (fp16) | ~128 MB |
| Adam 1차 모멘텀 (fp32 마스터) | ~255 MB |
| Adam 2차 모멘텀 (fp32 마스터) | ~255 MB |
| Gradient (fp16) | ~128 MB |
| Activation (배치 16, max_len 128) | ~수백 MB |
| 합계 | 약 1-1.5 GB |

큰 모델(BERT-large 340M)이나 큰 배치를 쓰면 한도(15.36 GB)에 빠르게 다가갑니다.
