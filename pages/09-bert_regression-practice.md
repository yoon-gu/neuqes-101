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
    return out
```

**위 코드 읽기** — 회귀의 핵심 한 줄입니다. 0-4로 저장된 `label` 에 `+1.0` 을 더해 별점 1-5로 되돌리되, `float(...)` 로 *실수* 라벨을 만듭니다. `Trainer` 는 `labels` 라는 컬럼명을 정답으로 인식하고, 라벨이 float이면 `problem_type="regression"` 과 맞물려 `MSELoss` 로 흘러갑니다. `max_length=128` 로 자른 건 attention 비용과 학습 시간을 T4 한도 안에 두기 위함입니다.

```python
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

```python
# BERT 최종 평가 (eval_dataset 기준)
bert_metrics = trainer.evaluate()
print("BERT evaluation:")
for k, v in bert_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
BERT evaluation:
             eval_loss: 0.6539
              eval_mse: 0.6539
              eval_mae: 0.6180
               eval_r2: 0.6644
```

**결과 해석**

`eval_loss` 와 `eval_mse` 가 0.6539로 같습니다 — 회귀 loss가 곧 MSE이기 때문입니다. MAE 0.618은 평균적으로 별점을 약 0.6점 틀린다는 뜻이고, R² 0.664는 별점 분산의 약 66%를 설명한다는 의미입니다.

같은 데이터를 Ch 2 방식(TF-IDF + `LinearRegression`)으로도 학습해 BERT와 직접 견줍니다.

```python
# 같은 4,000건으로 sklearn LinearRegression 학습 (Ch 2 방식)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# 토큰화 전 원문 회수
train_texts = train_ds["text"]
train_labels = np.array([float(l) + 1.0 for l in train_ds["label"]])
eval_texts = eval_ds["text"]
eval_labels = np.array([float(l) + 1.0 for l in eval_ds["label"]])

tfidf = TfidfVectorizer(max_features=10000)
X_tr = tfidf.fit_transform(train_texts)
X_ev = tfidf.transform(eval_texts)

linreg = LinearRegression().fit(X_tr, train_labels)
sk_pred = linreg.predict(X_ev)

print("sklearn LinearRegression evaluation:")
print(f"  mse: {mean_squared_error(eval_labels, sk_pred):.4f}")
print(f"  mae: {mean_absolute_error(eval_labels, sk_pred):.4f}")
print(f"  r2:  {r2_score(eval_labels, sk_pred):.4f}")
```

**▶ 실행 결과**

```text
sklearn LinearRegression evaluation:
  mse: 1.5597
  mae: 1.0086
  r2:  0.1996
```

```python
# 한 표로 비교
rows = [
    {"model": "sklearn LinearRegression",
     "mse": mean_squared_error(eval_labels, sk_pred),
     "mae": mean_absolute_error(eval_labels, sk_pred),
     "r2":  r2_score(eval_labels, sk_pred)},
    {"model": "DistilBERT fine-tuned",
     "mse": bert_metrics["eval_mse"],
     "mae": bert_metrics["eval_mae"],
     "r2":  bert_metrics["eval_r2"]},
]
pd.DataFrame(rows).round(4)
```

**▶ 실행 결과**

```text
                      model     mse     mae      r2
0  sklearn LinearRegression  1.5597  1.0086  0.1996
1     DistilBERT fine-tuned  0.6539  0.6180  0.6644
```

**결과 해석**

세 지표 모두 BERT가 크게 앞섭니다 — MSE 1.56 → 0.65, MAE 1.01 → 0.62, R² 0.20 → 0.66. 문맥을 attention으로 읽는 BERT가 단어 빈도만 보는 TF-IDF 회귀보다 별점을 훨씬 정확히 맞춘다는 가설이 이 수치로 확인됩니다.

시각화에 쓸 예측값을 모읍니다. `Trainer.predict` 로 BERT 예측을 받아 sklearn 예측과 한 long-form DataFrame으로 합칩니다.

```python
# BERT 예측값 직접 받기 (별도 evaluate 호출이지만 빠름)
preds_output = trainer.predict(eval_tok)
bert_pred = preds_output.predictions.flatten()

# seaborn 비교용 long-form DataFrame
df_compare = pd.DataFrame({
    "Actual star": np.concatenate([eval_labels, eval_labels]),
    "Predicted":   np.concatenate([bert_pred,   sk_pred]),
    "Model":       ["BERT"] * len(eval_labels) + ["sklearn"] * len(eval_labels),
})
df_compare["Residual"] = df_compare["Predicted"] - df_compare["Actual star"]
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
```

실제 별점별로 두 모델이 어떤 값을 출력했는지 split violin으로 좌우에 둡니다. 빨간 점선이 이상적인 정답선(정답 = 예측)입니다.

```python
fig, ax = plt.subplots(figsize=(11, 5))
sns.violinplot(
    data=df_compare, x="Actual star", y="Predicted", hue="Model",
    split=True, inner="quart", ax=ax,
)
for i, x_val in enumerate([1, 2, 3, 4, 5]):
    ax.plot([i - 0.4, i + 0.4], [x_val, x_val], "r--", linewidth=1, alpha=0.7)
ax.set_title("실제 별점별 예측 별점 분포")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/09-bert_regression-out1.png)

이번엔 잔차(예측 − 실제)를 y축에 둡니다. 0 기준선에 좁게 모일수록 정확하고, 한쪽으로 치우치면 bias가 있다는 뜻입니다.

```python
fig, ax = plt.subplots(figsize=(11, 5))
sns.violinplot(
    data=df_compare, x="Actual star", y="Residual", hue="Model",
    split=True, inner="quart", ax=ax,
)
ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax.set_title("잔차 = 예측 − 실제, 실제 별점별 분포")
ax.set_ylabel("잔차 (예측 − 실제)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/09-bert_regression-out2.png)
