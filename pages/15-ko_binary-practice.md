> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/15_ko_binary/15_ko_binary.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q transformers datasets
```

```python
import warnings
warnings.filterwarnings("ignore")

import io
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score,
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
Wed Jun 17 21:44:50 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   32C    P8             11W /   70W |       3MiB /  15360MiB |      0%      Default |
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

같은 한국어 문장을 한국어로 학습한 `klue/bert-base`와 영어 위주의 `distilbert-base-uncased`로 각각 토큰화해 결과를 나란히 비교합니다. 토큰 개수와 끊기는 단위를 보면 어느 토크나이저가 한국어를 의미 단위에 가깝게 쪼개는지 한눈에 드러납니다. 마지막에는 두 토크나이저의 단어 사전 크기도 함께 출력합니다.

```python
tokenizer_ko = AutoTokenizer.from_pretrained("klue/bert-base")
tokenizer_en = AutoTokenizer.from_pretrained("distilbert-base-uncased")

samples = [
    "이 영화 정말 재미있었어요",
    "별로였어요. 시간 낭비",
    "오랜만에 본 명작이네요!",
]

for sent in samples:
    print(f"text: {sent}")
    tok_ko = tokenizer_ko.tokenize(sent)
    tok_en = tokenizer_en.tokenize(sent)
    print(f"  klue/bert-base ({len(tok_ko):>2} tokens): {tok_ko}")
    print(f"  distilbert-en  ({len(tok_en):>2} tokens): {tok_en}")
    print()

print(f"klue/bert-base vocab size:        {tokenizer_ko.vocab_size:,}")
print(f"distilbert-base-uncased vocab:    {tokenizer_en.vocab_size:,}")
```

**▶ 실행 결과**

```text
text: 이 영화 정말 재미있었어요
  klue/bert-base ( 6 tokens): ['이', '영화', '정말', '재미있', '##었', '##어요']
  distilbert-en  (14 tokens): ['ᄋ', '##ᅵ', 'ᄋ', '##ᅧ', '##ᆼ', '##ᄒ', '##ᅪ', 'ᄌ', '##ᅥ', '##ᆼ', '##ᄆ', '##ᅡ', '##ᆯ', '[UNK]']

text: 별로였어요. 시간 낭비
  klue/bert-base ( 6 tokens): ['별로', '##였', '##어요', '.', '시간', '낭비']
  distilbert-en  (12 tokens): ['[UNK]', '.', 'ᄉ', '##ᅵ', '##ᄀ', '##ᅡ', '##ᆫ', 'ᄂ', '##ᅡ', '##ᆼ', '##ᄇ', '##ᅵ']

text: 오랜만에 본 명작이네요!
  klue/bert-base ( 8 tokens): ['오랜만', '##에', '본', '명작', '##이', '##네', '##요', '!']
  distilbert-en  (26 tokens): ['ᄋ', '##ᅩ', '##ᄅ', '##ᅢ', '##ᆫ', '##ᄆ', '##ᅡ', '##ᆫ', '##ᄋ', '##ᅦ', 'ᄇ', '##ᅩ', '##ᆫ', 'ᄆ', '##ᅧ', '##ᆼ', '##ᄌ', '##ᅡ', '##ᆨ', '##ᄋ', '##ᅵ', '##ᄂ', '##ᅦ', '##ᄋ', '##ᅭ', '!']

klue/bert-base vocab size:        32,000
distilbert-base-uncased vocab:    30,522
```

**결과 해석**

한국어로 학습된 `klue/bert-base`는 "재미있" + "##었" + "##어요"처럼 의미 단위에 가깝게 6-8개 토큰으로 끊지만, 영어 위주의 `distilbert-en`은 한글 음절을 자모 단위로 분해해 같은 문장을 14-26개 토큰으로 파편화하고 `[UNK]`까지 흘립니다. 한국어에서는 그 언어로 학습된 토크나이저를 쓰는 것이 토큰 효율과 표현력 모두에서 결정적임을 보여 줍니다.

실습에 쓸 NSMC(네이버 영화 리뷰) 데이터를 GitHub에서 직접 내려받아 train/test로 읽어 들입니다. 본문이 비어 있는 행은 미리 제거하고, 전체 행 수와 라벨 분포, 첫 3개 샘플을 출력해 데이터의 모양과 균형을 확인합니다.

```python
TRAIN_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
TEST_URL  = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

print("downloading NSMC train/test from GitHub...")
df_train_full = pd.read_csv(TRAIN_URL, sep="\t").dropna(subset=["document"])
df_test_full  = pd.read_csv(TEST_URL,  sep="\t").dropna(subset=["document"])
print(f"  train: {len(df_train_full):,} rows")
print(f"  test:  {len(df_test_full):,} rows")
print(f"  label distribution (train): {df_train_full['label'].value_counts().to_dict()}")
print(f"\nfirst 3 rows of train:")
for _, row in df_train_full.head(3).iterrows():
    print(f"  label={row['label']}  text={row['document'][:80]}")
```

**▶ 실행 결과**

```text
downloading NSMC train/test from GitHub...
  train: 149,995 rows
  test:  49,997 rows
  label distribution (train): {0: 75170, 1: 74825}

first 3 rows of train:
  label=0  text=아 더빙.. 진짜 짜증나네요 목소리
  label=1  text=흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나
  label=0  text=너무재밓었다그래서보는것을추천한다
```

**결과 해석**

NSMC는 부정(0) 75,170건과 긍정(1) 74,825건으로 거의 1:1로 균형 잡힌 영화 리뷰 데이터라, 정확도 지표를 그대로 신뢰할 수 있습니다.

T4에서 30분 안에 끝나도록 전체 데이터에서 train 5,000건과 eval 1,000건만 고정 시드로 뽑아 냅니다. 이어서 `datasets.Dataset` 형태로 바꾸고, 본문 컬럼 이름을 `transformers`가 기대하는 `text`로 통일합니다. 샘플 후에도 양성 비율이 거의 50%로 유지되는지 확인하는 것이 핵심입니다.

```python
# 5K train / 1K eval 로 subsample (T4 30분 룰)
SEED = 42
df_train = df_train_full.sample(n=5000, random_state=SEED).reset_index(drop=True)
df_eval  = df_test_full.sample(n=1000,  random_state=SEED).reset_index(drop=True)

print(f"sampled train: {len(df_train)}")
print(f"sampled eval:  {len(df_eval)}")
print(f"train positive rate: {df_train['label'].mean():.1%}")
print(f"eval  positive rate: {df_eval['label'].mean():.1%}")

# datasets.Dataset 형태로 변환
train_ds = Dataset.from_pandas(df_train[["document", "label"]])
eval_ds  = Dataset.from_pandas(df_eval[["document", "label"]])

# 컬럼 이름을 transformers 표준에 맞게 통일
train_ds = train_ds.rename_column("document", "text")
eval_ds  = eval_ds.rename_column("document", "text")
print()
print(train_ds)
```

**▶ 실행 결과**

```text
sampled train: 5000
sampled eval:  1000
train positive rate: 49.2%
eval  positive rate: 49.9%

Dataset({
    features: ['text', 'label'],
    num_rows: 5000
})
```

앞서 로드한 한국어 토크나이저로 모든 리뷰를 토큰 ID로 변환하고, 라벨을 정수형으로 정리해 모델 입력 형태를 갖춥니다. `max_length=128`로 잘라 길이를 제한하며, 변환이 끝난 뒤 토큰 길이의 평균·중앙값·최댓값을 찍어 자를 만큼 긴 리뷰가 있는지 살펴봅니다.

```python
tokenizer = tokenizer_ko   # klue/bert-base (위에서 로드)

def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(l) for l in batch["label"]]
    return out

train_tok = train_ds.map(tokenize_fn, batched=True).remove_columns(["text", "label"])
eval_tok  = eval_ds.map(tokenize_fn,  batched=True).remove_columns(["text", "label"])

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (int scalar 0=neg / 1=pos)")
# 토큰화된 첫 샘플의 길이
lens = [len(s) for s in train_tok["input_ids"]]
print(f"\nToken length stats — mean: {np.mean(lens):.1f}, median: {np.median(lens):.0f}, max: {max(lens)}")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample label: 1  (int scalar 0=neg / 1=pos)

Token length stats — mean: 21.9, median: 17, max: 117
```

**결과 해석**

토큰 길이가 평균 21.9개, 중앙값 17개로 짧고 최대도 117개라 `max_length=128`에서 사실상 잘리는 리뷰가 거의 없어, 영화평이라는 짧은 텍스트 특성과 설정이 잘 맞습니다.

`klue/bert-base`에 2차원 분류 헤드를 얹어 이진 분류 모델을 만듭니다. `problem_type="single_label_classification"`을 지정하면 `Trainer`가 자동으로 `CrossEntropyLoss`를 쓰며, `id2label`/`label2id`로 0과 1에 negative/positive 이름을 붙입니다. 출력에서 분류 헤드(classifier)가 새로 초기화되었다는 메시지와 파라미터 수, 은닉 차원·사전 크기를 확인합니다.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=2,
    problem_type="single_label_classification",
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
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
print(f"hidden size H:        {model.config.hidden_size}")
print(f"vocab size V:         {model.config.vocab_size:,}")
```

**▶ 실행 결과**

```text
[transformers] BertForSequenceClassification LOAD REPORT from: klue/bert-base
Key                                        | Status     | 
-------------------------------------------+------------+-
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.transform.dense.bias       | UNEXPECTED | 
cls.seq_relationship.bias                  | UNEXPECTED | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
classifier.weight                          | MISSING    | 
classifier.bias                            | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Parameters:             110,618,882  (110.6 M)
Trainable parameters:   110,618,882  (100.0%)
Classifier:           Linear(in_features=768, out_features=2, bias=True)
problem_type:         single_label_classification
hidden size H:        768
vocab size V:         32,000
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:45:11 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   33C    P8             12W /   70W |       3MiB /  15360MiB |      0%      Default |
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

평가 때 호출될 지표 계산 함수를 정의합니다. 모델이 내놓는 2차원 로짓을 softmax로 확률로 바꾼 뒤, 클래스 1 확률과 argmax 예측을 구해 정확도·정밀도·재현율·F1과 AUC까지 한 번에 반환합니다. AUC는 확률값을, 나머지 지표는 예측 라벨을 쓴다는 점을 눈여겨보세요.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits.shape = (B, 2)  → softmax → 클래스 1 확률
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_full = exp / exp.sum(axis=1, keepdims=True)
    probs = probs_full[:, 1]
    preds = probs_full.argmax(axis=1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(labels, probs)),
    }
```

학습 설정을 `TrainingArguments`에 모아 담고 `Trainer`를 구성한 뒤 실제 학습을 돌립니다. T4 제약에 맞춰 2 에폭·배치 16·`fp16=True`로 잡고, 매 에폭마다 평가하도록 설정합니다. 학습이 끝나면 평균 train loss를 출력해 수렴 여부를 가늠합니다.

```python
training_args = TrainingArguments(
    output_dir="./ch15_output",
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
Training done — mean train loss: 0.3008
```

**결과 해석**

2 에폭 학습 후 평균 train loss가 0.30까지 내려가, 한국어 데이터에서도 BERT 이진 분류가 안정적으로 수렴함을 보여 줍니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:46:00 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   54C    P0             32W /   70W |    2627MiB /  15360MiB |     79%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            8636      C   /usr/bin/python3                       2624MiB |
+-----------------------------------------------------------------------------------------+
```

학습이 끝난 모델을 평가 데이터에 적용해 앞서 정의한 지표들을 한 번에 측정합니다. `eval_`로 시작하는 항목만 골라 출력하므로, 한국어 데이터에서의 정확도·F1·AUC를 곧바로 확인할 수 있습니다.

```python
eval_metrics = trainer.evaluate()
print("klue/bert-base NSMC binary — evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
klue/bert-base NSMC binary — evaluation:
             eval_loss: 0.3995
         eval_accuracy: 0.8640
        eval_precision: 0.8773
           eval_recall: 0.8457
               eval_f1: 0.8612
              eval_auc: 0.9248
```

**결과 해석**

5,000건만 학습했는데도 정확도 0.864, F1 0.861, AUC 0.925로, 영어 DistilBERT 챕터에 견줄 만한 성능을 한국어에서도 그대로 재현합니다. precision 0.877과 recall 0.846이 균형 있게 높아 한쪽으로 치우치지 않은 분류기임을 알 수 있습니다.

평가셋 전체에 대해 예측을 다시 받아, 2차원 로짓을 softmax 확률로 바꾸고 `z = z1 - z0`로 단일 로짓을 만들어 둡니다. 이렇게 정리한 확률과 로짓의 분포 범위, 그리고 양성 예측 비율을 출력해 모델이 얼마나 확신을 갖고 양극단으로 판단하는지 살펴봅니다.

```python
preds_output = trainer.predict(eval_tok)
logits2 = preds_output.predictions
labels  = preds_output.label_ids.astype(int)

exp = np.exp(logits2 - logits2.max(axis=1, keepdims=True))
probs_full = exp / exp.sum(axis=1, keepdims=True)
probs = probs_full[:, 1]
logits = logits2[:, 1] - logits2[:, 0]

print(f"logits2 (raw)  shape: {logits2.shape}")
print(f"logit z = z1-z0 range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"prob range:           [{probs.min():.4f}, {probs.max():.4f}]")
print(f"positive prediction rate (prob >= 0.5): {(probs >= 0.5).mean():.1%}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
logits2 (raw)  shape: (1000, 2)
logit z = z1-z0 range: [-5.83, 5.09]
prob range:           [0.0029, 0.9939]
positive prediction rate (prob >= 0.5): 48.1%
```

**결과 해석**

확률이 0.003부터 0.994까지 양극단으로 넓게 퍼지고 logit z도 -5.83-+5.09 범위를 채워, 모델이 어중간하게 0.5 근처에 머무르지 않고 확신을 갖고 판단함을 보여 줍니다. 양성 예측 비율 48.1%는 실제 양성 비율 49.9%와 가까워 임계값 0.5가 적절합니다.

negative와 positive 두 클래스 각각의 정밀도·재현율·F1을 표로 정리해 출력합니다. 전체 정확도뿐 아니라 클래스별 성능을 따로 보면, 모델이 어느 한쪽 감성으로 치우치지 않았는지 확인할 수 있습니다.

```python
# 분류 리포트
print(classification_report(
    labels, probs_full.argmax(axis=1),
    target_names=["negative", "positive"],
    digits=4,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

    negative     0.8516    0.8822    0.8667       501
    positive     0.8773    0.8457    0.8612       499

    accuracy                         0.8640      1000
   macro avg     0.8645    0.8640    0.8639      1000
weighted avg     0.8645    0.8640    0.8640      1000
```

**결과 해석**

부정과 긍정 두 클래스의 F1이 0.867과 0.861로 거의 같아, 어느 한쪽 감성에 편향되지 않고 양쪽을 고르게 잘 맞춥니다.

실제 라벨별로 예측 확률의 분포를 밀도 곡선으로 그립니다. 부정·긍정 두 그룹의 봉우리가 임계선 0.5를 기준으로 얼마나 깔끔하게 갈라지는지, 그리고 0.5 부근에서 두 분포가 겹치는 경계 사례가 얼마나 되는지 시각적으로 확인합니다.

```python
sns.set_theme(style="whitegrid", context="talk")
PAL = {0: "#5B8DEF", 1: "#F47272"}
df_eval = pd.DataFrame({"prob": probs, "logit": logits, "label": labels})

fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df_eval, x="prob", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, clip=(0, 1), ax=ax,
)
ax.axvline(0.5, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_title("klue/bert-base NSMC — Probability Distribution by Actual Label")
ax.set_xlabel("Predicted probability  P(positive)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/15-ko_binary-out1.png)

**결과 해석**

실제 부정 리뷰는 확률 0 근처에, 긍정 리뷰는 1 근처에 봉우리가 쏠려 두 분포가 임계선 0.5를 기준으로 깔끔하게 갈라집니다. 0.5 부근에 겹치는 꼬리가 곧 모델이 망설이는 소수의 경계 사례입니다.

이번에는 확률 대신 `z = z1 - z0` 로짓을 가로축에 두고 같은 분포를 그립니다. sigmoid로 0-1 구간에 눌러 담기 전의 원본 신호가 0을 경계로 두 클래스를 어떻게 가르는지, 확률 그림을 양옆으로 펼친 모습으로 비교해 봅니다.

```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df_eval, x="logit", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, ax=ax,
)
ax.axvline(0.0, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_title("klue/bert-base NSMC — Logit Distribution  (z = z1 − z0)")
ax.set_xlabel("Logit  z = z1 − z0")
ax.set_ylabel("Density")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/15-ko_binary-out2.png)

**결과 해석**

logit z = z1 - z0로 펼친 분포는 확률 그림을 양옆으로 늘여 놓은 모습으로, 0을 기준으로 음수 쪽에 부정, 양수 쪽에 긍정이 분리됩니다. sigmoid가 z를 0-1 확률로 눌러 담기 전의 원본 신호가 이미 두 클래스를 잘 가르고 있음을 보여 줍니다.

예측 확률을 기준으로 가장 자신 있게 긍정으로 본 리뷰, 가장 자신 있게 부정으로 본 리뷰, 그리고 0.5에 가장 가까워 가장 망설인 리뷰를 한 건씩 뽑아 실제 문장과 함께 보여 줍니다. 확신의 정도가 문장의 감성 명료도와 어떻게 이어지는지 직접 눈으로 확인하기 위한 단계입니다.

```python
# 가장 자신있게 positive (probs 최대), 가장 자신있게 negative (probs 최소),
# 가장 망설이는 (|probs - 0.5| 최소) 3가지 샘플
texts = list(df_eval.assign(text=eval_ds["text"])["text"]) if "text" in eval_ds.column_names else list(eval_ds["text"])

# eval_tok 와 eval_ds 의 순서가 같으므로 인덱스 직접 사용
idx_top_pos    = int(np.argmax(probs))
idx_top_neg    = int(np.argmin(probs))
idx_uncertain  = int(np.argmin(np.abs(probs - 0.5)))

samples = [
    ("most confident positive", idx_top_pos),
    ("most confident negative", idx_top_neg),
    ("most uncertain (prob ≈ 0.5)", idx_uncertain),
]

for label_str, idx in samples:
    print("=" * 78)
    print(f"sample #{idx}  ({label_str})")
    print("=" * 78)
    print(f"text:        {texts[idx]}")
    print(f"true label:  {labels[idx]}  ({'positive' if labels[idx] == 1 else 'negative'})")
    print(f"prob(pos):   {probs[idx]:.4f}")
    print(f"logit z:     {logits[idx]:+.2f}")
    pred_label = int(probs[idx] >= 0.5)
    pred_str = "positive" if pred_label == 1 else "negative"
    match = "✓" if pred_label == labels[idx] else "✗"
    print(f"prediction:  {pred_label} ({pred_str})    match: {match}")
    print()
```

**▶ 실행 결과**

```text
==============================================================================
sample #580  (most confident positive)
==============================================================================
text:        아 최고.. 지금 수능 끝나고 보고 있어요ㅠㅠ 현실적인 30대의 사랑이야기~
true label:  1  (positive)
prob(pos):   0.9939
logit z:     +5.09
prediction:  1 (positive)    match: ✓

==============================================================================
sample #368  (most confident negative)
==============================================================================
text:        장난하는것도 아니고...정말 별로다..예전의 주온을 전혀 못 따라가는 졸작
true label:  0  (negative)
prob(pos):   0.0029
logit z:     -5.83
prediction:  0 (negative)    match: ✓

==============================================================================
sample #970  (most uncertain (prob ≈ 0.5))
==============================================================================
text:        이게 30년전 영화라니 최곱니다
true label:  1  (positive)
prob(pos):   0.5027
logit z:     +0.01
prediction:  1 (positive)    match: ✓
```

**결과 해석**

확신이 높은 두 사례("최고", "졸작")는 감성어가 또렷해 확률이 0.99와 0.003으로 양극단을 찍는 반면, 가장 망설인 "이게 30년전 영화라니 최곱니다"는 칭찬과 의외성이 뒤섞여 확률이 0.5027로 경계에 걸칩니다. 모델의 자신감이 곧 문장의 감성 명료도를 반영함을 잘 보여 줍니다.
