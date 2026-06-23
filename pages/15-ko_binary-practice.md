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
Mon Jun 22 03:56:11 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   44C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

한국어 토크나이저(`klue/bert-base`)와 영어 토크나이저(`distilbert-base-uncased`)를 둘 다 불러와, 같은 한국어 문장을 양쪽에 통과시켜 결과를 나란히 비교합니다. 이 챕터 교훈의 절반이 여기서 드러납니다.

```python
tokenizer_ko = AutoTokenizer.from_pretrained("klue/bert-base")
tokenizer_en = AutoTokenizer.from_pretrained("distilbert-base-uncased")

samples = [
    "이 영화 정말 재미있었어요",
    "별로였어요. 시간 낭비",
    "오랜만에 본 명작이네요!",
]

```

**위 코드 읽기** — `klue/bert-base` 는 한국어 위키·뉴스·댓글로 사전학습된 한국어 WordPiece 토크나이저, `distilbert-base-uncased` 는 영어용입니다. 두 토크나이저는 vocab 규모는 비슷해도 *담긴 어휘* 가 완전히 다릅니다. `samples` 에는 양쪽에 통과시킬 짧은 한국어 리뷰 세 개를 둡니다.

```python
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

**위 코드 읽기** — 같은 문장을 `tokenize()` 로 각각 쪼개 토큰 리스트와 개수를 출력합니다. 한국어 토크나이저는 `재미있` + `##었` + `##어요` 처럼 어간·어미 단위로, 영어 토크나이저는 자모 단위까지 산산조각 내거나 `[UNK]` 로 처리합니다. 마지막 두 줄은 양쪽 `vocab_size` 를 출력해 규모(32K vs 30K)는 비슷함을 보여줍니다.

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

`"이 영화 정말 재미있었어요"` 가 한국어 토크나이저로는 6 토큰(`이`, `영화`, `정말`, `재미있`, `##었`, `##어요`)인데 영어 토크나이저로는 14 토큰의 자모 부스러기 + `[UNK]` 로 깨집니다. WordPiece 자체는 같은 알고리즘이지만 *어떤 텍스트로 vocab 을 학습했는가* 만 달라도 한국어 표현력이 이렇게 갈립니다 — 영어 vocab 으로는 한국어 의미를 담을 토큰이 없습니다.

NSMC(네이버 영화 리뷰)의 train/test TSV 를 GitHub raw URL 에서 직접 받습니다. Hugging Face hub 의 nsmc 레포는 로더 스크립트 기반이라 최신 `datasets` 에서 deprecated 됐기 때문입니다. 라벨 분포와 첫 세 줄을 함께 찍어 데이터 형태를 확인합니다.

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

전체 train 약 15만 건이 긍정(74,825)·부정(75,170) 거의 완벽 균형입니다. 클래스가 균형이라 random baseline 의 loss 는 $\log 2 \approx 0.693$ — 학습 첫 step 의 loss 가 이 근처면 정상입니다. 첫 세 줄에서 보이듯 NSMC 는 짧은 구어체 한 줄 리뷰라 영어 Yelp 보다 정보가 적어 조금 더 어렵습니다.

T4 30분 룰에 맞춰 train 5,000건 / eval 1,000건만 추출하고, transformers 표준에 맞게 `document` 컬럼을 `text` 로 바꿔 `datasets.Dataset` 으로 변환합니다.

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

위에서 로드한 한국어 토크나이저로 데이터셋을 토큰화합니다. Ch 11 과 비교하면 토크나이저 인스턴스가 영어 → 한국어로 바뀐 *한 줄 차이* 뿐이고, 라벨을 `int` 로 두는 single-label 셋업은 그대로입니다.

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

토큰 길이 평균 21.9, 중앙값 17로 대부분 매우 짧고 최댓값(117)도 `max_length=128` 안에 들어옵니다. NSMC 한 줄 리뷰의 *짧음* 이 수치로 확인되며, truncation 으로 잘려나가는 정보가 거의 없어 학습도 빠릅니다.

`klue/bert-base` 본체에 `num_labels=2` 분류 헤드를 얹습니다. `problem_type="single_label_classification"` 으로 softmax + `CrossEntropyLoss`(방식 B) 가 자동 선택됩니다 — Ch 11 에서 `distilbert-base-uncased` 였던 자리만 한국어 모델로 교체한 셋업입니다.

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
cls.predictions.transform.dense.bias       | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
cls.seq_relationship.bias                  | UNEXPECTED | 
classifier.bias                            | MISSING    | 
classifier.weight                          | MISSING    | 

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

**결과 해석**

LOAD REPORT 의 `classifier.weight | MISSING` 은 정상입니다 — 사전학습 체크포인트엔 분류 헤드가 없어 새로 초기화된 것이고, 바로 이 헤드를 파인튜닝으로 학습합니다. `cls.predictions.*` 가 `UNEXPECTED` 인 것도 사전학습용 MLM 헤드라 분류엔 안 쓰여 무시됩니다. 총 110.6M 파라미터는 BERT-base 풀 사이즈(12 레이어)로, DistilBERT(약 67M, 6 레이어)의 약 1.5-2배 학습 시간이 듭니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:56:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   43C    P8             13W /   70W |       3MiB /  15360MiB |      0%      Default |
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

평가 시 호출될 지표 함수를 정의합니다. 2차원 logit 을 softmax 해 클래스 1(positive) 확률을 뽑고, accuracy·precision·recall·F1 에 더해 확률 기반의 AUC 까지 계산합니다.

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

Ch 11 과 동일한 hyperparams(2 에폭, lr 2e-5, batch 16, `fp16=True`)로 `Trainer` 를 구성해 학습합니다. T4 에서 bf16 은 미지원이라 항상 `fp16=True` 입니다.

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
Training done — mean train loss: 0.2939
```

**결과 해석**

평균 train loss 0.2939 는 random baseline $\log 2 \approx 0.693$ 보다 한참 아래로, 모델이 한국어 감성 신호를 학습했다는 뜻입니다. 짧은 한국어 리뷰에서도 긍정/부정 키워드를 충분히 잡아낸 것입니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 03:57:18 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   62C    P0             61W /   70W |    2627MiB /  15360MiB |     55%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A             661      C   /usr/bin/python3                       2624MiB |
+-----------------------------------------------------------------------------------------+
```

학습된 모델을 eval 1,000건에 대해 평가해 위에서 정의한 지표들을 출력합니다.

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
             eval_loss: 0.3887
         eval_accuracy: 0.8640
        eval_precision: 0.8773
           eval_recall: 0.8457
               eval_f1: 0.8612
              eval_auc: 0.9292
```

**결과 해석**

accuracy 86.4%, F1 0.861, AUC 0.929 로 NSMC 5K 샘플 + 2 에폭의 전형적 성능 구간(85-88%)에 듭니다. AUC 가 0.93 으로 높아 모델이 긍정/부정을 확률로 잘 분리하고 있음을 보여줍니다. 90%+ 가 목표라면 학습 데이터를 30K 이상으로 늘려야 합니다.

전체 eval 예측을 받아 2차원 raw logit 에서 softmax 확률과, 시각화에 쓸 1차원 logit $z = z_1 - z_0$ 를 만듭니다. 방식 B(2차원 softmax)를 방식 A(1차원 logit) 형태로 환산해 Ch 10·11 과 같은 그림을 그릴 수 있게 하는 단계입니다.

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
logit z = z1-z0 range: [-5.71, 5.19]
prob range:           [0.0033, 0.9944]
positive prediction rate (prob >= 0.5): 48.1%
```

**결과 해석**

확률이 0.0033 ~ 0.9944 까지 양극단으로 넓게 퍼져 있어 모델이 많은 샘플에 자신 있는 판단을 내립니다. positive 예측 비율 48.1% 가 실제 eval 양성 비율(49.9%)과 가까워, 임계값 0.5 기준 예측이 한쪽으로 치우치지 않았습니다.

클래스별 precision/recall/F1 을 한눈에 보는 분류 리포트입니다. negative·positive 가 균형 잡힌 데이터라 두 클래스 지표가 비슷하게 나오는지 확인합니다.

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

실제 라벨별로 예측 확률 $P(\text{positive})$ 의 분포를 KDE 로 겹쳐 그립니다. 두 곡선이 0.5 경계선 좌우로 잘 분리될수록 모델이 한국어 감성을 또렷하게 학습한 것입니다.

```python
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
PAL = {0: "#5B8DEF", 1: "#F47272"}
df_eval = pd.DataFrame({"prob": probs, "logit": logits, "label": labels})

fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df_eval, x="prob", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, clip=(0, 1), ax=ax,
)
ax.axvline(0.5, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_title("klue/bert-base NSMC — 실제 라벨별 확률 분포")
ax.set_xlabel("예측 확률  P(positive)")
ax.set_ylabel("밀도")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/15-ko_binary-out1.png)

같은 분포를 확률 대신 logit $z = z_1 - z_0$ 공간에서 다시 그립니다. 확률은 0~1 로 눌려 양극단이 뭉치지만, logit 공간에선 자신 있는 예측들이 0 양옆으로 멀리 퍼져 분리도가 더 잘 보입니다.

```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df_eval, x="logit", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette=PAL, ax=ax,
)
ax.axvline(0.0, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_title("klue/bert-base NSMC — logit 분포  (z = z1 − z0)")
ax.set_xlabel("logit  z = z1 − z0")
ax.set_ylabel("밀도")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/15-ko_binary-out2.png)

모델이 *가장 자신 있는* positive·negative 샘플과 *가장 망설이는*(prob ≈ 0.5) 샘플을 각각 골라 실제 한국어 리뷰를 직접 읽어봅니다. 짧은 리뷰가 모델 눈에 어떻게 보이는지 감을 잡는 단계입니다.

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
prob(pos):   0.9944
logit z:     +5.19
prediction:  1 (positive)    match: ✓

==============================================================================
sample #169  (most confident negative)
==============================================================================
text:        한마디로노잼 재미없음
true label:  0  (negative)
prob(pos):   0.0033
logit z:     -5.71
prediction:  0 (negative)    match: ✓

==============================================================================
sample #978  (most uncertain (prob ≈ 0.5))
==============================================================================
text:        영화보다가 진짜 기도드릴뻔했다. '제발 끝나게해주세요'라고..
true label:  0  (negative)
prob(pos):   0.5040
logit z:     +0.02
prediction:  1 (positive)    match: ✗
```

**결과 해석**

가장 자신 있는 두 샘플은 `"아 최고.."`(prob 0.9944), `"한마디로노잼 재미없음"`(prob 0.0033)처럼 감성이 노골적인 표현이라 모델이 거의 확신합니다. 반면 망설인 샘플 `"제발 끝나게해주세요"` 는 *반어적 부정* 인데 표면 단어에 명시적 부정어가 없어 prob 0.504 로 갈팡질팡하다 오답을 냈습니다 — 짧은 한국어 리뷰에서 반어가 모델에게 가장 어려운 케이스임을 보여줍니다.
