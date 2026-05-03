"""Build 10_bert_binary_sigmoid/10_bert_binary_sigmoid.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "10_bert_binary_sigmoid" / "10_bert_binary_sigmoid.ipynb"

cells = []
_counter = 0


def _cid():
    global _counter
    _counter += 1
    return f"cell{_counter:03d}"


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": text,
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ----- 1. Title -----
md(r"""# Chapter 10. BERT Binary 방식 A — sigmoid + BCEWithLogitsLoss

**목표**: Ch 4(sklearn)에서 본 *두 방식 동등성* 의 BERT 버전을 시작합니다. 이번 챕터는 **방식 A**인 sigmoid + BCE 패턴을 BERT로 학습합니다 (`num_labels=1`, `problem_type="multi_label_classification"`). 다음 Ch 11에서 같은 데이터를 **방식 B**(softmax + CE)로 학습한 뒤 두 결과를 비교합니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 10분 (모델 다운로드 + 2 에폭 학습 + 평가)

---

## 학습 흐름

1. 🚀 **실습**: Ch 3과 같은 Yelp 이진화 데이터를 BERT로 학습 — `num_labels=1` + sigmoid + `BCEWithLogitsLoss`
2. 🔬 **해부**: 학습 후 sigmoid 확률 분포 직접 확인, 평가 지표(accuracy/precision/recall/F1/AUC) 계산
3. 🛠️ **다음 챕터(Ch 11) 예고**: 같은 task에 `num_labels=2` + softmax + `CrossEntropyLoss` 로 다시 학습해 두 방식 결과 비교

---

> 📒 **사전 학습 자료**: Ch 4 (sklearn binary on softmax) — 두 방식이 수학적으로 동등하다는 것을 식으로 본 챕터. Ch 9 (BERT regression) — `Trainer` 기본 골격.""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression(multi_class="multinomial")` | `TfidfVectorizer()` | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` |
| 9 | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp (별점 1-5) | `Linear(H, 1)` | 없음 | `MSELoss` |
| **10 ← 여기** | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp 이진화 | **`Linear(H, 1)`** | **sigmoid** | **`BCEWithLogitsLoss`** |
| 11 (다음) | DistilBERT 파인튜닝 | 같음 | Yelp 이진화 | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 9)

| 축 | Ch 9 | Ch 10 |
|---|---|---|
| Task | 회귀 | **이진 분류 (방식 A)** |
| `num_labels` | 1 | **1** (그대로) |
| `problem_type` | `"regression"` | **`"multi_label_classification"`** ← BCE 자동 적용 트릭 |
| Activation | 없음 | **sigmoid** (output head는 1차원, 학습 시 logit이 sigmoid 통과 후 BCE) |
| Loss | `MSELoss` | **`BCEWithLogitsLoss`** |
| 라벨 | float (1-5) 별점 | **float [0.0 또는 1.0]** (multi-hot 1차원 벡터로 둠) |
| 데이터 | Yelp 별점 1-5 | **Yelp 이진화** (4-5 → 1, 1-2 → 0, 3 제외) |

### `num_labels=1` + `problem_type="multi_label_classification"` 의 트릭

`Trainer` 의 자동 loss 매핑은 이렇게 작동합니다 ([Ch 9에서 본 표](../09_bert_regression/09_bert_regression.ipynb)).

| `problem_type` | 자동 적용 loss | num_labels | 라벨 형식 |
|---|---|---|---|
| `"regression"` | `MSELoss` | 보통 1 | float |
| `"single_label_classification"` | `CrossEntropyLoss` | K (≥2) | int 인덱스 |
| `"multi_label_classification"` | **`BCEWithLogitsLoss`** | K (≥1) | **multi-hot float** |

방식 A는 *binary 분류이지만 num_labels=1* 형태를 유지해야 합니다. 그러려면 `multi_label_classification` 으로 두어 BCE를 적용시키되, *num_labels=1짜리 multi-label* 즉 라벨을 길이 1짜리 multi-hot 벡터(`[0.0]` 또는 `[1.0]`)로 만들면 됩니다. 이게 sklearn `LogisticRegression()` 의 sigmoid+BCE와 정확히 같은 셋업입니다.""")

# ----- 4. Loss 노트 -----
md(r"""## 📐 Loss 노트 — `BCEWithLogitsLoss` (Ch 3 그대로, BERT 맥락에서 다시)

수식과 직관은 Ch 3에서 봤습니다.

$$L = -\frac{1}{N}\sum_{i=1}^{N}\left[\,y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\,\right]$$

이번 챕터에서 새로운 점:

1. **모델이 BERT** 라 logit $z = w^\top h_{[CLS]} + b$ 의 *분포 표현 $h_{[CLS]}$* 가 768차원 hidden state를 압축한 결과입니다 (sklearn TF-IDF 입력보다 풍부).
2. `BCEWithLogits` 의 *Logits* — 모델 마지막 단의 raw 점수에 sigmoid를 따로 통과시키지 않고 BCE 안에서 한꺼번에 처리하기 때문에 **수치적으로 안정** 합니다.
3. `Trainer` 가 `problem_type="multi_label_classification"` 만 보고 자동으로 BCE를 골라줍니다. 우리는 라벨을 `[0.0]` 또는 `[1.0]` float 형태로 두기만 하면 됩니다.

**숫자로 감 잡기** — Ch 3 표 그대로:

| 정답 $y$ | 예측 확률 $\hat p$ | 손실 $-\log \hat p$ |
|---|---|---|
| 1 | 0.9 | 0.105 |
| 1 | 0.5 | 0.693 |
| 1 | 0.1 | **2.303** |

확률이 0에 가까울수록 손실이 로그 스케일로 폭증한다는 BCE의 성격은 sklearn에서든 BERT에서든 동일합니다. 다른 점은 *어떻게 그 확률을 만드느냐* 입니다 — sklearn은 단어 빈도, BERT는 attention으로 압축한 문장 표현.""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

Ch 7-9와 같은 `distilbert-base-uncased` WordPiece 토크나이저. 토크나이저·데이터 가공 파이프라인은 Ch 8에서 익힌 패턴을 그대로 적용합니다.

> **다음 챕터(Ch 11)**: 같은 토크나이저, 같은 데이터, 같은 BERT 본체. 변하는 건 출력 헤드가 1차원에서 2차원으로 늘어나고 sigmoid가 softmax로 바뀐다는 점뿐입니다.""")

# ----- 6. install + import -----
code(r"""!pip install -q transformers datasets""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score,
)

plt.rcParams["axes.unicode_minus"] = False

# 디바이스 자동 감지 — Colab T4(CUDA), Mac(MPS), 그 외는 CPU
USE_CUDA = torch.cuda.is_available()
USE_MPS  = torch.backends.mps.is_available()
DEVICE_KIND = "cuda" if USE_CUDA else ("mps" if USE_MPS else "cpu")
USE_FP16 = USE_CUDA  # fp16은 CUDA에서만, MPS에서는 fp32

print(f"PyTorch:        {torch.__version__}")
print(f"디바이스:       {DEVICE_KIND}")
if USE_CUDA:
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
elif USE_MPS:
    print("Apple Silicon MPS 사용 — Colab T4 대비 약간 느릴 수 있어요 (fp16 비활성)")
else:
    print("⚠️  CPU 런타임 — 학습이 매우 느립니다. T4 또는 MPS 권장.")""")

# ----- 7. nvidia-smi baseline -----
md(r"""**baseline VRAM** (CUDA에서만 의미; Mac MPS면 자동 skip):""")

code(r"""!command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo '(nvidia-smi 없음 — Mac/CPU 환경)'""")

# ----- 8. 데이터 준비 -----
md(r"""## 1. 🚀 데이터 — Yelp 이진화 (Ch 3·4와 동일)

별점 4-5는 `1.0` (긍정), 1-2는 `0.0` (부정), 3은 제외. 라벨을 *float 1차원 multi-hot 벡터* (`[0.0]` 또는 `[1.0]`) 형태로 둡니다 — 이게 BCE를 자동 적용시키는 핵심 형식.""")

code(r"""tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("yelp_review_full")
train_full = ds["train"].shuffle(seed=42).select(range(5000))
eval_full  = ds["test"].shuffle(seed=42).select(range(1000))

# 별점 3 제외 + 이진화
def to_binary(example):
    star = example["label"] + 1   # 0-4 → 1-5
    if star == 3:
        return False
    return True

def add_binary(batch):
    bins = []
    for lbl in batch["label"]:
        star = lbl + 1
        bins.append(1.0 if star >= 4 else 0.0)
    batch["binary"] = bins
    return batch

train_bin = train_full.filter(lambda x: (x["label"] + 1) != 3).map(add_binary, batched=True)
eval_bin  = eval_full.filter(lambda x:  (x["label"] + 1) != 3).map(add_binary, batched=True)

print(f"train (별점 3 제외 후): {len(train_bin)}")
print(f"eval  (별점 3 제외 후): {len(eval_bin)}")
print(f"train 긍정 비율: {sum(train_bin['binary']) / len(train_bin):.1%}")""")

# ----- 9. 토큰화 -----
code(r"""def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # 라벨을 길이 1짜리 multi-hot 벡터로 — Trainer가 BCEWithLogitsLoss 자동 적용
    out["labels"] = [[float(b)] for b in batch["binary"]]
    return out

train_tok = train_bin.map(tokenize_fn, batched=True).remove_columns(["text", "label", "binary"])
eval_tok  = eval_bin.map(tokenize_fn,  batched=True).remove_columns(["text", "label", "binary"])

print(train_tok)
print(f"\n첫 샘플 라벨: {train_tok[0]['labels']}  (길이 1짜리 float 벡터)")""")

# ----- 10. 모델 로드 -----
md(r"""## 2. 모델 로드 — 방식 A 셋업

`num_labels=1` + `problem_type="multi_label_classification"` 이 핵심.""")

code(r"""model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1,
    problem_type="multi_label_classification",   # ← BCEWithLogitsLoss 자동 매핑
)

def param_summary(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

total, trainable = param_summary(model)
print(f"파라미터 수:        {total:>13,}  ({total/1e6:.1f} M)")
print(f"학습되는 파라미터:  {trainable:>13,}  ({trainable/total:.1%})")
print(f"분류 헤드:          {model.classifier}")
print(f"problem_type:       {model.config.problem_type}")""")

code(r"""!command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo '(nvidia-smi 없음 — Mac/CPU 환경)'""")

# ----- 11. 학습 -----
md(r"""## 3. 학습 — Ch 9 골격 그대로

`compute_metrics` 만 binary 분류용으로 새로 짭니다 — sigmoid + threshold 0.5 로 0/1 예측을 만들고 accuracy/F1/AUC 계산.""")

code(r"""def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.flatten()         # (B, 1) → (B,)
    labels = labels.flatten().astype(int)

    # logit → 확률 (sigmoid 직접 적용)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(labels, probs)),
    }""")

code(r"""training_args = TrainingArguments(
    output_dir="./ch10_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=42,
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
print(f"\n학습 완료 — 평균 train loss: {train_result.training_loss:.4f}")""")

code(r"""!command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo '(nvidia-smi 없음 — Mac/CPU 환경)'""")

# ----- 12. 평가 -----
md(r"""## 4. 🔬 평가 — sigmoid 확률 분포 직접 확인

`Trainer.predict()` 로 logit을 받아 sigmoid를 통과시킨 확률 분포를 살펴봅니다.""")

code(r"""# 평가 metric
eval_metrics = trainer.evaluate()
print("BERT 방식 A 평가:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")""")

code(r"""# logit → 확률
preds_output = trainer.predict(eval_tok)
logits = preds_output.predictions.flatten()
probs  = 1.0 / (1.0 + np.exp(-logits))
labels = preds_output.label_ids.flatten().astype(int)

print(f"logit 범위:       [{logits.min():.2f}, {logits.max():.2f}]")
print(f"확률 범위:        [{probs.min():.4f}, {probs.max():.4f}]")
print(f"긍정 예측 비율 (확률 ≥ 0.5): {(probs >= 0.5).mean():.1%}")
print(f"\n앞 5개 샘플:")
print(pd.DataFrame({
    "label": labels[:5],
    "logit": logits[:5].round(2),
    "prob":  probs[:5].round(4),
    "pred":  (probs[:5] >= 0.5).astype(int),
}).to_string(index=False))""")

code(r"""# 확률 분포 시각화 — 정답 0/1 별로 따로
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(probs[labels == 0], bins=30, alpha=0.6, label="actual=0 (negative)", color="C0")
ax.hist(probs[labels == 1], bins=30, alpha=0.6, label="actual=1 (positive)", color="C1")
ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="threshold=0.5")
ax.set_xlabel("Predicted probability  P(y=1)")
ax.set_ylabel("Count")
ax.set_title("Method A (sigmoid + BCE) — Probability Distribution")
ax.legend()
plt.tight_layout()
plt.show()""")

code(r"""# 상세 분류 리포트
print(classification_report(
    labels, (probs >= 0.5).astype(int),
    target_names=["negative", "positive"],
    digits=4,
))""")

# ----- 13. 결과 보존 (다음 챕터에서 비교) -----
md(r"""## 5. 결과 저장 — Ch 11에서 비교용

다음 챕터 Ch 11에서 같은 데이터에 *방식 B* (softmax+CE)로 학습한 뒤 *이번 방식 A* 의 결과와 비교합니다. 평가 지표와 확률 예측을 디스크에 저장해 두면 비교가 깔끔해집니다.""")

code(r"""import json, os

os.makedirs("./shared_binary_results", exist_ok=True)

# numpy 배열을 그대로 저장
np.save("./shared_binary_results/method_a_probs.npy", probs)
np.save("./shared_binary_results/method_a_labels.npy", labels)

# metric 요약
method_a_summary = {
    "method": "A (sigmoid + BCE, num_labels=1)",
    "metrics": {
        k.replace("eval_", ""): v
        for k, v in eval_metrics.items()
        if k.startswith("eval_") and isinstance(v, float)
    },
}
with open("./shared_binary_results/method_a_summary.json", "w") as f:
    json.dump(method_a_summary, f, indent=2)

print("저장 완료: ./shared_binary_results/")
for f in sorted(os.listdir("./shared_binary_results")):
    size_kb = os.path.getsize(f"./shared_binary_results/{f}") / 1024
    print(f"  {f}  ({size_kb:.1f} KB)")""")

md(r"""**참고**: Colab은 세션이 끝나면 `./shared_binary_results/` 가 사라집니다. *Drive에 보존* 하려면 다음과 같이 마운트.

```python
from google.colab import drive
drive.mount("/content/drive")
import shutil
shutil.copytree("./shared_binary_results", "/content/drive/MyDrive/neuqes-101/shared_binary_results")
```

Ch 11 노트북은 같은 세션에서 이어 돌리거나, 같은 데이터·seed·모델로 다시 학습해서 결과를 만든 뒤 비교합니다.""")

# ----- 14. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModelForSequenceClassification(num_labels=1, problem_type="multi_label_classification")` | num_labels=1 + multi_label로 BCE 자동 매핑 | Ch 12-13에서 multi-hot 라벨로 재사용 |
| `sklearn.metrics.precision_recall_fscore_support` | 이진 분류 지표 한 묶음 | Ch 11·15·17에서 동일 |
| `sklearn.metrics.roc_auc_score` | AUC 계산 (확률 임계값 무관) | 분류 챕터마다 사용 |
| `numpy 1/(1+exp(-x))` | sigmoid 직접 구현 (모델 logit → 확률) | Ch 7에서 본 동일 패턴 |""")

# ----- 15. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. `num_labels=1` 인데 왜 `problem_type="single_label_classification"` 이 아닌 `"multi_label_classification"` 으로 두나요?
2. `BCEWithLogitsLoss` 의 *Logits* 가 의미하는 바는? sigmoid를 따로 적용하지 않는 이유는?
3. 학습 후 sigmoid 확률 분포가 정답 0과 정답 1 그룹에서 어떻게 다르게 보이나요? (시각화 그래프 해석)
4. 같은 binary 분류를 sklearn `LogisticRegression()` 으로 풀 때(Ch 3)와 BERT로 풀 때(Ch 10), accuracy 차이가 어디서 오나요?""")

# ----- 16. FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) `num_labels=1` 일 때 `problem_type="single_label_classification"` 으로 두면 안 되나요?

안 됩니다. `single_label_classification` 은 CrossEntropyLoss를 적용하는데, CE는 `num_labels >= 2` 를 가정합니다. `num_labels=1` 로 두고 CE를 적용하면 logit이 단 하나라 softmax가 항상 1을 출력하고, loss가 항상 0이 되어 학습이 안 됩니다.

방식 A의 정수 라벨이 0/1 두 개이지만 *출력은 1차원 logit* 이므로 — 이 형태를 `Trainer` 가 이해하게 하려면 `multi_label_classification` 으로 두어야 BCE가 자동 적용됩니다.

### Q2. (이론) 라벨을 길이 1 multi-hot 벡터(`[0.0]`, `[1.0]`)로 두는 이유는?

`BCEWithLogitsLoss` 가 *logits 와 같은 shape의 float 텐서* 를 라벨로 받기 때문입니다.

```python
# logits.shape = (batch, num_labels=1)
# labels.shape = (batch, num_labels=1)  ← 같은 shape여야 함
loss = BCEWithLogitsLoss()(logits, labels)
```

라벨을 scalar(`float(b)`)로 두면 batching 시 shape가 (batch,)가 되어 (batch, 1) logits와 안 맞습니다. `[float(b)]` 한 번 감싸 length-1 list로 두면 (batch, 1) shape이 자동으로 만들어집니다.

### Q3. (실무) `compute_metrics` 의 `eval_pred.predictions` 는 어떤 형태인가요?

`AutoModelForSequenceClassification` 의 출력은 *logits* 입니다 (sigmoid 적용 전).

- 방식 A (이번 챕터, num_labels=1): shape `(batch, 1)`. flatten 후 sigmoid 적용 → 확률.
- 방식 B (Ch 11, num_labels=2): shape `(batch, 2)`. softmax 적용 → 두 확률.

`compute_metrics` 안에서 우리가 sigmoid·argmax·threshold 같은 후처리를 직접 합니다.

### Q4. (이론) sklearn LogReg(Ch 3)와 BERT(Ch 10)의 accuracy 차이는 어디서?

세 가지 layer에서 차이가 옵니다.

1. **단어 표현**: TF-IDF는 단어 독립 벡터, BERT는 문맥 attention. `"not bad"` vs `"bad"` 구분이 BERT만 가능.
2. **모델 capacity**: TF-IDF + LogReg는 ~10K개 가중치, BERT는 67M개. 표현력 6,700배 차이.
3. **사전학습**: BERT는 위키피디아·BookCorpus로 미리 학습돼 일반 언어 지식이 모델에 인코딩됨.

다만 Yelp 별점 같은 *단어 빈도* 가 강한 신호인 task는 sklearn도 90% 이상 잘 맞춰서, 차이가 *극적이지 않은* 경우도 있습니다. 차이가 명확히 드러나는 task는 부정·반어가 많은 sentiment 데이터(SST-2)나 NLI(자연어 추론)입니다.

### Q5. (실무) 같은 데이터로 두 번 학습하면 결과가 똑같나요?

**거의 같지만 미세한 차이가 있습니다**. 이유:

- random seed (`seed=42`) 고정해도 *CUDA 비결정성* 이 남음 (cuDNN의 일부 알고리즘이 floating point 연산 순서를 비결정적으로 처리).
- DataLoader의 shuffle, dropout, layer init은 seed로 통제되지만 GPU 부동소수 연산 자체가 결정적이지 않음.

완전히 결정적으로 만들려면 `torch.use_deterministic_algorithms(True)` 같은 옵션이 있지만 속도가 느려집니다. 실무에선 *seed 여러 개로 학습 후 평균* 이 일반적.

### Q6. (실무) 학습 도중 학습률을 어떻게 조정하나요?

`Trainer` 는 기본으로 *linear warmup → linear decay* 스케줄러를 적용합니다. `lr_scheduler_type` 으로 바꿀 수 있습니다.

```python
TrainingArguments(
    ...,
    lr_scheduler_type="cosine",   # cosine decay (BERT 큰 모델에서 흔함)
    warmup_ratio=0.1,             # 첫 10%를 warmup
)
```

작은 데이터·짧은 학습이면 default linear가 무난. 학습 중 LR을 직접 모니터링하려면 `report_to="wandb"` 같은 트래커 필수.""")

# ----- 17. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# 라벨을 scalar로 두고 학습 시도
def tokenize_wrong(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [float(b) for b in batch["binary"]]   # scalar (감싸지 않음!)
    return out
```

힌트: `BCEWithLogitsLoss(logits, labels)` 에서 logits.shape는 `(B, 1)` 인데 labels.shape가 `(B,)` 가 되어 broadcasting 또는 shape mismatch 에러가 납니다. 정확히 어떤 메시지가 뜨는지 확인해 보세요.""")

# ----- 18. next -----
md(r"""## 다음 챕터 예고

**Chapter 11. BERT Binary 방식 B — softmax + CrossEntropyLoss**

- 같은 Yelp 이진화 데이터, 같은 BERT 본체
- `num_labels=2` + `problem_type="single_label_classification"` (BERT 표준)
- Activation은 softmax, Loss는 `CrossEntropyLoss`
- 학습 후 *방식 A의 저장된 결과* 와 직접 비교 — 두 방식이 거의 같은 확률 분포를 만들어내는지 확인 (Ch 4의 sklearn 동등성을 BERT에서 다시)
- 학습된 가중치 비교, 두 방식의 prediction agreement 측정""")


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT.relative_to(REPO)}  ({len(cells)} cells)")
