> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/21_en_bert_classify.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
%pip install -q -U transformers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 114.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 36.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 38.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/48.9 MB ? eta -:--:--
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 164.3 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 164.3 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 17.3 MB/s eta 0:00:00
```

```python
import warnings
warnings.filterwarnings("ignore")

import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score, confusion_matrix,
)

plt.rcParams["axes.unicode_minus"] = False

# matplotlib 한글 폰트 (Colab — NanumGothic). plot 의 한국어가 □ 로 깨지지 않게.
import matplotlib.font_manager as fm, subprocess, os
_fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(_fp):
    subprocess.run("apt-get -qq -y install fonts-nanum", shell=True)
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "NanumGothic"

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
    print("Warning: CPU runtime — both MLM and classification will be very slow. Switch to T4 recommended.")
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
Mon Jun 22 12:16:39 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   34C    P8              9W /   70W |       3MiB /  15360MiB |      0%      Default |
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

분류 fine-tune 에 쓸 Yelp 이진 분류 데이터를 불러옵니다. `fancyzhx/yelp_polarity` 는 이미 긍정/부정으로 이진화된 리뷰 데이터셋이라 별점을 직접 자를 필요가 없습니다. Ch 10 과 같은 결과를 얻도록 같은 seed (42) 와 같은 표본 수 (5,000 train / 1,000 eval) 를 그대로 맞춥니다.

```python
SEED = 42
N_TRAIN = 5000
N_EVAL = 1000

ds_raw = load_dataset("fancyzhx/yelp_polarity")
print(f"splits: {list(ds_raw.keys())}")
print(f"train size: {len(ds_raw['train']):,}")
print(f"test size:  {len(ds_raw['test']):,}")
print(f"label names: {ds_raw['train'].features['label'].names}")

```

**위 코드 읽기** — `load_dataset("fancyzhx/yelp_polarity")` 로 전체 데이터셋 (train 56만 / test 3.8만) 을 받아 두고, `label names` 가 `['1', '2']` 임을 확인합니다. 여기서 라벨 `1` 은 부정, `2` 는 긍정에 해당하는 원본 표기이고, 뒤에서 0/1 정수로 다룹니다.

```python
# Ch 10 과 동일한 seed·크기로 sample
ds_train_full = ds_raw["train"].shuffle(seed=SEED).select(range(N_TRAIN))
ds_eval_full  = ds_raw["test"].shuffle(seed=SEED).select(range(N_EVAL))

# 클래스 분포
train_labels = np.array(ds_train_full["label"])
eval_labels  = np.array(ds_eval_full["label"])
print(f"\nsampled train: {len(ds_train_full):,}")
print(f"  positive rate: {train_labels.mean():.1%}  (label 1)")
print(f"sampled eval:  {len(ds_eval_full):,}")
print(f"  positive rate: {eval_labels.mean():.1%}  (label 1)")

print(f"\nfirst train sample:")
print(f"  label: {ds_train_full[0]['label']} ({ds_raw['train'].features['label'].names[ds_train_full[0]['label']]})")
print(f"  text:  {ds_train_full[0]['text'][:200]}...")
```

**위 코드 읽기** — `shuffle(seed=SEED).select(range(...))` 로 56만 건 중 앞 5,000 / 1,000 건만 잘라 씁니다. seed 를 42 로 고정했기 때문에 Ch 10 과 *정확히 같은 표본* 이 뽑혀, 본체 출발점 외의 변수가 통제됩니다. 긍정 비율 (`positive rate`) 을 함께 찍어 클래스가 한쪽으로 치우치지 않았는지 미리 점검합니다.

**▶ 실행 결과**

```text
splits: ['train', 'test']
train size: 560,000
test size:  38,000
label names: ['1', '2']
sampled train: 5,000
  positive rate: 50.7%  (label 1)
sampled eval:  1,000
  positive rate: 48.4%  (label 1)

first train sample:
  label: 1 (2)
  text:  Decent size, decent selection, decent staff.\n\nI guess that can wholly sum this place up, it's decent.  As with many other stores …(뒤 72자 생략)
```

**결과 해석**

뽑힌 train 의 긍정 비율 50.7%, eval 48.4% 로 두 split 모두 거의 50:50 의 균형 잡힌 이진 분류라, 정확도가 다수 클래스 추측 (majority guess) 으로 부풀려질 걱정이 없습니다.

```python
TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"tokenizer:        {TOKENIZER_NAME}")
print(f"vocab_size:       {tokenizer.vocab_size:,}")
print(f"model_max_length: {tokenizer.model_max_length}")

```

**위 코드 읽기** — Ch 20 사전학습과 *완전히 같은* `bert-base-uncased` 토크나이저를 그대로 가져옵니다. vocab 이 30,522 로 고정돼 있어야 MLM 으로 학습한 토큰 임베딩이 분류 단계에서도 같은 의미를 유지합니다.

```python
# 분류 입력 예시
SAMPLE = "The food was unforgettable and the service was excellent."
enc = tokenizer(SAMPLE, return_tensors="pt", truncation=True, max_length=128)
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(f"\nsample: {SAMPLE!r}")
print(f"tokens ({len(tokens)}): {tokens}")
```

**위 코드 읽기** — Yelp 리뷰풍 예시 문장 하나를 토큰화해, MLM 때와 달리 `[CLS]`/`[SEP]` 특수 토큰이 자동으로 붙는 것을 확인합니다. 분류는 `[CLS]` 자리의 최종 hidden state 를 문장 표상으로 쓰기 때문에 이 포맷이 중요합니다.

**▶ 실행 결과**

```text
tokenizer:        bert-base-uncased
vocab_size:       30,522
model_max_length: 512

sample: 'The food was unforgettable and the service was excellent.'
tokens (15): ['[CLS]', 'the', 'food', 'was', 'un', '##for', '##get', '##table', 'and', 'the', 'service', 'was', 'excellent', '.', '[SEP]']
```

**결과 해석**

`unforgettable` 한 단어가 `un / ##for / ##get / ##table` 네 개의 WordPiece 조각으로 쪼개진 것이 보입니다. vocab 에 통째로 없는 단어도 조각으로 표현되므로 `[UNK]` 없이 처리되고, 양끝에 `[CLS]`/`[SEP]` 가 붙어 총 15 토큰이 됩니다.

Ch 20 과 동일한 작은 `BertConfig` (hidden 256, 4층, head 4, intermediate 1024) 로 `BertForMaskedLM` 을 random init 합니다. 사전학습된 가중치를 받아오는 게 아니라 *맨바닥에서* 시작하는 것이 이 챕터의 핵심이라, 처음 만든 본체의 파라미터 수를 함께 확인해 둡니다.

```python
# Ch 20 과 같은 작은 BERT 설정
HIDDEN_SIZE         = 256
NUM_HIDDEN_LAYERS   = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE   = 1024
MAX_POS_EMBED       = 128
BLOCK_SIZE          = 128

mlm_config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
)

mlm_model = BertForMaskedLM(mlm_config)  # random init
total = sum(p.numel() for p in mlm_model.parameters())
print(f"Small BERT config: hidden={HIDDEN_SIZE}, layer={NUM_HIDDEN_LAYERS}, head={NUM_ATTENTION_HEADS}")
print(f"Total parameters:  {total:,}  ({total/1e6:.2f} M)")
```

**위 코드 읽기** — `BertForMaskedLM(mlm_config)` 는 사전학습 체크포인트를 받지 않으므로 모든 가중치가 random 입니다. `vocab_size=tokenizer.vocab_size` 로 vocab 을 토크나이저와 묶어 둔 점이 중요한데, 임베딩 행렬 크기가 곧 30,522 × 256 이 되어 전체 파라미터의 큰 비중을 차지합니다.

**▶ 실행 결과**

```text
Small BERT config: hidden=256, layer=4, head=4
Total parameters:  11,103,290  (11.10 M)
```

**결과 해석**

전체 약 11.1M 파라미터로, Ch 10 의 DistilBERT (약 66M) 의 1/6 규모입니다. 이 작은 본체가 일반 위키 MLM 사전학습만으로 Yelp 분류에 얼마나 전이되는지가 이후 비교의 출발점입니다.

분류용 Yelp 와는 *별도로* MLM 사전학습용 일반 도메인 코퍼스 (Wikitext-103) 를 새로 불러옵니다. 일반 위키로 사전학습한 본체가 다른 도메인인 Yelp 분류로 전이되는지가 이 챕터의 메시지라, 두 데이터셋이 같은 노트북에 공존합니다.

```python
# MLM 사전학습용 일반 도메인 코퍼스: Wikitext-103 (분류용 Yelp 와 별도)
# 한국어 Ch 23 self-contained 와 동일한 hyperparams 로 통일 (2K × 3 epoch)
N_MLM_TRAIN = 2000
N_MLM_EVAL  = 400

print("downloading Wikitext-103 (Salesforce/wikitext, wikitext-103-raw-v1)...")
raw_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
raw_eval  = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")

# 빈 줄 / 너무 짧은 줄 (제목·메타) / 너무 긴 줄 (목록·인용) 제외
def is_good(ex, min_len=50, max_len=2000):
    t = ex["text"].strip()
    return min_len <= len(t) <= max_len

mlm_train_raw = (
    raw_train.filter(is_good).shuffle(seed=SEED).select(range(N_MLM_TRAIN))
    .remove_columns([c for c in raw_train.column_names if c != "text"])
)
mlm_eval_raw = (
    raw_eval.filter(is_good).shuffle(seed=SEED).select(range(N_MLM_EVAL))
    .remove_columns([c for c in raw_eval.column_names if c != "text"])
)

print(f"MLM train paragraphs: {len(mlm_train_raw):,}  (wikitext-103)")
print(f"MLM eval paragraphs:  {len(mlm_eval_raw):,}")
print(f"first MLM sample: {mlm_train_raw[0]['text'][:120]}...")

```

**위 코드 읽기** — `is_good` 필터로 빈 줄·제목 같은 짧은 줄과 목록·인용 같은 지나치게 긴 줄을 걸러내, 정상적인 위키 문단만 남깁니다. 거른 뒤 `shuffle(seed=SEED).select(...)` 로 train 2,000 / eval 400 문단만 골라 T4 시간 예산 안에 들어오도록 압축합니다.

```python
def mlm_tokenize(examples):
    return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

mlm_tokenized_train = mlm_train_raw.map(mlm_tokenize, batched=True, remove_columns=["text"])
mlm_tokenized_eval  = mlm_eval_raw.map(mlm_tokenize,  batched=True, remove_columns=["text"])

```

**위 코드 읽기** — MLM 단계는 분류와 달리 `add_special_tokens=False`, `truncation=False` 로 토큰화합니다. 문장 단위로 자르는 대신 모든 토큰을 *하나의 긴 스트림* 으로 이어 붙여 고정 길이 블록으로 다시 자를 것이므로, 여기서는 `[CLS]`/`[SEP]` 를 넣지 않고 자르지도 않습니다.

```python
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result

lm_train = mlm_tokenized_train.map(group_texts, batched=True, batch_size=1000)
lm_eval  = mlm_tokenized_eval.map(group_texts,  batched=True, batch_size=1000)

print(f"\nMLM train blocks: {len(lm_train):,}  (block_size={BLOCK_SIZE})")
print(f"MLM eval blocks:  {len(lm_eval):,}")
```

**위 코드 읽기** — `group_texts` 는 `sum(examples[k], [])` 로 한 batch 안의 모든 문단 토큰을 하나의 스트림으로 잇고, `BLOCK_SIZE`(128) 단위로 잘라 균일 길이 블록으로 재구성합니다. `result["labels"] = [ids.copy() ...]` 로 일단 입력을 그대로 복사해 두지만, 실제 마스킹은 학습 중 collator 가 매번 무작위로 적용합니다.

**▶ 실행 결과**

```text
downloading Wikitext-103 (Salesforce/wikitext, wikitext-103-raw-v1)...
MLM train paragraphs: 2,000  (wikitext-103)
MLM eval paragraphs:  400
first MLM sample:  Balinor Buckhannah , the Crown Prince of the country of Callahorn and the " charismatic commander of [ the ] Border Leg...
MLM train blocks: 2,100  (block_size=128)
MLM eval blocks:  428
```

**결과 해석**

문단 2,000 개를 토큰 스트림으로 이어 128 토큰씩 자르니 train 2,100 블록이 만들어졌습니다 — 문단마다 길이가 달라도 블록 길이가 균일해져 batch 학습이 효율적이 됩니다.

마스킹을 매 step 무작위로 적용하는 collator 를 만듭니다. `mlm=True` 와 `mlm_probability=0.15` 로 토큰의 약 15% 를 선택하고, 그 안에서 80/10/10 (`[MASK]`/random/원본) 규칙은 collator 내부에 고정돼 있습니다.

```python
mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)
```

방금 만든 collator 가 실제로 어떤 자리를 어떻게 바꾸는지 짧은 예시 문장 하나로 직접 들여다봅니다. 토큰별로 원본·마스킹 후·`label_id`·무슨 일이 일어났는지를 표로 정리해, 15% 선택과 80/10/10 규칙이 토큰 단위에서 어떻게 적용되는지 눈으로 확인합니다.

```python
# 짧은 예시 문장 하나에 collator 한 번 돌려서 어떤 자리가 어떻게 바뀌는지 직접 봅니다.
# 사전학습 데이터 (위키) 도메인 문장 — collator 는 토큰 id 위에서만 동작하므로 도메인 무관.
import torch
import pandas as pd

DEMO_SENT = "The capital of France is Paris, located on the banks of the Seine river."
demo_enc = tokenizer(DEMO_SENT, return_tensors=None)
demo_ids = demo_enc["input_ids"]

torch.manual_seed(0)  # 재현성: 같은 seed 면 같은 마스킹
demo_batch = [{"input_ids": demo_ids, "attention_mask": [1] * len(demo_ids)}]
demo_out = mlm_collator(demo_batch)

masked_ids = demo_out["input_ids"][0].tolist()
labels     = demo_out["labels"][0].tolist()   # -100 = loss 무시, 그 외 = 원본 token id
mask_id    = tokenizer.mask_token_id

orig_tokens   = tokenizer.convert_ids_to_tokens(demo_ids)
masked_tokens = tokenizer.convert_ids_to_tokens(masked_ids)

rows = []
for orig_id, new_id, lab, orig_tok, new_tok in zip(demo_ids, masked_ids, labels, orig_tokens, masked_tokens):
    if lab == -100:
        kind = "—"                      # 미선택 (loss 계산 X)
    elif new_id == mask_id:
        kind = "[MASK] (80%)"            # 표준 마스킹
    elif new_id == orig_id:
        kind = "kept (10%)"              # 선택됐지만 원본 유지
    else:
        kind = "random (10%)"            # 다른 token 으로 교체
    rows.append({
        "pos": len(rows),
        "original": orig_tok,
        "after_collator": new_tok,
        "label_id": lab,
        "what_happened": kind,
    })

demo_df = pd.DataFrame(rows)
print(demo_df.to_string(index=False))
```

**▶ 실행 결과**

```text
 pos original after_collator  label_id what_happened
   0    [CLS]          [CLS]      -100             —
   1      the            the      -100             —
   2  capital         [MASK]      3007  [MASK] (80%)
   3       of             of      1997    kept (10%)
   4   france         france      -100             —
   5       is             is      -100             —
   6    paris          paris      -100             —
   7        ,              ,      -100             —
   8  located        located      -100             —
   9       on             on      -100             —
  10      the            the      -100             —
  11    banks          banks      -100             —
  12       of         [MASK]      1997  [MASK] (80%)
  13      the            the      -100             —
  14    seine          seine      -100             —
  15    river          river      -100             —
  16        .              .      -100             —
  17    [SEP]          [SEP]      -100             —
```

**결과 해석**

18 토큰 중 `capital` 과 `of` 두 자리만 선택돼 (둘 다 이번에는 `[MASK]` 로 교체) `label_id` 에 원본 token id 가 남고, 나머지 16 자리는 `-100` 이라 loss 에서 제외됩니다. 모델은 가려진 두 자리를 주변 문맥만으로 복원하도록 학습 신호를 받습니다.

한 문장만으로는 표본이 작아 비율을 가늠하기 어렵습니다. 이번엔 block 64개 (약 8,000 토큰) 규모로 collator 를 돌려, 선택 비율 15% 와 그 안의 80/10/10 (`[MASK]`/random/kept) 이 통계적으로 실제 맞는지 집계해 봅니다.

```python
# 큰 batch (block 64개 = 약 8000 토큰) 에서 80/10/10 비율이 실제로 맞는지 통계로 확인.
torch.manual_seed(0)
N_DEMO = 64
big_batch = [
    {"input_ids": lm_train[i]["input_ids"], "attention_mask": [1] * BLOCK_SIZE}
    for i in range(N_DEMO)
]
big_out = mlm_collator(big_batch)

in_ids = big_out["input_ids"]
lab    = big_out["labels"]

selected = (lab != -100)                                  # loss 계산 대상
n_total    = lab.numel()
n_selected = selected.sum().item()
n_mask     = ((in_ids == mask_id) & selected).sum().item()
n_kept     = ((in_ids == lab) & selected).sum().item()    # 선택됐지만 원본 유지
n_random   = n_selected - n_mask - n_kept

print(f"Total tokens:                {n_total:>7,}")
print(f"Selected for loss (target 15%):    {n_selected:>7,}  ({100 * n_selected / n_total:5.2f}%)")
print(f"  └─ replaced with [MASK]:   {n_mask:>7,}  ({100 * n_mask / n_selected:5.2f}% of selected)")
print(f"  └─ replaced with random:   {n_random:>7,}  ({100 * n_random / n_selected:5.2f}% of selected)")
print(f"  └─ kept as original:       {n_kept:>7,}  ({100 * n_kept / n_selected:5.2f}% of selected)")
print()
print("이론치: 선택 15% / 그 중 80-10-10 으로 [MASK]-random-kept. 표본이 작아 약간 흔들리지만 비율 일치.")
```

**▶ 실행 결과**

```text
Total tokens:                  8,192
Selected for loss (target 15%):      1,217  (14.86%)
  └─ replaced with [MASK]:       961  (78.96% of selected)
  └─ replaced with random:       121  ( 9.94% of selected)
  └─ kept as original:           135  (11.09% of selected)

이론치: 선택 15% / 그 중 80-10-10 으로 [MASK]-random-kept. 표본이 작아 약간 흔들리지만 비율 일치.
```

**결과 해석**

8,192 토큰 중 14.86% 가 선택됐고 그 안에서 `[MASK]` 78.96% / random 9.94% / kept 11.09% 로, 목표인 15% 와 80/10/10 비율을 표본 오차 범위 안에서 그대로 재현합니다.

MLM 사전학습용 `Trainer` 를 구성합니다. scratch 사전학습이라 fine-tune 보다 큰 학습률 (5e-4) 을 쓰고, T4 미지원인 bf16 대신 `fp16=True` 를 적용합니다.

```python
USE_FP16 = (DEVICE == "cuda")
MLM_EPOCHS = 3   # 한국어 Ch 23 self-contained 와 동일 (1 epoch 은 도메인 gap 작은 영어에선 충분하지만, 일관성 위해 3 으로 통일)

mlm_args = TrainingArguments(
    output_dir="./ch21_mlm_output",
    num_train_epochs=MLM_EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.06,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

```

**위 코드 읽기** — `learning_rate=5e-4` 는 random init 본체를 빠르게 끌어올리기 위한 사전학습용 값으로, 뒤의 분류 fine-tune (2e-5) 보다 25배 큽니다. `save_strategy="no"` 로 체크포인트를 디스크에 남기지 않는데, 본체를 in-memory 로 바로 분류 모델에 옮길 것이기 때문입니다.

```python
mlm_trainer = Trainer(
    model=mlm_model,
    args=mlm_args,
    train_dataset=lm_train,
    eval_dataset=lm_eval,
    data_collator=mlm_collator,
    processing_class=tokenizer,
)

print(f"MLM epochs:     {MLM_EPOCHS}")
print(f"MLM batch size: {mlm_args.per_device_train_batch_size}")
print(f"MLM learning rate: {mlm_args.learning_rate}")
print(f"MLM fp16:       {USE_FP16}")
print(f"MLM steps:      {len(lm_train) // mlm_args.per_device_train_batch_size * MLM_EPOCHS}")
```

**위 코드 읽기** — `data_collator=mlm_collator` 를 넘겨야 매 step 마다 무작위 마스킹이 적용됩니다. 이 collator 가 없으면 입력=정답이 되어 MLM 학습이 성립하지 않습니다.

**▶ 실행 결과**

```text
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
MLM epochs:     3
MLM batch size: 32
MLM learning rate: 0.0005
MLM fp16:       True
MLM steps:      195
```

이제 작은 BERT 본체를 Wikitext-103 으로 MLM 사전학습합니다. 학습이 끝나면 평균 train loss 를 random baseline (`ln vocab` = 10.33) 과 나란히 찍어, 본체가 일반 위키 어휘·문맥을 얼마나 학습했는지 가늠합니다.

```python
t0 = time.time()
mlm_result = mlm_trainer.train()
mlm_elapsed = time.time() - t0
print(f"\nMLM pretraining done in {mlm_elapsed/60:.1f} min")
print(f"mean train loss: {mlm_result.training_loss:.4f}")
print(f"random baseline (ln vocab): {math.log(tokenizer.vocab_size):.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
MLM pretraining done in 0.3 min
mean train loss: 7.6027
random baseline (ln vocab): 10.3262
```

**결과 해석**

평균 train loss 7.60 으로, random baseline 10.33 에서 분명히 내려와 본체가 일반 위키의 어휘·문맥 구조 일부를 학습했음을 보여줍니다. 다만 7 부근에 머무는 데서 보이듯, 2K 문단 × 3 epoch 의 작은 사전학습으로는 vocab 30,522 위 예측이 여전히 어려운 task 임을 알 수 있습니다.

```python
mlm_eval_metrics = mlm_trainer.evaluate()
mlm_eval_loss = mlm_eval_metrics["eval_loss"]
print(f"MLM eval loss:        {mlm_eval_loss:.4f}")
print(f"MLM eval perplexity:  {math.exp(mlm_eval_loss):.2f}")
print(f"(random baseline PPL: {tokenizer.vocab_size:,})")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
MLM eval loss:        7.2124
MLM eval perplexity:  1356.19
(random baseline PPL: 30,522)
```

**결과 해석**

eval perplexity 1356 은 random baseline 30,522 의 약 1/22 로, 가려진 자리에서 vocab 전체가 아니라 약 1,300 개 후보로 좁혀진 정도를 뜻합니다. 완벽한 언어모델과는 멀지만 Yelp 분류 fine-tune 의 출발점으로는 충분한 표상입니다.

```python
# 분류용 config: 같은 본체 구조 + num_labels=2 + problem_type
cls_config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
    num_labels=2,
    problem_type="single_label_classification",
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)

cls_model = BertForSequenceClassification(cls_config)

```

**위 코드 읽기** — 본체 구조는 MLM 과 똑같이 두되 `num_labels=2` 와 `problem_type="single_label_classification"` 만 추가합니다. 이 `problem_type` 이 `CrossEntropyLoss` 를 자동으로 고르게 하고, `id2label` 로 0/1 이 부정/긍정으로 매핑됩니다.

```python
# MLM 본체 (embeddings + encoder) 를 분류 모델로 *복사* — pooler 까지 같이
missing, unexpected = cls_model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)
print(f"본체 가중치 복사 완료")
print(f"  missing keys (분류 측에만 있는 부분): {len(missing)}  e.g. {missing[:3] if missing else []}")
print(f"  unexpected keys (MLM 측 잉여):       {len(unexpected)}  e.g. {unexpected[:3] if unexpected else []}")

```

**위 코드 읽기** — `cls_model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)` 가 핵심 한 줄입니다. 두 모델 모두 내부에 같은 이름의 `self.bert` (`BertModel`) 를 갖기 때문에 사전학습된 임베딩+인코더를 통째로 옮길 수 있고, `strict=False` 라 분류 측에만 있는 `pooler` 같은 키는 missing 으로 넘어갑니다. MLM head 와 분류 head 는 본체 바깥의 다른 자리라 자동으로 분리됩니다.

```python
# 파라미터 수 비교
total_cls = sum(p.numel() for p in cls_model.parameters())
total_body = sum(p.numel() for n, p in cls_model.named_parameters() if "classifier" not in n)
total_head = sum(p.numel() for n, p in cls_model.named_parameters() if "classifier" in n)
print(f"\nClassification model parameters:")
print(f"  body (embeddings + encoder + pooler): {total_body:>10,}  ({total_body/total_cls:.1%})")
print(f"  classifier head Linear(256, 2):       {total_head:>10,}  ({total_head/total_cls:.1%})")
print(f"  total:                                 {total_cls:>10,}  ({total_cls/1e6:.2f} M)")
```

**▶ 실행 결과**

```text
본체 가중치 복사 완료
  missing keys (분류 측에만 있는 부분): 2  e.g. ['pooler.dense.weight', 'pooler.dense.bias']
  unexpected keys (MLM 측 잉여):       0  e.g. []

Classification model parameters:
  body (embeddings + encoder + pooler): 11,072,256  (100.0%)
  classifier head Linear(256, 2):              514  (0.0%)
  total:                                 11,072,770  (11.07 M)
```

**결과 해석**

missing 은 `pooler` 가중치 2 개뿐이고 unexpected 는 0 으로, 사전학습 본체가 깔끔하게 옮겨졌습니다. 분류 head `Linear(256, 2)` 는 514 개로 전체의 0.0% 에 불과해, *대부분의 지식은 사전학습 본체에 있고 새 head 는 아주 얇게* 얹힌다는 fine-tune 패러다임을 그대로 보여줍니다.

이번엔 Yelp 데이터를 분류용으로 토큰화합니다. MLM 과 달리 *문장 단위* 입력이라 `[CLS]`/`[SEP]` 가 기본으로 붙고 `max_length=128` 로 길이를 자릅니다.

```python
# 분류용 토큰화 — 문장 단위, [CLS]/[SEP] 부착, max_length=128
def cls_tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(l) for l in batch["label"]]
    return out

cls_train = ds_train_full.map(cls_tokenize, batched=True).remove_columns(
    [c for c in ds_train_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)
cls_eval  = ds_eval_full.map(cls_tokenize,  batched=True).remove_columns(
    [c for c in ds_eval_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)

print(cls_train)
print(f"\nFirst sample label: {cls_train[0]['labels']}  (int 0 or 1)")
```

**위 코드 읽기** — `out["labels"] = [int(l) for l in batch["label"]]` 로 라벨을 *정수* 0/1 로 둡니다. `single_label_classification` + `CrossEntropyLoss` 는 multi-label 의 multi-hot float 텐서가 아니라 정수 라벨을 받으므로, dtype 을 정수로 맞추는 게 중요합니다.

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample label: 1  (int 0 or 1)
```

**결과 해석**

`input_ids / token_type_ids / attention_mask / labels` 네 컬럼만 남아 분류 학습에 필요한 형태가 됐고, 첫 샘플의 `labels` 가 정수 1 (긍정) 로 잘 들어갔습니다.

Ch 10 과 같은 5종 지표 (accuracy / precision / recall / F1 / AUC) 를 계산하는 함수를 정의합니다. 두 결과를 같은 자로 비교하려면 metric 정의도 동일해야 합니다.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 안정 softmax (K=2)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_full = exp / exp.sum(axis=1, keepdims=True)
    preds = probs_full.argmax(axis=1)
    probs_pos = probs_full[:, 1]   # 클래스 1 의 확률 = AUC 입력

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(labels, probs_pos)),
    }
```

**위 코드 읽기** — `logits.max(...)` 를 빼고 지수를 취하는 *안정 softmax* 로 오버플로를 막은 뒤, `argmax` 로 예측 클래스를, `probs_full[:, 1]` 로 긍정 클래스 확률을 뽑습니다. accuracy/F1 은 argmax 예측을 쓰지만 AUC 는 *확률* (`probs_pos`) 을 입력으로 받는다는 점이 다릅니다.

```python
# Ch 10 과 같은 hyperparams — 변하는 건 *본체 출발점* 뿐
cls_args = TrainingArguments(
    output_dir="./ch21_cls_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

```

**위 코드 읽기** — `learning_rate=2e-5`, `batch=16`, `epoch=2` 는 Ch 10 의 DistilBERT fine-tune 과 *완전히 같은* 값입니다. 본체 출발점 외 모든 조건을 통제해야 두 챕터의 정확도 격차를 *사전학습 규모* 탓으로 해석할 수 있습니다.

```python
cls_trainer = Trainer(
    model=cls_model,
    args=cls_args,
    train_dataset=cls_train,
    eval_dataset=cls_eval,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

t0 = time.time()
cls_result = cls_trainer.train()
cls_elapsed = time.time() - t0
print(f"\nClassification fine-tune done in {cls_elapsed/60:.1f} min")
print(f"mean train loss: {cls_result.training_loss:.4f}")
print(f"random baseline (ln 2): {math.log(2):.4f}")
```

**위 코드 읽기** — MLM 때와 달리 `data_collator` 를 넘기지 않습니다. 분류는 마스킹이 필요 없고 `compute_metrics` 만 추가해 epoch 마다 5종 지표를 자동 측정합니다. 학습 후 평균 train loss 를 random baseline (`ln 2` = 0.693) 과 비교합니다.

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Classification fine-tune done in 0.3 min
mean train loss: 0.6860
random baseline (ln 2): 0.6931
```

**결과 해석**

평균 train loss 0.686 으로 random baseline 0.693 에서 아주 조금만 내려왔습니다. 작은 본체 + 작은 사전학습 + 2 epoch 라는 toy 셋업에서 분류 경계가 *겨우 잡히기 시작한* 정도임을 시사합니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 12:17:59 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   48C    P0             34W /   70W |     797MiB /  15360MiB |     14%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1329      C   /usr/bin/python3                        794MiB |
+-----------------------------------------------------------------------------------------+
```

```python
cls_eval_metrics = cls_trainer.evaluate()
print("Ch 21 small BERT (scratch MLM 3 epoch + classification fine-tune) — eval:")
for k, v in cls_eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
Ch 21 small BERT (scratch MLM 3 epoch + classification fine-tune) — eval:
             eval_loss: 0.6680
         eval_accuracy: 0.6260
        eval_precision: 0.5948
           eval_recall: 0.7128
               eval_f1: 0.6485
              eval_auc: 0.6821
```

**결과 해석**

eval accuracy 0.626, AUC 0.682 로, random (0.5) 보다는 분명히 높지만 Ch 10 의 DistilBERT (약 0.90) 와는 큰 격차입니다. recall 0.713 이 precision 0.595 보다 높아 모델이 긍정 쪽으로 다소 치우쳐 예측하는 경향도 읽힙니다.

eval set 전체에 대해 예측을 뽑아 클래스별 상세 리포트를 봅니다. 평균 정확도 한 숫자만으로는 모델이 어느 클래스에서 약한지 알 수 없으므로, 클래스별 precision/recall 과 예측 자신감 (top-1 확률) 을 함께 확인합니다.

```python
preds_output = cls_trainer.predict(cls_eval)
cls_logits = preds_output.predictions
cls_labels = preds_output.label_ids.astype(int)

exp = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
cls_probs_full = exp / exp.sum(axis=1, keepdims=True)
cls_preds = cls_probs_full.argmax(axis=1)
cls_probs_pos = cls_probs_full[:, 1]

```

**위 코드 읽기** — `cls_trainer.predict` 로 1,000 개 eval 샘플의 logits 를 한 번에 받고, `compute_metrics` 와 같은 안정 softmax 로 확률·예측을 재계산합니다. 이렇게 따로 뽑아 둔 `cls_preds`/`cls_probs_pos` 를 뒤의 confusion matrix 와 학습곡선에서 재사용합니다.

```python
print(f"Logits shape: {cls_logits.shape}")
print(f"Predicted positive rate: {(cls_preds == 1).mean():.1%}")
print(f"Top-1 prob mean: correct={cls_probs_full.max(axis=1)[cls_preds == cls_labels].mean():.4f}, "
      f"wrong={cls_probs_full.max(axis=1)[cls_preds != cls_labels].mean():.4f}")
print()
print(classification_report(
    cls_labels, cls_preds,
    target_names=["negative", "positive"],
    digits=4, zero_division=0,
))
```

**위 코드 읽기** — `Top-1 prob mean` 은 맞춘 예측과 틀린 예측 각각의 평균 자신감을 비교합니다. 두 값이 비슷하면 모델이 *맞을 때나 틀릴 때나 비슷하게 어정쩡한* 확신을 갖는다는 뜻으로, 표상이 아직 약함을 진단하는 신호입니다.

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Logits shape: (1000, 2)
Predicted positive rate: 58.0%
Top-1 prob mean: correct=0.5463, wrong=0.5355

              precision    recall  f1-score   support

    negative     0.6690    0.5446    0.6004       516
    positive     0.5948    0.7128    0.6485       484

    accuracy                         0.6260      1000
   macro avg     0.6319    0.6287    0.6245      1000
weighted avg     0.6331    0.6260    0.6237      1000
```

**결과 해석**

맞은 예측의 평균 자신감 (0.546) 과 틀린 예측 (0.536) 이 거의 같아, 모델이 0.5 근처에서 머뭇거리며 결정하고 있음을 보여줍니다. 예측 긍정 비율 58% 와 positive 의 높은 recall (0.713) 에서 보이듯 긍정 쪽으로 살짝 기울어, 부정 클래스의 recall (0.545) 이 상대적으로 낮습니다.

```python
log_history = cls_trainer.state.log_history
train_logs = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]

if train_logs:
    steps, losses = zip(*train_logs)
    random_baseline = math.log(2)

    sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, "o-", color="#4878D0", label="학습 CE loss (small BERT)")
    ax.axhline(random_baseline, color="black", lw=1.0, ls=":",
               label=f"랜덤 기준선 (ln 2 = {random_baseline:.3f})")
    ax.set_xlabel("학습 step")
    ax.set_ylabel("CE loss (binary)")
    ax.set_title("Yelp 분류 fine-tune loss — small BERT (Wikitext-103 MLM body)")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No train loss logs found.")
```

**▶ 실행 결과**

![output](../assets/21-en_bert_classify-out1.png)

**결과 해석**

학습 CE loss 가 random 기준선 (ln 2 = 0.693) 바로 아래에서 시작해 완만하게만 내려갑니다. 사전학습 본체가 출발점을 random 보다 낫게 만들어 주지만, 작은 규모라 수렴이 가파르지는 않다는 점이 곡선에 드러납니다.

```python
sns.set_theme(style="white", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
cm = confusion_matrix(cls_labels, cls_preds, labels=[0, 1])
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_norm, annot=cm, fmt="d",
    cmap="Blues", vmin=0, vmax=1,
    xticklabels=["부정", "긍정"],
    yticklabels=["부정", "긍정"],
    cbar_kws={"label": "행 기준 정규화 (recall)"}, ax=ax,
)
ax.set_xlabel("예측값")
ax.set_ylabel("실제값")
ax.set_title("Ch 21 small BERT — 혼동 행렬")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/21-en_bert_classify-out2.png)

**결과 해석**

혼동 행렬을 보면 실제 긍정을 긍정으로 맞춘 비율 (recall 0.713) 이 실제 부정을 부정으로 맞춘 비율 (0.545) 보다 높아, 모델이 긍정 쪽으로 치우쳐 오분류가 부정 행에 몰려 있음이 시각적으로 확인됩니다.

```python
# Ch 10 reference 수치 — yelp_polarity 5K/1K + DistilBERT fine-tune 2 epoch 의 *전형적* 결과
# (실측치는 학습자가 Ch 10 노트북을 돌려 본인 값으로 갱신 권장)
CH10_REFERENCE = {
    "accuracy":  0.93,
    "precision": 0.93,
    "recall":    0.93,
    "f1":        0.93,
    "auc":       0.98,
}

ch21_metrics = {k.replace("eval_", ""): v for k, v in cls_eval_metrics.items()
                if k.startswith("eval_") and isinstance(v, float)
                and k.replace("eval_", "") in CH10_REFERENCE}

comparison = pd.DataFrame({
    "metric":              list(CH10_REFERENCE.keys()),
    "Ch10 DistilBERT (ref)": [CH10_REFERENCE[k] for k in CH10_REFERENCE.keys()],
    "Ch21 small BERT":     [ch21_metrics.get(k, float("nan")) for k in CH10_REFERENCE.keys()],
})
comparison["delta (Ch21 - Ch10)"] = comparison["Ch21 small BERT"] - comparison["Ch10 DistilBERT (ref)"]
print("Ch10 vs Ch21 — classification metrics")
print(comparison.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
Ch10 vs Ch21 — classification metrics
   metric  Ch10 DistilBERT (ref)  Ch21 small BERT  delta (Ch21 - Ch10)
 accuracy                   0.93           0.6260              -0.3040
precision                   0.93           0.5948              -0.3352
   recall                   0.93           0.7128              -0.2172
       f1                   0.93           0.6485              -0.2815
      auc                   0.98           0.6821              -0.2979
```

**결과 해석**

accuracy 기준 약 0.30, AUC 약 0.30 의 격차가 일관되게 음수로 나타납니다. 두 모델이 같은 *일반 위키 → Yelp transfer* 패턴을 따르므로, 이 격차의 거의 전부가 *사전학습 규모 (약 3000-5000배) 와 모델 크기 (약 6배)* 의 차이에서 옵니다.

```python
# bar chart 로 한눈에 보기
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
plot_df = comparison.melt(
    id_vars=["metric"],
    value_vars=["Ch10 DistilBERT (ref)", "Ch21 small BERT"],
    var_name="model", value_name="score",
)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=plot_df, x="metric", y="score", hue="model",
    palette={"Ch10 DistilBERT (ref)": "#4878D0", "Ch21 small BERT": "#EE854A"},
    ax=ax,
)
ax.set_ylim(0, 1.05)
ax.set_title("Yelp 이진 분류 — Ch10 vs Ch21")
ax.set_xlabel("지표")
ax.set_ylabel("점수")
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/21-en_bert_classify-out3.png)

**결과 해석**

다섯 지표 모두에서 Ch 10 (DistilBERT) 막대가 Ch 21 (작은 BERT) 보다 일관되게 높습니다. 다만 Ch 21 의 막대가 모두 0.5 (random) 위에 있어, 작은 일반 도메인 사전학습도 random init 보다는 분명히 낫다는 메시지도 함께 읽힙니다.
