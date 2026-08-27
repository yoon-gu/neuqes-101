> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/21_en_bert_classify.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

```python
%pip install -q -U transformers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.7/11.7 MB 103.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 559.1/559.1 kB 32.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 252.4 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 252.4 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 252.4 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.1/50.1 MB 19.4 MB/s eta 0:00:00
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
    set_seed,
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

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Thu Aug 20 12:43:52 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   36C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## Yelp 이진 분류 데이터 로드 — Ch 10 과 같은 split

`fancyzhx/yelp_polarity` 는 *이미 이진화된* (긍정/부정) 5점 척도 yelp 리뷰 데이터셋. Ch 10 의 `Yelp/yelp_review_full` + 별점 이진화 와 *완전히 같은 형태* 의 결과가 나오도록 같은 seed·같은 sample 수를 적용. **5,000 train / 1,000 eval, seed 42**.

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
plain_text/train-00000-of-00001.parquet: downloading bytes:           |  0.00B            
plain_text/test-00000-of-00001.parquet: downloading bytes:           |  0.00B            
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

## 토크나이저 — `bert-base-uncased` (Ch 20 과 동일)

vocab 30,522 의 영어 WordPiece. MLM 사전학습과 분류 fine-tune 전 구간에서 *같은 토크나이저* 를 써야 본체가 학습한 임베딩의 의미가 유지됩니다.

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

## MLM 사전학습 — Ch 20 패턴 압축 재현 (Wikitext-103, 2K × 3 epoch)

이 노트북을 *self-contained* 로 만들기 위해 Ch 20 의 MLM 사전학습을 여기서 압축 재현합니다. Ch 20 (5K × 2 epoch) 보다 *데이터를 줄이고 (2K) epoch 를 늘려 (3)* 시간을 보존 — 한국어 Ch 23 self-contained 와 동일한 hyperparams. 같은 도메인 (위키) 표상을 *얕게라도* 새겨 fine-tune 의 출발점을 만듭니다 — random init 대비 우위가 실제로 얼마나 되는지는 부록의 A/C 비교가 실측으로 보여줍니다 — 순 효과는 실재하지만 수 %p 수준입니다 (`executed/appendix_compute_budget.ipynb` §5 표).

**MLM 사전학습 데이터는 *분류용 Yelp 와 별도*** — `Salesforce/wikitext`, config `wikitext-103-raw-v1` paragraphs 2K 를 *새로 로드*. 본 챕터의 *진짜 transfer 메시지* — *일반 위키 사전학습 → Yelp 분류 transfer* 가 노트북 한 구조에 자연스럽게 들어맞도록 *두 데이터셋이 공존*. 같은 토크나이저 (`bert-base-uncased`) 가 두 도메인을 모두 처리.

같은 작은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024) → `BertForMaskedLM(config)` random init → Wikitext-103 paragraphs 2K MLM 3 epoch.

> **사전학습 코퍼스가 얼마나 작은가** — 2K paragraphs 를 `block_size=128` 로 이어 붙이면 아래 셀에서 *2,100 block* 이 나옵니다. 토큰으로는 `2,100 × 128 = 268,800` — **약 27만 토큰** 입니다. DistilBERT 가 본 약 33억 토큰의 *약 1/12,000*. 이 숫자는 설정에서 결정되므로 재실행해도 같습니다.

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

# 모델 가중치에도 시드를 겁니다 — TrainingArguments(seed=) 는 Trainer 단계부터 적용되므로
# 이 줄이 없으면 random init 이 매 실행 달라져 loss·accuracy 가 실행마다 흔들립니다.
set_seed(SEED)

mlm_model = BertForMaskedLM(mlm_config)  # random init
total = sum(p.numel() for p in mlm_model.parameters())
print(f"Small BERT config: hidden={HIDDEN_SIZE}, layer={NUM_HIDDEN_LAYERS}, head={NUM_ATTENTION_HEADS}")
print(f"Total parameters:  {total:,}  ({total/1e6:.2f} M)")
```

**위 코드 읽기** — `BertForMaskedLM(mlm_config)` 는 사전학습 체크포인트를 받지 않으므로 모든 가중치가 random 입니다. `vocab_size=tokenizer.vocab_size` 로 vocab 을 토크나이저와 묶어 둔 점이 중요한데, 임베딩 행렬 크기가 곧 30,522 × 256 이 되어 전체 파라미터의 큰 비중을 차지합니다. 바로 위 `set_seed(SEED)` 는 이 random init 자체를 고정해, 같은 환경에서는 아래 학습 수치까지 재현되게 합니다.

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
wikitext-103-raw-v1/test-00000-of-00001.(…): downloading bytes:           |  0.00B            
wikitext-103-raw-v1/train-00000-of-00002(…): downloading bytes:           |  0.00B            
wikitext-103-raw-v1/train-00001-of-00002(…): downloading bytes:           |  0.00B            
wikitext-103-raw-v1/validation-00000-of-(…): downloading bytes:           |  0.00B            
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

### [MASK] 가 들어가는 원리 — 한 눈에 보는 80/10/10

`DataCollatorForLanguageModeling` 은 매 step 마다 *입력 토큰의 약 15%* 를 *무작위로* 선택하고, 선택된 위치마다 세 가지 중 하나를 적용합니다.

| 선택된 토큰 운명 | 비율 | 의도 |
| --- | --- | --- |
| `[MASK]` 로 교체 | **80%** | 표준 마스킹 — 모델이 *주변 문맥만으로* 원래 토큰을 맞추도록 |
| **다른 random 토큰** 으로 교체 | 10% | inference 때는 `[MASK]` 가 없으니, 모델이 *항상* 자기 입력을 *의심* 하게 만듦 |
| **원본 그대로** 유지 | 10% | 동일 — 입력과 정답이 일치하는 케이스도 학습 데이터에 포함 |

**나머지 85%** 의 토큰은 `labels = -100` 으로 두어 *loss 계산에서 제외* 됩니다 (PyTorch CE 의 `ignore_index` 기본값). 즉 한 step 의 MLM loss 는 *선택된 15% 자리만* 모아 평균한 값.

> 이 80/10/10 비율은 BERT 논문 (Devlin et al., 2018) 의 원안 그대로. `mlm_probability=0.15` 만 바꾸면 *선택률* 이 바뀌고, 80/10/10 자체는 collator 내부에 고정.

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

**관전 포인트**

- `what_happened` 가 `—` 인 자리(85%) 는 *입력과 정답이 그대로* — loss 에 기여하지 않습니다. 모델은 *문맥을 만들어 주는* 역할만.
- `[MASK]` 자리(약 12%) 가 본 task 의 *진짜 학습 신호*. 주변 토큰들의 attention 결과로 *가려진 자리* 의 vocab 분포를 예측.
- `random` (약 1.5%) 와 `kept` (약 1.5%) 는 *inference 분포 일치* 를 위한 정규화. 추론 시에는 `[MASK]` 가 없으므로 *입력을 절대 신뢰하면 안 된다* 는 신호를 학습에 섞어 줌.
- 매 epoch · 매 batch 마다 마스킹은 *새로 무작위* — 같은 문장이 epoch 마다 다른 자리에서 가려져 학습됩니다 (data augmentation 효과).

### `labels = -100` ignore_index 는 BERT-만의 트릭이 아닙니다 — Phase 4 (GPT) 의 핵심으로 다시

PyTorch `CrossEntropyLoss` 의 `ignore_index=-100` 은 *어느 토큰 자리의 loss 를 학습 신호로 쓸지* 고르는 범용 스위치입니다. 같은 트릭이 Phase 4 GPT 챕터에서 **사전학습 vs Instruction Tuning(SFT) 의 가장 큰 차이** 를 만듭니다.

| 단계 | `labels = ?` | loss 계산 자리 | 학습되는 것 |
|---|---|---|---|
| **MLM 사전학습** (이 챕터·Ch 20) | 선택된 약 15% 만 원본 token id, 나머지 = `-100` | 가려진 자리 | 주변 문맥으로 *가려진 토큰 복원* |
| **GPT CausalLM 사전학습** (Ch 24-26) | `input_ids.clone()` — *거의 모든 토큰* | (pad 만 `-100`) 사실상 *전 자리* | 모든 자리에서 *다음 토큰 예측* — 언어 분포 자체 |
| **SFT / Instruction Tuning** (Ch 28) | **prompt 부분 = `-100`**, *답변 토큰만* 원본 id | *답변 부분만* | "질문을 외우지 말고 답변하는 법" 만 학습 |

> **세 곳 모두 같은 `-100` 트릭, 적용 자리만 정반대.** MLM 은 *대부분을 가리고 일부만 학습*, GPT 사전학습은 *거의 가리지 않음*, SFT 는 *prompt 만 가림*. Phase 4 (특히 Ch 28 SFT, `SFTTrainer` 의 `response-only mask` 옵션) 에서 이 차이를 *코드 라인 한 줄 — `labels[prompt_mask] = -100`* 으로 직접 보게 될 겁니다.

지금 위 셀에서 본 `label_id = -100` 의 의미를 기억해 두면, Ch 28 의 *왜 모델이 instruction 을 따라가게 되는가* 가 한 줄로 이해됩니다.

### 같은 단어 "파인튜닝", BERT 시대와 GPT 시대의 의미가 살짝 다릅니다

이 챕터의 *fine-tune* 은 **BERT 시대 의미** — *사전학습된 본체 + 새 task-specific head (`Linear(H, 2)`)* 를 붙여 *downstream task* 마다 다른 모델로 분기. 본체는 *일반 표상*, head 는 *task 별 특화*. 분류·회귀·NER·QA 각각 다른 head 가 붙고 라벨 포맷도 다릅니다.

GPT 시대 (Phase 4 Ch 24 이후) 부터는 같은 단어가 *살짝 다른 의미* 를 가집니다.

| 축 | **BERT 파인튜닝** (이 챕터, Ch 9-18, Ch 23) | **GPT 파인튜닝 = SFT** (Ch 28) |
|---|---|---|
| 무엇을 바꾸나 | 본체 + **새 head** (task별 부착) | 본체 + **기존 LM head 그대로** |
| 출력 형식 | task별 다름 (class id / score / multi-hot) | *항상 토큰 시퀀스* — 형식 통일 |
| 학습 신호 | task별 loss (CE/BCE/MSE) | *항상 next-token CE*, 단 자리 마스킹만 다름 |
| 학습되는 것 | *task 의 출력 분포* (긍정/부정 결정 경계 등) | *행동 = "이런 입력엔 이런 형식으로 답하라"* |
| 라벨 | 정답 카테고리/값 | *모범 답안 토큰 시퀀스* |

> **BERT 파인튜닝은 *task 적응*, GPT 파인튜닝은 *행동 정렬*.** GPT 는 head 가 바뀌지 않으므로 "파인튜닝" 이 *동일한 next-token 예측 task 안에서 데이터만 바뀌는* 일이 됩니다 (사전학습 = 웹 텍스트, SFT = 모범 응답 쌍). 그래서 Phase 4 부터는 "fine-tuning ≈ SFT ≈ instruction tuning ≈ behavior alignment" 가 거의 동의어로 섞여 쓰입니다.

이 의미 차이는 *왜 GPT 모델 하나가 모든 task 를 해내는가* 의 핵심 이유 — head 가 task 별로 분기하지 않으니 *입력 프롬프트* 만 바꾸면 *같은 모델* 이 다른 일을 합니다. Ch 28 에서 직접 확인.

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
    warmup_steps=0.06,             # 1 미만이면 전체 step 대비 *비율* 로 해석 (구 warmup_ratio)
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
Epoch  Training Loss  Validation Loss
1      7.427901       7.483476
2      7.335162       7.241530
3      7.174193       7.277164
MLM pretraining done in 0.3 min
mean train loss: 7.6013
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
Training Loss  Validation Loss  Epoch
7.174193       7.221408         3
MLM eval loss:        7.2214
MLM eval perplexity:  1368.41
(random baseline PPL: 30,522)
```

**결과 해석**

eval perplexity 1368 은 random baseline 30,522 의 약 1/22 로, 가려진 자리에서 vocab 전체가 아니라 1,300여 개 후보로 좁혀진 정도를 뜻합니다. 완벽한 언어모델과는 멀지만 Yelp 분류 fine-tune 의 출발점으로는 충분한 표상입니다.

**관전 포인트** — Wikitext-103 paragraphs 에서 MLM loss 가 *random baseline 10.33* 에서 시작해 약 7 부근까지 떨어졌다면 본체가 *일반 위키 어휘·문맥 구조의 일부* 를 학습한 상태. perplexity 로 환산하면 vocab 30,522 중 *약 1,300 개 후보* 로 좁혀진 정도. Ch 20 의 2 epoch 와 비슷한 수준입니다. 다만 Ch 20 이 같은 Wikitext-103 에서 잰 *unigram (빈도만, 문맥 없음) 기준선* 이 **7.2525** 였으니, 이 정도 loss 는 *빈도 통계를 갓 넘어선* 단계로 읽는 편이 정확합니다 — 본체가 *일반 영어의 얕은 구조* 를 가진 채 Yelp 분류 fine-tune 에 들어갑니다.

> **체크포인트 저장은 생략** — 노트북 안에서 바로 본체 가중치를 분류 모델로 옮기기 때문. Ch 20 처럼 디스크에 저장하려면 `mlm_model.save_pretrained("./ch21_mlm_ckpt")` 한 줄.

## 헤드 교체 — MLM → 분류 + Fine-tune

이제 *방금 학습된 작은 BERT 본체* 를 분류 모델로 옮깁니다. 두 가지 흐름:

1. `BertForMaskedLM.bert` (embedding + encoder) 를 그대로 가져옴
2. 새 `BertForSequenceClassification(config)` 을 만들고, 1 의 본체를 *복사*. 분류 헤드는 새로 random init

이렇게 만든 모델을 같은 Yelp 데이터의 *라벨* 까지 사용해 분류 fine-tune. Ch 10 의 hyperparams 와 *완전히 같이* (`lr=2e-5, batch=16, epoch=2, fp16=True`) 둬서 *본체 출발점* 외 모든 조건을 통제.

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

# 분류 헤드도 새로 random init 되므로 같은 이유로 시드를 겁니다.
set_seed(SEED)

cls_model = BertForSequenceClassification(cls_config)

# MLM 본체 (embeddings + encoder) 를 분류 모델로 *복사* — pooler 까지 같이
missing, unexpected = cls_model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)
print(f"본체 가중치 복사 완료")
print(f"  missing keys (분류 측에만 있는 부분): {len(missing)}  e.g. {missing[:3] if missing else []}")
print(f"  unexpected keys (MLM 측 잉여):       {len(unexpected)}  e.g. {unexpected[:3] if unexpected else []}")

# 파라미터 수 비교
total_cls = sum(p.numel() for p in cls_model.parameters())
total_body = sum(p.numel() for n, p in cls_model.named_parameters() if "classifier" not in n)
total_head = sum(p.numel() for n, p in cls_model.named_parameters() if "classifier" in n)
print(f"\nClassification model parameters:")
print(f"  body (embeddings + encoder + pooler): {total_body:>10,}  ({total_body/total_cls:.1%})")
print(f"  classifier head Linear(256, 2):       {total_head:>10,}  ({total_head/total_cls:.1%})")
print(f"  total:                                 {total_cls:>10,}  ({total_cls/1e6:.2f} M)")
```

**위 코드 읽기** — 본체 구조는 MLM 과 똑같이 두되 `num_labels=2` 와 `problem_type="single_label_classification"` 만 추가합니다. 이 `problem_type` 이 `CrossEntropyLoss` 를 자동으로 고르게 하고, `id2label` 로 0/1 이 부정/긍정으로 매핑됩니다.

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

missing 은 `pooler` 가중치 2 개뿐이고 unexpected 는 0 으로, 사전학습 본체가 깔끔하게 옮겨졌습니다. 분류 head `Linear(256, 2)` 는 514 개로 전체의 0.0% 에 불과해, *대부분의 지식은 본체에 있고 head 는 얇은 어댑터* 라는 구조가 수치로 확인됩니다.

**`bert.load_state_dict` 가 한 일** — `BertForMaskedLM` 과 `BertForSequenceClassification` 둘 다 *내부에 같은 `BertModel`* (이름 `self.bert`) 을 갖습니다. 그 본체만 통째로 옮긴 셈. MLM head (`cls.predictions`) 와 분류 head (`classifier`) 는 *모델 객체의 다른 자리* 라 자동으로 분리됩니다.

> Ch 7-18 의 `AutoModelForSequenceClassification.from_pretrained(...)` 가 디스크에서 같은 일을 합니다. 우리는 *방금 학습한 본체* 를 디스크 없이 in-memory 로 옮긴 셈.

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
Epoch  Training Loss  Validation Loss  Accuracy  Precision  Recall    F1        Auc
1      0.689457       0.687375         0.495000  0.489318   0.993802  0.655760  0.666635
2      0.667249       0.661020         0.631000  0.605505   0.681818  0.641399  0.680821
Classification fine-tune done in 0.2 min
mean train loss: 0.6835
random baseline (ln 2): 0.6931
```

**결과 해석**

평균 train loss 0.6835 로 random baseline 0.6931 에서 아주 조금만 내려왔습니다. 작은 본체 + 작은 사전학습 + 2 epoch 라는 toy 셋업에서 분류 경계가 *겨우 잡히기 시작한* 정도임을 시사합니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Thu Aug 20 12:45:10 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   50C    P0             58W /   70W |     797MiB /  15360MiB |     31%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A             573      C   /usr/bin/python3                        794MiB |
+-----------------------------------------------------------------------------------------+
```
