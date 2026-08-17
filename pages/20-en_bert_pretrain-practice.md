> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/20_en_bert_pretrain/20_en_bert_pretrain.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

설치가 끝나면 **라이브러리 버전을 먼저 찍습니다.** 같은 `seed` 로 돌려도 `transformers`·`datasets` 가 올라가면 수치가 미세하게 달라질 수 있어서, 아래 출력이 이 챕터에 실린 값과 다르다면 *버전 차이* 를 먼저 의심하면 됩니다.

```python
%pip install -q -U transformers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 11.7/11.7 MB 264.4 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.7/11.7 MB 125.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 559.1/559.1 kB 47.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━━ 35.1/50.1 MB 246.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 250.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 250.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.1/50.1 MB 19.6 MB/s eta 0:00:00
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

import accelerate
import datasets
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
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

# 실행 환경 기록 — 재현이 안 될 때 가장 먼저 확인할 정보입니다.
print(f"PyTorch:        {torch.__version__}")
print(f"transformers:   {transformers.__version__}")
print(f"datasets:       {datasets.__version__}")
print(f"accelerate:     {accelerate.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device:         {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
elif DEVICE == "cpu":
    print("Warning: CPU runtime — MLM training will be very slow. Switch to T4 recommended.")
```

**▶ 실행 결과**

```text
PyTorch:        2.11.0+cu128
transformers:   5.15.0
datasets:       5.0.1
accelerate:     1.14.0
CUDA available: True
Device:         cuda
GPU:             Tesla T4
```

**결과 해석**

이 챕터에 실린 수치는 모두 위 조합(`transformers` 5.15.0 · `datasets` 5.0.1 · `accelerate` 1.14.0 · PyTorch 2.11.0, Tesla T4)에서 나온 값입니다. `Device: cuda` 가 아니면 학습이 몇 시간 단위로 늘어나니 런타임 유형을 먼저 T4 로 바꾸세요.

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Aug 17 09:01:01 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   53C    P8             14W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 토크나이저 — `bert-base-uncased` 그대로 로드

vocab 30,522 의 영어 WordPiece. *모델은 random init* 이지만 토크나이저는 *완성품* 을 가져옵니다.

```python
TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
```

**위 코드 읽기** `AutoTokenizer.from_pretrained("bert-base-uncased")` 가 내려받는 것은 vocab 과 WordPiece 분할 규칙뿐입니다 — 같은 이름의 *모델 weight* 는 전혀 건드리지 않으므로, 이번 챕터의 본체는 그대로 random init 입니다.

```python
print(f"tokenizer:        {TOKENIZER_NAME}")
print(f"vocab_size:       {tokenizer.vocab_size:,}")
print(f"model_max_length: {tokenizer.model_max_length}")
print(f"special tokens:")
for name in ("pad_token", "unk_token", "cls_token", "sep_token", "mask_token"):
    tok = getattr(tokenizer, name)
    tid = tokenizer.convert_tokens_to_ids(tok) if tok is not None else None
    print(f"  {name:>11}: {tok!r:>10}  (id={tid})")
```

**위 코드 읽기** 특수 토큰 중 `mask_token` 이 이번 챕터의 주인공입니다. `[MASK]` 가 vocab 안에 이미 고유 id 로 자리 잡고 있어야, 뒤에서 collator 가 그 id 로 토큰을 갈아끼울 수 있습니다.

```python
# 간단 시연 — 일반 위키풍 문장
SAMPLE = "The capital of France is Paris, located on the Seine river."
enc = tokenizer(SAMPLE, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(f"\nsample: {SAMPLE!r}")
print(f"tokens ({len(tokens)}): {tokens}")
print(f"ids:    {enc['input_ids'][0].tolist()}")
```

**위 코드 읽기** `tokenizer(SAMPLE)` 는 기본값이 `add_special_tokens=True` 라 앞뒤에 `[CLS]`·`[SEP]` 가 자동으로 붙습니다. 학습 데이터를 만들 때는 이 기본값을 일부러 꺼서 블록 단위로 이어 붙이게 되니, 두 호출의 차이를 기억해 두세요.

**▶ 실행 결과**

```text
tokenizer:        bert-base-uncased
vocab_size:       30,522
model_max_length: 512
special tokens:
    pad_token:    '[PAD]'  (id=0)
    unk_token:    '[UNK]'  (id=100)
    cls_token:    '[CLS]'  (id=101)
    sep_token:    '[SEP]'  (id=102)
   mask_token:   '[MASK]'  (id=103)

sample: 'The capital of France is Paris, located on the Seine river.'
tokens (15): ['[CLS]', 'the', 'capital', 'of', 'france', 'is', 'paris', ',', 'located', 'on', 'the', 'seine', 'river', '.', '[SEP]']
ids:    [101, 1996, 3007, 1997, 2605, 2003, 3000, 1010, 2284, 2006, 1996, 16470, 2314, 1012, 102]
```

**결과 해석**

`seine` 처럼 드문 고유명사까지 `##` 조각 없이 온전한 한 토큰으로 잡히는 것이, 30,522 vocab 이 위키 도메인을 얼마나 넉넉히 덮는지 보여 줍니다. Ch 19 에서 Yelp 5K 로 직접 학습한 8K vocab 이었다면 같은 문장이 훨씬 잘게 쪼개졌을 자리입니다.

## 데이터 — Wikitext-103 paragraphs (일반 도메인 사전학습 코퍼스)

원본 BERT 가 영어 Wikipedia + BookCorpus 라는 *일반 도메인* 코퍼스로 사전학습한 정신을 따라, 본 챕터도 **Wikitext-103** 본문으로 MLM 사전학습합니다 — *task 도메인 (Yelp 리뷰(식당·업체)) 으로 사전학습하면 domain-adaptive pretraining 에 가까워져 사전학습의 진짜 메시지 (일반 표상 학습 → 다른 task 로 transfer) 가 흐려지기 때문*.

**원본**: `Salesforce/wikitext`, config `wikitext-103-raw-v1` (CC-BY-SA, 정제된 영문 위키 paragraphs). HF Hub 정제본 — line 단위로 이미 정리되어 있어 빈 줄 / 너무 짧은 줄 / 너무 긴 줄만 제외하면 바로 사용 가능. Ch 21 의 분류 fine-tune (Yelp 이진) 은 *완전히 다른 도메인* — 사전학습 → fine-tune transfer 메시지가 정직해집니다. Ch 22 의 한국어 (한국어 Wikipedia paragraphs) 와 *대칭* 패턴.

Wikitext-103 전체를 그대로 쓰면 T4 30분 룰을 훌쩍 넘기므로, 원본 180만 줄에서 학습 5,000 / 평가 500 문단만 추려 씁니다. 라벨이 없는 순수 텍스트라는 점이 이전 분류 챕터들과 결정적으로 다른 지점입니다.

```python
SEED = 42
N_TRAIN_TEXT = 5000
N_EVAL_TEXT  = 500

print("downloading Wikitext-103 (Salesforce/wikitext, wikitext-103-raw-v1)...")
raw_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
raw_eval  = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
print(f"  raw train lines: {len(raw_train):,}")
print(f"  raw eval  lines: {len(raw_eval):,}")
```

**위 코드 읽기** `load_dataset(..., split="train")` 이 돌려주는 것은 문단이 아니라 *줄* 입니다. Wikitext 원본은 문서를 줄 단위로 흘려 놓은 형식이라, 제목 줄·빈 줄이 그대로 섞여 있어 곧바로 학습에 쓸 수 없습니다.

```python
# 빈 줄 / 너무 짧은 줄 (제목·메타) / 너무 긴 줄 (목록·인용) 제외
def is_good(ex, min_len=50, max_len=2000):
    t = ex["text"].strip()
    return min_len <= len(t) <= max_len

train_filtered = raw_train.filter(is_good).shuffle(seed=SEED).select(range(N_TRAIN_TEXT))
eval_filtered  = raw_eval.filter(is_good).shuffle(seed=SEED).select(range(N_EVAL_TEXT))
```

**위 코드 읽기** `is_good` 의 `min_len=50` 이 제목·메타 줄을, `max_len=2000` 이 목록·인용 덩어리를 걸러 내 *본문 문단만* 남깁니다. `shuffle(seed=SEED).select(...)` 는 앞쪽 문서에 쏠리지 않게 섞은 뒤 앞에서 자르는 표준 패턴으로, seed 를 고정했으므로 다시 돌려도 같은 표본이 뽑힙니다.

```python
# text 컬럼만 유지
ds_train_raw = train_filtered.remove_columns([c for c in train_filtered.column_names if c != "text"])
ds_eval_raw  = eval_filtered.remove_columns([c for c in eval_filtered.column_names if c != "text"])

print(f"\nsampled train: {len(ds_train_raw):,} paragraphs")
print(f"sampled eval:  {len(ds_eval_raw):,} paragraphs")
print()
print(f"sample text length stats (chars):")
lens = [len(t) for t in ds_train_raw["text"]]
print(f"  mean: {np.mean(lens):.1f}, median: {np.median(lens):.0f}, max: {max(lens)}")
print()
print(f"first sample previews:")
for i in range(3):
    t = ds_train_raw[i]["text"]
    print(f"  Sample {i}: {t[:120]}")
```

**▶ 실행 결과**

```text
downloading Wikitext-103 (Salesforce/wikitext, wikitext-103-raw-v1)...
wikitext-103-raw-v1/test-00000-of-00001.(…): downloading bytes:           |  0.00B            
wikitext-103-raw-v1/train-00000-of-00002(…): downloading bytes:           |  0.00B            
wikitext-103-raw-v1/train-00001-of-00002(…): downloading bytes:           |  0.00B            
wikitext-103-raw-v1/validation-00000-of-(…): downloading bytes:           |  0.00B            
  raw train lines: 1,801,350
  raw eval  lines: 3,760
sampled train: 5,000 paragraphs
sampled eval:  500 paragraphs

sample text length stats (chars):
  mean: 650.0, median: 614, max: 2002

first sample previews:
  Sample 0:  Balinor Buckhannah , the Crown Prince of the country of Callahorn and the " charismatic commander of [ the ] Border Leg
  Sample 1:  Bellomont was Member of Parliament for Droitwich from 1688 to 1695 . In the 1690s he became involved in the attempts by
  Sample 2:  Paulet was promoted to full lieutenant in 1791 and appointed to HMS Vulcan , though he was moved to HMS Assistance in A
```

**결과 해석**

원본 180만 줄에서 걸러 뽑은 5,000 문단의 평균 길이가 650자로, 제목 줄 같은 토막이 아니라 제대로 된 본문만 남았음을 확인할 수 있습니다. 미리보기 세 개가 모두 인물·역사 서술인 것도 위키 일반 도메인이라는 이번 사전학습 코퍼스의 성격 그대로입니다.

## 토큰화 + `group_texts` — HF `run_mlm.py` 표준 패턴

MLM 사전학습의 표준 입력 포맷은 *고정 길이 블록*. 변동 길이 문장에 그대로 padding 하면 *손실*: (a) 짧은 문장이 많으면 PAD 비율이 높아 GPU 시간 낭비, (b) 긴 문장은 truncation 으로 정보 손실.

**해결책**: 모든 문서를 *이어 붙여 토큰 스트림* 으로 만든 뒤, `block_size=128` 단위로 자름. 문장 경계가 사라지는 trade-off 가 있지만, BERT 사전학습은 *임의 위치의 토큰 예측* 이라 문장 경계가 중요하지 않음.

Wikitext-103 paragraphs 는 *제한 50-2000자 필터링* 으로 평균 문장 길이가 일정 (수백 자 위주). 5,000 paragraphs 가 `block_size=128` 로 잘리면 약 5,352 블록 (약 68만 토큰) 으로 정리됩니다. 코드는 HF `examples/pytorch/language-modeling/run_mlm.py` 의 `group_texts` 함수를 그대로 따른 표준 패턴.

```python
BLOCK_SIZE = 128

def tokenize_function(examples):
    # 특수 토큰 부착 안 함 — 블록 단위로 자를 거라 [CLS]/[SEP] 가 의미 없음
    return tokenizer(examples["text"], add_special_tokens=False, truncation=False)
```

**위 코드 읽기** 분류 챕터들과 정반대로 `add_special_tokens=False`, `truncation=False` 를 줍니다. 문단 경계를 지우고 하나의 토큰 스트림으로 이어 붙일 예정이라 `[CLS]`/`[SEP]` 가 중간에 끼면 방해가 되고, 자르는 일은 다음 셀의 `group_texts` 가 블록 단위로 맡습니다.

```python
tokenized_train = ds_train_raw.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
tokenized_eval = ds_eval_raw.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
print(f"tokenized_train: {tokenized_train}")
print(f"first 30 input_ids of sample 0: {tokenized_train[0]['input_ids'][:30]}")
```

**위 코드 읽기** `remove_columns=["text"]` 로 원본 문자열을 버려 이후 단계가 토큰 id 만 다루게 합니다. 이 시점의 `num_rows` 는 아직 *문단 수* 이며, 길이도 문단마다 제각각입니다.

**▶ 실행 결과**

```text
tokenized_train: Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask'],
    num_rows: 5000
})
first 30 input_ids of sample 0: [20222, 12131, 10131, 4819, 15272, 1010, 1996, 4410, 3159, 1997, 1996, 2406, 1997, 2655, 4430, 9691, 1998, 1 …(뒤 77자 생략)
```

**결과 해석**

`num_rows: 5000` 이 그대로이고 첫 id 가 `101`(`[CLS]`)이 아닌 것에서, 특수 토큰 없이 문단 단위 그대로 토큰화됐음을 확인할 수 있습니다. 고정 길이 블록으로 바뀌는 것은 다음 셀부터입니다.

```python
def group_texts(examples):
    '''HF 표준 group_texts — 모든 토큰 스트림을 이어 붙인 뒤 block_size 로 자름.'''
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    # block_size 배수로 잘라내기 (마지막 토막은 버림)
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    # labels = input_ids 사본 (collator 가 mask 위치만 골라냄)
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result
```

**위 코드 읽기** `sum(examples[k], [])` 가 batch 안의 모든 문단을 하나의 긴 리스트로 이어 붙이고, `(total_length // BLOCK_SIZE) * BLOCK_SIZE` 로 자투리를 버려 길이가 정확히 128 인 블록만 남깁니다. 마지막 줄의 `labels = input_ids` 사본이 MLM 의 관건으로, 여기서는 아직 전부 정답이고 *어느 자리를 가릴지* 는 collator 가 학습 중에 매 batch 새로 고릅니다.

```python
lm_train = tokenized_train.map(group_texts, batched=True, batch_size=1000)
lm_eval  = tokenized_eval.map(group_texts, batched=True, batch_size=1000)

print(f"lm_train: {lm_train}")
print(f"lm_eval:  {lm_eval}")
print(f"\nblock_size:           {BLOCK_SIZE}")
print(f"train blocks: {len(lm_train):,}  (approx. {len(lm_train) * BLOCK_SIZE:,} tokens)")
print(f"eval blocks:  {len(lm_eval):,}   (approx. {len(lm_eval) * BLOCK_SIZE:,} tokens)")
print(f"\nsample block 0 first 20 ids: {lm_train[0]['input_ids'][:20]}")
print(f"sample block 0 first 20 tok: {tokenizer.convert_ids_to_tokens(lm_train[0]['input_ids'][:20])}")
```

**▶ 실행 결과**

```text
lm_train: Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5352
})
lm_eval:  Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 535
})

block_size:           128
train blocks: 5,352  (approx. 685,056 tokens)
eval blocks:  535   (approx. 68,480 tokens)

sample block 0 first 20 ids: [20222, 12131, 10131, 4819, 15272, 1010, 1996, 4410, 3159, 1997, 1996, 2406, 1997, 2655, 4430, 9691, 1998, 1996, 1000, 23916]
sample block 0 first 20 tok: ['bali', '##nor', 'buck', '##han', '##nah', ',', 'the', 'crown', 'prince', 'of', 'the', 'country', 'of', 'call' …(뒤 52자 생략)
```

**결과 해석**

문단 5,000 개가 길이 128 짜리 블록 5,352 개(약 68.5만 토큰)로 재편됐습니다 — 행 수가 크게 늘지 않은 것은 문단 하나가 평균적으로 블록 하나 남짓이라는 뜻입니다. `features` 에 `labels` 가 새로 생긴 것도 확인할 수 있습니다.

## 작은 `BertConfig` + `BertForMaskedLM` — random init

표준 `bert-base-uncased` 는 hidden=768, layer=12, head=12, intermediate=3072 = **110M params** — T4 에서 scratch 학습은 *수일* 필요.

이번 챕터는 *입문용 작은 BERT* 로 축소:

| hyperparam | 표준 `bert-base-uncased` | 이번 챕터 (작은 BERT) |
|---|---|---|
| `hidden_size` | 768 | **256** |
| `num_hidden_layers` | 12 | **4** |
| `num_attention_heads` | 12 | **4** |
| `intermediate_size` | 3072 | **1024** |
| `max_position_embeddings` | 512 | **128** (BLOCK_SIZE 와 같음) |
| 총 파라미터 | 약 110M | **약 11M** (toy 규모) |

크기는 1/10 이지만 *MLM 학습이 진행되는지* 보기에는 충분. Ch 21 에서 분류 fine-tune 할 때 성능 비교가 진짜 결과.

```python
HIDDEN_SIZE         = 256
NUM_HIDDEN_LAYERS   = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE   = 1024
MAX_POS_EMBED       = 128  # = BLOCK_SIZE

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
)
```

**위 코드 읽기** `vocab_size=tokenizer.vocab_size` 로 토크나이저와 모델의 어휘 크기를 강제로 맞춥니다 — 이 둘이 어긋나면 임베딩 테이블 밖의 id 가 들어와 곧바로 에러가 납니다. `max_position_embeddings=128` 은 `BLOCK_SIZE` 와 같은 값으로, 어차피 128 길이 블록만 넣을 것이므로 필요 없는 위치 임베딩을 만들지 않습니다.

```python
# 모델 가중치에도 시드를 겁니다 — TrainingArguments(seed=) 는 Trainer 단계부터 적용되므로
# 이 줄이 없으면 random init 이 매 실행 달라져 loss·perplexity 가 실행마다 흔들립니다.
set_seed(SEED)

model = BertForMaskedLM(config)  # random init — pretrained weight 없음!
```

**위 코드 읽기** 이 챕터의 핵심 한 줄입니다. `from_pretrained` 가 아니라 `BertForMaskedLM(config)` 라는 **생성자 호출** 이므로 가중치가 전부 난수이며, 바로 앞의 `set_seed(SEED)` 가 그 난수를 고정해 실행마다 같은 loss·perplexity 가 나오게 합니다.

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n)
encoder = sum(p.numel() for n, p in model.named_parameters() if "encoder" in n)
head = sum(p.numel() for n, p in model.named_parameters() if "cls" in n)

print(f"Config: hidden={HIDDEN_SIZE}, layer={NUM_HIDDEN_LAYERS}, "
      f"head={NUM_ATTENTION_HEADS}, intermediate={INTERMEDIATE_SIZE}")
print(f"max_position_embeddings: {MAX_POS_EMBED}")
print()
print(f"Total parameters:    {total:>13,}  ({total/1e6:.2f} M)")
print(f"Trainable:           {trainable:>13,}")
print(f"  embeddings:        {emb:>13,}  ({emb/total:.1%})  ← vocab 30522 x hidden 256")
print(f"  encoder (4 layer): {encoder:>13,}  ({encoder/total:.1%})")
print(f"  MLM head:          {head:>13,}  ({head/total:.1%})  ← tied with embeddings")
```

**▶ 실행 결과**

```text
Config: hidden=256, layer=4, head=4, intermediate=1024
max_position_embeddings: 128

Total parameters:       11,103,290  (11.10 M)
Trainable:              11,103,290
  embeddings:            7,847,424  (70.7%)  ← vocab 30522 x hidden 256
  encoder (4 layer):     3,159,040  (28.5%)
  MLM head:                 96,826  (0.9%)  ← tied with embeddings
```

**관찰** — 작은 BERT 의 파라미터는 *임베딩 테이블이 70% 넘게* 차지합니다 (vocab 30522 × hidden 256 ≈ 7.8M). encoder body (4개 층) 자체는 약 3.2M 뿐입니다. 이게 *vocab 큰데 모델이 작은* 셋업의 특징 — 표준 BERT (vocab 30K × hidden 768 ≈ 23M / 110M = 21%) 와 비율이 매우 다릅니다.

> MLM head 의 weight 는 입력 임베딩과 *tied* (공유) — `BertForMaskedLM` 기본 동작. vocab 차원 출력 layer 가 임베딩 테이블과 같아 파라미터 절약.

## `DataCollatorForLanguageModeling` + Trainer 학습

collator 가 매 batch 마다 *무작위로 15% 토큰을 [MASK]* 로 바꾸고, 그 위치의 정답 토큰을 `labels` 로 표시 (나머지 위치는 -100 → CrossEntropyLoss 무시).

**MLM masking 규칙** (BERT 원논문):
- 선택된 15% 중 80%: 실제로 `[MASK]` 로 교체
- 10%: 무작위 다른 토큰으로 교체
- 10%: 원래 토큰 유지

이 비율은 *모델이 [MASK] 토큰 자체에 과도하게 의존하지 않게* 하는 트릭. `DataCollatorForLanguageModeling` 이 자동 처리.

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)
```

**위 코드 읽기** `mlm=True` 하나로 이 collator 는 BERT 식 마스킹 collator 가 됩니다 (`mlm=False` 면 Phase 4 GPT 챕터들이 쓰는 causal LM 용으로 바뀝니다). 마스킹이 데이터 전처리가 아니라 *collator* 에 있다는 점이 중요한데, 덕분에 같은 블록이 epoch 마다 다른 자리에서 가려집니다.

### [MASK] 가 들어가는 원리 — 한 눈에 보는 80/10/10

`DataCollatorForLanguageModeling` 은 매 step 마다 *입력 토큰의 약 15%* 를 *무작위로* 선택하고, 선택된 위치마다 세 가지 중 하나를 적용합니다.

| 선택된 토큰 운명 | 비율 | 의도 |
| --- | --- | --- |
| `[MASK]` 로 교체 | **80%** | 표준 마스킹 — 모델이 *주변 문맥만으로* 원래 토큰을 맞추도록 |
| **다른 random 토큰** 으로 교체 | 10% | inference 때는 `[MASK]` 가 없으니, 모델이 *항상* 자기 입력을 *의심* 하게 만듦 |
| **원본 그대로** 유지 | 10% | 동일 — 입력과 정답이 일치하는 케이스도 학습 데이터에 포함 |

**나머지 85%** 의 토큰은 `labels = -100` 으로 두어 *loss 계산에서 제외* 됩니다 (PyTorch CE 의 `ignore_index` 기본값). 즉 한 step 의 MLM loss 는 *선택된 15% 자리만* 모아 평균한 값.

> 이 `labels = -100` 트릭은 BERT-만의 것이 아닙니다 — Phase 4 GPT 사전학습은 *거의 모든 토큰* 을 학습 (`labels = input_ids`), SFT (Ch 28) 는 *prompt 만 -100, 답변만 학습*. 같은 트릭, 정반대 자리. Ch 21 에서 더 자세히.

위 표를 글로만 읽으면 감이 잘 안 오므로, 짧은 문장 하나를 collator 에 직접 통과시켜 *어느 자리가 무엇으로 바뀌었는지* 를 토큰 단위 표로 펼쳐 봅니다. 입력이 어떻게 변형됐는지와 `labels` 가 어디에만 살아남았는지를 나란히 놓고 보는 것이 요점입니다.

```python
# 짧은 예시 문장 하나에 collator 한 번 돌려서 어떤 자리가 어떻게 바뀌는지 직접 봅니다.
import pandas as pd

DEMO_SENT = "Pretraining a language model on Wikipedia teaches it general English structure."
demo_enc = tokenizer(DEMO_SENT, return_tensors=None)
demo_ids = demo_enc["input_ids"]

torch.manual_seed(0)  # 재현성: 같은 seed 면 같은 마스킹
demo_batch = [{"input_ids": demo_ids, "attention_mask": [1] * len(demo_ids)}]
demo_out = data_collator(demo_batch)
```

**위 코드 읽기** collator 를 Trainer 에 맡기지 않고 `data_collator(demo_batch)` 로 직접 한 번 호출한 것이 이 셀의 요령입니다. 마스킹은 매번 무작위이므로 `torch.manual_seed(0)` 을 걸어 아래 표가 항상 같은 자리를 가리키게 했습니다.

```python
masked_ids = demo_out["input_ids"][0].tolist()
labels     = demo_out["labels"][0].tolist()
mask_id    = tokenizer.mask_token_id

orig_tokens   = tokenizer.convert_ids_to_tokens(demo_ids)
masked_tokens = tokenizer.convert_ids_to_tokens(masked_ids)
```

**위 코드 읽기** collator 가 돌려준 것은 *변형된 입력* (`input_ids`) 과 *정답* (`labels`) 두 갈래입니다. 원본 `demo_ids` 를 따로 들고 있어야 셋을 나란히 비교할 수 있으므로 여기서 원본·변형본을 각각 토큰 문자열로 되돌립니다.

```python
rows = []
for orig_id, new_id, lab, orig_tok, new_tok in zip(demo_ids, masked_ids, labels, orig_tokens, masked_tokens):
    if lab == -100:
        kind = "—"
    elif new_id == mask_id:
        kind = "[MASK] (80%)"
    elif new_id == orig_id:
        kind = "kept (10%)"
    else:
        kind = "random (10%)"
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

**위 코드 읽기** 분류 기준이 곧 80/10/10 규칙 자체입니다 — `lab == -100` 이면 애초에 선택되지 않은 자리, 선택된 자리 중 `new_id == mask_id` 면 `[MASK]` 교체, `new_id == orig_id` 면 원본 유지, 나머지가 random 교체입니다.

**▶ 실행 결과**

```text
 pos  original after_collator  label_id what_happened
   0     [CLS]          [CLS]      -100             —
   1       pre            pre      -100             —
   2   ##train         [MASK]     23654  [MASK] (80%)
   3     ##ing         [MASK]      2075  [MASK] (80%)
   4         a              a      -100             —
   5  language       language      -100             —
   6     model          model      -100             —
   7        on             on      -100             —
   8 wikipedia      wikipedia      -100             —
   9   teaches        teaches      -100             —
  10        it             it      -100             —
  11   general        general      -100             —
  12   english         [MASK]      2394  [MASK] (80%)
  13 structure      structure      -100             —
  14         .              .      -100             —
  15     [SEP]          [SEP]      -100             —
```

**결과 해석**

16 토큰 중 3 자리만 `[MASK]` 로 바뀌었고 나머지는 전부 `label_id = -100` 이라 loss 에서 빠집니다. `pre`/`##train`/`##ing` 처럼 한 단어가 쪼개진 조각들이 개별적으로 선택되는 것도 볼 수 있는데, 마스킹 단위가 단어가 아니라 **WordPiece 토큰** 이기 때문입니다.

문장 하나로는 표본이 너무 작아 80/10/10 이 정말 맞는지 알 수 없습니다. 블록 64 개(8,192 토큰)를 한꺼번에 통과시켜 실제 비율을 세어 봅니다.

```python
# 큰 batch 통계 — 80/10/10 비율이 실제로 맞는지 확인
torch.manual_seed(0)
N_DEMO = 64
big_batch = [
    {"input_ids": lm_train[i]["input_ids"], "attention_mask": [1] * BLOCK_SIZE}
    for i in range(N_DEMO)
]
big_out = data_collator(big_batch)
```

**위 코드 읽기** 앞의 시연과 달리 실제 학습 데이터인 `lm_train` 블록을 그대로 넣습니다. 블록 길이가 모두 `BLOCK_SIZE` 로 같으므로 `attention_mask` 도 전부 1 — PAD 가 아예 없다는 점이 `group_texts` 방식의 이점입니다.

```python
in_ids = big_out["input_ids"]
lab    = big_out["labels"]

selected = (lab != -100)
n_total    = lab.numel()
n_selected = selected.sum().item()
n_mask     = ((in_ids == mask_id) & selected).sum().item()
n_kept     = ((in_ids == lab) & selected).sum().item()
n_random   = n_selected - n_mask - n_kept
```

**위 코드 읽기** `selected = (lab != -100)` 한 줄이 *loss 에 참여하는 자리* 전체를 골라내는 마스크입니다. 그 안에서 `[MASK]` 로 바뀐 자리와 원본이 유지된 자리를 세고, random 교체는 나머지로 역산합니다.

```python
print(f"Total tokens:                      {n_total:>7,}")
print(f"Selected for loss (target 15%):    {n_selected:>7,}  ({100 * n_selected / n_total:5.2f}%)")
print(f"  └─ replaced with [MASK]:         {n_mask:>7,}  ({100 * n_mask / n_selected:5.2f}% of selected)")
print(f"  └─ replaced with random:         {n_random:>7,}  ({100 * n_random / n_selected:5.2f}% of selected)")
print(f"  └─ kept as original:             {n_kept:>7,}  ({100 * n_kept / n_selected:5.2f}% of selected)")
print()
print("Target: 선택 15% / 그 중 80-10-10 으로 [MASK]-random-kept. 표본 크면 비율 안정.")
```

**▶ 실행 결과**

```text
Total tokens:                        8,192
Selected for loss (target 15%):      1,217  (14.86%)
  └─ replaced with [MASK]:             961  (78.96% of selected)
  └─ replaced with random:             121  ( 9.94% of selected)
  └─ kept as original:                 135  (11.09% of selected)

Target: 선택 15% / 그 중 80-10-10 으로 [MASK]-random-kept. 표본 크면 비율 안정.
```

**결과 해석**

8,192 토큰 표본에서 선택 비율 14.86%, 내부 분해 78.96 / 9.94 / 11.09% 로 목표치 15% · 80/10/10 을 거의 그대로 재현합니다. 소수점 단위 어긋남은 매 step 새로 뽑는 무작위 표본의 오차일 뿐이며, 실제 학습에서는 수백 step 에 걸쳐 평균화됩니다.

**관전 포인트**

- `what_happened` 가 `—` 인 자리 (약 85%) 는 *입력과 정답이 그대로* — loss 에 기여하지 않음. 모델은 *문맥을 만들어 주는* 역할만.
- `[MASK]` 자리 (약 12%) 가 본 task 의 *진짜 학습 신호*. 주변 토큰들의 attention 결과로 *가려진 자리* 의 vocab 분포를 예측.
- `random` (약 1.5%) 와 `kept` (약 1.5%) 는 *inference 분포 일치* 를 위한 정규화. 추론 시에는 `[MASK]` 가 없으므로 *입력을 절대 신뢰하면 안 된다* 는 신호를 학습에 섞어 줌.
- 매 epoch · 매 batch 마다 마스킹은 *새로 무작위* — 같은 문장이 epoch 마다 다른 자리에서 가려져 학습됨 (data augmentation 효과).

이제 학습 설정입니다. fine-tune 챕터들과 눈에 띄게 다른 곳은 학습률로, 사전학습된 가중치를 조심스레 건드리는 게 아니라 난수에서 출발하므로 훨씬 크게 잡습니다.

```python
USE_FP16 = (DEVICE == "cuda")   # T4 는 fp16, MPS/CPU 는 fp32
NUM_EPOCHS = 2
```

**위 코드 읽기** T4(Compute Capability 7.5)는 bf16 을 지원하지 않으므로 혼합 정밀도는 항상 `fp16` 입니다. CUDA 가 아닐 때 자동으로 fp32 로 떨어지게 해 두어 로컬에서 열어도 코드가 그대로 돕니다.

```python
training_args = TrainingArguments(
    output_dir="./ch20_output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-4,            # scratch 학습이라 fine-tune (2e-5) 보다 크게
    weight_decay=0.01,
    warmup_steps=0.06,             # 1 미만이면 전체 step 대비 *비율* 로 해석 (구 warmup_ratio)
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="no",            # 마지막에 직접 save_pretrained
    report_to="none",
    seed=SEED,
)
```

**위 코드 읽기** `learning_rate=5e-4` 는 fine-tune 관례(2e-5)의 스무 배가 넘는데, random init 본체를 처음부터 끌어올려야 하기 때문입니다. 대신 `warmup_steps=0.06` 으로 초반 6% 구간을 완만하게 올려 발산을 막고, `save_strategy="no"` 로 중간 체크포인트를 남기지 않아 디스크와 시간을 아낍니다.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train,
    eval_dataset=lm_eval,
    data_collator=data_collator,
    processing_class=tokenizer,
)
```

**위 코드 읽기** 분류 챕터들과 달리 `compute_metrics` 가 없습니다 — MLM 은 loss(그리고 그 지수인 perplexity)가 곧 평가 지표라 따로 붙일 metric 이 없기 때문입니다. `data_collator=data_collator` 로 넘긴 마스킹 담당이 매 batch 정답 자리를 새로 고릅니다.

```python
print(f"epochs:        {NUM_EPOCHS}")
print(f"batch size:    {training_args.per_device_train_batch_size}")
print(f"learning rate: {training_args.learning_rate}")
print(f"fp16:          {USE_FP16}")
print(f"train blocks:  {len(lm_train):,}")
print(f"steps / epoch: {len(lm_train) // training_args.per_device_train_batch_size}")
```

**▶ 실행 결과**

```text
epochs:        2
batch size:    32
learning rate: 0.0005
fp16:          True
train blocks:  5,352
steps / epoch: 167
```

**결과 해석**

한 epoch 이 167 step, 2 epoch 이라야 334 step 뿐입니다 — 사전학습이라기엔 대단히 짧은 분량이고, 뒤에서 loss 가 unigram 기준선 근처에 머무는 이유도 여기서 이미 예고됩니다.

### 학습 직전 baseline — 사전학습 전·후 비교 준비

`trainer.train()` 을 호출하기 *전* 의 모델 상태 (`BertForMaskedLM(config)` random init) 로 두 가지를 측정해 둡니다 — *학습 후와 나란히* 보면 *사전학습이 본체에 무엇을 새겼는지* 가 한 화면에 드러납니다.

1. **`eval_loss` / `perplexity`** — random init 이므로 vocab 30,522 균등 분포 (`ln V` ≈ 10.33) 근처가 기대치.
2. **같은 문장의 `[MASK]` top-5** — random init 의 logits 는 *아무 학습 신호도 담기지 않은 난수* 입니다. 따라서 **빈도 순위와 무관한 토큰들이 무작위로 뽑힙니다** — 조각 토큰 (`##…`)이나 희귀어가 섞이기도 하고, 평범한 중간 빈도 단어가 나오기도 합니다 (`set_seed(SEED)` 로 시드를 고정했으므로 다시 돌려도 같은 목록이 나옵니다 — `SEED` 를 바꾸면 완전히 달라집니다). *어떤 토큰이 흔한지조차 아직 반영되지 않은* 상태라는 점을 기억해 두세요 — 학습 후와 대비되는 지점입니다.

학습이 끝난 뒤 6-1 셀에서 *완전히 같은 문장* 으로 다시 측정해 *직접 비교* 합니다.

```python
# predict_mask 함수 정의 — 학습 전·후 두 번 호출하므로 먼저 정의
def predict_mask(text, top_k=5):
    '''text 안의 [MASK] 자리 top-k 토큰과 확률 반환.'''
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return None
    results = []
    for pos in mask_positions:
        probs = torch.softmax(logits[pos], dim=-1)
        top_p, top_i = probs.topk(top_k)
        candidates = [(tokenizer.convert_ids_to_tokens(int(i)), float(p))
                       for p, i in zip(top_p, top_i)]
        results.append((int(pos), candidates))
    return results
```

**위 코드 읽기** `(inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(...)` 로 `[MASK]` 가 놓인 **위치** 를 먼저 찾고, 그 위치의 logits 에만 softmax 를 걸어 top-k 를 뽑습니다. MLM 헤드는 모든 자리에 대해 vocab 분포를 내놓으므로, 우리가 궁금한 자리를 이렇게 직접 골라내야 합니다.

```python
# 검증용 문장 — 학습 전·후 동일하게 사용
# 위키 일반 도메인 (사전학습 직접 본 분포) + Yelp 도메인 (Ch 21 downstream, 다른 도메인 transfer)
test_sentences = [
    # 위키 도메인 — 사전학습 직접 본 분포, 향상 명확히 기대
    f"The capital of France is {tokenizer.mask_token}.",
    f"Water freezes at {tokenizer.mask_token} degrees Celsius.",
    # Yelp 도메인 (Ch 21 fine-tune 대상) — 다른 도메인 transfer 한계 확인
    f"The food at this restaurant was absolutely {tokenizer.mask_token}.",
    f"I would {tokenizer.mask_token} recommend this place.",
]
```

**위 코드 읽기** 검증 문장 네 개가 두 도메인으로 나뉩니다 — 앞 둘은 사전학습이 실제로 본 위키 분포, 뒤 둘은 Ch 21 에서 fine-tune 할 Yelp 분포입니다. 같은 목록을 학습 전·후에 그대로 재사용하므로 변화를 정확히 같은 잣대로 비교할 수 있습니다.

```python
# ---- 사전학습 전 eval_loss / perplexity ----
pre_eval = trainer.evaluate()
pre_eval_loss = pre_eval["eval_loss"]
pre_eval_ppl  = math.exp(pre_eval_loss)
random_baseline_loss = math.log(tokenizer.vocab_size)
```

**위 코드 읽기** `trainer.train()` 을 부르기 *전* 에 `trainer.evaluate()` 를 호출하는 것이 요점으로, 이 값이 random init 상태의 출발점 기록이 됩니다. 바로 옆의 `math.log(tokenizer.vocab_size)` 는 vocab 균등 분포일 때의 이론값이라, 두 수가 얼마나 붙어 있는지가 곧 "정말 아무것도 모르는 상태인가" 의 점검입니다.

```python
print("=" * 78)
print("BEFORE pretraining  (random init body)")
print("=" * 78)
print(f"  eval_loss       : {pre_eval_loss:.4f}   (random baseline ln V = {random_baseline_loss:.4f})")
print(f"  eval_perplexity : {pre_eval_ppl:,.0f}     (random baseline V    = {tokenizer.vocab_size:,})")
print()

# ---- 사전학습 전 [MASK] top-5 ----
pre_top5_records = []
for sent in test_sentences:
    results = predict_mask(sent, top_k=5)
    top5_tokens = [tok for tok, _ in results[0][1]] if results else []
    pre_top5_records.append({"sentence": sent, "top5_before": top5_tokens})
    print(f"input: {sent}")
    print(f"  top-5 before pretraining: {top5_tokens}")
    print()
```

**▶ 실행 결과**

```text
Training Loss  Validation Loss  Epoch
No log         10.372110        0
==============================================================================
BEFORE pretraining  (random init body)
==============================================================================
  eval_loss       : 10.3721   (random baseline ln V = 10.3262)
  eval_perplexity : 31,956     (random baseline V    = 30,522)
input: The capital of France is [MASK].
  top-5 before pretraining: ['76', 'fragments', 'community', 'plea', 'temporal']

input: Water freezes at [MASK] degrees Celsius.
  top-5 before pretraining: ['[unused556]', 'fragments', 'buildings', 'shoving', 'turnout']

input: The food at this restaurant was absolutely [MASK].
  top-5 before pretraining: ['plea', 'turnout', 'siegfried', 'harta', 'roared']

input: I would [MASK] recommend this place.
  top-5 before pretraining: ['ministries', 'terrifying', 'geometric', 'pained', 'ot']
```

**결과 해석**

학습 전 `eval_loss` 10.3721 은 이론값 `ln V = 10.3262` 와 사실상 같아, 본체가 어떤 토큰도 선호하지 않는 순수 난수 상태임을 확인해 줍니다. top-5 에 `[unused556]`·`ministries` 같은 희귀 토큰이 아무렇게나 섞이는 것도 같은 이야기로, *어떤 토큰이 흔한지조차* 아직 모르는 단계입니다.

준비가 끝났으니 실제 사전학습입니다. 한 줄이지만 이 챕터의 본론이며, 걸리는 시간과 loss 가 random baseline 에서 얼마나 내려오는지를 함께 기록합니다.

```python
t0 = time.time()
train_result = trainer.train()
elapsed = time.time() - t0
print(f"\nMLM pretraining done in {elapsed/60:.1f} min")
print(f"mean train loss: {train_result.training_loss:.4f}")
print(f"random baseline loss (uniform over vocab): {math.log(tokenizer.vocab_size):.4f}")
```

**위 코드 읽기** `train_result.training_loss` 는 마지막 step 의 loss 가 아니라 **전 구간 평균** 입니다. 초반의 10 부근 값들이 함께 평균되므로, 표에 찍히는 epoch 별 loss 보다 항상 높게 나옵니다.

**▶ 실행 결과**

```text
Epoch  Training Loss  Validation Loss
1      7.239586       7.226577
2      7.072258       7.064893
MLM pretraining done in 0.4 min
mean train loss: 7.4361
random baseline loss (uniform over vocab): 10.3262
```

**결과 해석**

10.33 에서 출발한 loss 가 1 epoch 만에 7.23 까지 떨어지고 2 epoch 에서 7.07 로 소폭 더 내려갑니다 — 큰 낙폭이 초반에 몰려 있다는 것은 *흔한 토큰* 부터 먼저 배웠다는 신호입니다. train 과 validation 이 거의 붙어 있어 과적합 걱정은 없고, 전체가 0.4분 만에 끝날 만큼 짧은 학습입니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Aug 17 09:02:00 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   67C    P0             64W /   70W |    3335MiB /  15360MiB |     66%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1229      C   /usr/bin/python3                       3332MiB |
+-----------------------------------------------------------------------------------------+
```

**결과 해석**

학습을 마친 뒤에도 15,360MiB 중 3,335MiB 만 쓰고 있습니다 — 약 11M 짜리 모델에 `BLOCK_SIZE=128` 이라 T4 메모리가 크게 남는다는 뜻이고, 🛠️ 변형에서 데이터나 블록 길이를 키울 여지가 여기서 나옵니다.
