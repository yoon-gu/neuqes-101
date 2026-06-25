> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/20_en_bert_pretrain/20_en_bert_pretrain.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

```python
%pip install -q -U transformers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 85.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 46.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 35.3 MB/s eta 0:00:00
   ━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.3/48.9 MB 249.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 149.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 149.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 17.2 MB/s eta 0:00:00
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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
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
    print("Warning: CPU runtime — MLM training will be very slow. Switch to T4 recommended.")
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
Mon Jun 22 04:08:21 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   36C    P8             12W /   70W |       3MiB /  15360MiB |      0%      Default |
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

print(f"tokenizer:        {TOKENIZER_NAME}")
print(f"vocab_size:       {tokenizer.vocab_size:,}")
print(f"model_max_length: {tokenizer.model_max_length}")
print(f"special tokens:")
for name in ("pad_token", "unk_token", "cls_token", "sep_token", "mask_token"):
    tok = getattr(tokenizer, name)
    tid = tokenizer.convert_tokens_to_ids(tok) if tok is not None else None
    print(f"  {name:>11}: {tok!r:>10}  (id={tid})")

# 간단 시연 — 일반 위키풍 문장
SAMPLE = "The capital of France is Paris, located on the Seine river."
enc = tokenizer(SAMPLE, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(f"\nsample: {SAMPLE!r}")
print(f"tokens ({len(tokens)}): {tokens}")
print(f"ids:    {enc['input_ids'][0].tolist()}")
```

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

## 데이터 — Wikitext-103 paragraphs (일반 도메인 사전학습 코퍼스)

원본 BERT 가 영어 Wikipedia + BookCorpus 라는 *일반 도메인* 코퍼스로 사전학습한 정신을 따라, 본 챕터도 **Wikitext-103** 본문으로 MLM 사전학습합니다 — *task 도메인 (Yelp 리뷰(식당·업체)) 으로 사전학습하면 domain-adaptive pretraining 에 가까워져 사전학습의 진짜 메시지 (일반 표상 학습 → 다른 task 로 transfer) 가 흐려지기 때문*.

**원본**: `Salesforce/wikitext`, config `wikitext-103-raw-v1` (CC-BY-SA, 정제된 영문 위키 paragraphs). HF Hub 정제본 — line 단위로 이미 정리되어 있어 빈 줄 / 너무 짧은 줄 / 너무 긴 줄만 제외하면 바로 사용 가능. Ch 21 의 분류 fine-tune (Yelp 이진) 은 *완전히 다른 도메인* — 사전학습 → fine-tune transfer 메시지가 정직해집니다. Ch 22 의 한국어 (한국어 Wikipedia paragraphs) 와 *대칭* 패턴.

```python
SEED = 42
N_TRAIN_TEXT = 5000
N_EVAL_TEXT  = 500

print("downloading Wikitext-103 (Salesforce/wikitext, wikitext-103-raw-v1)...")
raw_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
raw_eval  = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
print(f"  raw train lines: {len(raw_train):,}")
print(f"  raw eval  lines: {len(raw_eval):,}")

# 빈 줄 / 너무 짧은 줄 (제목·메타) / 너무 긴 줄 (목록·인용) 제외
def is_good(ex, min_len=50, max_len=2000):
    t = ex["text"].strip()
    return min_len <= len(t) <= max_len

train_filtered = raw_train.filter(is_good).shuffle(seed=SEED).select(range(N_TRAIN_TEXT))
eval_filtered  = raw_eval.filter(is_good).shuffle(seed=SEED).select(range(N_EVAL_TEXT))

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

## 토큰화 + `group_texts` — HF `run_mlm.py` 표준 패턴

MLM 사전학습의 표준 입력 포맷은 *고정 길이 블록*. 변동 길이 문장에 그대로 padding 하면 *손실*: (a) 짧은 문장이 많으면 PAD 비율이 높아 GPU 시간 낭비, (b) 긴 문장은 truncation 으로 정보 손실.

**해결책**: 모든 문서를 *이어 붙여 토큰 스트림* 으로 만든 뒤, `block_size=128` 단위로 자름. 문장 경계가 사라지는 trade-off 가 있지만, BERT 사전학습은 *임의 위치의 토큰 예측* 이라 문장 경계가 중요하지 않음.

Wikitext-103 paragraphs 는 *제한 50-2000자 필터링* 으로 평균 문장 길이가 일정 (수백 자 위주). 5,000 paragraphs 가 `block_size=128` 로 잘리면 약 5,352 블록 (약 68만 토큰) 으로 정리됩니다. 코드는 HF `examples/pytorch/language-modeling/run_mlm.py` 의 `group_texts` 함수를 그대로 따른 표준 패턴.

```python
BLOCK_SIZE = 128

def tokenize_function(examples):
    # 특수 토큰 부착 안 함 — 블록 단위로 자를 거라 [CLS]/[SEP] 가 의미 없음
    return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

tokenized_train = ds_train_raw.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
tokenized_eval = ds_eval_raw.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
print(f"tokenized_train: {tokenized_train}")
print(f"first 30 input_ids of sample 0: {tokenized_train[0]['input_ids'][:30]}")
```

**▶ 실행 결과**

```text
tokenized_train: Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask'],
    num_rows: 5000
})
first 30 input_ids of sample 0: [20222, 12131, 10131, 4819, 15272, 1010, 1996, 4410, 3159, 1997, 1996, 2406, 1997, 2655, 4430, 9691, 1998, 1 …(뒤 77자 생략)
```

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
| 총 파라미터 | 약 110M | **약 10M** (toy 규모) |

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

model = BertForMaskedLM(config)  # random init — pretrained weight 없음!

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

**관찰** — 작은 BERT 의 파라미터는 *임베딩 테이블이 절반 이상* 차지합니다 (vocab 30522 × hidden 256 ≈ 7.8M). encoder body 자체는 약 2M. 이게 *vocab 큰데 모델이 작은* 셋업의 특징 — 표준 BERT (vocab 30K × hidden 768 ≈ 23M / 110M = 21%) 와 비율이 매우 다릅니다.

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

### [MASK] 가 들어가는 원리 — 한 눈에 보는 80/10/10

`DataCollatorForLanguageModeling` 은 매 step 마다 *입력 토큰의 약 15%* 를 *무작위로* 선택하고, 선택된 위치마다 세 가지 중 하나를 적용합니다.

| 선택된 토큰 운명 | 비율 | 의도 |
| --- | --- | --- |
| `[MASK]` 로 교체 | **80%** | 표준 마스킹 — 모델이 *주변 문맥만으로* 원래 토큰을 맞추도록 |
| **다른 random 토큰** 으로 교체 | 10% | inference 때는 `[MASK]` 가 없으니, 모델이 *항상* 자기 입력을 *의심* 하게 만듦 |
| **원본 그대로** 유지 | 10% | 동일 — 입력과 정답이 일치하는 케이스도 학습 데이터에 포함 |

**나머지 85%** 의 토큰은 `labels = -100` 으로 두어 *loss 계산에서 제외* 됩니다 (PyTorch CE 의 `ignore_index` 기본값). 즉 한 step 의 MLM loss 는 *선택된 15% 자리만* 모아 평균한 값.

> 이 `labels = -100` 트릭은 BERT-만의 것이 아닙니다 — Phase 4 GPT 사전학습은 *거의 모든 토큰* 을 학습 (`labels = input_ids`), SFT (Ch 27) 는 *prompt 만 -100, 답변만 학습*. 같은 트릭, 정반대 자리. Ch 21 에서 더 자세히.

```python
# 짧은 예시 문장 하나에 collator 한 번 돌려서 어떤 자리가 어떻게 바뀌는지 직접 봅니다.
import pandas as pd

DEMO_SENT = "Pretraining a language model on Wikipedia teaches it general English structure."
demo_enc = tokenizer(DEMO_SENT, return_tensors=None)
demo_ids = demo_enc["input_ids"]

torch.manual_seed(0)  # 재현성: 같은 seed 면 같은 마스킹
demo_batch = [{"input_ids": demo_ids, "attention_mask": [1] * len(demo_ids)}]
demo_out = data_collator(demo_batch)

masked_ids = demo_out["input_ids"][0].tolist()
labels     = demo_out["labels"][0].tolist()
mask_id    = tokenizer.mask_token_id

orig_tokens   = tokenizer.convert_ids_to_tokens(demo_ids)
masked_tokens = tokenizer.convert_ids_to_tokens(masked_ids)

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

```python
# 큰 batch 통계 — 80/10/10 비율이 실제로 맞는지 확인
torch.manual_seed(0)
N_DEMO = 64
big_batch = [
    {"input_ids": lm_train[i]["input_ids"], "attention_mask": [1] * BLOCK_SIZE}
    for i in range(N_DEMO)
]
big_out = data_collator(big_batch)

in_ids = big_out["input_ids"]
lab    = big_out["labels"]

selected = (lab != -100)
n_total    = lab.numel()
n_selected = selected.sum().item()
n_mask     = ((in_ids == mask_id) & selected).sum().item()
n_kept     = ((in_ids == lab) & selected).sum().item()
n_random   = n_selected - n_mask - n_kept

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

**관전 포인트**

- `what_happened` 가 `—` 인 자리 (약 85%) 는 *입력과 정답이 그대로* — loss 에 기여하지 않음. 모델은 *문맥을 만들어 주는* 역할만.
- `[MASK]` 자리 (약 12%) 가 본 task 의 *진짜 학습 신호*. 주변 토큰들의 attention 결과로 *가려진 자리* 의 vocab 분포를 예측.
- `random` (약 1.5%) 와 `kept` (약 1.5%) 는 *inference 분포 일치* 를 위한 정규화. 추론 시에는 `[MASK]` 가 없으므로 *입력을 절대 신뢰하면 안 된다* 는 신호를 학습에 섞어 줌.
- 매 epoch · 매 batch 마다 마스킹은 *새로 무작위* — 같은 문장이 epoch 마다 다른 자리에서 가려져 학습됨 (data augmentation 효과).

```python
USE_FP16 = (DEVICE == "cuda")   # T4 는 fp16, MPS/CPU 는 fp32
NUM_EPOCHS = 2

training_args = TrainingArguments(
    output_dir="./ch20_output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-4,            # scratch 학습이라 fine-tune (2e-5) 보다 크게
    weight_decay=0.01,
    warmup_ratio=0.06,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="no",            # 마지막에 직접 save_pretrained
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train,
    eval_dataset=lm_eval,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print(f"epochs:        {NUM_EPOCHS}")
print(f"batch size:    {training_args.per_device_train_batch_size}")
print(f"learning rate: {training_args.learning_rate}")
print(f"fp16:          {USE_FP16}")
print(f"train blocks:  {len(lm_train):,}")
print(f"steps / epoch: {len(lm_train) // training_args.per_device_train_batch_size}")
```

**▶ 실행 결과**

```text
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
epochs:        2
batch size:    32
learning rate: 0.0005
fp16:          True
train blocks:  5,352
steps / epoch: 167
```

### 학습 직전 baseline — 사전학습 전·후 비교 준비

`trainer.train()` 을 호출하기 *전* 의 모델 상태 (`BertForMaskedLM(config)` random init) 로 두 가지를 측정해 둡니다 — *학습 후와 나란히* 보면 *사전학습이 본체에 무엇을 새겼는지* 가 한 화면에 드러납니다.

1. **`eval_loss` / `perplexity`** — random init 이므로 vocab 30,522 균등 분포 (`ln V` ≈ 10.33) 근처가 기대치.
2. **같은 문장의 `[MASK]` top-5** — random init 의 logits 는 거의 균등이라 *문맥과 무관한 토큰* (자주 등장하는 관사·전치사·특수문자 등) 이 뽑힙니다.

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

# ---- 사전학습 전 eval_loss / perplexity ----
pre_eval = trainer.evaluate()
pre_eval_loss = pre_eval["eval_loss"]
pre_eval_ppl  = math.exp(pre_eval_loss)
random_baseline_loss = math.log(tokenizer.vocab_size)

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
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
==============================================================================
BEFORE pretraining  (random init body)
==============================================================================
  eval_loss       : 10.3834   (random baseline ln V = 10.3262)
  eval_perplexity : 32,319     (random baseline V    = 30,522)
input: The capital of France is [MASK].
  top-5 before pretraining: ['broadband', '##eti', 'weights', 'smack', '##umen']

input: Water freezes at [MASK] degrees Celsius.
  top-5 before pretraining: ['##hp', 'damages', 'slogan', '##ssion', '[unused358]']

input: The food at this restaurant was absolutely [MASK].
  top-5 before pretraining: ['languages', 'sampling', '##pic', 'libretto', '##orus']

input: I would [MASK] recommend this place.
  top-5 before pretraining: ['ahmed', 'smack', 'stations', '##sam', 'now']
```

```python
t0 = time.time()
train_result = trainer.train()
elapsed = time.time() - t0
print(f"\nMLM pretraining done in {elapsed/60:.1f} min")
print(f"mean train loss: {train_result.training_loss:.4f}")
print(f"random baseline loss (uniform over vocab): {math.log(tokenizer.vocab_size):.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
MLM pretraining done in 0.4 min
mean train loss: 7.4179
random baseline loss (uniform over vocab): 10.3262
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 04:09:26 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   51C    P0             65W /   70W |    3335MiB /  15360MiB |     49%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5453      C   /usr/bin/python3                       3332MiB |
+-----------------------------------------------------------------------------------------+
```
