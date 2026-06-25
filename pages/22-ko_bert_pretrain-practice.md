> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/22_ko_bert_pretrain/22_ko_bert_pretrain.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

```python
%pip install -q -U transformers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 119.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/555.1 kB ? eta -:--:--
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 46.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 36.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━ 36.4/48.9 MB 263.3 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 146.2 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 146.2 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 17.0 MB/s eta 0:00:00
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

from datasets import Dataset
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
Mon Jun 22 12:19:24 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   45C    P8             11W /   70W |       3MiB /  15360MiB |      0%      Default |
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

## 한국어 Wikipedia 데이터 로드 — 일반 도메인 사전학습 코퍼스

원본 BERT 가 영어 Wikipedia + BookCorpus 라는 *일반 도메인* 코퍼스로 사전학습한 정신을 따라, 본 챕터도 **한국어 Wikipedia 본문** 으로 MLM 사전학습합니다 — *task 도메인 (NSMC 영화 리뷰) 으로 사전학습하면 domain-adaptive pretraining 에 가까워져 사전학습의 진짜 메시지 (일반 표상 학습 → 다른 task 로 transfer) 가 흐려지기 때문*.

**원본**: `wikimedia/wikipedia`, config `20231101.ko`. CC-BY-SA, HF Hub 정제본. article 단위 다운로드 후 paragraph 단위로 split 해 NSMC 5K 문장과 비슷한 토큰 양으로 맞춤. Ch 23 의 분류 fine-tune (NSMC 이진) 은 *완전히 다른 도메인* — 사전학습 → fine-tune transfer 메시지가 정직해집니다.

```python
from datasets import load_dataset

print("downloading Korean Wikipedia (wikimedia/wikipedia, 20231101.ko)...")
ds_raw = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
print(f"  total articles: {len(ds_raw):,}")
print()
print(f"first 3 article previews:")
for i in range(3):
    title = ds_raw[i]["title"]
    text  = ds_raw[i]["text"]
    print(f"  Article {i} ({title}): {text[:80].strip()}")
```

**▶ 실행 결과**

```text
downloading Korean Wikipedia (wikimedia/wikipedia, 20231101.ko)...
  total articles: 647,897

first 3 article previews:
  Article 0 (지미 카터): 제임스 얼 카터 주니어(, 1924년 10월 1일~)는 민주당 출신 미국의 제39대 대통령(1977년~1981년)이다.

생애

어린 시절 
지
  Article 1 (수학): 수학(數學, , 줄여서 math)은 수, 양, 구조, 공간, 변화 등의 개념을 다루는 학문이다. 널리 받아들여지는 명확한 정의는 없으나 현대 수
  Article 2 (수학 상수): 수학에서 상수란 그 값이 변하지 않는 불변량으로, 변수의 반대말이다. 물리 상수와는 달리, 수학 상수는 물리적 측정과는 상관없이 정의된다.

수
```

```python
SEED = 42
N_TRAIN_TEXT = 5000
N_EVAL_TEXT  = 500

# article 본문을 paragraph 단위로 잘라 N_TRAIN + N_EVAL 채우기.
# 너무 짧은 (제목·메타) 또는 너무 긴 (목록·인용) paragraph 제외.
def collect_paragraphs(ds, target, min_len=50, max_len=2000):
    out = []
    for ex in ds:
        for para in ex["text"].split("\n\n"):
            para = para.strip()
            if min_len <= len(para) <= max_len:
                out.append(para)
                if len(out) >= target:
                    return out
    return out

shuffled = ds_raw.shuffle(seed=SEED)
TARGET = N_TRAIN_TEXT + N_EVAL_TEXT
all_paragraphs = collect_paragraphs(shuffled, target=TARGET)

train_ds_raw = Dataset.from_dict({"text": all_paragraphs[:N_TRAIN_TEXT]})
eval_ds_raw  = Dataset.from_dict({"text": all_paragraphs[N_TRAIN_TEXT:N_TRAIN_TEXT + N_EVAL_TEXT]})

print(f"sampled train: {len(train_ds_raw):,} paragraphs")
print(f"sampled eval:  {len(eval_ds_raw):,} paragraphs")
print()
print(f"sample text length stats (chars):")
lens = [len(t) for t in train_ds_raw["text"]]
print(f"  mean: {np.mean(lens):.1f}, median: {np.median(lens):.0f}, max: {max(lens)}")
print()
print(f"first sample preview:")
for i in range(3):
    t = train_ds_raw[i]["text"]
    print(f"  Sample {i}: {t[:120]}")
```

**▶ 실행 결과**

```text
sampled train: 5,000 paragraphs
sampled eval:  500 paragraphs

sample text length stats (chars):
  mean: 194.6, median: 143, max: 1979

first sample preview:
  Sample 0: 원(元)은 시호에 쓰이는 글자다. 《일주서》 시법해에는 능사변중(能思辨衆), 행의열민(行義說民), 시건국도(始建國都), 주의덕행(主義行德)을 일컫는다 한다.
  Sample 1: 원황제 
 전한 원제
 조위 원제
 동진 원제
 후조 원제 (추존)
 북연 원제 (추존)
 염위 원제 (추존)
 하 원제 (추존)
 대 원제 (추존)
 양 원제
 한 원제 (추존)
 당 원제
 북송 원제
 금 원제
  Sample 2: 애그리게이션(aggregation)은 다음을 가리킨다.
 링크 애그리게이션(link aggregation)
 패킷 애그리게이션(packet aggregation)
 뉴스 애그리게이션(news aggregation)
```

## 토크나이저 — `klue/bert-base` 로드 + 영어 토크나이저와 한국어 비교

`klue/bert-base` 의 한국어 WordPiece (vocab 약 32,000) 를 그대로 가져옵니다. *모델은 random init* 이지만 토크나이저는 *완성품* — Ch 20 의 영어 패턴과 동일.

이어서 같은 한국어 문장을 *영어 토크나이저* (`bert-base-uncased`, Ch 20 에서 사용) 와 비교해 Ch 19 §5-4 의 cross-language 결론을 *직접* 확인합니다.

```python
TOKENIZER_NAME = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"tokenizer:        {TOKENIZER_NAME}")
print(f"vocab_size:       {tokenizer.vocab_size:,}")
print(f"model_max_length: {tokenizer.model_max_length}")
print(f"special tokens:")
for name in ("pad_token", "unk_token", "cls_token", "sep_token", "mask_token"):
    tok = getattr(tokenizer, name)
    tid = tokenizer.convert_tokens_to_ids(tok) if tok is not None else None
    print(f"  {name:>11}: {tok!r:>10}  (id={tid})")

# 간단 시연 — 한국어 문장
SAMPLE_KO = "이 영화 정말 재미있어요!"
enc = tokenizer(SAMPLE_KO, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(f"\nKorean sample: {SAMPLE_KO!r}")
print(f"tokens ({len(tokens)}): {tokens}")
print(f"ids:    {enc['input_ids'][0].tolist()}")
```

**▶ 실행 결과**

```text
tokenizer:        klue/bert-base
vocab_size:       32,000
model_max_length: 512
special tokens:
    pad_token:    '[PAD]'  (id=0)
    unk_token:    '[UNK]'  (id=1)
    cls_token:    '[CLS]'  (id=2)
    sep_token:    '[SEP]'  (id=3)
   mask_token:   '[MASK]'  (id=4)

Korean sample: '이 영화 정말 재미있어요!'
tokens (8): ['[CLS]', '이', '영화', '정말', '재미있', '##어요', '!', '[SEP]']
ids:    [2, 1504, 3771, 3944, 6001, 10283, 5, 3]
```

### 같은 한국어 문장을 두 토크나이저로 — Ch 19 §5-4 cross-language 검증

영어 토크나이저 (`bert-base-uncased`) 와 한국어 토크나이저 (`klue/bert-base`) 에 같은 한국어 문장을 통과시켜 토큰 리스트와 UNK 개수를 비교합니다.

```python
# 영어 토크나이저 (Ch 20 에서 사용한 것) 도 로드해 비교
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")

EN_NAME = "bert-base-uncased (EN)"
KO_NAME = "klue/bert-base (KO)"

ko_sentences = [
    "이 영화 정말 재미있어요!",
    "음식이 맛있었고 서비스도 훌륭했습니다.",
    "별로였어요. 시간 낭비.",
]

cross_rows = []
for sent in ko_sentences:
    for name, tok in [(EN_NAME, tokenizer_en), (KO_NAME, tokenizer)]:
        enc = tok(sent, add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(enc["input_ids"])
        n_unk = sum(1 for t in toks if t == "[UNK]")
        cross_rows.append({
            "sentence": sent,
            "tokenizer": name,
            "n_tokens": len(toks),
            "n_unk": n_unk,
            "unk_pct": round(n_unk / len(toks) * 100, 1) if toks else 0.0,
        })

cross_df = pd.DataFrame(cross_rows)
print(cross_df.to_string(index=False))
```

**▶ 실행 결과**

```text
             sentence              tokenizer  n_tokens  n_unk  unk_pct
       이 영화 정말 재미있어요! bert-base-uncased (EN)        15      1      6.7
       이 영화 정말 재미있어요!    klue/bert-base (KO)         6      0      0.0
음식이 맛있었고 서비스도 훌륭했습니다. bert-base-uncased (EN)        19      2     10.5
음식이 맛있었고 서비스도 훌륭했습니다.    klue/bert-base (KO)        12      0      0.0
        별로였어요. 시간 낭비. bert-base-uncased (EN)        13      1      7.7
        별로였어요. 시간 낭비.    klue/bert-base (KO)         7      0      0.0
```

```python
# 실제 토큰 리스트도 한 번 보여줍니다 (첫 12 토큰)
print("=" * 78)
for sent in ko_sentences:
    print(f"\n[Korean input] {sent}")
    for name, tok in [(EN_NAME, tokenizer_en), (KO_NAME, tokenizer)]:
        enc = tok(sent, add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(enc["input_ids"])
        head = toks[:12]
        n_unk = sum(1 for t in toks if t == "[UNK]")
        print(f"  {name:28} ({len(toks):>3} tokens, UNK {n_unk:>2}): {head}")
```

**▶ 실행 결과**

```text
==============================================================================

[Korean input] 이 영화 정말 재미있어요!
  bert-base-uncased (EN)       ( 15 tokens, UNK  1): ['ᄋ', '##ᅵ', 'ᄋ', '##ᅧ', '##ᆼ', '##ᄒ', '##ᅪ', 'ᄌ', '##ᅥ', '##ᆼ', '##ᄆ', '##ᅡ']
  klue/bert-base (KO)          (  6 tokens, UNK  0): ['이', '영화', '정말', '재미있', '##어요', '!']

[Korean input] 음식이 맛있었고 서비스도 훌륭했습니다.
  bert-base-uncased (EN)       ( 19 tokens, UNK  2): ['ᄋ', '##ᅳ', '##ᆷ', '##ᄉ', '##ᅵ', '##ᆨ', '##ᄋ', '##ᅵ', '[UNK]', 'ᄉ', '##ᅥ', '##ᄇ']
  klue/bert-base (KO)          ( 12 tokens, UNK  0): ['음식', '##이', '맛있', '##었', '##고', '서비스', '##도', '훌륭', '##했', '##습', '##니다', '.']

[Korean input] 별로였어요. 시간 낭비.
  bert-base-uncased (EN)       ( 13 tokens, UNK  1): ['[UNK]', '.', 'ᄉ', '##ᅵ', '##ᄀ', '##ᅡ', '##ᆫ', 'ᄂ', '##ᅡ', '##ᆼ', '##ᄇ', '##ᅵ']
  klue/bert-base (KO)          (  7 tokens, UNK  0): ['별로', '##였', '##어요', '.', '시간', '낭비', '.']
```

**관찰 — Ch 19 §5-4 결론의 실측 확인**

- **`bert-base-uncased` (영어)**: 한국어 문장이 *자모 단위* (`ᄋ`, `##ᅵ`, `##ᅧ` ...) 로 분해되거나 `[UNK]` 가 섞임. 토큰 수가 길게 폭증, *의미 단위* 가 사라짐. 모델이 이 표현으로 학습해도 *한국어 어휘 정보* 가 거의 없음.
- **`klue/bert-base` (한국어)**: 한국어 문장이 *어절·형태소* 단위 (`이`, `영화`, `정말`, `재미있`, `##어요`) 로 자연스럽게 쪼개짐. UNK 0개, 토큰 수가 짧고 *의미 단위* 가 보존.

> **결론** — 한국어 데이터로 BERT 를 사전학습하려면 한국어 토크나이저가 필수. Ch 20 의 영어 패턴을 한국어로 옮길 때 *토크나이저만 바꿔도* 같은 학습 동역학이 가능합니다. Ch 19 §5-4 가 *문제 제기* 였다면, 이번 챕터는 *해결책의 첫 단계*.

## 토큰화 + `group_texts` — Ch 20 패턴 그대로

MLM 사전학습 표준 입력 포맷. 모든 문서를 *이어 붙여 토큰 스트림* 으로 만든 뒤 `block_size=128` 단위로 자릅니다. 문장 경계가 사라지는 trade-off 는 있지만 BERT 사전학습은 *임의 위치의 토큰 예측* 이라 문장 경계가 중요하지 않습니다.

한국어 Wikipedia paragraphs 는 *제한 50-2000자 필터링* 으로 평균 문장 길이가 일정 (수십 자-수백 자). 5,000 paragraphs 이 `block_size=128` 로 잘리면 약 500-1,500 블록 정도로 정리됩니다. NSMC 한 줄 리뷰보다 길고 Yelp 보다는 짧은 중간 수준 — 일반 도메인 코퍼스다운 균형.

```python
BLOCK_SIZE = 128

def tokenize_function(examples):
    # 특수 토큰 부착 안 함 — 블록 단위로 자를 거라 [CLS]/[SEP] 가 의미 없음
    return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

tokenized_train = train_ds_raw.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
tokenized_eval = eval_ds_raw.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
print(f"tokenized_train: {tokenized_train}")
print(f"first 30 input_ids of sample 0: {tokenized_train[0]['input_ids'][:30]}")
print(f"first 30 tokens of sample 0:    {tokenizer.convert_ids_to_tokens(tokenized_train[0]['input_ids'][:30])}")
```

**▶ 실행 결과**

```text
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (610 > 512). Running this sequence through the model will result in indexing errors
tokenized_train: Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask'],
    num_rows: 5000
})
first 30 input_ids of sample 0: [1478, 12, 244, 13, 1497, 24307, 2170, 8026, 2259, 8034, 2062, 18, 170, 16164, 2112, 171, 1325, 2520, 2097, 2170, 2259, 797, 2063, 2447, 2284, 12, 471, 353, 1, 1]
first 30 tokens of sample 0:    ['원', '(', '元', ')', '은', '시호', '##에', '쓰이', '##는', '글자', '##다', '.', '《', '일주', '##서', '》', '시', '##법', '##해', '##에', '##는', '능', '##사', '##변', '##중', '(', '能', '思', '[UNK]', '[UNK]']
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
lm_eval  = tokenized_eval.map(group_texts,  batched=True, batch_size=1000)

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
    num_rows: 3924
})
lm_eval:  Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 429
})

block_size:           128
train blocks: 3,924  (approx. 502,272 tokens)
eval blocks:  429   (approx. 54,912 tokens)

sample block 0 first 20 ids: [1478, 12, 244, 13, 1497, 24307, 2170, 8026, 2259, 8034, 2062, 18, 170, 16164, 2112, 171, 1325, 2520, 2097, 2170]
sample block 0 first 20 tok: ['원', '(', '元', ')', '은', '시호', '##에', '쓰이', '##는', '글자', '##다', '.', '《', '일주', '##서', '》', '시', '##법', '##해', '##에']
```

## 작은 `BertConfig` + `BertForMaskedLM` — random init (Ch 20 과 동일)

본체 구조는 Ch 20 과 *완전히 동일* — hidden=256, layer=4, head=4, intermediate=1024. vocab 만 한국어 토크나이저 (32,000) 에 맞춤.

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
print(f"vocab_size:              {tokenizer.vocab_size:,}  (klue/bert-base)")
print()
print(f"Total parameters:    {total:>13,}  ({total/1e6:.2f} M)")
print(f"Trainable:           {trainable:>13,}")
print(f"  embeddings:        {emb:>13,}  ({emb/total:.1%})  (vocab {tokenizer.vocab_size} x hidden {HIDDEN_SIZE})")
print(f"  encoder (4 layer): {encoder:>13,}  ({encoder/total:.1%})")
print(f"  MLM head:          {head:>13,}  ({head/total:.1%})  (tied with embeddings)")
```

**▶ 실행 결과**

```text
Config: hidden=256, layer=4, head=4, intermediate=1024
max_position_embeddings: 128
vocab_size:              32,000  (klue/bert-base)

Total parameters:       11,483,136  (11.48 M)
Trainable:              11,483,136
  embeddings:            8,225,792  (71.6%)  (vocab 32000 x hidden 256)
  encoder (4 layer):     3,159,040  (27.5%)
  MLM head:                 98,304  (0.9%)  (tied with embeddings)
```

**관찰** — vocab 이 약 32,000 (Ch 20 의 30,522 보다 약간 큼) 이라 임베딩 테이블이 살짝 더 큽니다. 그래도 본체 구조는 동일 — encoder body 2M + 임베딩 8M 수준의 작은 BERT.

> Ch 20 과 마찬가지로 MLM head 의 weight 는 입력 임베딩과 *tied* (공유). vocab 차원 출력 layer 가 임베딩 테이블과 같아 파라미터 절약.

## `DataCollatorForLanguageModeling` + Trainer 학습

collator 가 매 batch 마다 *무작위로 약 15% 토큰을 [MASK]* 로 바꾸고, 그 위치의 정답 토큰을 `labels` 로 표시. 나머지 위치는 `-100` → CrossEntropyLoss 가 무시.

**MLM masking 규칙** (BERT 원논문) — Ch 20 / Ch 21 과 동일:
- 선택된 약 15% 중 80%: 실제로 `[MASK]` 로 교체
- 10%: 무작위 다른 토큰으로 교체
- 10%: 원래 토큰 유지

이 규칙은 *언어와 무관* — collator 코드가 토큰 id 만 보고 처리합니다.

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# collator 동작 확인 — 같은 입력을 두 번 처리해 mask 위치가 매번 다른지 보기
sample_batch = [lm_train[0], lm_train[1]]
out1 = data_collator(sample_batch)
out2 = data_collator(sample_batch)

print(f"batch shape: input_ids={tuple(out1['input_ids'].shape)}, labels={tuple(out1['labels'].shape)}")
mask_id = tokenizer.mask_token_id

n_masked_1 = (out1["input_ids"] == mask_id).sum().item()
n_masked_2 = (out2["input_ids"] == mask_id).sum().item()
total_tokens = out1["input_ids"].numel()
print(f"masked tokens (call 1): {n_masked_1:>4} / {total_tokens}  ({n_masked_1/total_tokens:.2%})")
print(f"masked tokens (call 2): {n_masked_2:>4} / {total_tokens}  ({n_masked_2/total_tokens:.2%})")

# labels 에서 -100 이 아닌 위치 = MLM loss 가 계산되는 위치
n_loss_pos = (out1["labels"] != -100).sum().item()
print(f"loss positions:        {n_loss_pos:>4} / {total_tokens}  "
      f"({n_loss_pos/total_tokens:.2%})  (labels != -100)")
```

**▶ 실행 결과**

```text
batch shape: input_ids=(2, 128), labels=(2, 128)
masked tokens (call 1):   35 / 256  (13.67%)
masked tokens (call 2):   26 / 256  (10.16%)
loss positions:          45 / 256  (17.58%)  (labels != -100)
```

### 5-1. 🔍 [MASK] 가 들어가는 원리 — 한 눈에 보는 80/10/10 (한국어 풀버전)

`DataCollatorForLanguageModeling` 은 매 step 마다 *입력 토큰의 약 15%* 를 *무작위로* 선택하고, 선택된 위치마다 세 가지 중 하나를 적용합니다.

| 선택된 토큰 운명 | 비율 | 의도 |
| --- | --- | --- |
| `[MASK]` 로 교체 | **80%** | 표준 마스킹 — 모델이 *주변 문맥만으로* 원래 토큰을 맞추도록 |
| **다른 random 토큰** 으로 교체 | 10% | inference 때는 `[MASK]` 가 없으니, 모델이 *항상* 자기 입력을 *의심* 하게 만듦 |
| **원본 그대로** 유지 | 10% | 동일 — 입력과 정답이 일치하는 케이스도 학습 데이터에 포함 |

**나머지 85%** 의 토큰은 `labels = -100` 으로 두어 *loss 계산에서 제외* 됩니다 (PyTorch CE 의 `ignore_index` 기본값). 즉 한 step 의 MLM loss 는 *선택된 15% 자리만* 모아 평균한 값.

> 이 `labels = -100` 트릭은 BERT-만의 것이 아닙니다 — Phase 4 GPT 사전학습은 *거의 모든 토큰* 을 학습 (`labels = input_ids`), SFT (Ch 27) 는 *prompt 만 -100, 답변만 학습*. 같은 트릭, 정반대 자리. Ch 21 / 영어 짝과 동일한 풀버전 시각화로 한국어 환경에서도 직접 확인.

```python
# 한국어 예시 문장 한 개에 collator 한 번 적용 — 어떤 자리가 어떻게 바뀌나
DEMO_SENT_KO = "이 영화는 정말 재미있었고 배우들 연기도 훌륭했습니다."
demo_enc = tokenizer(DEMO_SENT_KO, return_tensors=None)
demo_ids = demo_enc["input_ids"]

torch.manual_seed(0)  # 재현성: 같은 seed 면 같은 마스킹
demo_batch = [{"input_ids": demo_ids, "attention_mask": [1] * len(demo_ids)}]
demo_out = data_collator(demo_batch)

masked_ids = demo_out["input_ids"][0].tolist()
labels     = demo_out["labels"][0].tolist()   # -100 = loss 무시, 그 외 = 원본 token id
mask_id_local = tokenizer.mask_token_id

orig_tokens   = tokenizer.convert_ids_to_tokens(demo_ids)
masked_tokens = tokenizer.convert_ids_to_tokens(masked_ids)

rows = []
for orig_id, new_id, lab, orig_tok, new_tok in zip(demo_ids, masked_ids, labels, orig_tokens, masked_tokens):
    if lab == -100:
        kind = "-"                       # 미선택 (loss 계산 X)
    elif new_id == mask_id_local:
        kind = "[MASK] (80%)"             # 표준 마스킹
    elif new_id == orig_id:
        kind = "kept (10%)"               # 선택됐지만 원본 유지
    else:
        kind = "random (10%)"             # 다른 token 으로 교체
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
   0    [CLS]          [CLS]      -100             -
   1        이              이      -100             -
   2       영화         [MASK]      3771  [MASK] (80%)
   3      ##는            ##는      2259    kept (10%)
   4       정말             정말      -100             -
   5      재미있            재미있      -100             -
   6      ##었            ##었      -100             -
   7      ##고            ##고      -100             -
   8       배우             배우      -100             -
   9      ##들            ##들      -100             -
  10       연기             연기      -100             -
  11      ##도            ##도      -100             -
  12       훌륭         [MASK]      5825  [MASK] (80%)
  13      ##했            ##했      -100             -
  14      ##습            ##습      -100             -
  15     ##니다           ##니다      -100             -
  16        .              .      -100             -
  17    [SEP]          [SEP]      -100             -
```

```python
# 큰 batch 통계 — 80/10/10 비율이 실제로 맞는지 확인 (한국어 lm_train 사용)
torch.manual_seed(0)
N_DEMO = 64
big_batch = [
    {"input_ids": lm_train[i]["input_ids"], "attention_mask": [1] * BLOCK_SIZE}
    for i in range(N_DEMO)
]
big_out = data_collator(big_batch)

in_ids = big_out["input_ids"]
lab_big = big_out["labels"]

selected = (lab_big != -100)
n_total    = lab_big.numel()
n_selected = selected.sum().item()
n_mask     = ((in_ids == mask_id_local) & selected).sum().item()
n_kept     = ((in_ids == lab_big) & selected).sum().item()
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
Selected for loss (target 15%):      1,209  (14.76%)
  └─ replaced with [MASK]:             955  (78.99% of selected)
  └─ replaced with random:             119  ( 9.84% of selected)
  └─ kept as original:                 135  (11.17% of selected)

Target: 선택 15% / 그 중 80-10-10 으로 [MASK]-random-kept. 표본 크면 비율 안정.
```

**관전 포인트**

- `what_happened` 가 `-` 인 자리 (약 85%) 는 *입력과 정답이 그대로* — loss 에 기여하지 않습니다. 모델은 *문맥을 만들어 주는* 역할만.
- `[MASK]` 자리 (약 12%) 가 본 task 의 *진짜 학습 신호*. 주변 한국어 토큰들의 attention 결과로 *가려진 자리* 의 vocab 분포를 예측.
- `random` (약 1.5%) 과 `kept` (약 1.5%) 는 *inference 분포 일치* 를 위한 정규화. 추론 시에는 `[MASK]` 가 없으므로 *입력을 절대 신뢰하면 안 된다* 는 신호를 학습에 섞어 줌. 영어 (Ch 20·21) 와 같은 규칙.
- 매 epoch · 매 batch 마다 마스킹은 *새로 무작위* — 같은 한국어 문장이 epoch 마다 다른 자리에서 가려져 학습됨 (data augmentation 효과).

> **결론 한 줄** — *`[MASK]` 트릭은 언어와 무관, 본체만 한국어를 학습.* `DataCollatorForLanguageModeling` 코드는 한국어든 영어든 *토큰 id 위에서만* 동작합니다. 언어 차이는 *학습된 임베딩의 의미* 에 반영될 뿐, masking 메커니즘 자체는 동일.

### 5-2. 학습 시작

Ch 20 과 같은 hyperparams — epoch 2, batch 32, lr 5e-4 (scratch 사전학습 표준), warmup 0.06, fp16 (T4).

```python
USE_FP16 = (DEVICE == "cuda")   # T4 는 fp16, MPS/CPU 는 fp32
NUM_EPOCHS = 2

training_args = TrainingArguments(
    output_dir="./ch22_output",
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
train blocks:  3,924
steps / epoch: 122
```

### 학습 직전 baseline — 사전학습 전·후 비교 준비

`trainer.train()` 을 호출하기 *전* 의 모델 상태 (`BertForMaskedLM(config)` random init) 로 두 가지를 측정해 둡니다 — *학습 후와 나란히* 보면 *사전학습이 본체에 무엇을 새겼는지* 가 한 화면에 드러납니다.

1. **`eval_loss` / `perplexity`** — random init 이므로 vocab 32,000 균등 분포 (`ln V` ≈ 10.37) 근처가 기대치.
2. **같은 문장의 `[MASK]` top-5** — random init 의 logits 는 거의 균등이라 *문맥과 무관한 토큰* (자주 등장하는 조사·어미·특수문자 등) 이 뽑힙니다.

학습이 끝난 뒤 7번 셀에서 *완전히 같은 문장* 으로 다시 측정해 *직접 비교* 합니다.

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


# 검증용 한국어 문장 — 학습 전·후 동일하게 사용.
# 사전학습이 *위키 일반 도메인* 이므로 일반 문장 두 개 + NSMC 도메인 두 개 섞어 transfer 확인.
test_sentences = [
    # 위키 도메인 — 사전학습이 직접 본 분포, 향상 명확히 기대
    f"대한민국의 수도는 {tokenizer.mask_token}이다.",
    f"태양계에는 행성이 {tokenizer.mask_token} 개 있다.",
    # NSMC 도메인 (Ch 23 fine-tune 대상) — 다른 도메인 transfer 한계 확인
    f"이 영화 정말 {tokenizer.mask_token}.",
    f"배우 연기가 {tokenizer.mask_token} 좋았어요.",
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
  eval_loss       : 10.4255   (random baseline ln V = 10.3735)
  eval_perplexity : 33,709     (random baseline V    = 32,000)
input: 대한민국의 수도는 [MASK]이다.
  top-5 before pretraining: ['##희정', '해석', '찬성', '전한', 'par']

input: 태양계에는 행성이 [MASK] 개 있다.
  top-5 before pretraining: ['이씨', '저지른', '1958', '몰입', '끄집어내']

input: 이 영화 정말 [MASK].
  top-5 before pretraining: ['계약서', '서귀', '스페인어', '드세요', 'William']

input: 배우 연기가 [MASK] 좋았어요.
  top-5 before pretraining: ['계약서', '서귀', '드세요', '스페인어', 'William']
```

```python
t0 = time.time()
train_result = trainer.train()
elapsed = time.time() - t0
print(f"\nKorean MLM pretraining done in {elapsed/60:.1f} min")
print(f"mean train loss: {train_result.training_loss:.4f}")
print(f"random baseline loss (uniform over vocab): {math.log(tokenizer.vocab_size):.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Korean MLM pretraining done in 0.3 min
mean train loss: 7.7967
random baseline loss (uniform over vocab): 10.3735
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 12:20:24 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   59C    P0             41W /   70W |    3449MiB /  15360MiB |     54%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1984      C   /usr/bin/python3                       3446MiB |
+-----------------------------------------------------------------------------------------+
```
