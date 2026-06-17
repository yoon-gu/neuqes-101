## 환경 준비

```python
%pip install -q -U transformers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━ 10.3/11.2 MB 155.7 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 101.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 50.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 38.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━ 23.4/48.9 MB 225.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 157.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 157.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 17.8 MB/s eta 0:00:00
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

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 22:01:31 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   42C    P8             11W /   70W |       3MiB /  15360MiB |      0%      Default |
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

**결과 해석**

`klue/bert-base` 토크나이저는 한국어 문장을 `영화`, `재미있` 같은 의미 단위로 쪼개고, 어미 `##어요`만 서브워드로 붙여 8개 토큰으로 깔끔하게 처리합니다. 한국어 어휘로 학습된 vocab(32,000)이라 [UNK] 없이 자연스럽게 분절되는 점을 확인할 수 있습니다.

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

**결과 해석**

같은 한국어 문장도 영어 토크나이저는 토큰 수가 2배 가까이 많고 [UNK]가 6.7-10.5% 발생하지만, 한국어 토크나이저는 토큰 수가 절반이면서 [UNK]가 0%입니다. 사전학습할 언어에 맞는 vocab을 쓰는 것이 왜 중요한지 수치로 드러납니다.

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

**결과 해석**

영어 토크나이저는 한글을 자모(`ᄋ`, `##ᅵ`) 단위로 산산이 부수거나 [UNK]로 흘려 버리는 반면, 한국어 토크나이저는 `별로`, `시간`, `낭비` 같은 단어를 통째로 잡아냅니다. 같은 입력이라도 토큰화가 이렇게 달라야 학습 신호가 의미 단위로 모입니다.

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
masked tokens (call 1):   22 / 256  (8.59%)
masked tokens (call 2):   27 / 256  (10.55%)
loss positions:          30 / 256  (11.72%)  (labels != -100)
```

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

**결과 해석**

18개 토큰 중 `영화`와 `훌륭` 두 자리만 선택돼 학습 대상이 되고, 나머지는 label이 -100이라 loss에서 빠집니다. `영화`는 [MASK]로 가려졌고 `##는`은 선택은 됐지만 원본 그대로 둔(kept) 경우로, 80/10/10 규칙이 한 문장에서 어떻게 적용되는지 한눈에 보입니다.

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

**결과 해석**

8,192개 토큰에서 14.76%가 학습 대상으로 선택됐고, 그 안에서 [MASK]:random:kept가 약 79:10:11로 목표 80/10/10에 근접합니다. 한 문장에선 들쭉날쭉하던 비율이 표본이 커지자 안정적으로 수렴하는 것을 확인할 수 있습니다.

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
  eval_loss       : 10.4471   (random baseline ln V = 10.3735)
  eval_perplexity : 34,446     (random baseline V    = 32,000)
input: 대한민국의 수도는 [MASK]이다.
  top-5 before pretraining: ['서약', '정공', '##퐁', '회선', '슘']

input: 태양계에는 행성이 [MASK] 개 있다.
  top-5 before pretraining: ['양봉', '선거전', '##yp', '나중', 'D']

input: 이 영화 정말 [MASK].
  top-5 before pretraining: ['전유물', '대구', '##껄', '신기록', '학번']

input: 배우 연기가 [MASK] 좋았어요.
  top-5 before pretraining: ['전유물', '대구', '신기록', '가구', '##르']
```

**결과 해석**

학습 전 random init 모델은 eval_loss 10.45, perplexity 34,446으로 vocab 크기(32,000)에 해당하는 무작위 추측 수준입니다. [MASK] 예측도 `서약`, `##퐁`, `전유물`처럼 문맥과 무관한 토큰이라, 아직 한국어에 대해 아무것도 모른다는 출발점을 보여줍니다.

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
mean train loss: 7.8204
random baseline loss (uniform over vocab): 10.3735
```

**결과 해석**

2 epoch 학습으로 train loss가 무작위 기준 10.37에서 7.82까지 내려와, 짧은 시간에도 모델이 분포를 좁히기 시작했음을 보여줍니다. 다만 0.3분이라는 극히 짧은 학습이라 완성 모델과는 거리가 멀고, 뒤의 예측 결과에서 그 한계가 드러납니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 22:02:36 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   54C    P0             58W /   70W |    3449MiB /  15360MiB |     37%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1014      C   /usr/bin/python3                       3446MiB |
+-----------------------------------------------------------------------------------------+
```

```python
# 학습 로그에서 train loss 추출
log_history = trainer.state.log_history
train_logs = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]

if train_logs:
    steps, losses = zip(*train_logs)
    random_baseline = math.log(tokenizer.vocab_size)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, "o-", color="#4878D0", label="train MLM loss")
    ax.axhline(random_baseline, color="black", lw=1.0, ls=":",
               label=f"random baseline (ln V = {random_baseline:.2f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("MLM loss (CrossEntropy)")
    ax.set_title("MLM training loss — small BERT scratch on Korean Wikipedia")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No train loss logs found.")
```

**▶ 실행 결과**

![output](../assets/22-ko_bert_pretrain-out1.png)

**결과 해석**

train loss 곡선이 무작위 기준선(점선) 아래로 빠르게 떨어지며 학습이 정상적으로 진행됨을 보여줍니다. 다만 아직 곡선이 평평해지지 않아, 데이터나 epoch를 늘리면 더 내려갈 여지가 남아 있습니다.

```python
eval_metrics = trainer.evaluate()
eval_loss = eval_metrics["eval_loss"]
eval_ppl = math.exp(eval_loss)
print("=== eval (held-out Korean Wikipedia paragraphs) ===")
for k, v in eval_metrics.items():
    if isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
print()
print(f"  MLM loss:               {eval_loss:.4f}")
print(f"  perplexity (exp loss):  {eval_ppl:.2f}")
print(f"  random baseline PPL:    {tokenizer.vocab_size:,}  (uniform over vocab)")
print(f"  -> model narrowed vocab to approx. {eval_ppl:.0f} candidates per masked position")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
=== eval (held-out Korean Wikipedia paragraphs) ===
               eval_loss: 7.5589

  MLM loss:               7.5589
  perplexity (exp loss):  1917.77
  random baseline PPL:    32,000  (uniform over vocab)
  -> model narrowed vocab to approx. 1918 candidates per masked position
```

**결과 해석**

held-out 문단의 perplexity가 무작위 32,000에서 약 1,918로 줄어, 모델이 가려진 자리마다 후보를 32,000개에서 2,000개 안쪽으로 좁혔다는 뜻입니다. Ch 20 영어판과 마찬가지로 짧은 사전학습이라 여전히 높은 값이지만, 학습 신호가 분명히 들어갔습니다.

```python
# ---- 사전학습 후 eval_loss / perplexity ----
post_eval = trainer.evaluate()
post_eval_loss = post_eval["eval_loss"]
post_eval_ppl  = math.exp(post_eval_loss)

print("=" * 78)
print("AFTER pretraining  (2 epoch MLM)")
print("=" * 78)
print(f"  eval_loss       : {post_eval_loss:.4f}   (before: {pre_eval_loss:.4f})")
print(f"  eval_perplexity : {post_eval_ppl:,.2f}        (before: {pre_eval_ppl:,.0f})")
print(f"  -> narrowed vocab to approx. {post_eval_ppl:.0f} candidates per masked position")
print()

# ---- 사전학습 후 [MASK] top-5 ----
post_top5_records = []
for sent in test_sentences:
    results = predict_mask(sent, top_k=5)
    top5_tokens = [tok for tok, _ in results[0][1]] if results else []
    post_top5_records.append({"sentence": sent, "top5_after": top5_tokens})
    print(f"input: {sent}")
    print(f"  top-5 after pretraining: {top5_tokens}")
    print()
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
==============================================================================
AFTER pretraining  (2 epoch MLM)
==============================================================================
  eval_loss       : 7.5660   (before: 10.4471)
  eval_perplexity : 1,931.42        (before: 34,446)
  -> narrowed vocab to approx. 1931 candidates per masked position

input: 대한민국의 수도는 [MASK]이다.
  top-5 after pretraining: ['.', ',', '##의', '##다', '##년']

input: 태양계에는 행성이 [MASK] 개 있다.
  top-5 after pretraining: ['.', '##의', ',', '##다', ')']

input: 이 영화 정말 [MASK].
  top-5 after pretraining: ['.', ',', '##의', ')', '##다']

input: 배우 연기가 [MASK] 좋았어요.
  top-5 after pretraining: ['.', ',', '##의', '##다', ')']
```

**결과 해석**

학습 후 [MASK] 예측이 무작위 토큰에서 `.`, `,`, `##의`, `##다` 같은 한국어 고빈도 토큰으로 수렴했습니다. Ch 20 영어판과 똑같이, 짧은 사전학습은 우선 가장 흔한 조사·구두점부터 학습하므로 아직 `서울` 같은 내용어까지는 닿지 못한 단계입니다.

```python
# 사전·사후 수치 비교 표
metric_compare = pd.DataFrame({
    "metric":           ["eval_loss", "eval_perplexity"],
    "before (random)":  [pre_eval_loss,  pre_eval_ppl],
    "after (2 epoch)":  [post_eval_loss, post_eval_ppl],
    "random baseline":  [random_baseline_loss, float(tokenizer.vocab_size)],
})
print("Before vs After — eval metrics")
print(metric_compare.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
Before vs After — eval metrics
         metric  before (random)  after (2 epoch)  random baseline
      eval_loss          10.4471           7.5660          10.3735
eval_perplexity       34445.6443        1931.4213       32000.0000
```

```python
# 막대 그래프 두 장 (eval_loss / perplexity)
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

loss_values = [pre_eval_loss, post_eval_loss]
loss_labels = ["before (random)", "after (2 epoch)"]
axes[0].bar(loss_labels, loss_values, color=["#999999", "#EE854A"])
axes[0].axhline(random_baseline_loss, color="black", lw=1.0, ls=":",
                label=f"random baseline ln V = {random_baseline_loss:.2f}")
axes[0].set_ylabel("eval_loss")
axes[0].set_title("MLM eval_loss")
axes[0].legend(loc="upper right", fontsize=10)

ppl_values = [pre_eval_ppl, post_eval_ppl]
axes[1].bar(loss_labels, ppl_values, color=["#999999", "#EE854A"])
axes[1].set_yscale("log")
axes[1].axhline(tokenizer.vocab_size, color="black", lw=1.0, ls=":",
                label=f"random baseline V = {tokenizer.vocab_size:,}")
axes[1].set_ylabel("perplexity (log scale)")
axes[1].set_title("MLM perplexity")
axes[1].legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/22-ko_bert_pretrain-out2.png)

**결과 해석**

두 막대 모두 학습 후(주황) 값이 무작위 기준선(점선) 아래로 내려가, eval_loss와 perplexity가 함께 개선됐음을 시각적으로 확인할 수 있습니다. perplexity는 로그 스케일에서도 눈에 띄게 줄어 무작위 대비 약 16배 좁아진 셈입니다.

```python
# 표준 klue/bert-base 로드 — 학습이 충분히 잘 된 경우의 기준점
from transformers import AutoModelForMaskedLM

ref_model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")
ref_model.to(model.device)
ref_model.eval()

ref_param_count = sum(p.numel() for p in ref_model.parameters())
our_param_count = sum(p.numel() for p in model.parameters())
print(f"Our small BERT params: {our_param_count/1e6:.1f}M")
print(f"Reference BERT params: {ref_param_count/1e6:.1f}M  ({ref_param_count/our_param_count:.0f}x larger)")
```

**▶ 실행 결과**

```text
[transformers] BertForMaskedLM LOAD REPORT from: klue/bert-base
Key                         | Status     |  | 
----------------------------+------------+--+-
cls.seq_relationship.bias   | UNEXPECTED |  | 
bert.pooler.dense.bias      | UNEXPECTED |  | 
cls.seq_relationship.weight | UNEXPECTED |  | 
bert.pooler.dense.weight    | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Our small BERT params: 11.5M
Reference BERT params: 110.7M  (10x larger)
```

```python
# Reference 모델로 같은 문장의 top-5 측정
def predict_mask_with(text, ref, top_k=5):
    '''임의의 MLM 모델로 [MASK] 자리 top-k 예측.'''
    ref.eval()
    inputs = tokenizer(text, return_tensors="pt").to(ref.device)
    with torch.no_grad():
        outputs = ref(**inputs)
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


ref_top5_records = []
for sent in test_sentences:
    results = predict_mask_with(sent, ref_model, top_k=5)
    top5_tokens = [tok for tok, _ in results[0][1]] if results else []
    ref_top5_records.append({"sentence": sent, "top5_ref": top5_tokens})

# 참조 모델 메모리 해제
del ref_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

```python
# 3-way top-5 비교 표
rows = []
for pre, post, ref in zip(pre_top5_records, post_top5_records, ref_top5_records):
    rows.append({
        "sentence":          pre["sentence"],
        "top5_before":       ", ".join(pre["top5_before"]),
        "top5_ours":         ", ".join(post["top5_after"]),
        "top5_ref_bert":     ", ".join(ref["top5_ref"]),
    })

top5_compare = pd.DataFrame(rows)
print("Before (random) vs Ours (small BERT, ko wiki 5K) vs Reference (klue/bert-base, approx. 8.4B tokens)")
print("=" * 100)
for _, row in top5_compare.iterrows():
    print(f"input: {row['sentence']}")
    print(f"  before (random)            : {row['top5_before']}")
    print(f"  ours  (small, 5K para)     : {row['top5_ours']}")
    print(f"  ref   (klue/bert-base)     : {row['top5_ref_bert']}")
    print()
```

**▶ 실행 결과**

```text
Before (random) vs Ours (small BERT, ko wiki 5K) vs Reference (klue/bert-base, approx. 8.4B tokens)
====================================================================================================
input: 대한민국의 수도는 [MASK]이다.
  before (random)            : 서약, 정공, ##퐁, 회선, 슘
  ours  (small, 5K para)     : ., ,, ##의, ##다, ##년
  ref   (klue/bert-base)     : 서울, 광화문, 평양, 부산, 인천

input: 태양계에는 행성이 [MASK] 개 있다.
  before (random)            : 양봉, 선거전, ##yp, 나중, D
  ours  (small, 5K para)     : ., ##의, ,, ##다, )
  ref   (klue/bert-base)     : 여러, 몇, 두, 세, 다섯

input: 이 영화 정말 [MASK].
  before (random)            : 전유물, 대구, ##껄, 신기록, 학번
  ours  (small, 5K para)     : ., ,, ##의, ), ##다
  ref   (klue/bert-base)     : 좋아, [UNK], ., 좋아해, 좋아한다

input: 배우 연기가 [MASK] 좋았어요.
  before (random)            : 전유물, 대구, 신기록, 가구, ##르
  ours  (small, 5K para)     : ., ,, ##의, ##다, )
  ref   (klue/bert-base)     : 너무, 정말, 참, 굉장히, 아주
```

**결과 해석**

충분히 학습한 `klue/bert-base`는 `대한민국의 수도는 [MASK]`에 `서울`, 수량 자리에 `여러`-`몇`, 감정 자리에 `너무`-`정말`처럼 문맥에 맞는 내용어를 정확히 채웁니다. 우리 모델은 아직 고빈도 토큰에 머물러 있는데, 이 격차가 곧 약 8.4B 토큰과 5,000 문단의 학습량 차이를 그대로 보여줍니다.

```python
SAVE_DIR = "./ch22_small_bert_mlm_ko"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

import os
print(f"Saved to: {SAVE_DIR}")
print(f"Files:")
for f in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    if size > 1024 * 1024:
        size_str = f"{size / 1024 / 1024:.1f} MB"
    else:
        size_str = f"{size / 1024:.1f} KB"
    print(f"  {f:>30s}  {size_str}")
```

**▶ 실행 결과**

```text
Saved to: ./ch22_small_bert_mlm_ko
Files:
                     config.json  0.7 KB
               model.safetensors  43.8 MB
                  tokenizer.json  734.4 KB
           tokenizer_config.json  0.4 KB
```
