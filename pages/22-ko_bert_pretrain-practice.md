> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/22_ko_bert_pretrain/22_ko_bert_pretrain.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

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

한국어 Wikipedia 정제본(`wikimedia/wikipedia`, `20231101.ko`)을 article 단위로 내려받습니다. 일반 도메인 사전학습 코퍼스로, Ch 23 의 NSMC 영화 리뷰와는 의도적으로 다른 도메인입니다. 전체 article 수가 약 64만 개로 크니 다운로드가 본 챕터에서 가장 오래 걸리는 단계입니다.

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

**결과 해석**

전체 약 64만 article 이 로드됐고, 첫 미리보기에서 위키 본문이 *순한국어 + 한자·과학 용어* 가 섞인 일반 도메인 텍스트임이 보입니다 — 사전학습 코퍼스로 적합한 분포입니다.

article 본문을 그대로 쓰지 않고 paragraph(`\n\n` 단위) 로 잘라 학습 5,000 + 평가 500 개를 채웁니다. 너무 짧은 제목·메타나 너무 긴 목록·인용은 길이 필터(50-2000자)로 걸러 NSMC 5K 문장과 비슷한 토큰 양으로 맞추는 것이 목적입니다.

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

**결과 해석**

학습 5,000 / 평가 500 paragraph 가 채워졌고, 평균 약 195자·중앙값 143자로 NSMC 한 줄 리뷰보다 길고 Yelp 보다는 짧은 중간 길이입니다 — 일반 도메인 코퍼스다운 분포입니다.

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

`klue/bert-base` 토크나이저(vocab 32,000)가 그대로 로드됐고, 한국어 예시 문장이 `이 / 영화 / 정말 / 재미있 / ##어요` 처럼 어절·형태소 단위로 자연스럽게 8 토큰으로 쪼개집니다 — UNK 없이 의미 단위가 보존됩니다.

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

같은 한국어 문장에서 영어 토크나이저는 토큰 수가 2배 이상 길고 UNK 가 6.7-10.5% 섞이는 반면, 한국어 토크나이저는 토큰 수가 절반 이하이고 UNK 가 0%입니다 — Ch 19 §5-4 의 cross-language 결론이 실측으로 그대로 확인됩니다.

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

영어 토크나이저는 한국어를 `ᄋ`, `##ᅵ` 같은 *자모 단위* 로 분해하거나 `[UNK]` 로 떨어뜨려 의미 단위가 사라집니다. 반대로 한국어 토크나이저는 `음식 / ##이 / 맛있 / 서비스` 처럼 어절·형태소로 자연스럽게 쪼개므로, 한국어 데이터엔 한국어 토크나이저가 필수임이 토큰 리스트로 직접 보입니다.

paragraph 전체를 토큰 id 로 변환합니다. 곧바로 블록 단위로 자를 것이라 `[CLS]`/`[SEP]` 가 의미 없어 `add_special_tokens=False`, `truncation=False` 로 둡니다. 이 단계에서는 512 초과 경고가 떠도 무시해도 됩니다 — 다음 셀의 `group_texts` 가 어차피 128 블록으로 잘라내기 때문입니다.

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

**결과 해석**

토큰화된 데이터셋은 `input_ids` 등만 남고 5,000행 그대로입니다. 맨 앞 경고는 일부 paragraph 가 512 토큰을 넘는다는 안내로, 잘라내기를 다음 단계에 맡기므로 정상입니다. 토큰 리스트에서 한자(`能`, `思`)는 vocab 에 없어 `[UNK]` 로 떨어지지만, 한국어 본문 자체는 의미 단위로 잘 쪼개집니다.

```python
def group_texts(examples):
    '''HF 표준 group_texts — 모든 토큰 스트림을 이어 붙인 뒤 block_size 로 자름.'''
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    # block_size 배수로 잘라내기 (마지막 토막은 버림)
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
```

**위 코드 읽기** — `sum(examples[k], [])` 로 한 batch 안 모든 paragraph 의 토큰을 하나의 긴 스트림으로 이어 붙인 뒤, 그 길이를 `BLOCK_SIZE`(128)의 배수로 내림합니다. 즉 마지막에 남는 128 미만의 토막은 버려집니다.

```python
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    # labels = input_ids 사본 (collator 가 mask 위치만 골라냄)
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result
```

**위 코드 읽기** — 이어 붙인 스트림을 128 토큰씩 끊어 고정 길이 블록으로 만듭니다. `labels` 는 `input_ids` 의 사본일 뿐이고, 실제로 어느 자리를 학습할지(마스킹)는 다음 단계의 collator 가 매 batch 마다 동적으로 결정합니다.

```python


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

**결과 해석**

5,000 paragraph 가 128 토큰 블록으로 재조립되어 학습 3,924 블록(약 50만 토큰)·평가 429 블록이 만들어졌습니다. 노트북 개요는 약 500-1,500 블록을 예상했지만 실제로는 3,924 블록으로, paragraph 가 예상보다 길어 더 많은 블록이 나왔습니다.

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
```

**위 코드 읽기** — Ch 20 과 동일한 작은 본체(hidden=256, layer=4, head=4)를 구성하되 `vocab_size` 만 한국어 토크나이저의 32,000 에 맞춥니다. `BertForMaskedLM(config)` 는 사전학습 weight 없이 *random init* 된 모델이라, 이번 챕터의 학습이 그 빈 본체에 무엇을 새기는지를 직접 관찰하게 됩니다.

```python

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

**결과 해석**

전체 약 11.5M 파라미터 중 임베딩 테이블이 71.6%(약 8.2M)를 차지합니다 — vocab 32,000 × hidden 256 이라 본체보다 어휘 임베딩이 훨씬 큽니다. MLM head 는 임베딩과 weight 를 공유(tied)해 0.9%에 불과합니다.

MLM collator 를 만들고 동작을 확인합니다. 핵심은 마스킹이 *매 batch 마다 새로 무작위* 라는 점입니다 — 같은 입력을 두 번 넣어 마스크 자리가 매번 달라지는지 보고, `labels != -100` 인 자리(loss 가 계산되는 자리)도 함께 셉니다.

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

**결과 해석**

같은 입력인데도 1차 호출(35개)과 2차 호출(26개)의 마스크 개수가 달라, 마스킹이 매번 새로 무작위임이 확인됩니다. 작은 256 토큰 표본이라 비율이 13-17%로 흔들리지만, 큰 표본에서는 목표치 15%로 안정됩니다.

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
```

**위 코드 읽기** — 각 자리를 네 부류로 분류합니다. `labels == -100` 이면 학습에서 제외된 자리(`-`)이고, 선택된 자리는 토큰이 `[MASK]` 로 바뀌었는지·다른 토큰(random)으로 바뀌었는지·원본 그대로(kept)인지로 다시 나뉩니다 — BERT 의 80/10/10 규칙을 한 자리씩 눈으로 따라가는 셀입니다.

```python
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

18 토큰 중 `영화`·`훌륭` 두 자리만 `[MASK]` 로 가려졌고(`label_id` 에 원본 토큰 id 보존), `##는` 한 자리는 선택됐지만 원본 유지(kept)된 케이스입니다. 나머지 자리는 모두 `-100` 으로 loss 에서 제외되어, 모델은 이들을 *문맥* 으로만 활용합니다.

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

8,192 토큰의 큰 표본에서는 선택 비율이 14.76%로 목표 15%에 수렴하고, 그 안에서 79%/9.8%/11.2%로 BERT 의 80/10/10 규칙이 거의 정확히 재현됩니다. collator 가 토큰 id 만 보고 처리하므로 한국어에서도 영어와 동일한 비율이 나옵니다.

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

`[MASK]` 자리의 top-k 후보를 뽑는 `predict_mask` 함수를 정의하고, 학습 *전* 의 random init 상태로 eval_loss·perplexity 와 네 문장의 top-5 를 측정해 둡니다. 학습 후 *완전히 같은 문장* 으로 다시 측정해 나란히 비교하기 위한 baseline 입니다.

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

**결과 해석**

학습 전 eval_loss 가 10.43 으로 random baseline `ln V`(10.37)에 거의 일치 — 모델이 vocab 32,000 을 사실상 균등 추측하는 상태입니다. top-5 도 `##희정`·`계약서`·`William` 처럼 문맥과 무관한 토큰이 뽑혀, 아직 한국어 언어 구조를 전혀 학습하지 못했음을 보여줍니다.

이제 한국어 MLM 사전학습을 실행합니다. 작은 본체 + 5K paragraph 라 T4 에서 수십 초 안에 끝나며, 평균 train loss 가 random baseline(10.37)에서 얼마나 내려갔는지가 학습이 진행됐다는 첫 신호입니다.

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

**결과 해석**

학습이 약 0.3분만에 끝났고 평균 train loss 가 7.80 으로 random baseline 10.37 보다 확실히 내려가, 본체가 한국어 구조의 일부를 학습하기 시작했음을 보여줍니다. 다만 잘 학습된 작은 BERT 목표 영역(2.3-3.0)에는 못 미치는데, 데이터·모델·학습 시간이 작아 자연스러운 결과입니다.

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

```python
# 학습 로그에서 train loss 추출
log_history = trainer.state.log_history
train_logs = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]

if train_logs:
    steps, losses = zip(*train_logs)
    random_baseline = math.log(tokenizer.vocab_size)

    sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, "o-", color="#4878D0", label="학습 MLM loss")
    ax.axhline(random_baseline, color="black", lw=1.0, ls=":",
               label=f"랜덤 기준선 (ln V = {random_baseline:.2f})")
    ax.set_xlabel("학습 step")
    ax.set_ylabel("MLM loss (CrossEntropy)")
    ax.set_title("MLM 학습 loss — 한국어 위키백과 위에서 처음부터 학습한 small BERT")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No train loss logs found.")
```

**▶ 실행 결과**

![output](../assets/22-ko_bert_pretrain-out1.png)

**결과 해석**

학습 step 이 진행되며 MLM loss 가 랜덤 기준선(점선, 약 10.37)에서 빠르게 떨어집니다 — 첫 100 step 안의 급락이 vocab 과 학습이 정상 작동한다는 신호입니다.

eval set 의 perplexity 를 측정합니다. perplexity 는 `exp(eval_loss)` 로, *마스크 자리마다 모델이 몇 개 후보로 좁혔는가* 로 해석할 수 있습니다 (랜덤 기준선은 vocab 전체인 32,000).

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
               eval_loss: 7.5138

  MLM loss:               7.5138
  perplexity (exp loss):  1833.24
  random baseline PPL:    32,000  (uniform over vocab)
  -> model narrowed vocab to approx. 1833 candidates per masked position
```

**결과 해석**

eval perplexity 가 약 1,833 으로, 랜덤 기준선 32,000 에서 크게 내려왔습니다 — 모델이 각 마스크 자리의 후보를 약 1,800 개 수준으로 좁혔다는 뜻입니다. 작은 toy 셋업이라 큰 BERT(perplexity 수십 수준)에는 못 미치지만 학습 방향은 분명합니다.

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
  eval_loss       : 7.5249   (before: 10.4255)
  eval_perplexity : 1,853.59        (before: 33,709)
  -> narrowed vocab to approx. 1854 candidates per masked position

input: 대한민국의 수도는 [MASK]이다.
  top-5 after pretraining: ['.', '##의', ',', '##에', '##는']

input: 태양계에는 행성이 [MASK] 개 있다.
  top-5 after pretraining: ['.', '##의', '##다', '##에', ',']

input: 이 영화 정말 [MASK].
  top-5 after pretraining: ['.', '##의', ',', ')', '##다']

input: 배우 연기가 [MASK] 좋았어요.
  top-5 after pretraining: ['.', '##의', ',', ')', '##다']
```

**결과 해석**

학습 후 perplexity 가 약 1,854 로 학습 전 33,709 에서 18배가량 줄었습니다. 다만 top-5 가 `.`·`##의`·`,` 같은 *고빈도 조사·문장부호* 위주라, 정답 토큰(`서울`, `8` 등)을 안정적으로 맞히기에는 데이터·모델 크기가 부족함이 드러납니다 — 학습 전의 무작위 토큰과는 질적으로 다른, 빈도 통계는 학습한 상태입니다.

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
      eval_loss          10.4255           7.5249          10.3735
eval_perplexity       33708.8968        1853.5880       32000.0000
```

**결과 해석**

eval_loss 가 10.43(랜덤 기준선 수준)에서 7.52 로, perplexity 가 약 33,709 에서 1,854 로 내려간 것이 한 표에 정리됩니다 — 사전학습이 본체에 새긴 변화의 직접적 증거입니다.

```python
# 막대 그래프 두 장 (eval_loss / perplexity)
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

loss_values = [pre_eval_loss, post_eval_loss]
loss_labels = ["학습 전 (랜덤)", "학습 후 (2 epoch)"]
axes[0].bar(loss_labels, loss_values, color=["#999999", "#EE854A"])
axes[0].axhline(random_baseline_loss, color="black", lw=1.0, ls=":",
                label=f"랜덤 기준선 ln V = {random_baseline_loss:.2f}")
axes[0].set_ylabel("eval_loss")
axes[0].set_title("MLM eval_loss")
axes[0].legend(loc="upper right", fontsize=10)

ppl_values = [pre_eval_ppl, post_eval_ppl]
axes[1].bar(loss_labels, ppl_values, color=["#999999", "#EE854A"])
axes[1].set_yscale("log")
axes[1].axhline(tokenizer.vocab_size, color="black", lw=1.0, ls=":",
                label=f"랜덤 기준선 V = {tokenizer.vocab_size:,}")
axes[1].set_ylabel("perplexity (log scale)")
axes[1].set_title("MLM perplexity")
axes[1].legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/22-ko_bert_pretrain-out2.png)

**결과 해석**

두 막대 모두 학습 후(주황)가 랜덤 기준선(점선) 아래로 내려가, eval_loss 와 perplexity(로그 스케일) 양쪽에서 학습 효과가 시각적으로 분명합니다.

이번엔 *학습이 충분히 잘 됐을 때의 기준점* 으로 표준 `klue/bert-base`(110M)를 로드합니다. 같은 토크나이저를 쓰므로 모델만 바꿔 같은 문장에 적용하면 우리 작은 BERT 와 직접 비교할 수 있습니다.

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
bert.pooler.dense.weight    | UNEXPECTED |  | 
cls.seq_relationship.weight | UNEXPECTED |  | 
bert.pooler.dense.bias      | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Our small BERT params: 11.5M
Reference BERT params: 110.7M  (10x larger)
```

**결과 해석**

표준 `klue/bert-base` 는 110.7M 으로 우리 작은 BERT(11.5M)의 약 10배 크기입니다. LOAD REPORT 의 UNEXPECTED 키(`pooler`, `seq_relationship`)는 MLM 에 쓰이지 않는 헤드라 무시해도 됩니다.

앞서 정의한 `predict_mask` 는 전역 `model` 에 고정돼 있어, 참조 모델에는 임의의 MLM 모델을 인자로 받는 `predict_mask_with` 를 따로 정의합니다. 같은 `test_sentences` 4개를 표준 `klue/bert-base` 에 통과시켜 top-5 를 `ref_top5_records` 에 모아 둔 뒤, 110M 참조 모델은 곧장 메모리에서 해제해 T4 VRAM 을 비웁니다. 다음 셀의 3-way 비교 표가 이 기록을 학습 전·후 결과와 나란히 묶습니다.

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
  before (random)            : ##희정, 해석, 찬성, 전한, par
  ours  (small, 5K para)     : ., ##의, ,, ##에, ##는
  ref   (klue/bert-base)     : 서울, 광화문, 평양, 부산, 인천

input: 태양계에는 행성이 [MASK] 개 있다.
  before (random)            : 이씨, 저지른, 1958, 몰입, 끄집어내
  ours  (small, 5K para)     : ., ##의, ##다, ##에, ,
  ref   (klue/bert-base)     : 여러, 몇, 두, 세, 다섯

input: 이 영화 정말 [MASK].
  before (random)            : 계약서, 서귀, 스페인어, 드세요, William
  ours  (small, 5K para)     : ., ##의, ,, ), ##다
  ref   (klue/bert-base)     : 좋아, [UNK], ., 좋아해, 좋아한다

input: 배우 연기가 [MASK] 좋았어요.
  before (random)            : 계약서, 서귀, 드세요, 스페인어, William
  ours  (small, 5K para)     : ., ##의, ,, ), ##다
  ref   (klue/bert-base)     : 너무, 정말, 참, 굉장히, 아주
```

**결과 해석**

세 모델의 격차가 정확히 *데이터·모델 크기·학습 시간* 의 격차로 드러납니다. 표준 `klue/bert-base` 는 `대한민국의 수도는 [MASK]` 에 `서울`, `[MASK] 좋았어요` 에 `너무`·`정말` 처럼 정답을 top-5 에 자연스럽게 올리는 반면, 우리 작은 BERT 는 방향은 학습 전보다 나아졌지만 아직 고빈도 조사·부호에 머뭅니다.

학습한 모델과 토크나이저를 *같은 폴더* 에 저장합니다. Ch 23 에서 이 폴더를 `AutoModelForSequenceClassification.from_pretrained(...)` 한 줄로 불러와 분류 헤드를 얹을 것이라, 본체와 토크나이저가 함께 있어야 합니다.

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

**결과 해석**

`config.json` + `model.safetensors`(약 44MB) + 토크나이저 파일이 HF 표준 레이아웃으로 저장됐습니다 — Ch 23 fine-tune 의 출발 체크포인트입니다.
