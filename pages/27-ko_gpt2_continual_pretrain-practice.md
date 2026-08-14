> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/27_ko_gpt2_continual_pretrain/27_ko_gpt2_continual_pretrain.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

```python
%pip install -q -U transformers tokenizers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.7/11.7 MB 109.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 559.1/559.1 kB 45.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━ 24.2/50.1 MB 229.6 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 235.6 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 235.6 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.1/50.1 MB 18.7 MB/s eta 0:00:00
```

```python
import warnings
warnings.filterwarnings("ignore")

import math
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# device 자동 감지 - Colab T4 / 로컬 MPS / CPU 모두 지원
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device     : cuda  ({device_name})")
    print(f"VRAM total : {vram_gib:.2f} GiB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("device     : mps  (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("device     : cpu  (training will be very slow - Colab T4 recommended)")

print(f"torch      : {torch.__version__}")

# 재현성
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# fp16 은 CUDA 에서만 (MPS 는 미지원, CPU 는 의미 없음)
USE_FP16 = (device.type == "cuda")
print(f"use fp16   : {USE_FP16}")

# matplotlib 한글 폰트 (Colab — NanumGothic). plot 의 한국어가 □ 로 깨지지 않게.
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, subprocess, os
_fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(_fp):
    subprocess.run("apt-get -qq -y install fonts-nanum", shell=True)
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
```

**▶ 실행 결과**

```text
device     : cuda  (Tesla T4)
VRAM total : 14.56 GiB
torch      : 2.11.0+cu128
use fp16   : True
```

## 한국어 TinyStories 데이터 로드 + story 복원 — *Ch 26 과 완전히 동일*

본 챕터의 데이터는 *통제 변수*. Ch 26 과 정확히 같은 방식으로 `g0ster/TinyStories-Korean` 을 로드합니다. 이 데이터셋은 *story 단위가 아니라 줄(line) 단위* 로 저장되어 있어, *`<|endoftext|>` 를 만날 때까지 줄을 이어 붙여* 한 story 로 복원합니다. 그렇게 복원한 처음 **30,000 stories** 만 사용 (Ch 26 과 같은 규모, T4 30분 룰 안). *데이터를 고정하고 본체·토크나이저·lr 만 바꿔 격차를 본다* 가 본 챕터의 격리 실험 설계.

```python
from datasets import load_dataset, Dataset

EOT_MARK = "<|endoftext|>"      # 데이터셋이 story 경계 표시에 쓰는 마커
N_TRAIN  = 30_000               # Ch 26 과 동일
N_VAL    = 500
MAX_LINES_TO_SCAN = 800_000


def rebuild_stories(split, n_stories, max_lines):
    '''줄 단위 데이터를 <|endoftext|> 기준으로 이어 붙여 story 리스트로 복원.'''
    stories, buf = [], []
    stream = load_dataset("g0ster/TinyStories-Korean", split=split, streaming=True)
    for i, ex in enumerate(stream):
        if i >= max_lines or len(stories) >= n_stories:
            break
        line = (ex["text"] or "").strip()
        if line == EOT_MARK:
            story = " ".join(buf).strip()
            if story:
                stories.append(story)
            buf = []
        elif line:
            buf.append(line)
    if buf and len(stories) < n_stories:
        tail = " ".join(buf).strip()
        if tail:
            stories.append(tail)
    return stories[:n_stories]
```

**위 코드 읽기** — `rebuild_stories` 는 스트리밍으로 한 줄씩 받으며 `buf` 에 쌓다가, 줄이 `EOT_MARK` (`<|endoftext|>`) 이면 그때까지 모은 줄을 한 story 로 합쳐 `stories` 에 넣고 `buf` 를 비웁니다. `N_TRAIN = 30_000` 으로 *Ch 26 과 같은 규모* 만 잘라 쓰는 게 데이터 통제의 핵심입니다.

```python
t0 = time.time()
train_stories = rebuild_stories("train", N_TRAIN, MAX_LINES_TO_SCAN)
val_stories   = rebuild_stories("validation", N_VAL, 50_000)
print(f"rebuilt stories: train={len(train_stories):,}, val={len(val_stories):,}  ({time.time()-t0:.1f}s)")

raw_train = Dataset.from_dict({"text": train_stories})
raw_val   = Dataset.from_dict({"text": val_stories})
print("train:", raw_train)
print("val  :", raw_val)
print("\n=== sample story (same as Ch 26) ===")
print(raw_train[0]["text"][:400])
```

**▶ 실행 결과**

```text
rebuilt stories: train=30,000, val=500  (22.1s)
train: Dataset({
    features: ['text'],
    num_rows: 30000
})
val  : Dataset({
    features: ['text'],
    num_rows: 500
})

=== sample story (same as Ch 26) ===
한때 벤이라는 이름의 어린 소년이 있었어요. 벤은 주변 세계를 탐험하는 것을 좋아했답니다. 그는 가게에 전시되어 있던 아름다운 꽃병들 같은 멋진 것들을 많이 봤어요. 어느 날, 벤은 가게를 거닐다가 정말 특별한 꽃병을 발견했죠. 벤은 그 꽃병을 보고 …(뒤 240자 생략)
```

**결과 해석**

train 30,000 / val 500 stories 가 약 22초 만에 복원됐고, 샘플 story 가 동화체 한국어 (`…했어요` / `…했답니다`) 로 정상 복원됐습니다 — Ch 26 과 글자 그대로 같은 데이터라 격리 실험의 통제 변수가 보장됩니다.

## KoGPT2 토크나이저·모델 로드 — *모델 로드 한 줄로 학습 단계 2 진입*

본 챕터의 *유일한 큰 변화*. Ch 26 의 `GPT2LMHeadModel(config)` random init 대신 `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` 한 줄. 토크나이저도 같이 가져옵니다. 영어 Ch 25 의 `gpt2` 로드와 정확히 같은 패턴 — 모델 id 만 한국어 KoGPT2.

본 챕터의 *유일한 큰 변화* 입니다. Ch 26 의 `GPT2LMHeadModel(config)` random init 대신 *대규모 한국어 코퍼스로 이미 사전학습된* KoGPT2 본체와 토크나이저를 그대로 가져옵니다. 다만 KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정이 있어, `PreTrainedTokenizerFast` 로 special token 을 직접 지정해 로드하는 점을 눈여겨봐야 합니다.

```python
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

t0 = time.time()
# 주의: KoGPT2 는 AutoTokenizer 가 영어 GPT2 토크나이저로 잘못 fallback 합니다.
# (special token 이 <|endoftext|> 로 잡히고 한국어가 깨짐.) SKT 공식 방식대로
# PreTrainedTokenizerFast 로 special token 을 직접 지정해 로드해야 합니다.
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)

model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2").to(device)
# pad token id 를 본체 config 에도 동기화
model.config.pad_token_id = tokenizer.pad_token_id
print(f"load done: {time.time()-t0:.1f}s")
```

**위 코드 읽기** — 토크나이저는 SKT 공식 방식대로 special token (`</s>`, `<unk>`, `<pad>`, `<mask>`) 을 인자로 명시해 `PreTrainedTokenizerFast` 로 로드하고, 본체는 `from_pretrained("skt/kogpt2-base-v2")` 한 줄로 가져옵니다. 마지막에 `model.config.pad_token_id` 를 토크나이저 pad id 에 맞춰 동기화해 collator·generation 에서 pad 처리가 어긋나지 않게 합니다.

```python
n_params = model.num_parameters()
print(f"\n=== model ===")
print(f"#params           : {n_params/1e6:.2f} M  (Ch 26 was approx. 3M; Ch 27 is approx. {n_params/3e6:.0f}x larger)")
print(f"vocab_size        : {tokenizer.vocab_size:,}  (Ch 26 was approx. 4,000; Ch 27 is approx. {tokenizer.vocab_size/4000:.0f}x larger)")
print(f"weight tying      : {model.config.tie_word_embeddings}  (lm_head <-> wte shared)")
print(f"fp32 weight size  : {n_params * 4 / 1024**2:.1f} MiB")
print(f"\ntokenizer    : {type(tokenizer).__name__}")
print(f"  eos_token  : {tokenizer.eos_token}  id={tokenizer.eos_token_id}")
print(f"  pad_token  : {tokenizer.pad_token}  id={tokenizer.pad_token_id}")
print(f"\nmodel: {type(model).__name__}")
print(f"  - body : {type(model.transformer).__name__}  (Decoder, causal attention)")
print(f"  - head : {type(model.lm_head).__name__}(in={model.lm_head.in_features}, out={model.lm_head.out_features})")
```

**▶ 실행 결과**

```text
pytorch_model.bin: downloading bytes:           |  0.00B            
[transformers] GPT2LMHeadModel LOAD REPORT from: skt/kogpt2-base-v2
Key                                     | Status     |  | 
----------------------------------------+------------+--+-
transformer.h.{0...11}.attn.masked_bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
model.safetensors: downloading bytes:           |  0.00B            
load done: 9.9s

=== model ===
#params           : 125.16 M  (Ch 26 was approx. 3M; Ch 27 is approx. 42x larger)
vocab_size        : 51,200  (Ch 26 was approx. 4,000; Ch 27 is approx. 13x larger)
weight tying      : True  (lm_head <-> wte shared)
fp32 weight size  : 477.5 MiB

tokenizer    : TokenizersBackend
  eos_token  : </s>  id=1
  pad_token  : <pad>  id=3

model: GPT2LMHeadModel
  - body : GPT2Model  (Decoder, causal attention)
  - head : Linear(in=768, out=51200)
```

**결과 해석**

본체가 125.16M params (Ch 26 의 약 3M 대비 약 42배), vocab 51,200 (약 13배) 로 로드됐고, LM head 가 `Linear(768, 51200)` 으로 weight tying (`lm_head ↔ wte`) 된 표준 GPT2 구조입니다. `masked_bias` UNEXPECTED 경고는 구버전 GPT2 체크포인트의 흔적이라 무시해도 되고, `pad_token` 이 `<pad>`(id=3) 로 제대로 잡혀 토크나이저 함정을 피했음을 확인할 수 있습니다.

### Ch 26 ↔ Ch 27 코드 diff — *모델·토크나이저 로드 두 줄 차이*

```python
# Ch 26 (한국어 GPT scratch) - BBPE 직접 학습 후 random init 모델
# bbpe = Tokenizer(BPE(unk_token=None))
# trainer = BpeTrainer(vocab_size=4000, ...)
# bbpe.train_from_iterator(text_iter, trainer)
# tokenizer = PreTrainedTokenizerFast(tokenizer_object=bbpe, bos_token=EOS, eos_token=EOS, pad_token=EOS)
# config = GPT2Config(vocab_size=4000, n_layer=4, n_head=4, n_embd=256, ...)
# model = GPT2LMHeadModel(config)

# Ch 27 (continual pretraining) - 단 몇 줄로 (KoGPT2 는 토크나이저 로드만 주의)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
```

> *trainer·collator·loss 는 같음* — *모델 로드 한 줄 + 토크나이저 로드* 로 학습 단계 2 (continual pretraining) 에 진입합니다. 그게 본 챕터의 메시지. 영어 Ch 25 의 `from_pretrained("gpt2")` 와 같은 구조 — 다만 **KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정** 이 있어 `PreTrainedTokenizerFast` + special token 명시가 필요합니다 (실무에서 자주 만나는 *사전학습 모델별 토크나이저 로드 주의점*).

## 토큰화 + `group_texts` — *Ch 26 과 완전히 같은 패턴*

HF causal LM 학습 표준 패턴 (`run_clm.py`) 그대로. Ch 26 과 정확히 같습니다 — *데이터·전처리·collator 는 통제 변수*.

다만 `BLOCK_SIZE` 는 Ch 26 과 동일하게 유지 (128) — *KoGPT2 본체의 `n_positions=1024` 까지 가능하지만, T4 + 30분 룰 안에서 비교 가능성 우선*.

토큰화와 `group_texts` 는 Ch 26 과 *한 글자도 다르지 않은* HF causal LM 학습 표준 패턴 (`run_clm.py`) 입니다. 전처리는 통제 변수라, 같은 30K stories 가 *KoGPT2 BBPE (vocab 51,200)* 로 토큰화되면 Ch 26 의 직접 학습 BBPE (vocab 약 4,000) 보다 토큰 수가 줄어드는 점만 관전 포인트입니다.

```python
BLOCK_SIZE = 128   # Ch 26 과 동일

def tokenize_fn(batch):
    return tokenizer(batch["text"])

tok_train = raw_train.map(tokenize_fn, batched=True, remove_columns=["text"], desc="tokenize train")
tok_val   = raw_val.map(tokenize_fn,   batched=True, remove_columns=["text"], desc="tokenize val")

# 각 story 끝에 EOS 부착 (story 경계 표시)
def add_eos(batch):
    new_ids, new_mask = [], []
    for ids in batch["input_ids"]:
        ids = ids + [tokenizer.eos_token_id]
        new_ids.append(ids)
        new_mask.append([1] * len(ids))
    return {"input_ids": new_ids, "attention_mask": new_mask}

tok_train = tok_train.map(add_eos, batched=True, desc="add eos train")
tok_val   = tok_val.map(add_eos,   batched=True, desc="add eos val")
```

**위 코드 읽기** — 각 story 를 토큰화한 뒤 `add_eos` 로 끝에 `eos_token_id` 를 붙여 *story 경계* 를 표시합니다. 이렇게 해야 다음 단계에서 story 들을 한 스트림으로 이어 붙여도 모델이 어디가 이야기 끝인지 학습할 수 있습니다.

```python
def group_texts(batch):
    concatenated = {k: sum(batch[k], []) for k in batch.keys()}
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE
    return {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }

lm_train = tok_train.map(group_texts, batched=True, desc="group train")
lm_val   = tok_val.map(group_texts,   batched=True, desc="group val")
```

**위 코드 읽기** — `group_texts` 는 batch 안 모든 토큰을 `sum(..., [])` 로 하나의 긴 스트림으로 이어 붙인 뒤 `BLOCK_SIZE`(128) 배수로 내림해 고정 길이 chunk 로 자릅니다. 가변 길이 story 가 PAD 낭비 없이 빈틈없는 학습 블록으로 바뀌는 단계입니다.

```python
print(f"\ntrain chunks: {len(lm_train):,}  (block_size={BLOCK_SIZE})")
print(f"val   chunks: {len(lm_val):,}")
print(f"approx. train tokens: {len(lm_train) * BLOCK_SIZE / 1e6:.2f} M")
print("\nfirst chunk decode (first 200 chars):")
print(tokenizer.decode(lm_train[0]["input_ids"])[:200])
```

**▶ 실행 결과**

```text
train chunks: 48,513  (block_size=128)
val   chunks: 796
approx. train tokens: 6.21 M

first chunk decode (first 200 chars):
한때 벤이라는 이름의 어린 소년이 있었어요. 벤은 주변 세계를 탐험하는 것을 좋아했답니다. 그는 가게에 전시되어 있던 아름다운 꽃병들 같은 멋진 것들을 많이 봤어요. 어느 날, 벤은 가게를 거닐다가 정말 특별한 꽃병을 발견했죠. 벤은 그 꽃병을 보고 …(뒤 60자 생략)
```

**결과 해석**

30K stories 가 48,513 chunks (block_size=128, 약 6.21M 토큰) 로 묶였습니다. 첫 chunk 를 decode 하면 원본 동화가 깨짐 없이 그대로 복원돼, KoGPT2 BBPE 의 encode→decode 왕복이 한국어에서 정상 동작함을 확인할 수 있습니다.

**비교 관전 포인트** — 같은 30K stories 가 *KoGPT2 BBPE (vocab 51,200)* 로 토큰화되면 Ch 26 의 *직접 학습 BBPE (vocab 약 4,000)* 보다 *토큰 수가 적습니다* — vocab 이 클수록 한 토큰이 더 긴 byte 시퀀스를 표현하므로. 같은 데이터의 토큰 수 차이가 *토크나이저 vocab 크기의 직접적 효과* (영어 Ch 24→25 에서 본 결의 한국어 재확인).

## 학습 *전* generation — *이미 잘 만들어진 한국어 본체* 라는 사실 확인

Ch 26 의 *random init baseline* 은 *한국어와 거리 먼 byte 조각* 이었습니다. Ch 27 의 학습 전 baseline 은 *KoGPT2 가 대규모 한국어 코퍼스로 이미 사전학습된 본체* 라 *학습 시작 시점에 이미 자연스러운 한국어 generation* 이 가능합니다 (다만 TinyStories 동화체는 아님).

같은 한국어 prompt 3개로 *KoGPT2 학습 직전 (BEFORE)* generation 을 기록 — 학습 후 (§6) 와 나란히 비교해 *continual pretraining 이 본체에 어떤 변화를 주는가* 를 직접 봅니다.

학습을 시작하기 전에, *continual pretraining 이 본체에 무엇을 바꾸는가* 를 측정할 기준선을 먼저 기록합니다. 같은 한국어 prompt 3개에 대해 KoGPT2 *그대로* 의 generation 을 저장해 두고, 학습 후 결과와 나란히 비교하기 위함입니다.

```python
PROMPTS = [
    "옛날 옛날에",
    "작은 소녀가",
    "큰 개가",
]
GEN_KWARGS = dict(max_new_tokens=60, do_sample=True, temperature=0.8, top_k=50)


@torch.no_grad()
def generate_text(active_model, prompt: str, gen_tokenizer=None, **kwargs):
    tok = gen_tokenizer if gen_tokenizer is not None else tokenizer
    enc = tok(prompt, return_tensors="pt").to(active_model.device)
    out = active_model.generate(
        **enc,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        **kwargs,
    )
    return tok.decode(out[0], skip_special_tokens=True)


torch.manual_seed(SEED)
model.eval()
before_outputs = []
print("=" * 70)
print("BEFORE continual pretraining - KoGPT2 pretrained on Korean corpus, as-is")
print("=" * 70)
for p in PROMPTS:
    text = generate_text(model, p, **GEN_KWARGS)
    before_outputs.append(text)
    print(f"\n[prompt] {p}")
    print(text)
```

**▶ 실행 결과**

```text
======================================================================
BEFORE continual pretraining - KoGPT2 pretrained on Korean corpus, as-is
======================================================================
[prompt] 옛날 옛날에
옛날 옛날에 네들이 말했잖아. 그게 뭐냐면 우리가 그~ 네들한테 이십년에 다 사십년 동안 이십년 동안 사십년 동안 살았으니까 그~ 삼십년 동안 이렇게 살았으면 뭐가 좋겠느냐 이거야. 그런
[prompt] 작은 소녀가
작은 소녀가 된 후 소녀에게 한 번도 성적으로 관심을 갖지 않았다.
그녀가 소녀가 된 후에는 그녀의 사랑을 알게 되었다.
그녀가 고등학교에 진학했을 때, 그들은 소녀를 볼 때마다 미소를 짓고 그녀를 바라봤다.
마치 그처럼 그녀는 소녀처럼 소녀를 보았다.
그때마다 소녀는 소녀를 위해 무엇인가를 해 주고 있었다.
[prompt] 큰 개가
큰 개가 되어버린 것입니다.
그래서 모든 것을 초월하여 하나로 통합하고, 하나로 통합하는 것이 바로 그 무엇이라는 것을 강조하고 있는 것입니다.
하나의 힘을 하나로 통합하는 것이 바로 통합인 것입니다.
우리는 하나님의 말씀을 통해서 하나님의 능력을 충분히 깨닫습니다.
그 능력은 하나님의 말씀을 통해서 우리에게 전달되기도 하고, 또 하나
```

**결과 해석**

학습 전 KoGPT2 는 *이미 자연스러운 한국어 문장* 을 생성하지만 *일반 도메인 풍* (구어체 / 종교 텍스트 / 산문) 이라 TinyStories 동화체와는 거리가 멉니다 — Ch 26 의 random init baseline 이 byte 조각이던 것과 달리, Ch 27 은 *random 이 아니라 잘 만들어진 본체* 에서 출발한다는 증거입니다.

**해석 가이드 — *Ch 26 random init* vs *Ch 27 KoGPT2 사전학습* 의 직전 비교**

- **Ch 26 학습 직전 (random init)**: *한국어와 거리 먼 음절·byte 조각 / 의미 없는 짧은 토큰 반복*
- **Ch 27 학습 직전 (KoGPT2 사전학습 그대로)**: *이미 자연스러운 한국어 문장* — *주어 + 서술어* 구조, 다양한 도메인 어휘. 다만 *TinyStories 동화 풍은 아님* — 일반 한국어 산문 / 뉴스 / 대화 등 (사전학습 데이터 분포 반영)

> 이 차이가 *학습 시작점의 차이*. Ch 27 은 *random 에서 시작하지 않습니다* — *이미 잘 만들어진 본체* 에서 시작해 *TinyStories 동화 풍 적응* 만 더하는 게 학습 단계 2 (continual pretraining) 의 본질. 영어 Ch 25 (gpt2 BEFORE) 의 한국어판.

## Continual Pretraining — *trainer 코드는 Ch 26 과 거의 동일*

Ch 26 과 *완전히 같은 구조* 의 `Trainer` 코드. 변하는 곳은 **lr (`5e-4 → 2e-5`)** 와 **step 설정 (max_steps → 1 epoch, 48,513 chunks / eff. batch 16 ≈ 약 3,000 step)**, 그리고 **메모리 제약 대응 (batch 4 + grad accum 4)** 입니다.

### 왜 lr 가 작아지는가 — `2e-5` 의 정확한 의미

Ch 26 (scratch) 의 lr `5e-4` 는 *random init 본체* 가 *빠르게 의미 있는 표상* 을 학습하기 위한 표준 값. Ch 27 (continual pretraining) 는 *이미 학습된 본체* 라 *큰 lr 면 사전학습된 표상이 망가질 위험* — **catastrophic forgetting**. `2e-5` 는 HF 의 continual pretraining / fine-tuning 표준 lr 중 가장 작은 쪽으로, *사전학습 표상 보존* 을 우선. (영어 Ch 25 와 같은 `2e-5`.)

### `DataCollatorForLanguageModeling(mlm=False)` — *Ch 26 과 한 글자도 다르지 않음*

학습 단계 2 의 정의: *collator 안 바뀜, loss 안 바뀜, trainer 안 바뀜*. *데이터·본체·lr 만 바뀜*.

이제 continual pretraining 본체입니다. collator·loss·trainer 는 Ch 26 과 같고, 변하는 건 *lr (`5e-4 → 2e-5`)* 와 *step 설정 (1 epoch)*, 그리고 *메모리 대응 (batch 4 + grad accum 4)* 뿐입니다. lr 이 작아진 이유는 *이미 학습된 표상이 큰 lr 에 망가지는 catastrophic forgetting* 을 피하기 위해서라는 점을 눈여겨봐야 합니다.

```python
from transformers import (DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, TrainerCallback)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

**위 코드 읽기** — `DataCollatorForLanguageModeling(mlm=False)` 는 Ch 26 과 정확히 같은 causal LM collator 로, `labels = input_ids.clone()` 을 자동으로 만들고 pad 자리만 `-100` 으로 가립니다. *거의 모든 자리가 학습 신호* 라는 점이 Ch 28 SFT 의 `labels[:prompt_len] = -100` 과 대비되는 단계 2 의 기준선입니다.

```python
args = TrainingArguments(
    output_dir="./out_ko_gpt2_continual_pretrain",
    num_train_epochs=1,                    # 본체 이미 학습됨 - 1 epoch 충분
    per_device_train_batch_size=4,         # KoGPT2 125M + T4 16GB
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,         # effective batch = 16
    learning_rate=2e-5,                    # <- Ch 26 의 5e-4 와 다른 유일한 큰 차이
    weight_decay=0.01,
    warmup_steps=0.06,                     # 1 미만이면 전체 step 대비 *비율* 로 해석 (구 warmup_ratio)
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    fp16=USE_FP16,                         # T4 는 bf16 불가
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    seed=SEED,
)
```

**위 코드 읽기** — `learning_rate=2e-5` 가 Ch 26 (`5e-4`) 대비 유일한 큰 차이이고, `batch_size=4` + `gradient_accumulation_steps=4` 로 effective batch 16 을 맞춰 125M 본체를 T4 16GB 안에서 돌립니다. `num_train_epochs=1` 인데도 chunk 가 많아 약 3,000 step 이 되고, `fp16=True` 는 bf16 불가인 T4 대응입니다.

```python
class VRAMCallback(TrainerCallback):
    '''step 별 peak VRAM 기록 (로깅 윈도우 단위로 reset). CUDA 에서만 유효.'''

    def __init__(self):
        self.steps, self.peak_MiB = [], []

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            self.steps.append(state.global_step)
            self.peak_MiB.append(peak)
            torch.cuda.reset_peak_memory_stats()


vram_cb = VRAMCallback()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=lm_train,
    eval_dataset=lm_val,
    data_collator=collator,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== continual pretraining summary ===")
print(f"elapsed       : {elapsed/60:.2f} min")
print(f"global_step   : {train_out.global_step}")
print(f"train_loss    : {train_out.training_loss:.4f}")
print(f"vocab ln (random baseline): {math.log(tokenizer.vocab_size):.4f}  (we start MUCH lower than this)")
if torch.cuda.is_available():
    print(f"final peak    : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
```

**▶ 실행 결과**

```text
[transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
Step  Training Loss  Validation Loss
100   3.043552       2.899030
200   2.798506       2.683229
300   2.712242       2.597310
400   2.669398       2.540780
500   2.612974       2.499535
600   2.595014       2.468091
700   2.548348       2.438415
800   2.489526       2.418997
900   2.492317       2.405184
1000  2.502229       2.384735
1100  2.461211       2.373174
1200  2.451401       2.358372
1300  2.439108       2.347635
1400  2.437176       2.332937
1500  2.408855       2.325852
1600  2.419213       2.315121
1700  2.418423       2.304565
1800  2.395895       2.298695
1900  2.377906       2.292478
2000  2.399462       2.284666
2100  2.351143       2.278726
2200  2.408343       2.274161
2300  2.357115       2.270545
2400  2.357828       2.266827
2500  2.378545       2.264369
2600  2.342011       2.262938
2700  2.337422       2.261098
2800  2.345940       2.260038
2900  2.349167       2.259549
3000  2.341200       2.259395
...
=== continual pretraining summary ===
elapsed       : 17.09 min
global_step   : 3033
train_loss    : 2.4862
vocab ln (random baseline): 10.8435  (we start MUCH lower than this)
final peak    : 1455 MiB
```

**결과 해석**

T4 에서 약 17분, 3,033 step 만에 누적 평균 `train_loss` 가 2.49 로 끝났습니다. random baseline `ln(51200) ≈ 10.84` 보다 *훨씬 낮은 지점에서 시작해 더 낮아진* 것이 사전학습된 본체의 시작 이점이고, peak VRAM 1,455 MiB 로 T4 16GB 에 여유 있게 들어갑니다.

학습 로그에서 train/eval loss 곡선과 step별 peak VRAM 을 함께 그려, *어디서 시작해 어디까지 내려갔는지* 와 *메모리 여유* 를 한눈에 확인합니다.

```python
# loss curve + VRAM trace
log = trainer.state.log_history
train_pts = [(r["step"], r["loss"]) for r in log if "loss" in r and "eval_loss" not in r]
eval_pts  = [(r["step"], r["eval_loss"]) for r in log if "eval_loss" in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# loss
ax1.plot([s for s, _ in train_pts], [l for _, l in train_pts], "-",
         color="tab:blue", alpha=0.6, label="train")
if eval_pts:
    ax1.plot([s for s, _ in eval_pts], [l for _, l in eval_pts], "s-",
             color="tab:red", label="eval")
ax1.axhline(math.log(tokenizer.vocab_size), ls=":", color="gray",
            label=f"무작위 추측 기준선 = ln({tokenizer.vocab_size}) approx. {math.log(tokenizer.vocab_size):.2f}")
ax1.set_xlabel("step"); ax1.set_ylabel("cross-entropy loss")
ax1.set_title("KoGPT2 이어서 사전학습 (TinyStories-Korean) - loss")
ax1.grid(True, alpha=0.3); ax1.legend()

# VRAM (CUDA 만)
if vram_cb.steps:
    ax2.plot(vram_cb.steps, vram_cb.peak_MiB, "o-", color="tab:green",
             label="최대 VRAM (로그 구간별)")
    ax2.set_title(f"VRAM trace  (bs=4, grad_accum=4, fp16, n_pos={BLOCK_SIZE})")
else:
    ax2.text(0.5, 0.5, "VRAM 추적은 CUDA 에서만 가능",
             ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("VRAM 추적 - CUDA 전용")
ax2.set_xlabel("step"); ax2.set_ylabel("VRAM (MiB)")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()
```

**▶ 실행 결과**

![output](../assets/27-ko_gpt2_continual_pretrain-out1-1.png)

**결과 해석**

loss 곡선이 random baseline 점선 (`ln(51200) ≈ 10.84`) 보다 한참 아래에서 시작해 약 2.5 부근까지 완만히 내려가는 모양 — Ch 26 이 random baseline 에서 급강하하던 것과 달리, *이미 낮은 지점에서 출발해 천천히 도메인 적응* 하는 단계 2 의 전형적 곡선입니다. VRAM 도 학습 내내 1.5GiB 안쪽으로 안정적입니다.

**관전 포인트** — Ch 26 과 달리 *첫 step loss 가 random baseline `ln(51200) ≈ 10.84` 부근이 아니라 약 3.0-4.0 부근* 에서 시작합니다. *KoGPT2 가 이미 일반 한국어 분포를 학습해 둔 덕분에 TinyStories 평가에서도 시작 loss 가 낮음*. 학습 진행과 함께 약 2.0-2.5 로 더 떨어지는데, 이게 *TinyStories 동화 도메인 적응* 의 효과. 곡선이 *random baseline 으로부터 빠르게 떨어지는 Ch 26* vs *이미 낮은 지점에서 시작해 천천히 더 떨어지는 Ch 27* 의 모양 차이가 한눈에 보입니다 (영어 Ch 24→25 와 같은 결).

## 학습 *후* generation — *continual pretraining 의 효과*

같은 `PROMPTS / GEN_KWARGS` 로 학습 후 모델에서 다시 생성. *BEFORE (KoGPT2 그대로) → AFTER (continual pretrained on TinyStories-Korean)* 비교가 *학습 단계 2 가 본체에 새긴 도메인 적응* 을 직접 드러냅니다.

학습이 끝난 모델에 *같은 prompt·GEN_KWARGS* 로 다시 생성해, BEFORE 와 무엇이 달라졌는지 봅니다. seed 도 동일하게 고정해 *데이터 1 epoch 의 차이* 만 비교에 남깁니다.

```python
torch.manual_seed(SEED)
model.eval()
after_outputs = []
print("=" * 70)
print("AFTER continual pretraining - KoGPT2 + TinyStories-Korean 30K")
print("=" * 70)
for p in PROMPTS:
    text = generate_text(model, p, **GEN_KWARGS)
    after_outputs.append(text)
    print(f"\n[prompt] {p}")
    print(text)
```

**▶ 실행 결과**

```text
======================================================================
AFTER continual pretraining - KoGPT2 + TinyStories-Korean 30K
======================================================================
[prompt] 옛날 옛날에
옛날 옛날에 릴리라는 이름의 작은 소녀가 있었어요. 그녀는 친구들과 함께 큰 공원에 가는 것을 좋아했지요. 어느 날, 공원에 큰 개가 서 있는 것을 봤어요. 릴리는 그 개를 보고 이렇게 말했죠, "개야, 내가 도와줄게." 개는 꼬리를 흔들며 짖었고, 릴리는
[prompt] 작은 소녀가
작은 소녀가 있었어요. 그녀는 예쁜 드레스를 가지고 있었고, 정말 예뻤답니다. 릴리는 그 드레스를 입는 걸 정말 좋아했지요. 어느 날, 그들은 공원으로 갔어요. 그때 릴리는 엄마를 보았어요. 엄마는 릴리를 위해 예쁜 드레스를 입었지요. 릴리는 정말 행복했어
[prompt] 큰 개가
큰 개가 친구들에게 말했죠, "걱정 마, 작은 개야. 우리는 너를 돕고 싶어, 작은 개야." 그들은 공을 되찾으려고 해요. 그들은 계속 시도해요. 그들은 공을 잡으려고 해요. 하지만 공이 너무 커서 쉽지 않아요. 그들은 미끄러져 넘어지고 말죠.
```

**결과 해석**

같은 prompt 가 이제 *동화 풍* 으로 이어집니다 — `릴리`, `작은 소녀`, `공원`, `친구`, `예쁜 드레스` 같은 TinyStories 어휘와 짧고 단순한 `…했어요` 문장으로 톤이 바뀌었습니다. 본체는 그대로 125M 이고 *lr 한 숫자 + 1 epoch 데이터* 만으로 generation 의 도메인이 이동한 것이 continual pretraining 의 효과입니다.

앞서 저장한 `before_outputs` 와 방금 얻은 `after_outputs` 를 prompt별로 나란히 출력해, *같은 모델·같은 prompt* 에서 학습 전후가 어떻게 갈리는지 직접 대조합니다.

```python
# Ch 27 within-model BEFORE vs AFTER comparison
print("=" * 78)
print("Ch 27 BEFORE (KoGPT2 as-is) vs AFTER (KoGPT2 + TinyStories-Korean continual pretrain)")
print("=" * 78)
for p, before, after in zip(PROMPTS, before_outputs, after_outputs):
    print(f"\nPROMPT  : {p}")
    print("-" * 78)
    print(f"BEFORE  : {before[len(p):].strip()[:280]}")
    print(f"AFTER   : {after[len(p):].strip()[:280]}")
```

**▶ 실행 결과**

```text
==============================================================================
Ch 27 BEFORE (KoGPT2 as-is) vs AFTER (KoGPT2 + TinyStories-Korean continual pretrain)
==============================================================================

PROMPT  : 옛날 옛날에
------------------------------------------------------------------------------
BEFORE  : 네들이 말했잖아. 그게 뭐냐면 우리가 그~ 네들한테 이십년에 다 사십년 동안 이십년 동안 사십년 동안 살았으니까 그~ 삼십년 동안 이렇게 살았으면 뭐가 좋겠느냐 이거야. 그런
AFTER   : 릴리라는 이름의 작은 소녀가 있었어요. 그녀는 친구들과 함께 큰 공원에 가는 것을 좋아했지요. 어느 날, 공원에 큰 개가 서 있는 것을 봤어요. 릴리는 그 개를 보고 이렇게 말했죠, "개야, 내가 도와줄게." 개는 꼬리를 흔들며 짖었고, 릴리는

PROMPT  : 작은 소녀가
------------------------------------------------------------------------------
BEFORE  : 된 후 소녀에게 한 번도 성적으로 관심을 갖지 않았다.
그녀가 소녀가 된 후에는 그녀의 사랑을 알게 되었다.
그녀가 고등학교에 진학했을 때, 그들은 소녀를 볼 때마다 미소를 짓고 그녀를 바라봤다.
마치 그처럼 그녀는 소녀처럼 소녀를 보았다.
그때마다 소녀는 소녀를 위해 무엇인가를 해 주고 있었다.
AFTER   : 있었어요. 그녀는 예쁜 드레스를 가지고 있었고, 정말 예뻤답니다. 릴리는 그 드레스를 입는 걸 정말 좋아했지요. 어느 날, 그들은 공원으로 갔어요. 그때 릴리는 엄마를 보았어요. 엄마는 릴리를 위해 예쁜 드레스를 입었지요. 릴리는 정말 행복했어

PROMPT  : 큰 개가
------------------------------------------------------------------------------
BEFORE  : 되어버린 것입니다.
그래서 모든 것을 초월하여 하나로 통합하고, 하나로 통합하는 것이 바로 그 무엇이라는 것을 강조하고 있는 것입니다.
하나의 힘을 하나로 통합하는 것이 바로 통합인 것입니다.
우리는 하나님의 말씀을 통해서 하나님의 능력을 충분히 깨닫습니다.
그 능력은 하나님의 말씀을 통해서 우리에게 전달되기도 하고, 또 하나
AFTER   : 친구들에게 말했죠, "걱정 마, 작은 개야. 우리는 너를 돕고 싶어, 작은 개야." 그들은 공을 되찾으려고 해요. 그들은 계속 시도해요. 그들은 공을 잡으려고 해요. 하지만 공이 너무 커서 쉽지 않아요. 그들은 미끄러져 넘어지고 말죠.
```

**결과 해석**

세 prompt 모두 BEFORE 는 구어체·종교·산문 등 *일반 도메인* 으로 흩어졌지만 AFTER 는 일관되게 *동화체* 로 수렴합니다. head·task·loss 가 그대로인데 출력 분포만 새 데이터 쪽으로 옮겨간 것이 *task adaptation 이 아닌 데이터 적응* (continual pretraining) 임을 보여줍니다.

**해석 가이드 — continual pretraining 의 도메인 적응 효과**

- **BEFORE (KoGPT2 그대로)**: 자연스러운 한국어이지만 *일반 도메인 풍* — 산문 / 뉴스 / 대화 톤. *옛날 옛날에* 같은 동화 도입에 대해서도 *동화 스타일 이어쓰기보다 일반 산문 이어쓰기* 경향
- **AFTER (KoGPT2 + TinyStories-Korean 1 epoch)**: 같은 prompt 가 *동화 풍* 으로 이어짐 — 짧고 단순한 문장, 동화 어휘 (소녀 / 엄마 / 친구 / 숲 / 행복 / 토끼 ...), TinyStories 특유의 *반복적이고 어린이 어휘 한정* 톤

> 본체는 *같은 125M params 모델* 이고, *한 줄 코드 차이 (lr) + 한 epoch 의 데이터* 만으로 *generation 톤 자체가 도메인 적응*. 그게 *continual pretraining 의 정량적 가치* — *task adaptation 의미의 fine-tune (head 교체 / 새 loss) 이 아닙니다*, *같은 task 의 데이터만 바뀐 단계 1 의 연장*. 영어 Ch 25 (gpt2 AFTER) 의 한국어 재확인.

## 3-way generation 비교 — Ch 26 (scratch) vs Ch 27 BEFORE vs Ch 27 AFTER

Ch 26 의 *작은 from-scratch 모델* (약 3M, 한국어 TinyStories 1500 step) 의 generation 결과를 *옆에 두고* 비교합니다. *Ch 26 노트북 §7 의 "TRAINED model" generation 출력* 을 직접 인용 (사용자가 본인 결과로 갱신 가능).

### 세 셋업의 차이

| 셋업 | 본체 | 사전학습 | TinyStories 학습 |
|---|---|---|---|
| Ch 26 (scratch) | 약 3M params, random init | 없음 (from scratch) | 1500 step 사전학습 자체 |
| **Ch 27 BEFORE** | 125M params (KoGPT2) | **대규모 한국어 코퍼스** | 없음 (KoGPT2 그대로) |
| **Ch 27 AFTER** | 125M params (KoGPT2) | **대규모 한국어 코퍼스** | **1 epoch continual pretraining** |

마지막으로 Ch 26 의 *3M scratch* 모델 generation 을 옆에 두어 *3-way 비교* 를 만듭니다. `ch26_outputs` 는 Ch 26 노트북 §7 의 출력을 인용한 것으로, 본인 결과로 갱신하면 비교가 더 정확해집니다.

```python
# Ch 26 의 TRAINED model generation 결과 인용
# (Ch 26 노트북 §7 "TRAINED model" 출력에서 본인 결과로 갱신하시면 비교가 정확해집니다)
ch26_outputs = {
    "옛날 옛날에": (
        "옛날 옛날에 작은 소녀가 살았어요. 소녀는 숲에서 친구들과 놀았어요. "
        "어느 날 소녀는 큰 토끼를 만났어요. 토끼는 소녀에게 인사했어요."
    ),
    "작은 소녀가": (
        "작은 소녀가 엄마와 공원에 갔어요. 소녀는 꽃을 보고 행복했어요. "
        "엄마는 소녀에게 웃어주었어요. 둘은 함께 집으로 돌아갔어요."
    ),
    "큰 개가": (
        "큰 개가 마당에서 뛰어놀았어요. 개는 공을 가지고 놀았어요. "
        "한 아이가 와서 개와 함께 놀았어요. 개는 꼬리를 흔들며 좋아했어요."
    ),
}

print("=" * 80)
print("3-way comparison: Ch 26 (3M scratch) vs Ch 27 BEFORE vs Ch 27 AFTER")
print("=" * 80)
for p, before, after in zip(PROMPTS, before_outputs, after_outputs):
    ch26_text = ch26_outputs.get(p, "(Ch 26 result not recorded for this prompt)")
    print(f"\nPROMPT          : {p}")
    print("-" * 80)
    print(f"Ch 26 (scratch) : {ch26_text[:240]}")
    print(f"Ch 27 BEFORE    : {before[len(p):].strip()[:240]}")
    print(f"Ch 27 AFTER     : {after[len(p):].strip()[:240]}")
```

**▶ 실행 결과**

```text
================================================================================
3-way comparison: Ch 26 (3M scratch) vs Ch 27 BEFORE vs Ch 27 AFTER
================================================================================

PROMPT          : 옛날 옛날에
--------------------------------------------------------------------------------
Ch 26 (scratch) : 옛날 옛날에 작은 소녀가 살았어요. 소녀는 숲에서 친구들과 놀았어요. 어느 날 소녀는 큰 토끼를 만났어요. 토끼는 소녀에게 인사했어요.
Ch 27 BEFORE    : 네들이 말했잖아. 그게 뭐냐면 우리가 그~ 네들한테 이십년에 다 사십년 동안 이십년 동안 사십년 동안 살았으니까 그~ 삼십년 동안 이렇게 살았으면 뭐가 좋겠느냐 이거야. 그런
Ch 27 AFTER     : 릴리라는 이름의 작은 소녀가 있었어요. 그녀는 친구들과 함께 큰 공원에 가는 것을 좋아했지요. 어느 날, 공원에 큰 개가 서 있는 것을 봤어요. 릴리는 그 개를 보고 이렇게 말했죠, "개야, 내가 도와줄게." 개는 꼬리를 흔들며 짖었고, 릴리는

PROMPT          : 작은 소녀가
--------------------------------------------------------------------------------
Ch 26 (scratch) : 작은 소녀가 엄마와 공원에 갔어요. 소녀는 꽃을 보고 행복했어요. 엄마는 소녀에게 웃어주었어요. 둘은 함께 집으로 돌아갔어요.
Ch 27 BEFORE    : 된 후 소녀에게 한 번도 성적으로 관심을 갖지 않았다.
그녀가 소녀가 된 후에는 그녀의 사랑을 알게 되었다.
그녀가 고등학교에 진학했을 때, 그들은 소녀를 볼 때마다 미소를 짓고 그녀를 바라봤다.
마치 그처럼 그녀는 소녀처럼 소녀를 보았다.
그때마다 소녀는 소녀를 위해 무엇인가를 해 주고 있었다.
Ch 27 AFTER     : 있었어요. 그녀는 예쁜 드레스를 가지고 있었고, 정말 예뻤답니다. 릴리는 그 드레스를 입는 걸 정말 좋아했지요. 어느 날, 그들은 공원으로 갔어요. 그때 릴리는 엄마를 보았어요. 엄마는 릴리를 위해 예쁜 드레스를 입었지요. 릴리는 정말 행복했어

PROMPT          : 큰 개가
--------------------------------------------------------------------------------
Ch 26 (scratch) : 큰 개가 마당에서 뛰어놀았어요. 개는 공을 가지고 놀았어요. 한 아이가 와서 개와 함께 놀았어요. 개는 꼬리를 흔들며 좋아했어요.
Ch 27 BEFORE    : 되어버린 것입니다.
그래서 모든 것을 초월하여 하나로 통합하고, 하나로 통합하는 것이 바로 그 무엇이라는 것을 강조하고 있는 것입니다.
하나의 힘을 하나로 통합하는 것이 바로 통합인 것입니다.
우리는 하나님의 말씀을 통해서 하나님의 능력을 충분히 깨닫습니다.
그 능력은 하나님의 말씀을 통해서 우리에게 전달되기도 하고, 또 하나
Ch 27 AFTER     : 친구들에게 말했죠, "걱정 마, 작은 개야. 우리는 너를 돕고 싶어, 작은 개야." 그들은 공을 되찾으려고 해요. 그들은 계속 시도해요. 그들은 공을 잡으려고 해요. 하지만 공이 너무 커서 쉽지 않아요. 그들은 미끄러져 넘어지고 말죠.
```

**결과 해석**

Ch 26 (3M scratch) 도 동화체 한국어는 만들지만 어휘가 단조롭고, Ch 27 BEFORE 는 어휘 폭은 넓되 동화체가 아니며, Ch 27 AFTER 는 *동화체 + 넓은 어휘력* 을 모두 갖춥니다. 다만 이 격차가 *모델 크기 (3M→125M)* 때문인지 *사전학습* 때문인지는 두 변수가 함께 변해 분리되지 않습니다 (FAQ Q3 참고).

**해석 가이드 — 세 셋업의 격차**

- **Ch 26 (3M scratch, 한국어 TinyStories 1500 step)**: *동화 풍 단순 한국어* 가능 — 작은 모델·작은 데이터로도 말이 되는 한국어 생성. 다만 어휘는 동화 도메인에 한정, 번역체 데이터라 다소 어색할 수 있음
- **Ch 27 BEFORE (KoGPT2 그대로)**: *다양한 도메인 한국어* 가능. 자연스러운 산문이지만 *TinyStories 동화 풍은 아님*
- **Ch 27 AFTER (KoGPT2 + TinyStories-Korean continual pretrain)**: *동화 풍 + 자연스러움 + 일반 도메인 어휘력* 결합. *작은 from-scratch 의 도메인 특화 + 큰 사전학습 모델의 어휘 폭* 이 모두

> **세 셋업의 비교가 던지는 질문** — Ch 27 AFTER 가 Ch 26 보다 *훨씬 좋아 보인다면*, 이게 *모델 크기 (3M → 125M, 약 40배) 의 위력인가, 사전학습 (대규모 한국어 코퍼스) 의 위력인가?* — 본 챕터의 셋업으로는 *분리 불가능*. 두 요인이 *함께 변함*. FAQ Q3 에서 더 자세히 (영어 Ch 25 Q3 의 한국어판).

## 학습 곡선 비교 — Ch 26 vs Ch 27 의 학습 효율

*같은 데이터 (한국어 TinyStories 30K)* 에 대한 *random init vs 사전학습 본체* 의 학습 효율 격차를 표로 정리.

| 항목 | Ch 26 (3M scratch) | **Ch 27 (125M continual pretrain)** |
|---|---|---|
| 시작 loss | 약 8.29 (`ln(4000)`, random baseline) | **약 3.0-4.0** (KoGPT2 pretrained, TinyStories 평가) |
| 도달 loss (학습 끝, 누적 평균 `train_loss`) | 약 4.5 | **약 2.49** |
| 학습 step | 1,500 | **약 3,000** (1 epoch, 48,513 chunks / eff. batch 16) |
| 학습 시간 (T4) | 약 1분 | **약 17분** |
| Vocab 차원 | 약 4,000 | **51,200** (loss 단위 다름 — 직접 비교 어려움) |
| Generation 품질 | 동화 풍 단순 한국어 | **자연스러운 동화 + 일반 도메인 어휘** |

> **요점**: Ch 27 은 *시작부터 낮은 loss* 에서 출발해 더 낮은 지점까지 내려갑니다 — 사전학습된 본체의 *시작 이점*. step·시간은 Ch 26 보다 오히려 더 큽니다 (125M 본체 + chunk 수가 많아 1 epoch 이 길어짐). 다만 *loss 의 절대값* 은 vocab 단위가 달라 직접 비교 어려움 (vocab 약 13배 차이). *Generation 품질* 로는 §7 의 3-way 비교가 정성적 차이를 보여줍니다.

> Ch 27 의 결과만 보면 *대규모 사전학습 + continual pretraining* 이 압도적으로 보이지만, *3M params + 대규모 사전학습* (가상의 비교군) 이라면 어떻게 될까요 — *모델 크기와 사전학습 데이터를 분리하는 비교* 는 본 챕터의 셋업으로는 어렵습니다. 그게 *실험 설계의 한계* 이자 *학습 단계 2 의 실용성* — 실무는 보통 *큰 사전학습 모델을 그대로 가져와 continual pretraining* 하는 게 비용 대비 최선이라 (영어 Ch 25 §8 의 한국어판).
