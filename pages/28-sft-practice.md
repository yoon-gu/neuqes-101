> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/28_sft/28_sft.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

`trl` (Transformer Reinforcement Learning) 라이브러리가 이번 챕터에 새로 등장합니다 — `SFTTrainer` 와 SFT 용 데이터 collator 를 제공. `transformers` / `datasets` / `accelerate` 와 함께 설치합니다.

```python
%pip install -q -U trl transformers tokenizers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 825.1/825.1 kB 22.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 11.1/11.2 MB 198.4 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 111.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 49.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 38.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━ 19.0/48.9 MB 232.1 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 155.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 155.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 155.5 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 16.1 MB/s eta 0:00:00
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

import trl
print(f"trl          : {trl.__version__}")

# device 자동 감지 - Colab T4 / 로컬 MPS / CPU 모두 지원
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device       : cuda  ({device_name})")
    print(f"VRAM total   : {vram_gib:.2f} GiB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("device       : mps  (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("device       : cpu  (training will be very slow - Colab T4 recommended)")

print(f"torch        : {torch.__version__}")

# 재현성
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# fp16 은 CUDA 에서만 (MPS 는 미지원, CPU 는 의미 없음)
USE_FP16 = (device.type == "cuda")
print(f"use fp16     : {USE_FP16}")

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
trl          : 1.6.0
device       : cuda  (Tesla T4)
VRAM total   : 14.56 GiB
torch        : 2.11.0+cu128
use fp16     : True
```

## KoAlpaca instruction 데이터 로드 + 포맷

**`beomi/KoAlpaca-v1.1a`** — 한국어 instruction tuning 데이터셋. 각 샘플은 `instruction` (지시) 과 `output` (응답) 필드를 가집니다 (`url` 필드는 출처 — 학습에 사용 안 함). T4 + 30분 룰 안에서 **약 3,000 샘플** 만 subset 으로 사용합니다.

KoGPT2 는 chat template 이 없으니 *직접 포맷* — `### 명령어:\n{instruction}\n\n### 응답:\n{output}`. 여기서 **`### 응답:\n` 가 response_template** (답변 시작 경계).

KoAlpaca 데이터셋을 불러와 instruction-response 쌍의 구조를 먼저 눈으로 확인합니다. 빈 샘플을 거른 뒤 3,000 개만 추려 T4 + 30분 룰 안에서 학습할 수 있게 줄이는데, `MAX_CHARS` 로 답변 길이를 통제해 평균 시퀀스 길이가 들쭉날쭉해지지 않도록 잡아 둡니다.

```python
from datasets import load_dataset

N_SFT = 3000          # T4 + 30분 룰 - subset
MAX_CHARS = 600       # 너무 긴 답변은 잘라 평균 길이 통제 (학습 안정 + 속도)

raw = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
print("raw dataset:", raw)
print("\nfields:", raw.column_names)
print("\n=== sample 0 ===")
ex0 = raw[0]
print("instruction:", ex0["instruction"][:200])
print("output     :", ex0["output"][:200])

# instruction / output 모두 비어있지 않은 샘플만, 길이 통제 후 subset
def keep(ex):
    return bool(ex["instruction"].strip()) and bool(ex["output"].strip())

raw = raw.filter(keep)
raw = raw.shuffle(seed=SEED).select(range(min(N_SFT, len(raw))))
print(f"\nafter filter + subset: {len(raw):,} samples")
```

**▶ 실행 결과**

```text
raw dataset: Dataset({
    features: ['instruction', 'output', 'url'],
    num_rows: 21155
})

fields: ['instruction', 'output', 'url']

=== sample 0 ===
instruction: 양파는 어떤 식물 부위인가요? 그리고 고구마는 뿌리인가요?
output     : 양파는 잎이 아닌 식물의 줄기 부분입니다. 고구마는 식물의 뿌리 부분입니다. 

식물의 부위의 구분에 대해 궁금해하는 분이라면 분명 이 질문에 대한 답을 찾고 있을 것입니다. 양파는 잎이 아닌 줄기 부분입니다. 고구마는 다른 질문과 답변에서 언급된 것과 같이 뿌리 부분입니다. 따라서, 양파는 식물의 줄기 부분이 되고, 고구마는 식물의 뿌리 부분입니다.
after filter + subset: 3,000 samples
```

**결과 해석**

원본 KoAlpaca 는 21,155 행이지만 필터 + subset 으로 정확히 3,000 샘플만 남았습니다. sample 0 처럼 각 행이 `instruction` (지시) 과 `output` (응답) 한 쌍으로 이뤄져 있어, 이 형식 자체가 Ch 27 의 연속 텍스트와 갈라지는 첫 변경점입니다.

### 포맷 함수 — `prompt` / `completion` 두 컬럼으로

`trl.SFTTrainer` 는 *`prompt` + `completion` 두 컬럼* 형식을 받으면 *completion (답변) 부분만 자동으로 학습 대상* 으로 잡습니다 (`completion_only_loss=True`). 그래서 우리는 instruction 을 prompt 쪽에, output 을 completion 쪽에 넣되, **response_template `### 응답:\n` 까지를 prompt 에 포함** 시켜 *답변 시작 경계* 를 명확히 합니다.

`### 응답:\n` 를 response_template 로 고정하고, instruction 을 그 앞 prompt 에, output 을 그 뒤 completion 에 배치하는 포맷 함수를 정의합니다. response_template 까지를 prompt 에 포함시키는 것이 핵심으로, collator 가 이 경계를 기준으로 답변 시작점을 잡기 때문입니다.

```python
RESPONSE_TEMPLATE = "### 응답:\n"   # 이 뒤부터가 '답변' (학습 대상)


def build_prompt(instruction: str) -> str:
    '''KoGPT2 용 instruction 포맷. RESPONSE_TEMPLATE 로 끝나 답변 경계를 명시.'''
    return f"### 명령어:\n{instruction}\n\n{RESPONSE_TEMPLATE}"


def to_prompt_completion(ex):
    output = ex["output"].strip()
    if len(output) > MAX_CHARS:
        output = output[:MAX_CHARS]
    return {
        "prompt": build_prompt(ex["instruction"].strip()),
        "completion": output,
    }


sft_ds = raw.map(to_prompt_completion, remove_columns=raw.column_names, desc="format")
print("formatted dataset:", sft_ds)
print("\n=== formatted sample 0 ===")
print("--- prompt ---")
print(sft_ds[0]["prompt"])
print("--- completion ---")
print(sft_ds[0]["completion"][:200])
```

**▶ 실행 결과**

```text
formatted dataset: Dataset({
    features: ['prompt', 'completion'],
    num_rows: 3000
})

=== formatted sample 0 ===
--- prompt ---
### 명령어:
나무가 말라 죽을 때 왜 속부터 썩는 걸까요? 그리고, 나무 속에 전선을 넣을 수 있는 방법이 있을까요?

### 응답:

--- completion ---
나무 내부에 있는 심재는 죽은 세포들로 이루어져 있습니다. 이 부분은 변재와는 달리 생명력이 없기 때문에 균에 저항력이 떨어져 있습니다. 그래서 변재와는 달리 균에 대항하기 어렵습니다. 일단 균이 침입해서 번져나가면, 막을 수 있는 방법이 없어서 썩어 …(뒤 60자 생략)
```

**결과 해석**

데이터셋이 `prompt` / `completion` 두 컬럼으로 재편됐고, prompt 가 `### 응답:\n` 으로 정확히 끝나는 것을 sample 0 출력에서 볼 수 있습니다. 이 두 컬럼 형식은 뒤에서 `SFTTrainer` 가 completion 부분만 자동으로 학습 대상으로 잡게 하는 입력 규약입니다.

## KoGPT2 토크나이저·모델 로드 — *Ch 27 과 동일한 본체*

본 챕터의 본체는 *Ch 27 과 완전히 같은 KoGPT2*. 토크나이저도 같은 방식 (`PreTrainedTokenizerFast` + special token 명시 — `AutoTokenizer` 함정 회피). encode → decode 왕복으로 한국어가 깨지지 않는지 한 줄 검증합니다.

KoGPT2 본체와 토크나이저를 Ch 27 과 같은 방식으로 로드합니다. KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 토크나이저로 잘못 fallback 하는 함정이 있어, `PreTrainedTokenizerFast` 로 special token 을 직접 지정해 불러오고 encode → decode 왕복으로 한국어가 깨지지 않는지 한 줄 검증합니다.

```python
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

t0 = time.time()
# 주의: KoGPT2 는 AutoTokenizer 가 영어 GPT2 토크나이저로 잘못 fallback 합니다.
# SKT 공식 방식대로 PreTrainedTokenizerFast 로 special token 을 직접 지정해 로드.
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)

model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2").to(device)
model.config.pad_token_id = tokenizer.pad_token_id
print(f"load done: {time.time()-t0:.1f}s")

# encode -> decode 왕복 검증 (한국어 깨짐 방지)
probe = "옛날 옛날에 작은 소녀가"
roundtrip = tokenizer.decode(tokenizer(probe)["input_ids"])
print(f"\nroundtrip check: {roundtrip!r}  ({'OK' if roundtrip == probe else 'BROKEN'})")

n_params = model.num_parameters()
print(f"\n=== model ===")
print(f"#params      : {n_params/1e6:.2f} M  (same body as Ch 27)")
print(f"vocab_size   : {tokenizer.vocab_size:,}")
print(f"tokenizer    : {type(tokenizer).__name__}")
print(f"  eos_token  : {tokenizer.eos_token}  id={tokenizer.eos_token_id}")
print(f"  pad_token  : {tokenizer.pad_token}  id={tokenizer.pad_token_id}")
```

**▶ 실행 결과**

```text
[transformers] GPT2LMHeadModel LOAD REPORT from: skt/kogpt2-base-v2
Key                                     | Status     |  | 
----------------------------------------+------------+--+-
transformer.h.{0...11}.attn.masked_bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
load done: 7.2s

roundtrip check: '옛날 옛날에 작은 소녀가'  (OK)

=== model ===
#params      : 125.16 M  (same body as Ch 27)
vocab_size   : 51,200
tokenizer    : TokenizersBackend
  eos_token  : </s>  id=1
  pad_token  : <pad>  id=3
```

**결과 해석**

roundtrip check 가 `OK` 로 나와 한국어가 토큰화·복원 과정에서 깨지지 않음을 확인했고, 파라미터 수 125.16M·vocab 51,200 으로 Ch 27 과 완전히 같은 본체임이 드러납니다. 즉 바뀐 것은 본체가 아니라 데이터 형식과 마스킹 자리뿐입니다.
