> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/31_grpo/31_grpo.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

`trl` 의 **`GRPOTrainer`** 와 **`GRPOConfig`**, 그리고 **`reward_funcs`** (verifier 함수) 가 이번 챕터에 새로 등장합니다. `transformers` / `datasets` / `accelerate` 와 함께 설치합니다.

> ⚠️ `trl` 은 버전마다 `GRPOTrainer` / `GRPOConfig` API 변동이 큽니다 (인자 이름이 버전에 따라 바뀝니다 — 예: `max_completion_length` 는 있지만 `max_prompt_length` 는 버전에 따라 없음). 본 노트북은 설치된 `trl` 버전을 셋업 셀에서 출력하고, *버전 간 안정적인 핵심 경로* (`num_generations` + `reward_funcs` + `max_completion_length` + `prompt` 컬럼) 만 사용합니다.

```python
%pip install -q -U trl transformers tokenizers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 825.1/825.1 kB 21.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━ 6.2/11.2 MB 185.6 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 120.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 48.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 39.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━ 26.9/48.9 MB 179.3 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 215.2 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 215.2 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 19.4 MB/s eta 0:00:00
```

```python
import warnings
warnings.filterwarnings("ignore")

import math
import os
import random
import re
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

## verifiable 데이터 — `prompt` + 정답 (산술)

GRPO 데이터의 핵심은 **정답을 자동 검증할 수 있어야** 한다는 것입니다. 코드(테스트 실행) 는 무겁고 환경 의존이 크니, 본 챕터는 *가장 깨끗한 verifiable task* 인 **산술(arithmetic)** 로 시작합니다 — 정답이 *정수 하나* 라 *문자열 매칭만으로 채점* 됩니다.

각 샘플은 `(prompt, answer)` 두 컬럼입니다:
- `prompt`: 풀어야 할 문제 (예: `"3 + 5 = ?"`) — 모델에 입력
- `answer`: 정답 (예: `"8"`) — *verifier 가 채점할 때만* 사용 (모델 입력 아님)

> 합성 산술이라 *정답을 우리가 알고* 있으니, *verifier (정답 매칭) 가 완벽* 합니다. 이것이 verifiable reward 의 이상적 형태 — *reward 가 잡음 없이 정확*. (GSM8K 같은 실제 수학 데이터셋도 같은 방식이지만, 답 추출이 더 까다롭습니다 — FAQ 참고.)

GRPO 는 verifier 가 자동으로 채점할 수 있는 task 가 필요합니다. 정답이 명확한 산술 문제를 만들어, `prompt` (모델 입력) 와 `answer` (채점용 정답) 를 짝지어 둡니다. prompt 는 Ch 28 SFT 와 동일한 instruction 포맷으로 감싸 학습·추론 포맷을 맞춥니다.

```python
from datasets import Dataset

# Ch 28 SFT / Ch 30 DPO 와 같은 instruction 포맷으로 prompt 를 감쌉니다.
RESPONSE_TEMPLATE = "### 응답:\n"


def build_prompt(question: str) -> str:
    '''Ch 28 SFT 와 동일한 instruction 포맷 (학습·추론 포맷 일치).'''
    return f"### 명령어:\n{question}\n\n{RESPONSE_TEMPLATE}"


def make_arithmetic(n: int, max_operand: int = 9, seed: int = 0):
    '''산술 prompt + 정답. verifier 가 정답을 자동 검증할 수 있는 task.'''
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.randint(1, max_operand)
        b = rng.randint(1, max_operand)
        op = rng.choice(["+", "-"])
        ans = a + b if op == "+" else a - b
        rows.append({
            "prompt": build_prompt(f"{a} {op} {b} = ?"),
            "answer": str(ans),          # 정답 (verifier 채점용)
        })
    return Dataset.from_list(rows)


N_TRAIN = 256       # T4 + 30분 룰 - rollout 이 무거우니 작게
grpo_ds = make_arithmetic(N_TRAIN, max_operand=9, seed=SEED)
eval_ds = make_arithmetic(64, max_operand=9, seed=SEED + 1)   # 전·후 비교용

print(f"train: {len(grpo_ds)} samples,  eval: {len(eval_ds)} samples")
print("\n=== sample 0 ===")
print("--- prompt (model input) ---")
print(grpo_ds[0]["prompt"])
print("--- answer (for verifier scoring, not model input) ---")
print(grpo_ds[0]["answer"])
```

**▶ 실행 결과**

```text
train: 256 samples,  eval: 64 samples

=== sample 0 ===
--- prompt (model input) ---
### 명령어:
7 + 7 = ?

### 응답:

--- answer (for verifier scoring, not model input) ---
14
```

## SFT 모델 (policy) 로드

GRPO 는 *SFT 모델에서 출발* 합니다 (Ch 28 의 SFT 체크포인트가 정석). 노트북 단독 실행을 위해 **base KoGPT2 로 시작** 합니다 — 보통은 *이미 지시를 따르는 SFT 모델* 에서 GRPO 를 시작해야 *rollout 이 의미 있는 답* 을 내고 verifier 가 *섞인 reward* (잘한 답 + 못한 답) 를 줄 수 있습니다.

토크나이저는 Ch 27·28·30 과 동일 (`PreTrainedTokenizerFast` + special token 명시 — `AutoTokenizer` 함정 회피).

학습 대상인 policy 모델과 토크나이저를 불러옵니다. KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 토크나이저로 잘못 fallback 되므로 (Ch 27), `PreTrainedTokenizerFast` 로 special token 을 직접 지정해 로드합니다. 단독 실행을 위해 SFT 체크포인트 대신 base 모델에서 시작합니다.

```python
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

t0 = time.time()
# 주의: KoGPT2 는 AutoTokenizer 가 영어 GPT2 토크나이저로 잘못 fallback (Ch 27).
# PreTrainedTokenizerFast 로 special token 을 직접 지정해 로드.
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)

# policy = 학습 대상. 보통은 Ch 28 SFT 체크포인트를 쓰지만, 단독 실행을 위해 base 로 시작.
SFT_MODEL = "skt/kogpt2-base-v2"   # Ch 28 SFT 체크포인트 경로가 있으면 여기에
policy = AutoModelForCausalLM.from_pretrained(SFT_MODEL).to(device)
policy.config.pad_token_id = tokenizer.pad_token_id
print(f"load done: {time.time()-t0:.1f}s")

n_params = policy.num_parameters()
print(f"\n=== policy model ===")
print(f"#params      : {n_params/1e6:.2f} M")
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
load done: 17.1s

=== policy model ===
#params      : 125.16 M
vocab_size   : 51,200
tokenizer    : TokenizersBackend
  eos_token  : </s>  id=1
  pad_token  : <pad>  id=3
```

## 5 SFT 워밍스타트 — GRPO 의 비제로 시작점 만들기

GRPO 는 한 prompt 에 여러 답(group)을 생성해 *그룹 안에서* 잘한 답의 확률을 올립니다. 그런데 base KoGPT2 는 산술을 거의 못 풀어 **그룹의 보상이 전부 0** 이 되기 쉽고, 그러면 advantage 가 모두 0 이라 *학습 신호가 없습니다*(GRPO 의 cold-start 함정).

그래서 표준 RLHF 파이프라인처럼 **GRPO 전에 짧은 SFT 로 "포맷 + 산술"을 먼저 가르칩니다.** 산술 prompt 와 정답을 지도학습해 모델이 일부 문제를 맞히기 시작하면(비제로 정확도), 그룹 안에 정답·오답이 섞여 advantage 가 생기고 GRPO 가 비로소 작동합니다. Ch 28 에서 한 그 SFT 를, 이번엔 산술 task 에 맞춰 워밍스타트로 씁니다.

base 모델은 산술을 거의 못 풀어 group 이 전부 오답이 되기 쉽습니다 (std=0 → advantage 0 → 학습 신호 없음). GRPO 가 신호를 얻으려면 모델이 가끔이라도 정답을 내야 하므로, 먼저 정답을 포함한 예제로 짧게 지도학습해 비제로 시작점을 만듭니다.

```python
# === SFT 워밍스타트 === GRPO 전에 산술 포맷+정답을 지도학습 (비제로 시작점)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

sft_ds = make_arithmetic(3000, max_operand=9, seed=SEED + 7)   # SFT 용 (정답 포함 학습)
def _to_sft(ex):
    # prompt + 정답 + EOS 를 통째로 언어모델링 (정답 생성을 학습)
    return tokenizer(ex["prompt"] + ex["answer"] + tokenizer.eos_token,
                     truncation=True, max_length=48, padding="max_length")
sft_tok = sft_ds.map(_to_sft, remove_columns=sft_ds.column_names)

sft_args = TrainingArguments(
    output_dir="./out_grpo_sft", num_train_epochs=5,
    per_device_train_batch_size=32, learning_rate=5e-4,
    warmup_ratio=0.1, lr_scheduler_type="cosine", max_grad_norm=1.0,
    fp16=torch.cuda.is_available(), logging_steps=50, save_strategy="no", report_to="none")
sft_trainer = Trainer(model=policy, args=sft_args, train_dataset=sft_tok,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
t0 = time.time(); sft_trainer.train()
print(f"SFT 워밍스타트 완료 ({(time.time()-t0)/60:.1f}min) - policy 가 이제 산술 포맷을 안다")
```

**▶ 실행 결과**

```text
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
[transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
<IPython.core.display.HTML object>
SFT 워밍스타트 완료 (1.2min) - policy 가 이제 산술 포맷을 안다
```

## verifier (reward function) 정의 + group advantage 손계산

여기가 본 챕터의 *개념 핵심*. **verifier 함수** 를 정의하고, 한 prompt 에 *여러 답* 을 채점한 뒤 *group relative advantage* 를 손으로 계산해 §의 표를 재현합니다. `GRPOTrainer` 가 매 step·매 prompt 내부에서 하는 일을 *축소판으로 재현* 하는 셈입니다.

### verifier — 생성 답에서 정답 추출 → 매칭 → reward

`trl` 의 reward 함수 시그니처는 **`reward_func(completions, **kwargs)`** 입니다:
- `completions`: policy 가 생성한 답들의 *리스트* (group)
- `**kwargs`: 데이터셋의 *나머지 컬럼* 이 *리스트로* 전달 (우리의 `answer` 컬럼이 `answer=[...]` 로 들어옴)
- 반환: 각 completion 의 **reward 리스트** (`list[float]`)

GRPO 의 핵심은 verifier 입니다. 생성된 답에서 정수를 추출해 정답과 일치하면 1.0, 아니면 0.0 을 주는 이진 보상 함수를 정의합니다. `trl` 의 reward 함수는 `completions` (생성 답 리스트) 와 `answer` (정답 리스트) 를 받아 reward 리스트를 돌려주는 시그니처를 씁니다.

```python
def extract_answer(text: str):
    '''생성 답에서 정답 정수를 추출. 응답 블록만 보고(모델이 다음 문제를 이어 생성해도
    그 숫자를 집지 않도록 "###" 앞까지만) 첫 정수를 집는다.'''
    seg = text.split("###")[0]
    m = re.search(r"-?\d+", seg)
    return m.group(0) if m else None


def reward_correct(completions, answer, **kwargs):
    '''verifier: 생성 답의 정수가 정답과 일치하면 1.0, 아니면 0.0 (이진 검증가능 보상).
    trl reward_func 시그니처: completions(생성 답 리스트) + answer(정답 리스트) -> reward 리스트.'''
    return [1.0 if (extract_answer(c) == str(g)) else 0.0
            for c, g in zip(completions, answer)]


# verifier 시연 - 한 prompt("3 + 5 = ?", 정답 8) 에 4개 답 (일부 맞음/틀림)
demo_completions = ["The answer is 8.", "answer: 7", "8", "I don't know"]
demo_answers = ["8", "8", "8", "8"]
demo_rewards = reward_correct(demo_completions, answer=demo_answers)
print("=" * 56)
print("verifier demo - prompt: '3 + 5 = ?', gold answer: 8")
print("=" * 56)
for c, r in zip(demo_completions, demo_rewards):
    print(f"  reward={r:.1f}  <- completion: {c!r}")
print(f"\nrewards (group): {demo_rewards}")
```

**▶ 실행 결과**

```text
========================================================
verifier demo - prompt: '3 + 5 = ?', gold answer: 8
========================================================
  reward=1.0  <- completion: 'The answer is 8.'
  reward=0.0  <- completion: 'answer: 7'
  reward=1.0  <- completion: '8'
  reward=0.0  <- completion: "I don't know"

rewards (group): [1.0, 0.0, 1.0, 0.0]
```

**결과 해석**

정답 8 이 포함된 `'The answer is 8.'` 과 `'8'` 은 1.0, 7 을 낸 답과 모르겠다는 답은 0.0 을 받았습니다. 한 group 안에 정답·오답이 섞여 있어 다음 단계의 group advantage 가 0 이 아닌 학습 신호를 만들 수 있습니다.

### group relative advantage 손계산 — reward → advantage

verifier 가 매긴 reward $[1, 0, 1, 0]$ 를 *group 평균 대비 상대값* 으로 바꿉니다 (§의 수식):

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \varepsilon}$$

이게 `GRPOTrainer` 가 *critic 없이* advantage 를 만드는 방법 — *group 평균이 baseline*.

GRPO 가 PPO 와 다른 핵심이 여기 있습니다. critic (value model) 없이, 같은 prompt 에서 나온 group 의 reward 평균을 baseline 으로 삼아 advantage 를 계산합니다. 평균보다 높은 답은 확률을 올리고 (advantage>0), 낮은 답은 내립니다 (advantage<0). 손으로 직접 계산해 group 구성에 따라 advantage 가 어떻게 달라지는지 살펴봅니다.

```python
def group_advantage(rewards, eps=1e-4):
    '''GRPO 의 group relative advantage = (r - mean) / (std + eps). critic 불필요.'''
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


rewards = np.array(demo_rewards)
adv = group_advantage(rewards)

print("=" * 60)
print("group relative advantage - by hand (group mean as baseline, no critic)")
print("=" * 60)
print(f"rewards          : {rewards}")
print(f"group mean       : {rewards.mean():.3f}   <- baseline (replaces critic)")
print(f"group std        : {rewards.std():.3f}")
print(f"advantage        : {np.round(adv, 3)}")
print("-" * 60)
for i, (r, a) in enumerate(zip(rewards, adv)):
    arrow = "prob UP (above avg)" if a > 0 else ("prob DOWN (below avg)" if a < 0 else "no signal")
    print(f"  y{i+1}: reward={r:.0f}  advantage={a:+.2f}  -> {arrow}")

# 다른 group 들도 - 동료 구성에 따라 advantage 가 어떻게 달라지나
print("\nadvantage for various group compositions:")
for rw in [[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]:
    a = group_advantage(rw)
    note = "  (all same -> no learning signal)" if np.allclose(a, 0) else ""
    print(f"  rewards={rw} -> advantage={np.round(a, 2)}{note}")
```

**▶ 실행 결과**

```text
============================================================
group relative advantage - by hand (group mean as baseline, no critic)
============================================================
rewards          : [1. 0. 1. 0.]
group mean       : 0.500   <- baseline (replaces critic)
group std        : 0.500
advantage        : [ 1. -1.  1. -1.]
------------------------------------------------------------
  y1: reward=1  advantage=+1.00  -> prob UP (above avg)
  y2: reward=0  advantage=-1.00  -> prob DOWN (below avg)
  y3: reward=1  advantage=+1.00  -> prob UP (above avg)
  y4: reward=0  advantage=-1.00  -> prob DOWN (below avg)

advantage for various group compositions:
  rewards=[1, 0, 1, 0] -> advantage=[ 1. -1.  1. -1.]
  rewards=[1, 1, 1, 0] -> advantage=[ 0.58  0.58  0.58 -1.73]
  rewards=[1, 0, 0, 0] -> advantage=[ 1.73 -0.58 -0.58 -0.58]
  rewards=[1, 1, 1, 1] -> advantage=[0. 0. 0. 0.]  (all same -> no learning signal)
  rewards=[0, 0, 0, 0] -> advantage=[0. 0. 0. 0.]  (all same -> no learning signal)
```

**결과 해석**

reward 가 `[1,0,1,0]` 이면 정답은 +1, 오답은 -1 로 갈립니다. 반면 group 이 전부 정답 `[1,1,1,1]` 이거나 전부 오답 `[0,0,0,0]` 이면 advantage 가 모두 0 — 비교 대상이 없어 학습 신호가 사라집니다. group 안에 정답·오답이 섞여야 GRPO 가 배웁니다.

**무엇을 보고 있나** — 위 두 출력은 `GRPOTrainer` 가 *매 step, 매 prompt* 내부에서 하는 계산입니다:

- **verifier** 가 *사람 없이 자동* 으로 reward 를 매깁니다 (정답 매칭). preference 라벨이 필요 없습니다
- **group advantage** 가 *critic 없이* 만들어집니다 — *그룹 동료들의 평균* 이 baseline. 평균보다 잘한 답은 +, 못한 답은 −
- **group 전체가 같으면 (전부 정답·전부 오답) advantage = 0** → 학습 신호 없음. *그룹 안에 다양성* (잘한 답 + 못한 답) 이 있어야 GRPO 가 작동합니다

> 이 두 부품 — *verifier (reward)* 와 *group advantage (baseline)* — 이 GRPO 의 전부입니다. 아래 §4 에서 `GRPOTrainer` 에 이 verifier 를 넘기면, 나머지 (rollout · advantage · 정책 갱신) 는 자동입니다.

## `GRPOTrainer` 로 GRPO 학습 — *새 trainer, verifier 로 정렬*

`trl.GRPOTrainer` 는 본 챕터에 처음 등장합니다. §3 에서 손으로 한 *verifier reward → group advantage* 를, *매 step* *rollout (여러 답 생성) → 채점 → advantage → 정책 갱신* 으로 자동 수행합니다. 설정은 `GRPOConfig` (`TrainingArguments` 상속) 로 주며, **`num_generations`** 가 group size 입니다.

> **rollout 주의 (T4 시간·메모리)**: GRPO 는 *매 step 여러 답을 생성* 하므로 무겁습니다 (DPO 보다 generation 비용이 큼). T4 + 30분 룰을 지키려면: **group size 작게 (`num_generations=4`) + 짧은 generation (`max_completion_length` 작게) + 작은 batch + 적은 step**. 시간이 빡빡하면 `N_TRAIN` 이나 step 을 더 줄이세요.

> **`trl` 버전 주의**: `GRPOConfig` 는 `max_completion_length` 를 받지만 `max_prompt_length` 는 버전에 따라 없습니다. `beta` 는 KL 제약의 세기로, 0 으로 두면 reference 없이(ref-free) 돌지만 정책이 SFT 모델에서 멀어지는 것을 막을 닻이 사라집니다. 본 노트북은 *작은 KL 앵커 (`beta=0.04`)* 로 reference (= SFT 모델) 근처에 묶어 collapse·reward hacking 을 완화합니다.

GRPO 전후를 비교하려면 흔들리지 않는 측정 기준이 필요합니다. sampling 은 실행마다 결과가 달라져 delta 를 읽기 어려우므로, greedy (`do_sample=False`) 로 정확도를 결정적으로 측정하는 함수를 만들고 학습 전 정확도를 먼저 기록합니다.

```python
from trl import GRPOTrainer, GRPOConfig


# GRPO 전·후 비교용 - greedy(do_sample=False) 로 결정화 측정.
# sampling 측정은 실행마다 베이스라인이 흔들려 delta 를 못 읽는다 -> greedy 로 고정.
@torch.no_grad()
def eval_accuracy(model, dataset, n=64, max_new=24):
    model.eval()
    correct = 0
    for ex in dataset.select(range(min(n, len(dataset)))):
        enc = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
                             pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += int(extract_answer(text) == str(ex["answer"]))
    return correct / min(n, len(dataset))


acc_before = eval_accuracy(policy, eval_ds, n=64)
print(f"BEFORE GRPO - arithmetic accuracy (greedy verifier pass rate): {acc_before:.3f}")
```

**▶ 실행 결과**

```text
BEFORE GRPO - arithmetic accuracy (greedy verifier pass rate): 0.875
```

## 5 🎯 난이도 필터 — GRPO 가 배울 *신호* 만들기

GRPO 의 advantage 는 그룹 안에서 $(r-\text{mean})/\text{std}$ 입니다. 그런데 한 자리 산술은 prompt 마다 정답률이 **0 또는 1 로 양극화** 되기 쉽습니다 - SFT 후 쉬운 문제는 8개 답이 *전부 정답*, 못 푸는 문제는 *전부 오답*. 그러면 그룹 보상의 **표준편차가 0** 이라 advantage 가 전부 0 → *학습 신호가 아예 없습니다*.

그래서 GRPO 가 실제로 배우려면 **그룹 안에 정답과 오답이 섞여야** 합니다. SFT 직후 각 prompt 의 정답률을 재서, *중간 난이도(약 25-87.5%)* 인 prompt 만 GRPO 학습셋으로 남깁니다. 이것이 reward 를 손대지 않고(이진 검증가능 보상 그대로) advantage 분산을 살리는 가장 직접적인 방법입니다.

앞서 봤듯 group 이 전부 정답이거나 전부 오답이면 advantage 가 0 이라 학습 신호가 없습니다. 그래서 각 prompt 에 여러 답을 생성해 정답률 (pass rate) 을 재고, 너무 쉽지도 어렵지도 않은 중간 난이도 문제만 남겨 group 안에 정답·오답이 섞이도록 (std>0) 데이터셋을 거릅니다.

```python
# 각 prompt 에 k 개 답을 생성해 정답률(pass rate)을 측정
@torch.no_grad()
def pass_rate(model, prompt, gold, k=8):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=0.7, top_p=0.95,
                         num_return_sequences=k, pad_token_id=tokenizer.pad_token_id)
    c = sum(extract_answer(tokenizer.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)) == str(gold)
            for g in gen)
    return c / k

# 중간 난이도(2-7/8 정답)인 prompt 만 남겨 그룹 std>0 보장
pool = make_arithmetic(500, max_operand=9, seed=SEED + 3)
keep = [ex for ex in pool if 0.25 <= pass_rate(policy, ex["prompt"], ex["answer"]) <= 0.875]
grpo_ds = Dataset.from_list(keep[:256]) if len(keep) >= 16 else grpo_ds  # 너무 적으면 원본 유지
print(f"난이도 필터: pool {len(pool)} -> 중간난이도 {len(grpo_ds)}개 (그룹에 정답·오답 섞임 → advantage std>0)")
```

**▶ 실행 결과**

```text
난이도 필터: pool 500 -> 중간난이도 256개 (그룹에 정답·오답 섞임 → advantage std>0)
```

이제 GRPO 본 학습입니다. `GRPOConfig` 에서 group 크기 (`num_generations`)·KL 앵커 강도 (`beta`)·작은 learning rate 등 collapse 를 막는 설정을 잡고, `reward_funcs` 에 앞서 만든 verifier 를 넘기면 `GRPOTrainer` 가 rollout → 채점 → group advantage → 정책 갱신을 자동으로 돌립니다.

```python
GROUP_SIZE = 8   # num_generations - rollout group size (T4 룰: 작게)

grpo_config = GRPOConfig(
    output_dir="./out_kogpt2_grpo",
    num_train_epochs=2,
    # 배치·그룹: micro-batch 당 prompt 2개(16/8) × grad_accum 8 = step 당 unique prompt 16
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    num_generations=GROUP_SIZE,               # 한 prompt 당 생성 답 개수(그룹 크기)
    max_completion_length=24,
    mask_truncated_completions=True,          # 잘린 생성은 loss 에서 제외
    # 탐색: eval(greedy)·rollout 정합, 산술은 저엔트로피라 0.7 로 낮춤
    temperature=0.7,
    top_p=0.95,
    # 신호 정규화: std 나눗셈 제거(난이도 bias·과증폭 차단)
    scale_rewards=False,
    loss_type="dr_grpo",
    # collapse 방지: lr 낮추고(정석 ~1e-6 근처) clip 강화, 짧은 RL 에 cosine 금지
    learning_rate=5e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.1,
    max_grad_norm=0.2,
    beta=0.04,                                # KL 앵커(참조=SFT 모델), ref-free(0) 금지
    fp16=USE_FP16,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    use_vllm=False,
    seed=SEED,
)


class VRAMCallback(__import__("transformers").TrainerCallback):
    '''step 별 peak VRAM 기록 (로깅 윈도우 단위 reset). CUDA 에서만 유효.'''

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

# reward_funcs 에 verifier 를 넘기면 rollout -> 채점 -> group advantage -> 정책 갱신 자동.
trainer = GRPOTrainer(
    model=policy,
    reward_funcs=reward_correct,   # <- verifier (callable 또는 list). 데이터의 answer 컬럼이 kwargs 로 전달
    args=grpo_config,
    train_dataset=grpo_ds,
    processing_class=tokenizer,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== GRPO summary ===")
print(f"elapsed     : {elapsed/60:.2f} min")
print(f"global_step : {train_out.global_step}")
print(f"train_loss  : {train_out.training_loss:.4f}")
if torch.cuda.is_available():
    print(f"final peak  : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
```

**▶ 실행 결과**

```text
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
[transformers] GPT2LMHeadModel LOAD REPORT from: skt/kogpt2-base-v2
Key                                     | Status     |  | 
----------------------------------------+------------+--+-
transformer.h.{0...11}.attn.masked_bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
<IPython.core.display.HTML object>
=== GRPO summary ===
elapsed     : 0.48 min
global_step : 32
train_loss  : 5171634.2480
final peak  : 2886 MiB
```

**결과 해석**

32 step, 약 0.5 분 만에 끝났고 peak VRAM 은 2886 MiB 로 T4 에 충분히 들어갑니다. `train_loss` 값 자체는 GRPO 목적함수의 부호·스케일이 분류 loss 와 달라 절대값으로 해석하지 않습니다 — 실제 효과는 다음 셀의 정확도 변화로 확인합니다.

## GRPO 전·후 정확도 비교 — *verifier pass rate 가 올랐는가*

본 챕터의 핵심 데모. *같은 eval 셋* (학습에 안 쓴 산술 문제) 에 대해 *GRPO 전* 과 *후* 의 **정확도 (verifier pass rate)** 를 비교합니다.

- **GRPO 전**: policy 가 산술을 잘 못 풀어 pass rate 낮음
- **GRPO 후**: *정답 방향* 으로 정책이 강화되어 pass rate ↑ (정답을 더 자주 생성)

정확도가 *올랐다면* verifiable reward 로 능력이 정렬된 직접 증거입니다.

```python
acc_after = eval_accuracy(policy, eval_ds, n=64)

print(f"AFTER  GRPO - arithmetic accuracy (verifier pass rate): {acc_after:.3f}")
print(f"BEFORE GRPO - arithmetic accuracy                     : {acc_before:.3f}")
print(f"delta                                                 : {acc_after - acc_before:+.3f}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
bars = ax.bar(["GRPO 전", "GRPO 후"], [acc_before, acc_after],
              color=["tab:gray", "tab:green"], alpha=0.85)
for b, v in zip(bars, [acc_before, acc_after]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
            ha="center", va="bottom")
ax.set_ylabel("정확도 (verifier pass rate)")
ax.set_ylim(0, 1)
ax.set_title("GRPO 전 vs 후 - 산술 정확도")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
```

**▶ 실행 결과**

```text
AFTER  GRPO - arithmetic accuracy (verifier pass rate): 0.891
BEFORE GRPO - arithmetic accuracy                     : 0.875
delta                                                 : +0.016
```

![output](../assets/31-grpo-out1.png)

**결과 해석**

정확도가 0.875 → 0.891 로 +0.016 올랐습니다. 작은 모델·짧은 학습이라 변화 폭은 미미하지만, *verifier reward + group advantage* 만으로 (사람 라벨·critic·reward model 없이) 정렬 방향이 양으로 움직였다는 것이 핵심입니다.

**해석 가이드 — verifiable reward alignment 의 증거**

- **before (gray)**: policy 가 산술을 잘 못 풀어 pass rate 가 낮습니다 (base KoGPT2 는 산술에 약함)
- **after (green)**: *정답 방향* 으로 정책이 강화되어 pass rate 가 오릅니다 — 모델이 *정답을 더 자주 생성*

> **핵심**: GRPO 는 *preference 라벨 없이*, *verifier 가 자동 채점한 reward* 만으로 능력을 정렬합니다. group 안에서 *정답이 평균보다 잘한 답* 으로 강화되며, 그 효과가 *정확도(pass rate) 상승* 으로 나타납니다.

> ⚠️ KoGPT2 (125M) 는 작은 base 모델이고 (정석은 SFT 모델에서 출발), 학습 step·group size 도 작아 효과가 *미묘* 할 수 있습니다. 관전 포인트는 *극적 향상* 이 아니라 ***정확도가 정답 방향으로 올랐는가*** 입니다. 또한 *group 안에 정답·오답이 섞여야* (std>0) 학습 신호가 생기므로, base 모델이 *가끔이라도 정답을 내야* GRPO 가 작동합니다 — §6 의 reward 곡선에서 확인.

## 학습 곡선 — reward / reward std / completion 길이

`GRPOTrainer` 는 학습 중 *loss* 뿐 아니라 *reward (group 평균)·reward_std·completion 길이* 같은 GRPO 고유 지표를 로깅합니다 (`trainer.state.log_history`). reward 가 오르고, reward_std 가 *0 이 아닌* (= group 안에 다양성이 있는) 구간에서 학습이 일어났는지 확인합니다.

학습 로그에서 step 별 group 평균 reward·reward std·loss 를 꺼내 그려, GRPO 가 진행되며 보상이 어떻게 움직였는지와 peak VRAM 을 확인합니다.

```python
log = trainer.state.log_history
steps = [r["step"] for r in log if "loss" in r]
losses = [r["loss"] for r in log if "loss" in r]


def series(key):
    return [(r["step"], r[key]) for r in log if key in r]


reward_s = series("reward")          # group 평균 reward
reward_std_s = series("reward_std")  # group reward 표준편차 (다양성 지표)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

if reward_s:
    ax1.plot([s for s, _ in reward_s], [v for _, v in reward_s], "o-",
             color="tab:green", label="reward (group 평균)")
if reward_std_s:
    ax1.plot([s for s, _ in reward_std_s], [v for _, v in reward_std_s], "s--",
             color="tab:orange", alpha=0.7, label="reward std (group 다양성)")
ax1.set_xlabel("step"); ax1.set_ylabel("reward")
ax1.set_title("GRPO - reward 와 reward std")
ax1.grid(True, alpha=0.3); ax1.legend()

if steps and losses:
    ax2.plot(steps, losses, "-", color="tab:blue", alpha=0.8, label="GRPO loss")
ax2.set_xlabel("step"); ax2.set_ylabel("GRPO loss")
ax2.set_title("GRPO - loss")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()

if torch.cuda.is_available() and vram_cb.steps:
    print(f"peak VRAM (max over training): {max(vram_cb.peak_MiB):.0f} MiB"
          f"  (policy only, ref-free, num_generations={GROUP_SIZE}, fp16)")
```

**▶ 실행 결과**

![output](../assets/31-grpo-out2.png)

```text
peak VRAM (max over training): 3670 MiB  (policy only, ref-free, num_generations=8, fp16)
```
