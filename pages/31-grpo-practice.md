> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/31_grpo/31_grpo.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
%pip install -q -U trl transformers tokenizers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 825.1/825.1 kB 49.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 119.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 45.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 35.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━ 22.4/48.9 MB 221.2 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 146.3 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 146.3 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 16.9 MB/s eta 0:00:00
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
```

**▶ 실행 결과**

```text
trl          : 1.6.0
device       : cuda  (Tesla T4)
VRAM total   : 14.56 GiB
torch        : 2.11.0+cu128
use fp16     : True
```

GRPO 학습용 데이터를 만듭니다. 산술 문제를 prompt로 만들고 정답을 따로 보관하는데, 이 정답은 모델 입력이 아니라 verifier가 생성 답을 자동 채점할 때만 씁니다. 보상 모델 없이 규칙으로 채점할 수 있는 task라는 점을 눈여겨보세요.

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

정렬할 policy 모델과 토크나이저를 로드합니다. 보통은 Ch 28 SFT 체크포인트를 출발점으로 삼지만 여기서는 단독 실행을 위해 base 모델로 시작합니다. KoGPT2는 `AutoTokenizer`가 영어 토크나이저로 잘못 fallback되므로 `PreTrainedTokenizerFast`로 special token을 직접 지정해 로드하는 점을 봐 두세요.

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
load done: 22.2s

=== policy model ===
#params      : 125.16 M
vocab_size   : 51,200
tokenizer    : TokenizersBackend
  eos_token  : </s>  id=1
  pad_token  : <pad>  id=3
```

GRPO의 핵심인 verifier(보상 함수)를 정의합니다. DPO처럼 별도 보상 모델을 학습하는 대신, 생성 답에서 마지막 정수를 뽑아 정답과 일치하면 1.0, 아니면 0.0을 주는 규칙 기반 채점입니다. 아래 데모에서 같은 prompt에 대한 여러 답이 어떻게 0/1로 갈리는지 확인하세요.

```python
def extract_last_int(text: str):
    '''생성 답에서 마지막 정수를 추출 (없으면 None). 산술 task 의 정답 후보.'''
    matches = re.findall(r"-?\d+", text)
    return matches[-1] if matches else None


def reward_correct(completions, answer, **kwargs):
    '''verifier: 생성 답의 마지막 정수가 정답과 일치하면 1.0, 아니면 0.0.

    trl reward_func 시그니처:
      - completions: 생성된 답 리스트 (group)
      - answer     : 데이터셋의 'answer' 컬럼이 리스트로 전달 (정답)
      - 반환       : 각 completion 의 reward 리스트
    '''
    rewards = []
    for comp, gold in zip(completions, answer):
        pred = extract_last_int(comp)
        rewards.append(1.0 if (pred is not None and pred == str(gold)) else 0.0)
    return rewards


# verifier 시연 - 한 prompt("3 + 5 = ?", 정답 8) 에 4개 답 (일부 맞음/틀림)
demo_completions = [
    "The answer is 8.",         # 맞음 -> 1.0
    "answer: 7",                # 틀림 -> 0.0
    "8",                        # 맞음 -> 1.0
    "I don't know",             # 숫자 없음 -> 0.0
]
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

GRPO의 group relative advantage를 손으로 계산해 봅니다. 한 prompt에서 만든 여러 rollout의 reward를 그룹 평균으로 빼고 표준편차로 나누어, critic(가치망) 없이 그룹 평균을 baseline으로 삼는 방식입니다. 그룹 구성이 전부 같으면 advantage가 0이 되어 학습 신호가 사라진다는 점을 마지막 출력에서 확인하세요.

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

GRPO 전후를 비교하기 위해 학습 전 정확도를 먼저 측정합니다. eval 셋의 각 prompt에 답을 여러 개 생성하고 verifier로 채점해 pass rate(정확도)를 구합니다. base 모델이라 산술을 거의 못 맞히는 출발점을 확인해 두세요.

```python
from trl import GRPOTrainer, GRPOConfig


# GRPO 전·후 비교용 - eval 셋에서 정확도(verifier pass rate) 측정
@torch.no_grad()
def eval_accuracy(model, dataset, n=64, n_sample=2, max_new=24):
    '''각 prompt 에 n_sample 개 답을 생성해 verifier pass rate (정확도) 계산.'''
    model.eval()
    correct, total = 0, 0
    for ex in dataset.select(range(min(n, len(dataset)))):
        enc = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new, do_sample=True, temperature=1.0,
            top_p=0.95, num_return_sequences=n_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        for g in gen:
            text = tokenizer.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_last_int(text)
            correct += int(pred is not None and pred == str(ex["answer"]))
            total += 1
    return correct / max(total, 1)


acc_before = eval_accuracy(policy, eval_ds, n=64, n_sample=2)
print(f"BEFORE GRPO - arithmetic accuracy (verifier pass rate): {acc_before:.3f}")
```

**▶ 실행 결과**

```text
BEFORE GRPO - arithmetic accuracy (verifier pass rate): 0.000
```

**결과 해석**

base KoGPT2는 산술을 한 번도 맞히지 못해 정확도가 0.000입니다. GRPO가 끌어올려야 할 기준선입니다.

`GRPOConfig`와 `GRPOTrainer`로 실제 정렬을 수행합니다. `num_generations=4`로 한 prompt당 4개의 rollout을 생성해 그룹 내 상대 비교로 정책을 갱신하고, `beta=0.0`으로 reference 모델 없이(ref-free) 메모리를 아낍니다. `reward_funcs`에 verifier를 넘기면 rollout → 채점 → group advantage → 정책 갱신이 자동으로 돌아갑니다.

```python
GROUP_SIZE = 4   # num_generations - rollout group size (T4 룰: 작게)

grpo_config = GRPOConfig(
    output_dir="./out_kogpt2_grpo",
    num_train_epochs=1,
    per_device_train_batch_size=GROUP_SIZE,   # group rollout 이 한 batch 에 들어가도록
    gradient_accumulation_steps=4,
    num_generations=GROUP_SIZE,               # <- 한 prompt 당 생성 답 개수 (group size)
    max_completion_length=24,                 # 짧은 산술 답 - generation 비용 통제
    temperature=1.0,                          # rollout 다양성 (group 안에 정답·오답 섞이게)
    learning_rate=1e-5,
    beta=0.0,                                 # 0 = ref-free (reference 없이, 메모리 절약)
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    fp16=USE_FP16,                            # T4 는 bf16 불가
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    use_vllm=False,                           # vLLM 없이 HF generate 로 rollout (Colab 호환)
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
<IPython.core.display.HTML object>
=== GRPO summary ===
elapsed     : 0.77 min
global_step : 64
train_loss  : -0.0000
final peak  : 1451 MiB
```

**결과 해석**

ref-free에 num_generations=4로 64 step 학습이 0.77분, peak VRAM 1451 MiB에 끝나 T4 30분 룰 안에 충분히 들어옵니다. train_loss가 0 근처인 것은 advantage 가중 정책 손실의 특성으로, 정렬 효과는 loss 값이 아니라 정확도 변화로 봐야 합니다.

GRPO 후 정확도를 다시 측정해 학습 전과 막대그래프로 비교합니다. 같은 eval 셋·같은 verifier로 채점하므로 두 값의 차이가 곧 GRPO의 정렬 효과입니다.

```python
acc_after = eval_accuracy(policy, eval_ds, n=64, n_sample=2)

print(f"AFTER  GRPO - arithmetic accuracy (verifier pass rate): {acc_after:.3f}")
print(f"BEFORE GRPO - arithmetic accuracy                     : {acc_before:.3f}")
print(f"delta                                                 : {acc_after - acc_before:+.3f}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
bars = ax.bar(["before GRPO", "after GRPO"], [acc_before, acc_after],
              color=["tab:gray", "tab:green"], alpha=0.85)
for b, v in zip(bars, [acc_before, acc_after]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
            ha="center", va="bottom")
ax.set_ylabel("accuracy (verifier pass rate)")
ax.set_ylim(0, 1)
ax.set_title("GRPO before vs after - arithmetic accuracy")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
```

**▶ 실행 결과**

```text
AFTER  GRPO - arithmetic accuracy (verifier pass rate): 0.047
BEFORE GRPO - arithmetic accuracy                     : 0.000
delta                                                 : +0.047
```

![output](../assets/31-grpo-out1.png)

**결과 해석**

정확도가 0.000에서 0.047로 올라(delta +0.047) GRPO가 짧은 학습만으로도 정렬 방향이 옳음을 보여줍니다. 절대값이 낮은 것은 작은 base 모델·256 샘플·1 에폭이라는 T4 제약 때문이며, 변형 코너의 group size 확대나 더 큰 데이터로 끌어올릴 수 있습니다.

학습 로그에서 group 평균 reward와 reward 표준편차, GRPO loss의 추이를 그립니다. reward가 오르는지, 그리고 그룹 안에 정답·오답이 섞여 학습 신호를 만드는 reward std가 유지되는지를 함께 확인하는 그림입니다.

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
             color="tab:green", label="reward (group mean)")
if reward_std_s:
    ax1.plot([s for s, _ in reward_std_s], [v for _, v in reward_std_s], "s--",
             color="tab:orange", alpha=0.7, label="reward std (group diversity)")
ax1.set_xlabel("step"); ax1.set_ylabel("reward")
ax1.set_title("GRPO - reward and reward std")
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
peak VRAM (max over training): 2230 MiB  (policy only, ref-free, num_generations=4, fp16)
```

**결과 해석**

학습 전 구간 peak VRAM이 2230 MiB로 T4의 16GB에 여유 있게 들어옵니다. ref-free(`beta=0.0`)라 reference 모델을 띄우지 않고 policy만 메모리에 올린 덕분이며, num_generations를 키우면 rollout 비용과 함께 이 값도 늘어납니다.
