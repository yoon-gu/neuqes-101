> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/32_diffusion_intro/32_diffusion_intro.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
%pip install -q -U transformers tokenizers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 118.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 48.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 39.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━ 34.4/48.9 MB 261.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 148.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 148.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 16.8 MB/s eta 0:00:00
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
import torch.nn as nn
import torch.nn.functional as F

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
```

**▶ 실행 결과**

```text
device     : cuda  (Tesla T4)
VRAM total : 14.56 GiB
torch      : 2.11.0+cu128
use fp16   : True
```

```python
from datasets import load_dataset

N_TRAIN = 100_000      # 더 길게 돌리려면 키우세요 (full 은 약 2.1M stories)
N_VAL   = 500

raw_train = load_dataset("roneneldan/TinyStories", split=f"train[:{N_TRAIN}]")
raw_val   = load_dataset("roneneldan/TinyStories", split=f"validation[:{N_VAL}]")
print("train:", raw_train)
print("val  :", raw_val)
print("\n=== sample story ===")
print(raw_train[0]["text"][:400])
```

**▶ 실행 결과**

```text
train: Dataset({
    features: ['text'],
    num_rows: 100000
})
val  : Dataset({
    features: ['text'],
    num_rows: 500
})

=== sample story ===
One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to …(뒤 71자 생략)

Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, w …(뒤 43자 생략)

To
```

```python
# 작은 모델엔 작은 vocab — TinyStories 에 BPE 2048 직접 학습 + [MASK] 추가
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
VOCAB = 2048
def corpus_iter(bs=1000):
    for i in range(0, len(raw_train), bs):
        yield raw_train[i:i+bs]["text"]
_tk = Tokenizer(models.BPE(unk_token="[UNK]"))
_tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
_tk.decoder = decoders.ByteLevel()
_tk.train_from_iterator(corpus_iter(), trainer=trainers.BpeTrainer(
    vocab_size=VOCAB, special_tokens=["[PAD]", "[UNK]", "[MASK]"]))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tk, pad_token="[PAD]",
                                    unk_token="[UNK]", mask_token="[MASK]")
print(f"vocab_size : {tokenizer.vocab_size}")
print(f"[MASK]     : '{tokenizer.mask_token}'  id={tokenizer.mask_token_id}")

# [MASK] 가 섞인 시퀀스 = diffusion 의 노이즈. 일부를 가려본다
import random as _r; _r.seed(0)
sample = "Once upon a time, a little rabbit went to the forest."
enc = tokenizer(sample, add_special_tokens=False)["input_ids"]
md = enc[:]
for i in sorted(_r.sample(range(len(enc)), max(1, len(enc)//3))):
    md[i] = tokenizer.mask_token_id
print("\noriginal :", sample)
print("masked   :", tokenizer.decode(md))
```

**▶ 실행 결과**

```text
vocab_size : 2048
[MASK]     : '[MASK]'  id=2

original : Once upon a time, a little rabbit went to the forest.
masked   : [MASK] upon a time[MASK] a[MASK] rabbit went to the forest[MASK]
```

```python
BLOCK_SIZE = 128

def tokenize_fn(batch):
    # add_special_tokens=False - [CLS]/[SEP] 없이 순수 토큰 스트림
    return tokenizer(batch["text"], add_special_tokens=False)

tok_train = raw_train.map(tokenize_fn, batched=True, remove_columns=raw_train.column_names, desc="tokenize train")
tok_val   = raw_val.map(tokenize_fn,   batched=True, remove_columns=raw_val.column_names,   desc="tokenize val")

# group_texts - 모든 토큰을 이어붙여 BLOCK_SIZE 단위로 자름
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

# 학습엔 input_ids 만 필요 (마스킹은 collator 가 매번 새로 함)
lm_train = lm_train.remove_columns([c for c in lm_train.column_names if c != "input_ids"])
lm_val   = lm_val.remove_columns([c for c in lm_val.column_names if c != "input_ids"])

print(f"\ntrain chunks: {len(lm_train):,}  (block_size={BLOCK_SIZE})")
print(f"val   chunks: {len(lm_val):,}")
print(f"approx. train tokens: {len(lm_train) * BLOCK_SIZE / 1e6:.2f} M")
print("\nfirst chunk decode (first 200 chars):")
print(tokenizer.decode(lm_train[0]["input_ids"])[:200])
```

**▶ 실행 결과**

```text
train chunks: 189,030  (block_size=128)
val   chunks: 853
approx. train tokens: 24.20 M

first chunk decode (first 200 chars):
 One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted t …(뒤 60자 생략)
```

```python
class DiffusionCollator:
    '''매 배치마다 t ~ U(eps, 1) 을 뽑아 그 비율로 토큰을 [MASK] 치환.'''

    def __init__(self, tokenizer, eps=0.02, seed=42):
        self.mask_id = tokenizer.mask_token_id
        self.eps = eps
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, examples):
        ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        B, L = ids.shape

        # 샘플별 마스킹 비율 t ~ U(eps, 1)
        t = torch.rand(B, generator=self.gen) * (1.0 - self.eps) + self.eps          # (B,)

        # 각 토큰을 독립적으로 확률 t 로 마스킹
        mask = torch.rand(B, L, generator=self.gen) < t.unsqueeze(1)                  # (B, L) bool
        # 적어도 한 자리는 가리도록 보정 (t 가 아주 작아 전부 안 가려진 경우 방지)
        no_mask_rows = ~mask.any(dim=1)
        if no_mask_rows.any():
            j = torch.randint(0, L, (int(no_mask_rows.sum()),), generator=self.gen)
            mask[no_mask_rows, j] = True

        input_ids = ids.clone()
        input_ids[mask] = self.mask_id
        labels = ids.clone()
        labels[~mask] = -100                                     # 가린 자리만 학습 신호

        attention_mask = torch.ones(B, L, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "t": t}


diff_collator = DiffusionCollator(tokenizer)

# collator 출력 확인 - 같은 두 chunk 를 여러 번 돌리면 매번 다른 비율로 가려짐
print("=== diffusion collator demo (same 2 chunks, masking ratio varies each call) ===")
for trial in range(3):
    batch = diff_collator([lm_train[0], lm_train[1]])
    labels = batch["labels"]
    t = batch["t"]
    for b in range(labels.shape[0]):
        n_masked = (labels[b] != -100).sum().item()
        frac = 100 * n_masked / labels.shape[1]
        print(f"trial {trial} | sample {b}: t={t[b]:.3f}  ->  masked {n_masked:>3d}/{labels.shape[1]} ({frac:5.1f}%)")
    print()
```

**▶ 실행 결과**

```text
=== diffusion collator demo (same 2 chunks, masking ratio varies each call) ===
trial 0 | sample 0: t=0.885  ->  masked 111/128 ( 86.7%)
trial 0 | sample 1: t=0.917  ->  masked 119/128 ( 93.0%)

trial 1 | sample 0: t=0.756  ->  masked  98/128 ( 76.6%)
trial 1 | sample 1: t=0.732  ->  masked  98/128 ( 76.6%)

trial 2 | sample 0: t=0.937  ->  masked 121/128 ( 94.5%)
trial 2 | sample 1: t=0.046  ->  masked   9/128 (  7.0%)
```

```python
from transformers import BertConfig, BertForMaskedLM

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=BLOCK_SIZE,
    pad_token_id=tokenizer.pad_token_id,
)

model = BertForMaskedLM(config).to(device)
n_params = model.num_parameters()
print(f"#params           : {n_params/1e6:.2f} M")
print(f"vocab_size        : {config.vocab_size}")
print(f"\nmodel: {type(model).__name__}")
print(f"  - body : {type(model.bert).__name__}  (Encoder, bidirectional attention)")
print(f"  - head : MLM head -> Linear(in={config.hidden_size}, out={config.vocab_size})")
```

**▶ 실행 결과**

```text
#params           : 3.79 M
vocab_size        : 2048

model: BertForMaskedLM
  - body : BertModel  (Encoder, bidirectional attention)
  - head : MLM head -> Linear(in=256, out=2048)
```

```python
@torch.no_grad()
def diffusion_generate(active_model, length=64, steps=16, temperature=1.0, top_k=50,
                       prompt_ids=None, record_trajectory=False):
    '''전부 [MASK] 에서 시작해 steps 번 denoise. prompt_ids 를 주면 앞부분 고정 (조건부 생성).

    기본은 sampling (temperature>0). temperature=0 으로 두면 greedy 인데,
    작은 모델 + 전부-[MASK] 출발에서는 greedy 가 최빈 토큰('.')만 뽑는 *collapse* 가 잘 일어나
    sampling 을 기본값으로 둡니다 (아래 한계 노트 참고).'''
    active_model.eval()
    dev = active_model.device
    mask_id = tokenizer.mask_token_id

    x = torch.full((1, length), mask_id, dtype=torch.long, device=dev)
    fixed = torch.zeros(length, dtype=torch.bool, device=dev)   # 절대 마스킹 안 할 자리 (prompt)
    if prompt_ids is not None:
        p = torch.tensor(prompt_ids[:length], device=dev)
        x[0, :len(p)] = p
        fixed[:len(p)] = True
    n_gen = int((~fixed).sum().item())                          # 생성해야 할 자리 수

    traj = []
    for step in range(steps):
        logits = active_model(input_ids=x).logits[0]            # (L, V)
        probs = logits.softmax(dim=-1)
        if temperature > 0:
            scaled = logits / temperature
            if top_k > 0:                                       # top-k 로 후보 제한
                kth = scaled.topk(top_k, dim=-1).values[:, -1, None]
                scaled = scaled.masked_fill(scaled < kth, float("-inf"))
            pred = torch.multinomial(scaled.softmax(-1), 1).squeeze(-1)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        else:
            conf, pred = probs.max(dim=-1)                      # greedy (최빈 토큰 collapse 주의)

        is_mask = (x[0] == mask_id) & (~fixed)                  # 지금 마스킹된 (생성 대상) 자리
        # 일단 마스킹된 자리를 예측으로 채운 잠정 시퀀스
        x_new = torch.where(is_mask, pred, x[0])

        # 이 step 이 끝났을 때 남겨둘 [MASK] 수 (선형 스케줄: n_gen -> 0)
        n_remain = int(round(n_gen * (1.0 - (step + 1) / steps)))
        if n_remain > 0:
            # 마스킹됐던 자리들 중 confidence 가 낮은 n_remain 개를 다시 [MASK] 로
            conf_masked = conf.clone()
            conf_masked[~is_mask] = float("inf")               # 마스킹 안 됐던 자리는 후보에서 제외
            remask_idx = conf_masked.topk(n_remain, largest=False).indices
            x_new[remask_idx] = mask_id

        x[0] = x_new
        if record_trajectory:
            traj.append(x[0].clone())

    text = tokenizer.decode(x[0], skip_special_tokens=True)
    return (text, traj) if record_trajectory else text
```

```python
torch.manual_seed(SEED)
print("=" * 70)
print("UNTRAINED model - parallel denoise from all-[MASK]")
print("=" * 70)
for i in range(3):
    text = diffusion_generate(model, length=48, steps=16)
    print(f"\n[sample {i}] {text}")
```

**▶ 실행 결과**

```text
======================================================================
UNTRAINED model - parallel denoise from all-[MASK]
======================================================================
[sample 0]  together block mommy smo mommyblem warely� picturesndergetheranced fellday Emnder cars greendren pictures waitoomced squirrel sa …(뒤 103자 생략)

[sample 1]  him pret birthday smoow Jimmy belie You birthday mommy pictures mommy sc sk birthday gre sc mommy skho birthday ha7 wait mail mo …(뒤 112자 생략)

[sample 2] aduched mommy smo explainedriesdayelyely gre mommypblem sk goodbye grender mommy�ho birthdayblemblem waitred pictures mommynderar …(뒤 96자 생략)
```

```python
from transformers import Trainer, TrainingArguments, TrainerCallback


class DiffusionTrainer(Trainer):
    '''masked-diffusion loss: 가려진 자리 CE 를 샘플별로 1/t 재가중.'''

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        t = inputs["t"]                                          # (B,)
        labels = inputs["labels"]                               # (B, L)
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        logits = outputs.logits                                 # (B, L, V)
        B, L, V = logits.shape

        per_tok = F.cross_entropy(
            logits.view(-1, V), labels.view(-1),
            ignore_index=-100, reduction="none",
        ).view(B, L)                                            # 가린 자리만 비-0 (나머지 -100 -> 0)

        # 샘플별: (가린 자리 CE 합 / L) * (1/t)
        per_ex = per_tok.sum(dim=1) / L
        loss = (per_ex / t.to(per_ex.dtype)).mean()
        return (loss, outputs) if return_outputs else loss


args = TrainingArguments(
    output_dir="./out_diffusion_intro",
    max_steps=30000,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    fp16=USE_FP16,                       # T4 는 bf16 불가
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=150,
    save_strategy="no",
    report_to="none",
    label_names=["labels"],
    remove_unused_columns=False,         # 'labels','t' 를 collator 가 만들므로 보존
    seed=SEED,
)


class VRAMCallback(TrainerCallback):
    def __init__(self):
        self.steps, self.peak_MiB = [], []

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            self.steps.append(state.global_step)
            self.peak_MiB.append(torch.cuda.max_memory_allocated() / 1024**2)
            torch.cuda.reset_peak_memory_stats()


vram_cb = VRAMCallback()

trainer = DiffusionTrainer(
    model=model,
    args=args,
    train_dataset=lm_train,
    eval_dataset=lm_val,
    data_collator=diff_collator,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== training summary ===")
print(f"elapsed       : {elapsed/60:.2f} min")
print(f"global_step   : {train_out.global_step}")
print(f"train_loss    : {train_out.training_loss:.4f}")
print(f"random baseline (ln vocab): {math.log(tokenizer.vocab_size):.4f}")
if torch.cuda.is_available():
    print(f"final peak    : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
=== training summary ===
elapsed       : 18.92 min
global_step   : 30000
train_loss    : 3.7148
random baseline (ln vocab): 7.6246
final peak    : 61 MiB
```

```python
# loss curve + VRAM trace
log = trainer.state.log_history
train_pts = [(r["step"], r["loss"]) for r in log if "loss" in r and "eval_loss" not in r]
eval_pts  = [(r["step"], r["eval_loss"]) for r in log if "eval_loss" in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot([s for s, _ in train_pts], [l for _, l in train_pts], "-",
         color="tab:blue", alpha=0.6, label="train")
if eval_pts:
    ax1.plot([s for s, _ in eval_pts], [l for _, l in eval_pts], "s-",
             color="tab:red", label="eval")
ax1.axhline(math.log(tokenizer.vocab_size), ls=":", color="gray",
            label=f"uniform baseline = ln({tokenizer.vocab_size}) approx. {math.log(tokenizer.vocab_size):.2f}")
ax1.set_xlabel("step"); ax1.set_ylabel("diffusion denoising loss (1/t reweighted)")
ax1.set_title("Small mask-diffusion LM on TinyStories - loss")
ax1.grid(True, alpha=0.3); ax1.legend()

if vram_cb.steps:
    ax2.plot(vram_cb.steps, vram_cb.peak_MiB, "o-", color="tab:green",
             label="peak VRAM (per log window)")
    ax2.set_title(f"VRAM trace  (bs=32, fp16, L={BLOCK_SIZE})")
else:
    ax2.text(0.5, 0.5, "VRAM trace available on CUDA only",
             ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("VRAM trace - CUDA only")
ax2.set_xlabel("step"); ax2.set_ylabel("VRAM (MiB)")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()
```

**▶ 실행 결과**

![output](../assets/32-diffusion_intro-out1.png)

```python
torch.manual_seed(SEED)
print("=" * 70)
print("TRAINED model - parallel denoise from all-[MASK]")
print("=" * 70)
for i in range(3):
    text = diffusion_generate(model, length=48, steps=16)
    print(f"\n[sample {i}] {text}")
```

**▶ 실행 결과**

```text
======================================================================
TRAINED model - parallel denoise from all-[MASK]
======================================================================

[sample 0]  Ben are twins. They run to the park. They run to slide and run. They want to reach the park. They see the big slide. They see the noise. They

They and Ben are happy. They are happy.

[sample 1] 
"OK, Ben. We will play the balloon," Ben says.

"We can play the park!" Ben says..

They run to the park. They. They seek in the park. They
They
[sample 2] . They understand. They are happy. They
TheyThey see the park again. They They see the dog. They are not happy in the park. They hug their mom. They back to the park. They are very happy. They
```

```python
# denoise 궤적 - [MASK] 가 단어로 채워지는 과정을 step 별로
torch.manual_seed(SEED)
text, traj = diffusion_generate(model, length=40, steps=12, record_trajectory=True)

def render(ids):
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    return " ".join("____" if tk == tokenizer.mask_token else tk for tk in toks)

print("=" * 78)
print("DENOISE TRAJECTORY  ('____' = still [MASK])  - filled in parallel, by confidence")
print("=" * 78)
n_steps = len(traj)
for step in [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]:
    n_mask = (traj[step] == tokenizer.mask_token_id).sum().item()
    print(f"\nstep {step:>2d}/{n_steps-1}  ([MASK] remaining: {n_mask:>2d})")
    print("  " + render(traj[step]))

print("\n" + "=" * 78)
print("FINAL:", text)
```

**▶ 실행 결과**

```text
==============================================================================
DENOISE TRAJECTORY  ('____' = still [MASK])  - filled in parallel, by confidence
==============================================================================

step  0/11  ([MASK] remaining: 37)
  ____ ____ ____ ____ ____ ____ ____ ____ ____ . ____ ____ ____ ____ Ġand ____ ____ ____ ____ ____ ____ ____ ____ . ____ ____ ____ ____ ____ …(뒤 55자 생략)

step  3/11  ([MASK] remaining: 27)
  Ġand ____ ____ ____ ____ ____ ____ ____ ____ . ĠThey ____ Ġto ____ Ġand ____ ____ ____ ____ ____ ____ ____ ____ . Ċ Ċ They ____ ____ Ġto Ġ …(뒤 45자 생략)

step  6/11  ([MASK] remaining: 17)
  Ġand ____ . . ĠThey ____ ____ Ġthe ____ . ĠThey ____ Ġto ____ Ġand ____ . . ĠThey ____ ____ ____ ____ . Ċ Ċ They ____ Ġback Ġto Ġthe Ġand …(뒤 37자 생략)

step  9/11  ([MASK] remaining:  7)
  Ġand ____ . . ĠThey Ġrun Ġto Ġthe Ġpark . ĠThey ____ Ġto ____ Ġand Ġrun . . ĠThey ____ ____ Ġand Ġfun . Ċ Ċ They Ġgo Ġback Ġto Ġthe Ġand _ …(뒤 36자 생략)

step 11/11  ([MASK] remaining:  0)
  Ġand Ġrun . . ĠThey Ġrun Ġto Ġthe Ġpark . ĠThey Ġwant Ġto Ġswing Ġand Ġrun . . ĠThey Ġhave Ġfun Ġand Ġfun . Ċ Ċ They Ġgo Ġback Ġto Ġthe Ġa …(뒤 43자 생략)

==============================================================================
FINAL:  and run.. They run to the park. They want to swing and run.. They have fun and fun.

They go back to the and slide. They say they are to stop
```
