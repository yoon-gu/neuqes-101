> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/28_sft/28_sft.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

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

```python
# trl 1.x 의 SFT collator. 버전마다 위치가 다를 수 있어 폴백 import.
try:
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling as SFTCollator
except Exception:
    from trl import DataCollatorForLanguageModeling as SFTCollator  # 일부 버전

# 한 샘플을 prompt / completion 으로 직접 토큰화 (SFTTrainer 내부와 같은 절차)
sample = sft_ds[0]
prompt_text = sample["prompt"]
completion_text = sample["completion"]

p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
c_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
c_ids = c_ids + [tokenizer.eos_token_id]   # SFTTrainer 는 답변 끝에 EOS 부착

input_ids = p_ids + c_ids
completion_mask = [0] * len(p_ids) + [1] * len(c_ids)   # 0 = prompt, 1 = 답변

print(f"prompt tokens     : {len(p_ids)}")
print(f"completion tokens : {len(c_ids)}  (incl. EOS)")
print(f"total tokens      : {len(input_ids)}")

# collator 적용 - prompt 부분이 -100 으로 마스킹됨
collator = SFTCollator(pad_token_id=tokenizer.pad_token_id, completion_only_loss=True)
batch = collator([{"input_ids": input_ids, "completion_mask": completion_mask}])
labels = batch["labels"][0].tolist()
ids = batch["input_ids"][0].tolist()

n_learn = sum(1 for l in labels if l != -100)
print(f"\nlabels learned    : {n_learn} / {len(labels)}  (prompt masked = {len(labels) - n_learn})")
```

**▶ 실행 결과**

```text
prompt tokens     : 38
completion tokens : 142  (incl. EOS)
total tokens      : 180

labels learned    : 142 / 180  (prompt masked = 38)
```

```python
# 토큰별 표 - position | token | input_id | label | learn?
rows = []
for i, (tid, lab) in enumerate(zip(ids, labels)):
    rows.append({
        "pos": i,
        "token": repr(tokenizer.decode([tid])),
        "input_id": tid,
        "label": lab,
        "learn?": "Y (response)" if lab != -100 else "- (prompt, -100)",
    })
label_table = pd.DataFrame(rows)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)
print("=" * 78)
print("Per-token labels - prompt is masked (-100), only response is learned")
print("=" * 78)
print(label_table.to_string(index=False))
```

**▶ 실행 결과**

```text
==============================================================================
Per-token labels - prompt is masked (-100), only response is learned
==============================================================================
 pos   token  input_id  label           learn?
   0      ''       739   -100 - (prompt, -100)
   1     '#'       378   -100 - (prompt, -100)
   2     '#'       378   -100 - (prompt, -100)
   3     '#'       378   -100 - (prompt, -100)
   4    '명령'     14266   -100 - (prompt, -100)
   5     '어'      8006   -100 - (prompt, -100)
   6     ':'       401   -100 - (prompt, -100)
   7    '\n'       375   -100 - (prompt, -100)
   8   '나무가'     18306   -100 - (prompt, -100)
   9    '말라'     15020   -100 - (prompt, -100)
  10    '죽을'     14909   -100 - (prompt, -100)
  11     '때'      9068   -100 - (prompt, -100)
  12     '왜'     10401   -100 - (prompt, -100)
  13     '속'      9238   -100 - (prompt, -100)
  14    '부터'      9148   -100 - (prompt, -100)
  15     '썩'     23623   -100 - (prompt, -100)
  16     '는'      7162   -100 - (prompt, -100)
  17     '걸'      9539   -100 - (prompt, -100)
  18     '까'      6969   -100 - (prompt, -100)
  19     '요'      8084   -100 - (prompt, -100)
  20     '?'       406   -100 - (prompt, -100)
  21  '그리고,'     39678   -100 - (prompt, -100)
  22    '나무'     10221   -100 - (prompt, -100)
  23    '속에'     10671   -100 - (prompt, -100)
  24   '전선을'     46886   -100 - (prompt, -100)
  25    '넣을'     44361   -100 - (prompt, -100)
  26     '수'      9025   -100 - (prompt, -100)
  27    '있는'      9080   -100 - (prompt, -100)
  28   '방법이'     15517   -100 - (prompt, -100)
  29    '있을'      9846   -100 - (prompt, -100)
  30 '까요?\n'     15092   -100 - (prompt, -100)
  31    '\n'       375   -100 - (prompt, -100)
  32     '#'       378   -100 - (prompt, -100)
  33     '#'       378   -100 - (prompt, -100)
  34     '#'       378   -100 - (prompt, -100)
... (출력 145줄 생략) ...
```

```python
# 요약 시각화 - prompt vs response 토큰 수, loss 기여 비율
n_prompt = len(labels) - n_learn
n_resp = n_learn

fig, ax = plt.subplots(figsize=(9, 1.8))
ax.barh([0], [n_prompt], color="lightgray", edgecolor="gray",
        label=f"prompt (가림, -100): {n_prompt} tokens")
ax.barh([0], [n_resp], left=[n_prompt], color="tab:green", edgecolor="darkgreen",
        label=f"response (학습됨): {n_resp} tokens")
ax.set_yticks([])
ax.set_xlabel("토큰 위치")
ax.set_title("SFT labels: prompt 은 가리고 (-100), response 만 loss 에 기여")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2)
plt.tight_layout(); plt.show()
```

**▶ 실행 결과**

![output](../assets/28-sft-out1.png)

```python
from trl import SFTTrainer, SFTConfig

# SFT 학습 전 generation 비교를 위해 '학습 전' 모델 상태를 기록해 둠 (§5 에서 사용)
PROMPTS = [
    "피보나치 수열을 설명해줘",
    "건강한 식습관 3가지를 알려줘",
    "파이썬으로 리스트를 뒤집는 방법은?",
    "아침에 일찍 일어나는 팁을 알려줘",
]
GEN_KWARGS = dict(max_new_tokens=80, do_sample=True, temperature=0.8,
                  top_k=50, repetition_penalty=1.3)


@torch.no_grad()
def generate_answer(active_model, instruction: str, **kwargs):
    '''instruction 을 포맷해 답변을 생성. RESPONSE_TEMPLATE 뒤부터를 답변으로 디코드.'''
    text = build_prompt(instruction)
    enc = tokenizer(text, return_tensors="pt").to(active_model.device)
    out = active_model.generate(
        **enc,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # 답변 부분만 잘라내기 (response_template 이후)
    if RESPONSE_TEMPLATE.strip() in full:
        return full.split(RESPONSE_TEMPLATE.strip(), 1)[-1].strip()
    return full[len(text):].strip()


torch.manual_seed(SEED)
model.eval()
before_outputs = []
print("=" * 70)
print("BEFORE SFT - raw KoGPT2 (no instruction tuning yet)")
print("=" * 70)
for p in PROMPTS:
    ans = generate_answer(model, p, **GEN_KWARGS)
    before_outputs.append(ans)
    print(f"\n[instruction] {p}")
    print(f"[answer] {ans[:240]}")
```

**▶ 실행 결과**

```text
======================================================================
BEFORE SFT - raw KoGPT2 (no instruction tuning yet)
======================================================================
[instruction] 피보나치 수열을 설명해줘
[answer] 일단 한 번만 들어주면 끝나요
이제 본격적으로 사용하셔야겠죠?
다음부터는 내가 쓰는게 다인 듯!
내 안에 있는 피보라인의 모든 부분을 소개해드려요! momeljae.eats & pet_bang bong.
#미소천사 님이네요.
아무튼 저는 매일 미소에 대한
[instruction] 건강한 식습관 3가지를 알려줘
[answer] #diet #dieter #dietfood #eatclean <16.01.13.Sun>  
오늘은 정말 맛있는 날!
오랜만에 먹는 떡볶이가 나왔는데~ 진짜 너무 맛있었다
그리고 빵투샷도 있네용ᄒᄒ!ᄏᄏᄏ 대박이어서
다음에 또 먹어야지염
[instruction] 파이썬으로 리스트를 뒤집는 방법은?
[answer] 이벤트 응모 이벤트도 진행중이라, 오늘부터 이벤트에 신청하면 추첨을 통하여
2人1파이어보틀 세트를 선물로 받을 수 있는데,
(당첨된사람은 모두 파운데이션)
그래서인지 구매를 하면 제일 먼저 할인이 되는거 같아요~!
아니면 다들 미리미리 준비해서 갔는데...
그냥
[instruction] 아침에 일찍 일어나는 팁을 알려줘
[answer] crutsof_blogger.co.kr
오늘도 일상이 너무 행복해서요
이제는 아침을 거르지 않고 집에서 운동장 가기!
일단 점심시간이고 퇴근~~
우리 가족끼리 가자고 했다가 저는 저녁까지 먹고 싶다! 이젠 더 이상 뭐, 어떡해?.....
미세먼지
```

```python
sft_config = SFTConfig(
    output_dir="./out_kogpt2_sft",
    num_train_epochs=1,                     # SFT 는 1-3 epoch 이 표준 - T4 룰 안에서 1
    per_device_train_batch_size=2,          # KoGPT2 125M + instruction 은 시퀀스가 길어 작게
    gradient_accumulation_steps=8,          # effective batch = 16
    learning_rate=2e-5,                     # SFT 표준 lr
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    max_length=512,                         # instruction + response 길이 상한
    completion_only_loss=True,              # <- 핵심: 답변 부분만 loss (prompt = -100)
    packing=False,                          # 샘플 경계 유지 (마스킹이 정확하려면 packing 끔)
    fp16=USE_FP16,                          # T4 는 bf16 불가
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=2,
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

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=sft_ds,
    processing_class=tokenizer,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== SFT summary ===")
print(f"elapsed     : {elapsed/60:.2f} min")
print(f"global_step : {train_out.global_step}")
print(f"train_loss  : {train_out.training_loss:.4f}")
if torch.cuda.is_available():
    print(f"final peak  : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
```

**▶ 실행 결과**

```text
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
[transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
<IPython.core.display.HTML object>
=== SFT summary ===
elapsed     : 2.40 min
global_step : 188
train_loss  : 3.7007
final peak  : 1453 MiB
```

```python
log = trainer.state.log_history
train_pts = [(r["step"], r["loss"]) for r in log if "loss" in r and "eval_loss" not in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

if train_pts:
    ax1.plot([s for s, _ in train_pts], [l for _, l in train_pts], "-",
             color="tab:blue", alpha=0.8, label="train (response 만)")
ax1.set_xlabel("step"); ax1.set_ylabel("cross-entropy loss (response 토큰만)")
ax1.set_title("KoGPT2 SFT (KoAlpaca) - loss")
ax1.grid(True, alpha=0.3); ax1.legend()

if vram_cb.steps:
    ax2.plot(vram_cb.steps, vram_cb.peak_MiB, "o-", color="tab:green",
             label="최대 VRAM (로그 구간별)")
    ax2.set_title("VRAM trace  (bs=2, grad_accum=8, fp16)")
else:
    ax2.text(0.5, 0.5, "VRAM 추적은 CUDA 에서만 가능",
             ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("VRAM 추적 - CUDA 전용")
ax2.set_xlabel("step"); ax2.set_ylabel("VRAM (MiB)")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()
```

**▶ 실행 결과**

![output](../assets/28-sft-out2.png)

```python
torch.manual_seed(SEED)
model.eval()
after_outputs = []
print("=" * 70)
print("AFTER SFT - KoGPT2 + KoAlpaca instruction tuning")
print("=" * 70)
for p in PROMPTS:
    ans = generate_answer(model, p, **GEN_KWARGS)
    after_outputs.append(ans)
    print(f"\n[instruction] {p}")
    print(f"[answer] {ans[:240]}")
```

**▶ 실행 결과**

```text
======================================================================
AFTER SFT - KoGPT2 + KoAlpaca instruction tuning
======================================================================
[instruction] 피보나치 수열을 설명해줘
[answer] 수열이란 어떤 원인에 의해서 생기는 열입니다.
이러한 현상은 두 가지 종류가 있습니다. 첫째는 물 또는 공기 중에 포함된 물이 모두 다 녹아서 수증기가 만들어지는 현상이고, 둘째는 물에 녹아있는 산소가 열을 가해서 증발하는 현상입니다. 이 때, 물은 대부분 산화되지 않도록 보호 작용을 합니다. 
또한, 물을 끓여 끓이면 산소를 흡수하여 수분이 되어 다시 용해되기 때문에 물과 산소도 함께 배출되어 열이 발생하게 됩니다. 이러한
[instruction] 건강한 식습관 3가지를 알려줘
[answer] 1. 건강한 식사를 위해서는 균형과 식이섭취가 중요합니다. 아침식사는 탄수화물과 단백질은 풍부하지만, 단백질이 부족하면 소화도 잘 되지 않는 경우가 있습니다. 따라서 적당한 양의 영양소를 섭취하는 것이 좋으며, 저녁은 신선한 채소와 함께 먹는 것을 추천드립니다. 
2. 단백질을 풍부한 음식으로 만들기 위해 필요한 비타민 C, E는 필수적이며, 단백질 대신 과일과 채소를 풍부하게 먹어야 합니다. 또한, 칼슘을 충분히 함유하고 있
[instruction] 파이썬으로 리스트를 뒤집는 방법은?
[answer] 리스트에 파일 이름을 등록하여 해당 페이지에 접속한 뒤, 그 계정을 열지 않고 다시 연결하면, 파일을 열어보내면 된다.
- 리스트는 'inficitecture guide' 또는 '이디렉티브', '이미지(Deady Leak)'입니다.
- 이 디바이스는 P2P 사이트인 링크드인을 통해 제공되며, 현재까지는
[instruction] 아침에 일찍 일어나는 팁을 알려줘
[answer] 1. 아침에 일어나서 가장 먼저 하는 것은 아침체온 관리입니다.
2. 저녁에 잠자리에 들기 전에 꼭 하고 싶은 것이 있다면 아침을 먹고 하루를 시작하는 것입니다.
3. 아침에 일어날 때는 식욕을 억제해 몸의 신진대사를 활발하게 합니다.
4. 오후에는 몸을 따뜻하게 하기 위해 비타민이 풍부한 영양소를 보충합니다.
5. 낮 동안에는 잠을 자지 않는 습관을 가지면 좋습니다. 
6. 밤에 옷을 입는 것도 권장됩니다.
```

```python
# BEFORE vs AFTER 나란히 비교
print("=" * 80)
print("BEFORE SFT (raw KoGPT2) vs AFTER SFT (KoGPT2 + KoAlpaca) - instruction following")
print("=" * 80)
comparison = []
for p, before, after in zip(PROMPTS, before_outputs, after_outputs):
    print(f"\nINSTRUCTION : {p}")
    print("-" * 80)
    print(f"BEFORE      : {before[:300]}")
    print(f"AFTER       : {after[:300]}")
    comparison.append({
        "instruction": p,
        "before (raw)": before[:80] + ("..." if len(before) > 80 else ""),
        "after (sft)": after[:80] + ("..." if len(after) > 80 else ""),
    })

print("\n\n=== compact table ===")
print(pd.DataFrame(comparison).to_string(index=False))
```

**▶ 실행 결과**

```text
================================================================================
BEFORE SFT (raw KoGPT2) vs AFTER SFT (KoGPT2 + KoAlpaca) - instruction following
================================================================================

INSTRUCTION : 피보나치 수열을 설명해줘
--------------------------------------------------------------------------------
BEFORE      : 일단 한 번만 들어주면 끝나요
이제 본격적으로 사용하셔야겠죠?
다음부터는 내가 쓰는게 다인 듯!
내 안에 있는 피보라인의 모든 부분을 소개해드려요! momeljae.eats & pet_bang bong.
#미소천사 님이네요.
아무튼 저는 매일 미소에 대한
AFTER       : 수열이란 어떤 원인에 의해서 생기는 열입니다.
이러한 현상은 두 가지 종류가 있습니다. 첫째는 물 또는 공기 중에 포함된 물이 모두 다 녹아서 수증기가 만들어지는 현상이고, 둘째는 물에 녹아있는 산소가 열을 가해서 증발하는 현상입니다. 이 때, 물은 대부분 산화되지 않도록 보호 작용을 합니다. 
또한, 물을 끓여 끓이면 산소를 흡수하여 수분이 되어 다시 용해되기 때문에 물과 산소도 함께 배출되어 열이 발생하게 됩니다. 이러한

INSTRUCTION : 건강한 식습관 3가지를 알려줘
--------------------------------------------------------------------------------
BEFORE      : #diet #dieter #dietfood #eatclean <16.01.13.Sun>  
오늘은 정말 맛있는 날!
오랜만에 먹는 떡볶이가 나왔는데~ 진짜 너무 맛있었다
그리고 빵투샷도 있네용ᄒᄒ!ᄏᄏᄏ 대박이어서
다음에 또 먹어야지염
AFTER       : 1. 건강한 식사를 위해서는 균형과 식이섭취가 중요합니다. 아침식사는 탄수화물과 단백질은 풍부하지만, 단백질이 부족하면 소화도 잘 되지 않는 경우가 있습니다. 따라서 적당한 양의 영양소를 섭취하는 것이 좋으며, 저녁은 신선한 …(뒤 21자 생략)
2. 단백질을 풍부한 음식으로 만들기 위해 필요한 비타민 C, E는 필수적이며, 단백질 대신 과일과 채소를 풍부하게 먹어야 합니다. 또한, 칼슘을 충분히 함유하고 있으므로

INSTRUCTION : 파이썬으로 리스트를 뒤집는 방법은?
--------------------------------------------------------------------------------
BEFORE      : 이벤트 응모 이벤트도 진행중이라, 오늘부터 이벤트에 신청하면 추첨을 통하여
2人1파이어보틀 세트를 선물로 받을 수 있는데,
(당첨된사람은 모두 파운데이션)
그래서인지 구매를 하면 제일 먼저 할인이 되는거 같아요~!
아니면 다들 미리미리 준비해서 갔는데...
그냥
AFTER       : 리스트에 파일 이름을 등록하여 해당 페이지에 접속한 뒤, 그 계정을 열지 않고 다시 연결하면, 파일을 열어보내면 된다.
- 리스트는 'inficitecture guide' 또는 '이디렉티브', '이미지(Deady Leak)'입니다.
- 이 디바이스는 P2P 사이트인 링크드인을 통해 제공되며, 현재까지는

INSTRUCTION : 아침에 일찍 일어나는 팁을 알려줘
... (출력 21줄 생략) ...
```
