> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/33_diffusion_train/33_diffusion_train.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
%pip install -q -U transformers tokenizers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 108.5 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/555.1 kB ? eta -:--:--
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 49.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 39.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━ 42.7/48.9 MB 261.2 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 285.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 285.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 19.2 MB/s eta 0:00:00
```

```python
import math, time, torch
import torch.nn.functional as F
from datasets import load_dataset

SEED = 42
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()
print("torch", torch.__version__, "| device", device, "| fp16", USE_FP16)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

**▶ 실행 결과**

```text
torch 2.11.0+cu128 | device cuda | fp16 True
GPU: Tesla T4
```

### TinyStories 로드 (Ch 24/26과 같은 데이터)

작은 모델을 살릴 만큼의 분량으로 학습/검증 split을 잘라 옵니다. 앞 100,000편을 학습에, 검증 500편을 평가에 씁니다.

```python
raw_train = load_dataset("roneneldan/TinyStories", split="train[:100000]")
raw_val   = load_dataset("roneneldan/TinyStories", split="validation[:500]")
print(raw_train)
print(raw_val[0]["text"][:160])
```

**▶ 실행 결과**

```text
Dataset({
    features: ['text'],
    num_rows: 100000
})
Spot. Spot saw the shiny car and said, "Wow, Kitty, your car is so bright and clean!" Kitty smiled and replied, "Thank you, Spot. I polish it every day."

After
```

### TinyStories에 BPE 2048 직접 학습 + `[MASK]`

작은 모델에 맞춰 vocab을 직접 학습합니다. `[MASK]`를 special token으로 더해 흡수형 마스킹에 씁니다.

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

VOCAB = 2048
def corpus_iter(bs=1000):
    for i in range(0, len(raw_train), bs):
        yield raw_train[i:i+bs]["text"]

_tk = Tokenizer(models.BPE(unk_token="[UNK]"))
_tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
_tk.decoder = decoders.ByteLevel()
_trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["[PAD]", "[UNK]", "[MASK]"])
_tk.train_from_iterator(corpus_iter(), trainer=_trainer)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=_tk, pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]")
print("vocab_size :", tokenizer.vocab_size)
print("mask_id    :", tokenizer.mask_token_id, "| pad_id:", tokenizer.pad_token_id)
print("sample tok :", tokenizer.tokenize("Once upon a time there was a little cat.")[:14])
```

**▶ 실행 결과**

```text
vocab_size : 2048
mask_id    : 2 | pad_id: 0
sample tok : ['ĠOnce', 'Ġupon', 'Ġa', 'Ġtime', 'Ġthere', 'Ġwas', 'Ġa', 'Ġlittle', 'Ġcat', '.']
```

### 토큰화 + `group_texts` (BLOCK_SIZE=128)

문장 경계를 무시하고 토큰을 길게 이어 붙인 뒤 `BLOCK_SIZE=128` 단위로 잘라, 길이가 고정된 학습 청크를 만듭니다. 특수 토큰 없이 토큰화한 뒤 이어 붙이므로 청크 하나가 여러 동화에 걸칠 수 있습니다.

```python
BLOCK_SIZE = 128
def tok_fn(b):
    return tokenizer(b["text"], add_special_tokens=False)
tt = raw_train.map(tok_fn, batched=True, remove_columns=raw_train.column_names, desc="tok train")
tv = raw_val.map(tok_fn, batched=True, remove_columns=raw_val.column_names, desc="tok val")

def group_texts(b):
    cat = sum(b["input_ids"], [])
    n = (len(cat) // BLOCK_SIZE) * BLOCK_SIZE
    return {"input_ids": [cat[i:i+BLOCK_SIZE] for i in range(0, n, BLOCK_SIZE)]}
lm_train = tt.map(group_texts, batched=True, remove_columns=tt.column_names, desc="group train")
lm_val   = tv.map(group_texts, batched=True, remove_columns=tv.column_names, desc="group val")
print(f"train chunks {len(lm_train):,} | val {len(lm_val):,} | approx {len(lm_train)*BLOCK_SIZE/1e6:.2f}M tokens")
```

**▶ 실행 결과**

```text
train chunks 189,030 | val 853 | approx 24.20M tokens
```

### Diffusion collator — 매 배치 가변 비율 마스킹

`t ~ U(0.02, 1)`로 마스킹 비율을 뽑고(하한 절단), 가린 자리만 학습 신호로 둡니다.

```python
class DiffusionCollator:
    def __init__(self, tok, eps=0.02, seed=SEED):
        self.mask_id = tok.mask_token_id
        self.eps = eps
        self.gen = torch.Generator().manual_seed(seed)   # Trainer seed 와 분리
    def __call__(self, examples):
        ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        B, L = ids.shape
        t = torch.rand(B, generator=self.gen) * (1.0 - self.eps) + self.eps
        mask = torch.rand(B, L, generator=self.gen) < t.unsqueeze(1)
```

**위 코드 읽기** — 샘플마다 마스킹 비율 `t`를 `U(0.02, 1)`에서 따로 뽑습니다(`* (1.0 - self.eps) + self.eps`로 하한 절단). 그리고 각 자리를 독립적으로 확률 `t`로 가릴지 정해 `mask`를 만듭니다. 같은 배치라도 행마다 가려지는 비율이 다릅니다.

```python
        no = ~mask.any(dim=1)
        if no.any():
            j = torch.randint(0, L, (int(no.sum()),), generator=self.gen)
            mask[no, j] = True
```

**위 코드 읽기** — `t`가 매우 작아 한 자리도 안 가려진 행(`no`)은 학습 신호가 0이 됩니다. 그런 행은 무작위 한 자리(`j`)를 강제로 가려, 모든 샘플이 최소 하나의 손실 항을 갖게 합니다.

```python
        inp = ids.clone(); inp[mask] = self.mask_id
        lab = ids.clone(); lab[~mask] = -100
        return {"input_ids": inp, "attention_mask": torch.ones(B, L, dtype=torch.long),
                "labels": lab, "t": t}
coll = DiffusionCollator(tokenizer)
print("collator ready, mask_id =", coll.mask_id)
```

**위 코드 읽기** — 입력은 가린 자리를 `mask_id`로 덮고(`inp[mask] = self.mask_id`), 라벨은 가리지 않은 자리를 `-100`으로 두어 손실에서 제외합니다(`lab[~mask] = -100`). 뽑은 `t`도 함께 반환해 손실의 `1/t` 시간가중에 씁니다.

**▶ 실행 결과**

```text
collator ready, mask_id = 2
```

### 작은 BERT-MLM 모델 (Ch 24와 동급, 본체는 그대로)

`BertForMaskedLM`을 흡수형 diffusion 백본으로 재활용합니다. hidden 256 / 4층 / 4헤드로 작게 두고, vocab을 2048로 줄인 덕에 임베딩이 차지하던 비중이 크게 떨어집니다.

```python
from transformers import BertConfig, BertForMaskedLM
cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=256, num_hidden_layers=4,
                 num_attention_heads=4, intermediate_size=1024,
                 max_position_embeddings=BLOCK_SIZE, pad_token_id=tokenizer.pad_token_id)
model = BertForMaskedLM(cfg).to(device)
np_ = model.num_parameters()
emb = tokenizer.vocab_size * cfg.hidden_size
print(f"#params {np_/1e6:.2f}M | embedding share {emb/np_:.1%}  (Ch32: ~70%)")
```

**▶ 실행 결과**

```text
#params 3.79M | embedding share 13.9%  (Ch32: ~70%)
```

**결과 해석** — 전체 3.79M 중 임베딩 비중이 13.9%로, vocab 30522일 때의 약 70%에서 크게 줄었습니다. 아낀 용량이 그대로 본체로 돌아가 문맥 추론에 쓰입니다.

### 시간가중 `1/t` 손실로 30000 step 학습

손실은 마스크된 자리에만 교차엔트로피를 매기고, 거기에 `1/t` 시간가중을 곱하는 흡수형 NELBO입니다. `Trainer.compute_loss`를 오버라이드해 직접 구현합니다.

```python
from transformers import Trainer, TrainingArguments

class DiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        t = inputs["t"]; labels = inputs["labels"]
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        B, L, V = out.logits.shape
        per = F.cross_entropy(out.logits.view(-1, V), labels.view(-1),
                              ignore_index=-100, reduction="none").view(B, L)
        loss = ((per.sum(dim=1) / L) / t.to(per.dtype)).mean()
        return (loss, out) if return_outputs else loss
```

**위 코드 읽기** — `reduction="none"`으로 자리별 CE를 따로 받고(`ignore_index=-100`이 마스크 안 된 자리를 자동으로 0 처리), `(B, L)`로 되돌립니다. 핵심은 `((per.sum(dim=1) / L) / t...).mean()` — 샘플별 CE 합을 길이 `L`로 나눈 뒤 다시 `t`로 나눠 `1/t` 시간가중을 건 흡수형 NELBO입니다.

```python
args = TrainingArguments(
    output_dir="./out33", max_steps=30000,
    per_device_train_batch_size=64, per_device_eval_batch_size=64,
    learning_rate=3e-4, weight_decay=0.01, warmup_steps=500,
    lr_scheduler_type="cosine", max_grad_norm=1.0, fp16=USE_FP16,
    logging_steps=250, eval_strategy="steps", eval_steps=2000, save_strategy="no",
    report_to="none", label_names=["labels"], remove_unused_columns=False, seed=SEED)
```

**위 코드 읽기** — diffusion은 자리당 학습 신호가 희박해 `max_steps=30000`으로 길게 돕니다(AR의 약 20배). `fp16=USE_FP16`로 T4에서 메모리·속도를 확보하고, `remove_unused_columns=False`로 collator가 넘긴 `t` 컬럼이 버려지지 않게 막습니다.

```python
trainer = DiffusionTrainer(model=model, args=args, train_dataset=lm_train,
                           eval_dataset=lm_val, data_collator=coll)
t0 = time.time(); r = trainer.train(); el = (time.time()-t0)/60
print(f"\n=== summary ===\nelapsed {el:.2f} min | step {r.global_step} | train_loss {r.training_loss:.4f}")
print(f"random baseline ln(V) = {math.log(tokenizer.vocab_size):.4f}")
if torch.cuda.is_available():
    print(f"peak VRAM {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
```

**위 코드 읽기** — 커스텀 collator를 `data_collator`로 넘겨 매 배치 가변 마스킹이 적용됩니다. 학습이 끝나면 train_loss를 무작위 기준선 `ln(V)`와 비교해, 모델이 단순 빈도 모사를 넘어섰는지 한눈에 확인합니다.

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
=== summary ===
elapsed 18.50 min | step 30000 | train_loss 3.5916
random baseline ln(V) = 7.6246
peak VRAM 627 MiB
```

**결과 해석** — 30000 step을 18.5분에 끝내 T4 30분 예산 안에 들어옵니다. train_loss 3.59는 무작위 기준선 `ln(V)=7.62`의 절반 이하로, 모델이 유니그램 빈도 모사를 넘어 조건부 구조를 학습했다는 신호입니다. peak VRAM도 627 MiB로 여유가 큽니다.

### carry-over 샘플러로 생성

전부 `[MASK]`에서 시작해 블록 단위로 채웁니다. 기본값은 반복억제 설정(temperature 0.8 · top_p 0.92 · rep penalty 1.3 · 인접중복 금지)입니다.

```python
@torch.no_grad()
def generate(model, length=128, block=32, temperature=0.8, top_p=0.92, top_k=0,
             rep_penalty=1.3, no_immediate_repeat=True, prompt_ids=None):
    """carry-over semi-AR + 반복 억제(rep penalty / 인접중복 금지 / top-p)."""
    model.eval()
    mask_id = tokenizer.mask_token_id
    x = torch.full((1, length), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros(length, dtype=torch.bool, device=device)
    if prompt_ids is not None:
        p = torch.tensor(prompt_ids[:length], device=device)
        x[0, :len(p)] = p; fixed[:len(p)] = True
    nblocks = (length + block - 1) // block
    for b in range(nblocks):
        lo, hi = b * block, min((b + 1) * block, length)
        steps = hi - lo
        for s in range(steps):
            logits = model(input_ids=x).logits[0].float()        # (L, V)
            logits[:, mask_id] = -1e9
```

**위 코드 읽기** — 전부 `mask_id`인 상태에서 시작해(`x = torch.full(...)`), 32 토큰 `block` 단위로 왼→오 진행합니다. 매 step 전체를 다시 예측하되 생성 결과에 `[MASK]`가 나오지 않게 `logits[:, mask_id] = -1e9`로 막습니다. 프롬프트가 있으면 `fixed`로 고정해 건드리지 않습니다.

```python
            # 반복 패널티: 이미 확정된 토큰들의 로짓을 깎음
            if rep_penalty and rep_penalty != 1.0:
                comm = x[0][x[0] != mask_id]
                if comm.numel() > 0:
                    u = torch.unique(comm)
                    col = logits[:, u]
                    logits[:, u] = torch.where(col > 0, col / rep_penalty, col * rep_penalty)
            # 인접중복 금지: 각 자리에서 '왼쪽 토큰과 같은 토큰' 예측 차단
            if no_immediate_repeat:
                left = torch.roll(x[0], 1); left[0] = mask_id
                valid = left != mask_id
                logits[valid, left[valid]] = -1e9
```

**위 코드 읽기** — 반복 억제 두 장치입니다. `rep_penalty`는 이미 확정된 토큰(`u`)의 로짓을 깎아 같은 단어가 다시 뽑힐 확률을 낮추고(부호에 따라 나누거나 곱함), `no_immediate_repeat`는 바로 왼쪽 토큰(`left`)과 같은 예측을 `-1e9`로 막아 `the the` 같은 인접 중복을 봉쇄합니다.

```python
            probs = (logits / max(temperature, 1e-6)).softmax(-1)
            if top_k and top_k > 0:
                kth = probs.topk(top_k, dim=-1).values[:, -1, None]
                probs = probs.masked_fill(probs < kth, 0.0)
            if top_p and top_p < 1.0:
                sp, si = probs.sort(dim=-1, descending=True)
                rm = (sp.cumsum(-1) - sp) > top_p
                sp = sp.masked_fill(rm, 0.0)
                probs = torch.zeros_like(probs).scatter(-1, si, sp)
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)
            pred = torch.multinomial(probs, 1).squeeze(-1)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
```

**위 코드 읽기** — `temperature`로 분포를 날카롭게 한 뒤 top-k·top-p로 꼬리를 자르고, 정규화해 자리마다 토큰을 샘플링합니다(`pred`). 동시에 그 자리의 확신도 `conf`(뽑힌 토큰의 확률)도 챙겨, 다음에서 어디를 확정할지 고르는 데 씁니다.

```python
            cur = (x[0] == mask_id) & (~fixed)
            cur[:lo] = False; cur[hi:] = False
            nleft = int(cur.sum())
            if nleft == 0: break
            nreveal = nleft if s == steps - 1 else max(1, nleft // (steps - s))
            cc = conf.clone(); cc[~cur] = -1e9
            idx = cc.topk(nreveal).indices
            x[0, idx] = pred[idx]
    return tokenizer.decode(x[0], skip_special_tokens=True)
```

**위 코드 읽기** — carry-over의 핵심입니다. 아직 마스크인 자리(`cur`) 중 확신도(`cc`)가 높은 곳부터 `nreveal`개만 골라 확정하고(`x[0, idx] = pred[idx]`), 한 번 확정한 토큰은 다시 마스크로 되돌리지 않습니다. block 마지막 step에서는 남은 자리를 전부 확정해 빈칸 없이 마무리합니다.

```python
pid = tokenizer("Once upon a time", add_special_tokens=False)["input_ids"]
torch.manual_seed(SEED)
print("=== unconditional (all-[MASK] -> generate, default sampler) ===")
for i in range(3):
    print(f"[{i}] {generate(model)[:340]}")
print("\n=== conditional (prompt 'Once upon a time' fixed) ===")
for i in range(3):
    print(f"[{i}] {generate(model, prompt_ids=pid)[:340]}")
```

**▶ 실행 결과**

```text
=== unconditional (all-[MASK] -> generate, default sampler) ===
[0]  say, "Yes, Ben. We have a ball. They are very good friends."

"They go to the park and play," Lily says.
Ben follows his mom's house. He hopes to play with their toys again. She is happy and happy.
Lily smiles at her. She shows him back to Tom and kiss. She gives Anna to his dad. She hugs her. She says, "Thank you, I love me. You're welc
[1]  you want to play with me." She looked at Tom and put her toys in the room. 

"It's okay, Lily! I'm sorry for you. But you don't know that we should have a good friend. You are not nice. And they can share your dolls o …(뒤 122자 생략)
[2] , so it zoomed in the air! He got very scared, but it started to run away. 

The boy was sad and wished he had never been here for being playing with his friends. They knew that they would play together again. Once up …(뒤 123자 생략)

=== conditional (prompt 'Once upon a time' fixed) ===
[0]  Once upon a time, there was a little girl named Lily. She loved to play with her toys and play with her mommy and her friends. One day, …(뒤 83자 생략)

Lily's mom said, "Let's go inside!" Her mommy replied, "Yes, we can slide together and have fun after you." 
As they we
[1]  Once upon a time, there was a little girl named Lily. She loved to play outside and watch her friends. One day, she went for bed with her mom.
 
Lily's mommy said, "I want to go inside the park!" Her mom replied, "Yes, I can do it." So, they walked home with their toys. They were so h …(뒤 45자 생략)

After a
[2]  Once upon a time, there was a little girl named Lily. She loved to play with her toys and run in the park together. One day, she saw a …(뒤 106자 생략)

Lily's mom replied, "I'm sorry! I didn't know what to do." 

Her mom explained, "Why don't have 
```

**결과 해석** — 작은 3.79M 모델·짧은 학습이라 완벽하진 않아 "She is happy and happy", "Why don't have"처럼 어색한 구절이 남습니다. 그래도 인물(Lily, Ben)·대화·배경이 이어지는 동화가 나오고, 특히 조건부 생성은 프롬프트 "Once upon a time" 뒤로 일관된 이야기를 펼쳐 모델이 조건부 구조를 학습했음을 보여줍니다.
