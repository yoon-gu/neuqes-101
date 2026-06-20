> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/34_ko_diffusion/34_ko_diffusion.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

80/10/10 콜레이터로 작은 모델을 학습시켜 한국어 동화를 생성해 봅니다. (80/10/10이 무엇인지는 위 마스킹 노트에서 봤고, 순진한 100% `[MASK]`가 왜 무너지는지는 뒤 🔬 해부에서 다룹니다.) T4에서 30000 step, 약 20분.

```python
%pip install -q -U transformers tokenizers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 119.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 50.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 40.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━ 22.4/48.9 MB 219.4 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 171.3 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 171.3 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 22.2 MB/s eta 0:00:00
```

```python
import math, time, torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
SEED=42; torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()
print("device", device, "| fp16", USE_FP16)
```

**▶ 실행 결과**

```text
device cuda | fp16 True
```

### 한국어 TinyStories 복원 (Ch 26과 동일)

```python
EOT="<|endoftext|>"; N_TRAIN,N_VAL,MAXL=50_000,500,1_500_000
def rebuild(split,n,maxl):
    stories,buf=[],[]
    for i,ex in enumerate(load_dataset("g0ster/TinyStories-Korean",split=split,streaming=True)):
        if i>=maxl or len(stories)>=n: break
        line=(ex["text"] or "").strip()
        if line==EOT:
            s=" ".join(buf).strip()
            if s: stories.append(s)
            buf=[]
        elif line: buf.append(line)
    if buf and len(stories)<n:
        s=" ".join(buf).strip()
        if s: stories.append(s)
    return stories[:n]
raw_train=Dataset.from_dict({"text":rebuild("train",N_TRAIN,MAXL)})
raw_val=Dataset.from_dict({"text":rebuild("validation",N_VAL,50_000)})
print("stories", len(raw_train), len(raw_val))
print(raw_train[0]["text"][:120])
```

**▶ 실행 결과**

```text
stories 50000 500
한때 벤이라는 이름의 어린 소년이 있었어요. 벤은 주변 세계를 탐험하는 것을 좋아했답니다. 그는 가게에 전시되어 있던 아름다운 꽃병들 같은 멋진 것들을 많이 봤어요. 어느 날, 벤은 가게를 거닐다가 정말 특별한 꽃병
```

### BPE 4000 + initial_alphabet + [MASK]

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
VOCAB=4000
def corpus_iter(bs=1000):
    for i in range(0,len(raw_train),bs): yield raw_train[i:i+bs]["text"]
_tk=Tokenizer(models.BPE(unk_token="[UNK]"))
_tk.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
_tk.decoder=decoders.ByteLevel()
_tk.train_from_iterator(corpus_iter(), trainer=trainers.BpeTrainer(
    vocab_size=VOCAB, special_tokens=["[PAD]","[UNK]","[MASK]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tokenizer=PreTrainedTokenizerFast(tokenizer_object=_tk, pad_token="[PAD]", unk_token="[UNK]", mask_token="[MASK]")
print("vocab", tokenizer.vocab_size, "mask_id", tokenizer.mask_token_id)
```

**▶ 실행 결과**

```text
vocab 4000 mask_id 2
```

### 토큰화 + group

```python
BLOCK=128
tt=raw_train.map(lambda b: tokenizer(b["text"],add_special_tokens=False), batched=True, remove_columns=raw_train.column_names)
tv=raw_val.map(lambda b: tokenizer(b["text"],add_special_tokens=False), batched=True, remove_columns=raw_val.column_names)
def group(b):
    cat=sum(b["input_ids"],[]); n=(len(cat)//BLOCK)*BLOCK
    return {"input_ids":[cat[i:i+BLOCK] for i in range(0,n,BLOCK)]}
lm_train=tt.map(group,batched=True,remove_columns=tt.column_names)
lm_val=tv.map(group,batched=True,remove_columns=tv.column_names)
print("chunks", len(lm_train), len(lm_val))
```

**▶ 실행 결과**

```text
chunks 75941 746
```

### 수정 — diffusion 콜레이터에 80/10/10 (순진한 100% [MASK] 대신)

가변 마스킹률 t는 유지(생성용)하되, 선택된 자리에 80% [MASK] / 10% 랜덤 / 10% 원본유지. 이게 모델이 [MASK]→유니그램 지름길로 새는 걸 막는다.

```python
N_SPECIAL=3
class DiffMLMCollator:
    def __init__(self, tok, eps=0.05, tmax=1.0, seed=SEED):
        self.mask_id=tok.mask_token_id; self.vocab=tok.vocab_size
        self.eps=eps; self.tmax=tmax; self.gen=torch.Generator().manual_seed(seed)
    def __call__(self, ex):
        ids=torch.tensor([e["input_ids"] for e in ex], dtype=torch.long)
        B,L=ids.shape
        t=torch.rand(B,generator=self.gen)*(self.tmax-self.eps)+self.eps
        sel=torch.rand(B,L,generator=self.gen)<t.unsqueeze(1)
        no=~sel.any(1)
        if no.any():
            j=torch.randint(0,L,(int(no.sum()),),generator=self.gen); sel[no,j]=True
        labels=ids.clone(); labels[~sel]=-100
        inp=ids.clone()
        r=torch.rand(B,L,generator=self.gen)
        inp[sel&(r<0.8)]=self.mask_id                                    # 80% [MASK]
        rp=sel&(r>=0.8)&(r<0.9); nr=int(rp.sum())                        # 10% 랜덤
        if nr: inp[rp]=torch.randint(N_SPECIAL,self.vocab,(nr,),generator=self.gen)
        # 10% 원본 유지
        return {"input_ids":inp,"attention_mask":torch.ones(B,L,dtype=torch.long),"labels":labels}
coll=DiffMLMCollator(tokenizer)
```

### 작은 모델 (256/4L) — 용량 아닌 마스킹이 문제였음

```python
from transformers import BertConfig, BertForMaskedLM
cfg=BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=256, num_hidden_layers=4,
               num_attention_heads=4, intermediate_size=1024,
               max_position_embeddings=BLOCK, pad_token_id=tokenizer.pad_token_id)
model=BertForMaskedLM(cfg).to(device)
print("params(M)", round(model.num_parameters()/1e6,2))
```

**▶ 실행 결과**

```text
params(M) 4.29
```

### 학습 — plain CE(BertForMaskedLM 기본) + lr 5e-4, 30000 step

```python
from transformers import Trainer, TrainingArguments
args=TrainingArguments(output_dir="./out34", max_steps=30000,
    per_device_train_batch_size=64, learning_rate=5e-4, weight_decay=0.01,
    warmup_steps=1000, lr_scheduler_type="cosine", max_grad_norm=1.0, fp16=USE_FP16,
    logging_steps=500, save_strategy="no", report_to="none", remove_unused_columns=False, seed=SEED)
trainer=Trainer(model=model, args=args, train_dataset=lm_train, data_collator=coll)
t0=time.time(); r=trainer.train()
print(f"elapsed {(time.time()-t0)/60:.2f}min | step {r.global_step} | train_loss {r.training_loss:.4f} | baseline ln(V) {math.log(tokenizer.vocab_size):.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
elapsed 20.20min | step 30000 | train_loss 4.1260 | baseline ln(V) 8.2940
```


**결과 해석**

순진한 100% `[MASK]`였다면 7.06(유니그램 벽)에서 멈췄을 텐데, 80/10/10 콜레이터로 바꾸자 train_loss가 baseline 8.29에서 4.13까지 내려갑니다. 모델이 비로소 한국어 문맥을 읽기 시작했다는 신호입니다.

### carry-over 샘플러로 한국어 생성

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
            cur = (x[0] == mask_id) & (~fixed)
            cur[:lo] = False; cur[hi:] = False
            nleft = int(cur.sum())
            if nleft == 0: break
            nreveal = nleft if s == steps - 1 else max(1, nleft // (steps - s))
            cc = conf.clone(); cc[~cur] = -1e9
            idx = cc.topk(nreveal).indices
            x[0, idx] = pred[idx]
    return tokenizer.decode(x[0], skip_special_tokens=True)

pid = tokenizer("옛날 옛날에", add_special_tokens=False)["input_ids"]
torch.manual_seed(SEED)
print("=== unconditional (all-[MASK] -> generate, default sampler) ===")
for i in range(3):
    print(f"[{i}] {generate(model)[:340]}")
print("\n=== conditional (prompt 'Once upon a time' fixed) ===")
for i in range(3):
    print(f"[{i}] {generate(model, prompt_ids=pid)[:340]}")

pid = tokenizer("옛날 옛날에", add_special_tokens=False)["input_ids"]
torch.manual_seed(SEED)
print("=== conditional ('옛날 옛날에') ===")
for i in range(3):
    print(f"[{i}] {generate(model, prompt_ids=pid)[:300]}")
print("\n=== unconditional ===")
for i in range(2):
    print(f"[{i}] {generate(model)[:300]}")
```

**▶ 실행 결과**

```text
=== unconditional (all-[MASK] -> generate, default sampler) ===
[0]  공원에 도착해서 맥스는 맥스라는 큰 공을 봤어요. "와, 정말 멋진 공이야! 우리 공으로 같이 놀자!" 맥스는 웃으며 "그래, 가자!"라고 말했어요. 그들은 함께 놀면서 재미있게 놀았어요. 결국, 릴리와 그녀의 친구들은 다시 즐겁게 놀 수 있 …(뒤 204자 생략)
[1] 옛날 옛적에, 팀이라는 작은 소년이 있었어요. 팀은 하루 종일 장난감 자동차를 가지고 노는 것을 좋아했죠. 맑은 어느 날, 팀은 마당에서 수 있는 가장 좋아하는 장난감을 발견했어요. 그는 그 자동차와 함께 놀고 싶어서 마당을 뛰어다니며 뛰어다녔 …(뒤 204자 생략)
[2]  장난감 자동차를 가지고 노는 것을 좋아했습니다. 어느 날, 팀은 가장 좋아하는 자동차와 놀고 있었습니다. 그는 자신의 자동차들과 함께 차를 운전하고 싶어 했습니다. 팀은 매우 빠른적이라고 생각했습니다. 그래서 그는 친구 수에게 갔습니다. 그녀 …(뒤 204자 생략)

=== conditional (prompt 'Once upon a time' fixed) ===
[0] 옛날 옛날에, 수라는 이름의 작은 소녀가 있었어요. 그녀는 장난감 자동차를 가지고 노는 것을 정말 좋아했지요. 어느 날, 수는 자신의 자동차에 큰 장난감 자동차를 발견했어요. 그 차는 아주 무거웠어요. 그래서 친구 수에게 그 자동차로 놀고 싶어 …(뒤 204자 생략)
[1] 옛날 옛날에, 안나라는 이름의 어린 소녀가 있었어요. 그녀는 가장 좋아하는 장난감과 예쁜 인형을 가지고 있었지요. 안나는 그 인형 가지고 노는 것을 정말 좋아했답니다. 어느 날, 안나가 공원에 있는 나무 밑에서 새로운 장난감을 발견했어요. 바로 …(뒤 204자 생략)
[2] 옛날 옛날에, 사라라는 이름의 작은 소녀가 있었어요. 그녀는 매우 사랑하는 아끼는 큰 인형을 가지고 있었지요. 사라는 친구들과 함께 노는 것을 좋아했죠. 어느 날, 사라가 자신의 인형에게 전화를 했어요. "봐, 내 인형이 네 장난감 자동차와 놀 …(뒤 204자 생략)
=== conditional ('옛날 옛날에') ===
[0] 옛날 옛날에, 큰 빨간 기차가 있었어요. 그 기차는 많은 색깔의 바퀴를 가지고 있었지요. 매일 파란 기차를 함께 노는 것을 좋아했답니다. 어느 날, 파란색 기차는 빨간색이고 노란색 기차를 발견했어요. 새도 그 차로를 고치고 싶어 했죠. 차는 매 …(뒤 164자 생략)
[1] 옛날 옛날에, 샐리라는 이름의 어린 소녀가 있었습니다. 샐리는 매우 독립적인 아이였죠. 어느 날, 그녀는 큰 성을 발견했어요. 그 용은 어디서 가고 있는지 보고 싶어 했습니다. 그래서 그는 자신의 성으로 달려가 예쁜 성과 용을 가지고 놀았어요. …(뒤 164자 생략)
[2] 옛날 옛날에, 루시라는 작은 소녀가 있었어요. 그녀는 매우 영리했어요. 그는 매일 춤추고 음악을 연주하는 것을 좋아했답니다 맑은 어느 날, 루시는 새로운 음악을 연주하고 싶어졌어요. 그래서 그녀는 친구들과 음악을 연주하며 음악을 듣기 위해 연주 …(뒤 164자 생략)

=== unconditional ===
[0]  알게 되었답니다!옛날 옛적에 팀이라는 작은 소년이 있었어요. 그는 자신의 장난감 자동차를 가지고 노는 것을 좋아했죠. 어느 날, 팀은 바닥에 놓고 있는 큰 상자를 발견했어요. 그 상자는 매우 궁금해졌죠. 팀은 그 자동차로 놀고 싶어 했죠. 그 …(뒤 164자 생략)
[1]  정말 좋은 일이야."옛날 옛적에 팀이라는 어린 소년이 있었어요. 그는 장난감 장난감을 가지고 노는 것을 좋아했지요. 어느 날, 팀은 방에서 놀 수 있는 큰 상자를 발견했어요. 그 상자에는 장난감이 들어있었답니다! 팀은 매우 신이 나서 엄마에게 …(뒤 164자 생략)
```


**결과 해석**

전부 `[MASK]`에서 시작한 무조건 생성인데도 인물(맥스·팀·수·안나)과 배경(공원·마당), 대화, 서사가 갖춰진 한국어 동화가 나옵니다. "맥스라는 큰 공"처럼 자잘한 흠은 4.29M 작은 모델의 한계이지만, 80/10/10 한 가지로 붕괴를 벗어났다는 건 분명합니다.

### 진단 — 고정-t(0.15) acc + infill

```python
g=torch.Generator().manual_seed(0)
def fixed_t_acc(tv_=0.15,n=128):
    cor=tot=0
    for ex in lm_val.select(range(min(n,len(lm_val)))):
        ids=torch.tensor(ex["input_ids"]); m=torch.rand(len(ids),generator=g)<tv_
        if not m.any(): m[0]=True
        inp=ids.clone(); inp[m]=tokenizer.mask_token_id
        with torch.no_grad(): pr=model(inp.unsqueeze(0).to(device)).logits[0].argmax(-1).cpu()
        cor+=(pr[m]==ids[m]).sum().item(); tot+=int(m.sum())
    return cor/tot
print(f"[diag] fixed-t(0.15) top-1 acc = {fixed_t_acc():.3f}   (naive diffusion 0.081)")
```

**▶ 실행 결과**

```text
[diag] fixed-t(0.15) top-1 acc = 0.652   (naive diffusion 0.081)
```


**결과 해석**

샘플러를 배제하고 모델만 잰 고정-t(0.15) top-1 정확도가 0.652입니다. 순진한 이식의 0.081(거의 찍기)에서 8배 넘게 뛰었고, 영어 Ch 33의 0.717에 근접합니다.
