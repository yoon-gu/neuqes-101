"""Ch 32 검증 (로컬 MPS) — 영어 diffusion(vocab 2048, 100% mask, 1/t loss)을
Ch 32 기본 샘플러(MaskGIT confidence remasking)로 생성. 기본 샘플러가 coherent한지 확인.
사용: python run_local_ch32.py [max_steps]"""
import math, time, sys, torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import (PreTrainedTokenizerFast, BertConfig, BertForMaskedLM,
                          Trainer, TrainingArguments)

SEED = 42; torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, flush=True)

# 1) 영어 TinyStories
raw_train = load_dataset("roneneldan/TinyStories", split="train[:50000]")
raw_val   = load_dataset("roneneldan/TinyStories", split="validation[:500]")
print("stories", len(raw_train), len(raw_val), flush=True)

# 2) BPE 2048 + [MASK]
VOCAB = 2048
def corpus_iter(bs=1000):
    for i in range(0, len(raw_train), bs): yield raw_train[i:i+bs]["text"]
_tk = Tokenizer(models.BPE(unk_token="[UNK]"))
_tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
_tk.decoder = decoders.ByteLevel()
_tk.train_from_iterator(corpus_iter(), trainer=trainers.BpeTrainer(
    vocab_size=VOCAB, special_tokens=["[PAD]", "[UNK]", "[MASK]"]))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tk, pad_token="[PAD]",
                                    unk_token="[UNK]", mask_token="[MASK]")
print("vocab", tokenizer.vocab_size, "mask_id", tokenizer.mask_token_id, flush=True)

# 3) group
BLOCK = 128
tt = raw_train.map(lambda b: tokenizer(b["text"], add_special_tokens=False), batched=True, remove_columns=raw_train.column_names)
tv = raw_val.map(lambda b: tokenizer(b["text"], add_special_tokens=False), batched=True, remove_columns=raw_val.column_names)
def group(b):
    cat = sum(b["input_ids"], []); n = (len(cat)//BLOCK)*BLOCK
    return {"input_ids": [cat[i:i+BLOCK] for i in range(0, n, BLOCK)]}
lm_train = tt.map(group, batched=True, remove_columns=tt.column_names)
lm_val   = tv.map(group, batched=True, remove_columns=tv.column_names)
print("chunks", len(lm_train), len(lm_val), flush=True)

# 4) 100% [MASK] collator (Ch 33과 동일 — 영어는 80/10/10 없이도 됨)
class DiffusionCollator:
    def __init__(self, tok, eps=0.02, seed=SEED):
        self.mask_id = tok.mask_token_id; self.eps = eps
        self.gen = torch.Generator().manual_seed(seed)
    def __call__(self, ex):
        ids = torch.tensor([e["input_ids"] for e in ex], dtype=torch.long)
        B, L = ids.shape
        t = torch.rand(B, generator=self.gen) * (1.0 - self.eps) + self.eps
        mask = torch.rand(B, L, generator=self.gen) < t.unsqueeze(1)
        no = ~mask.any(1)
        if no.any():
            j = torch.randint(0, L, (int(no.sum()),), generator=self.gen); mask[no, j] = True
        inp = ids.clone(); inp[mask] = self.mask_id
        lab = ids.clone(); lab[~mask] = -100
        return {"input_ids": inp, "attention_mask": torch.ones(B, L, dtype=torch.long), "labels": lab, "t": t}

cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=256, num_hidden_layers=4,
                 num_attention_heads=4, intermediate_size=1024,
                 max_position_embeddings=BLOCK, pad_token_id=tokenizer.pad_token_id)
model = BertForMaskedLM(cfg).to(device)
print("params(M)", round(model.num_parameters()/1e6, 2), flush=True)

class DiffTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        t = inputs["t"]; labels = inputs["labels"]
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        B, L, V = out.logits.shape
        per = F.cross_entropy(out.logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="none").view(B, L)
        loss = ((per.sum(1)/L) / t.to(per.dtype)).mean()
        return (loss, out) if return_outputs else loss

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
args = TrainingArguments(output_dir="./_drafts/ch32_validate/out", max_steps=MAX_STEPS,
    per_device_train_batch_size=64, learning_rate=3e-4, weight_decay=0.01,
    warmup_steps=500, lr_scheduler_type="cosine", max_grad_norm=1.0, fp16=False,
    logging_steps=500, save_strategy="no", report_to="none", label_names=["labels"],
    remove_unused_columns=False, seed=SEED)
trainer = DiffTrainer(model=model, args=args, train_dataset=lm_train, data_collator=DiffusionCollator(tokenizer))
t0 = time.time(); r = trainer.train()
print(f"\n=== elapsed {(time.time()-t0)/60:.2f}min | step {r.global_step} | train_loss {r.training_loss:.4f} | baseline ln(V) {math.log(tokenizer.vocab_size):.4f} ===", flush=True)

# 5) Ch 32 기본 샘플러 (MaskGIT confidence remasking)
@torch.no_grad()
def diffusion_generate(model, length=128, steps=16, temperature=1.0, top_k=50, prompt_ids=None):
    model.eval(); mask_id = tokenizer.mask_token_id
    x = torch.full((1, length), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros(length, dtype=torch.bool, device=device)
    if prompt_ids is not None:
        p = torch.tensor(prompt_ids[:length], device=device); x[0, :len(p)] = p; fixed[:len(p)] = True
    n_gen = int((~fixed).sum().item())
    for step in range(steps):
        logits = model(input_ids=x).logits[0]; probs = logits.softmax(-1)
        if temperature > 0:
            scaled = logits / temperature
            if top_k > 0:
                kth = scaled.topk(top_k, dim=-1).values[:, -1, None]; scaled = scaled.masked_fill(scaled < kth, float("-inf"))
            pred = torch.multinomial(scaled.softmax(-1), 1).squeeze(-1); conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        else:
            conf, pred = probs.max(-1)
        is_mask = (x[0] == mask_id) & (~fixed); x_new = torch.where(is_mask, pred, x[0])
        n_remain = int(round(n_gen * (1.0 - (step+1)/steps)))
        if n_remain > 0:
            cm = conf.clone(); cm[~is_mask] = float("inf")
            x_new[cm.topk(n_remain, largest=False).indices] = mask_id
        x[0] = x_new
    return tokenizer.decode(x[0], skip_special_tokens=True)

pid = tokenizer("Once upon a time", add_special_tokens=False)["input_ids"]
torch.manual_seed(SEED)
print("\n=== [기본 샘플러] unconditional (all-[MASK]) ===", flush=True)
for i in range(4): print(f"[{i}] {diffusion_generate(model)[:300]}", flush=True)
print("\n=== [기본 샘플러] conditional ('Once upon a time') ===", flush=True)
for i in range(2): print(f"[{i}] {diffusion_generate(model, prompt_ids=pid)[:300]}", flush=True)

# 6) 모델 품질
g = torch.Generator().manual_seed(0)
def fixed_t_acc(tv_=0.15, n=128):
    cor = tot = 0
    for ex in lm_val.select(range(min(n, len(lm_val)))):
        ids = torch.tensor(ex["input_ids"]); m = torch.rand(len(ids), generator=g) < tv_
        if not m.any(): m[0] = True
        inp = ids.clone(); inp[m] = tokenizer.mask_token_id
        with torch.no_grad():
            pr = model(inp.unsqueeze(0).to(device)).logits[0].argmax(-1).cpu()
        cor += (pr[m] == ids[m]).sum().item(); tot += int(m.sum())
    return cor/tot
print(f"\n[diag] fixed-t(0.15) top-1 acc = {fixed_t_acc():.3f}", flush=True)
