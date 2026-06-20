"""Ch 35 저-마스킹 실험 — 로컬 Mac MPS 실행. t<=0.30 컷이 한국어 조건부 학습을 살리는지.
사용: python run_local_mps.py [max_steps]"""
import math, time, sys, torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import (PreTrainedTokenizerFast, BertConfig, BertForMaskedLM,
                          Trainer, TrainingArguments)

SEED = 42
torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, flush=True)

# 1) 한국어 데이터 복원
EOT = "<|endoftext|>"; N_TRAIN, N_VAL, MAXL = 50_000, 500, 1_500_000
def rebuild(split, n, maxl):
    stories, buf = [], []
    for i, ex in enumerate(load_dataset("g0ster/TinyStories-Korean", split=split, streaming=True)):
        if i >= maxl or len(stories) >= n: break
        line = (ex["text"] or "").strip()
        if line == EOT:
            s = " ".join(buf).strip()
            if s: stories.append(s)
            buf = []
        elif line: buf.append(line)
    if buf and len(stories) < n:
        s = " ".join(buf).strip()
        if s: stories.append(s)
    return stories[:n]
t0 = time.time()
raw_train = Dataset.from_dict({"text": rebuild("train", N_TRAIN, MAXL)})
raw_val   = Dataset.from_dict({"text": rebuild("validation", N_VAL, 50_000)})
print(f"stories train={len(raw_train)} val={len(raw_val)} ({time.time()-t0:.1f}s)", flush=True)

# 2) BPE 4000 + [MASK] (initial_alphabet)
VOCAB = 4000
def corpus_iter(bs=1000):
    for i in range(0, len(raw_train), bs): yield raw_train[i:i+bs]["text"]
_tk = Tokenizer(models.BPE(unk_token="[UNK]"))
_tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
_tk.decoder = decoders.ByteLevel()
_tk.train_from_iterator(corpus_iter(), trainer=trainers.BpeTrainer(
    vocab_size=VOCAB, special_tokens=["[PAD]", "[UNK]", "[MASK]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tk, pad_token="[PAD]",
                                    unk_token="[UNK]", mask_token="[MASK]")
print("vocab", tokenizer.vocab_size, "mask_id", tokenizer.mask_token_id, flush=True)

# 3) 토큰화 + group
BLOCK = 128
tt = raw_train.map(lambda b: tokenizer(b["text"], add_special_tokens=False),
                   batched=True, remove_columns=raw_train.column_names)
tv = raw_val.map(lambda b: tokenizer(b["text"], add_special_tokens=False),
                 batched=True, remove_columns=raw_val.column_names)
def group(b):
    cat = sum(b["input_ids"], []); n = (len(cat)//BLOCK)*BLOCK
    return {"input_ids": [cat[i:i+BLOCK] for i in range(0, n, BLOCK)]}
lm_train = tt.map(group, batched=True, remove_columns=tt.column_names)
lm_val   = tv.map(group, batched=True, remove_columns=tv.column_names)
print("chunks train", len(lm_train), "val", len(lm_val), flush=True)

# 4) collator — 고-마스킹 컷 t in [0.02, 0.30]
class Coll:
    def __init__(self, tok, eps=0.02, seed=SEED):
        self.mask_id = tok.mask_token_id; self.eps = eps
        self.gen = torch.Generator().manual_seed(seed)
    def __call__(self, ex):
        ids = torch.tensor([e["input_ids"] for e in ex], dtype=torch.long)
        B, L = ids.shape
        t = torch.rand(B, generator=self.gen) * (0.30 - self.eps) + self.eps
        mask = torch.rand(B, L, generator=self.gen) < t.unsqueeze(1)
        no = ~mask.any(1)
        if no.any():
            j = torch.randint(0, L, (int(no.sum()),), generator=self.gen); mask[no, j] = True
        inp = ids.clone(); inp[mask] = self.mask_id
        lab = ids.clone(); lab[~mask] = -100
        return {"input_ids": inp, "attention_mask": torch.ones(B, L, dtype=torch.long),
                "labels": lab, "t": t}

# 5) 작은 모델 (256/4L)
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
        per = F.cross_entropy(out.logits.view(-1, V), labels.view(-1),
                              ignore_index=-100, reduction="none").view(B, L)
        loss = ((per.sum(1)/L) / t.to(per.dtype)).mean()
        return (loss, out) if return_outputs else loss

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
args = TrainingArguments(output_dir="./35_ko_lowmask/out_local", max_steps=MAX_STEPS,
    per_device_train_batch_size=64, learning_rate=4e-4, weight_decay=0.01,
    warmup_steps=1000, lr_scheduler_type="cosine", max_grad_norm=1.0, fp16=False,
    logging_steps=250, save_strategy="no", report_to="none", label_names=["labels"],
    remove_unused_columns=False, seed=SEED, use_cpu=False)
trainer = DiffTrainer(model=model, args=args, train_dataset=lm_train,
                      data_collator=Coll(tokenizer))
t0 = time.time(); r = trainer.train()
print(f"\n=== elapsed {(time.time()-t0)/60:.2f}min | step {r.global_step} | "
      f"train_loss {r.training_loss:.4f} | baseline ln(V) {math.log(tokenizer.vocab_size):.4f} ===", flush=True)

# 6) 핵심 지표: 고정-t(0.15) top-1 acc (붕괴면 ~0.07, 탈출이면 상승)
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
print(f"[진단] 고정-t(0.15) top-1 acc = {fixed_t_acc():.3f}   (붕괴 0.07 / 탈출 0.2+)", flush=True)

# 7) infill 데모 — 한국어 문장 일부를 [MASK]로 가리고 복원
demo = "옛날 옛날에 작은 토끼가 숲으로 갔어요"
ids = tokenizer(demo, add_special_tokens=False)["input_ids"]
import random; random.seed(0)
mids = ids[:]; pos = sorted(random.sample(range(len(ids)), max(1, len(ids)//4)))
for p in pos: mids[p] = tokenizer.mask_token_id
with torch.no_grad():
    pr = model(torch.tensor([mids]).to(device)).logits[0].argmax(-1).cpu().tolist()
filled = [pr[i] if mids[i] == tokenizer.mask_token_id else ids[i] for i in range(len(ids))]
print("[infill] 원문:", demo, flush=True)
print("[infill] 가림:", tokenizer.decode([tokenizer.mask_token_id if m == tokenizer.mask_token_id else x for x, m in zip(ids, mids)]), flush=True)
print("[infill] 복원:", tokenizer.decode(filled), flush=True)
