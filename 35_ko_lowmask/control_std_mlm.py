"""대조 실험 — 표준 MLM(고정 15% 마스킹, plain CE, Ch 22 방식)을 같은 한국어 데이터에.
diffusion 목적함수(1/t·고마스킹)가 붕괴 원인인지 vs 한국어 자체가 안 되는지 가른다.
acc 높으면 → 표준 MLM은 됨 → diffusion 설정이 문제. acc 낮으면 → 데이터/근본."""
import math, time, sys, torch
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import (PreTrainedTokenizerFast, BertConfig, BertForMaskedLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)

SEED = 42; torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, flush=True)

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
raw_train = Dataset.from_dict({"text": rebuild("train", N_TRAIN, MAXL)})
raw_val   = Dataset.from_dict({"text": rebuild("validation", N_VAL, 50_000)})
print("stories", len(raw_train), len(raw_val), flush=True)

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
print("vocab", tokenizer.vocab_size, flush=True)

BLOCK = 128
tt = raw_train.map(lambda b: tokenizer(b["text"], add_special_tokens=False), batched=True, remove_columns=raw_train.column_names)
tv = raw_val.map(lambda b: tokenizer(b["text"], add_special_tokens=False), batched=True, remove_columns=raw_val.column_names)
def group(b):
    cat = sum(b["input_ids"], []); n = (len(cat)//BLOCK)*BLOCK
    return {"input_ids": [cat[i:i+BLOCK] for i in range(0, n, BLOCK)]}
lm_train = tt.map(group, batched=True, remove_columns=tt.column_names)
lm_val   = tv.map(group, batched=True, remove_columns=tv.column_names)
print("chunks", len(lm_train), len(lm_val), flush=True)

cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=256, num_hidden_layers=4,
                 num_attention_heads=4, intermediate_size=1024,
                 max_position_embeddings=BLOCK, pad_token_id=tokenizer.pad_token_id)
model = BertForMaskedLM(cfg).to(device)
print("params(M)", round(model.num_parameters()/1e6, 2), flush=True)

# ★ 표준 MLM collator: 고정 15%, 80/10/10, plain CE (Trainer 기본 loss)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
args = TrainingArguments(output_dir="./35_ko_lowmask/out_ctrl", max_steps=MAX_STEPS,
    per_device_train_batch_size=64, learning_rate=5e-4, weight_decay=0.01,
    warmup_steps=500, lr_scheduler_type="cosine", max_grad_norm=1.0, fp16=False,
    logging_steps=250, save_strategy="no", report_to="none", remove_unused_columns=False, seed=SEED)
trainer = Trainer(model=model, args=args, train_dataset=lm_train, data_collator=collator)
t0 = time.time(); r = trainer.train()
print(f"\n=== STD-MLM elapsed {(time.time()-t0)/60:.2f}min | step {r.global_step} | "
      f"train_loss {r.training_loss:.4f} | baseline ln(V) {math.log(tokenizer.vocab_size):.4f} ===", flush=True)

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
print(f"[대조] 표준-MLM 고정-t(0.15) top-1 acc = {fixed_t_acc():.3f}   (diffusion은 0.084)", flush=True)
