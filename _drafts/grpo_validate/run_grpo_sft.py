"""Ch 31 GRPO 검증 — SFT 워밍스타트 → 비제로 시작점 → GRPO 개선 수치 찾기.
사용: python run_grpo_sft.py [max_operand] [sft_epochs]"""
import re, random, sys, time, torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, PreTrainedTokenizerFast, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)

SEED = 42; random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL = "skt/kogpt2-base-v2"
MAXOP = int(sys.argv[1]) if len(sys.argv) > 1 else 9
SFT_EPOCHS = float(sys.argv[2]) if len(sys.argv) > 2 else 3
print(f"device={device} max_operand={MAXOP} sft_epochs={SFT_EPOCHS}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)

RESP = "### 응답:\n"
def build_prompt(q): return f"### 명령어:\n{q}\n\n{RESP}"
def make_arith(n, seed):
    rng = random.Random(seed); rows = []
    for _ in range(n):
        a = rng.randint(1, MAXOP); b = rng.randint(1, MAXOP); op = rng.choice(["+", "-"])
        ans = a + b if op == "+" else a - b
        rows.append({"prompt": build_prompt(f"{a} {op} {b} = ?"), "answer": ans})
    return Dataset.from_list(rows)

train_ds = make_arith(2000, SEED)
eval_ds  = make_arith(128, SEED + 1)

def extract_last_int(t):
    m = re.findall(r"-?\d+", t); return m[-1] if m else None

@torch.no_grad()
def eval_acc(model, ds, n=128, n_sample=2, max_new=16):
    model.eval(); cor = tot = 0
    for ex in ds.select(range(min(n, len(ds)))):
        enc = tokenizer(ex["prompt"], return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.8,
                             top_p=0.95, num_return_sequences=n_sample, pad_token_id=tokenizer.pad_token_id)
        for g in gen:
            txt = tokenizer.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_last_int(txt)
            cor += int(pred is not None and pred == str(ex["answer"])); tot += 1
    return cor / tot

print(f"[0] BASE accuracy (SFT 전): {eval_acc(model, eval_ds):.3f}", flush=True)

# ===== SFT 워밍스타트 =====
def to_text(ex): return {"text": ex["prompt"] + str(ex["answer"]) + tokenizer.eos_token}
sft_ds = train_ds.map(to_text)
def tok_fn(b):
    o = tokenizer(b["text"], truncation=True, max_length=64, padding="max_length")
    return o
sft_tok = sft_ds.map(tok_fn, batched=True, remove_columns=sft_ds.column_names)
sft_args = TrainingArguments(output_dir="./_drafts/grpo_validate/sft_out",
    num_train_epochs=SFT_EPOCHS, per_device_train_batch_size=16, learning_rate=5e-4,
    warmup_ratio=0.1, lr_scheduler_type="cosine", fp16=torch.cuda.is_available(),
    logging_steps=100, save_strategy="no", report_to="none")
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
sft_trainer = Trainer(model=model, args=sft_args, train_dataset=sft_tok, data_collator=collator)
t0 = time.time(); sft_trainer.train()
print(f"[SFT] {(time.time()-t0)/60:.1f}min", flush=True)
acc_sft = eval_acc(model, eval_ds)
print(f"[1] SFT 후 accuracy (= GRPO 시작점): {acc_sft:.3f}", flush=True)

# ===== GRPO =====
from trl import GRPOTrainer, GRPOConfig
def reward_correct(completions, answer, **kw):
    return [1.0 if (extract_last_int(c) is not None and extract_last_int(c) == str(g)) else 0.0
            for c, g in zip(completions, answer)]
GROUP = 8
grpo_args = GRPOConfig(output_dir="./_drafts/grpo_validate/grpo_out",
    num_train_epochs=2, per_device_train_batch_size=GROUP, gradient_accumulation_steps=2,
    num_generations=GROUP, max_completion_length=16, temperature=1.0, learning_rate=1e-5,
    beta=0.0, warmup_ratio=0.1, lr_scheduler_type="cosine", max_grad_norm=1.0,
    fp16=torch.cuda.is_available(), logging_steps=20, save_strategy="no", report_to="none")
grpo_trainer = GRPOTrainer(model=model, args=grpo_args, train_dataset=train_ds,
                           reward_funcs=reward_correct)
t0 = time.time(); grpo_trainer.train()
print(f"[GRPO] {(time.time()-t0)/60:.1f}min", flush=True)
acc_grpo = eval_acc(model, eval_ds)
print(f"\n=== 결과 ===", flush=True)
print(f"BASE(SFT전) {eval_acc(model, eval_ds) if False else '—'}  SFT후(GRPO전) {acc_sft:.3f}  GRPO후 {acc_grpo:.3f}", flush=True)
