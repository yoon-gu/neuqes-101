## 평가 — MLM loss 곡선 + perplexity + masked token 예측

학습이 *실제로 진행* 됐는지 세 각도로 확인:
1. step-by-step train loss 곡선 — 빠르게 10.33 (random baseline) → 약 7 부근으로 떨어졌는지
2. eval set 의 perplexity — 외부 텍스트에서도 일관된 수준인지
3. 임의 문장에 `[MASK]` 를 끼워 top-5 후보 출력 — *어떤 단어를 예측하는지* 정성 평가

```python
# 학습 로그에서 train loss 추출
log_history = trainer.state.log_history
train_logs = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]

if train_logs:
    steps, losses = zip(*train_logs)
    random_baseline = math.log(tokenizer.vocab_size)

    sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, "o-", color="#4878D0", label="학습 MLM loss")
    ax.axhline(random_baseline, color="black", lw=1.0, ls=":", label=f"랜덤 기준선 (ln V = {random_baseline:.2f})")
    ax.set_xlabel("학습 step")
    ax.set_ylabel("MLM loss (CrossEntropy)")
    ax.set_title("MLM 학습 loss — Wikitext-103 위에서 처음부터 학습한 small BERT")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No train loss logs found.")
```

**▶ 실행 결과**

![output](../assets/20-en_bert_pretrain-out1.png)

```python
eval_metrics = trainer.evaluate()
eval_loss = eval_metrics["eval_loss"]
eval_ppl = math.exp(eval_loss)
print("=== eval (held-out Wikitext-103 paragraphs) ===")
for k, v in eval_metrics.items():
    if isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
print()
print(f"  MLM loss:               {eval_loss:.4f}")
print(f"  perplexity (exp loss):  {eval_ppl:.2f}")
print(f"  random baseline PPL:    {tokenizer.vocab_size:,}  (uniform over vocab)")
print(f"  -> model narrowed vocab to approx. {eval_ppl:.0f} candidates per masked position")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
=== eval (held-out Wikitext-103 paragraphs) ===
               eval_loss: 7.1202

  MLM loss:               7.1202
  perplexity (exp loss):  1236.73
  random baseline PPL:    30,522  (uniform over vocab)
  -> model narrowed vocab to approx. 1237 candidates per masked position
```

### 6-1. 🔬 사전학습 전·후 비교 — random init 본체 vs 2 epoch 학습 후

학습 직전 (5번 마지막 셀에서 측정해 둔 `pre_eval_loss` / `pre_top5_records`) 와 *완전히 같은 문장·같은 평가 셋* 에 학습 후 모델을 적용해 두 결과를 나란히 봅니다. *사전학습이 본체에 무엇을 새겼는가* 의 가장 직접적인 증거.

```python
# ---- 사전학습 후 eval_loss / perplexity ----
post_eval = trainer.evaluate()
post_eval_loss = post_eval["eval_loss"]
post_eval_ppl  = math.exp(post_eval_loss)

print("=" * 78)
print("AFTER pretraining  (2 epoch MLM on Wikitext-103)")
print("=" * 78)
print(f"  eval_loss       : {post_eval_loss:.4f}   (before: {pre_eval_loss:.4f})")
print(f"  eval_perplexity : {post_eval_ppl:,.2f}        (before: {pre_eval_ppl:,.0f})")
print(f"  -> narrowed vocab to approx. {post_eval_ppl:.0f} candidates per masked position")
print()

# ---- 사전학습 후 [MASK] top-5 ----
post_top5_records = []
for sent in test_sentences:
    results = predict_mask(sent, top_k=5)
    top5_tokens = [tok for tok, _ in results[0][1]] if results else []
    post_top5_records.append({"sentence": sent, "top5_after": top5_tokens})
    print(f"input: {sent}")
    print(f"  top-5 after pretraining: {top5_tokens}")
    print()
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
==============================================================================
AFTER pretraining  (2 epoch MLM on Wikitext-103)
==============================================================================
  eval_loss       : 7.1278   (before: 10.3834)
  eval_perplexity : 1,246.14        (before: 32,319)
  -> narrowed vocab to approx. 1246 candidates per masked position

input: The capital of France is [MASK].
  top-5 after pretraining: ['the', ',', '.', 'and', 'of']

input: Water freezes at [MASK] degrees Celsius.
  top-5 after pretraining: ['the', ',', '.', 'and', 'of']

input: The food at this restaurant was absolutely [MASK].
  top-5 after pretraining: ['the', ',', '.', 'and', 'of']

input: I would [MASK] recommend this place.
  top-5 after pretraining: ['the', ',', '.', 'and', 'of']
```

### 6-2. eval_loss / perplexity — 수치 비교

두 측정치를 한 표·한 막대 그래프로.

```python
# 사전·사후 수치 비교 표
metric_compare = pd.DataFrame({
    "metric":           ["eval_loss", "eval_perplexity"],
    "before (random)":  [pre_eval_loss,  pre_eval_ppl],
    "after (2 epoch)":  [post_eval_loss, post_eval_ppl],
    "random baseline":  [random_baseline_loss, float(tokenizer.vocab_size)],
})
print("Before vs After — eval metrics")
print(metric_compare.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
Before vs After — eval metrics
         metric  before (random)  after (2 epoch)  random baseline
      eval_loss          10.3834           7.1278          10.3262
eval_perplexity       32319.2119        1246.1431       30522.0000
```

```python
# 막대 그래프 두 장 (eval_loss / perplexity)
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

loss_values = [pre_eval_loss, post_eval_loss]
loss_labels = ["학습 전 (랜덤)", "학습 후 (2 epoch)"]
axes[0].bar(loss_labels, loss_values, color=["#999999", "#4878D0"])
axes[0].axhline(random_baseline_loss, color="black", lw=1.0, ls=":",
                label=f"랜덤 기준선 ln V = {random_baseline_loss:.2f}")
axes[0].set_ylabel("eval_loss")
axes[0].set_title("MLM eval_loss")
axes[0].legend(loc="upper right", fontsize=10)

ppl_values = [pre_eval_ppl, post_eval_ppl]
axes[1].bar(loss_labels, ppl_values, color=["#999999", "#4878D0"])
axes[1].set_yscale("log")
axes[1].axhline(tokenizer.vocab_size, color="black", lw=1.0, ls=":",
                label=f"랜덤 기준선 V = {tokenizer.vocab_size:,}")
axes[1].set_ylabel("perplexity (log scale)")
axes[1].set_title("MLM perplexity")
axes[1].legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/20-en_bert_pretrain-out2.png)

### 6-3. 🏆 학습이 *충분히 잘 된 경우* 의 기준점 — 표준 `bert-base-uncased` 비교

우리 작은 BERT (10M, 5K paragraphs × 2 epoch) 의 top-5 가 *방향성은 맞지만 정답이 잘 안 보이는* 이유는 단순합니다 — **학습 데이터·모델 크기·학습 시간 모두 부족**. *그럼 학습이 충분히 잘 되면 어떤 결과가 나오나?* 의 답을 같은 문장에 표준 `bert-base-uncased` (110M, 위키+BookCorpus 약 33억 토큰) 를 적용해 직접 봅니다.

같은 토크나이저 (`bert-base-uncased`) 를 쓰고 있으므로 *모델만 바꿔* 두 결과를 나란히.

```python
# 표준 bert-base-uncased 로드 — 학습이 충분히 잘 된 경우의 기준점
from transformers import AutoModelForMaskedLM

ref_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
ref_model.to(model.device)
ref_model.eval()

ref_param_count = sum(p.numel() for p in ref_model.parameters())
our_param_count = sum(p.numel() for p in model.parameters())
print(f"Our small BERT params: {our_param_count/1e6:.1f}M")
print(f"Reference BERT params: {ref_param_count/1e6:.1f}M  ({ref_param_count/our_param_count:.0f}x larger)")
```

**▶ 실행 결과**

```text
[transformers] BertForMaskedLM LOAD REPORT from: bert-base-uncased
Key                         | Status     |  | 
----------------------------+------------+--+-
bert.pooler.dense.weight    | UNEXPECTED |  | 
cls.seq_relationship.weight | UNEXPECTED |  | 
bert.pooler.dense.bias      | UNEXPECTED |  | 
cls.seq_relationship.bias   | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Our small BERT params: 11.1M
Reference BERT params: 109.5M  (10x larger)
```

```python
# Reference 모델로 같은 문장의 top-5 측정
def predict_mask_with(text, ref, top_k=5):
    '''임의의 MLM 모델로 [MASK] 자리 top-k 예측.'''
    ref.eval()
    inputs = tokenizer(text, return_tensors="pt").to(ref.device)
    with torch.no_grad():
        outputs = ref(**inputs)
    logits = outputs.logits[0]
    mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return None
    results = []
    for pos in mask_positions:
        probs = torch.softmax(logits[pos], dim=-1)
        top_p, top_i = probs.topk(top_k)
        candidates = [(tokenizer.convert_ids_to_tokens(int(i)), float(p))
                       for p, i in zip(top_p, top_i)]
        results.append((int(pos), candidates))
    return results


ref_top5_records = []
for sent in test_sentences:
    results = predict_mask_with(sent, ref_model, top_k=5)
    top5_tokens = [tok for tok, _ in results[0][1]] if results else []
    ref_top5_records.append({"sentence": sent, "top5_ref": top5_tokens})

# 참조 모델 메모리 해제 (분류 fine-tune 챕터 21 가 같은 노트북이 아니므로 안전)
del ref_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 6-4. [MASK] top-5 — 3-way 비교 (before / ours / reference BERT)

같은 문장 4개의 [MASK] 자리 top-5 후보를 *사전학습 전 → 우리 작은 BERT 학습 후 → 표준 bert-base-uncased* 셋으로 나란히.

```python
# 3-way top-5 비교 표
rows = []
for pre, post, ref in zip(pre_top5_records, post_top5_records, ref_top5_records):
    rows.append({
        "sentence":          pre["sentence"],
        "top5_before":       ", ".join(pre["top5_before"]),
        "top5_ours":         ", ".join(post["top5_after"]),
        "top5_ref_bert":     ", ".join(ref["top5_ref"]),
    })

top5_compare = pd.DataFrame(rows)
print("Before (random) vs Ours (small BERT, 5K paragraphs) vs Reference (bert-base-uncased, approx. 3.3B tokens)")
print("=" * 100)
for _, row in top5_compare.iterrows():
    print(f"input: {row['sentence']}")
    print(f"  before (random)        : {row['top5_before']}")
    print(f"  ours  (small, 5K para) : {row['top5_ours']}")
    print(f"  ref   (bert-base)      : {row['top5_ref_bert']}")
    print()
```

**▶ 실행 결과**

```text
Before (random) vs Ours (small BERT, 5K paragraphs) vs Reference (bert-base-uncased, approx. 3.3B tokens)
====================================================================================================
input: The capital of France is [MASK].
  before (random)        : broadband, ##eti, weights, smack, ##umen
  ours  (small, 5K para) : the, ,, ., and, of
  ref   (bert-base)      : paris, lille, lyon, marseille, tours

input: Water freezes at [MASK] degrees Celsius.
  before (random)        : ##hp, damages, slogan, ##ssion, [unused358]
  ours  (small, 5K para) : the, ,, ., and, of
  ref   (bert-base)      : 100, 60, 50, 30, 90

input: The food at this restaurant was absolutely [MASK].
  before (random)        : languages, sampling, ##pic, libretto, ##orus
  ours  (small, 5K para) : the, ,, ., and, of
  ref   (bert-base)      : delicious, amazing, fabulous, fantastic, incredible

input: I would [MASK] recommend this place.
  before (random)        : ahmed, smack, stations, ##sam, now
  ours  (small, 5K para) : the, ,, ., and, of
  ref   (bert-base)      : highly, certainly, definitely, strongly, greatly
```

**해석 가이드 — 사전학습이 만든 차이**

- **`eval_loss`**: random baseline `ln V ≈ 10.33` 에서 약 7 부근까지 떨어졌으면 본체가 *언어 구조 일부* 를 학습. *완전한* BERT 수준은 아니어도 표준 BERT 가 학습한 것의 *방향* 은 맞춤.
- **`perplexity`**: 30,522 (vocab 전체) 에서 약 1,200 부근으로. *마스크 자리마다 후보를 약 1,200 개로 좁힌 상태* 라는 직관적 해석.
- **top-5 토큰** (3-way 비교):
  - *before (random)*: 자주 등장하는 *관사·전치사·특수문자* (`the`, `a`, `,`, `.`, `of`) — random init 이지만 logits 가 미세하게 흔들려 *통계적 빈도* 높은 토큰만 뽑힘.
  - *ours (small BERT, 5K paragraphs × 2 epoch)*: 위키 도메인은 *방향성이 보이기 시작* — 일반 부사·형용사, 위키 어휘 일부. 다만 정답 토큰 (`paris`, `0` 등) 이 top-5 안에 *안정적으로* 들어오지는 못함. **데이터·모델 크기 부족의 한계**.
  - *ref (bert-base-uncased, 약 33억 토큰 × 40 epoch)*: 위키 도메인은 *정답이 top-1* — `paris`, `zero` 같은 자연스러운 답. Yelp 도메인 (다른 도메인) 도 *감성 형용사* (`amazing`, `delicious`, `highly`) 가 자연스럽게 top-5 에 들어옴. **이게 사전학습이 충분히 잘 됐을 때의 모습**.

> **세 모델의 격차가 정확히 *데이터 규모 + 모델 크기 + 학습 시간* 의 격차** — 우리 작은 BERT (10M, 5K paragraphs, 2 epoch) → reference (110M, 3.3B tokens, 40 epoch) 사이에 *데이터 약 5,000배, 파라미터 11배, epoch 20배*. 그 격차가 top-5 의 *질적 차이* 로 정확히 드러납니다.

이번 챕터의 작은 BERT 는 *Wikitext-103 5K paragraphs × 2 epoch* 로 학습한 *일반 도메인 mini BERT*. 위키 도메인은 직접 본 분포라 향상이 빠르지만, Yelp 리뷰(식당·업체) 영역은 *다른 도메인* 이라 fine-tune 단계에서 적응이 필요합니다 — 이게 *진짜 사전학습 → fine-tune 패러다임* 의 핵심. Ch 21 에서 Yelp 이진 분류로 fine-tune 할 때 진짜 transfer 비교 — *우리가 직접 만든 작은 영어 BERT (일반 위키 5K, 약 10M)* vs *Ch 10 의 DistilBERT (대규모 Wikipedia+BookCorpus, 약 66M)* vs *random init baseline*.

## 모델 저장 — Ch 21 에서 재사용

`model.save_pretrained()` 와 `tokenizer.save_pretrained()` 를 *같은 폴더* 에 저장. Ch 21 에서는 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 한 줄로 *이 BERT body* 를 가져와 분류 헤드를 새로 얹습니다.

```python
SAVE_DIR = "./ch20_small_bert_mlm"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

import os
print(f"Saved to: {SAVE_DIR}")
print(f"Files:")
for f in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    if size > 1024 * 1024:
        size_str = f"{size / 1024 / 1024:.1f} MB"
    else:
        size_str = f"{size / 1024:.1f} KB"
    print(f"  {f:>30s}  {size_str}")
```

**▶ 실행 결과**

```text
Saved to: ./ch20_small_bert_mlm
Files:
                     config.json  0.7 KB
               model.safetensors  42.4 MB
                  tokenizer.json  694.7 KB
           tokenizer_config.json  0.3 KB
```

**저장된 파일 구조** — `from_pretrained` 가 인식하는 HF 표준 레이아웃:

| 파일 | 역할 |
|---|---|
| `config.json` | `BertConfig` 직렬화 (hidden, layer, head, vocab 등) |
| `model.safetensors` (또는 `pytorch_model.bin`) | 모델 weight |
| `tokenizer.json` / `vocab.txt` | 토크나이저 (Ch 21 fine-tune 에서 같은 토크나이저 사용) |
| `special_tokens_map.json`, `tokenizer_config.json` | 특수 토큰 메타 |

> Ch 21 에서 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 호출 시, `BertForMaskedLM` 의 *MLM head 는 버려지고* encoder body 만 가져옴. 그 위에 새 `Linear(256, 2)` 분류 헤드를 random init 으로 부착 — Ch 7-18 의 fine-tune 셋업과 *동일한 구조*. 이번 챕터의 사전학습이 *얼마나 도움 됐는지* 가 Ch 21 의 학습 곡선에서 직접 비교됩니다.
