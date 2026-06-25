## `datasets.map` — 5,000건 일괄 토큰화

샘플 하나씩 `tokenizer(...)` 부르는 건 5,000번 호출이 필요하고 느립니다. `dataset.map(fn, batched=True)` 로 *배치 단위* 일괄 호출이 표준.

```python
def tokenize_fn(batch):
    # batch는 dict of lists: {"text": [..., ...], "label": [..., ...]}
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
```

**위 코드 읽기** — `batched=True` 로 부르면 `tokenize_fn` 이 받는 `batch` 는 샘플 하나가 아니라 *리스트의 dict* 입니다. 그래서 `batch["text"]` 가 문자열 리스트가 되고, 토크나이저가 이 리스트를 한 번에 처리해 1샘플씩 부르는 것보다 훨씬 빠릅니다.

```python
# batched=True: tokenize_fn을 batch_size개씩 묶어 호출 (기본 1,000)
tokenized = small.map(tokenize_fn, batched=True, batch_size=200)

print(tokenized)
print(f"\nFirst sample input_ids length: {len(tokenized[0]['input_ids'])}  (= 128, fixed)")
print(f"First sample attention_mask sum: {sum(tokenized[0]['attention_mask'])}  (real tokens)")
```

**위 코드 읽기** — `map` 은 원본 컬럼(`text`, `label`)은 그대로 두고 `input_ids`/`token_type_ids`/`attention_mask` 를 *덧붙인* 새 데이터셋을 돌려줍니다. 결과는 디스크에 자동 캐시되어 같은 함수·데이터로 다시 부르면 즉시 로드됩니다.

**▶ 실행 결과**

```text
Dataset({
    features: ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask'],
    num_rows: 5000
})

First sample input_ids length: 128  (= 128, fixed)
First sample attention_mask sum: 75  (real tokens)
```

### `dataset.filter` — 조건에 맞는 샘플만 선별

`filter` 는 조건을 만족하는 샘플만 남깁니다. 별점이 높은(label ≥ 3) 긍정 리뷰만, 그리고 단어 100개 이하의 짧은 리뷰만 각각 골라내 전체 대비 비율을 확인합니다.

```python
# 별점 4-5 (label 3-4) 만
positive = small.filter(lambda x: x["label"] >= 3)
print(f"Positive samples: {len(positive):,} / {len(small):,} = {len(positive)/len(small):.1%}")

# 짧은 텍스트만 (예: 100단어 이하)
short = small.filter(lambda x: len(x["text"].split()) <= 100)
print(f"Short samples:    {len(short):,} / {len(small):,} = {len(short)/len(small):.1%}")
```

**▶ 실행 결과**

```text
Positive samples: 1,996 / 5,000 = 39.9%
Short samples:    2,520 / 5,000 = 50.4%
```

### `with_format("torch")` — 텐서 형식으로

학습에 넣으려면 출력이 Python 리스트가 아니라 PyTorch 텐서여야 합니다. `with_format("torch", columns=[...])` 로 지정한 컬럼만 텐서로 내보내는 새 데이터셋을 만들고, 첫 샘플의 dtype·shape를 확인합니다.

```python
# 모델에 바로 먹일 수 있도록 PyTorch tensor로 변환
tokenized_torch = tokenized.with_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"],
)

sample = tokenized_torch[0]
print(f"input_ids:      {type(sample['input_ids']).__name__}, dtype={sample['input_ids'].dtype}, shape={sample['input_ids'].shape}")
print(f"attention_mask: {type(sample['attention_mask']).__name__}, shape={sample['attention_mask'].shape}")
print(f"label:          {sample['label']}  (0-4 = stars 1-5)")
```

**▶ 실행 결과**

```text
input_ids:      Tensor, dtype=torch.int64, shape=torch.Size([128])
attention_mask: Tensor, shape=torch.Size([128])
label:          4  (0-4 = stars 1-5)
```

## `DataLoader` 변환 — Ch 9 학습 입력 미리보기

PyTorch `DataLoader` 는 dataset을 받아 *배치 + shuffle* 을 자동 처리합니다. Ch 9의 `Trainer` 가 내부에서 이걸 만들어 쓰지만, 직접 만들 줄 알면 디버깅에 유리.

텐서 형식 데이터셋을 `DataLoader` 에 넣으면 배치 묶기와 셔플이 자동으로 처리됩니다. 첫 배치를 꺼내 각 키의 shape(`input_ids`/`attention_mask` 가 `[batch_size, max_length]`)와 라벨 값을 확인합니다 — Ch 9의 `Trainer` 가 내부에서 만들어 쓰는 입력의 모습입니다.

```python
from torch.utils.data import DataLoader

loader = DataLoader(tokenized_torch, batch_size=8, shuffle=True)

# 첫 배치 살펴보기
batch = next(iter(loader))
print(f"batch keys:           {list(batch.keys())}")
print(f"input_ids shape:      {batch['input_ids'].shape}      (= [batch_size, max_length])")
print(f"attention_mask shape: {batch['attention_mask'].shape}")
print(f"label shape:          {batch['label'].shape}")
print(f"label values:         {batch['label'].tolist()}")
```

**▶ 실행 결과**

```text
batch keys:           ['label', 'input_ids', 'attention_mask']
input_ids shape:      torch.Size([8, 128])      (= [batch_size, max_length])
attention_mask shape: torch.Size([8, 128])
label shape:          torch.Size([8])
label values:         [1, 3, 3, 3, 1, 1, 3, 3]
```

## `DataCollator` — 동적 padding을 배치 시점에

위 `DataLoader` 코드는 `padding="max_length"` 로 *모든* 샘플을 128로 미리 padding한 상태였습니다. 짧은 문장은 대부분 패딩 자리라 메모리 낭비가 큽니다.

**더 좋은 방법: 토큰화 시엔 padding 안 하고**, `DataLoader` 가 *배치를 만들 때마다 그 배치 안 longest까지만* padding (동적 padding). 이걸 담당하는 게 `DataCollator`.

`DataCollator` 는 `DataLoader` 의 `collate_fn` 자리에 들어가는 함수입니다. 매 배치마다 N개 샘플을 받아 batch-level 변환을 적용한 뒤 텐서로 묶어주는 역할을 하고, Hugging Face는 task별로 여러 종류를 제공합니다.

동적 padding을 쓰려면 토큰화 단계에서는 일부러 padding을 *빼고* truncation만 적용합니다. 그러면 샘플마다 길이가 제각각이 되어 그대로는 하나의 텐서로 묶을 수 없는데, 앞 10개의 길이를 출력해 이 점을 직접 확인합니다.

```python
from transformers import DataCollatorWithPadding

# 토큰화 시엔 padding을 *빼고* truncation만 (길이가 들쭉날쭉)
def tokenize_dynamic(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

tokenized_dyn = small.select(range(50)).map(tokenize_dynamic, batched=True)
tokenized_dyn = tokenized_dyn.remove_columns(["text"])  # 깔끔하게 input만

# 각 샘플의 input_ids 길이는 모두 다름 (padding 없으니까)
sample_lens = [len(tokenized_dyn[i]["input_ids"]) for i in range(10)]
print(f"First 10 sample token lengths: {sample_lens}")
print(f"  → all different — cannot batch as-is into a tensor")
```

**▶ 실행 결과**

```text
First 10 sample token lengths: [75, 128, 89, 28, 128, 128, 128, 128, 64, 128]
  → all different — cannot batch as-is into a tensor
```

길이가 제각각인 샘플을 `DataCollatorWithPadding` 에 맡깁니다. 이 collator를 `DataLoader` 의 `collate_fn` 자리에 넣으면 *배치를 만들 때마다 그 배치 안 longest까지만* padding하므로, 배치별 shape와 채움률(`fill`)이 어떻게 달라지는지 출력합니다.

```python
# DataCollatorWithPadding이 collate_fn 자리에서 매 배치 동적 padding
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
dyn_loader = DataLoader(
    tokenized_dyn, batch_size=8, shuffle=False,
    collate_fn=collator,
)

print(f"Per-batch shape (batch_size=8, varies):")
print(f"{'batch':>6}  {'shape':>16}  {'real_tok':>10}  {'total':>6}  {'fill':>6}")
for i, batch in enumerate(dyn_loader):
    real = batch["attention_mask"].sum().item()
    total = batch["attention_mask"].numel()
    print(f"{i:>6}  {str(tuple(batch['input_ids'].shape)):>16}  {real:>10}  {total:>6}  {real/total:>6.0%}")
    if i >= 4: break
```

**▶ 실행 결과**

```text
Per-batch shape (batch_size=8, varies):
 batch             shape    real_tok   total    fill
     0          (8, 128)         832    1024     81%
     1          (8, 128)         782    1024     76%
     2          (8, 128)         761    1024     74%
     3          (8, 128)         770    1024     75%
     4          (8, 128)         762    1024     74%
```

**정적 vs 동적 padding 비교**

| 방식 | 토큰화 시점 | 배치 shape | 실제 토큰 비율 | 메모리/속도 |
|---|---|---|---|---|
| **정적** (`padding="max_length"=128`) | 모든 샘플을 128로 미리 padding | 항상 (B, 128) | 보통 30-50% | 일정, 낭비 큼 |
| **동적** (`DataCollatorWithPadding`) | padding 없이 truncation만 | (B, longest in batch) — 매번 다름 | 보통 70-95% | 변동, 효율 |

위 출력에서 채움률(실제 토큰/전체)이 정적 방식보다 훨씬 높을 겁니다. 같은 학습이라도 동적 padding이 *2배 가까이 빠른 경우* 도 흔합니다.

### 다양한 `DataCollator` 종류

| Collator | 용도 | 자주 쓰이는 곳 |
|---|---|---|
| `DataCollatorWithPadding` | 분류·회귀 — `input_ids`/`attention_mask` 동적 padding | **Ch 9-13 모든 분류 학습 (기본)** |
| `DataCollatorForLanguageModeling` | MLM — 입력의 15%를 `[MASK]` 로 가려 라벨 생성 | BERT 사전학습 재현, MLM 헤드 학습 |
| `DataCollatorForSeq2Seq` | seq2seq — encoder/decoder input 둘 다 padding | T5, BART 같은 인코더-디코더 학습 |
| `DataCollatorForTokenClassification` | NER 같은 토큰 단위 라벨링 — labels도 padding | NER, POS tagging |
| `default_data_collator` | 단순 stacking — padding 없이 길이 같은 샘플들에 | 이미 padding 끝낸 데이터 |

### 향후 학습 코드 관점 — Ch 9-13에서 실제로 어떻게 쓰이나

이번 챕터에서 만든 *데이터 파이프라인 부품들*이 다음 챕터부터 학습 코드에서 어떤 자리에 들어가는지 미리 그려두면, Ch 9 이후 코드가 훨씬 익숙해 보입니다.

#### 패턴 A — `Trainer` (커리큘럼 기본, Ch 9-13 대부분)

```python
# Ch 9 회귀, Ch 10 binary, Ch 11 multi-class, Ch 12 multi-label
# 모두 같은 골격. 바뀌는 건 num_labels / problem_type / 데이터뿐.

# 토크나이저 + 데이터셋 (이번 Ch 8에서 한 작업)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tok(b): return tokenizer(b["text"], truncation=True, max_length=128)
train_ds = small.map(tok, batched=True).remove_columns(["text"])
# padding 없이 둠. DataCollator가 매 배치 알아서 처리

# 모델 (Ch 7의 from_pretrained 패턴)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1, problem_type="regression",
)

# Trainer 한 줄로 묶음
trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,        # ← 이게 핵심: Trainer가 이걸 보고
                                #   자동으로 DataCollatorWithPadding 만듦
)
trainer.train()
```

**`tokenizer=...` 한 줄이 collator를 자동 생성합니다.** 학습자가 명시적으로 `data_collator=DataCollatorWithPadding(...)` 를 적을 일이 거의 없는 이유.

#### 패턴 B — 직접 학습 루프 (커스텀이 필요할 때)

```python
# Ch 13 auxiliary loss 처럼 Trainer 자동 매핑이 안 맞을 때, 또는
# 디버깅·연구용 커스텀 학습 코드를 짜야 할 때.

train_ds = small.map(tok, batched=True).remove_columns(["text"])

# DataLoader + collator 직접 조립 — Ch 8에서 본 패턴
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collator)

for batch in loader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss          # 또는 직접 계산
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### 두 패턴의 매핑

| 이번 Ch 8에서 만든 것 | 패턴 A (Trainer) | 패턴 B (직접 루프) |
|---|---|---|
| `tokenizer = AutoTokenizer.from_pretrained(...)` | `Trainer(tokenizer=...)` | `DataCollatorWithPadding(tokenizer=...)` |
| `dataset.map(tok, batched=True)` | `Trainer(train_dataset=...)` | `DataLoader(dataset, ...)` |
| `DataCollatorWithPadding(...)` | (Trainer가 자동 생성) | `DataLoader(collate_fn=...)` |
| `with_format("torch")` | (Trainer가 처리) | `DataLoader` 가 텐서 변환 |

**요점**: 이번 챕터에서 익힌 부품들이 Ch 9-13에서 *그대로 입력으로* 들어갑니다. `Trainer` 는 그 부품들을 묶어 학습 루프를 자동화한 것뿐이며, 안에서 일어나는 일은 패턴 B와 같습니다. 커스텀 학습이 필요해지면 패턴 B로 분해해 다시 짤 수 있다는 점이 Ch 8을 손에 익혀두는 가장 큰 이유입니다.

## Collator 추가 실습

DataCollator의 다양한 모습을 직접 돌려봅니다 — 동적 padding의 효율을 *숫자로* 확인하고, BERT 사전학습에 쓰는 MLM collator를 시연하며, 마지막으로 *직접 작성한 collator* 까지 보면 Ch 13 보조 loss 같은 커스텀 학습이 어떻게 가능한지 감이 잡힙니다.

### 실험 1 — 정적 vs 동적 padding 효율을 숫자로

같은 50 샘플을 두 방식으로 처리해 *전체 토큰 수* 와 *실제 토큰 비율* 을 직접 비교합니다.

```python
# 정적 padding 데이터셋 (max_length=128 고정)
def tokenize_static(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_static = small.select(range(50)).map(tokenize_static, batched=True).remove_columns(["text"])
tokenized_static = tokenized_static.with_format("torch", columns=["input_ids", "attention_mask", "label"])

static_loader = DataLoader(tokenized_static, batch_size=8, shuffle=False)
# dyn_loader는 앞 실습에서 만들어둔 것 그대로 사용

def loader_stats(loader, name):
    real_total = grid_total = 0
    for batch in loader:
        real_total += batch["attention_mask"].sum().item()
        grid_total += batch["attention_mask"].numel()
    return {"method": name, "real_tokens": real_total, "total_tokens": grid_total, "fill_rate": real_total / grid_total}

stats_static = loader_stats(static_loader, "static (max_length=128)")
stats_dyn    = loader_stats(dyn_loader,    "dynamic (DataCollator)")

df_stats = pd.DataFrame([stats_static, stats_dyn])
df_stats["fill_rate"] = df_stats["fill_rate"].apply(lambda r: f"{r:.1%}")
print(df_stats.to_string(index=False))

ratio = stats_dyn["total_tokens"] / stats_static["total_tokens"]
print(f"\nDynamic vs static total tokens: {ratio:.1%} → ~{1-ratio:.0%} reduction")
print("(this much self-attention compute saved → faster training, less memory)")
```

**▶ 실행 결과**

```text
                 method  real_tokens  total_tokens fill_rate
static (max_length=128)         4873          6400     76.1%
 dynamic (DataCollator)         4873          6400     76.1%

Dynamic vs static total tokens: 100.0% → ~0% reduction
(this much self-attention compute saved → faster training, less memory)
```

**결과 해석**

여기서는 절감이 0%로 나옵니다 — 이 50개 샘플 중 긴 리뷰가 많아 거의 모든 배치의 longest가 max_length(128)에 닿았기 때문입니다. 동적 padding의 이득은 *짧은 문장이 많이 섞인* 배치에서 커지므로, 데이터 분포에 따라 절감 폭이 크게 달라진다는 점을 보여주는 사례입니다.

### 실험 2 — `DataCollatorForLanguageModeling` 으로 MLM masking 직접 보기

BERT 사전학습은 입력 토큰의 15%를 `[MASK]` 로 가리고 모델이 맞추도록 학습됩니다 (Masked Language Modeling). 그 masking 자체를 담당하는 게 `DataCollatorForLanguageModeling` — 이번 챕터의 학습엔 안 쓰지만, BERT의 기원을 이해하는 데 직접 보는 게 빠릅니다.

```python
from transformers import DataCollatorForLanguageModeling

mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,   # 입력의 15% 가림 (BERT 원논문)
)

# tokenized_dyn 의 앞 3개 샘플로 시연
small_batch = [tokenized_dyn[i] for i in range(3)]
mlm_out = mlm_collator(small_batch)

print(f"input_ids shape: {mlm_out['input_ids'].shape}")
print(f"labels shape:    {mlm_out['labels'].shape}  (-100 positions ignored by loss)")

# masked 위치 통계
mask_id = tokenizer.mask_token_id
n_mask = (mlm_out["input_ids"] == mask_id).sum().item()
n_total = mlm_out["input_ids"].numel()
print(f"\n[MASK] tokens: {n_mask} / {n_total} ({n_mask/n_total:.1%})")
print("  (of the 15% masked: 80% [MASK], 10% random token, 10% kept — so [MASK] rate ~12%)")
```

**▶ 실행 결과**

```text
input_ids shape: torch.Size([3, 128])
labels shape:    torch.Size([3, 128])  (-100 positions ignored by loss)

[MASK] tokens: 44 / 384 (11.5%)
  (of the 15% masked: 80% [MASK], 10% random token, 10% kept — so [MASK] rate ~12%)
```

**결과 해석**

`labels` shape이 `input_ids` 와 같지만 마스킹된 자리 외에는 모두 `-100` 으로 채워져 loss에서 무시됩니다. 실제 `[MASK]` 토큰 비율이 15%가 아니라 11.5%인 건, 가릴 15% 중 80%만 `[MASK]` 로 바꾸고 10%는 랜덤 토큰, 10%는 원본 유지하는 BERT의 규칙 때문입니다.

masking이 토큰 단위로 정확히 무슨 일을 하는지 첫 샘플 앞 25토큰을 표로 펼쳐 봅니다. 원래 토큰(`original`) / collator가 내보낸 토큰(`masked`) / loss가 맞춰야 할 정답(`label`)을 나란히 두고, 바뀐 자리에 `*` 를 표시합니다.

```python
# 첫 샘플 — 원래 vs masked vs label 비교
i = 0
orig_ids = small_batch[i]["input_ids"][:25]
mask_ids = mlm_out["input_ids"][i][:25].tolist()
label_ids = mlm_out["labels"][i][:25].tolist()

rows = []
for orig, masked, lbl in zip(orig_ids, mask_ids, label_ids):
    orig_tok = tokenizer.decode([orig])
    masked_tok = tokenizer.decode([masked])
    if lbl == -100:
        label_str = "(ignored)"
    else:
        label_str = tokenizer.decode([lbl])
    changed = "*" if orig != masked else ""
    rows.append({"original": orig_tok, "masked": masked_tok, "label": label_str, "changed": changed})

print(pd.DataFrame(rows).to_string(index=False))
print("\nRows marked * are positions where the collator masked or replaced the token — the model learns to predict the original token there.")
```

**▶ 실행 결과**

```text
  original     masked     label changed
     [CLS]      [CLS] (ignored)        
         i          i (ignored)        
     stalk      stalk (ignored)        
      this       this (ignored)        
     truck barrington     truck       *
         .     [MASK]         .       *
         i     [MASK]         i       *
         '          ' (ignored)        
        ve         ve (ignored)        
      been       been (ignored)        
        to         to (ignored)        
industrial industrial (ignored)        
     parks      parks (ignored)        
     where      where (ignored)        
         i          i (ignored)        
   pretend    pretend (ignored)        
        to         to (ignored)        
        be         be (ignored)        
         a          a         a        
      tech       tech (ignored)        
    worker     worker (ignored)        
  standing     [MASK]  standing       *
        in         in (ignored)        
      line       line (ignored)        
         ,          , (ignored)        

Rows marked * are positions where the collator masked or replaced the token — the model learns to predict the original token there.
```

**결과 해석**

`*` 표시 행을 보면 세 가지 규칙이 한눈에 드러납니다 — `truck` 자리는 `barrington` 이라는 *랜덤 토큰* 으로 바뀌었고, `.` 와 `i` 자리는 `[MASK]` 로 가려졌습니다. 가려지지 않은 자리의 label은 모두 `(ignored)`(`-100`)라, 모델은 오직 `*` 자리의 원래 토큰만 예측하도록 학습됩니다.

### 실험 2b — GPT-style CLM 도 같은 collator로

같은 `DataCollatorForLanguageModeling` 에 **`mlm=False`** 만 주면 GPT 같은 *autoregressive 사전학습* 용 collator가 됩니다. 차이는 단순합니다.

| 비교 | MLM (BERT, `mlm=True`) | **CLM** (GPT, `mlm=False`) |
|---|---|---|
| 입력 | 일부 토큰을 `[MASK]` 로 가림 | 가리지 않음, 원본 그대로 |
| labels | masked 자리만 원래 토큰, 나머지 `-100` | `input_ids` 와 동일 (padding은 `-100`) |
| 학습 목적 | 양방향 문맥에서 가린 토큰 예측 | 왼쪽 문맥에서 *다음 토큰* 예측 |
| 모델 구조 | 양방향 attention | causal mask (왼쪽만 보기) |
| shift 처리 | 없음 | 모델 forward 내부에서 자동 shift-by-one |

GPT-2 토크나이저는 BERT와 다른 점이 두 가지 있어 약간의 셋업이 필요합니다.

1. `pad_token` 이 *없습니다*. 사전학습 시 패딩을 안 썼기 때문 — `eos_token` 을 pad로 재활용.
2. WordPiece 대신 BPE라 `Ġ` (공백) 접두사 표기가 등장 (Ch 7에서 이미 봄).

```python
from transformers import AutoTokenizer

# GPT-2 토크나이저 + pad_token 셋업
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# 작은 토큰화 데이터셋 (GPT-2 토크나이저 사용)
def gpt_tokenize(batch):
    return gpt_tokenizer(batch["text"], truncation=True, max_length=48)

gpt_tokenized = small.select(range(20)).map(gpt_tokenize, batched=True).remove_columns(["text"])

# CLM collator — mlm=False 가 핵심
clm_collator = DataCollatorForLanguageModeling(
    tokenizer=gpt_tokenizer,
    mlm=False,
)

clm_batch = clm_collator([gpt_tokenized[i] for i in range(3)])
print(f"input_ids shape:  {clm_batch['input_ids'].shape}")
print(f"labels shape:     {clm_batch['labels'].shape}")

# 핵심 관찰: input_ids 와 labels 가 (padding 제외하고) 동일
i = 0
real_len = (clm_batch["attention_mask"][i] == 1).sum().item()
input_first = clm_batch["input_ids"][i][:real_len].tolist()
label_first = clm_batch["labels"][i][:real_len].tolist()
print(f"\nFirst sample: input_ids == labels?  (excluding padding, all should match)")
print(f"  Matching positions: {sum(int(a == b) for a, b in zip(input_first, label_first))} / {real_len}")
```

**▶ 실행 결과**

```text
input_ids shape:  torch.Size([3, 48])
labels shape:     torch.Size([3, 48])

First sample: input_ids == labels?  (excluding padding, all should match)
  Matching positions: 48 / 48
```

**결과 해석**

48자리가 모두 일치한다는 건 CLM에서는 `labels` 가 `input_ids` 의 복사본이라는 뜻입니다 — MLM처럼 일부를 가리지 않습니다. 정답을 그대로 입력에 두고도 컨닝이 안 되는 이유는, 모델이 forward 내부에서 한 칸씩 밀어(shift-by-one) *다음* 토큰을 맞추고 causal mask로 오른쪽을 못 보기 때문입니다.

```python
# 첫 샘플 — input vs label (패딩 자리 -100 확인)
i = 0
ids = clm_batch["input_ids"][i][:20].tolist()
lbls = clm_batch["labels"][i][:20].tolist()

rows = []
for tok_id, lbl in zip(ids, lbls):
    tok = gpt_tokenizer.decode([tok_id])
    lbl_str = "(ignored)" if lbl == -100 else gpt_tokenizer.decode([lbl])
    rows.append({"position_token": tok, "label": lbl_str})

print(pd.DataFrame(rows).to_string(index=False))
print("\nObservation: at non-padding positions, input_ids and labels match.")
print("    The model shifts by one inside forward() and predicts the *next token*.")
print("    e.g. 'Hello world' → at 'Hello' position learn to predict 'world'.")
```

**▶ 실행 결과**

```text
position_token       label
             I           I
         stalk       stalk
          this        this
         truck       truck
             .           .
                          
             I           I
           've         've
          been        been
            to          to
    industrial  industrial
         parks       parks
         where       where
             I           I
       pretend     pretend
            to          to
            be          be
             a           a
          tech        tech
        worker      worker

Observation: at non-padding positions, input_ids and labels match.
    The model shifts by one inside forward() and predicts the *next token*.
    e.g. 'Hello world' → at 'Hello' position learn to predict 'world'.
```

### 실험 3 — 커스텀 `collate_fn` 직접 작성

위에서 본 `DataCollatorWithPadding` 와 `DataCollatorForLanguageModeling` 은 `transformers` 가 task별로 미리 만들어준 도구입니다. 그러나 *우리 task* 가 표준 형식에서 벗어나면 (예: Ch 13 보조 loss처럼 라벨이 두 종류) 직접 함수를 작성해야 합니다.

`collate_fn` 의 시그니처는 단순합니다 — 입력은 *샘플 dict의 리스트*, 출력은 *batch dict* (각 키에 stacked 텐서).

위에서 본 `DataCollatorWithPadding` 와 `DataCollatorForLanguageModeling` 은 `transformers` 가 task별로 미리 만들어준 도구입니다. 그러나 *우리 task* 가 표준 형식에서 벗어나면 (예: Ch 14 보조 loss처럼 라벨이 두 종류) 직접 함수를 작성해야 합니다.

```python
def custom_collate(batch_list):
    # 샘플 dict 리스트 → batch dict.
    # Ch 13 보조 loss를 미리 흉내내 라벨을 두 종류로 만들어 둠:
    #   - main_label: 0-4 정수 (분류 라벨)
    #   - aux_score:  0.0-1.0 float (정규화한 별점)

    # input_ids/attention_mask 는 DataCollatorWithPadding 에 위임 (편함)
    pad_input = collator(
        [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch_list]
    )

    # 라벨을 두 형태로 변환
    raw_labels = torch.tensor([item["label"] for item in batch_list], dtype=torch.long)
    aux_scores = raw_labels.float() / 4.0   # 0-4 → 0-1 (정규화)

    pad_input["main_label"] = raw_labels
    pad_input["aux_score"]  = aux_scores
    return pad_input


custom_loader = DataLoader(
    tokenized_dyn,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate,
)

batch = next(iter(custom_loader))
print(f"input_ids shape:        {batch['input_ids'].shape}")
print(f"attention_mask shape:   {batch['attention_mask'].shape}")
print(f"main_label (int):       {batch['main_label'].tolist()}")
print(f"aux_score (float):      {[round(x, 3) for x in batch['aux_score'].tolist()]}")
```

**▶ 실행 결과**

```text
input_ids shape:        torch.Size([4, 128])
attention_mask shape:   torch.Size([4, 128])
main_label (int):       [4, 2, 4, 0]
aux_score (float):      [1.0, 0.5, 1.0, 0.0]
```

**관찰**: `collate_fn` 한 함수가 *batch 단위 변환의 집결지* 입니다. 라벨 형식 변환, 추가 메타정보 부착, 다중 라벨 dict 등 무엇이든 여기서 처리할 수 있습니다.

Ch 13 보조 loss는 이 패턴을 그대로 사용 — 메인 라벨(multi-hot)과 보조 라벨(별점 float)을 한 dict 안에 같이 담아서 모델이 두 loss를 합쳐 계산하도록 합니다. *학습 코드의 어디에도 collate_fn 정의가 없는데 학습이 잘 되네?* 라고 느낀다면, 그건 `Trainer` 가 `tokenizer` 만 보고 `DataCollatorWithPadding` 을 자동 생성한 결과일 가능성이 큽니다 — 직접 짜야 하는 순간엔 위 패턴이 출발점입니다.
