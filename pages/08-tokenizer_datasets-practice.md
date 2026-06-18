> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/08_tokenizer_datasets/08_tokenizer_datasets.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q transformers datasets
```

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
print("\nNo model weights loaded in this chapter; VRAM stays roughly flat.")
```

**▶ 실행 결과**

```text
PyTorch:        2.11.0+cu128
CUDA available: True
GPU:            Tesla T4

No model weights loaded in this chapter; VRAM stays roughly flat.
```

Phase 0에서 쓰던 Yelp 리뷰 데이터를 `datasets` 허브에서 그대로 내려받습니다. 반환되는 `DatasetDict` 의 train/test split 구성과 각 split의 행 수를 출력해 데이터 규모를 먼저 확인합니다.

```python
ds = load_dataset("Yelp/yelp_review_full")
print(ds)
```

**▶ 실행 결과**

```text
DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 650000
    })
    test: Dataset({
        features: ['label', 'text'],
        num_rows: 50000
    })
})
```

데이터의 실제 모양을 들여다봅니다. `features` 로 라벨이 5단계 별점(`ClassLabel`)이고 텍스트가 문자열임을 확인하고, 첫 샘플의 label과 text 앞부분을 찍어 어떤 리뷰인지 감을 잡습니다.

```python
# train split의 첫 샘플 + features 확인
print(f"train samples: {len(ds['train']):,}")
print(f"test samples:  {len(ds['test']):,}")
print(f"\nfeatures: {ds['train'].features}")
print(f"\nFirst sample:")
print(f"  label: {ds['train'][0]['label']}  (0-4 = stars 1-5)")
print(f"  text:  {ds['train'][0]['text'][:200]}...")
```

**▶ 실행 결과**

```text
train samples: 650,000
test samples:  50,000

features: {'label': ClassLabel(names=['1 star', '2 star', '3 stars', '4 stars', '5 stars']), 'text': Value('string')}

First sample:
  label: 4  (0-4 = stars 1-5)
  text:  dr. goldberg offers everything i look for in a general practitioner.  he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-no...
```

65만 건 전체는 T4 30분 제약에 너무 큽니다. `shuffle` 로 섞은 뒤 `select` 로 5,000건만 떼어내 Phase 0과 같은 작업 분량으로 맞춥니다. `seed=42` 를 고정해 매번 같은 subsample을 얻도록 합니다.

```python
# 5,000건만 subsample (Phase 0와 동일한 처리)
small = ds["train"].shuffle(seed=42).select(range(5000))
print(small)
print(f"\nfirst sample text: {small[0]['text'][:150]}...")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['label', 'text'],
    num_rows: 5000
})

first sample text: I stalk this truck.  I've been to industrial parks where I pretend to be a tech worker standing in line, strip mall parking lots, and of course the fa...
```

DistilBERT의 토크나이저를 불러옵니다. 실제 클래스 이름과 vocab 크기, 그리고 빈자리를 채우는 데 쓰는 `pad_token` 과 그 id를 확인합니다. pad 토큰은 뒤에서 padding을 다룰 때 핵심이 됩니다.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(f"Class:     {type(tokenizer).__name__}")
print(f"vocab:     {tokenizer.vocab_size:,}")
print(f"pad_token: {tokenizer.pad_token}  (id={tokenizer.pad_token_id})")
```

**▶ 실행 결과**

```text
Class:     BertTokenizer
vocab:     30,522
pad_token: [PAD]  (id=0)
```

리뷰 한 건을 토크나이저에 통과시켜 텍스트가 정수 id 시퀀스로 바뀌는 과정을 봅니다. 맨 앞 id가 `[CLS]`(101)로 시작하는 점, 그리고 `decode` 로 id를 다시 토큰 문자열로 되돌려 어떤 단위로 쪼개졌는지 확인합니다.

```python
sample = small[0]["text"]
print(f"Input (first 150 chars): {sample[:150]}...\n")

out = tokenizer(sample)
print(f"input_ids length: {len(out['input_ids'])}")
print(f"First 30 IDs:      {out['input_ids'][:30]}")
print(f"Decoded first 30:  {tokenizer.decode(out['input_ids'][:30])}")
```

**▶ 실행 결과**

```text
Input (first 150 chars): I stalk this truck.  I've been to industrial parks where I pretend to be a tech worker standing in line, strip mall parking lots, and of course the fa...

input_ids length: 75
First 30 IDs:      [101, 1045, 23899, 2023, 4744, 1012, 1045, 1005, 2310, 2042, 2000, 3919, 6328, 2073, 1045, 9811, 2000, 2022, 1037, 6627, 7309, 3061, 1999, 2240, 1010, 6167, 6670, 5581, 7167, 1010]
Decoded first 30:  [CLS] i stalk this truck. i ' ve been to industrial parks where i pretend to be a tech worker standing in line, strip mall parking lots,
```

길이가 크게 다른 두 문장을 한 번에 토큰화하며 padding의 효과를 봅니다. `padding=False` 면 문장마다 길이가 제각각이지만, `padding=True` 면 배치 안 가장 긴 길이에 맞춰 짧은 쪽을 채웁니다. 채운 자리는 `attention_mask` 에서 0으로 표시돼 모델이 무시할 부분을 알려줍니다.

```python
# 길이가 다른 두 문장을 묶기
short_text = "Great service!"
long_text = small[0]["text"]
texts = [short_text, long_text]

# padding=False (기본): 각 문장 길이 그대로
out_no_pad = tokenizer(texts, padding=False)
print(f"padding=False:")
for i, ids in enumerate(out_no_pad["input_ids"]):
    print(f"  sentence {i}: {len(ids)} tokens")

# padding=True: 가장 긴 길이까지만 padding
out_dyn = tokenizer(texts, padding=True, return_tensors="pt")
print(f"\npadding=True (return_tensors='pt'):")
print(f"  input_ids shape: {out_dyn['input_ids'].shape}")
print(f"  attention_mask sentence 0: {out_dyn['attention_mask'][0][:20]}")
print(f"  attention_mask sentence 1: {out_dyn['attention_mask'][1][:20]}")
```

**▶ 실행 결과**

```text
padding=False:
  sentence 0: 5 tokens
  sentence 1: 75 tokens

padding=True (return_tensors='pt'):
  input_ids shape: torch.Size([2, 75])
  attention_mask sentence 0: tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  attention_mask sentence 1: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

이번엔 `padding="max_length"` 로 배치 길이가 아니라 고정 길이 128에 무조건 맞춥니다. `attention_mask` 에서 1의 비율을 계산해, 짧은 문장이 섞이면 상당 부분이 padding으로 채워져 연산이 낭비됨을 숫자로 확인합니다.

```python
out_fixed = tokenizer(texts, padding="max_length", max_length=128, return_tensors="pt")
print(f"shape: {out_fixed['input_ids'].shape}  (batch 2, max_length=128)")

# attention_mask에서 1의 비율 = 실제 토큰 비율
real_ratio = out_fixed["attention_mask"].sum().item() / out_fixed["attention_mask"].numel()
print(f"\nattention_mask=1 ratio: {real_ratio:.1%}")
print(f"  → short sentence is mostly padding (compute wasted)")
```

**▶ 실행 결과**

```text
shape: torch.Size([2, 128])  (batch 2, max_length=128)

attention_mask=1 ratio: 31.2%
  → short sentence is mostly padding (compute wasted)
```

반대로 너무 긴 입력은 잘라내야 합니다. BERT 계열은 512 토큰이 한계라, truncation 없이 넣으면 길이 초과 경고가 뜹니다. `truncation=True` 와 `max_length=128` 을 주면 앞에서부터 잘라 길이를 맞추고, 마지막 자리에는 항상 `[SEP]` 가 붙는 점을 확인합니다.

```python
# 매우 긴 텍스트 (512 토큰 초과)
very_long = "Hello world! This is a sentence. " * 200

# truncation 없이 (경고 또는 에러)
out_full = tokenizer(very_long)
print(f"truncation=False: {len(out_full['input_ids'])} tokens  (may exceed BERT limit 512)")

# truncation=True + max_length=128
out_trunc = tokenizer(very_long, truncation=True, max_length=128)
print(f"truncation=True, max_length=128: {len(out_trunc['input_ids'])} tokens")
print(f"  Last token: {tokenizer.decode([out_trunc['input_ids'][-1]])} (= [SEP], always appended)")
```

**▶ 실행 결과**

```text
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (1602 > 512). Running this sequence through the model will result in indexing errors
truncation=False: 1602 tokens  (may exceed BERT limit 512)
truncation=True, max_length=128: 128 tokens
  Last token: [SEP] (= [SEP], always appended)
```

`max_length` 를 얼마로 둘지 정하려면 실제 데이터의 길이 분포를 알아야 합니다. 1,000건의 토큰 길이를 모아 min/mean/median/분위수를 보고, max_length를 64-512로 바꿀 때 각각 몇 %가 잘리는지 계산해 길이 선택의 trade-off를 눈으로 봅니다.

```python
# 5,000건의 토큰 길이 분포
lengths = []
for i in range(min(1000, len(small))):
    n = len(tokenizer.tokenize(small[i]["text"]))
    lengths.append(n)
lengths = np.array(lengths)

print(f"Token length distribution over 1,000 samples:")
print(f"  min:    {lengths.min()}")
print(f"  mean:   {lengths.mean():.0f}")
print(f"  median: {int(np.median(lengths))}")
print(f"  p90:    {int(np.percentile(lengths, 90))}")
print(f"  p95:    {int(np.percentile(lengths, 95))}")
print(f"  p99:    {int(np.percentile(lengths, 99))}")
print(f"  max:    {lengths.max()}")

print(f"\nFraction truncated at various max_length:")
for max_len in [64, 128, 256, 512]:
    truncated_pct = (lengths > max_len).mean() * 100
    print(f"  max_length={max_len}: {truncated_pct:5.1f}% truncated")
```

**▶ 실행 결과**

```text
Token length distribution over 1,000 samples:
  min:    2
  mean:   177
  median: 131
  p90:    366
  p95:    470
  p99:    770
  max:    1263

Fraction truncated at various max_length:
  max_length=64:  77.1% truncated
  max_length=128:  50.9% truncated
  max_length=256:  21.6% truncated
  max_length=512:   3.9% truncated
```

**결과 해석**

리뷰 토큰 길이가 median 131, p99 770으로 꼬리가 깁니다. max_length=128로 자르면 절반(50.9%)이 잘리지만 512로 늘리면 4%만 잘립니다. 길이를 키울수록 정보 손실은 줄지만 self-attention 연산이 제곱으로 늘어, T4 30분 제약과 맞바꿔야 하는 선택입니다.

이제 한 건씩이 아니라 5,000건 전체를 한 번에 토큰화합니다. `map(batched=True)` 는 `tokenize_fn` 에 샘플을 묶음으로 넘겨 빠르게 처리하고, 결과로 `input_ids`·`attention_mask` 컬럼이 데이터셋에 추가됩니다. 모든 샘플이 128 길이로 고정됐는지 확인합니다.

```python
def tokenize_fn(batch):
    # batch는 dict of lists: {"text": [..., ...], "label": [..., ...]}
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

# batched=True: tokenize_fn을 batch_size개씩 묶어 호출 (기본 1,000)
tokenized = small.map(tokenize_fn, batched=True, batch_size=200)

print(tokenized)
print(f"\nFirst sample input_ids length: {len(tokenized[0]['input_ids'])}  (= 128, fixed)")
print(f"First sample attention_mask sum: {sum(tokenized[0]['attention_mask'])}  (real tokens)")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask'],
    num_rows: 5000
})

First sample input_ids length: 128  (= 128, fixed)
First sample attention_mask sum: 75  (real tokens)
```

`filter` 로 조건에 맞는 샘플만 골라내는 방법도 같이 봅니다. 별점 4-5(label 3 이상)인 긍정 리뷰와, 단어 100개 이하의 짧은 텍스트를 각각 추려 전체 대비 비율을 출력합니다. 데이터 부분집합을 만드는 표준 도구입니다.

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

토큰화한 데이터셋은 아직 파이썬 리스트라 모델에 바로 못 넣습니다. `with_format("torch")` 로 지정한 컬럼만 PyTorch 텐서로 바꿉니다. `input_ids` 의 dtype이 `int64`, shape이 128로 잡히는지 확인해 모델 입력 형식을 맞춥니다.

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

텐서로 바꾼 데이터셋을 `DataLoader` 로 감싸 배치 단위로 꺼내봅니다. 첫 배치를 뽑아 `input_ids` shape이 `[batch_size, max_length]` 인 8x128로 묶이는지, label이 배치 크기만큼 함께 따라오는지 확인합니다.

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

고정 길이 padding의 낭비를 줄이는 방법이 동적 padding입니다. 토큰화 단계에서는 padding을 빼고 truncation만 적용해 샘플마다 길이가 제각각인 상태로 둡니다. 앞 10개 길이를 찍어, 이대로는 하나의 텐서로 묶을 수 없음을 확인합니다.

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

길이가 제각각인 샘플은 `DataCollatorWithPadding` 을 `collate_fn` 자리에 끼워 해결합니다. 이 collator는 배치를 만들 때마다 그 배치 안 가장 긴 길이에 맞춰 동적으로 padding합니다. 배치별 shape과 채움 비율(fill)을 찍어 매 배치 길이가 달라질 수 있음을 봅니다.

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

이 50샘플에서는 동적 padding이 정적과 똑같이 76.1%·절감 0%로 나옵니다. max_length=128로 자른 리뷰의 절반 이상이 128 토큰을 꽉 채워, 배치마다 가장 긴 샘플이 거의 항상 128이라 동적 padding도 결국 128까지 채우기 때문입니다. 동적 padding의 이득은 짧은 문장이 섞인 데이터에서 커지고, 긴 리뷰처럼 대부분이 상한에 닿는 데이터에서는 이렇게 사라질 수 있습니다.

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

masking이 실제로 토큰을 어떻게 바꾸는지 첫 샘플 앞부분을 표로 펼쳐 봅니다. 원래 토큰, collator가 가린 토큰, 그리고 모델이 맞춰야 할 label을 나란히 두면 가려진 자리(`*`)만 label에 원래 토큰이 남고 나머지는 `-100`(ignored)으로 빠지는 게 보입니다.

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

CLM에서는 토큰을 가리지 않으므로 input과 label이 같은 위치에서 일치합니다. 첫 샘플 앞 20자리를 표로 펼쳐 토큰과 label이 그대로 겹치는지 확인합니다. 다음 토큰 예측을 위한 shift-by-one은 collator가 아니라 모델 forward 안에서 처리됩니다.

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
