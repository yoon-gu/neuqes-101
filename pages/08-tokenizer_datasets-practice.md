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

로드한 `DatasetDict` 의 train split이 어떤 구조인지 들여다봅니다. 샘플 수, `features` 스키마(라벨이 `ClassLabel`, 텍스트가 `string`), 그리고 첫 샘플의 라벨·텍스트를 직접 출력해 데이터의 모습을 확인합니다.

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

65만 건 전체를 다룰 필요는 없으므로, `shuffle(seed=42)` 로 결정론적으로 섞은 뒤 `select(range(5000))` 로 앞 5,000건만 골라냅니다. Phase 0(Ch 1-6)에서 쓴 것과 동일한 subsample이라, 같은 데이터가 이번엔 토크나이저를 통과하는 모습을 보게 됩니다.

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

Ch 7과 같은 `distilbert-base-uncased` 토크나이저를 불러옵니다. 클래스 이름과 vocab 크기, 그리고 패딩에 쓰일 `[PAD]` 토큰의 id를 확인해 둡니다 — 뒤에서 padding을 줄 때 이 id가 빈자리를 채웁니다.

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

옵션 없이 `tokenizer(sample)` 를 호출하면 기본 동작을 봅니다. 첫 샘플 텍스트를 토큰화해 `input_ids` 의 길이, 앞 30개 id, 그리고 그 id를 다시 텍스트로 `decode` 한 결과를 나란히 출력합니다.

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

**결과 해석**

decode 결과 맨 앞의 `[CLS]` 는 토크나이저가 자동으로 붙인 특수 토큰(id 101)이고, 그 뒤로 소문자화된 원문이 이어집니다 — `uncased` 모델이라 대문자가 모두 소문자로 바뀐 점에 주목하세요. `i ' ve` 처럼 `I've` 가 세 토큰으로 쪼개진 것도 WordPiece가 구두점을 분리한 결과입니다.

길이가 크게 다른 두 문장을 한 배치로 묶어 padding의 효과를 봅니다. 먼저 `padding=False` (기본)로 각 문장이 제 길이 그대로인지 확인하고, 이어 `padding=True` 로 묶었을 때 짧은 문장이 긴 문장 길이까지 채워지면서 `attention_mask` 가 어떻게 0/1로 표시되는지 비교합니다.

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

**결과 해석**

`padding=True` 는 배치 안 가장 긴 문장(75 토큰)에 맞춰 짧은 문장을 채우므로 shape이 `[2, 75]` 가 됩니다. 짧은 문장 0번은 앞 5칸만 `attention_mask=1` 이고 나머지는 `0` 으로, 모델이 그 자리를 패딩으로 인식해 무시하도록 표시된 것입니다.

이번엔 `padding="max_length"` 로 두 문장을 *항상 128* 까지 채웁니다. 배치 안 longest와 무관하게 고정 길이가 되며, `attention_mask` 의 1 비율을 계산해 짧은 문장에서 얼마나 많은 자리가 패딩으로 낭비되는지 숫자로 확인합니다.

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

**결과 해석**

실제 토큰이 전체의 31.2%뿐이라는 건 약 69%가 패딩이라는 뜻 — 이만큼의 self-attention 계산이 그저 버려집니다. 고정 길이(`max_length`)가 주는 일정한 shape의 대가가 이 낭비이며, 뒤에서 동적 padding으로 이 비율을 끌어올리게 됩니다.

BERT 계열은 사전학습 한도가 512 토큰이라 그보다 긴 입력은 그대로 모델에 넣을 수 없습니다. 일부러 긴 텍스트를 만들어, `truncation` 없이 호출하면 경고와 함께 1,602 토큰이 나오는 것과 `truncation=True, max_length=128` 로 자르면 정확히 128로 맞춰지는 것을 비교합니다.

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

**결과 해석**

`truncation` 없이 1,602 토큰을 만들면 transformers가 512 초과를 경고합니다 — 이 상태로 모델에 넣으면 인덱싱 에러가 납니다. `truncation=True` 로 자른 뒤에도 마지막 토큰이 `[SEP]` 인 점이 핵심으로, 토크나이저는 단순히 뒤를 잘라내는 게 아니라 잘린 끝에 문장 종료 토큰을 항상 다시 붙여 줍니다.

`max_length` 를 어림짐작이 아니라 데이터로 정하기 위해, 앞 1,000건의 토큰 길이 분포(min/mean/median/percentile)를 구합니다. 이어 `64/128/256/512` 각각에서 몇 %가 잘리는지 계산해, 길이 상한과 정보 손실의 trade-off를 한눈에 봅니다.

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

평균은 177이지만 median이 131이라 분포가 오른쪽으로 길게 늘어진 형태(소수의 매우 긴 리뷰가 평균을 끌어올림)입니다. `max_length=128` 이면 절반가량(50.9%)이 잘리지만, 잘리는 건 대부분 뒷부분 세부 묘사라 별점 예측에는 영향이 작아 이 커리큘럼의 표준값으로 씁니다 — 손실을 더 줄이려면 256/512 쪽으로 올리되 그만큼 계산이 늘어납니다.

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
