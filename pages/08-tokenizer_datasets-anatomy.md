## 토크나이저 옵션 직접 실험

Ch 7과 같은 `distilbert-base-uncased` 토크나이저로 시작합니다 (사전학습 모델 그대로).

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

### 옵션 없이 — 한 문장 토큰화 (기본 동작)

```python
tokenizer(text)
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

### 두 문장 배치 + `padding=True` — *동적 패딩*

여러 문장을 한 배치로 묶으려면 길이가 모두 같아야 텐서가 만들어집니다. `padding=True` 옵션을 주면 한 배치 안에서 **가장 긴 문장 길이에 맞춰 짧은 문장만 패딩으로 채우므로**, 배치마다 필요한 만큼만 늘어나 가장 효율적입니다.

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

### `padding="max_length"`, `max_length=128` — *고정 길이*

배치마다 길이가 달라지는 게 싫을 때 (TPU·정적 그래프 환경) 항상 `max_length` 까지 padding합니다.

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

### `truncation=True` — 긴 입력 자르기

BERT 계열은 사전학습 시 `max_length=512` 로 학습돼서 그보다 긴 입력은 처리할 수 없습니다. `truncation=True` 로 자동 절단.

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

### attention_mask가 self-attention에서 하는 일

핵심: 패딩 토큰이 *다른 토큰의 표현에 영향을 주지 않도록* 막습니다.

```python
# 모델 내부에서 (단순화):
attention_scores = Q @ K.T / sqrt(d_k)
attention_scores[mask == 0] = -inf       # 패딩 자리 점수를 -inf로
attention_weights = softmax(attention_scores)  # softmax 후 그 자리는 ~0
output = attention_weights @ V
```

`-inf` 가 softmax를 거치면 `e^(-inf) = 0` 이 되어 패딩 토큰의 가중치가 정확히 0이 됩니다. 결과적으로 아무리 길게 패딩을 붙여도 학습 결과는 변하지 않고, 그만큼의 계산이 그저 버려질 뿐입니다.

확인: 위 padding="max_length" 출력에서 attention_mask=1 비율이 낮으면 그만큼 *낭비된 계산* 입니다.

### `max_length` 결정 — 데이터 길이 분포 보고 정하기

너무 작으면 *정보 손실* (긴 리뷰가 잘림), 너무 크면 *낭비* (대부분 패딩). 실제 데이터의 토큰 길이 분포를 보고 정합니다.

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

**해석**: 학습 시 `max_length=128` 로 두면 95% 이상 정상, 5% 정도만 잘립니다 (Yelp 리뷰가 대부분 짧음). `max_length=512` 면 거의 모든 리뷰를 보존하지만 평균 패딩 비율이 60-70%라 메모리·시간 낭비.

이 커리큘럼은 **`max_length=128`** 을 표준으로 씁니다 (T4 30분 제약 + 무난한 정보 보존 균형).
