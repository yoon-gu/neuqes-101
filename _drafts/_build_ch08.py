"""Build 08_tokenizer_datasets/08_tokenizer_datasets.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "08_tokenizer_datasets" / "08_tokenizer_datasets.ipynb"

cells = []
_counter = 0


def _cid():
    global _counter
    _counter += 1
    return f"cell{_counter:03d}"


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": text,
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ----- 1. Title -----
md(r"""# Chapter 8. Tokenizer 옵션 깊이 + `datasets` 라이브러리

**목표**: Ch 7에서 만난 WordPiece 토크나이저와 사전학습 모델로 *Phase 0의 Yelp 데이터를 다시* 만납니다. `padding` / `truncation` / `max_length` 옵션의 의미를 손에 익히고, `datasets` 라이브러리로 65만 건 코퍼스를 메모리 걱정 없이 다룹니다. Ch 9 학습의 *입력 파이프라인* 이 이 챕터에서 완성됩니다.

**환경**: Google Colab — CPU도 OK (이번 챕터도 학습 없음). T4 권장.

**예상 소요 시간**: 약 10분 (모델 가중치 다운로드는 안 함, 토크나이저 + 데이터 로딩만)

---

## 학습 흐름

1. 🚀 **실습**: `datasets.load_dataset` 으로 Yelp 65만 건 로드 → 5,000건 subsample
2. 🔬 **해부**: 토크나이저 옵션 3종 (`padding`, `truncation`, `max_length`) 직접 실험 + `attention_mask` 가 학습에 어떻게 쓰이는지
3. 🛠️ **변형**: `datasets.map` 으로 5,000건 일괄 토큰화 → `DataLoader` 까지 변환 (Ch 9 학습 입력의 모습)""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2-6 | sklearn 모델들 | `TfidfVectorizer()` | Yelp 변형 | 1차원/K차원 | 없음/sigmoid/softmax | MSE/BCE/CE |
| 7 | `pipeline("sentiment-analysis")` | `AutoTokenizer.from_pretrained(...)` | 간단 영어 예시 | 사전학습 헤드 | softmax | — |
| **8 ← 여기** | (모델 없음 — 토크나이저·데이터 파이프라인만) | `AutoTokenizer.from_pretrained(...)` | **Yelp 5,000 (Phase 0과 동일)** | — | — | — |

전체 19챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 7)

| 축 | Ch 7 | Ch 8 |
|---|---|---|
| 모델 | `pipeline` + `AutoModelForSequenceClassification` | **모델 로드 없음** (다음 챕터 학습 준비 단계) |
| 토크나이저 | WordPiece — 한 문장 시연 | **WordPiece + 옵션 학습** (`padding` / `truncation` / `max_length`) |
| 데이터 | 간단 영어 예시 문장 | **`datasets` 로 Yelp 65만 → 5,000 subsample** (Phase 0과 동일 데이터) |
| 데이터 라이브러리 | (없음) | **`datasets`** 첫 등장 — `load_dataset`, `map`, `filter`, `with_format` |
| 학습 단계 | 추론만 | 학습·추론 모두 없음 — 데이터 파이프라인 *연습* |

**왜 이 챕터?** Ch 9 BERT 회귀에서 `Trainer.train()` 한 줄을 부르려면 *그 한 줄에 무엇을 먹여야 하는지* 알아야 합니다 — `Dataset` 객체, `padding/truncation` 결정, `DataLoader` 변환. 이 챕터는 그 입력 형태를 *학습 없이* 미리 손에 익히는 자리입니다.

**Phase 0와의 다리**: Yelp 5,000건은 Ch 1-6에서 줄곧 쓴 데이터. 같은 텍스트가 TF-IDF에서 sparse vector로 갔던 길이, 이번엔 WordPiece에서 `input_ids` + `attention_mask` 텐서 쌍으로 가는 길을 봅니다.""")

# ----- 4. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트 — `padding` / `truncation` / `max_length`

WordPiece가 출력하는 시퀀스 길이는 입력 텍스트마다 다릅니다. 그런데 모델은 **고정된 shape의 텐서 배치** 를 받아야 하므로, 이 둘 사이를 맞추는 세 가지 옵션이 있습니다.

| 옵션 | 의미 | 언제 쓰나 |
|---|---|---|
| `padding=False` | 패딩 없음 (기본값). 시퀀스 길이가 다 다름 | 한 문장씩 처리할 때 |
| `padding=True` | **배치 안 가장 긴 시퀀스 길이까지** padding | 일반 학습 (효율적, dynamic padding) |
| `padding="max_length"` | **항상 `max_length` 까지** padding (짧으면 패딩, 길면 자름) | TPU·고정 shape 필요할 때 |
| `truncation=True` | `max_length` 초과분은 *잘라냄* | 항상 같이 두는 게 안전 (긴 입력 방지) |
| `max_length=N` | 길이 상한 (모델별 사전학습 한도 — BERT 512) | 메모리/속도 trade-off |

**패딩이 들어간 자리는 `attention_mask=0`** 으로 표시됩니다. 모델은 이 mask를 보고 self-attention에서 패딩 토큰을 무시합니다 — *아무리 길게 패딩해도 학습 결과에 영향 없음* (속도·메모리만 손해).

이번 챕터에서 위 세 옵션을 직접 호출해 input_ids와 attention_mask가 어떻게 변하는지 봅니다.""")

# ----- 5. install -----
code(r"""!pip install -q transformers datasets""")

# ----- 6. import + GPU check -----
code(r"""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

print(f"PyTorch:       {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
print("\n이번 챕터는 모델 가중치 로드 없음 → VRAM 거의 변화 없습니다.")""")

# ----- 7. Yelp 로드 도입 -----
md(r"""## 1. 🚀 `datasets` 로 Yelp 로드

`load_dataset("yelp_review_full")` 한 줄로 Hugging Face Hub에서 65만 건 학습 데이터를 받아옵니다 (50K test). 처음 받으면 ~150MB 다운로드 + 디스크 캐시.

**주목할 점** — `datasets` 는 Apache Arrow 형식으로 디스크에 저장하고 *메모리맵* 으로 접근합니다. 65만 건 전체가 RAM에 올라가는 게 아니라, 인덱싱하는 시점에만 디스크에서 읽어요. RAM 영향이 거의 없는 게 핵심.""")

# ----- 8. load_dataset -----
code(r"""ds = load_dataset("yelp_review_full")
print(ds)""")

# ----- 9. 데이터 구조 -----
code(r"""# train split의 첫 샘플 + features 확인
print(f"train 샘플 수: {len(ds['train']):,}")
print(f"test 샘플 수:  {len(ds['test']):,}")
print(f"\nfeatures: {ds['train'].features}")
print(f"\n첫 샘플:")
print(f"  label: {ds['train'][0]['label']}  (0-4 = 별점 1-5)")
print(f"  text:  {ds['train'][0]['text'][:200]}...")""")

# ----- 10. subsample -----
code(r"""# 5,000건만 subsample (Phase 0와 동일한 처리)
small = ds["train"].shuffle(seed=42).select(range(5000))
print(small)
print(f"\nfirst sample text: {small[0]['text'][:150]}...")""")

# ----- 11. 토크나이저 로드 -----
md(r"""## 2. 🔬 토크나이저 옵션 직접 실험

Ch 7과 같은 `distilbert-base-uncased` 토크나이저로 시작합니다 (사전학습 모델 그대로).""")

code(r"""tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print(f"클래스:    {type(tokenizer).__name__}")
print(f"vocab:     {tokenizer.vocab_size:,}")
print(f"pad_token: {tokenizer.pad_token}  (id={tokenizer.pad_token_id})")""")

# ----- 12. 단일 문장 토큰화 -----
md(r"""### 옵션 없이 — 한 문장 토큰화 (기본 동작)

```python
tokenizer(text)
```
""")

code(r"""sample = small[0]["text"]
print(f"입력 (앞 150자): {sample[:150]}...\n")

out = tokenizer(sample)
print(f"input_ids 길이: {len(out['input_ids'])}")
print(f"앞 30개 ID:    {out['input_ids'][:30]}")
print(f"디코딩 앞 30개: {tokenizer.decode(out['input_ids'][:30])}")""")

# ----- 13. 두 문장 + padding=True -----
md(r"""### 두 문장 배치 + `padding=True` — *동적 패딩*

여러 문장을 묶어 배치로 만들 때, 길이가 다 다르면 텐서로 만들 수 없습니다. `padding=True` 는 **배치 안 가장 긴 시퀀스 길이까지만** 패딩 — 가장 효율적입니다.""")

code(r"""# 길이가 다른 두 문장을 묶기
short_text = "Great service!"
long_text = small[0]["text"]
texts = [short_text, long_text]

# padding=False (기본): 각 문장 길이 그대로
out_no_pad = tokenizer(texts, padding=False)
print(f"padding=False:")
for i, ids in enumerate(out_no_pad["input_ids"]):
    print(f"  문장 {i}: {len(ids)}개 토큰")

# padding=True: 가장 긴 길이까지만 padding
out_dyn = tokenizer(texts, padding=True, return_tensors="pt")
print(f"\npadding=True (return_tensors='pt'):")
print(f"  input_ids shape: {out_dyn['input_ids'].shape}")
print(f"  attention_mask 첫 문장: {out_dyn['attention_mask'][0][:20]}")
print(f"  attention_mask 둘째 문장: {out_dyn['attention_mask'][1][:20]}")""")

# ----- 14. padding=max_length -----
md(r"""### `padding="max_length"`, `max_length=128` — *고정 길이*

배치마다 길이가 달라지는 게 싫을 때 (TPU·정적 그래프 환경) 항상 `max_length` 까지 padding합니다.""")

code(r"""out_fixed = tokenizer(texts, padding="max_length", max_length=128, return_tensors="pt")
print(f"shape: {out_fixed['input_ids'].shape}  (배치 2, max_length=128)")

# attention_mask에서 1의 비율 = 실제 토큰 비율
real_ratio = out_fixed["attention_mask"].sum().item() / out_fixed["attention_mask"].numel()
print(f"\nattention_mask=1 비율: {real_ratio:.1%}")
print(f"  → 짧은 문장은 거의 패딩만 (계산 낭비)")""")

# ----- 15. truncation -----
md(r"""### `truncation=True` — 긴 입력 자르기

BERT 계열은 사전학습 시 `max_length=512` 로 학습돼서 그보다 긴 입력은 처리할 수 없습니다. `truncation=True` 로 자동 절단.""")

code(r"""# 매우 긴 텍스트 (512 토큰 초과)
very_long = "Hello world! This is a sentence. " * 200

# truncation 없이 (경고 또는 에러)
out_full = tokenizer(very_long)
print(f"truncation=False: {len(out_full['input_ids'])} 토큰  (BERT 한도 512 초과 가능)")

# truncation=True + max_length=128
out_trunc = tokenizer(very_long, truncation=True, max_length=128)
print(f"truncation=True, max_length=128: {len(out_trunc['input_ids'])} 토큰")
print(f"  마지막 토큰: {tokenizer.decode([out_trunc['input_ids'][-1]])} (= [SEP], 항상 끝에 붙음)")""")

# ----- 16. attention_mask 깊이 -----
md(r"""### attention_mask가 self-attention에서 하는 일

핵심: 패딩 토큰이 *다른 토큰의 표현에 영향을 주지 않도록* 막습니다.

```python
# 모델 내부에서 (단순화):
attention_scores = Q @ K.T / sqrt(d_k)
attention_scores[mask == 0] = -inf       # 패딩 자리 점수를 -inf로
attention_weights = softmax(attention_scores)  # softmax 후 그 자리는 ~0
output = attention_weights @ V
```

`-inf` 가 softmax를 거치면 `e^(-inf) = 0` → 패딩 토큰은 가중치가 정확히 0. 그래서 *아무리 길게 패딩해도 학습 결과는 같음* — 속도·메모리만 손해.

확인: 위 padding="max_length" 출력에서 attention_mask=1 비율이 낮으면 그만큼 *낭비된 계산* 입니다.""")

# ----- 17. max_length 결정 도입 -----
md(r"""### `max_length` 결정 — 데이터 길이 분포 보고 정하기

너무 작으면 *정보 손실* (긴 리뷰가 잘림), 너무 크면 *낭비* (대부분 패딩). 실제 데이터의 토큰 길이 분포를 보고 정합니다.""")

code(r"""# 5,000건의 토큰 길이 분포
lengths = []
for i in range(min(1000, len(small))):
    n = len(tokenizer.tokenize(small[i]["text"]))
    lengths.append(n)
lengths = np.array(lengths)

print(f"1,000개 샘플의 토큰 길이 분포:")
print(f"  min:    {lengths.min()}")
print(f"  mean:   {lengths.mean():.0f}")
print(f"  median: {int(np.median(lengths))}")
print(f"  p90:    {int(np.percentile(lengths, 90))}")
print(f"  p95:    {int(np.percentile(lengths, 95))}")
print(f"  p99:    {int(np.percentile(lengths, 99))}")
print(f"  max:    {lengths.max()}")

print(f"\n다양한 max_length에서 잘리는 비율:")
for max_len in [64, 128, 256, 512]:
    truncated_pct = (lengths > max_len).mean() * 100
    print(f"  max_length={max_len}: {truncated_pct:5.1f}% 가 잘림")""")

# ----- 18. max_length 해석 -----
md(r"""**해석**: 학습 시 `max_length=128` 로 두면 95% 이상 정상, 5% 정도만 잘립니다 (Yelp 리뷰가 대부분 짧음). `max_length=512` 면 거의 모든 리뷰를 보존하지만 평균 패딩 비율이 60-70%라 메모리·시간 낭비.

이 커리큘럼은 **`max_length=128`** 을 표준으로 씁니다 (T4 30분 제약 + 무난한 정보 보존 균형).""")

# ----- 19. datasets.map 도입 -----
md(r"""## 3. 🛠️ `datasets.map` — 5,000건 일괄 토큰화

샘플 하나씩 `tokenizer(...)` 부르는 건 5,000번 호출이 필요하고 느립니다. `dataset.map(fn, batched=True)` 로 *배치 단위* 일괄 호출이 표준.""")

code(r"""def tokenize_fn(batch):
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
print(f"\n첫 샘플의 input_ids 길이: {len(tokenized[0]['input_ids'])}  (= 128, 고정)")
print(f"첫 샘플의 attention_mask 합: {sum(tokenized[0]['attention_mask'])}  (실제 토큰 수)")""")

# ----- 20. .filter -----
md(r"""### `dataset.filter` — 조건에 맞는 샘플만 선별""")

code(r"""# 별점 4-5 (label 3-4) 만
positive = small.filter(lambda x: x["label"] >= 3)
print(f"긍정 샘플: {len(positive):,} / {len(small):,} = {len(positive)/len(small):.1%}")

# 짧은 텍스트만 (예: 100단어 이하)
short = small.filter(lambda x: len(x["text"].split()) <= 100)
print(f"짧은 샘플: {len(short):,} / {len(small):,} = {len(short)/len(small):.1%}")""")

# ----- 21. with_format -----
md(r"""### `with_format("torch")` — 텐서 형식으로""")

code(r"""# 모델에 바로 먹일 수 있도록 PyTorch tensor로 변환
tokenized_torch = tokenized.with_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"],
)

sample = tokenized_torch[0]
print(f"input_ids:      {type(sample['input_ids']).__name__}, dtype={sample['input_ids'].dtype}, shape={sample['input_ids'].shape}")
print(f"attention_mask: {type(sample['attention_mask']).__name__}, shape={sample['attention_mask'].shape}")
print(f"label:          {sample['label']}  (0-4 = 별점 1-5)")""")

# ----- 22. DataLoader 도입 -----
md(r"""## 4. `DataLoader` 변환 — Ch 9 학습 입력 미리보기

PyTorch `DataLoader` 는 dataset을 받아 *배치 + shuffle* 을 자동 처리합니다. Ch 9의 `Trainer` 가 내부에서 이걸 만들어 쓰지만, 직접 만들 줄 알면 디버깅에 유리.""")

code(r"""from torch.utils.data import DataLoader

loader = DataLoader(tokenized_torch, batch_size=8, shuffle=True)

# 첫 배치 살펴보기
batch = next(iter(loader))
print(f"batch keys:           {list(batch.keys())}")
print(f"input_ids shape:      {batch['input_ids'].shape}      (= [batch_size, max_length])")
print(f"attention_mask shape: {batch['attention_mask'].shape}")
print(f"label shape:          {batch['label'].shape}")
print(f"label 값:             {batch['label'].tolist()}")""")

# ----- 23. DataCollator 미리보기 -----
md(r"""**조금 더 — `DataCollatorWithPadding`**: 위 코드는 `padding="max_length"` 로 *모든* 샘플을 128로 만들어둔 상태. 그러나 *동적 패딩* (배치 내 longest까지만)이 더 효율적입니다. 이때 `DataCollatorWithPadding` 을 `DataLoader` 의 `collate_fn` 으로 넘깁니다 — Ch 9 `Trainer` 가 자동으로 사용.

```python
from transformers import DataCollatorWithPadding

# 토큰화는 padding 안 하고 truncation만:
def tokenize_fn_dyn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)
tokenized_dyn = small.map(tokenize_fn_dyn, batched=True)
tokenized_dyn = tokenized_dyn.with_format("torch", columns=["input_ids", "attention_mask", "label"])

# DataLoader가 배치 시점에 동적 padding 적용
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
dyn_loader = DataLoader(tokenized_dyn, batch_size=8, shuffle=True, collate_fn=collator)
```""")

# ----- 24. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

### `datasets`

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `datasets.load_dataset` | Hugging Face Hub에서 데이터셋 다운로드 + Apache Arrow 캐시 | Ch 9 학습 데이터 로드 |
| `Dataset.shuffle(seed)` | 결정론적 셔플 | 재현 가능한 학습 분할 |
| `Dataset.select(indices)` | 지정 인덱스만 선택 | subsample, train/val/test 분할 |
| `Dataset.map(fn, batched=True, batch_size=...)` | 배치 단위 변환 (토큰화 등). 결과 자동 캐시 | Ch 9 입력 전처리 |
| `Dataset.filter(fn)` | 조건 만족 샘플만 | 라벨 필터링, 길이 제한 등 |
| `Dataset.with_format("torch", columns=[...])` | PyTorch tensor 출력 | 모든 학습 챕터 |

### `transformers` 토크나이저 옵션

| 옵션 | 의미 |
|---|---|
| `padding=True` | 배치 내 가장 긴 길이까지 padding (동적) |
| `padding="max_length"` | 항상 max_length까지 padding (고정) |
| `truncation=True` | max_length 초과분 잘라냄 |
| `max_length=N` | 길이 상한 (기본 모델 한도) |
| `return_tensors="pt"` | PyTorch 텐서로 반환 (`"tf"`, `"np"` 도 가능) |
| `tokenizer.decode(ids)` | ID → 텍스트 역변환 |

### PyTorch 데이터 도구

| 이름 | 한 줄 설명 |
|---|---|
| `torch.utils.data.DataLoader` | 배치 + shuffle + multiprocessing |
| `transformers.DataCollatorWithPadding` | DataLoader의 `collate_fn` — 배치 시점에 동적 padding |""")

# ----- 25. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. `padding=True` 와 `padding="max_length"` 는 각각 어떤 상황에 적합한가요? 메모리·속도 차이는?
2. `attention_mask=0` 인 자리는 self-attention에서 어떻게 무시되나요? (`-inf` 트릭)
3. `datasets` 가 65만 건 데이터를 RAM에 다 안 올리고 인덱싱할 수 있는 이유는 무엇인가요?
4. `dataset.map(fn, batched=True)` 와 `batched=False` 의 속도 차이는 어디에서 오나요?""")

# ----- 26. FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) `padding=True` 와 `padding="max_length"` 중 뭘 써야 하나요?

대부분 **`padding=True` (동적)** 이 효율적입니다.

- **`padding=True`**: 배치 안 가장 긴 시퀀스 길이까지만 padding. 짧은 배치는 적은 토큰만 처리해 속도·메모리 절약. 일반 학습/추론에서 표준.
- **`padding="max_length"`**: 항상 같은 길이. TPU나 정적 graph(예: TorchScript)처럼 *shape이 매 배치 동일해야 하는* 환경에서. CPU/GPU 학습엔 보통 불필요.

`Trainer` 와 `DataCollatorWithPadding` 조합이 동적 padding을 자동 처리하므로, 데이터 전처리 시엔 **`padding=False`** (또는 생략) + collator에 padding을 맡기는 게 흔한 패턴입니다.

### Q2. (실무) `max_length` 를 작게 하면 학습이 빨라지지만 성능에 영향은?

self-attention 비용은 *시퀀스 길이의 제곱* — `max_length=128` vs `512` 면 4배 차이가 아니라 **16배 차이**. 작게 하면 매우 빨라지고 메모리도 4배 절약.

성능 영향은 **데이터 분포에 따라**:

```python
# 데이터의 95th percentile을 보고 정함
p95 = int(np.percentile(token_lengths, 95))
chosen_max = ((p95 // 32) + 1) * 32   # 32의 배수로 올림 (GPU 친화)
```

Yelp는 평균 ~150 토큰이라 `max_length=128` 이 95% 이상 정상 처리. 더 긴 문서가 많은 데이터(예: 논문 abstract)는 `256` 또는 `512` 가 안전.

### Q3. (이론) `attention_mask` 가 정확히 모델에서 어떻게 쓰이나요?

self-attention 계산 직전에 *mask=0인 위치의 점수를 `-inf` 로 바꿉니다*.

```python
# 단순화한 BERT의 attention 계산
scores = Q @ K.T / sqrt(d_k)              # (seq, seq) 점수
scores = scores + (1.0 - mask) * -10000   # 패딩 자리에 큰 음수
weights = softmax(scores, dim=-1)         # softmax 후 패딩 자리 ≈ 0
output = weights @ V
```

softmax 안에서 `e^(-10000) ≈ 0` 이라 패딩 토큰은 다른 토큰의 표현에 *전혀* 기여하지 않습니다. 학습된 모델 입장에선 패딩이 있든 없든 같은 결과 — 단지 *계산을 낭비* 한 셈.

### Q4. (실무) 데이터셋이 너무 커서 `map` 이 오래 걸리는데 어떻게 하나요?

세 가지 가속 기법.

1. **`batched=True`**: 토크나이저는 batch 호출이 1샘플 호출보다 훨씬 빠름 (Rust 백엔드 활용). 보통 5-10배.
2. **`num_proc=N`**: 여러 프로세스 병렬화. CPU 코어 수만큼.
   ```python
   tokenized = small.map(tokenize_fn, batched=True, num_proc=4)
   ```
3. **자동 캐시**: `map` 결과는 디스크에 자동 저장. 같은 함수·같은 데이터로 다시 부르면 즉시 로드 (해시로 식별). Colab 세션이 끝나면 캐시 사라지지만, Drive 마운트로 보존 가능.

### Q5. (이론) `datasets` 가 메모리 효율적인 이유는 무엇인가요?

핵심: **Apache Arrow + memory-mapped files**.

- **Apache Arrow**: column-oriented 바이너리 포맷. 같은 type 데이터를 연속된 메모리에 저장 → cache hit 좋음, 압축 효율 좋음.
- **memory-mapped (mmap)**: 디스크 파일을 가상 메모리에 *연결만* 해두고 *접근하는 페이지만* 실제 RAM에 올림. 65만 건 데이터셋 객체를 변수에 할당해도 RAM 사용량은 거의 안 늘어남.

```python
import psutil
process = psutil.Process()
mem_before = process.memory_info().rss / 1024**2

ds = load_dataset("yelp_review_full")  # 65만 건

mem_after = process.memory_info().rss / 1024**2
print(f"메모리 증가: {mem_after - mem_before:.1f} MB  (수십 MB 정도)")
```

대조적으로 pandas로 같은 데이터를 읽으면 GB 단위 메모리가 필요합니다.

### Q6. (실무) `dataset.set_format` 과 `with_format` 의 차이는?

- `set_format(type, columns)`: **in-place** 변경. dataset 객체 자체의 출력 형식 변경.
- `with_format(type, columns)`: **새 dataset 반환**. 원본은 그대로.

```python
# in-place
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# tokenized[0] 이 이제 torch tensor

# 새 객체
tokenized_torch = tokenized.with_format("torch", columns=[...])
# tokenized 는 그대로, tokenized_torch가 새 형식
```

대규모 파이프라인에선 `with_format` 으로 *변환마다 새 객체* 가 안전 (디버깅 시 원본 비교).""")

# ----- 27. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

다음 코드를 돌려보고, 두 결과의 길이가 왜 다른지 예측해보세요.

```python
text = "Hugging Face is amazing!"
out1 = tokenizer(text)
out2 = tokenizer(text, add_special_tokens=False)

print(f"기본:                    {len(out1['input_ids'])}, {tokenizer.decode(out1['input_ids'])}")
print(f"add_special_tokens=False: {len(out2['input_ids'])}, {tokenizer.decode(out2['input_ids'])}")
```

힌트: 기본값에선 `[CLS]` 와 `[SEP]` 가 자동으로 *2개* 추가됩니다 — `add_special_tokens=False` 로 끄면 빠집니다. 분류 작업에선 거의 항상 켜둬야 하지만(BERT 사전학습 시 [CLS] 자리에 분류 신호가 모이도록 학습됐으므로), 디버깅·연구용으로 끄는 경우가 있습니다.""")

# ----- 28. next -----
md(r"""## 다음 챕터 예고

**Chapter 9. BERT 회귀 — 첫 파인튜닝, 첫 `Trainer`**

- Phase 0의 Yelp 별점 회귀(Ch 2)를 *DistilBERT 파인튜닝* 으로 다시 풀기 — sklearn `LinearRegression` 1초 학습 vs BERT T4 GPU 5-10분 학습
- **`Trainer` 본격 등장** — `TrainingArguments`, `compute_metrics`, evaluation loop
- 이번 챕터의 데이터 파이프라인(`datasets.map` + `with_format("torch")`)이 그대로 입력으로 들어감
- Loss는 `MSELoss` (Ch 2와 동일) — `problem_type="regression"` 으로 자동 매핑
- **GPU 필수** — fp16 옵션 등장""")


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT.relative_to(REPO)}  ({len(cells)} cells)")
