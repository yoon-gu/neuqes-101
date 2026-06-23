**목표**: Ch 7에서 만난 WordPiece 토크나이저와 사전학습 모델로 *Phase 0의 Yelp 데이터를 다시* 만납니다. `padding` / `truncation` / `max_length` 옵션의 의미를 손에 익히고, `datasets` 라이브러리로 65만 건 코퍼스를 메모리 걱정 없이 다룹니다. Ch 9 학습의 *입력 파이프라인* 이 이 챕터에서 완성됩니다.

**환경**: Google Colab — CPU도 OK (이번 챕터도 학습 없음). T4 권장.

**예상 소요 시간**: 약 10분 (모델 가중치 다운로드는 안 함, 토크나이저 + 데이터 로딩만)


## 학습 흐름

1. 🚀 **실습**: `datasets.load_dataset` 으로 Yelp 65만 건 로드 → 5,000건 subsample
2. 🔬 **해부**: 토크나이저 옵션 3종 (`padding`, `truncation`, `max_length`) 직접 실험 + `attention_mask` 가 학습에 어떻게 쓰이는지
3. 🛠️ **변형**: `datasets.map` 으로 5,000건 일괄 토큰화 → `DataLoader` 까지 변환 (Ch 9 학습 입력의 모습)

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2-6 | sklearn 모델들 | `TfidfVectorizer()` | Yelp 변형 | 1차원/K차원 | 없음/sigmoid/softmax | MSE/BCE/CE |
| 7 | `pipeline("sentiment-analysis")` | `AutoTokenizer.from_pretrained(...)` | 간단 영어 예시 | 사전학습 헤드 | softmax | — |
| **8 ← 여기** | (모델 없음 — 토크나이저·데이터 파이프라인만) | `AutoTokenizer.from_pretrained(...)` | **Yelp 5,000 (Phase 0과 동일)** | — | — | — |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 7)

| 축 | Ch 7 | Ch 8 |
|---|---|---|
| 모델 | `pipeline` + `AutoModelForSequenceClassification` | **모델 로드 없음** (다음 챕터 학습 준비 단계) |
| 토크나이저 | WordPiece — 한 문장 시연 | **WordPiece + 옵션 학습** (`padding` / `truncation` / `max_length`) |
| 데이터 | 간단 영어 예시 문장 | **`datasets` 로 Yelp 65만 → 5,000 subsample** (Phase 0과 동일 데이터) |
| 데이터 라이브러리 | (없음) | **`datasets`** 첫 등장 — `load_dataset`, `map`, `filter`, `with_format` |
| 학습 단계 | 추론만 | 학습·추론 모두 없음 — 데이터 파이프라인 *연습* |

**왜 이 챕터?** Ch 9 BERT 회귀에서 `Trainer.train()` 한 줄을 부르려면, 그 한 줄에 어떤 입력이 들어가는지 미리 알고 있어야 합니다. `Dataset` 객체, `padding`/`truncation` 결정, `DataLoader` 변환이 그 한 줄을 떠받치는 부품들입니다. 이 챕터는 그 입력 형태를 *학습 없이* 미리 손에 익히는 자리입니다.

**Phase 0와의 다리**: Yelp 5,000건은 Ch 1-6에서 줄곧 쓴 데이터. 같은 텍스트가 TF-IDF에서 sparse vector로 갔던 길이, 이번엔 WordPiece에서 `input_ids` + `attention_mask` 텐서 쌍으로 가는 길을 봅니다.

## 토크나이저 노트 — `padding` / `truncation` / `max_length`

WordPiece가 출력하는 시퀀스 길이는 입력 텍스트마다 다릅니다. 그런데 모델은 **고정된 shape의 텐서 배치** 를 받아야 하므로, 이 둘 사이를 맞추는 세 가지 옵션이 있습니다.

| 옵션 | 의미 | 언제 쓰나 |
|---|---|---|
| `padding=False` | 패딩 없음 (기본값). 시퀀스 길이가 다 다름 | 한 문장씩 처리할 때 |
| `padding=True` | **배치 안 가장 긴 시퀀스 길이까지** padding | 일반 학습 (효율적, dynamic padding) |
| `padding="max_length"` | **항상 `max_length` 까지** padding (짧으면 패딩, 길면 자름) | TPU·고정 shape 필요할 때 |
| `truncation=True` | `max_length` 초과분은 *잘라냄* | 항상 같이 두는 게 안전 (긴 입력 방지) |
| `max_length=N` | 길이 상한 (모델별 사전학습 한도 — BERT 512) | 메모리/속도 trade-off |

**패딩이 들어간 자리는 `attention_mask=0`** 으로 표시됩니다. 모델은 이 mask를 보고 self-attention에서 패딩 토큰을 무시하므로, 아무리 길게 패딩을 붙여도 학습 결과는 달라지지 않습니다. 다만 그만큼 속도와 메모리만 낭비될 뿐입니다.

이번 챕터에서 위 세 옵션을 직접 호출해 input_ids와 attention_mask가 어떻게 변하는지 봅니다.

## `datasets` 로 Yelp 로드

`load_dataset("Yelp/yelp_review_full")` 한 줄로 Hugging Face Hub에서 65만 건 학습 데이터를 받아옵니다 (50K test). 처음 받으면 ~150MB 다운로드 + 디스크 캐시.

**주목할 점**: `datasets` 는 Apache Arrow 형식으로 디스크에 저장하고 메모리맵으로 접근합니다. 65만 건이 한꺼번에 RAM에 올라가는 게 아니라, 인덱싱하는 시점에만 디스크에서 필요한 부분을 읽어 옵니다. 그래서 데이터셋이 아무리 커도 RAM 사용량에는 거의 영향이 없습니다.

## 토크나이저 옵션 직접 실험

Ch 7과 같은 `distilbert-base-uncased` 토크나이저로 시작합니다 (사전학습 모델 그대로).

### 옵션 없이 — 한 문장 토큰화 (기본 동작)

```python
tokenizer(text)
```

### 두 문장 배치 + `padding=True` — *동적 패딩*

여러 문장을 한 배치로 묶으려면 길이가 모두 같아야 텐서가 만들어집니다. `padding=True` 옵션을 주면 한 배치 안에서 **가장 긴 문장 길이에 맞춰 짧은 문장만 패딩으로 채우므로**, 배치마다 필요한 만큼만 늘어나 가장 효율적입니다.

### `padding="max_length"`, `max_length=128` — *고정 길이*

배치마다 길이가 달라지는 게 싫을 때 (TPU·정적 그래프 환경) 항상 `max_length` 까지 padding합니다.

### `truncation=True` — 긴 입력 자르기

BERT 계열은 사전학습 시 `max_length=512` 로 학습돼서 그보다 긴 입력은 처리할 수 없습니다. `truncation=True` 로 자동 절단.

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

**해석**: `max_length=128` 이면 절반가량(약 51%)이 잘리지만(평균 177·중앙값 131 토큰) 잘리는 건 대부분 리뷰 뒷부분이라 별점 예측 영향은 작아 표준값으로 씁니다. `max_length=512` 면 3.9%만 잘리지만 평균 패딩 60-70%로 메모리·시간 낭비.

이 커리큘럼은 **`max_length=128`** 을 표준으로 씁니다 (T4 30분 제약 + 무난한 정보 보존 균형).

## `datasets.map` — 5,000건 일괄 토큰화

샘플 하나씩 `tokenizer(...)` 부르는 건 5,000번 호출이 필요하고 느립니다. `dataset.map(fn, batched=True)` 로 *배치 단위* 일괄 호출이 표준.

### `dataset.filter` — 조건에 맞는 샘플만 선별

### `with_format("torch")` — 텐서 형식으로

## `DataLoader` 변환 — Ch 9 학습 입력 미리보기

PyTorch `DataLoader` 는 dataset을 받아 *배치 + shuffle* 을 자동 처리합니다. Ch 9의 `Trainer` 가 내부에서 이걸 만들어 쓰지만, 직접 만들 줄 알면 디버깅에 유리.

## `DataCollator` — 동적 padding을 배치 시점에

위 `DataLoader` 코드는 `padding="max_length"` 로 *모든* 샘플을 128로 미리 padding한 상태였습니다. 짧은 문장은 대부분 패딩 자리라 메모리 낭비가 큽니다.

**더 좋은 방법: 토큰화 시엔 padding 안 하고**, `DataLoader` 가 *배치를 만들 때마다 그 배치 안 longest까지만* padding (동적 padding). 이걸 담당하는 게 `DataCollator`.

`DataCollator` 는 `DataLoader` 의 `collate_fn` 자리에 들어가는 함수입니다. 매 배치마다 N개 샘플을 받아 batch-level 변환을 적용한 뒤 텐서로 묶어주는 역할을 하고, Hugging Face는 task별로 여러 종류를 제공합니다.

**정적 vs 동적 padding 비교**

| 방식 | 토큰화 시점 | 배치 shape | 실제 토큰 비율 | 메모리/속도 |
|---|---|---|---|---|
| **정적** (`padding="max_length"=128`) | 모든 샘플을 128로 미리 padding | 항상 (B, 128) | 보통 30-50% | 일정, 낭비 큼 |
| **동적** (`DataCollatorWithPadding`) | padding 없이 truncation만 | (B, longest in batch) — 매번 다름 | 보통 70-95% | 변동, 효율 |

위 출력에서 채움률(실제 토큰/전체)이 정적 방식보다 훨씬 높을 겁니다. 같은 학습이라도 동적 padding이 *2배 가까이 빠른 경우* 도 흔합니다.

### 다양한 `DataCollator` 종류

| Collator | 용도 | 자주 쓰이는 곳 |
|---|---|---|
| `DataCollatorWithPadding` | 분류·회귀 — `input_ids`/`attention_mask` 동적 padding | **Ch 9-13 모든 분류·회귀 학습 (기본)** |
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

## 이 장의 구성

[[SubPages]]
