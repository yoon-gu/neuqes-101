## 이번 챕터에 등장한 라이브러리·함수

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
| `transformers.DataCollatorWithPadding` | DataLoader의 `collate_fn` — 배치 시점에 동적 padding |

## 체크포인트 질문

1. `padding=True` 와 `padding="max_length"` 는 각각 어떤 상황에 적합한가요? 메모리·속도 차이는?
2. `attention_mask=0` 인 자리는 self-attention에서 어떻게 무시되나요? (`-inf` 트릭)
3. `datasets` 가 65만 건 데이터를 RAM에 다 안 올리고 인덱싱할 수 있는 이유는 무엇인가요?
4. `dataset.map(fn, batched=True)` 와 `batched=False` 의 속도 차이는 어디에서 오나요?

## FAQ

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

ds = load_dataset("Yelp/yelp_review_full")  # 65만 건

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

대규모 파이프라인에선 `with_format` 으로 *변환마다 새 객체* 가 안전 (디버깅 시 원본 비교).

## 삽질 코너 (선택)

다음 코드를 돌려보고, 두 결과의 길이가 왜 다른지 예측해보세요.

```python
text = "Hugging Face is amazing!"
out1 = tokenizer(text)
out2 = tokenizer(text, add_special_tokens=False)

print(f"기본:                    {len(out1['input_ids'])}, {tokenizer.decode(out1['input_ids'])}")
print(f"add_special_tokens=False: {len(out2['input_ids'])}, {tokenizer.decode(out2['input_ids'])}")
```

힌트: 기본값에선 `[CLS]` 와 `[SEP]` 가 자동으로 *2개* 추가됩니다 — `add_special_tokens=False` 로 끄면 빠집니다. 분류 작업에선 거의 항상 켜둬야 하지만(BERT 사전학습 시 [CLS] 자리에 분류 신호가 모이도록 학습됐으므로), 디버깅·연구용으로 끄는 경우가 있습니다.

## 다음 챕터 예고

**Chapter 9. BERT 회귀 — 첫 파인튜닝, 첫 `Trainer`**

- Phase 0의 Yelp 별점 회귀(Ch 2)를 *DistilBERT 파인튜닝* 으로 다시 풀기 — sklearn `LinearRegression` 1초 학습 vs BERT T4 GPU 5-10분 학습
- **`Trainer` 본격 등장** — `TrainingArguments`, `compute_metrics`, evaluation loop
- 이번 챕터의 데이터 파이프라인(`datasets.map` + `with_format("torch")`)이 그대로 입력으로 들어감
- Loss는 `MSELoss` (Ch 2와 동일) — `problem_type="regression"` 으로 자동 매핑
- **GPU 필수** — fp16 옵션 등장
