## 이번 챕터에 등장한 라이브러리·함수 (Ch 24 와의 차이만)

| 이름 | 한 줄 설명 | Ch 24 와 차이 |
|---|---|---|
| `load_dataset("g0ster/TinyStories-Korean", streaming=True)` | 한국어 TinyStories 줄 단위 스트리밍 로드 | 영어 → 한국어, story 복원 로직 추가 |
| `tokenizers.BPE + ByteLevel` (한국어 코퍼스) | byte-level BPE (BBPE) 직접 학습 | 학습 코퍼스 영어 → 한국어, vocab 2,048 → 약 4,000 |
| `AutoTokenizer.from_pretrained("gpt2")` (비교용) | 영어 BPE 로 한국어 토큰화 비교 | 신규 (cross-language 실측) |
| `transformers.GPT2Config / GPT2LMHeadModel(config)` (동일) | 작은 GPT random init | (Ch 24 동일, vocab 만 다름) |
| `DataCollatorForLanguageModeling(mlm=False)` (동일) | `labels = input_ids.clone()` 자동 | (Ch 24 동일) |
| `group_texts` 패턴 (동일) | 가변 길이 → 고정 length 블록 스트림 | (Ch 24 동일) |
| `model.generate(do_sample=True, ...)` (동일) | sampling generation | (Ch 24 동일) |
| `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` (선택) | KoGPT2 reference | gpt2 → KoGPT2 |

## 체크포인트 질문

1. 영어 gpt2 BPE 로 한국어를 토큰화하면 토큰 수가 폭증합니다 (셀 비교 표). *byte-level* 이라 UNK 는 없는데, 왜 그래도 *직접 학습한 BBPE* 가 한국어 학습에 유리할까요? (의미 단위 보존 vs byte 조각 분해 관점)
2. random baseline 이 Ch 24 (vocab 2,048) 의 `ln(2048) ≈ 7.62` 에서 Ch 26 (vocab 약 4,000) 의 `ln(4000) ≈ 8.29` 로 바뀝니다. 이 차이가 *학습 동역학* 에 의미 있는 영향을 주나요? *학습 종료 loss 의 절대값* 을 영어 챕터와 비교할 때는요?
3. 본 챕터의 collator 가 만드는 `labels` 는 *거의 모든 자리* 가 학습 신호입니다. Ch 28 (한국어 SFT) 에서는 `labels[:prompt_len] = -100` 한 줄로 *prompt 부분만 제외* 합니다. *왜 이 한 줄이 모델이 한국어 instruction 을 따라가게 만드는지* 설명해 보세요.
4. 같은 본체 구조·loss·trainer 로 영어 (Ch 24) 와 한국어 (Ch 26) 를 학습했습니다. 학습 곡선·generation 품질이 두 언어에서 비슷하게 나온다면, *언어 자체의 난도 차이* 와 *데이터 규모·번역체 차이* 중 무엇이 generation 품질에 더 큰 영향을 줄까요?

## FAQ

### Q1. (이론) 한국어 BBPE 는 영어 BPE 와 무엇이 다른가요? byte-level 이 한글을 어떻게 처리하나요?

**알고리즘은 *완전히 같습니다* — 다른 건 *학습 코퍼스* 뿐**. 둘 다 byte-level BPE (BBPE):

- *가장 작은 단위가 byte (256개)* 라 한글·이모지·한자 *어떤 유니코드도 UNK 없이* 표현.
- 한글 `가` 는 UTF-8 로 3 byte (`0xEA 0xB0 0x80`). BBPE 는 학습 중 *자주 함께 등장하는 byte 쌍* 을 반복 병합해 *글자·어절 단위* 토큰을 만듦.

차이는 *어떤 코퍼스로 학습했는가*. 영어 코퍼스로 학습한 gpt2 BPE 는 *영어 byte 패턴* 의 병합 규칙을 갖고 있어, 한국어를 넣으면 한글 byte 가 거의 *원자 단위 그대로* 나와 토큰 수가 폭증합니다.

```python
# 같은 문장, 두 토크나이저
sent = "옛날 옛날에 작은 토끼가 살았어요."
len(gpt2_tok(sent, add_special_tokens=False)["input_ids"])    # 많음 (byte 조각)
len(tokenizer(sent, add_special_tokens=False)["input_ids"])   # 적음 (의미 단위)
```

한국어 코퍼스 위에 직접 학습하면 *한국어 어절의 byte 패턴* 이 병합 규칙에 반영되어 *훨씬 적은, 의미 있는 토큰* 으로 압축됩니다.

### Q2. (이론) 왜 한국어도 *scratch* 부터 학습하나요? Ch 24 (영어 scratch) 를 이미 했는데.

**같은 시연 가치 + 한국어 특유의 필요성** 때문입니다.

1. *시연 가치* — 사전학습이 본체에 *next-token 분포* 를 새기는 과정을 *한국어로* 직접 봅니다 (Ch 20→22 가 BERT 에서 한 것의 GPT 판).
2. *한국어 특유의 필요성* — 영어는 gpt2 BPE 를 그대로 가져다 continual pretraining (Ch 25) 이 가능했지만, *영어 BPE 는 한국어를 byte 조각으로 쪼개* 사실상 못 씁니다. 그래서 한국어는 *토크나이저부터 새로* — 자연스럽게 *scratch* 가 됩니다.

```python
# 한국어는 토크나이저 + 본체 모두 처음부터
bbpe = Tokenizer(BPE(unk_token=None))      # 토크나이저 직접 학습
model = GPT2LMHeadModel(config)            # 본체 random init
```

*토크나이저는 본체와 운명공동체* — 토크나이저가 한국어를 못 다루면 본체 weight 도 유효한 신호를 받지 못합니다.

### Q3. (실무) 다음 챕터 (Ch 27 KoGPT2 continual pretraining) 와의 관계는?

Ch 27 = *대규모 한국어 사전학습 모델 KoGPT2 (`skt/kogpt2-base-v2`, 125M) 를 같은 한국어 TinyStories 로* **continual pretraining**. 본 챕터의 *작은 from-scratch 모델* 과 *완전 반대 출발점* — Ch 24→25 (영어) 의 한국어 짝:

| 축 | Ch 26 (본 챕터) | Ch 27 (다음) |
|---|---|---|
| 모델 크기 | 약 4.2M params | **약 125M (약 30배)** |
| 사전학습 | from scratch (random init) | **대규모 한국어 코퍼스 사전학습** |
| 토크나이저 | 직접 학습 BBPE vocab 약 4,000 | **KoGPT2 BBPE (그대로)** |
| 한국어 TinyStories 학습 | 사전학습 그 자체 (1500 steps) | **continual pretraining** (수백 steps) |
| Generation 품질 | 동화 풍 단순 한국어 | **자연스러운 동화 + 일반 도메인 폭** |

**핵심 메시지**: *대규모 한국어 사전학습 본체* + *작은 도메인 continual pretraining* 이 *작은 from-scratch 모델* 보다 *빠르게, 좋게* 도달합니다. *왜 실무는 from-scratch 가 아니라 사전학습 모델을 가져와 계속 학습하는가* 의 한국어 답.

### Q4. (이론) 한국어 CausalLM 사전학습도 `labels = -100` 트릭을 쓰나요?

**거의 안 씁니다 — pad 자리만**. Ch 24 (영어) 와 동일하게, 본 챕터 collator 출력은 *거의 모든 자리* 가 학습 신호 (`labels = input_ids.clone()`). `group_texts` 로 chunk 길이가 모두 같으면 pad 도 없어 `-100` 이 0개일 수 있습니다.

같은 트릭이 **Ch 28 (한국어 SFT)** 에서 *결정적 한 줄* 로 부활합니다:

```python
# Ch 28 의 SFT 데이터 - "instruction + response" 형식
prompt = "### 질문: 한국의 수도는?\n### 답변: "
response = "서울입니다."

input_ids = tokenizer(prompt + response)["input_ids"]
labels = input_ids.copy()
prompt_len = len(tokenizer(prompt)["input_ids"])
labels[:prompt_len] = [-100] * prompt_len   # <- 이 한 줄이 SFT 의 핵심
```

이 한 줄로 모델은 *prompt 를 외우지 않고 response 만 학습* → *주어진 instruction 에 response 를 생성* 하게 됩니다. 본 챕터의 collator 출력 (거의 모든 자리 = 학습 신호) 을 손에 익혀 두면 Ch 28 의 그 한 줄이 단번에 이해됩니다.

### Q5. (실무) 한국어 generation 품질이 영어 (Ch 24) 보다 낮아 보이면 무엇을 의심하나요?

세 가지 원인을 순서대로 점검합니다.

1. **데이터 양·품질** — `g0ster/TinyStories-Korean` 은 *기계 번역본* 이라 원문 영어 TinyStories 보다 문장이 다소 어색하거나 일관성이 떨어질 수 있습니다. story 복원 (`<|endoftext|>` 기준) 이 제대로 됐는지, story 수가 충분한지 확인.

```python
# story 복원 검증 - 한 story 가 통째로 나오는지
print(raw_train[0]["text"][:300])   # 문장이 자연스럽게 이어지면 복원 정상
print(f"n_stories: {len(raw_train):,}")
```

2. **토크나이저** — vocab 약 4,000 이 너무 작으면 한 어절이 여러 byte 조각으로 쪼개져 학습이 어렵습니다. `VOCAB_SIZE` 를 6,000-8,000 으로 키워 비교.

3. **학습량** — `max_steps` 또는 `N_TRAIN` 을 키우면 한국어 문장 자연스러움이 올라갑니다 (T4 30분 룰 주의).

> *번역체 데이터* 라 영어판보다 다소 어색한 건 *정상* — 본 챕터의 목표는 *완벽한 한국어 생성* 이 아니라 *학습 전·후의 질적 도약* 을 한국어로 확인하는 것. 자연스러운 한국어 generation 은 Ch 27 (KoGPT2 continual pretraining) 의 영역.

### Q6. (이론) Ch 24 (영어) 와 Ch 26 (한국어) 의 학습 곡선을 비교하려면?

같은 hyperparam·같은 BLOCK_SIZE·같은 step 수로 학습된 두 모델의 *상대* 비교가 의미 있습니다.

```python
metrics = {
    "language":          ["EN (Ch 24)",  "KO (Ch 26)"],
    "vocab_size":        [2048,          4000],
    "random_baseline":   [7.62,          8.29],
    "final_train_loss":  ["measure",     "measure"],
}
```

vocab 크기가 다르므로 *random baseline (`ln V`) 이 다릅니다*. 절대 loss 를 직접 비교하기보다 *random baseline 대비 얼마나 떨어졌는가* (상대 하락폭) 로 비교해야 공정합니다. 또 *번역체 한국어* 는 원문 영어보다 *반복·패턴* 이 적을 수 있어 같은 step 에서 loss 가 약간 높게 나올 수 있습니다 — *언어 난도* 보다 *데이터 특성* 차이가 더 큽니다.

### Q7. (실무) 학습 첫 step loss 가 `ln(4000) ≈ 8.29` 가 아니라 *5.0* 이라면? *15.0* 이라면?

- **5.0 (너무 낮음)**: vocab 크기 가정이 틀렸거나 (실제 vocab 이 더 작음), 토크나이저가 prompt 를 *비정상적으로 적은 토큰* 으로 만들거나, 데이터가 *극도로 반복적* 이어서 첫 배치가 쉬운 경우. `tokenizer.vocab_size` 와 `math.log(tokenizer.vocab_size)` 를 출력해 baseline 을 확인.

```python
print(f"vocab: {tokenizer.vocab_size}, ln V = {math.log(tokenizer.vocab_size):.2f}")
```

- **15.0 (너무 높음)**: `ln(4000) ≈ 8.29` 보다 훨씬 높다면 *vocab 불일치* (모델 config 의 `vocab_size` 와 토크나이저가 다름) 또는 *입력 id 가 vocab 범위를 벗어남* 을 의심. `config.vocab_size == tokenizer.vocab_size` 인지 점검하세요. random init 의 자연스러운 시작은 *baseline 근처* 입니다.

## 다음 챕터 예고

**Chapter 27. KoGPT2 Continual Pretraining 으로 한국어 TinyStories 에 적응 — *Ch 25 의 한국어판***

- `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` - 대규모 한국어 코퍼스로 사전학습된 125M params KoGPT2 로드
- **같은 한국어 TinyStories** 데이터 (본 챕터와 동일) 로 **continual pretraining** (계속 사전학습 — *같은 CausalLM task, 새 데이터, head 그대로*. *task adaptation 의미의 fine-tune 이 아니라 단계 2*)
- **핵심 비교**: 본 챕터 (약 4.2M, from scratch) vs Ch 27 (125M, continual pretraining) 의 한국어 generation 품질·학습 곡선 격차
- *trainer 자체는 Ch 26 과 동일* (`transformers.Trainer` + `DataCollatorForLanguageModeling(mlm=False)`) - *변하는 건 모델 로드 한 줄 + lr (scratch 5e-4 → continual pretraining 2e-5)*
- Ch 24→Ch 25 (영어) 의 한국어 짝 - *왜 실무가 from-scratch 가 아니라 대규모 사전학습 모델 위에 계속 학습하는가* 의 한국어 정량 답변

> **변하는 축**: *모델 크기 + 사전학습 여부* (3M / scratch → 125M / pretrained). 데이터·토크나이저 규약·loss·trainer 는 같음. Phase 4 의 *학습 단계 2 (continual pretraining)* 가 한국어에서 자리 잡는 챕터. *진짜 행동 정렬 (SFT)* 은 Ch 28 에서 본격 등장합니다.
