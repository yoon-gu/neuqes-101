## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | Ch 24 와 공유? |
|---|---|---|
| `AutoModelForCausalLM.from_pretrained("gpt2")` | OpenAI gpt2 (124M, WebText 사전학습) 본체 로드 | **새로 등장** (Ch 24 는 `GPT2LMHeadModel(config)` random init) |
| `AutoTokenizer.from_pretrained("gpt2")` | gpt2 BPE 토크나이저 (vocab 50,257) 로드 | **새로 등장** (Ch 24 는 직접 학습 BPE) |
| `tokenizer.pad_token = tokenizer.eos_token` | gpt2 의 pad 컨벤션 (EOS 재활용) | **새로 등장** (Ch 24 는 PreTrainedTokenizerFast 인자로 직접 지정) |
| `transformers.Trainer` | HuggingFace 표준 학습 루프 | **공유** (Ch 24 와 동일 클래스, 동일 인자 구조) |
| `DataCollatorForLanguageModeling(mlm=False)` | CausalLM collator (`labels = input_ids.clone()` 자동) | **공유** (Ch 24 와 정확히 같음) |
| `group_texts` 패턴 (HF run_clm.py 표준) | 가변 길이 텍스트 → 고정 길이 블록 스트림 | **공유** |
| `model.generate(do_sample=True, ...)` | sampling-based text generation | **공유** |
| `warmup_steps` 1 미만 비율 해석 | 전체 step 대비 비율 기반 warmup — 구 `warmup_ratio` (continual pretraining 표준) | **약간 다름** (Ch 24 는 `warmup_steps=100` 절대값) |
| `num_train_epochs` (vs `max_steps`) | epoch 수 기반 학습 (continual pretraining 1 epoch 충분) | **약간 다름** (Ch 24 는 `max_steps=1500`) |
| `gradient_accumulation_steps` | 작은 배치를 누적해 큰 effective batch (T4 + 124M 메모리 제약) | **새로 등장** (Ch 24 는 3.7M 이라 불필요)

## 체크포인트 질문

1. Ch 24 의 학습 첫 step loss 는 약 `ln(2048) ≈ 7.62` 에서 시작했습니다. Ch 25 는 random baseline 이 `ln(50257) ≈ 10.82` 인데 *시작 loss 가 약 3.0-4.0* 입니다. 두 챕터의 *시작 loss 차이* 가 의미하는 바는 무엇인가요? (랜덤 추측 / 사전학습된 본체 / vocab 차원의 관계)
2. 본 챕터의 lr (`2e-5`) 가 Ch 24 의 lr (`3e-4`) 보다 *약 15배 작은* 이유를 *catastrophic forgetting* 키워드로 설명해 보세요. 만약 Ch 25 에서도 `3e-4` 를 썼다면 무슨 일이 일어날까요?
3. *Continual pretraining* (단계 2, 본 챕터) 과 *SFT* (단계 3, Ch 28) 의 가장 큰 차이는 *`labels = -100` 자리* 입니다. 본 챕터의 collator 가 만드는 `labels` 패턴과, Ch 28 에서 등장할 `labels[:prompt_len] = -100` 한 줄의 차이를 직접 비교해 설명해 보세요.
4. Ch 25 AFTER 의 generation 이 Ch 24 보다 좋아 보인다면, *모델 크기 (3.7M → 124M)* 의 효과인가 *사전학습 데이터 (없음 → WebText 40GB)* 의 효과인가? 본 챕터의 실험 셋업으로는 둘을 분리할 수 있나요? (분리하려면 어떤 추가 실험이 필요할까요?)

## FAQ

### Q1. (이론) *Continual pretraining* 이 정확히 무엇인가요? *fine-tune* 과 어떻게 다른가요?

**Continual pretraining** (계속 사전학습 / continual learning) 은 GPT 시대 학습 4단계 중 *단계 2*:

| 단계 | 정확 용어 | 의미 |
|---|---|---|
| 1 | **Pretraining** | random init 본체 + 일반 코퍼스 |
| 2 | **Continual pretraining** ← *본 챕터* | *사전학습된 본체* + *새 도메인 데이터*. **head 그대로, task 그대로, loss 그대로** |
| 3 | **SFT** (Instruction tuning) | instruction-response 쌍, `labels[:prompt_len] = -100` |
| 4 | **Alignment** (DPO / RLHF / GRPO) | preference 또는 verifier reward |

**`fine-tune` 이라는 단어는 *세 의미가 섞여 쓰입니다***:

- **BERT 시대 fine-tune**: *task adaptation* — 본체 + 새 head (`Linear(H, K)`) + 새 task loss. Ch 9-23 의 분류 챕터들.
- **GPT 시대 continual pretraining**: *데이터 적응* — 본체 + 같은 head + 같은 loss + 새 데이터. **본 챕터.**
- **GPT 시대 SFT**: *행동 정렬* — 본체 + 같은 head + 같은 loss + instruction-response 데이터 + `-100` 마스킹. Ch 28.

정확히 부르자면 본 챕터는 *fine-tune 이 아니라 continual pretraining*. *task 가 안 바뀌고 데이터만 바뀐* 게 핵심.

### Q2. (실무) 왜 lr 가 Ch 24 의 `3e-4` 보다 작은 `2e-5` 인가요?

**catastrophic forgetting 방지** 가 핵심 이유. *사전학습된 본체* 는 weight 들이 *이미 의미 있는 표상* 을 학습한 상태. 큰 lr 로 학습하면:

1. *사전학습된 표상이 새 데이터에 맞춰 크게 흔들림*
2. *원래 알던 일반 도메인 지식 (WebText 풍 영어 전반)* 이 *TinyStories 동화 풍* 으로 *덮어쓰기*
3. 결과: *TinyStories 도메인은 잘 하지만 일반 영어 능력 손실* — 이게 catastrophic forgetting

작은 lr (`2e-5`) 는 *사전학습된 표상 보존* 을 우선합니다. *기존 weight 에서 살짝만 떨어진 지점으로 이동* — 도메인 적응은 하되 일반 능력은 유지.

```python
# Ch 24 (scratch) - 큰 lr 로 표상 빨리 학습
TrainingArguments(learning_rate=3e-4, ...)

# Ch 25 (continual pretraining) - 작은 lr 로 표상 보존
TrainingArguments(learning_rate=2e-5, ...)
```

HF 의 continual pretraining / fine-tuning 표준 lr 범위: `1e-5` - `5e-5`. SFT (Ch 28) 도 비슷한 범위.

### Q3. (이론) Ch 24 (3.7M) 가 같은 데이터로 학습했는데 Ch 25 (124M) 결과가 훨씬 좋다면, *모델 크기의 위력* 인가 *사전학습의 위력* 인가?

**둘이 *섞여서* 분리 불가능** 입니다. 본 챕터의 셋업은 *두 변수가 동시에 변함*:

- Ch 24 → Ch 25 변화: 모델 크기 *3.7M → 124M (약 33배)* + 사전학습 *없음 → WebText 약 40GB*

진짜 *모델 크기와 사전학습 효과를 분리* 하려면 *2 × 2 격자* 실험이 필요:

| 셋업 | 모델 크기 | 사전학습 | 본 커리큘럼 |
|---|---|---|---|
| (a) | 3.7M | 없음 | **Ch 24** |
| (b) | 3.7M | WebText 풍 사전학습 | (미실험) |
| (c) | 124M | 없음 | (미실험 — 124M scratch + TinyStories 만 학습) |
| (d) | 124M | WebText 약 40GB | **Ch 25** |

본 커리큘럼에는 (a) 와 (d) 만 있어 *둘의 차이* 만 보입니다. (b) 와 (c) 는 *T4 + 30분 룰* 안에 어렵습니다 (124M scratch 는 *TinyStories 만으로 의미 있는 학습이 부족함*, 3.7M WebText 사전학습은 *데이터 규모 자체가 30분에 안 맞음*).

**실용적 결론**: 실무에서는 (b)(c) 가 *비용 대비 비효율* 이라 (d) 패턴이 표준. *대규모 사전학습 모델을 가져와 작은 도메인 데이터로 continual pretraining* — 본 챕터의 패턴이 그 자체로 *실무 표준 레시피*.

### Q4. (실무) gpt2 토크나이저로 *한국어* TinyStories-Korean (Ch 26 도메인) 을 학습하면 어떻게 되나요?

**거의 작동하지 않습니다** — gpt2 BPE 는 *WebText (영어 중심)* 위에 학습되어 *한국어 토큰* 이 vocab 에 거의 없습니다. 한국어 텍스트는 *byte 단위로 잘게 쪼개져* 표현됩니다.

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
sample_korean = "옛날 옛적에 작은 토끼가 살았어요."
enc = tokenizer(sample_korean)
# 같은 의미의 영어보다 보통 5-10배 많은 토큰으로 쪼개집니다
print(f"Korean token count : {len(enc['input_ids'])}")
print(f"tokens (first 10)  : {tokenizer.convert_ids_to_tokens(enc['input_ids'])[:10]}")
# byte-level 조각들이라 vocab 전체가 사실상 한국어를 표현하지 못합니다
```

결과:
- *토큰 수 폭증* — 같은 의미가 약 5-10배 많은 토큰으로 표현 → context window 낭비
- *임베딩이 한국어 단어 단위 의미를 못 가짐* — byte 단위 임베딩만 있어 학습 효율 매우 낮음
- *gpt2 의 사전학습 표상이 한국어로 transfer 안 됨* — WebText 에 한국어가 거의 없어 일반화 약함

그래서 **Ch 26 (한국어 GPT scratch)** 는 *한국어 데이터로 BPE 를 처음부터 학습 + 한국어 GPT 본체도 처음부터 학습* 패턴을 택합니다. *토크나이저 + 본체가 운명공동체* 이므로 *한국어로 처음부터* 가 정공법.

### Q5. (이론·실무) *catastrophic forgetting* 이 무엇인가요? Ch 25 에서 실제로 일어나나요?

**Catastrophic forgetting** (재앙적 망각): 새 데이터로 학습할 때 *이전에 학습한 표상이 덮어쓰기* 되어 *원래 알던 능력이 손실* 되는 현상.

Ch 25 에서는 *짧은 (1 epoch) continual pretraining + 작은 lr (`2e-5`)* 라 *catastrophic forgetting 이 강하지 않음*. 다만 *변형 1 (epoch 늘리기)* 또는 *큰 lr (`1e-3` 등)* 을 시도하면 다음 신호가 보입니다:

- 비-동화 prompt (예: `"Albert Einstein was a"`) 에 대해 *gpt2 가 원래 답했을 만한 일반 영어* 가 *동화풍 톤* 으로 끌려감
- *generation 다양성 하락* — 모든 prompt 에 *little / mommy / friend* 류 동화 단어가 자주 등장
- *evaluation* — 만약 GLUE 같은 일반 영어 벤치마크에 *학습 전 / 후* 를 측정하면 *학습 후가 더 낮은 점수*

방지법:
1. **짧은 학습 + 작은 lr** (본 챕터 패턴)
2. **regularization** — replay (이전 데이터 일부 섞어 학습) / EWC (Elastic Weight Consolidation) 등
3. **adapter / LoRA** — 본체 weight 는 freeze 하고 작은 adapter 만 학습 → 본체 표상 보존

본 챕터는 *방법 1* 만 적용. *방법 3 (LoRA)* 는 본 커리큘럼 범위 밖이지만 *실무에서는 표준 옵션*.

### Q6. (실무) *Trainer 가 Ch 24 와 같다* 는데 진짜 같나요?

**클래스도, 인자 구조도 같습니다** — 인스턴스화하는 `transformers.Trainer` 의 *클래스 자체가 동일* 합니다. Ch 24 / Ch 25 의 Trainer 인자만 나란히 비교해 보세요:

```python
# Ch 24
trainer = Trainer(
    model=model,                              # GPT2LMHeadModel (3.7M, random init)
    args=args,                                # TrainingArguments(lr=3e-4, max_steps=1500, ...)
    train_dataset=lm_train,                   # TinyStories
    eval_dataset=lm_val,
    data_collator=collator,                   # DataCollatorForLanguageModeling(mlm=False)
    callbacks=[vram_cb],
)

# Ch 25
trainer = Trainer(
    model=model,                              # AutoModelForCausalLM (124M, gpt2 pretrained)
    args=args,                                # TrainingArguments(lr=2e-5, num_train_epochs=1, ...)
    train_dataset=lm_train,                   # TinyStories (동일)
    eval_dataset=lm_val,
    data_collator=collator,                   # DataCollatorForLanguageModeling(mlm=False)  <- 동일
    callbacks=[vram_cb],
)
```

다른 곳: *model 인자에 넘기는 인스턴스* 와 *args 의 lr / step 설정* 두 곳. 나머지는 *글자 그대로 동일*.

> 그게 *학습 단계 2 (continual pretraining)* 의 미적 본질 — *trainer / collator / loss 코드 재사용*. 단계 3 (SFT, Ch 28) 에서도 *대부분 같습니다*, 다만 *collator 가 `labels[:prompt_len] = -100` 마스킹* 한다는 점이 추가될 뿐.

### Q7. (이론) 다음 챕터 (Ch 26 한국어 GPT scratch) 와의 관계는?

Ch 26 는 *Ch 24 의 한국어판* — *작은 GPT + 한국어 TinyStories-Korean + BPE 직접 학습* 패턴. *Ch 25 의 한국어판 (한국어 사전학습 GPT + continual pretraining)* 이 *아닌* 이유:

- **한국어 사전학습 GPT 가 부족** — `skt/kogpt2-base-v2` (125M) 등이 있지만 *영어 gpt2 만큼 표준화된 토크나이저·본체 조합* 이 아님. KoGPT2 는 Ch 27 (continual pretraining)·Ch 28 (SFT) 에서 등장
- **한국어 토크나이저 새로 학습이 필요** — Q4 에서 봤듯 영어 BPE 는 한국어를 못 다룸. *한국어 BBPE 를 직접 학습* 하는 게 정공법
- **Phase 4 의 한국어 사전학습 단계 1 챕터** — Ch 22 (한국어 BERT scratch) 의 GPT 판

Ch 24 → Ch 26 흐름:

| Ch | 언어 | 본체 | 토크나이저 | 단계 |
|---|---|---|---|---|
| 22 | 한국어 | 작은 BERT scratch | klue/bert-base 가져옴 | BERT 시대 사전학습 |
| 24 | 영어 | 작은 GPT scratch | BPE 직접 학습 | GPT 시대 단계 1 |
| **25 ← 본 챕터** | 영어 | gpt2 (124M, WebText) | gpt2 BPE 그대로 | **GPT 시대 단계 2** |
| 26 | 한국어 | 작은 GPT scratch | BPE 직접 학습 (한국어) | GPT 시대 단계 1 (한국어판) |

> Ch 25 ↔ Ch 26 사이엔 *축이 두 개 동시에 바뀝니다* (언어 + 학습 단계). 본 챕터 한정으로는 *Ch 24 ↔ Ch 25 가 한 축 격리* 임을 기억해 두시면 됩니다.

## 다음 챕터 예고

**Chapter 26. 한국어 작은 GPT scratch — *Ch 24 의 한국어판***

- `roneneldan/TinyStories` 의 한국어 짝 (TinyStories-Korean 또는 유사 한국어 동화 데이터셋) 으로 *한국어 GPT scratch* 학습
- *한국어 BPE 직접 학습* — gpt2 BPE 가 한국어를 못 다루는 이유 (Q4 참고) 를 출발점으로, 한국어 코퍼스 위에 새 BPE 학습
- *작은 GPT2LMHeadModel(config)* random init — Ch 24 와 *같은 패턴, 데이터 + 토크나이저만 한국어*
- 비교: Ch 22 (한국어 BERT scratch) 와 같은 한국어 사전학습 단계 1 이지만 *encoder MLM → decoder CausalLM*

**Phase 4 GPT 시대 4단계 흐름 정리**:

| 챕터 | 단계 | 본체 | 데이터 | 핵심 |
|---|---|---|---|---|
| Ch 24 | 1 (영어) | 작은 GPT scratch | TinyStories | 단계 1 출발 |
| **Ch 25 ← 여기** | **2** | **`gpt2` 124M** | **TinyStories (동일)** | **단계 2: continual pretraining** |
| Ch 26 | 1 (한국어) | 작은 GPT scratch | TinyStories-Korean | 한국어 단계 1 |
| Ch 27 | 2 (한국어) | KoGPT2 125M | TinyStories-Korean | **한국어 단계 2: continual pretraining** (본 챕터의 한국어 짝) |
| Ch 28 | 3 | KoGPT2 + SFT | KoAlpaca 등 instruction 데이터 | **단계 3: SFT** |
| Ch 30 | 4 | SFT 모델 + DPO | preference 쌍 데이터 | **단계 4: DPO** |
| Ch 31 | 4 | SFT 모델 + GRPO | verifier reward | **단계 4: GRPO** |

> *왜 영어 사전학습 모델 (gpt2) 을 한국어에 그대로 적용하기 어려운가* 의 답이 Ch 26 의 동기입니다 — *토크나이저가 한국어를 못 다루면 본체 weight 가 유효한 신호가 아니라* 사실상 random init 과 같은 상태. 한국어는 *처음부터* 가 정공법.

> **변하는 축** (Ch 25 → Ch 26): *언어 + 학습 단계* 두 축. *직접 짝* 은 Ch 24 ↔ Ch 26 (같은 단계 1, 언어만 다름) / **Ch 25 ↔ Ch 27** (같은 단계 2 continual pretraining, KoGPT2 + 한국어 TinyStories) 입니다. 영어·한국어가 *scratch + continual pretraining* 으로 완전 대칭.
