## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.GPT2Config` | GPT-2 구조 hyperparam (n_layer, n_head, n_embd, n_positions, ...) | Ch 25 - `gpt2` 사전학습 config 그대로 로드 |
| `transformers.GPT2LMHeadModel` | decoder + LM head 내장, causal attention 자동 처리 | Ch 25 - `AutoModelForCausalLM.from_pretrained("gpt2")` |
| `transformers.GPT2LMHeadModel(config)` (random init) | from scratch 사전학습 모델 생성 | Ch 26 - 한국어 TinyStories scratch |
| `tokenizers.BPE + ByteLevel` | byte-level BPE 토크나이저 직접 학습 | Ch 26 - 한국어 BBPE |
| `DataCollatorForLanguageModeling(mlm=False)` | CausalLM collator - `labels = input_ids.clone()` 자동 | Ch 25-26 동일 / Ch 28 SFT 는 *`-100` 자리만 다름* |
| `group_texts` 패턴 (HF run_clm.py 표준) | 가변 길이 텍스트 → 고정 length 토큰 블록 스트림 | Ch 25-26 동일 |
| `model.generate(do_sample=True, ...)` | sampling-based text generation (temperature / top_k / top_p) | Ch 25-30 전반에서 활용 |
| `<\|endoftext\|>` 단일 special token | GPT-2 컨벤션 (bos = eos = pad 겸용) | Ch 25 - 같은 컨벤션 |

## 체크포인트 질문

1. GPT CausalLM 사전학습은 *거의 모든 자리* 가 학습 신호인데, BERT MLM 은 *15% 자리* 만 학습 신호입니다. 왜 BERT 는 *전 자리* 를 학습하지 못할까요? (causal attention vs bidirectional attention 의 정보 누출 관점)
2. `tie_word_embeddings=True` (weight tying) 가 `Linear(H, V)` 의 파라미터를 어떻게 절약하나요? vocab 2,048 / hidden 256 일 때 절약되는 파라미터 수를 직접 계산해 보세요.
3. 학습 첫 step loss 가 `ln(2048) ≈ 7.62` 가 아니라 *5.0* 이라면 무엇을 의심해야 하나요? *15.0* 이라면?
4. Ch 28 (SFT) 에서는 `labels[:prompt_len] = -100` 한 줄로 *prompt 부분만 학습 신호에서 제외* 합니다. 이번 챕터의 collator 출력 (거의 모든 자리가 학습 신호) 과 비교해 *왜 이 한 줄이 모델이 instruction 을 따라가게 만드는지* 설명해 보세요.

## FAQ

### Q1. (이론) 왜 GPT 는 *거의 모든 토큰* 을 학습 신호로 쓰고 BERT 는 15% 만 쓰나요?

**causal attention vs bidirectional attention 의 정보 누출 차이** 때문입니다.

BERT 의 *bidirectional* attention 은 토큰 $i$ 의 hidden 이 좌·우 모든 토큰을 다 봅니다. 만약 *모든 자리* 의 정답 토큰을 *예측 task* 로 두면, 모델은 *자기 자신을 그대로 복사* 하는 trivial 해를 학습합니다 (input 이 hidden 에 그대로 들어 있으니까). 그래서 BERT 는 일부 토큰을 `[MASK]` 로 *가려야만* 의미 있는 학습 신호가 생깁니다 - *주변 문맥* 으로 *가려진 자리* 를 복원.

GPT 의 *causal* attention 은 토큰 $i$ 의 hidden 이 *과거 (j ≤ i)* 만 봅니다. 미래 토큰을 못 보니 *next-token 예측* 이 trivial 하지 않습니다 - 모든 자리에서 *다음 토큰* 을 예측해도 cheating 이 안 됩니다. 그래서 *전 자리* 가 학습 신호.

코드 한 줄로 갈리는 차이:

```python
# BERT MLM
DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
# - 약 15% 자리만 학습 신호 (labels = original token id)
# - 나머지 85% 는 labels = -100 (loss 계산 제외)

# GPT CausalLM
DataCollatorForLanguageModeling(tokenizer, mlm=False)
# - 거의 모든 자리가 학습 신호 (labels = input_ids.clone())
# - pad 토큰 자리만 -100
```

한 step 의 *토큰 학습 효율* 은 GPT 가 약 5-6배 높습니다 (15% vs 거의 100%). 그래서 같은 step 수라도 GPT 가 더 많은 토큰을 학습.

### Q2. (이론) TinyStories 는 *일반 도메인* 인가요 *task corpus* 인가요? 왜 일반 위키 (Wikitext-103) 가 아닌가요?

**TinyStories 는 *합성된 simple 스토리* 라 *generation 시연 가치* 가 우선** 인 데이터입니다. *진정한 일반 도메인 사전학습* 의 의미에서는 Wikitext-103 보다 약하지만, 본 챕터의 목적은 *작은 모델로 generation 이 어떻게 동작하는지를 직접 보는 것* - 일반 위키 (Wikitext-103) 로 같은 셋업을 돌리면 3M 모델이 *문장 구조를 학습하기 전에 학습이 끝남*. TinyStories 의 단순한 어휘·문법 덕분에 *작은 모델로도 grammatical 한 생성이 가능* 합니다.

Ch 25 가 그 *trade-off 의 반대편* 을 다룹니다 - *큰 모델 (gpt2 124M) + 대규모 일반 코퍼스 (WebText)* 의 사전학습된 본체를 TinyStories 로 **continual pretraining**. *작은 + 합성 도메인 from-scratch* vs *큰 + 일반 도메인 사전학습 후 continual pretraining* 의 generation 품질 격차가 핵심 비교.

### Q3. (이론) BPE 토크나이저는 Ch 19 의 WordPiece / WordLevel 과 어떻게 다른가요?

세 방식 모두 *vocab 학습 알고리즘* 이지만, *어떻게 subword 를 만드는가* 가 다릅니다.

- **WordLevel**: 공백 + 빈도 - 단어 통째로 vocab 등록. UNK 가 많음.
- **WordPiece (BERT)**: 빈도 + likelihood 기반 subword 병합. 단어 *중간* subword 에 `##` 접두사 (`unhappiness` → `["un", "##happiness"]`).
- **BPE (GPT-2)**: *byte 쌍 빈도* 기반 반복 병합. 접두사 없이 byte 시퀀스 그대로 (`unhappiness` → `["un", "happiness"]`).

본 챕터는 *byte-level BPE* - 가장 작은 단위가 *byte (256개)* 라 *어떤 유니코드 문자열* (이모지, 한글, 특수 기호) 도 UNK 없이 표현 가능. Ch 19 의 직접 학습 챕터에서 본 세 알고리즘 중 *GPT 계열이 BPE 를 선호* 하는 이유.

```python
# 본 챕터의 BPE 학습 - vocab 2,048
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

bpe = Tokenizer(BPE(unk_token=None))
bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(vocab_size=2048, initial_alphabet=ByteLevel.alphabet())
bpe.train_from_iterator(text_iter, trainer)
```

### Q4. (이론) `tie_word_embeddings=True` (weight tying) 가 정확히 무엇을 공유하나요?

**Input embedding (`wte`) 의 weight 와 LM head (`Linear(H, V)`) 의 weight 가 *같은 텐서를 공유*** 합니다.

```python
# 개념적으로
model.lm_head.weight = model.transformer.wte.weight   # 같은 텐서, 같은 메모리
```

직관: input embedding 은 *vocab token → hidden* 변환, LM head 는 *hidden → vocab logit* 변환. 둘이 *transpose 관계* 라 같은 weight 를 공유해도 의미가 통합니다. 효과:

- **파라미터 절약**: `vocab_size × hidden_size` 만큼 (본 챕터: 2,048 × 256 = 524,288 = 약 0.5M params). 전체 3.7M 모델의 약 14% - 작은 모델에선 비중이 큼.
- **학습 안정**: input 과 output 이 *같은 임베딩 공간* 을 공유 → 일관성 ↑.

GPT-2 의 기본값. 우리 모델도 자동으로 적용됩니다 (`config.tie_word_embeddings=True`).

### Q5. (실무) temperature / top_k / top_p sampling 의 의미는?

`model.generate(do_sample=True, ...)` 의 세 핵심 hyperparam:

- **`temperature`** (T): logits 를 *나눠* softmax → T<1 은 분포를 뾰족하게 (안전), T>1 은 평탄화 (다양). $p_i = \text{softmax}(\text{logits}_i / T)$.
- **`top_k`**: 매 step *상위 k 개* 후보로만 한정. 작으면 안전하지만 반복적.
- **`top_p`** (nucleus): *누적 확률 p* 이내 후보로 한정. 모델이 확신 있을 땐 좁게 (top-1 이 0.9), 애매할 땐 넓게 (top-100 이 0.9) 자동 조정.

```python
# 셋이 같이 쓰일 때의 적용 순서
# 1. logits / T  → softmax
# 2. top_k 로 후보 잘라냄
# 3. top_p 로 후보 더 잘라냄
# 4. 남은 후보에서 multinomial sampling
model.generate(do_sample=True, temperature=0.8, top_k=50, top_p=0.9, max_new_tokens=60)
```

일반적 추천: *대화* 에는 `T=0.7-0.9, top_p=0.9`, *코드 / 수식* 에는 `T=0.2-0.4, top_k=20`, *창의적 생성* 에는 `T=1.0-1.2, top_p=0.95`.

### Q6. (실무) `labels = -100` 트릭이 CausalLM 사전학습에서 거의 안 쓰이는데, 그럼 언제 쓰이나요?

본 챕터의 collator 출력에서 봤듯 CausalLM 사전학습은 *pad 토큰 자리만* `-100` (그것도 `group_texts` 로 chunk 길이가 모두 같으면 pad 도 없음). 거의 안 쓰임.

하지만 같은 트릭이 **Ch 28 (SFT, Instruction Tuning)** 에서 *결정적 한 줄* 로 부활합니다:

```python
# Ch 28 의 SFT 데이터 - "instruction + response" 형식
# 모델이 *response 부분만* 학습하길 원함 (instruction 은 외우면 안 됨)
prompt = "### 질문: 한국의 수도는?\n### 답변: "
response = "서울입니다."

input_ids = tokenizer(prompt + response)["input_ids"]
labels = input_ids.copy()
prompt_len = len(tokenizer(prompt)["input_ids"])
labels[:prompt_len] = [-100] * prompt_len   # <- 이 한 줄이 SFT 의 핵심
```

이 한 줄로 모델은 *prompt 토큰을 외우지 않고 response 만 학습* - 같은 instruction 에 대한 *다양한 response* 가 학습 가능하고, 추론 시에는 *주어진 instruction 에 대해 response 를 생성* 하게 됩니다. *모델이 instruction 을 따라간다* 는 게 이 한 줄의 효과.

본 챕터의 collator 출력 (거의 모든 자리 = 학습 신호) 을 손에 익혀 두면 Ch 28 의 `labels[:prompt_len] = -100` 가 단번에 이해됩니다.

### Q7. (실무) 다음 챕터 (Ch 25 — continual pretraining) 와의 비교는 어떻게 되나요?

Ch 25 = *OpenAI 가 사전학습한 `gpt2` (124M params, WebText 약 40GB) 를 TinyStories 로* **continual pretraining** (같은 CausalLM task, 같은 LM head — *task adaptation 의미의 fine-tune 이 아님*). 본 챕터의 *작은 from-scratch 모델* 과 *완전 반대 출발점*:

| 축 | Ch 24 (본 챕터) | Ch 25 (다음) |
|---|---|---|
| 모델 크기 | 약 3.7M params | **약 124M (약 33배)** |
| 사전학습 | from scratch (random init) | **OpenAI WebText 약 40GB 사전학습** |
| TinyStories 학습 | 사전학습 그 자체 (1500 steps) | **continual pretraining** (1 epoch, 약 3,200 steps) |
| 토크나이저 | 직접 학습 BPE vocab 2,048 | **gpt2 BPE vocab 50,257 (그대로)** |
| Generation 품질 | grammatical 한 동화 풍 영어 | **자연스러운 동화 + 일반 도메인 폭** |
| 학습 시간 | 약 1분 (사전학습) | **약 19분** (124M 본체 continual pretraining) |

**핵심 메시지**: *대규모 일반 사전학습된 본체* + *작은 도메인 continual pretraining* 이 *작은 from-scratch 모델* 보다 *낮은 loss 에서 출발해 더 좋게* 도달합니다 (학습 시간은 124M 본체라 오히려 더 깁니다). *왜 실무는 보통 from-scratch 가 아니라 사전학습 모델을 가져와 새 데이터로 계속 학습하거나 SFT 하는가* 의 답. (단계 3 SFT 는 Ch 28 에서 본격.)

## 다음 챕터 예고

**Chapter 25. GPT2 (124M) Continual Pretraining 으로 TinyStories 에 적응 — *대규모 사전학습 모델의 도메인 계속 학습***

- `AutoModelForCausalLM.from_pretrained("gpt2")` - OpenAI WebText 약 40GB 로 사전학습된 124M params 모델 로드
- **같은 TinyStories 30K** 데이터 (본 챕터와 동일) 로 **continual pretraining** (계속 사전학습 / continual learning — *같은 CausalLM task, 새 데이터, head 그대로*. *task adaptation 의미의 fine-tune 이 아니라 단계 2*)
- **핵심 비교**: 본 챕터 (3.7M, from scratch, 약 1분) vs Ch 25 (124M, continual pretraining, 약 19분) 의 generation 품질·학습 곡선 격차
- *trainer 자체는 Ch 24 와 동일* (`transformers.Trainer` + `DataCollatorForLanguageModeling(mlm=False)`) — *변하는 건 모델 로드 한 줄 + lr (scratch 3e-4 → continual pretraining 2e-5)*
- 작은 데이터 + 큰 사전학습 모델 = *왜 실무가 from-scratch 가 아니라 사전학습 모델 위에 계속 학습 패턴인가* 의 정량 답변
- *진짜 task adaptation 의미의 fine-tune (instruction tuning)* 은 Ch 28 SFT 에서 본격 등장

> **변하는 축**: *모델 크기 + 사전학습 여부* (3.7M / scratch → 124M / pretrained). 데이터·토크나이저 규약·loss·trainer 는 같음. Phase 4 의 *학습 단계 2 (continual pretraining)* 가 본격적으로 자리 잡는 챕터.
