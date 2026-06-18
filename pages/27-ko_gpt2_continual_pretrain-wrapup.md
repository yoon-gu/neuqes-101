## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | Ch 26 과 차이 |
|---|---|---|
| `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` | KoGPT2 (125M, 대규모 한국어 사전학습) 본체 로드 | **새로 등장** (Ch 26 은 `GPT2LMHeadModel(config)` random init) |
| `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` | KoGPT2 BBPE 토크나이저 (vocab 51,200) 로드 | **새로 등장** (Ch 26 은 직접 학습 BBPE) |
| `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token` | KoGPT2 의 pad 컨벤션 (없으면 EOS 재활용) | **새로 등장** (Ch 26 은 PreTrainedTokenizerFast 인자로 직접 지정) |
| `transformers.Trainer` | HuggingFace 표준 학습 루프 | **공유** (Ch 26 과 동일 클래스, 동일 인자 구조) |
| `DataCollatorForLanguageModeling(mlm=False)` | CausalLM collator (`labels = input_ids.clone()` 자동) | **공유** (Ch 26 과 정확히 같음) |
| `group_texts` 패턴 (HF run_clm.py 표준) | 가변 길이 텍스트 → 고정 길이 블록 스트림 | **공유** |
| `model.generate(do_sample=True, ...)` | sampling-based text generation | **공유** |
| `warmup_ratio` (vs `warmup_steps`) | epoch 비율 기반 warmup (continual pretraining 표준) | **약간 다름** (Ch 26 은 `warmup_steps=100`) |
| `num_train_epochs` (vs `max_steps`) | epoch 수 기반 학습 (continual pretraining 1 epoch 충분) | **약간 다름** (Ch 26 은 `max_steps=1500`) |
| `gradient_accumulation_steps` | 작은 배치를 누적해 큰 effective batch (T4 + 125M 메모리 제약) | **새로 등장** (Ch 26 은 약 3M 이라 불필요)

## 체크포인트 질문

1. Ch 26 의 학습 첫 step loss 는 약 `ln(4000) ≈ 8.29` 에서 시작했습니다. Ch 27 은 random baseline 이 `ln(51200) ≈ 10.84` 인데 *시작 loss 가 약 3.0-4.0* 입니다. 두 챕터의 *시작 loss 차이* 가 의미하는 바는 무엇인가요? (랜덤 추측 / 사전학습된 본체 / vocab 차원의 관계)
2. 본 챕터의 lr (`2e-5`) 가 Ch 26 의 lr (`5e-4`) 보다 *약 25배 작은* 이유를 *catastrophic forgetting* 키워드로 설명해 보세요. 만약 Ch 27 에서도 `5e-4` 를 썼다면 무슨 일이 일어날까요?
3. *Continual pretraining* (단계 2, 본 챕터) 과 *SFT* (단계 3, Ch 28) 의 가장 큰 차이는 *`labels = -100` 자리* 입니다. 본 챕터의 collator 가 만드는 `labels` 패턴과, Ch 28 에서 등장할 `labels[:prompt_len] = -100` 한 줄의 차이를 직접 비교해 설명해 보세요.
4. Ch 27 AFTER 의 generation 이 Ch 26 보다 좋아 보인다면, *모델 크기 (3M → 125M)* 의 효과인가 *사전학습 데이터 (없음 → 대규모 한국어 코퍼스)* 의 효과인가? 본 챕터의 실험 셋업으로는 둘을 분리할 수 있나요? (분리하려면 어떤 추가 실험이 필요할까요?)

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

### Q2. (실무) 왜 lr 가 Ch 26 의 `5e-4` 보다 작은 `2e-5` 인가요?

**catastrophic forgetting 방지** 가 핵심 이유. *사전학습된 본체* 는 weight 들이 *이미 의미 있는 표상* 을 학습한 상태. 큰 lr 로 학습하면:

1. *사전학습된 표상이 새 데이터에 맞춰 크게 흔들림*
2. *원래 알던 일반 도메인 지식 (일반 한국어 전반)* 이 *TinyStories 동화 풍* 으로 *덮어쓰기*
3. 결과: *TinyStories 도메인은 잘 하지만 일반 한국어 능력 손실* — 이게 catastrophic forgetting

작은 lr (`2e-5`) 는 *사전학습된 표상 보존* 을 우선합니다. *기존 weight 에서 살짝만 떨어진 지점으로 이동* — 도메인 적응은 하되 일반 능력은 유지.

```python
# Ch 26 (scratch) - 큰 lr 로 표상 빨리 학습
TrainingArguments(learning_rate=5e-4, ...)

# Ch 27 (continual pretraining) - 작은 lr 로 표상 보존
TrainingArguments(learning_rate=2e-5, ...)
```

HF 의 continual pretraining / fine-tuning 표준 lr 범위: `1e-5` - `5e-5`. SFT (Ch 28) 도 비슷한 범위. (영어 Ch 25 와 같은 `2e-5`.)

### Q3. (이론) Ch 26 (3M) 가 같은 데이터로 학습했는데 Ch 27 (125M) 결과가 훨씬 좋다면, *모델 크기의 위력* 인가 *사전학습의 위력* 인가?

**둘이 *섞여서* 분리 불가능** 입니다. 본 챕터의 셋업은 *두 변수가 동시에 변함*:

- Ch 26 → Ch 27 변화: 모델 크기 *3M → 125M (약 40배)* + 사전학습 *없음 → 대규모 한국어 코퍼스*

진짜 *모델 크기와 사전학습 효과를 분리* 하려면 *2 × 2 격자* 실험이 필요:

| 셋업 | 모델 크기 | 사전학습 | 본 커리큘럼 |
|---|---|---|---|
| (a) | 3M | 없음 | **Ch 26** |
| (b) | 3M | 대규모 한국어 사전학습 | (미실험) |
| (c) | 125M | 없음 | (미실험 — 125M scratch + TinyStories 만 학습) |
| (d) | 125M | 대규모 한국어 코퍼스 | **Ch 27** |

본 커리큘럼에는 (a) 와 (d) 만 있어 *둘의 차이* 만 보입니다. (b) 와 (c) 는 *T4 + 30분 룰* 안에 어렵습니다 (125M scratch 는 *TinyStories 만으로 의미 있는 학습이 부족함*, 3M 대규모 사전학습은 *데이터 규모 자체가 30분에 안 맞음*).

**실용적 결론**: 실무에서는 (b)(c) 가 *비용 대비 비효율* 이라 (d) 패턴이 표준. *대규모 사전학습 모델을 가져와 작은 도메인 데이터로 continual pretraining* — 본 챕터의 패턴이 그 자체로 *실무 표준 레시피*. (영어 Ch 25 Q3 의 한국어판.)

### Q4. (실무) KoGPT2 토크나이저로 *영어* TinyStories 를 학습하면 어떻게 되나요? (Ch 25 Q4 의 거울)

**작동은 하지만 비효율적** 입니다 — KoGPT2 BBPE 는 *byte-level* 이라 영어도 UNK 없이 표현하지만, *한국어 코퍼스 중심* 으로 학습돼 영어 어절의 병합 규칙이 약합니다. 그래서 *같은 영어 문장이 영어 gpt2 BPE 보다 다소 많은 토큰* 으로 쪼개질 수 있습니다.

```python
from transformers import PreTrainedTokenizerFast, AutoTokenizer
ko_tok = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)                                          # KoGPT2 는 반드시 이 방식으로
en_tok = AutoTokenizer.from_pretrained("gpt2")   # 영어 gpt2 는 AutoTokenizer OK
sent = "Once upon a time, a little rabbit lived in the forest."
print(len(ko_tok(sent)["input_ids"]))   # 한국어 중심 vocab -> 영어는 다소 많은 토큰
print(len(en_tok(sent)["input_ids"]))   # 영어 중심 vocab -> 적은 토큰
```

핵심 교훈은 Ch 25 Q4 와 동일 — *토크나이저는 본체와 운명공동체*. 영어는 영어 사전학습 모델 (gpt2), 한국어는 한국어 사전학습 모델 (KoGPT2) 을 가져오는 게 정공법. 그래서 영어 Ch 25 와 한국어 Ch 27 이 *각자의 언어에 맞는 본체* 로 같은 단계 2 를 수행합니다.

### Q5. (이론·실무) *catastrophic forgetting* 이 무엇인가요? Ch 27 에서 실제로 일어나나요?

**Catastrophic forgetting** (재앙적 망각): 새 데이터로 학습할 때 *이전에 학습한 표상이 덮어쓰기* 되어 *원래 알던 능력이 손실* 되는 현상.

Ch 27 에서는 *짧은 (1 epoch) continual pretraining + 작은 lr (`2e-5`)* 라 *catastrophic forgetting 이 강하지 않음*. 다만 *변형 1 (epoch 늘리기)* 또는 *변형 4 (큰 lr)* 를 시도하면 다음 신호가 보입니다:

- 비-동화 prompt (예: `"대한민국의 수도는"`) 에 대해 *KoGPT2 가 원래 답했을 만한 일반 한국어* 가 *동화풍 톤* 으로 끌려감
- *generation 다양성 하락* — 모든 prompt 에 *소녀 / 엄마 / 친구* 류 동화 단어가 자주 등장
- *evaluation* — 만약 일반 한국어 벤치마크에 *학습 전 / 후* 를 측정하면 *학습 후가 더 낮은 점수*

방지법:
1. **짧은 학습 + 작은 lr** (본 챕터 패턴)
2. **regularization** — replay (이전 데이터 일부 섞어 학습) / EWC (Elastic Weight Consolidation) 등
3. **adapter / LoRA** — 본체 weight 는 freeze 하고 작은 adapter 만 학습 → 본체 표상 보존

본 챕터는 *방법 1* 만 적용. *방법 3 (LoRA)* 는 본 커리큘럼 범위 밖이지만 *실무에서는 표준 옵션*.

### Q6. (실무) 왜 KoGPT2 는 `AutoTokenizer` 가 아니라 `PreTrainedTokenizerFast` 로 로드하나요?

**`PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` 는 영어 GPT2 토크나이저로 잘못 fallback** 하기 때문입니다. 그 결과 special token 이 `<|endoftext|>` 로 잡히고 한국어가 깨집니다 — 직접 확인해 보면:

```python
from transformers import AutoTokenizer
bad = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
print(type(bad).__name__)                 # GPT2Tokenizer (slow, 영어 fallback)
print(bad.encode("옛날 옛날에"))            # [501, 500, 529, ...] (잘못된 id)
print(repr(bad.decode(bad.encode("옛날 옛날에"))))   # '�����' (깨짐)
```

SKT 공식 방식 — `PreTrainedTokenizerFast` + special token 명시:

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
print(type(tokenizer).__name__)            # fast tokenizer, vocab 51,200
print(tokenizer.encode("옛날 옛날에"))      # [12346, 35970] (정상)
print(repr(tokenizer.decode([12346, 35970])))   # '옛날 옛날에' (정상 왕복)
model.config.pad_token_id = tokenizer.pad_token_id   # 본체 config 동기화
```

이렇게 로드하면 `pad_token` 이 `<pad>` 로 제대로 잡혀 별도 가드도 필요 없습니다. 영어 Ch 25 의 gpt2 는 `AutoTokenizer` 가 정상 작동했지만 (`pad` 만 eos 로 보충), KoGPT2 는 *로드 방식 자체* 가 다릅니다 — **모델 카드 example code 확인 + encode/decode 왕복 검증** 이 이런 함정을 막는 습관입니다.

### Q7. (이론) 다음 챕터 (Ch 28 KoGPT2 SFT) 와의 관계는?

Ch 28 = *같은 KoGPT2 본체* 를 *instruction-response 쌍 데이터* 로 **SFT** (단계 3). 본 챕터 (단계 2, continual pretraining) 와의 결정적 차이는 *`labels = -100` 자리*:

```python
# Ch 27 (본 챕터, continual pretraining) - 거의 모든 자리 학습
# collator 가 자동으로: labels = input_ids.clone()  (pad 만 -100)

# Ch 28 (SFT) - prompt 부분만 -100
prompt = "### 질문: 한국의 수도는?\n### 답변: "
response = "서울입니다."
input_ids = tokenizer(prompt + response)["input_ids"]
labels = input_ids.copy()
prompt_len = len(tokenizer(prompt)["input_ids"])
labels[:prompt_len] = [-100] * prompt_len   # <- 이 한 줄이 SFT 의 핵심
```

| 항목 | Ch 27 (continual pretraining) | Ch 28 (SFT) |
|---|---|---|
| 본체 | KoGPT2 125M | **KoGPT2 125M (동일)** |
| 데이터 | 연속된 일반 텍스트 (TinyStories) | **instruction-response 쌍** |
| `labels = -100` 자리 | pad 만 | **prompt 부분** |
| 효과 | 도메인 적응 | **instruction 따라가기 (행동 정렬)** |

본 챕터의 collator 출력 (거의 모든 자리 = 학습 신호) 을 손에 익혀 두면, Ch 28 의 *왜 모델이 한국어 instruction 을 따라가게 되는가* 가 *한 줄 `labels[:prompt_len] = -100`* 으로 단번에 이해됩니다. 그게 Phase 4 thread 의 클라이맥스.

## 다음 챕터 예고

**Chapter 28. KoGPT2 SFT — instruction 데이터로 행동 정렬 (Phase 4 단계 3)**

- `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` — *본 챕터와 같은 KoGPT2 본체*. 다만 출발점을 *continual pretrained 모델* 또는 *base 모델* 로 선택 가능
- **instruction-response 쌍 데이터** (예: KoAlpaca 류) 로 **SFT** (Supervised Fine-Tuning) — *행동 정렬*. *task adaptation 의미의 fine-tune 도, 단순 continual pretraining 도 아닌 단계 3*
- **핵심 한 줄**: `labels[:prompt_len] = -100` — *prompt 는 외우지 않고 response 만 학습*. 본 챕터의 collator 출력 (거의 모든 자리 학습) 과 *정반대 자리*
- *trainer 자체는 본 챕터와 거의 동일* (`transformers.Trainer`) — *변하는 건 데이터 형식 + collator 의 `-100` 마스킹*
- Phase 4 GPT 시대 thread 의 클라이맥스 — *왜 모델이 한국어 instruction 을 따라가게 되는가* 의 정확한 답

**Phase 4 GPT 시대 4단계 흐름 정리**:

| 챕터 | 단계 | 본체 | 데이터 | 핵심 |
|---|---|---|---|---|
| Ch 24 | 1 (영어) | 작은 GPT scratch | 영어 TinyStories | 단계 1 출발 |
| Ch 25 | 2 (영어) | `gpt2` 124M | 영어 TinyStories (동일) | 단계 2: continual pretraining |
| Ch 26 | 1 (한국어) | 작은 GPT scratch | 한국어 TinyStories | 한국어 단계 1 |
| **Ch 27 ← 여기** | **2 (한국어)** | **KoGPT2 125M** | **한국어 TinyStories (동일)** | **한국어 단계 2: continual pretraining** |
| Ch 28 | 3 | KoGPT2 + SFT | 한국어 instruction 데이터 | **단계 3: SFT** (`labels[:prompt_len] = -100`) |
| Ch 29-30 | 4 | SFT 모델 + DPO / GRPO | preference / verifier reward | **단계 4: Alignment** |

> **변하는 축** (Ch 27 → Ch 28): *학습 단계* (continual pretraining → SFT). 본체·언어는 같고, *데이터 형식 + `labels = -100` 자리* 만 바뀜. 본 챕터의 collator 출력 (거의 모든 자리 학습) 이 *그 한 줄과 정확히 대비되는 기준선* — Ch 28 이 Phase 4 thread 의 클라이맥스인 이유.
