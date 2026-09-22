## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | Ch 27 과 차이 |
|---|---|---|
| `trl.SFTTrainer` | SFT 특화 trainer (prompt/completion 전처리 + completion 마스킹 자동) | **새로 등장** (Ch 27 은 `transformers.Trainer`) |
| `trl.SFTConfig` | `SFTTrainer` 설정 (`TrainingArguments` 상속 + SFT 옵션) | **새로 등장** |
| `SFTConfig(completion_only_loss=True)` | *답변 부분만* loss (prompt = `-100`) — SFT 의 핵심 옵션 | **새로 등장** |
| `trl` 의 SFT 데이터 준비 + collator | 데이터 준비 단계에서 `completion_mask` 로 prompt 를 `-100` 마스킹한 `labels` 를 생성, collator 는 패딩만 담당 (trl 1.10 기준) | **새로 등장** (Ch 27 은 `transformers.DataCollatorForLanguageModeling(mlm=False)`) |
| `prompt` / `completion` 데이터 형식 | instruction-response 쌍 표준 형식 | **새로 등장** (Ch 27 은 단일 `text` 컬럼) |
| `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` | KoGPT2 본체 로드 | **공유** (Ch 27 과 같은 본체) |
| `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` | KoGPT2 BBPE 토크나이저 (AutoTokenizer 함정 회피) | **공유** (Ch 27 과 동일) |
| `model.generate(repetition_penalty=...)` | 반복 억제 sampling (작은 모델의 반복 완화) | **약간 다름** (반복 페널티 추가) |

> `trl` 은 버전마다 API 변동이 큰 라이브러리입니다 (`DataCollatorForCompletionOnlyLM` 처럼 버전에 따라 사라진 클래스도 있습니다). 본 노트북은 *`prompt`/`completion` 데이터 + `completion_only_loss=True`* 라는 *최신 trl 의 표준 경로* 를 씁니다 — 이 경로가 버전 간 가장 안정적입니다. 설치된 `trl` 버전은 셋업 셀의 출력에서 확인하세요.

## 체크포인트 질문

1. SFT 에서 *왜 prompt 부분을 `-100` 으로 가리나요?* 만약 prompt 도 학습 대상에 포함하면 (response-only 가 아니라 전체 학습) 모델은 무엇을 강화하게 되고, 왜 그게 instruction following 에 불리한가요?
2. *Continual pretraining (Ch 27)* 과 *SFT (Ch 28)* 는 *같은 본체 + 같은 next-token CrossEntropyLoss* 를 씁니다. 그런데도 둘은 다른 단계입니다 — 정확히 *무엇 두 가지* 가 다른가요? (데이터 형식 / `labels = -100` 자리)
3. `### 응답:\n` (response_template) 의 역할은 무엇인가요? collator 는 이 문자열을 어떻게 사용해 prompt 와 답변을 나누나요?
4. *"같은 모델이 입력 프롬프트만 바꾸면 다른 task 를 한다"* — BERT 시대 (task 마다 새 head) 와 비교해 이게 왜 가능한가요? SFT 가 그 능력에서 하는 역할은?

## FAQ

### Q1. (이론) SFT 가 continual pretraining (Ch 27) 과 정확히 뭐가 다른가요?

*같은 본체, 같은 next-token `CrossEntropyLoss`* 를 쓴다는 점은 같습니다. 다른 건 **두 가지**:

| 항목 | Continual pretraining (Ch 27) | SFT (Ch 28) |
|---|---|---|
| **데이터 형식** | 연속된 일반 텍스트 (TinyStories) | **instruction-response 쌍** (KoAlpaca) |
| **`labels = -100` 자리** | pad 만 (거의 모든 자리 학습) | **prompt 부분 전체** (답변만 학습) |

```python
# Ch 27 (continual pretraining) - 거의 모든 자리 학습
# collator: labels = input_ids.clone()  (pad 만 -100)

# Ch 28 (SFT) - prompt 는 -100, 답변만 학습
SFTConfig(completion_only_loss=True)   # collator 가 prompt 를 자동 -100 마스킹
```

continual pretraining 은 *모델의 지식·도메인* 을 다듬고, SFT 는 *모델의 행동 (지시를 따르는 법)* 을 정렬합니다. 그래서 실무에서는 보통 *pretraining → (continual pretraining) → SFT → alignment* 순서로 *쌓아* 갑니다.

### Q2. (이론) 왜 prompt 를 `-100` 으로 가리나요? 가리지 않으면 어떻게 되나요?

*질문을 외우는 것* 과 *답하는 법을 배우는 것* 의 차이입니다. prompt 도 학습 대상에 넣으면:

1. loss 의 *상당 부분* 이 *prompt 토큰* 에서 나옵니다 (prompt 가 보통 답변보다 길거나 비슷). 모델은 *질문을 받아쓰는* 데 gradient 를 씁니다
2. 모델이 *주어진 질문 분포* 에 과적합 — *새로운 질문* 에 약해질 수 있습니다
3. 우리가 원하는 *"질문이 주어졌을 때 답하는 능력"* (조건부 생성) 이 희석됩니다

`labels[:prompt_len] = -100` 한 줄로 *prompt 는 조건 (given), 답변만 학습 대상 (target)* 으로 분리합니다. §3 에서 본 collator 출력이 정확히 이 효과입니다.

```python
# 개념적으로 (collator 가 자동으로 해 주는 일)
input_ids = tokenizer(prompt + response)["input_ids"]
labels = input_ids.copy()
prompt_len = len(tokenizer(prompt)["input_ids"])
labels[:prompt_len] = [-100] * prompt_len   # <- prompt 를 loss 에서 제외
```

### Q3. (실무) `SFTTrainer` 는 `transformers.Trainer` 와 뭐가 다른가요?

`SFTTrainer` 는 `transformers.Trainer` 를 *상속* 한 서브클래스입니다 — 학습 루프 (forward / backward / optimizer step) 는 *완전히 동일*. 다른 건 *데이터 전처리를 자동화* 한다는 점:

- *prompt + completion* 을 토큰화해 이어 붙이고, 답변 끝에 **EOS 를 자동 부착**
- `completion_only_loss=True` 면 *completion 마스킹* (`completion_mask` 생성 → prompt `-100`) 을 자동
- (옵션) `packing`, chat template 적용 등 SFT 편의 기능

```python
# transformers.Trainer (Ch 27) - 직접 토큰화 + group_texts + collator 설정
# trl.SFTTrainer (Ch 28) - prompt/completion 데이터만 주면 위 과정 자동
trainer = SFTTrainer(model=model, args=SFTConfig(completion_only_loss=True),
                     train_dataset=sft_ds, processing_class=tokenizer)
```

즉 *같은 학습 루프, 더 적은 보일러플레이트*. SFT 의 *마스킹 같은 디테일* 을 라이브러리가 처리해 줍니다.

### Q4. (이론) chat template 이 뭔가요? KoGPT2 는 왜 직접 포맷하나요?

**chat template** 은 *대화 메시지 (`{"role": "user", "content": ...}`) 를 모델이 학습한 형식의 문자열로 변환하는 규칙* 입니다. instruction-tuned 모델 (예: Llama-Instruct, Qwen-Chat) 은 토크나이저에 chat template 이 내장돼 있어 `tokenizer.apply_chat_template(messages)` 한 줄로 포맷됩니다.

**KoGPT2 는 *base 모델* (instruction tuning 안 됨) 이라 chat template 이 없습니다.** 그래서 우리가 *직접* 포맷합니다:

```python
def build_prompt(instruction):
    return f"### 명령어:\n{instruction}\n\n### 응답:\n"
```

> *우리가 정한 포맷* (`### 명령어:` / `### 응답:`) 으로 SFT 하면, *추론 시에도 같은 포맷* 으로 입력해야 합니다. SFT 가 *그 포맷을 모델의 chat template 으로* 가르치는 셈입니다. instruction-tuned 모델을 *직접 만드는* 과정이 곧 *그 모델의 chat template 을 정의* 하는 일입니다.

### Q5. (실무) 더 큰 모델을 SFT 하려면? LoRA / QLoRA 는 무엇인가요?

KoGPT2 (125M) 는 T4 에서 full fine-tuning 이 가능하지만, *7B 급 이상* 은 full SFT 가 *T4 메모리 (16GB) 를 초과* 합니다. 그래서 실무 표준은 **LoRA** (Low-Rank Adaptation):

- 본체 weight 는 *freeze* (그대로 두고)
- 각 layer 에 *작은 low-rank adapter 행렬 (`r=8-64`)* 만 추가해 *그것만 학습*
- 학습 파라미터가 *전체의 약 0.1-1%* → 메모리·시간 대폭 절감

**QLoRA** 는 여기에 *본체를 4bit 양자화* 까지 더해 *더 큰 모델 (예: 70B) 도 단일 GPU* 에서 SFT 가능하게 합니다.

```python
from peft import LoraConfig
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"],
                         lora_dropout=0.05, task_type="CAUSAL_LM")
trainer = SFTTrainer(model=model, args=sft_config, train_dataset=sft_ds,
                     processing_class=tokenizer, peft_config=peft_config)
```

본 챕터는 *full SFT* (LoRA 없이) 로 *마스킹의 원리* 에 집중했습니다. LoRA 는 *메모리 기법* 일 뿐 *마스킹·loss 원리는 동일* 합니다.

### Q6. (실무) SFT 후 답변 품질이 거친데요? (반복, 사실 오류)

KoGPT2 는 *125M 의 작은 base 모델* 이고, 본 챕터의 SFT 는 *약 3K 샘플 / 1 epoch* 로 *최소 규모* 입니다. 그래서:

- **반복** — `model.generate(repetition_penalty=1.3)`, `no_repeat_ngram_size=3` 등으로 완화
- **사실 오류 / 환각** — 작은 모델의 근본 한계. 더 큰 모델 + RAG (검색 증강) 로 보완
- **포맷 일관성** — 더 많은 데이터 / epoch 로 개선

> 본 챕터의 목표는 *답변의 정확도* 가 아니라 ***instruction 을 따라가는 행동 자체가 생겼는가*** 입니다. §6 의 BEFORE/AFTER 에서 *지시를 따르는 방향* 으로 바뀌었다면 SFT 의 핵심 (행동 정렬) 은 성공한 것입니다. 품질은 *모델 크기 + 데이터 + LoRA* 의 영역.

### Q7. (이론) 다음 단계 alignment (DPO, Ch 30) 는 SFT 와 뭐가 다른가요?

SFT 는 *"좋은 답변 하나" 를 따라 학습* 합니다 (정답 demonstration 모방). 하지만 *"여러 답변 중 어느 게 더 나은가"* 라는 *선호 (preference)* 는 가르치지 못합니다. **alignment (DPO 등)** 가 그 단계:

| 단계 | 데이터 | 학습 신호 |
|---|---|---|
| **SFT (Ch 28)** | instruction → *하나의* 좋은 답변 | 그 답변을 *따라 생성* |
| **DPO (Ch 30)** | instruction → *(chosen, rejected) 쌍* | chosen 을 *더 선호*, rejected 를 *덜 선호* |

> 흥미롭게도 DPO 도 *`labels = -100` thread 를 잇습니다* — chosen / rejected 각각의 *response 부분에서만* log-likelihood 를 계산합니다 (prompt 는 양쪽 공통이라 제외). 즉 *답변 부분만 본다* 는 본 챕터의 원리가 alignment 까지 이어집니다. DPO 는 Ch 30 에서 본격.

```python
# Ch 30 미리보기 (DPO)
# from trl import DPOTrainer, DPOConfig
# 데이터: {"prompt": ..., "chosen": ..., "rejected": ...}
# chosen 의 response 확률은 높이고, rejected 의 response 확률은 낮춤
```

## 다음 챕터 예고

**Chapter 29. 벤치마크 평가 — SFT 모델을 분야별 벤치마크로 측정**

- 평가 대상은 **Qwen2.5-0.5B-Instruct** — 본 챕터의 SFT 모델은 Ch29의 §7 섹션에서 대조(비교) 용도로만 서술
- **KoBEST(HellaSwag·BoolQ subset) MC 직접 구현 + 산술 생성 평가 + `lm-evaluation-harness` 시연 1건** — MMLU/KMMLU/GSM8K/LogicKor 등은 분야 소개(§6 지도)로만 다룸, 실제 실행 대상 아님
- *분류(Ch 1-23) vs 생성(Ch 24-)* 평가 방식의 근본 차이 — 정답이 하나가 아니라 log-likelihood/생성 채점이 필요한 이유
- 그 다음 Ch 30 (DPO) — *preference 정렬*. **`labels = -100` thread 가 DPO 에서도 이어집니다** — chosen/rejected *response 부분만* 계산

**Phase 4 GPT 시대 4단계 흐름 정리**:

| 챕터 | 단계 | 본체 | 데이터 | `labels = -100` 자리 |
|---|---|---|---|---|
| Ch 24·26 | 1 (pretraining) | 작은 GPT scratch | TinyStories (영/한) | pad 만 |
| Ch 25·27 | 2 (continual pretraining) | gpt2 / KoGPT2 | TinyStories (동일) | pad 만 |
| **Ch 28 ← 여기** | **3 (SFT)** | **KoGPT2 (동일)** | **KoAlpaca instruction-response** | **prompt 부분 (답변만 학습)** |
| Ch 30·31 | 4 (alignment) | SFT 모델 + ref | preference / verifier reward | response 부분만 (RL 내부) |

> **변하는 축** (Ch 27 → Ch 28): *학습 단계* (continual pretraining → SFT). 본체·토크나이저·loss 종류는 같고, *데이터 형식 + `labels = -100` 자리* 가 바뀝니다. 본 챕터에서 그 *마스킹 자리* 를 collator 출력으로 *눈으로 확인* 했습니다 — 그게 Phase 4 두 thread 의 클라이맥스입니다.
