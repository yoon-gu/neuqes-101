**목표**: Phase 4 의 *한국어 단계 2 (continual pretraining)* 챕터. Ch 26 에서 *random init 작은 GPT (약 3M params) 를 한국어 TinyStories 로 from scratch 사전학습* 했다면, 이번엔 **KoGPT2 (`skt/kogpt2-base-v2`, 125M params, 대규모 한국어 코퍼스로 사전학습된 본체)** 를 *같은 한국어 TinyStories 데이터* 로 **continual pretraining** (계속 사전학습 / continual learning) 합니다. **같은 CausalLM task, 같은 LM head, 같은 collator, 같은 loss** — 변하는 건 *모델 로드 한 줄 + 토크나이저 + 학습률* 뿐. 그게 GPT 시대 *학습 단계 2 (continual pretraining)* 의 본질입니다. Ch 24→Ch 25 (영어) 의 한국어 짝.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 25-30분 (데이터 로드·story 복원 약 3분 + KoGPT2 로드·토큰화 약 2분 + 학습 전 baseline generation 약 1분 + continual pretraining 약 17분 + 학습 후 generation + 3-way 비교 약 2분)


## 학습 흐름

1. 📊 **누적 추적표 + Phase 4 영어·한국어 대칭** — Ch 25(영어 단계2) ↔ Ch 27(한국어 단계2)
2. 🔄 **변경점 (Diff from Ch 26)** — *모델 출발점 + 토크나이저 + lr* 만 변함. *데이터·trainer·collator·loss 는 동일*
3. 🌏 **GPT 시대 학습 4단계 표** — 본 챕터의 위치 (단계 2). Ch 27 은 *SFT 가 아님* 을 명확히
4. 📐 **Loss** — 변화 없음 (CE next-token). 다만 *시작점이 random 이 아닌 대규모 한국어 사전학습 본체* 라는 게 핵심
5. 🔤 **토크나이저 노트** — *KoGPT2 BBPE 그대로* (vocab 51,200). Ch 26 의 직접 학습 BBPE (vocab 약 4,000) 와 비교
6. 🚀 **실습**: 한국어 TinyStories 30K → KoGPT2 로드 → 학습 전 generation → continual pretraining → 학습 후 generation
7. 🆚 **3-way generation 비교** — Ch 26 (3M scratch) vs Ch 27 BEFORE (KoGPT2 그대로) vs Ch 27 AFTER (continual pretraining)
8. 📦 **등장 라이브러리** / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)


> 📒 **사전 학습 자료**: Ch 26 (한국어 GPT scratch + 한국어 TinyStories), Ch 25 (영어 gpt2 continual pretraining — 본 챕터의 영어 짝). 본 챕터는 Ch 26 과 *데이터·trainer·collator·loss 모두 같고 본체 출발점·토크나이저·lr 만 다른* 격리 실험. *trainer 코드 차이가 극단적으로 적음* — 그게 *학습 단계 2 (continual pretraining)* 의 본질입니다.

## 누적 추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 24 | 작은 GPT2 (약 3M, scratch) | BPE (직접 학습, 영어, vocab 2,048) | 영어 TinyStories 30K | `Linear(H, V)` (LM head, weight tied) | `CrossEntropyLoss` (next-token) |
| 25 | `gpt2` (124M, OpenAI WebText 사전학습) | BPE (gpt2 그대로, vocab 50,257) | 영어 TinyStories (Ch 24 와 동일) | `Linear(H, V)` (LM head 그대로) | `CrossEntropyLoss` (next-token) - continual pretraining |
| 26 | 작은 GPT2 (한국어, 약 3M, scratch) | BBPE (직접 학습, 한국어, vocab 약 4,000) | 한국어 TinyStories 30K | `Linear(H, V)` (LM head, weight tied) | `CrossEntropyLoss` (next-token) |
| **27 ← 여기** | **KoGPT2 `skt/kogpt2-base-v2` (125M, 대규모 한국어 사전학습)** | **BBPE (KoGPT2 그대로, vocab 51,200)** | **한국어 TinyStories 30K (Ch 26 과 동일)** | **`Linear(H, V)` (LM head 그대로)** | **`CrossEntropyLoss` (next-token) — *continual pretraining*** |
| 28 (다음) | KoGPT2 + SFT | KoGPT2 BBPE (그대로) | 한국어 instruction 데이터 | `Linear(H, V)` (LM head 그대로) | `CrossEntropyLoss` (`labels[:prompt_len] = -100`) |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.


## Phase 4 의 영어·한국어 대칭 — Ch 25 ↔ Ch 27

Phase 4 는 영어 (Ch 24-25) 와 한국어 (Ch 26-27) 가 *같은 학습 단계* 를 *언어만 바꿔* 반복하는 구조입니다. 본 챕터는 영어 Ch 25 (gpt2 continual pretraining) 의 한국어 짝.

| 학습 단계 | 영어 | **한국어** |
|---|---|---|
| **단계 1: Pretraining** (random init → scratch) | Ch 24 (작은 GPT, 영어 TinyStories) | Ch 26 (작은 GPT, 한국어 TinyStories) |
| **단계 2: Continual pretraining** (사전학습 본체 + 새 데이터) | Ch 25 (`gpt2` 124M + 영어 TinyStories) | **Ch 27 ← 여기 (KoGPT2 125M + 한국어 TinyStories)** |

> Ch 24 ↔ Ch 26 (scratch, 언어만 다름) 과 **Ch 25 ↔ Ch 27** (continual pretraining, 언어만 다름) 이 *영어·한국어 완전 대칭*. 본 챕터는 한국어 단계 2.


### Ch 26 ↔ Ch 27 격리 실험의 통제 변수

| 항목 | Ch 26 | **Ch 27 (본 챕터)** | 같음 / 다름 |
|---|---|---|---|
| 데이터 | 한국어 TinyStories 30K | 한국어 TinyStories 30K | **같음** (통제 변수) |
| Trainer 클래스 | `transformers.Trainer` | `transformers.Trainer` | **같음** |
| Data collator | `DataCollatorForLanguageModeling(mlm=False)` | `DataCollatorForLanguageModeling(mlm=False)` | **같음** |
| Loss | CE next-token (`labels = input_ids.clone()`) | CE next-token (`labels = input_ids.clone()`) | **같음** |
| 본체 출발점 | `GPT2LMHeadModel(config)` random init (약 3M) | `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` (125M) | **다름** |
| 토크나이저 | BBPE 직접 학습 (vocab 약 4,000) | `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` (vocab 51,200) | **다름** (본체와 운명공동체) |
| 학습률 | 5e-4 (scratch 표준) | **2e-5** (continual pretraining 표준) | **다름** |
| 학습 step | 약 1,500 | **약 3,000** (48,513 chunks / eff. batch 16, 1 epoch) | **다름** (lr 만 작아짐) |

> **세 줄 차이가 곧 *학습 단계 2 (continual pretraining)* 의 정의** — 같은 task, 같은 collator, 같은 loss, 같은 trainer. *모델 로드 한 줄 + 토크나이저 한 줄 + lr 한 숫자* 만 바꾸면 됩니다. Ch 25 (영어) 에서 본 격리 실험의 한국어 재확인.

## 변경점 (Diff from Ch 26)

| 축 | Ch 26 (한국어 GPT scratch) | Ch 27 (본 챕터, KoGPT2 continual pretraining) |
|---|---|---|
| **본체** | 작은 GPT2 (약 3M params, random init) | **KoGPT2 `skt/kogpt2-base-v2`** (125M, 대규모 한국어 코퍼스 사전학습) ← *출발점 변화* |
| **토크나이저** | BBPE 직접 학습 (vocab 약 4,000) | **`PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)`** (vocab 51,200) ← *본체에 맞춰 함께 변함* |
| 데이터 | 한국어 TinyStories 30K | **한국어 TinyStories 30K (동일)** ← 통제 변수 |
| Trainer | `transformers.Trainer` | **`transformers.Trainer` (동일)** |
| Data collator | `DataCollatorForLanguageModeling(mlm=False)` | **(동일)** |
| Loss | CE next-token (`labels = input_ids.clone()`) | **(동일)** |
| **학습률** | 5e-4 | **2e-5** ← *유일한 hyperparam 큰 차이* |
| 학습 step | 약 1,500 (T4 약 1분) | **약 3,000 (T4 약 17분)** — 125M 본체라 step 당 비용이 큼 |
| Generation 품질 | 동화 풍 단순 한국어 | **자연스러운 동화 + 일반 도메인 폭** ← 메시지 |

> **핵심**: *Ch 26 ↔ Ch 27 은 데이터·trainer·collator·loss 모두 같고 본체 출발점·토크나이저·lr 만 다름*. **trainer 코드 차이가 극단적으로 적음** — 그게 GPT 시대 학습 단계 2 (continual pretraining) 의 본질. *task adaptation 의미의 fine-tune 이 아닙니다* — head 바뀌지 않고, task (next-token 예측) 바뀌지 않습니다. 데이터만 바뀝니다. Ch 24→Ch 25 (영어) 에서 본 그 격리 실험의 한국어 짝.

## GPT 시대 학습 4단계 — 본 챕터의 위치 (단계 2, 한국어)

Ch 24 에서 도입한 GPT 시대 학습 4단계 표의 *단계 2 (continual pretraining)* 의 *한국어판* 이 본 챕터입니다.

| 단계 | 정확 용어 | 의미 | `labels = -100` 자리 | 영어 | 한국어 | 본 챕터? |
|---|---|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | 일반 코퍼스 위에 random init 본체부터 학습 | pad 만 | Ch 24 | Ch 26 | |
| 2 | **Continual pretraining** (계속 사전학습 / continual learning) | *사전학습된 본체* 를 *새 데이터* 로 *같은 CausalLM task* 더 학습. **head 그대로, task 그대로, 데이터만 새로** | pad 만 (단계 1 과 동일) | Ch 25 | **Ch 27 ← 여기** | ✅ |
| 3 | **SFT** (Supervised Fine-Tuning / Instruction tuning) | instruction-response 쌍으로 *행동 정렬*. `labels[:prompt_len] = -100` 으로 답변 부분만 학습 | **prompt 부분** | - | Ch 28 | |
| 4 | **Alignment** (DPO / RLHF / GRPO) | preference 또는 verifier reward 로 *선호 정렬* | (RL 내부) | - | Ch 30-31 | |

### Ch 27 은 *SFT 가 아닙니다*

본 챕터를 *fine-tune* 으로 부르면 *단계 2 / 3 / 4* 가 모두 섞여 혼동이 생깁니다. Ch 27 의 정확한 위치는:

- **`task adaptation` 의미의 fine-tune 이 아님** — output head 안 바뀜 (LM head 그대로), task 안 바뀜 (next-token 예측 그대로), loss 안 바뀜 (CE)
- **`instruction tuning` 의미의 SFT 가 아님** — `labels = -100` 자리가 *pad 만* (Ch 26 과 동일). prompt-response 쌍 데이터 형식이 아니라 *연속된 일반 텍스트*
- **`continual pretraining` 그 자체** — *사전학습된 본체 + 새 도메인 데이터 + 같은 CausalLM task* 의 조합. *데이터만 바뀐 단계 1 의 연장*

> SFT (단계 3) 는 Ch 28 에서 본격. *왜 모델이 한국어 instruction 을 따라가게 되는가* 는 `labels[:prompt_len] = -100` 한 줄로 정확히 설명됩니다. 본 챕터의 collator 출력 (거의 모든 자리 = 학습 신호) 이 그 한 줄과 정확히 대비되는 *학습 단계 2 의 기준선*. 그 클라이맥스가 Ch 28.

## Loss — 변화 없음, 다만 *시작점* 이 대규모 한국어 사전학습 본체

Ch 26 과 *완전히 동일* 한 `CrossEntropyLoss` (next-token, `mlm=False`). `labels = input_ids.clone()`, pad 만 `-100`. 단 *vocab 차원이 약 4,000 → 51,200 로 커진* 영향과, *모델 본체가 random init 이 아닌 이미 대규모 한국어 코퍼스로 학습된 상태* 라는 두 차이가 *loss 곡선의 시작 지점* 을 결정합니다.

### Random baseline 의 변화

| 토크나이저 | vocab 차원 | `ln(vocab)` (uniform CE) | 챕터 |
|---|---|---|---|
| 직접 학습 BBPE (한국어) | 약 4,000 | 약 8.29 | Ch 26 |
| **KoGPT2 BBPE** | **51,200** | **10.84** | **Ch 27** |
| `gpt2` BPE (영어) | 50,257 | 10.82 | Ch 25 (참고) |

*만약* KoGPT2 본체가 random init 이었다면 첫 step loss 가 약 10.84 부근에서 시작할 것입니다. 하지만 **KoGPT2 본체는 이미 대규모 한국어 코퍼스로 사전학습되어 있어** *TinyStories 평가에서도 시작 loss 가 random baseline 보다 훨씬 낮습니다* — 그게 학습 단계 2 의 핵심 차이.

### 숫자로 감 잡기 — *시작점 ↔ 도달점*

| 상태 | 정답 토큰 확률 | $-\log p$ |
|---|---|---|
| 균등 추측 (KoGPT2 vocab 51,200) | $1/51200$ | **10.84** ← random baseline (도달 불필요) |
| KoGPT2 사전학습 그대로, TinyStories 평가 | $0.05$ - $0.10$ 범위 | **2.5 - 3.0** ← *우리 시작점* (이미 좋음) |
| Continual pretraining 후 (수백 step) | $0.10$ - $0.20$ 범위 | **1.6 - 2.3** ← *우리 도달점* |
| Reference: 학습 길게 했을 때 | $0.25$+ | 약 1.4 |

> Ch 26 의 시작 loss `약 8.3` (random baseline) 와 Ch 27 의 시작 loss `약 2.5-3.0` (사전학습된 본체의 평가 loss) 의 차이가 *대규모 한국어 사전학습이 본체에 미리 새겨둔 next-token 분포* 의 정량적 가치. *Ch 27 은 random 에서 시작하지 않습니다*.

### Perplexity 환산

$\text{PPL} = e^{L}$:

| CLM loss | PPL | 해석 |
|---|---|---|
| 10.84 | 51,200 | 균등 추측 (51K vocab 전체) |
| 3.0 | 20 | 약 20 개 후보 ← Ch 27 시작 영역 |
| 2.0 | 7.4 | 약 7 개 후보 ← Ch 27 도달 영역 |
| 1.4 | 4.1 | 거의 결정적 |

> *vocab 51K 의 거대한 공간에서 평균 7-20 개 후보로 좁힌* 상태에서 시작해 더 좁힙니다. Ch 26 의 vocab 약 4,000 과 정량적 비교는 어렵지만 (vocab 단위가 다름), *generation 품질* 로는 직접 비교 가능 — 그게 본 챕터 §7 (3-way 비교) 의 역할.

## 토크나이저 노트 — KoGPT2 BBPE 그대로 (vocab 51,200)

본 챕터에서는 토크나이저를 *학습하지 않습니다*. `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` 로 SKT 가 대규모 한국어 코퍼스 위에 학습해 둔 byte-level BPE (BBPE, vocab 51,200) 를 그대로 가져옵니다 (special token 명시 필요 — 바로 아래 주의 참고).

### 왜 직접 학습하지 않는가 — *토크나이저는 본체와 운명공동체*

KoGPT2 본체의 input embedding `wte` (51200 × 768) 와 LM head (768 × 51200) 는 *KoGPT2 가 학습한 그 vocab id 체계* 에 맞춰 학습되어 있습니다. 만약 *다른 vocab* (예: Ch 26 의 직접 학습 BBPE 약 4,000) 을 붙이면:

- token id `100` 이 *KoGPT2 가 학습한 토큰* 과 *완전히 다른 byte 조각* 을 가리킴
- `wte[100]` 의 vector 는 *KoGPT2 가 학습한 token 100* 의 의미인데, *우리가 붙인 token 100* 은 무관한 byte
- 결과: 본체 weight 가 *유효한 신호가 아님*. 사실상 random init 과 같은 상태에서 시작

따라서 **사전학습 모델을 가져올 때는 그 모델이 학습한 토크나이저를 *반드시 함께* 가져와야** 합니다. Ch 19 에서 다룬 *토크나이저는 모델과 운명공동체* 원칙의 정확한 적용 사례 — 영어 Ch 25 (gpt2 BPE 그대로) 의 한국어 재확인.

### Ch 26 ↔ Ch 27 토크나이저 비교

| 항목 | Ch 26 | **Ch 27 (본 챕터)** |
|---|---|---|
| 알고리즘 | byte-level BPE (BBPE) | byte-level BPE (BBPE) (같은 종류) |
| Vocab 크기 | 약 4,000 | **51,200** (약 13배) |
| 학습 코퍼스 | 한국어 TinyStories 30K (약 4-6M 토큰) | **대규모 한국어 코퍼스** (SKT 가 학습) |
| 학습 주체 | 본 챕터에서 직접 학습 (Ch 26) | **SKT 가 미리 학습** (그대로 사용) |
| 특수 토큰 | `<|endoftext|>` (bos = eos = pad) | KoGPT2 컨벤션 (`</s>` 등, pad 는 별도 지정 필요할 수 있음) |

### KoGPT2 토크나이저 로드 주의 — `AutoTokenizer` 가 잘못 fallback 합니다

KoGPT2 (`skt/kogpt2-base-v2`) 는 **`AutoTokenizer.from_pretrained(...)` 가 영어 GPT2 토크나이저로 잘못 fallback** 하는 *알려진 함정* 이 있습니다. 그러면 special token 이 `<|endoftext|>` 로 잡히고, 한국어가 *완전히 깨진 토큰* 으로 인코딩됩니다 (예: `"옛날 옛날에"` → `[501, 500, ...]` → `'�����'`).

SKT 공식 방식대로 **`PreTrainedTokenizerFast` + special token 명시** 로 로드해야 합니다:

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
# 이렇게 로드하면 "옛날 옛날에" -> [12346, 35970] -> "옛날 옛날에" 정상 왕복
```

> 실무 교훈: *사전학습 모델마다 권장 토크나이저 로드 방식이 다를 수 있습니다.* 모델 카드의 example code 를 확인하고, *encode → decode 왕복* 으로 한 번 검증하는 습관이 이런 함정을 막습니다.

EOS 토큰을 pad 로 재활용. `group_texts` 패턴에서 chunk 길이가 모두 같으면 pad 가 거의 없어 실용적으로는 영향 없음. (영어 Ch 25 에서 `tokenizer.pad_token = tokenizer.eos_token` 했던 것과 같은 패턴 — 다만 KoGPT2 는 이미 pad 가 지정돼 있을 수도 있어 `if None` 가드.)

### 한국어를 *제대로* 다루는 vocab

Ch 26 에서 봤듯, *영어 gpt2 BPE 로 한국어를 토큰화하면 한글이 byte 조각으로 잘게 쪼개져 토큰 수가 폭증* 합니다. KoGPT2 BBPE 는 *한국어 코퍼스 위에 학습* 되어 한국어 어절을 *의미 있는 토큰* 으로 압축합니다 — 그래서 영어 gpt2 를 한국어에 쓰는 대신 *한국어 사전학습 모델 KoGPT2* 를 가져오는 게 정공법. Ch 26 (한국어는 scratch) 와 Ch 27 (한국어 사전학습 본체) 이 *영어 Ch 24-25 와 정확히 같은 대칭* 을 이루는 이유.

## 환경 셋업

## 한국어 TinyStories 데이터 로드 + story 복원 — *Ch 26 과 완전히 동일*

본 챕터의 데이터는 *통제 변수*. Ch 26 과 정확히 같은 방식으로 `g0ster/TinyStories-Korean` 을 로드합니다. 이 데이터셋은 *story 단위가 아니라 줄(line) 단위* 로 저장되어 있어, *`<|endoftext|>` 를 만날 때까지 줄을 이어 붙여* 한 story 로 복원합니다. 그렇게 복원한 처음 **30,000 stories** 만 사용 (Ch 26 과 같은 규모, T4 30분 룰 안). *데이터를 고정하고 본체·토크나이저·lr 만 바꿔 격차를 본다* 가 본 챕터의 격리 실험 설계.

## KoGPT2 토크나이저·모델 로드 — *모델 로드 한 줄로 학습 단계 2 진입*

본 챕터의 *유일한 큰 변화*. Ch 26 의 `GPT2LMHeadModel(config)` random init 대신 `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` 한 줄. 토크나이저도 같이 가져옵니다. 영어 Ch 25 의 `gpt2` 로드와 정확히 같은 패턴 — 모델 id 만 한국어 KoGPT2.

### Ch 26 ↔ Ch 27 코드 diff — *모델·토크나이저 로드 두 줄 차이*

```python
# Ch 26 (한국어 GPT scratch) - BBPE 직접 학습 후 random init 모델
# bbpe = Tokenizer(BPE(unk_token=None))
# trainer = BpeTrainer(vocab_size=4000, ...)
# bbpe.train_from_iterator(text_iter, trainer)
# tokenizer = PreTrainedTokenizerFast(tokenizer_object=bbpe, bos_token=EOS, eos_token=EOS, pad_token=EOS)
# config = GPT2Config(vocab_size=4000, n_layer=4, n_head=4, n_embd=256, ...)
# model = GPT2LMHeadModel(config)

# Ch 27 (continual pretraining) - 단 몇 줄로 (KoGPT2 는 토크나이저 로드만 주의)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
```

> *trainer·collator·loss 는 같음* — *모델 로드 한 줄 + 토크나이저 로드* 로 학습 단계 2 (continual pretraining) 에 진입합니다. 그게 본 챕터의 메시지. 영어 Ch 25 의 `from_pretrained("gpt2")` 와 같은 구조 — 다만 **KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정** 이 있어 `PreTrainedTokenizerFast` + special token 명시가 필요합니다 (실무에서 자주 만나는 *사전학습 모델별 토크나이저 로드 주의점*).

## 토큰화 + `group_texts` — *Ch 26 과 완전히 같은 패턴*

HF causal LM 학습 표준 패턴 (`run_clm.py`) 그대로. Ch 26 과 정확히 같습니다 — *데이터·전처리·collator 는 통제 변수*.

다만 `BLOCK_SIZE` 는 Ch 26 과 동일하게 유지 (128) — *KoGPT2 본체의 `n_positions=1024` 까지 가능하지만, T4 + 30분 룰 안에서 비교 가능성 우선*.

**비교 관전 포인트** — 같은 30K stories 가 *KoGPT2 BBPE (vocab 51,200)* 로 토큰화되면 Ch 26 의 *직접 학습 BBPE (vocab 약 4,000)* 보다 *토큰 수가 적습니다* — vocab 이 클수록 한 토큰이 더 긴 byte 시퀀스를 표현하므로. 같은 데이터의 토큰 수 차이가 *토크나이저 vocab 크기의 직접적 효과* (영어 Ch 24→25 에서 본 결의 한국어 재확인).

## 학습 *전* generation — *이미 잘 만들어진 한국어 본체* 라는 사실 확인

Ch 26 의 *random init baseline* 은 *한국어와 거리 먼 byte 조각* 이었습니다. Ch 27 의 학습 전 baseline 은 *KoGPT2 가 대규모 한국어 코퍼스로 이미 사전학습된 본체* 라 *학습 시작 시점에 이미 자연스러운 한국어 generation* 이 가능합니다 (다만 TinyStories 동화체는 아님).

같은 한국어 prompt 3개로 *KoGPT2 학습 직전 (BEFORE)* generation 을 기록 — 학습 후 (§6) 와 나란히 비교해 *continual pretraining 이 본체에 어떤 변화를 주는가* 를 직접 봅니다.

**해석 가이드 — *Ch 26 random init* vs *Ch 27 KoGPT2 사전학습* 의 직전 비교**

- **Ch 26 학습 직전 (random init)**: *한국어와 거리 먼 음절·byte 조각 / 의미 없는 짧은 토큰 반복*
- **Ch 27 학습 직전 (KoGPT2 사전학습 그대로)**: *이미 자연스러운 한국어 문장* — *주어 + 서술어* 구조, 다양한 도메인 어휘. 다만 *TinyStories 동화 풍은 아님* — 일반 한국어 산문 / 뉴스 / 대화 등 (사전학습 데이터 분포 반영)

> 이 차이가 *학습 시작점의 차이*. Ch 27 은 *random 에서 시작하지 않습니다* — *이미 잘 만들어진 본체* 에서 시작해 *TinyStories 동화 풍 적응* 만 더하는 게 학습 단계 2 (continual pretraining) 의 본질. 영어 Ch 25 (gpt2 BEFORE) 의 한국어판.

## Continual Pretraining — *trainer 코드는 Ch 26 과 거의 동일*

Ch 26 과 *완전히 같은 구조* 의 `Trainer` 코드. 변하는 곳은 **lr (`5e-4 → 2e-5`)** 와 **step 설정 (max_steps → 1 epoch, 48,513 chunks / eff. batch 16 ≈ 약 3,000 step)**, 그리고 **메모리 제약 대응 (batch 4 + grad accum 4)** 입니다.

### 왜 lr 가 작아지는가 — `2e-5` 의 정확한 의미

Ch 26 (scratch) 의 lr `5e-4` 는 *random init 본체* 가 *빠르게 의미 있는 표상* 을 학습하기 위한 표준 값. Ch 27 (continual pretraining) 는 *이미 학습된 본체* 라 *큰 lr 면 사전학습된 표상이 망가질 위험* — **catastrophic forgetting**. `2e-5` 는 HF 의 continual pretraining / fine-tuning 표준 lr 중 가장 작은 쪽으로, *사전학습 표상 보존* 을 우선. (영어 Ch 25 와 같은 `2e-5`.)

### `DataCollatorForLanguageModeling(mlm=False)` — *Ch 26 과 한 글자도 다르지 않음*

학습 단계 2 의 정의: *collator 안 바뀜, loss 안 바뀜, trainer 안 바뀜*. *데이터·본체·lr 만 바뀜*.

**관전 포인트** — Ch 26 과 달리 *첫 step loss 가 random baseline `ln(51200) ≈ 10.84` 부근이 아니라 약 3.0-4.0 부근* 에서 시작합니다. *KoGPT2 가 이미 일반 한국어 분포를 학습해 둔 덕분에 TinyStories 평가에서도 시작 loss 가 낮음*. 학습 진행과 함께 약 2.0-2.5 로 더 떨어지는데, 이게 *TinyStories 동화 도메인 적응* 의 효과. 곡선이 *random baseline 으로부터 빠르게 떨어지는 Ch 26* vs *이미 낮은 지점에서 시작해 천천히 더 떨어지는 Ch 27* 의 모양 차이가 한눈에 보입니다 (영어 Ch 24→25 와 같은 결).

## 학습 *후* generation — *continual pretraining 의 효과*

같은 `PROMPTS / GEN_KWARGS` 로 학습 후 모델에서 다시 생성. *BEFORE (KoGPT2 그대로) → AFTER (continual pretrained on TinyStories-Korean)* 비교가 *학습 단계 2 가 본체에 새긴 도메인 적응* 을 직접 드러냅니다.

**해석 가이드 — continual pretraining 의 도메인 적응 효과**

- **BEFORE (KoGPT2 그대로)**: 자연스러운 한국어이지만 *일반 도메인 풍* — 산문 / 뉴스 / 대화 톤. *옛날 옛날에* 같은 동화 도입에 대해서도 *동화 스타일 이어쓰기보다 일반 산문 이어쓰기* 경향
- **AFTER (KoGPT2 + TinyStories-Korean 1 epoch)**: 같은 prompt 가 *동화 풍* 으로 이어짐 — 짧고 단순한 문장, 동화 어휘 (소녀 / 엄마 / 친구 / 숲 / 행복 / 토끼 ...), TinyStories 특유의 *반복적이고 어린이 어휘 한정* 톤

> 본체는 *같은 125M params 모델* 이고, *한 줄 코드 차이 (lr) + 한 epoch 의 데이터* 만으로 *generation 톤 자체가 도메인 적응*. 그게 *continual pretraining 의 정량적 가치* — *task adaptation 의미의 fine-tune (head 교체 / 새 loss) 이 아닙니다*, *같은 task 의 데이터만 바뀐 단계 1 의 연장*. 영어 Ch 25 (gpt2 AFTER) 의 한국어 재확인.

## 3-way generation 비교 — Ch 26 (scratch) vs Ch 27 BEFORE vs Ch 27 AFTER

Ch 26 의 *작은 from-scratch 모델* (약 3M, 한국어 TinyStories 1500 step) 의 generation 결과를 *옆에 두고* 비교합니다. *Ch 26 노트북 §7 의 "TRAINED model" generation 출력* 을 직접 인용 (사용자가 본인 결과로 갱신 가능).

### 세 셋업의 차이

| 셋업 | 본체 | 사전학습 | TinyStories 학습 |
|---|---|---|---|
| Ch 26 (scratch) | 약 3M params, random init | 없음 (from scratch) | 1500 step 사전학습 자체 |
| **Ch 27 BEFORE** | 125M params (KoGPT2) | **대규모 한국어 코퍼스** | 없음 (KoGPT2 그대로) |
| **Ch 27 AFTER** | 125M params (KoGPT2) | **대규모 한국어 코퍼스** | **1 epoch continual pretraining** |

**해석 가이드 — 세 셋업의 격차**

- **Ch 26 (3M scratch, 한국어 TinyStories 1500 step)**: *동화 풍 단순 한국어* 가능 — 작은 모델·작은 데이터로도 말이 되는 한국어 생성. 다만 어휘는 동화 도메인에 한정, 번역체 데이터라 다소 어색할 수 있음
- **Ch 27 BEFORE (KoGPT2 그대로)**: *다양한 도메인 한국어* 가능. 자연스러운 산문이지만 *TinyStories 동화 풍은 아님*
- **Ch 27 AFTER (KoGPT2 + TinyStories-Korean continual pretrain)**: *동화 풍 + 자연스러움 + 일반 도메인 어휘력* 결합. *작은 from-scratch 의 도메인 특화 + 큰 사전학습 모델의 어휘 폭* 이 모두

> **세 셋업의 비교가 던지는 질문** — Ch 27 AFTER 가 Ch 26 보다 *훨씬 좋아 보인다면*, 이게 *모델 크기 (3M → 125M, 약 40배) 의 위력인가, 사전학습 (대규모 한국어 코퍼스) 의 위력인가?* — 본 챕터의 셋업으로는 *분리 불가능*. 두 요인이 *함께 변함*. FAQ Q3 에서 더 자세히 (영어 Ch 25 Q3 의 한국어판).

## 학습 곡선 비교 — Ch 26 vs Ch 27 의 학습 효율

*같은 데이터 (한국어 TinyStories 30K)* 에 대한 *random init vs 사전학습 본체* 의 학습 효율 격차를 표로 정리.

| 항목 | Ch 26 (3M scratch) | **Ch 27 (125M continual pretrain)** |
|---|---|---|
| 시작 loss | 약 8.29 (`ln(4000)`, random baseline) | **약 3.0-4.0** (KoGPT2 pretrained, TinyStories 평가) |
| 도달 loss (학습 끝, 누적 평균 `train_loss`) | 약 4.5 | **약 2.49** |
| 학습 step | 1,500 | **약 3,000** (1 epoch, 48,513 chunks / eff. batch 16) |
| 학습 시간 (T4) | 약 1분 | **약 17분** |
| Vocab 차원 | 약 4,000 | **51,200** (loss 단위 다름 — 직접 비교 어려움) |
| Generation 품질 | 동화 풍 단순 한국어 | **자연스러운 동화 + 일반 도메인 어휘** |

> **요점**: Ch 27 은 *시작부터 낮은 loss* 에서 출발해 더 낮은 지점까지 내려갑니다 — 사전학습된 본체의 *시작 이점*. step·시간은 Ch 26 보다 오히려 더 큽니다 (125M 본체 + chunk 수가 많아 1 epoch 이 길어짐). 다만 *loss 의 절대값* 은 vocab 단위가 달라 직접 비교 어려움 (vocab 약 13배 차이). *Generation 품질* 로는 §7 의 3-way 비교가 정성적 차이를 보여줍니다.

> Ch 27 의 결과만 보면 *대규모 사전학습 + continual pretraining* 이 압도적으로 보이지만, *3M params + 대규모 사전학습* (가상의 비교군) 이라면 어떻게 될까요 — *모델 크기와 사전학습 데이터를 분리하는 비교* 는 본 챕터의 셋업으로는 어렵습니다. 그게 *실험 설계의 한계* 이자 *학습 단계 2 의 실용성* — 실무는 보통 *큰 사전학습 모델을 그대로 가져와 continual pretraining* 하는 게 비용 대비 최선이라 (영어 Ch 25 §8 의 한국어판).

## 이 장의 구성

- [27-1. 실습](27-ko_gpt2_continual_pretrain-practice.md)
- [27-2. 변형 — 더 많은 epoch / 다른 도메인 / catastrophic forgetting 시연](27-ko_gpt2_continual_pretrain-variation.md)
- [27-3. 정리와 FAQ](27-ko_gpt2_continual_pretrain-wrapup.md)
