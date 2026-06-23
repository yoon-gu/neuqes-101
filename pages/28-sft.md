**목표**: Phase 4 의 *학습 단계 3 (SFT, Supervised Fine-Tuning / Instruction Tuning)* 챕터. Ch 27 에서 *KoGPT2 본체를 한국어 TinyStories 로 continual pretraining* 했다면, 이번엔 같은 **KoGPT2 (`skt/kogpt2-base-v2`, 125M)** 본체를 **KoAlpaca instruction-response 쌍 데이터** 로 **SFT** 합니다. 본체도 같고, loss 종류 (next-token `CrossEntropyLoss`) 도 같습니다. 바뀌는 건 **데이터 형식 (연속 텍스트 → instruction-response 쌍)** + **`labels` 마스킹 (pad 만 → prompt 부분 전체)** + **trainer (`Trainer` → `SFTTrainer`)** 입니다. *그 한 줄 `labels[:prompt_len] = -100` 이 모델을 지시를 따르게 만듭니다*.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 10-15분 (KoAlpaca 로드·포맷 약 2분 + KoGPT2 로드 약 2분 + collator labels 마스킹 시각화 약 1분 + SFT 학습 약 2-3분 (3,000 샘플 1 epoch, 188 step) + SFT 전·후 instruction following 비교 약 3분)


## 학습 흐름

1. 📊 **누적 추적표** (Ch 25/26/27 + **28 강조** + Ch 29 예고) + GPT 학습 4단계 표 (Ch 28 = 단계 3)
2. 🔄 **변경점 (Diff from Ch 27)** — *데이터 형식 + trainer + labels 마스킹* 만 변함. 본체·토크나이저·loss 종류는 그대로
3. 🎯 **`labels = -100` thread 완성 표** — MLM 15% / CausalLM 거의 전부 / SFT 답변만. *세 단계를 한 화면에*. 이 챕터가 thread 의 종착점
4. ⚠️ **파인튜닝 의미 변화 완성** — BERT task head vs GPT SFT behavior alignment. *진짜* instruction following
5. 📐 **Loss** — next-token CE 동일, 단 *어느 자리에서 계산하는가* 가 핵심 변화 (response-only)
6. 🔤 **토크나이저 노트** — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 방식). instruction 포맷 토큰화 + response_template 위치
7. 🚀 **실습**: KoAlpaca 로드 → KoGPT2 로드 → **collator labels 마스킹 직접 시각화 (클라이맥스)** → `SFTTrainer` 학습 → SFT 전·후 instruction following 비교
8. 📦 **등장 라이브러리** (`trl.SFTTrainer` 첫 등장) / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)


> 📒 **사전 학습 자료**: Ch 27 (KoGPT2 continual pretraining — 본 챕터와 *같은 본체*), Ch 24-26 (GPT 사전학습), Ch 20-22 (MLM 의 `labels = -100`). 본 챕터는 Phase 4 의 두 thread (`labels = -100` 자리 / "파인튜닝" 의미 변화) 의 *클라이맥스*. **`### 응답:` 뒤만 학습한다** 는 한 줄이 *왜 GPT 하나가 모든 task 를 해내는가* 의 답입니다.

## 누적 추적표

| Ch | 모델 | 토크나이저 | 데이터 | `labels = -100` 자리 | Loss |
|---|---|---|---|---|---|
| 25 | `gpt2` (124M, 사전학습) | BPE (gpt2 그대로, vocab 50,257) | 영어 TinyStories 30K | pad 만 | `CrossEntropyLoss` (next-token) - continual pretraining |
| 26 | 작은 GPT2 (한국어, 약 3M, scratch) | BBPE (직접 학습, vocab 약 4,000) | 한국어 TinyStories 30K | pad 만 | `CrossEntropyLoss` (next-token) |
| 27 | KoGPT2 `skt/kogpt2-base-v2` (125M) | BBPE (KoGPT2 그대로, vocab 51,200) | 한국어 TinyStories 30K | pad 만 | `CrossEntropyLoss` (next-token) - continual pretraining |
| **28 ← 여기** | **KoGPT2 `skt/kogpt2-base-v2` (125M, 동일)** | **BBPE (KoGPT2 그대로, vocab 51,200, 동일)** | **KoAlpaca instruction-response 쌍 (약 3K, 3,000 샘플)** | **prompt 부분 (`### 응답:` 앞 전부)** | **`CrossEntropyLoss` (next-token, *답변 부분만*) — SFT** |
| 29 (다음) | Ch 28 SFT 모델 + 비교 | (동일) | 분야별 벤치마크 (KMMLU / HAERAE / MMLU ...) | - (평가만) | - (`lm-evaluation-harness`) |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.


## GPT 시대 학습 4단계 — 본 챕터의 위치 (단계 3, SFT)

Ch 24 에서 도입한 GPT 시대 학습 4단계 표. 본 챕터는 *단계 3 (SFT)* — *행동 정렬* 이 처음 일어나는 단계입니다.

| 단계 | 정확 용어 | 의미 | `labels = -100` 자리 | 본 커리큘럼 | 본 챕터? |
|---|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | random init 본체 + 일반 코퍼스 | pad 만 | Ch 24 (영어), Ch 26 (한국어) | |
| 2 | **Continual pretraining** (계속 사전학습) | 사전학습된 본체 + 새 데이터, *같은 CausalLM task* | pad 만 | Ch 25 (영어), Ch 27 (한국어) | |
| 3 | **SFT** (Supervised Fine-Tuning / Instruction tuning) | instruction-response 쌍으로 *행동 정렬*. *답변 부분만* 학습 | **prompt 부분** | **Ch 28 ← 여기** | ✅ |
| 4 | **Alignment** (DPO / RLHF / GRPO) | preference 또는 verifier reward 로 *선호 정렬* | (RL 내부, response 부분만) | Ch 30 (DPO), Ch 31 (GRPO) | |

### 단계 2 (Ch 27) → 단계 3 (Ch 28) 의 결정적 변화

- **단계 2 (continual pretraining)**: *연속된 일반 텍스트* 를 *거의 모든 자리* 에서 학습. 모델은 *다음 토큰 분포* 를 도메인에 맞게 다듬을 뿐 — *지시를 따르는 법* 은 배우지 않음
- **단계 3 (SFT)**: *instruction-response 쌍* 에서 *답변 토큰만* 학습. 모델은 *주어진 지시에 어떻게 응답하는가* 를 배움 — **행동 정렬 (behavior alignment)**

> *같은 본체, 같은 loss 종류, 단 데이터 형식 + 마스킹 자리만 바뀌어* 모델이 *지시를 따라가게* 됩니다. 그 한 줄이 `labels[:prompt_len] = -100`. 본 챕터는 그 한 줄을 *눈으로 확인* 하는 챕터입니다.

## 변경점 (Diff from Ch 27)

| 축 | Ch 27 (KoGPT2 continual pretraining) | Ch 28 (본 챕터, KoGPT2 SFT) |
|---|---|---|
| 본체 | KoGPT2 `skt/kogpt2-base-v2` (125M) | **KoGPT2 `skt/kogpt2-base-v2` (125M, 동일)** ← 고정 |
| 토크나이저 | `PreTrainedTokenizerFast` (KoGPT2 BBPE, vocab 51,200) | **(동일)** ← 고정 |
| Loss 종류 | next-token `CrossEntropyLoss` | **next-token `CrossEntropyLoss` (종류 동일)** ← 고정 |
| **데이터 형식** | 연속된 일반 텍스트 (TinyStories) | **instruction-response 쌍 (KoAlpaca)** ← *변화 1* |
| **Trainer** | `transformers.Trainer` | **`trl.SFTTrainer`** ← *변화 2* (새 클래스, 첫 등장) |
| **`labels = -100` 자리** | pad 만 (거의 모든 자리 학습) | **prompt 부분 전체 (`### 응답:` 앞)** ← *변화 3, 핵심* |
| 효과 | 도메인 적응 (동화 풍) | **instruction 따라가기 (행동 정렬)** ← 메시지 |
| lr | 2e-5 | 2e-5 (SFT 표준, 동일 범위) |

> **핵심**: *본체·토크나이저·loss 종류는 그대로*. 바뀌는 건 *데이터 형식 + trainer + 마스킹 자리* 세 가지. 그중에서도 **`labels` 마스킹 자리** 가 *왜 모델이 instruction 을 따라가게 되는가* 의 진짜 원인입니다. Ch 27 의 collator 가 *거의 모든 자리* 를 학습했다면, Ch 28 은 *prompt 를 전부 가리고 답변만* 학습합니다 — *정확히 정반대 자리*.

## `labels = -100` thread 의 완성 — 세 단계를 한 화면에

커리큘럼 전체를 관통하는 thread 의 *종착점* 입니다. `labels = -100` 은 *그 자리를 loss 에서 제외* 하라는 의미 (`CrossEntropyLoss(ignore_index=-100)`). **어느 자리를 -100 으로 두느냐가 곧 모델이 무엇을 학습하느냐** 를 결정합니다.

| 단계 | 챕터 | task | `labels = -100` 자리 | loss 계산 자리 | 모델이 배우는 것 |
|---|---|---|---|---|---|
| **MLM** | Ch 20·21·22 | 양방향 빈칸 채우기 | 선택 안 된 약 85% (= 안 가린 자리) | **선택된 약 15% (가린 자리) 만** | 문맥으로 *가려진 단어* 복원 |
| **CausalLM** | Ch 24·25·26·27 | 다음 토큰 예측 | **pad 만** (거의 없음) | **거의 전 토큰** | *다음에 올 토큰* 분포 |
| **SFT (본 챕터)** | **Ch 28** | instruction 따라가기 | **prompt 부분 전체** (`### 응답:` 앞) | **답변 토큰만** | *지시에 대한 응답* 생성 |

### 세 단계를 한눈에 — *같은 `-100`, 다른 자리*

```
MLM (Ch 21):     [the] [MASK] [sat] [on] [the] [MASK]
labels:          -100   cat   -100  -100 -100   mat       <- 가린 15% 만 학습

CausalLM (Ch 27): [옛날] [옛날에] [작은] [소녀가] [살았어요]
labels:           옛날에  작은    소녀가  살았어요   <eos>      <- 거의 전부 학습 (shift)

SFT (Ch 28):     [### 명령어:] [피보나치 설명] [### 응답:] [피보나치는] [수열입니다]
labels:           -100  -100   -100  -100  -100   -100   피보나치는  수열입니다   <- 답변만 학습
```

> **MLM 은 일부 (15%) 만 가리고**, **CausalLM 은 거의 안 가리고**, **SFT 는 prompt 전부 가립니다**. 세 task 모두 *같은 `CrossEntropyLoss(ignore_index=-100)`* 를 쓰지만 *-100 의 자리* 가 다릅니다. **Ch 28 에서 이 thread 가 완성됩니다** — 아래 §3 에서 KoGPT2 의 실제 collator 출력을 print 로 *눈으로 확인* 합니다 (Ch 21 의 `[MASK]` 80/10/10 시각화의 SFT 판).

핵심 메시지: **모델이 instruction 을 "따라간다" 는 것은, instruction 토큰 자체는 학습하지 않고 그에 대한 *응답만* 학습한다는 의미**. 만약 prompt 도 함께 학습하면 모델은 *질문 자체를 외우는* 쪽으로 기웁니다 — 우리가 원하는 건 *질문에 답하는 법* 입니다. 그 차이가 한 줄 `labels[:prompt_len] = -100` 입니다.

## "파인튜닝" 의미 변화의 완성 — task head → behavior alignment

커리큘럼 전체에서 *fine-tune* 이라는 단어는 *세 의미* 로 쓰였습니다. Ch 28 이 그 세 번째 의미 (*행동 정렬*) 의 도착점입니다.

| 의미 | 무엇이 바뀌나 | 챕터 | 비유 |
|---|---|---|---|
| **① task adaptation** (BERT 파인튜닝) | 본체 + **새 head** (`Linear(H, K)`) + **새 task loss** | Ch 9-23 (분류) | *새 도구* 를 손에 붙임 |
| **② 데이터 적응** (GPT continual pretraining) | head 그대로, **같은 next-token task**, *새 데이터* | Ch 25·27 | *같은 도구* 로 *새 재료* 연습 |
| **③ 행동 정렬** (GPT SFT) | head 그대로, **같은 next-token CE**, *instruction 형식 데이터* + *prompt 마스킹* | **Ch 28 ← 여기** | *도구는 그대로*, *지시를 따르는 법* 을 깨움 |

### BERT 파인튜닝 (①) vs GPT SFT (③) — *진짜* instruction following

- **BERT 파인튜닝 (①)**: task 마다 *다른 head* 를 붙입니다. 감정분류 head, NER head, QA head... *task 하나당 모델 하나*. head 가 task 를 정의
- **GPT SFT (③)**: *head 는 LM head 하나 그대로*. *입력 프롬프트 형식* 만 바꾸면 *같은 모델* 이 번역도, 요약도, 질의응답도 합니다. **task 가 head 가 아니라 prompt 에 인코딩됨**

> *왜 GPT 하나가 모든 task 를 해내는가?* — 답은 SFT 입니다. 본체는 *입력 프롬프트만 바꾸면 다른 일* 을 하도록 *행동 정렬* 되어 있습니다. **SFT 가 그 능력을 깨우는 단계**. BERT 가 *task 마다 head 를 갈아끼우던* 시대에서, GPT 가 *prompt 하나로 모든 task* 를 하는 시대로의 전환 — 그 전환점이 본 챕터입니다.

이게 *진짜* behavior tuning 입니다 — *task 적응 (①)* 도, *데이터 적응 (②)* 도 아닌, *모델의 행동 자체* 를 instruction 을 따르도록 정렬하는 단계.

## Loss — next-token CE 동일, 단 *어느 자리에서 계산하는가*

본 챕터의 loss 는 Ch 27 과 *같은 종류* — next-token `CrossEntropyLoss`:

$$L_{\text{CLM}} = -\sum_{i} \log P(x_{i+1} \mid x_{\leq i})$$

다른 점은 **어느 위치 $i$ 에서 합산하느냐** 입니다. SFT 는 *답변 부분 토큰만* 합산합니다:

$$L_{\text{SFT}} = -\sum_{i \in \text{response}} \log P(x_{i+1} \mid x_{\leq i}) \qquad (\text{prompt 부분은 } -100 \text{ 으로 제외})$$

### 왜 prompt 도 학습하면 안 되나 — 숫자로 감 잡기

instruction `"### 명령어:\n2+2 는?\n\n### 응답:\n"` 뒤에 답변 `"4 입니다."` 가 오는 한 샘플을 생각해 봅시다. 토큰이 *prompt 12개 + 답변 4개* 라고 하면:

| 학습 방식 | loss 합산 자리 | 모델이 강화하는 것 |
|---|---|---|
| **전체 학습** (prompt 포함) | 16개 토큰 전부 | *"### 명령어:" → "2+2 는?"* 같은 *질문 자체의 패턴* 까지 외움 |
| **response-only** (본 챕터) | 답변 4개만 | *prompt 가 주어졌을 때 답변* 하는 능력만 |

전체 학습을 하면 loss 의 *대부분* 이 *prompt 토큰* 에서 나옵니다 (prompt 가 보통 더 김). 그러면 모델은 *질문을 받아쓰는 데* gradient 를 낭비합니다. 우리가 원하는 건 *질문을 외우는 게 아니라 답하는 법* — 그래서 prompt 를 `-100` 으로 가립니다.

### response-only 의 직관

| 토큰 위치 | label | loss 기여 | 의미 |
|---|---|---|---|
| `### 명령어:` ... `### 응답:` (prompt) | `-100` | **0** (제외) | *주어진 조건* — 외울 필요 없음 |
| `4` `입니다` `.` `<eos>` (답변) | 원본 token id | **포함** | *이걸 생성하는 법* 을 학습 |

> *prompt 는 조건 (given), 답변은 학습 대상 (target)*. 이 구분이 SFT 의 전부입니다. `trl` 의 collator 가 `### 응답:` 위치를 찾아 그 *앞을 전부 -100* 으로 만듭니다 — §3 에서 직접 봅니다.

## 토크나이저 노트 — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 방식 그대로)

본 챕터의 토크나이저는 *Ch 27 과 완전히 동일*. KoGPT2 BBPE (vocab 51,200) 를 그대로 가져옵니다. **단 KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정** 이 있어 (Ch 27 §토크나이저 노트), `PreTrainedTokenizerFast` + special token 명시로 로드합니다.

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
```

### instruction 포맷 + response_template 위치

KoGPT2 는 *chat template 이 없습니다* (instruction-tuned 모델이 아니므로). 그래서 instruction-response 를 *직접 포맷* 합니다:

```
### 명령어:
{instruction}

### 응답:
{output}
```

여기서 **`### 응답:\n`** 가 **response_template** — *이 문자열 이후부터가 답변* 이라는 경계 표시입니다. `trl` collator 는 이 경계를 기준으로 *앞은 prompt (= -100), 뒤는 답변 (= 학습)* 으로 나눕니다.

### 같은 문장이 어떻게 토큰화되는가

instruction 포맷 `### 명령어:\n피보나치 설명\n\n### 응답:\n` 을 KoGPT2 BBPE 로 토큰화하면:

- `###` → `#`·`#`·`#` (3 토큰), `명령어` → `명령`·`어` (2 토큰), `:` → 1 토큰, `\n` → 1 토큰 ...
- 한국어 어절 (`피보나치`, `설명`) 은 KoGPT2 가 한국어 코퍼스로 학습한 *의미 있는 토큰* 으로 압축

> **response_template `### 응답:\n` 자체도 토큰 시퀀스** 입니다. collator 는 이 *토큰 시퀀스* 를 input_ids 안에서 찾아 그 *직후 위치* 부터 답변으로 간주합니다. 그래서 response_template 은 *데이터에 일관되게 등장하고, 본문과 충돌하지 않는* 문자열이어야 합니다 (`### 응답:` 처럼 특수한 마커가 적합).

다음 챕터 (Ch 29 벤치마크 평가) 에서도 *같은 KoGPT2 토크나이저* 를 그대로 사용합니다 — 토크나이저는 Ch 27 이후 고정.

## 환경 셋업

`trl` (Transformer Reinforcement Learning) 라이브러리가 이번 챕터에 새로 등장합니다 — `SFTTrainer` 와 SFT 용 데이터 collator 를 제공. `transformers` / `datasets` / `accelerate` 와 함께 설치합니다.

## KoAlpaca instruction 데이터 로드 + 포맷

**`beomi/KoAlpaca-v1.1a`** — 한국어 instruction tuning 데이터셋. 각 샘플은 `instruction` (지시) 과 `output` (응답) 필드를 가집니다 (`url` 필드는 출처 — 학습에 사용 안 함). T4 + 30분 룰 안에서 **약 3,000 샘플** 만 subset 으로 사용합니다.

KoGPT2 는 chat template 이 없으니 *직접 포맷* — `### 명령어:\n{instruction}\n\n### 응답:\n{output}`. 여기서 **`### 응답:\n` 가 response_template** (답변 시작 경계).

### 포맷 함수 — `prompt` / `completion` 두 컬럼으로

`trl.SFTTrainer` 는 *`prompt` + `completion` 두 컬럼* 형식을 받으면 *completion (답변) 부분만 자동으로 학습 대상* 으로 잡습니다 (`completion_only_loss=True`). 그래서 우리는 instruction 을 prompt 쪽에, output 을 completion 쪽에 넣되, **response_template `### 응답:\n` 까지를 prompt 에 포함** 시켜 *답변 시작 경계* 를 명확히 합니다.

## KoGPT2 토크나이저·모델 로드 — *Ch 27 과 동일한 본체*

본 챕터의 본체는 *Ch 27 과 완전히 같은 KoGPT2*. 토크나이저도 같은 방식 (`PreTrainedTokenizerFast` + special token 명시 — `AutoTokenizer` 함정 회피). encode → decode 왕복으로 한국어가 깨지지 않는지 한 줄 검증합니다.

## collator 의 `labels` 마스킹 직접 시각화 — **이 챕터의 클라이맥스**

여기가 본 챕터의 핵심. `trl` 의 SFT collator 가 한 instruction-response 샘플을 받아 **prompt 부분을 전부 `-100` 으로, 답변 부분만 원본 token id 로** 만드는 것을 *눈으로* 확인합니다. Ch 21 의 `[MASK]` 80/10/10 시각화의 *SFT 판* 입니다.

### 동작 원리

1. `SFTTrainer` 가 *prompt + completion* 을 토큰화해 이어 붙이고, *completion 부분에 1, prompt 부분에 0* 인 `completion_mask` 를 만듭니다 (response_template `### 응답:\n` 가 prompt 의 끝).
2. collator 가 `labels = input_ids.clone()` 한 뒤 *`completion_mask == 0` 인 자리 (= prompt) 를 전부 `-100`* 으로 덮습니다.
3. 그래서 loss 는 *답변 토큰에서만* 계산됩니다 — `labels[:prompt_len] = -100` 의 효과.

**무엇을 보고 있나** — 위 표의 `learn?` 열을 보면:

- **prompt 부분** (`### 명령어:` ... `### 응답:\n` 까지) → `label = -100` → *loss 에서 제외*. 모델은 *이 질문 자체* 를 외우지 않습니다
- **답변 부분** (`### 응답:\n` *이후* 의 모든 토큰 + EOS) → `label = 원본 token id` → *loss 에 포함*. 모델은 *이 답변을 생성하는 법* 만 배웁니다

> Ch 21 의 `[MASK]` 시각화는 *문장의 약 15% 를 가렸다* 면, 여기서는 *prompt 전체를 가립니다* — **정반대 방향의 마스킹**. 그리고 이게 `labels = -100` thread 의 *세 번째이자 마지막 단계*. MLM(15% 만 학습) → CausalLM(거의 전부 학습) → **SFT(답변만 학습)**. 한 줄 `labels[:prompt_len] = -100` 의 효과를 *눈으로 확인* 했습니다.

## `SFTTrainer` 로 SFT 학습 — *새 trainer, 같은 loss 종류*

`trl.SFTTrainer` 는 본 챕터에 처음 등장하는 클래스입니다. `transformers.Trainer` 를 상속해 *SFT 에 특화된 전처리* (prompt/completion 토큰화, EOS 부착, completion 마스킹) 를 자동으로 해 줍니다. 설정은 `SFTConfig` (── `TrainingArguments` 를 상속) 로 주며, **`completion_only_loss=True`** 가 *답변 부분만 학습* 하라는 핵심 옵션입니다.

## 학습 곡선 — *답변 부분에서만 계산된* loss

아래 loss 는 *답변 토큰에서만* 계산된 값입니다 (prompt 는 `-100` 으로 제외). Ch 27 의 loss (거의 모든 자리) 와는 *합산 대상* 이 다르므로 절대값을 직접 비교하지는 않습니다.

## SFT 전·후 instruction following 비교 — *행동 정렬이 일어났는가*

본 챕터의 핵심 데모. *같은 instruction* 을 *SFT 전 (raw KoGPT2)* 과 *SFT 후* 에 각각 넣어 답변을 비교합니다.

- **SFT 전 (raw KoGPT2)**: instruction 을 *지시로 인식하지 못하고* — 질문을 *이어쓰기* 하거나, 블로그·해시태그·SNS 잡담체로 흘러가는 경향, 엉뚱한 방향으로 흐름
- **SFT 후**: instruction 을 *따라* — 질문에 *대답* 하는 구조화된 답변

이 차이가 *행동 정렬 (behavior alignment)* 의 직접 증거입니다.

**해석 가이드 — behavior alignment 의 증거**

- **BEFORE (raw KoGPT2)**: 같은 *125M 본체* 인데도 instruction 을 *지시로 받아들이지 못합니다*. `"피보나치 수열을 설명해줘"` 를 넣으면 *설명* 대신 *질문을 이어 쓰거나*, 일반 산문으로 흘러가거나, 블로그·해시태그·SNS 잡담체로 흘러가는 경향
- **AFTER (KoGPT2 + KoAlpaca SFT)**: *같은 본체* 가 이제 instruction 을 *따라* — 질문에 *대답하는* 구조로 응답. 짧은 SFT (1 epoch, 약 3K 샘플) 만으로도 *행동의 방향* 이 바뀝니다

> **핵심**: 본체는 *한 토큰도 바꾸지 않은 같은 125M KoGPT2* 입니다 (continual pretraining 처럼 *데이터만* 바뀐 게 아니라, *데이터 형식 + 마스킹 자리* 가 바뀌었습니다). 그 결과 *모델의 행동 자체* 가 instruction 을 따르도록 정렬됐습니다. **이게 *왜 GPT 하나가 모든 task 를 해내는가* 의 답** — 입력 프롬프트 형식만 바꾸면 다른 일을 하도록, SFT 가 그 능력을 *깨웠습니다*.

> ⚠️ KoGPT2 는 125M 의 *작은* 모델이고 SFT 데이터·시간도 작아서, 답변 품질 자체는 거칠 수 있습니다 (사실 오류, 반복 등). 본 챕터의 관전 포인트는 *답변의 정확도* 가 아니라 ***instruction 을 따라가는 행동 자체가 생겼는가*** 입니다. 품질은 *더 큰 모델 + 더 많은 데이터 + LoRA* 로 끌어올립니다 (FAQ 참고).

## 이 장의 구성

[[SubPages]]
