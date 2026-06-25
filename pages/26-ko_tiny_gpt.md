**목표**: Phase 4 의 *한국어 단계 1 (pretraining)* 챕터. Ch 24 에서 *영어 작은 GPT (약 3M params) 를 영어 TinyStories 로 from scratch 사전학습* 했다면, 이번엔 **완전히 같은 본체 구조** 로 **한국어 GPT 사전학습** 을 합니다. 변하는 축은 **언어** — 토크나이저는 한국어 코퍼스 위에 직접 학습한 **byte-level BPE (BBPE)**, 데이터는 **`g0ster/TinyStories-Korean`** (영어 TinyStories 의 한국어 번역본). 본체 구조·loss·trainer·hyperparams 는 Ch 24 와 동일. 같은 prompt 에 *학습 전 / 학습 후* generation 을 나란히 비교해 *사전학습이 본체에 어떤 next-token 분포를 새겼는가* 를 한국어로 직접 봅니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 10-15분 (데이터 로드·story 복원 약 3분 + BBPE 토크나이저 학습 약 3분 + 학습 전 generation 약 30초 + 모델 학습 약 1분 + 학습 후 generation 약 2분)

## 학습 흐름

1. 📊 **누적 추적표 + Phase 4 영어·한국어 대칭** — Ch 24(영어 단계1) ↔ Ch 26(한국어 단계1)
2. 🔄 **변경점** (Diff from Ch 24) — 언어 축: 토크나이저 학습 코퍼스 + 데이터만 한국어
3. 🌏 **Phase 4 학습 4단계 표** — Ch 26 = 단계 1 (한국어 pretraining)
4. 📐 **Loss** — Ch 24 와 동일 (CE next-token). vocab 크기 차이로 random baseline `ln V` 미세 변화만
5. 🔤 **토크나이저 노트** — BBPE 직접 학습 (한국어). Ch 24 영어 BPE 와 비교 + Ch 19 연결
6. 🚀 **실습**: 한국어 TinyStories story 복원 → BBPE 직접 학습 → 작은 `GPT2LMHeadModel` random init
7. 🔬 **사전·사후 generation 비교** — 같은 한국어 prompt 3-4개, *학습 전 (random init) vs 학습 후* 나란히
8. 📦 **등장 라이브러리** / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)

> 📒 **사전 학습 자료**: Ch 24 (영어 GPT scratch + TinyStories), Ch 19 (토크나이저 직접 학습), Ch 22 (영어→한국어 BERT 의 *언어 축* 변화 패턴). 본 챕터는 Ch 24 의 셀 구조를 그대로 가져와 *언어 한 축만* 한국어로 바꿉니다. Ch 20(영어 BERT)→Ch 22(한국어 BERT) 가 그랬듯, Ch 24(영어 GPT)→Ch 26(한국어 GPT) 의 한국어 대칭본.

## 변화 추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 23 | 작은 BERT (한국어, scratch) + 분류 head | `klue/bert-base` (가져옴) | NSMC 이진 | `Linear(H, 2)` | `CrossEntropyLoss` |
| 24 | 작은 GPT2 (약 3M, scratch) | BPE (직접 학습, 영어, vocab 2,048) | 영어 TinyStories 30K stories | `Linear(H, V)` (LM head, weight tied) | `CrossEntropyLoss` (next-token) |
| 25 | `gpt2` (124M, OpenAI WebText 사전학습) | BPE (gpt2 그대로, vocab 50,257) | 영어 TinyStories (Ch 24 와 동일) | `Linear(H, V)` (LM head 그대로) | `CrossEntropyLoss` (next-token) - continual pretraining |
| **26 ← 여기** | **작은 GPT2 (약 3M, scratch)** | **BBPE (직접 학습, 한국어, vocab 약 4,000)** | **한국어 TinyStories 30K stories** | **`Linear(H, V)` (LM head, weight tied)** | **`CrossEntropyLoss` (next-token)** |
| 27 (다음) | KoGPT2 (`skt/kogpt2-base-v2`, 125M, 대규모 한국어 사전학습) | KoGPT2 BBPE (그대로) | 한국어 TinyStories (Ch 26 과 동일) | `Linear(H, V)` (LM head 그대로) | `CrossEntropyLoss` (next-token) - continual pretraining |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.

## Phase 4 의 영어·한국어 대칭 — Ch 24 ↔ Ch 26

Phase 4 는 영어 (Ch 24-25) 와 한국어 (Ch 26-27) 가 *같은 학습 단계* 를 *언어만 바꿔* 반복하는 구조입니다. Ch 20(영어 BERT)→Ch 22(한국어 BERT) 의 *언어 축 변화* 가 GPT 에서 그대로 반복됩니다.

| 학습 단계 | 영어 | **한국어** |
|---|---|---|
| **단계 1: Pretraining** (random init → scratch 사전학습) | Ch 24 (작은 GPT, 영어 TinyStories) | **Ch 26 ← 여기 (작은 GPT, 한국어 TinyStories)** |
| **단계 2: Continual pretraining** (사전학습 본체 + 새 데이터) | Ch 25 (`gpt2` 124M + 영어 TinyStories) | Ch 27 (KoGPT2 125M + 한국어 TinyStories) |

> **본 챕터 = 한국어 단계 1**. Ch 24 와 *본체·loss·trainer·hyperparams 모두 동일*, *토크나이저 학습 코퍼스 + 데이터만 한국어*. 검증 가설: *언어가 달라도 작은 GPT + 30K stories from-scratch 의 학습 동역학은 비슷하다* — Ch 20↔Ch 22 (BERT) 에서 확인한 결을 GPT 에서 재확인.

## 변경점 (Diff from Ch 24)

| 축 | Ch 24 (영어 GPT scratch) | Ch 26 (한국어 GPT scratch) |
|---|---|---|
| **언어** | 영어 | **한국어** ← *유일한 변화* |
| 토크나이저 학습 코퍼스 | 영어 TinyStories | **한국어 TinyStories** |
| 토크나이저 알고리즘 | byte-level BPE (vocab 2,048) | **byte-level BPE (BBPE, vocab 약 4,000)** - 한글은 byte 단위라 어휘를 약간 키움 |
| 데이터 | `roneneldan/TinyStories` (영어 동화) | **`g0ster/TinyStories-Korean`** (한국어 번역 동화) |
| 본체 구조 | `GPT2Config(n_layer=4, n_head=4, n_embd=256)` 약 3M | (그대로) |
| 모델 클래스 | `GPT2LMHeadModel(config)` random init | (그대로) |
| Collator | `DataCollatorForLanguageModeling(mlm=False)` | (그대로) |
| Loss | `CrossEntropyLoss` (next-token, vocab 2,048 logits) | **`CrossEntropyLoss`** (next-token, vocab 약 4,000 logits) |
| 학습률 | 3e-4 (scratch) | **5e-4** ← 한국어 vocab 에 맞춘 미세 조정 |
| 산출물 | 영어 동화 풍 generation | **한국어 동화 풍 generation** |

> **핵심 변경은 *언어 축*** — Phase 4 안에서 토크나이저 학습 코퍼스와 데이터가 한국어로 바뀝니다. 본체 구조는 Ch 24 와 동일하고, 학습률만 한국어 vocab 에 맞춰 `3e-4 → 5e-4` 로 미세 조정했습니다. *같은 코드를 한국어 토크나이저 + 한국어 데이터로 돌렸을 때 같은 결 (말이 되는 한국어 동화) 이 나오는가* 가 본 챕터의 검증 포인트.

### 왜 한국어는 토크나이저를 *직접* 학습하나 — Ch 25 의 결론 잇기

Ch 25 에서 `gpt2` (영어 WebText) 의 BPE 를 *그대로* 가져와 continual pretraining 했습니다. 영어는 그게 자연스럽습니다 — gpt2 BPE 가 영어 어휘를 잘 커버하니까요. 하지만 *영어 gpt2 BPE 로 한국어를 토큰화하면 한글이 byte 단위로 잘게 쪼개져 토큰 수가 폭증* 합니다 (Ch 25 Q4 / Ch 19 §5-4 의 cross-language 결론). 그래서 한국어는 *한국어 코퍼스 위에 새 토크나이저를 학습* 하는 게 정공법 — 그게 Ch 26 가 다시 *scratch* 인 이유. *토크나이저는 본체와 운명공동체* 원칙이 한국어에서 직접 학습을 강제합니다.

## Phase 4 학습 4단계 표 — Ch 26 = 단계 1 (한국어 pretraining)

Ch 24 에서 도입한 *GPT 시대 학습 4단계* 표. 본 챕터는 단계 1 (pretraining) 의 *한국어판* 입니다.

| 단계 | 정확 용어 | 의미 | 학습 신호 (`labels`) | 영어 | 한국어 |
|---|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | random init 본체부터 일반 코퍼스로 학습 | 거의 모든 토큰 (pad 만 `-100`) | Ch 24 | **Ch 26 ← 여기** |
| 2 | **Continual pretraining** (계속 사전학습) | 사전학습된 본체 + 새 데이터 + 같은 task | 거의 모든 토큰 (단계 1 과 동일) | Ch 25 | Ch 27 |
| 3 | **SFT** (Supervised Fine-Tuning) | instruction-response 쌍으로 행동 정렬 | **답변 토큰만** (`labels[:prompt_len] = -100`) | - | Ch 28 |
| 4 | **Alignment** (DPO / GRPO) | preference / verifier reward 로 선호 정렬 | (RL 내부) | - | Ch 30-31 |

**영어·한국어 대칭** — 단계 1·2 가 영어 (Ch 24·25) ↔ 한국어 (Ch 26·27) 로 짝지어집니다.

| | 단계 1 (Pretraining) | 단계 2 (Continual pretraining) |
|---|---|---|
| 영어 | Ch 24 (작은 GPT scratch) | Ch 25 (`gpt2` 124M) |
| **한국어** | **Ch 26 (작은 GPT scratch) ← 여기** | Ch 27 (KoGPT2 125M) |

> 세 단계 모두 *모델 클래스 그대로* (`AutoModelForCausalLM`), *출력 형식 그대로* (토큰 시퀀스), *학습 신호 종류 그대로* (next-token CE). 다른 점은 *데이터 형식* 과 *`labels = -100` 자리* 뿐. 본 챕터 (단계 1) 는 *pad 만 `-100`* — 거의 모든 자리가 학습 신호. Ch 28 (SFT) 의 *prompt 만 `-100`* 이 정반대 자리 (클라이맥스).

## `labels = -100` thread — 한국어에서 한 줄 재확인

Ch 20·22 의 BERT MLM 에서 봤던 `labels = -100` ignore_index 트릭은 GPT CausalLM 사전학습에서도 등장하지만 *적용 자리가 정반대* 였습니다 (Ch 24 에서 영어로 확인). 한국어에서도 *완전히 동일* — collator 코드는 토큰 id 위에서만 동작하므로 언어와 무관합니다.

| 단계 | 챕터 | `labels` 구성 | loss 계산 자리 |
|---|---|---|---|
| MLM 사전학습 | Ch 20 (영어), Ch 22 (한국어) | 선택된 약 15% 만 원본 token id, 나머지 `-100` | *가려진 자리만* |
| GPT CausalLM 사전학습 (영어) | Ch 24 | `input_ids.clone()` - pad 만 `-100` | 거의 *전 자리* |
| **GPT CausalLM 사전학습 (한국어)** | **Ch 26 (본 챕터)** | **`input_ids.clone()` - pad 만 `-100`** | **거의 *전 자리*** |
| SFT / Instruction Tuning | Ch 28 (한국어 SFT) | **prompt 부분 `-100`**, 답변 토큰만 원본 id | *답변 부분만* |

> 같은 `-100` 트릭, *적용 자리만 정반대*. 한국어 CausalLM 사전학습도 *거의 모든 자리* 가 학습 신호 (MLM 대비 약 5-6배 효율). 본 챕터에서는 `DataCollatorForLanguageModeling(mlm=False)` 이 자동으로 `labels = input_ids.clone()` 을 만듭니다 — 뒤 collator 출력 셀에서 한국어 토큰으로 직접 확인합니다. Ch 28 (한국어 SFT) 의 *왜 모델이 instruction 을 따라가게 되는가* 는 *한 줄 `labels[:prompt_len] = -100`* 으로 설명되는데, 그 토대가 *이 챕터의 collator 출력* 입니다.

## Loss — `CrossEntropyLoss` (next-token), Ch 24 와 동일

이번 챕터는 *언어만 바뀌고* loss 함수는 Ch 24 와 동일한 next-token CrossEntropyLoss. 다만 vocab 크기가 달라 random baseline `ln V` 가 미세하게 이동합니다.

### 수식 (Ch 24 와 동일)

입력 토큰 시퀀스 $x = (x_1, \dots, x_n)$ 에 대해, 각 위치 $i$ 에서 *그 다음 토큰* $x_{i+1}$ 을 예측:

$$L_{\text{CLM}} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(x_{i+1} \mid x_1, \dots, x_i)$$

- $P(x_{i+1} \mid x_{\leq i})$: 모델이 *지금까지 본 토큰만으로* 다음 토큰을 예측할 확률 (vocab 약 4,000 차원 softmax)
- 평균 분모 $n-1$: pad 가 아닌 *거의 모든* 자리에서 loss 계산

### vocab 차이가 random baseline 에 주는 미세한 영향

| 토크나이저 | vocab size $V$ | random baseline $\ln V$ | random PPL $= V$ |
|---|---|---|---|
| BPE (Ch 24, 영어) | 2,048 | **7.62** | 2,048 |
| BBPE (Ch 26, 한국어) | 약 4,000 | **약 8.29** | 약 4,000 |

한글은 byte 단위로 표현되어 *같은 의미를 담으려면 vocab 을 약간 키우는* 게 자연스럽습니다 (그렇지 않으면 한 글자가 여러 byte 조각으로 잘게 쪼개짐). vocab 을 약 4,000 으로 잡으면 random baseline 이 약 8.29.

### 숫자로 감 잡기 (vocab 약 4,000)

| 모델 상태 | 정답 토큰 확률 | $-\log p$ |
|---|---|---|
| 균등 추측 (random init 직후) | $1/4000 = 2.5 \times 10^{-4}$ | **약 8.29** ← random baseline |
| 약하게 학습 (정답 확률 0.01-0.02) | $0.01$ - $0.02$ | **3.9 - 4.6** ← 1500 step (1분) 도달 영역 (번역체) |
| 잘 학습된 작은 GPT (정답 확률 0.05-0.15) | $0.05$ - $0.15$ | $1.9$ - $3.0$ (더 길게 학습 시) |
| 큰 사전학습 GPT (정답 확률 0.3+) | $0.3$ | 1.20 |

**관전 포인트**:
- 학습 첫 step loss 가 약 8.3 부근이면 random init 직후 *균등 추측* 상태. 첫 100 step 안에 빠르게 떨어지면 vocab + 모델 정상.
- 1분 (1500 step) 학습으로 누적 평균 `train_loss` 가 *약 4.5* 까지 내려갑니다. 번역체 한국어라 영어 챕터 (약 3.8) 보다 다소 높지만, *vocab 후보를 좁히는* 단계로 진입한 수준 — Ch 24 (영어) 와 같은 결. 더 길게 학습하면 약 2.5-3.0 까지 내려갑니다.

> Ch 24 의 `ln(2048) ≈ 7.62` 와 같은 직관. *vocab 차원* 만 약간 커진 것 (약 4,000) — 학습 동역학에는 영향 없고, *학습 종료 loss 의 절대값* 을 영어 챕터와 비교할 때만 미세 보정.

## 토크나이저 노트 — BBPE 직접 학습 (한국어, vocab 약 4,000)

Ch 24 와 같은 종류 (byte-level BPE) 의 토크나이저를 *한국어 코퍼스 위에* 직접 학습합니다. Ch 19 의 토크나이저 직접 학습 + Ch 24 의 영어 BPE 의 *한국어판*.

| 토크나이저 | 학습 코퍼스 | 등장 챕터 |
|---|---|---|
| WordLevel / WordPiece | (직접 학습) | Ch 19 |
| BPE (byte-level, 영어) | 영어 TinyStories | Ch 24 (직접 학습) |
| **BPE (byte-level, 한국어 = BBPE)** | **한국어 TinyStories** | **Ch 26 (본 챕터, 직접 학습)** |

### byte-level BPE 가 한글을 다루는 법 — UNK 없음

byte-level BPE 의 핵심: *가장 작은 단위가 byte (256개)* 라 *어떤 유니코드 문자열* (한글, 이모지, 한자) 도 *UNK 없이* 표현 가능합니다. 한글 한 글자 `가` 는 UTF-8 로 3 byte (`EA B0 80`) — BBPE 는 이 byte 들을 학습 중 *자주 함께 등장하는 쌍* 으로 병합해 *글자·어절 단위* 토큰을 만들어 갑니다.

- **영어 BPE (Ch 24)**: `"Once upon a time"` → 자주 등장하는 표현이라 적은 토큰으로 압축
- **한국어 BBPE (Ch 26)**: `"옛날 옛날에"` → 한국어 코퍼스에 자주 등장하는 어절은 적은 토큰으로 압축, 드문 어절은 byte 조각으로 분할

### 같은 한국어 문장: 영어 BPE (gpt2) vs 한국어 BBPE (본 챕터)

`gpt2` 의 영어 BPE 로 `"옛날 옛날에 작은 토끼가 살았어요"` 를 토큰화하면 한글 한 글자가 *여러 byte 조각* 으로 잘게 쪼개져 토큰 수가 폭증합니다 (UNK 는 없지만 의미 단위가 사라짐). 한국어 BBPE 로 학습하면 *같은 문장이 훨씬 적은, 의미 있는 토큰* 으로 표현됩니다 — 뒤 토크나이저 학습 셀에서 직접 비교합니다.

### 특수 토큰 컨벤션 (Ch 24 와 동일)

GPT 계열은 특수 토큰을 *최소화* — `<|endoftext|>` 하나로 bos = eos = pad 겸용. 마침 한국어 TinyStories 데이터도 story 경계를 `<|endoftext|>` 로 표시하고 있어 컨벤션이 자연스럽게 일치합니다.

> Ch 19 의 "토크나이저는 모델과 운명공동체" 원칙이 본 챕터에서도 유효 — vocab 약 4,000 의 BBPE 를 직접 학습한 뒤, *같은 vocab 으로 GPT 본체를 random init* 합니다. Ch 27 에서는 *반대로* KoGPT2 의 사전학습된 BBPE + 본체를 그대로 가져와 continual pretraining — 토크나이저 + 모델이 *함께* 변하는 게 Ch 26-27 의 핵심 비교 (Ch 24-25 의 한국어 짝).

## 이 장의 구성

[[SubPages]]
