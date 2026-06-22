**목표**: Phase 4 의 첫 챕터. Ch 7-23 까지 다룬 **BERT (encoder, MLM, task head 부착 fine-tune)** 패러다임에서, 이번엔 **GPT (decoder-only, causal LM, LM head 그대로)** 패러다임으로 전환합니다. `GPT2LMHeadModel` 을 *random init* 으로 from scratch 띄우고, **TinyStories** subset 으로 next-token 예측 사전학습 → 같은 prompt 에 *학습 전 / 학습 후* generation 결과를 나란히 비교합니다. Ch 20·22 의 *사전·사후 [MASK] 비교* 와 같은 깊이로, *사전학습이 본체에 어떤 next-token 분포를 새겼는가* 를 직접 봅니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 8-12분 (데이터 로드 약 2분 + BPE 토크나이저 학습 약 3분 + 학습 전 generation 약 30초 + 모델 학습 약 1분 + 학습 후 generation + reference 비교 약 2분)


## 학습 흐름

1. 📊 **변화 추적표 + Phase 전환 도입부** — Encoder (BERT) → Decoder (GPT) 큰 그림 한 화면
2. 🔄 **변경점** — 모델 패밀리 (encoder → decoder), 학습 목표 (MLM → CausalLM), 토크나이저 (WordPiece → BPE 직접 학습)
3. 📐 **Loss** — `CrossEntropyLoss(next-token)`. MLM 의 *15% 자리* vs CausalLM 의 *거의 모든 자리* 차이
4. 🔤 **토크나이저 노트** — BPE 직접 학습 (Ch 19 의 WordPiece/WordLevel 과 비교)
5. 🚀 **실습**: TinyStories 30K stories → BPE vocab=2048 학습 → 작은 `GPT2LMHeadModel` (약 3M params) 학습
6. 🔬 **사전·사후 generation 비교** — 같은 prompt 3개, *학습 전 (random init) vs 학습 후* 나란히, 그리고 reference `gpt2` (124M, WebText) 도 함께
7. 🛠️ **변형**: `temperature / top_k / top_p` sampling 비교
8. 📦 **등장 라이브러리** / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)


> 📒 **사전 학습 자료**: Ch 20-23 (작은 BERT scratch MLM + 분류 fine-tune). Ch 24 는 *같은 from-scratch 사전학습* 흐름인데, 본체가 *encoder (BERT) → decoder (GPT)*, 학습 목표가 *MLM → CausalLM*, 산출물이 *fine-tune 체크포인트 → generation 모델* 로 바뀝니다.

## 변화 추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 20 | 작은 BERT (영어, scratch) | `bert-base-uncased` (가져옴) | Wikitext-103 paragraphs | MLM head | `CrossEntropyLoss` (masked 15%) |
| 21 | Ch 20 + 분류 헤드 | (Ch 20 과 동일) | Yelp 이진 (다른 도메인) | `Linear(H, 2)` | `CrossEntropyLoss` |
| 22 | 작은 BERT (한국어, scratch) | `klue/bert-base` (가져옴) | 한국어 위키 paragraphs | MLM head | `CrossEntropyLoss` (masked 15%) |
| 23 | Ch 22 + 분류 헤드 | (Ch 22 와 동일) | NSMC 이진 (다른 도메인) | `Linear(H, 2)` | `CrossEntropyLoss` |
| **24 ← 여기** | **작은 GPT2 (직접, scratch)** | **BPE (직접 학습, vocab=2048)** | **TinyStories 30K stories** | **`Linear(H, V)` (LM head, weight tied)** | **`CrossEntropyLoss` (next-token, 거의 모든 자리)** |
| 25 (다음) | `gpt2` (124M, OpenAI WebText 사전학습) | BPE (GPT2 그대로) | TinyStories (Ch 24 와 동일) | `Linear(H, V)` (LM head 그대로) | `CrossEntropyLoss` (next-token) - **continual pretraining** |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.


## Phase 전환 — Encoder (BERT) → Decoder (GPT) 패러다임

Ch 7-23 의 BERT 챕터들이 *encoder + masked token 예측 + task head 부착 fine-tune* 패러다임이라면, Phase 4 (Ch 24-31) 는 *decoder + next-token 예측 + LM head 그대로 + SFT(behavior alignment)* 패러다임입니다. 본 챕터가 그 출발점.

| 축 | Phase 1·2·3 (BERT, Ch 7-23) | **Phase 4 (GPT, Ch 24-31)** |
|---|---|---|
| 본체 | Encoder (양방향 attention) | **Decoder (causal / masked attention)** |
| 사전학습 task | MLM (가려진 토큰 예측) | **CausalLM (next-token 예측)** |
| 학습 신호 위치 | 선택된 약 15% 만 (`-100` 다수) | **거의 모든 토큰** (`-100` pad 만) |
| 출력 head | task 별 부착 (`Linear(H, K)`) | **LM head (`Linear(H, V)`) 그대로** |
| Downstream 적응 | head 교체 + 본체 fine-tune (*task 적응*) | **SFT (*behavior alignment*)** + alignment (DPO/GRPO) |
| "Fine-tune" 의미 | task 별 특화 | **prompt 만 바꿔도 다른 일** |

> 본 챕터는 그 *출발점* — 작은 GPT 를 처음부터 학습해 *next-token 예측이 어떻게 generation 으로 이어지는지* 를 직접 봅니다. Ch 25 (대규모 사전학습 `gpt2` 를 TinyStories 로 **continual pretraining**) / Ch 28 (SFT) / Ch 30-31 (DPO / GRPO) 가 같은 본체 위에 차곡차곡 쌓여 갑니다.

## Phase 3 → Phase 4 다리 — *왜 갑자기 decoder 인가, 그리고 생성형 AI 의 등장*

Ch 23 (한국어 BERT 분류) 에서 Ch 24 (GPT) 로 넘어오면 *왜 갑자기 encoder 를 버리고 decoder 로 가지?* 라는 의문이 듭니다. 그 사이를 잇는 큰 그림을 한 화면에 정리합니다.

### Transformer 의 세 갈래 — encoder / decoder / encoder-decoder

원래 Transformer (Vaswani et al. 2017, *Attention Is All You Need*) 는 *번역* 을 위한 **encoder-decoder** 구조였습니다. 이후 두 절반이 *각자* 떨어져 나와 독립 계열이 됩니다.

| 갈래 | attention | 대표 모델 | 잘하는 task | 본 커리큘럼 |
|---|---|---|---|---|
| **Encoder-only** | 양방향 (bidirectional) | BERT, RoBERTa, DistilBERT, klue/bert-base | 분류·NER·추출형 QA (*이해*) | **Phase 1-3 (Ch 7-23)** |
| **Decoder-only** | causal (좌→오 마스킹) | GPT-2/3/4, LLaMA, Mistral, KoGPT2 | generation·in-context learning·SFT/RLHF (*생성*) | **Phase 4 (Ch 24-31)** |
| **Encoder-Decoder (seq2seq)** | encoder 양방향 + decoder causal + **cross-attention** | T5, BART, mBART, KoBART, KE-T5 | 번역·요약·생성형 QA (*변환*) | (본 커리큘럼은 미포함 — 맥락만) |

> 본 커리큘럼은 *encoder-only (이해)* → *decoder-only (생성)* 의 두 축을 직접 구현하며 잇습니다. **seq2seq** 는 *입력 시퀀스를 출력 시퀀스로 변환* (번역·요약) 하는 제3의 길로, 두 절반을 다시 합치고 *cross-attention* 으로 연결합니다 — 본 커리큘럼에선 다루지 않지만 *지형의 한 축* 으로 기억해 두세요.

### 왜 BERT 는 generation 이 어려운가

BERT 의 *양방향* attention 은 토큰 $i$ 가 *좌·우 전부* 를 봅니다. 그래서 *다음 토큰을 좌→오로 하나씩 뽑는* autoregressive 생성에는 부적합합니다 — 미래 토큰을 이미 보고 있으니 "다음을 예측" 이 성립하지 않습니다. BERT 가 일부만 `[MASK]` 로 가려 복원하는 (MLM) 것도 이 *양방향 cheating* 을 막기 위함이었습니다 (Ch 20).

generation 을 하려면 둘 중 하나가 필요합니다:
- **causal masking** — 미래를 가려 *좌→오 순차 생성* 을 가능케 함 → **decoder (GPT), 본 챕터부터**
- **반복 denoise** — 양방향을 유지한 채 *마스킹 비율을 일반화* 해 병렬 생성 → **diffusion LM (Phase 5, Ch 32)**

> 즉 BERT 의 양방향성은 *이해* 엔 강점, *순차 생성* 엔 약점입니다. 이 한 가지 차이가 Phase 4 (decoder) 와 Phase 5 (diffusion) 두 갈래를 가릅니다.

### attention 의 진화

| 단계 | attention 방식 | 모델 |
|---|---|---|
| 양방향 self | 모든 위치가 서로를 봄 | BERT (encoder) |
| **causal masked self** | 과거(좌)만 봄 → 미래 누출 차단 | **GPT (decoder), 본 챕터** |
| cross-attention | decoder 가 encoder 출력을 참조 | seq2seq (T5/BART) |

### 생성형 AI 등장 타임라인 (큰 흐름)

| 연도 | 사건 | 의미 |
|---|---|---|
| 2017 | Transformer (*Attention Is All You Need*) | encoder-decoder, attention 의 출발 |
| 2018 | BERT (encoder) / GPT-1 (decoder) | *이해* 와 *생성* 두 갈래 분기 |
| 2019 | GPT-2 | 큰 decoder 의 generation 품질 — 본 챕터의 reference 모델 |
| 2020 | GPT-3 | **in-context learning** (예시만 줘도 task 수행, fine-tune 없이) |
| 2022 | InstructGPT / ChatGPT | **SFT + RLHF** 로 *지시를 따르는* 정렬 (Ch 28·30·31 의 주제) |
| 2023+ | GPT-4 / LLaMA / Mistral / KoGPT 등 | decoder-only LLM 의 시대 |

> 본 커리큘럼 Phase 4 (Ch 24-31) 가 이 타임라인을 *압축 재현* 합니다 — *작은 GPT 사전학습 (본 챕터)* → *continual pretraining (Ch 25)* → *SFT (Ch 28)* → *alignment (Ch 30-31)*. 2017-2022 의 흐름을 손으로 한 번 따라가 보는 셈입니다.

> 📚 **참고** — *T4 30분 룰 너머* 로 GPT/LLM scratch 학습을 더 키워 보고 싶다면 [FareedKhan-dev/train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch) 가 좋은 출발점입니다. Transformer 를 PyTorch 로 직접 구현하고 *13M → 2B params* 까지 consumer GPU 로 학습하는 과정 + **GPU별 실용 모델 크기 표** (예: T4 16GB ≈ 1.5-2B, RTX 4090 24GB ≈ 4B, A100 40GB ≈ 6-8B) 가 인상적입니다 — 본 챕터의 약 3M 모델이 *어디까지 커질 수 있는지* 의 감을 줍니다.

## 변경점 (Diff from Ch 23)

| 축 | Ch 23 (한국어 BERT 분류 fine-tune) | Ch 24 (GPT scratch + TinyStories) |
|---|---|---|
| **모델 패밀리** | Encoder (`BertForSequenceClassification`) | **Decoder (`GPT2LMHeadModel`)** ← *Phase 전환의 핵심* |
| 사전학습 task | MLM (Ch 22 산출물 본체 + 분류 head) | **CausalLM (next-token, from scratch)** |
| 토크나이저 | `klue/bert-base` WordPiece (가져옴, vocab 32K) | **BPE 직접 학습** (vocab 2,048) |
| 데이터 | NSMC 한국어 영화 리뷰 (이진 라벨) | **TinyStories 영어 short stories** (라벨 없음) |
| Output head | `Linear(H, 2)` (새로 부착) | **`Linear(H, V)` LM head** (모델이 내장 + weight tied) |
| Loss | `CrossEntropyLoss` (분류, K=2) | **`CrossEntropyLoss` (next-token, K=V=2048)** |
| 산출물 | 분류 정확도 | **generation 텍스트** (`model.generate()`) |

> **변경점이 한꺼번에 많은 이유** — Phase 가 바뀌는 *전환 챕터* 라 *축 자체* 가 새로 정의됩니다. Ch 25 부터는 다시 *한 가지 축* 만 바뀝니다 (Ch 25: 본체 출발점 = scratch → 사전학습 모델, 같은 *continual pretraining* task / Ch 26: 언어 (한국어 scratch) / Ch 27: 한국어 continual pretraining / Ch 28: 학습 단계 = pretraining → SFT, `labels[prompt] = -100`).

## Loss — `CrossEntropyLoss` (next-token)

수식은 MLM 의 CE 와 *완전히 같음*. 다만 *어느 자리에서 loss 가 계산되는가* 가 다릅니다.

### 수식

입력 토큰 시퀀스 $x = (x_1, \dots, x_n)$ 에 대해, 각 위치 $i$ 에서 *그 다음 토큰* $x_{i+1}$ 을 예측:

$$L_{\text{CLM}} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(x_{i+1} \mid x_1, \dots, x_i)$$

- $P(x_{i+1} \mid x_{\leq i})$: 모델이 *지금까지 본 토큰만으로* 다음 토큰을 예측할 확률 (vocab 2,048 차원 softmax)
- 평균 분모 $n-1$: pad 가 아닌 *거의 모든* 자리에서 loss 계산 (MLM 의 15% 와 대비)

### 숫자로 감 잡기 (vocab=2048)

| 모델 상태 | 정답 토큰 확률 | $-\log p$ |
|---|---|---|
| 균등 추측 (random init 직후) | $1/2048 \approx 4.88 \times 10^{-4}$ | **7.62** ← random baseline |
| 약하게 학습 (정답 확률 0.02) | $0.02$ | **3.91** ← 1500 step (1분) 도달 영역 |
| 잘 학습된 작은 GPT (정답 확률 0.05-0.15) | $0.05$ - $0.15$ | $1.9$ - $3.0$ (더 길게 학습 시) |
| 큰 사전학습 GPT (정답 확률 0.3+) | $0.3$ | 1.20 |
| 완벽 (정답 확률 1.0) | $1.0$ | 0.00 |

**관전 포인트**:
- 학습 첫 step loss 가 약 7.6 부근이면 random init 직후 *균등 추측* 상태. 첫 100 step 안에 빠르게 떨어지면 vocab + 모델 정상.
- 1분 (1500 step) 학습으로 누적 평균 `train_loss` 가 *약 3.8* 까지 내려갑니다. *vocab 후보를 좁히는* 단계로 충분히 진입한 수준 - TinyStories 의 단순한 어휘·문법 덕분에 3M 짜리 작은 모델로도 도달 가능합니다. 더 길게 학습하면 약 2-3 까지 더 내려갑니다.

### Perplexity (PPL)

$\text{PPL} = e^{L}$ — *다음 토큰을 평균 몇 후보 중에서 고민하는가*:

| CLM loss | PPL | 해석 |
|---|---|---|
| 7.62 | 2,048 | 균등 (전체 vocab) |
| 4.0 | 55 | 약 50 개 후보 ← 1500 step (1분) 도달 영역 |
| 2.5 | 12 | 약 12 개 후보 (더 길게 학습 시) |
| 1.0 | 2.7 | 거의 결정적 |

> MLM 의 `ln(30522) ≈ 10.33` random baseline 과 같은 직관. *vocab 차원* 만 작아진 것 (2,048).

## `labels = -100` thread 환기 — MLM 의 *15% 만* vs CausalLM 의 *거의 모든 자리*

Ch 20·22 의 MLM 에서 봤던 `labels = -100` ignore_index 트릭이 GPT CausalLM 사전학습에서도 등장하지만, **적용 자리가 정반대** 입니다.

| 단계 | 챕터 | `labels` 구성 | loss 계산 자리 |
|---|---|---|---|
| MLM 사전학습 | Ch 20 (영어), Ch 22 (한국어) | 선택된 약 15% 만 원본 token id, 나머지 = `-100` | *가려진 자리만* |
| **GPT CausalLM 사전학습** | **Ch 24 (영어, 본 챕터), Ch 26 (한국어)** | **`input_ids.clone()` - pad 만 `-100`** | **거의 *전 자리*** |
| SFT / Instruction Tuning | Ch 28 (한국어 KoGPT2 SFT) | **prompt 부분 = `-100`**, *답변 토큰만* 원본 id | *답변 부분만* |

> 같은 `-100` 트릭, *적용 자리만 정반대*. MLM 은 *대부분을 가리고 일부만 학습*, GPT 사전학습은 *거의 가리지 않음*, SFT 는 *prompt 만 가림*. 한 step 에 학습되는 토큰 수만 봐도 *GPT 사전학습은 MLM 대비 약 5-6배 효율* (15% vs 거의 100%).

본 챕터에서는 `DataCollatorForLanguageModeling(mlm=False)` 이 자동으로 `labels = input_ids.clone()` 을 만들어 줍니다 — 뒤에 collator 출력 셀에서 직접 확인하겠습니다. Ch 28 의 *왜 모델이 instruction 을 따라가게 되는가* 는 *한 줄 코드 `labels[prompt_mask] = -100`* 로 정확히 설명되는데, 그 코드를 이해할 토대가 *이 챕터의 collator 출력* 입니다.

## GPT 시대의 학습은 *네 단계* — 용어가 BERT 와 다릅니다

Ch 21·23 에서 본 *fine-tune* 은 **BERT 시대 의미** — *사전학습된 본체 + 새 task-specific head (`Linear(H, K)`)*. 분류·회귀·QA 마다 다른 head, *task 별 특화*. 한 모델 = 한 task.

Phase 4 GPT 시대는 *fine-tune* 한 단어가 *여러 의미* 로 섞여 쓰입니다. 학술적으로는 **네 단계** 로 분리됩니다.

| 단계 | 정확 용어 | 의미 | 학습 신호 | 본 커리큘럼 |
|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | 일반 코퍼스 위에 random init 본체부터 학습 | 모든 토큰 (`labels = input_ids`) | **Ch 24** (영어 scratch, TinyStories), **Ch 26** (한국어 scratch) |
| 2 | **Continual pretraining** (계속 사전학습 / continual learning) | *사전학습된 본체* 를 *새 데이터* 로 *같은 CausalLM task* 더 학습. **head 그대로, task 그대로, 데이터만 새로** | 모든 토큰 (pretraining 과 동일) | **Ch 25** (`gpt2` + TinyStories) |
| 3 | **SFT** (Supervised Fine-Tuning / Instruction tuning) | instruction-response 쌍으로 *행동 정렬*. `labels[prompt] = -100` 으로 답변 부분만 학습 | **답변 토큰만** | **Ch 28** (KoGPT2 + KoAlpaca) |
| 4 | **Alignment** (DPO / RLHF / GRPO) | preference 또는 verifier reward 로 *선호 정렬* | preference log-likelihood ratio / RL advantage | **Ch 30** (DPO), **Ch 31** (GRPO) |

**세 가지 공통점** (모두 GPT 시대):
- **모델 클래스 그대로** — `AutoModelForCausalLM` (BERT 처럼 task head 부착 안 함)
- **출력 형식 그대로** — 토큰 시퀀스
- **학습 신호 종류 그대로** — next-token CE (alignment 만 예외)

**다른 점은 *데이터 형식* 과 *어느 토큰에 학습 신호를 주는가*** — `labels = -100` 자리만 변함.

| 단계 | 데이터 | `labels = -100` 자리 |
|---|---|---|
| Pretraining (Ch 24·26) | 일반 텍스트 | pad 만 |
| **Continual pretraining (Ch 25)** | **새 도메인 텍스트** | **pad 만 (Pretraining 과 동일)** |
| SFT (Ch 28) | instruction + response | **prompt 부분** |
| Alignment (Ch 30·31) | preference 쌍 / verifier reward | (RL 내부) |

> **BERT 의 "fine-tune" 은 *task 적응* 한 가지였지만, GPT 의 "fine-tune" 은 *continual pretraining / SFT / alignment* 셋이 섞인 통칭**. 정확히 말하려면 단계별 용어를 구분합니다. 본 챕터는 단계 1 (사전학습) 그 자체. Ch 25 에서 단계 2 (continual pretraining) 가, Ch 28 에서 단계 3 (SFT — 진짜 *행동 정렬*) 이, Ch 30-31 에서 단계 4 (alignment) 가 본격 등장합니다.

> *왜 GPT 모델 하나가 모든 task 를 해내는가* 의 답은 단계 3 (SFT) 부터 — head 가 task 별로 분기하지 않으니 *입력 프롬프트* 만 바꾸면 같은 모델이 다른 일을 합니다.

## 토크나이저 노트 — BPE 직접 학습 (vocab=2048)

GPT-2 와 같은 종류 (byte-level BPE) 의 작은 vocab 을 직접 학습합니다. Ch 19 에서 봤던 WordPiece / WordLevel 토크나이저 학습 절차의 *BPE 판*.

| 토크나이저 | 학습 방식 | 등장 챕터 |
|---|---|---|
| WordLevel | 공백 + 빈도 - 가장 단순 | Ch 19 (직접 학습) |
| WordPiece | 빈도 기반 subword (BERT) | Ch 7-23 (BERT 챕터들 - 가져옴), Ch 19 (직접 학습) |
| **BPE (byte-level)** | **빈도 높은 byte 쌍 반복 병합 (GPT-2)** | **Ch 24 (본 챕터, 직접 학습), Ch 25-26 (GPT 챕터들)** |

### WordPiece vs BPE 의 결합 방식 차이

같은 입력 `"unhappiness"` 에 대해:

- **WordPiece** (BERT): `["un", "##happiness"]` 또는 `["un", "##happy", "##ness"]` - 단어 *중간* subword 에 `##` 접두사. 단어 경계를 명시.
- **BPE** (GPT-2): `["un", "happiness"]` 또는 `["un", "h", "app", "iness"]` - 접두사 없이 *byte 시퀀스 그대로*. 단어 경계는 *공백 자체가 한 byte* 로 처리.

byte-level BPE 의 핵심 장점: *어떤 유니코드 문자열이든* (이모지, 한글, 특수 기호) UNK 없이 표현 가능 - 가장 작은 단위가 *byte (256개)* 라 vocab 에 모든 byte 를 포함하면 *완전 가역*.

### 특수 토큰 컨벤션

GPT-2 는 특수 토큰을 *최소화* 합니다 - `<|endoftext|>` 하나만 사용 (bos = eos = pad 겸용). BERT 의 `[CLS] [SEP] [MASK] [PAD] [UNK]` 5종과 대비.

> Ch 19 의 "토크나이저는 모델과 운명공동체" 원칙이 본 챕터에서도 유효 - vocab 2,048 의 BPE 를 직접 학습한 뒤, *같은 vocab 으로 GPT 본체를 random init* 합니다. Ch 25 에서는 *반대로* - `gpt2` (124M) 의 vocab 50,257 BPE 를 그대로 가져와 같은 TinyStories 데이터로 **continual pretraining**. 토크나이저 + 모델이 *함께* 변하는 게 Ch 24-25 의 핵심 비교.

## 환경 셋업

## TinyStories 데이터 로드

`roneneldan/TinyStories` 는 GPT-3.5 / GPT-4 가 *4세 어린이가 이해할 단어만* 으로 생성한 짧은 영어 동화 약 2.1M 편 (Eldan & Li 2023, arXiv:2305.07759). 어휘·문법이 단순해 **3-5M 파라미터** 짜리 작은 모델로도 grammatical 한 생성이 가능합니다.

학습 split 의 처음 **30,000 stories** 만 사용 (T4 30분 룰 안).

## BPE 토크나이저 직접 학습

`tokenizers.BPE` + ByteLevel pre-tokenizer 로 vocab 2,048 의 BPE 를 코퍼스에서 직접 학습합니다. Ch 19 의 토크나이저 학습 절차와 같은 패턴 - 다른 점은 *알고리즘* 만 (WordPiece → BPE).

**관전 포인트** — `Once upon a time` 같이 TinyStories 에 *자주 등장* 하는 표현은 *적은 수의 토큰* 으로 압축, `rabbit` 처럼 덜 등장한 단어는 *여러 byte 조각* 으로 분할되는 경향. vocab 2,048 은 작은 모델에 맞춘 *최소한의 크기* 입니다.

## 토큰화 + `group_texts` (HF 표준 CLM 전처리)

HuggingFace 의 causal LM 학습 표준 패턴 (`run_clm.py`) 그대로:

1. 전체 코퍼스를 토큰화 (배치 단위)
2. 각 story 끝에 `<|endoftext|>` 부착 (story 경계 표시)
3. 모든 토큰을 이어붙여 1D 스트림으로 만든 뒤 `block_size=128` 단위로 잘라 chunk 화
4. 각 chunk 가 한 학습 sample - `DataCollatorForLanguageModeling(mlm=False)` 가 `labels = input_ids` 를 자동으로 채워 next-token prediction loss 가 됨

Ch 20·22 의 `group_texts` 와 *완전히 같은 패턴*. MLM 챕터들에선 *마스킹* 만 추가됐다면, CausalLM 챕터에선 *labels = input_ids* 그대로.

### Collator 가 만드는 `labels` 확인 - *거의 모든 자리* 가 학습 신호

`DataCollatorForLanguageModeling(mlm=False)` 가 *내부적으로* `labels = input_ids.clone()` 을 만들어 `-100` 자리는 *없거나 pad 토큰 자리만* 임을 직접 확인합니다. Ch 20·22 의 MLM collator 가 약 85% 를 `-100` 으로 채웠던 것과 *정확히 반대*.

> **`-100` thread 환기** - MLM 은 *마스킹된 자리만* 학습, CausalLM 은 *거의 모든 자리* 학습. 같은 PyTorch `CrossEntropyLoss(ignore_index=-100)` 트릭이 *적용 자리만 정반대*. Ch 28 (SFT) 에서는 *prompt 자리만 -100* - 같은 트릭의 세 번째 적용. 그 한 줄 코드가 *모델이 instruction 을 따라가게 만드는 핵심* 입니다.

본 챕터의 collator 셋업이 *그 토대* - `labels = input_ids.clone()` 의 직관을 손에 익혀 두면 Ch 28 의 `labels[:prompt_len] = -100` 가 단번에 이해됩니다.

## `GPT2LMHeadModel` from scratch

`GPT2Config` 의 핵심 필드만 작게 잡고 *random init* (사전학습 X) 시작.

- `n_layer=4, n_head=4, n_embd=256` → 약 3M params, BERT 챕터들의 small DistilBERT 와 비슷한 스케일
- `n_positions = BLOCK_SIZE = 128` - 학습한 만큼만 context 사용
- bos / eos / pad token id 를 토크나이저와 동기화
- `tie_word_embeddings=True` (기본) - LM head 와 input embedding 의 weight 를 공유 → 파라미터 절약

### BERT 와의 차이가 코드로 드러나는 곳

- `BertForMaskedLM` 이 아니라 `GPT2LMHeadModel` - 클래스 자체가 *causal attention 내장*
- `from_pretrained(...)` 없이 `GPT2LMHeadModel(config)` - 무작위 초기화 from scratch (Ch 20·22 의 `BertForMaskedLM(config)` 와 같은 패턴, *모델 패밀리만* 다름)

## 학습 *전* generation - 비교 기준선 (random init baseline)

Ch 20·22 의 *사전학습 전 [MASK] top-5 후보* 와 같은 역할. random init 모델은 통계적으로 *어느 토큰이든 거의 균등한 확률* 로 뽑으니, 생성 텍스트가 *영어와 거리가 먼 byte 조각 / 의미 없는 짧은 단어 나열* 이 나옵니다.

같은 prompt 와 sampling 설정을 학습 *전 / 후* 모두에서 호출 → loss 곡선 없이도 *학습이 본체에 무엇을 새겼는가* 가 한 화면에 드러납니다.

**관전 포인트** - 학습 전 출력은 *무작위 토큰 나열* (반복되는 짧은 byte 조각, 의미 없는 단어들). Ch 20·22 의 *학습 전 [MASK] top-5* 가 *the / a / of / , / .* 같은 통계적 빈도 토큰이었던 것과 같은 현상의 *generation 판* 입니다. 학습 후 출력과 *나란히 비교* 하면 사전학습이 본체에 *next-token 분포* 를 새긴 증거를 직접 보게 됩니다.

## `Trainer` 로 사전학습

BERT 챕터들 (Ch 20·22) 과 *완전히 같은* Trainer 패턴 - 바뀌는 건 모델 클래스와 collator 의 `mlm=False` 두 곳.

- `DataCollatorForLanguageModeling(mlm=False)` → `labels = input_ids` (next-token prediction)
- `max_steps=1500`, `batch_size=32`, `fp16=True` - T4 약 1분
- `eval_steps=150` 으로 train / val loss 추이 관찰

**관전 포인트** - 학습 첫 step loss 가 약 7.6 (random baseline `ln(2048)`) 부근에서 시작해 *수백 step 안에 약 4-5* 로 빠르게 떨어지고, 1500 step 끝에 누적 평균 `train_loss` 가 *약 3.8* 까지 내려가면 정상 (`train_loss` 는 학습 내내 본 step 들의 누적 평균이라 마지막 step 의 순간 loss 보다 다소 높게 보입니다). perplexity 로 환산하면 vocab 2,048 중 *수십 개 후보* 로 좁힌 상태 - 다음 토큰을 *어느 정도 결정적* 으로 뽑는 수준.

## 학습 *후* generation + before/after 비교

같은 `PROMPTS / GEN_KWARGS` 로 학습 후 모델에서 다시 생성하고, §5 의 학습 전 결과와 나란히 비교합니다. **이 챕터의 합격 기준**: 학습 후 텍스트가 *전* 보다 명확히 *영어 문장* 에 가까워졌는가 - Ch 20·22 의 *사전·사후 [MASK] top-5* 비교의 *generation 판*.

**해석 가이드 - 사전학습이 만든 차이**

- **BEFORE (random init)**: *영어와 거리가 먼 byte 조각 / 의미 없는 짧은 단어 반복*. logits 가 random 초기값이라 sampling 이 통계적 빈도 토큰들 사이에서만 흔들림.
- **AFTER (TinyStories 30K × 1500 steps)**: *말이 되는 영어 문장* - 짧지만 *주어 + 동사 + 목적어* 구조, *동화 풍 어휘* (rabbit, forest, friend, mom, happy, ...). 완벽하진 않아도 *학습이 본체에 next-token 분포를 새긴 증거* 가 한 줄에서 명확.

> Ch 20·22 의 *사전·사후 [MASK] top-5* 비교에서 `[MASK]` 자리에 *the / a / of* 같은 빈도 토큰만 뽑히던 random init 모델이, 학습 후엔 *문맥에 맞는 정답 토큰* 을 top-5 에 담아내던 그 변화의 *generation 판* 입니다.

### Reference 비교 - `gpt2` (124M, OpenAI WebText) 의 같은 prompt generation

같은 prompt 3개를 *학습이 충분히 잘 된* 표준 `gpt2` (124M params, WebText 약 40GB 사전학습) 에 넣어 *우리 작은 GPT (약 3M, TinyStories 30K)* 와 격차를 직접 비교. Ch 20 의 *3-way [MASK] top-5 비교* (before / ours / `bert-base-uncased`) 와 같은 패턴.

T4 에서 약 1분 추가. 데이터·파라미터 격차가 generation 품질의 격차로 어떻게 드러나는지 한 화면에.

**해석 가이드 - 데이터·파라미터 규모가 만든 격차**

- **BEFORE (random)**: 영어와 거리 먼 byte 조각.
- **OURS (3M, TinyStories 30K × 1500 steps)**: *동화 풍 단순 영어* - 어휘는 동화 도메인에 강하지만 (rabbit, forest, mom, friend, ...) *복잡한 문장 구조 / 추상적 어휘* 는 약함.
- **REF (gpt2 124M, WebText 약 40GB)**: *다양한 도메인 어휘 + 자연스러운 문장 흐름* - 같은 prompt 에 대해 *동화풍이 아닌 일반 산문 / 뉴스 / 대화* 등 다양한 톤. 학습 데이터 분포 (WebText) 의 다양성이 generation 다양성으로 직결.

> **세 모델의 격차가 정확히 *모델 크기 + 데이터 크기 + 데이터 다양성* 의 격차** - 우리 작은 GPT (3M, TinyStories 30K stories) → reference `gpt2` (124M, WebText 약 40GB) 사이에 *파라미터 약 40배, 데이터 규모 약 수천 배, 도메인 다양성 격차*. 그게 generation 의 *질적 차이* 로 정확히 드러납니다.

> Ch 25 가 이 격차를 *데이터 축을 통제하고* 좁히는 챕터입니다 - `gpt2` (124M) 의 사전학습 *위에* 같은 TinyStories 30K 로 **continual pretraining**. *대규모 일반 사전학습 모델을 작은 도메인 데이터로 적응* 시킬 때의 generation 품질이, 우리 from-scratch 작은 GPT 와 어떻게 다른지 직접 비교.

## 이 장의 구성

[[SubPages]]
