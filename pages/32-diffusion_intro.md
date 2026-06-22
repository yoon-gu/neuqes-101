Phase 5 의 첫 챕터. Ch 24-31 까지 다룬 **GPT (decoder, autoregressive, 왼→오 순차 생성)** 패러다임에서, 이번엔 **Diffusion LM (encoder/bidirectional, masked-denoise, 문장 전체를 병렬로 생성)** 패러다임으로 전환합니다.

학습이 끝나면 전부 `[MASK]` 인 빈 캔버스에서 시작해, 왼→오가 아니라 **병렬로** 토큰을 채워가며 영어 동화를 만들어냅니다.

> *"Once upon a time, there was a boy named Timmy. He had found a big toy ball in the park. He went to a big house with his toys. He liked to play with him."*

핵심 한 줄: **BERT MLM (Ch 20-23) 의 *고정 15% 마스킹* 을 *0-100% 가변 마스킹* 으로 일반화하고, 한 번에 복원하는 대신 *여러 번 반복 denoise* 하면 그게 generation 입니다.** Ch 1 부터 추적해 온 *마스킹 + 토크나이저* 시각이 여기서 클라이맥스에 도달합니다 — 가려서 맞히던 BERT 가, 가리는 비율을 끝까지 밀어붙이면 *무에서 문장을 만들어내는 생성 모델* 이 됩니다.

작은 BERT-style 모델을 *random init* 으로 from scratch 띄우고, **TinyStories** 로 *가변 마스킹 denoising* 목표로 학습 → reverse process (전부 `[MASK]` 에서 시작해 반복 denoise) 로 텍스트를 *왼→오가 아닌 병렬* 로 생성하는 과정을 직접 눈으로 봅니다.

**환경**: Google Colab **T4 GPU 필수**. **예상 소요**: 약 25분 (데이터·토큰화 약 5분 + 학습 약 20분 + 생성·궤적).


## 학습 흐름

1. 📊 **변화 추적표 + Phase 전환 도입부** — Autoregressive (GPT) → Diffusion 큰 그림
2. 🔄 **변경점** — 생성 방식 (순차 → 병렬 denoise), attention (causal → bidirectional), 마스킹 (고정 15% → 가변 0-100%)
3. 📐 **Loss** — masked-diffusion denoising loss. MLM 의 CE 를 *가변 마스킹 비율 t* 로 일반화 + `1/t` 재가중
4. 💡 **마스킹 thread 클라이맥스** — BERT 의 *고정 15%* vs diffusion 의 *가변 0-100%*. 같은 `-100` 트릭
5. 🔤 **토크나이저 노트** — 작은 모델엔 작은 vocab (ByteLevel BPE 2048 직접 학습 + `[MASK]`)
6. 🚀 **실습**: TinyStories → 작은 BERT-style 모델 → 가변 마스킹 denoising 학습
7. 🔬 **Reverse process generation** — 전부 `[MASK]` 에서 반복 denoise. 마스크가 *병렬로* 단어로 채워지는 궤적 직접 관찰
8. 🛠️ **변형**: denoise step 수 비교 (1 / 4 / 16 / 32), 조건부 생성 (prompt 고정)
9. ⚖️ **AR vs Diffusion 비교** — Ch 24 (GPT) 와 나란히. Ch 33 (샘플러)·Ch 34 (한국어) 예고
10. 📦 **등장 라이브러리** / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)


> 📒 **사전 학습 자료**: Ch 20-23 (BERT MLM 사전학습 — 고정 15% 마스킹), Ch 24 (GPT from scratch — autoregressive generation). 본 챕터는 *둘을 잇습니다* — BERT 의 *마스킹-복원* 메커니즘을 Ch 24 의 *generation* 목적에 다시 씁니다. 다른 점은 *마스킹 비율을 0-100% 로 일반화* 하고 *복원을 여러 번 반복* 한다는 것뿐.

## 변화 추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | 생성/학습 방식 | Loss |
|---|---|---|---|---|---|---|
| 20 | 작은 BERT (영어, scratch) | `bert-base-uncased` (가져옴) | Wikitext-103 | MLM head | 고정 15% 마스킹-복원 | `CrossEntropyLoss` (masked 15%) |
| 24 | 작은 GPT2 (직접, scratch) | BPE (직접 학습) | TinyStories | `Linear(H, V)` | autoregressive (왼→오 순차) | `CrossEntropyLoss` (next-token) |
| 31 | SFT base + GRPO | BBPE | verifiable-reward | `Linear(H, V)` + group adv. | autoregressive + RL | `GRPO loss` |
| **32 ← 여기** | **작은 BERT-style (직접, scratch)** | **ByteLevel BPE 2048 (직접 학습 + `[MASK]`)** | **TinyStories** | **`Linear(H, V)`** | **parallel denoise (가변 마스킹 + 반복 복원)** | **masked-diffusion denoising loss (`1/t` 재가중)** |
| 33 (다음) | MDLM (170M) / DiffuGPT (124M) 사전학습 | (각 모델 토크나이저) | 영어 사전학습 추론 시연 | `Linear(H, V)` | parallel denoise (추론만) | — |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.


## Phase 전환 — Autoregressive (GPT) → Diffusion LM

Ch 24-31 의 GPT 챕터들이 *decoder + next-token 예측 + 왼→오 순차 생성* 패러다임이라면, Phase 5 (Ch 32-34) 는 *encoder/bidirectional + masked-denoise + 문장 전체 병렬 생성* 패러다임입니다. 본 챕터가 그 출발점.

| 축 | Phase 4 (GPT, Ch 24-31) | **Phase 5 (Diffusion, Ch 32-34)** |
|---|---|---|
| attention | Causal (과거만 봄) | **Bidirectional (양방향 다 봄)** |
| 학습 목표 | next-token 예측 | **가변 마스킹 denoising** |
| 생성 순서 | 왼→오 *한 토큰씩 순차* | **문장 전체를 *동시에* 여러 번 denoise** |
| 생성 step 수 | 토큰 수 = step 수 (길면 느림) | **step 수를 *자유롭게 조절* (4 / 16 / 32 ...)** |
| 출발 상태 | prompt 토큰들 | **전부 `[MASK]` (무에서 시작)** |
| 본체 계보 | GPT (Ch 24) | **BERT (Ch 20)** — MLM 을 일반화 |

> **핵심 직관**: GPT 가 *왼쪽부터 한 글자씩 받아쓰기* 라면, diffusion 은 *흐릿한 전체 그림을 여러 번 선명하게 다듬기* 입니다. 이미지 생성에서 노이즈를 점점 걷어내듯, 텍스트에서는 `[MASK]` 를 점점 진짜 단어로 바꿔 갑니다. 본 챕터는 그 메커니즘을 *작은 모델로 직접 구현* 해 봅니다. Ch 33 (MDLM 170M / DiffuGPT 124M 사전학습) 이 *같은 원리의, 충분한 규모로 학습된 실전 모델* 입니다.

## 변경점 (Diff from Ch 31)

| 축 | Ch 24-31 (GPT, autoregressive) | Ch 32 (Diffusion LM) |
|---|---|---|
| **생성 방식** | next-token, 왼→오 *순차* | **masked-denoise, 문장 전체 *병렬*** ← *Phase 전환의 핵심* |
| attention | Causal (`GPT2LMHeadModel`) | **Bidirectional (`BertForMaskedLM` 계열)** |
| 학습 목표 | `CrossEntropyLoss` (next-token, 거의 모든 자리) | **masked-diffusion loss (가변 비율 `t` 마스킹 + `1/t` 재가중)** |
| 마스킹 | 없음 (causal mask 가 미래 차단) | **입력 토큰을 `t` 비율로 `[MASK]` 치환** |
| 토크나이저 | BPE (GPT 계열) | **ByteLevel BPE 2048 직접 학습 + `[MASK]`** |
| 생성 출발 | prompt 토큰 | **전부 `[MASK]` 인 시퀀스** |
| 생성 step | 토큰 길이 만큼 | **임의 step 수 (속도-품질 trade-off 조절 가능)** |

> **변경점이 한꺼번에 많은 이유** — Phase 가 바뀌는 *전환 챕터* 라 *축 자체* 가 새로 정의됩니다. 하지만 본질은 *Ch 20 의 BERT MLM 을 재활용* 한 것 — *bidirectional + 마스킹-복원* 은 이미 다 배운 메커니즘이고, *마스킹 비율을 가변* 으로 만들고 *복원을 반복* 한 것만 새롭습니다. 다음 두 챕터는 다시 *한 가지씩* 만 바뀝니다 — **Ch 33: 샘플러를 바꿔 생성 품질을 끌어올리고, Ch 34: 한국어로 확장**합니다.

## Loss — masked-diffusion denoising loss

BERT MLM 의 CrossEntropyLoss 와 *뼈대는 같습니다* — 가려진 자리의 정답 토큰을 맞히는 CE. 다른 점은 두 가지:

1. 마스킹 비율이 *고정 15%* 가 아니라 *매 샘플마다 $t \sim U(0, 1)$ 로 뽑은 가변 비율*
2. 비율 $t$ 만큼 가렸으니, loss 를 *$1/t$ 로 재가중* 해 *어떤 마스킹 비율이든 공정하게* 평균

### 수식

깨끗한 토큰 시퀀스 $x_0 = (x_1, \dots, x_L)$ 에 대해, 비율 $t$ 를 뽑고 각 토큰을 *독립적으로 확률 $t$* 로 `[MASK]` 치환해 $x_t$ 를 만듭니다. 모델은 $x_t$ 전체를 보고 *가려진 자리* 의 원본 토큰을 예측:

$$L = \mathbb{E}_{t \sim U(0,1)} \left[ \frac{1}{t} \cdot \frac{1}{L} \sum_{i:\, x_t^{(i)} = \texttt{[MASK]}} -\log P_\theta\!\left(x_0^{(i)} \mid x_t\right) \right]$$

- $\sum_{i:\, x_t^{(i)}=\texttt{[MASK]}}$: *가려진 자리에서만* loss 계산 (Ch 20-23 의 `-100` 트릭과 동일)
- $1/t$ 재가중: $t$ 가 작으면 (조금 가림) 가려진 토큰이 적으니 합이 작아지는데, $1/t$ 로 곱해 *마스킹 비율에 무관* 하게 스케일을 맞춤. 이 재가중 덕분에 *학습 목표가 log-likelihood 의 upper bound* 가 됩니다 (LLaDA / MDLM 의 핵심 항)
- $t \sim U(0,1)$: 매 step 마다 *다른 난이도* 의 복원 문제를 풀게 함 — 5% 만 가린 쉬운 문제부터 95% 가린 거의-무에서-생성 문제까지

### 숫자로 감 잡기 (vocab = 2,048)

random init 직후 모델은 가려진 자리를 *균등 추측* → 정답 확률 $1/2048$, 토큰당 $-\log p \approx \ln(2048) = 7.62$.

| 마스킹 비율 $t$ | 가린 토큰 수 (L=128) | 가린 자리 합 ($\approx t L \times 7.62$) | $\times \frac{1}{t}\frac{1}{L}$ 후 | 해석 |
|---|---|---|---|---|
| 0.10 | 약 13 | 약 99 | **7.62** | 조금 가린 쉬운 복원 |
| 0.50 | 약 64 | 약 488 | **7.62** | 절반 가림 |
| 0.90 | 약 115 | 약 877 | **7.62** | 거의 무에서 생성 |

**관전 포인트**:
- `1/t` 재가중 덕분에 *어떤 t 든 baseline loss 가 똑같이 `ln(vocab) ≈ 7.62`* 로 정렬됩니다. 직접 학습한 BPE 2048 의 random baseline `ln(2048) ≈ 7.62` 와 같은 값 — *마스킹 비율만 일반화* 했지 loss 의 척도는 그대로입니다.
- 학습 첫 step loss 가 약 7.6 부근에서 시작해 빠르게 떨어지면 정상. 목표는 약 3.7 부근 (작은 모델 + TinyStories).
- $t$ 가 1 에 가까울수록 (거의 다 가림) *문맥 정보가 거의 없어* 복원이 어려움 → diffusion 생성이 *여러 step 에 나눠* 조금씩 푸는 이유.

## 마스킹 thread 클라이맥스 — *고정 15%* (BERT) → *가변 0-100%* (diffusion)

Ch 1 부터 *토크나이저와 마스킹* 을 일관되게 추적해 왔습니다. 그 흐름이 여기서 정점에 닿습니다.

| 단계 | 챕터 | 마스킹 비율 | 복원 횟수 | 용도 |
|---|---|---|---|---|
| MLM 사전학습 | Ch 20 (영어), Ch 22 (한국어) | **고정 15%** | 1회 | 표현 학습 (downstream fine-tune 용) |
| GPT CausalLM | Ch 24-31 | 마스킹 없음 (causal mask) | — | autoregressive 생성 |
| **Mask-diffusion** | **Ch 32 (본 챕터)** | **가변 $t \sim U(0,1)$** | **반복 (4-32 step)** | **병렬 생성** |

핵심은 **셋이 모두 같은 `labels = -100` 트릭** 을 쓴다는 점입니다 — *가려진 자리만* loss 계산, 나머지는 `-100` 으로 무시. Ch 20 에서 손에 익힌 그 패턴이 그대로 재등장합니다.

> **"가린다" 의 의미가 바뀝니다.** BERT 에서 마스킹은 *표현을 배우기 위한 수단* (15% 만 살짝 가려 문맥으로 복원). Diffusion 에서 마스킹은 *생성 그 자체* — 100% 가린 `[MASK]` 시퀀스에서 출발해 한 step 씩 단어를 채우면, 그게 *무에서 문장을 만들어내는 것* 입니다. **같은 메커니즘, 다른 목적.** 가리는 비율을 끝까지 밀어붙였더니 *복원이 생성이 되었습니다.*

이 챕터에서 학습 collator 는 매 배치마다 $t$ 를 새로 뽑아 *가변 비율* 로 가립니다. Ch 20 의 `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` 가 *고정 15%* 였다면, 여기서는 *직접 만든 가변 collator* 가 *0-100% 를 매번 다르게* 가립니다.

## 토크나이저 노트 — 작은 모델엔 작은 vocab (BPE 2048 + `[MASK]`)

Diffusion LM 의 주인공 토큰은 **`[MASK]`** 입니다. 전부 `[MASK]` 인 빈 캔버스에서 시작해 토큰을 채우는 게 곧 생성이니까요.

그런데 작은 from-scratch 모델에는 **작은 vocab** 이 중요합니다. `bert-base-uncased` 의 WordPiece 는 vocab 이 30,522 개라, hidden 256 짜리 작은 모델에 그대로 붙이면 임베딩 테이블이 파라미터의 대부분을 잡아먹어 정작 문맥을 배우는 본체에 쓸 용량이 없습니다. 그래서 이 챕터는 Ch 19·24 처럼 **TinyStories 코퍼스에 ByteLevel BPE 2048 을 직접 학습** 하고, 거기에 `[PAD]`·`[UNK]`·`[MASK]` 특수 토큰을 더해 씁니다.

| 토크나이저 | vocab | `[MASK]` | 본 챕터 적합성 |
|---|---|---|---|
| WordPiece (`bert-base-uncased`) | 30,522 | 내장 | 작은 모델엔 임베딩 과대 |
| BPE (GPT-2) | 50,257 | 없음 | 별도 추가 필요 |
| **ByteLevel BPE (직접 학습)** | **2048** | **추가** | **작은 모델에 딱 — 본 챕터** |

`[MASK]` 토큰이 *forward process (가리기)* 와 *reverse process (복원/생성)* 양쪽의 핵심입니다.

> Ch 1 부터 추적한 *토크나이저 시각* 의 클라이맥스 — *같은 문장이 어떻게 토큰화되는가* 를 넘어, 이제 *`[MASK]` 토큰 자체가 생성의 캔버스* 가 됩니다.

## 환경 셋업

## TinyStories 데이터 로드

Ch 24 (GPT) 와 *완전히 같은 데이터* — `roneneldan/TinyStories` (Eldan & Li 2023, arXiv:2305.07759). GPT-3.5 / GPT-4 가 *4세 어린이 어휘* 로 생성한 짧은 영어 동화. 어휘·문법이 단순해 작은 모델로도 의미 있는 생성이 가능합니다.

*데이터를 Ch 24 와 동일* 하게 둔 이유: 나중에 *같은 데이터에서 AR (Ch 24) vs Diffusion (본 챕터) 생성 방식만 다른* 비교를 하기 위함입니다.

학습 split 의 처음 **30,000 stories** 만 사용 (T4 30분 룰 안).

## ByteLevel BPE 2048 직접 학습 + `[MASK]` 추가

Ch 19·24 처럼 TinyStories 코퍼스에 ByteLevel BPE 를 vocab 2,048 으로 직접 학습하고, `[PAD]`·`[UNK]`·`[MASK]` 특수 토큰을 더해 씁니다. 핵심은 `[MASK]` 토큰의 존재 (직접 학습이라 `special_tokens` 로 명시 추가).

**관전 포인트** — `[MASK]` 가 섞인 시퀀스가 바로 diffusion 의 *중간 상태* $x_t$ 입니다. 학습은 *가려진 자리를 맞히는 것*, 생성은 *전부 `[MASK]` 에서 시작해 반복적으로 채우는 것*. Ch 20 의 MLM 과 토큰 수준에서는 똑같이 생겼습니다 — 차이는 *마스킹 비율* 과 *반복 횟수*.

## 토큰화 + `group_texts` (고정 길이 블록 스트림)

Ch 20·24 와 같은 전처리 패턴 — 전체 코퍼스를 토큰화해 이어 붙이고 `block_size=128` 단위로 자릅니다. 특수 토큰 (`[CLS]`, `[SEP]`) 은 넣지 않고 *순수 텍스트 스트림* 으로 만듭니다 (diffusion 은 문장 전체를 한 캔버스로 다루므로 경계 토큰이 불필요).

## Diffusion collator — *가변 비율* 마스킹 직접 구현

여기가 BERT MLM 과 갈리는 지점입니다. Ch 20 은 `DataCollatorForLanguageModeling(mlm_probability=0.15)` 로 *고정 15%* 를 가렸지만, diffusion 은 **매 샘플마다 $t \sim U(\epsilon, 1)$ 을 뽑아 그 비율로** 가립니다.

- 각 토큰을 *독립적으로 확률 $t$* 로 `[MASK]` 치환 (LLaDA 의 forward process 와 동일)
- `labels`: 가려진 자리는 원본 토큰 id, 나머지는 `-100` (Ch 20 의 `-100` 트릭 그대로)
- `t`: $1/t$ 재가중을 위해 샘플별 비율도 함께 반환

`add_special_tokens=False` 로 토큰화했으므로 시퀀스 안에 특수 토큰이 없어 *모든 자리가 마스킹 가능* 합니다.

**관전 포인트** — Ch 20 MLM collator 가 *항상 약 15%* 를 가렸다면, 이 collator 는 *호출마다 0-100% 사이 아무 값* 으로 가립니다. 같은 chunk 가 어떤 step 엔 5% 만, 다른 step 엔 90% 가려진 채 학습됩니다 → 모델이 *모든 난이도의 복원* 을 골고루 학습 → 생성 시 *어떤 마스킹 비율에서도* denoise 가능.

> **`-100` thread**: 가려진 자리만 `labels`, 나머지는 `-100`. Ch 20 (MLM 15%) → Ch 28 (SFT, prompt 만 `-100`) → 본 챕터 (가변 마스킹) — 같은 트릭의 세 번째 변주.

## 작은 BERT-style 모델 from scratch

diffusion 의 본체는 *bidirectional encoder* — 가려진 자리를 *좌·우 양방향 문맥* 으로 복원해야 하니 BERT 계열이 자연스럽습니다. `BertForMaskedLM` 을 *random init* 으로 작게 띄웁니다 (Ch 20 의 작은 BERT 와 같은 패턴).

- `num_hidden_layers=4, num_attention_heads=4, hidden_size=256` → 약 3.79M params (작은 vocab 2048 덕분에 임베딩도 가벼움)
- `max_position_embeddings = BLOCK_SIZE = 128`
- MLM head (`Linear(H, V)`) 가 *가려진 자리의 토큰 분포* 를 출력 — 이게 곧 diffusion 의 denoiser

### GPT (Ch 24) 와 코드로 갈리는 곳

- `GPT2LMHeadModel` 이 아니라 `BertForMaskedLM` — *causal mask 없는 bidirectional attention*
- 같은 `from_pretrained` 없이 `BertForMaskedLM(config)` random init — Ch 20·22 와 동일

## Reverse process — 병렬 denoise 생성 함수

diffusion 생성의 핵심. **전부 `[MASK]` 인 시퀀스에서 시작**해 여러 step 에 걸쳐 점점 진짜 토큰으로 채웁니다 (LLaDA 의 *low-confidence remasking* 방식):

1. 현재 `[MASK]` 자리들을 모델이 *한꺼번에* 예측 (병렬!)
2. 각 예측의 *confidence* (softmax 최대 확률) 계산
3. *확신 높은* 자리부터 확정, *확신 낮은* 자리는 다시 `[MASK]` 로 남김
4. 스케줄에 따라 남기는 `[MASK]` 수를 step 마다 줄여 마지막엔 0개

GPT 의 *왼→오 순차* 와 결정적으로 다른 점: **채우는 순서가 위치가 아니라 confidence 순** — 문장 중간이나 끝 단어가 앞 단어보다 먼저 확정될 수 있습니다.

## 학습 *전* denoise - 비교 기준선 (random init baseline)

학습 전 모델은 가려진 자리를 *균등 추측* 하니, denoise 결과가 *의미 없는 토큰 나열* 이 나옵니다. 학습 후와 나란히 비교하기 위한 기준선 (Ch 20·22 의 *사전학습 전 [MASK] top-5*, Ch 24 의 *학습 전 generation* 과 같은 역할).

**관전 포인트** - 학습 전엔 *영어 문장과 거리가 먼 토큰 나열*. logits 가 random 이라 confidence 순서도 무의미. 학습 후 같은 함수로 다시 생성해 비교하면 *diffusion 학습이 본체에 무엇을 새겼는가* 가 드러납니다.

## `Trainer` 로 diffusion 학습 — `1/t` 재가중 loss

BERT/GPT 챕터들과 같은 `Trainer` 패턴이지만, *loss 를 직접 정의* 합니다. `BertForMaskedLM` 의 기본 loss 는 *가려진 자리 CE 평균* 인데, diffusion 은 거기에 *샘플별 `1/t` 재가중* 을 더해야 합니다 (`compute_loss` 오버라이드).

- `DiffusionCollator` → 매 배치 가변 마스킹 + `t` 반환
- `compute_loss` → 가려진 자리 CE 를 샘플별로 합산해 `1/t` 곱한 뒤 평균
- `max_steps=30000`, `batch_size=64`, `fp16=True` - T4 약 19분

**관전 포인트** - `1/t` 재가중 덕분에 첫 step loss 가 약 7.6 (`ln(2048)`) 부근에서 시작 (직접 학습한 BPE 2048 의 random baseline 과 같은 값!). 빠르게 떨어져 30000 step 끝에 *약 3.7* 부근에서 안정화되면 정상. 작은 모델 + TinyStories 라 완벽하진 않지만 *가려진 자리를 문맥으로 복원* 하는 능력이 본체에 새겨집니다.

## 학습 *후* denoise + 궤적 시각화

같은 `diffusion_generate` 로 학습 후 생성하고, **denoise 궤적** (각 step 의 시퀀스) 을 출력해 *마스크가 단어로 채워지는 과정* 을 직접 봅니다. 이게 이 챕터의 하이라이트 — GPT 의 왼→오 순차와 달리, *문장 전체가 동시에 흐릿하게 떠오르다 선명해지는* 모습.

**해석 가이드 - 이게 autoregressive 와 결정적으로 다른 점**

- **step 0**: 거의 전부 `____` (`[MASK]`). 모델이 *가장 확신하는* 몇 자리만 먼저 채워짐 — *위치 순서가 아니라 confidence 순서*. 문장 끝/중간 단어가 앞보다 먼저 나타날 수 있음.
- **중간 step**: 단어들이 *여기저기 동시에* 떠오름. GPT 라면 왼쪽부터 한 칸씩 채워졌을 자리가, diffusion 에선 *전 영역이 함께* 선명해짐.
- **마지막 step**: 모든 `[MASK]` 가 채워진 완성 문장.

> Ch 24 의 GPT generation 이 *왼→오 받아쓰기* 였다면, 여기선 *흐릿한 전체 그림을 반복적으로 다듬기*. 같은 TinyStories 데이터, 같은 "다음 단어가 뭘까" 직관이지만 *생성 메커니즘이 근본적으로 다릅니다.*

## 솔직한 이야기 — 생성은 되지만 *반복* 이 보인다

학습이 끝난 모델은 전부 `[MASK]` 에서 출발해도 *영어 동화* 를 만들어냅니다 — 인물·대화·배경이 있는 문장이 병렬 denoise 로 채워집니다. 다만 자세히 읽어 보면 **같은 조각이 반복** 되는 게 눈에 띕니다.

> *"Once upon a time, there was a **a** boy named **named** Timmy. ... They are happy friends and happy. They are **to play and play**."*

`named named`, `was a was a`, `play and play` 처럼요. 이건 *모델이 잘못 배운 게 아닙니다.* 고정-$t$(0.15) 복원 정확도가 0.7 안팎까지 오른, 조건부 구조를 제대로 익힌 모델입니다. 반복의 원인은 **샘플러** 에 있습니다.

- 이 챕터의 기본 샘플러는 매 step *confidence 가 높은 자리를 채우고 낮은 자리를 다시 `[MASK]`* 로 두는 방식인데, 한번 "안전한" 고빈도 토큰(`a`, `the`, 자주 나오는 이름)이 높은 confidence 를 받으면 그 토큰이 거듭 뽑히기 쉽습니다.
- 즉 *모델의 확률 분포는 멀쩡한데, 거기서 문장을 어떻게 뽑아내느냐* 가 아직 거친 것입니다.

> 그래서 **다음 Ch 33 은 모델은 그대로 두고 샘플러만 바꿉니다** — carry-over semi-AR + 반복 억제(temperature·top-p·repetition penalty·인접 중복 금지)로 이 반복을 잡아 한결 깔끔한 생성을 얻습니다. 이 챕터에서 "diffusion 이 글을 만든다"를 확인했다면, 다음 챕터는 "그 글을 더 잘 뽑아낸다"입니다.

## Autoregressive (Ch 24) vs Diffusion (본 챕터) 비교

같은 TinyStories, 같은 "언어모델" 이지만 생성 메커니즘이 근본적으로 다릅니다.

| 축 | Autoregressive (GPT, Ch 24) | Diffusion (본 챕터) |
|---|---|---|
| attention | causal (과거만) | **bidirectional (양방향)** |
| 생성 순서 | 왼→오 *위치 순* | **confidence 순 (위치 무관)** |
| 생성 step | 토큰 수 = step (고정) | **임의 (1-32+ 조절)** |
| 병렬성 | 생성 시 순차 (느림) | **여러 자리 동시 생성 (잠재적 고속)** |
| infilling (중간 채우기) | 구조적으로 어려움 | **자연스럽게 가능** (양방향) |
| 출발 상태 | prompt | **전부 `[MASK]`** |
| 성숙도 | 표준 (대부분의 LLM) | **신생 (LLaDA, Trida 등 등장 중)** |

> **왜 diffusion 이 주목받는가**: ① *병렬 생성* 으로 잠재적 속도 이점 (autoregressive 는 토큰 수만큼 순차), ② *양방향 문맥* 으로 infilling·편집에 강점, ③ step 수로 *속도-품질* 을 추론 시점에 조절. 아직 autoregressive 만큼 성숙하진 않지만 *대안 패러다임* 으로 빠르게 발전 중입니다. Ch 33 에서 *사전학습된 작은 diffusion LM (MDLM 170M / DiffuGPT 124M)* 으로 제대로 된 생성을, Ch 34 에서 *한국어 diffusion + AR 직접 비교* 를 다룹니다.

## 이 챕터 알고리즘의 논문 계보

본 챕터에서 *직접 구현* 한 세 요소는 아래 논문들의 방법을 *교육용으로 단순화* 해 옮긴 것입니다. 어느 요소가 어느 논문의 무엇에 대응하는지 정리합니다.

| 구현 요소 (본 챕터) | 대응 논문·수식 | 일치 |
|---|---|---|
| 가변 마스킹 forward (`t ~ U(0,1)`, 토큰별 독립 마스킹) | **LLaDA** Eq. 8 / **D3PM** absorbing-state(=mask) kernel | 동일 |
| `1/t` 재가중 denoising loss (가린 자리 CE 합을 `t·L` 로 정규화) | **LLaDA** Eq. 3 = $-\mathbb{E}[\frac{1}{t}\sum_i \mathbb{1}[x_t^{(i)}{=}\texttt{M}]\log p_\theta]$ / **MDLM** weighted MLM-CE (NELBO) | 동일 |
| low-confidence remasking 생성 (전부 `[MASK]` 시작 → confidence 낮은 자리만 유지) | **LLaDA** sampling (low-confidence remasking) / **MaskGIT** confidence 병렬 디코딩 | 동일 |

> 참고로 LLaDA 논문의 loss 는 본문 수식엔 `1/L` 이 없지만 *구현(Algorithm 1)에서 `t·L` 로 정규화* 합니다. 본 챕터 코드의 `per_tok.sum()/L` 후 `/t` 평균이 정확히 `sum/(t·L)` 으로 *구현 레벨까지 일치* 합니다. 이 loss 는 *negative log-likelihood 의 upper bound* (LLaDA Eq. 4).

### 읽는 순서 추천 (계보)

1. **D3PM** — Austin et al. 2021, [arXiv:2107.03006](https://arxiv.org/abs/2107.03006). 이산 diffusion + *absorbing(=mask) 상태*. 이론 시초.
2. **MaskGIT** — Chang et al. 2022, [arXiv:2202.04200](https://arxiv.org/abs/2202.04200). *confidence 기반 반복 병렬 디코딩* — 본 챕터 생성 절차의 원조 (원래 이미지 분야).
3. **MDLM** — Sahoo et al. 2024, [arXiv:2406.07524](https://arxiv.org/abs/2406.07524). masked diffusion loss = *"고전 MLM loss 들의 가중 혼합"* (NELBO). 본 챕터 `1/t` 재가중의 이론 근거.
4. **LLaDA** — Nie et al. 2025, [arXiv:2502.09992](https://arxiv.org/abs/2502.09992). 위를 *LLM 스케일* 로. **본 챕터가 직접 따른** forward·loss·sampling. 8B 라 Ch 33 의 *대형 맛보기(선택)* 로 다룹니다.

> ⚠️ **혼동 주의** — **Diffusion-LM** (Li et al. 2022, [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)) 은 이름은 비슷하지만 *연속 임베딩 공간* 에서 Gaussian noise 를 더하는 diffusion 이라 본 챕터의 *이산 mask-diffusion* 과 **다른 계열** 입니다. Ch 33 (MDLM/DiffuGPT)·34 는 본 챕터와 같은 이산 mask-diffusion.

> 본 챕터는 *단순화판* 입니다 — 실제 LLaDA 는 semi-autoregressive remasking 등 변형, 대규모 사전학습, 정교한 스케줄을 더합니다. 하지만 *핵심 메커니즘 (가변 마스킹 + `1/t` loss + confidence 병렬 denoise)* 은 동일하므로, 본 챕터를 손으로 구현해 보면 위 논문들의 알고리즘 절을 그대로 읽어낼 수 있습니다.

## 이 장의 구성

[[SubPages]]
