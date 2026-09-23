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
| 31 | SFT base + GRPO | Character BPE | verifiable-reward | `Linear(H, V)` + group adv. | autoregressive + RL | `GRPO loss` |
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

## 이 장의 구성

[[SubPages]]
