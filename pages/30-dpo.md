**목표**: Phase 4 의 *학습 단계 4 (Alignment, 선호 정렬)* 의 첫 챕터. Ch 28 에서 **SFT** 로 KoGPT2 를 *지시를 따르게* (행동 정렬) 만들었고, Ch 29 에서 *능력을 벤치마크로 측정* 했습니다. 이제 **사람의 선호에 맞춰 정렬 (alignment)** 합니다. **DPO (Direct Preference Optimization)** 는 SFT 모델을 *preference 쌍 (chosen / rejected)* 으로 학습 — *좋은 답의 확률은 올리고, 나쁜 답의 확률은 내립니다*. 바뀌는 건 **데이터 (instruction-response → preference 쌍)** + **trainer (`SFTTrainer` → `trl.DPOTrainer`)** + **loss (next-token CE → DPO sigmoid)** + **frozen reference 모델 추가** 입니다.

**환경**: Google Colab **T4 GPU 필수**. policy + reference *두 모델* 을 동시에 올리므로 batch 를 작게 + gradient accumulation 으로 VRAM 을 관리합니다.

**예상 소요 시간**: 약 22-30분 (preference 데이터 로드·필터 약 2분 + SFT 모델 로드 약 2분 + DPO loss 직관 시각화 약 1분 + DPOTrainer 학습 약 15-22분 + DPO 전·후 reward margin 비교 약 3분)

## 학습 흐름

1. 📊 **누적 추적표** (Ch 27/28/29 + **30 강조** + Ch 31 예고) + GPT 학습 4단계 표 (Ch 30 = 단계 4 alignment, DPO)
2. 🔄 **변경점 (Diff from Ch 28 SFT)** — *데이터 + trainer + loss + reference 모델* 이 변함
3. 🎯 **alignment 의 의미** — SFT (지시 따름) → alignment (선호·품질 정렬). RLHF 흐름 + DPO 가 PPO 간소화인 이유
4. 📐 **DPO Loss** — 수식 + 직관 + 수치 예시 표. β 의 역할, frozen reference 가 필요한 이유
5. 🎯 **`labels = -100` thread 연결** — DPO 도 *response 부분만* log-prob 계산
6. 🔤 **토크나이저 노트** — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 이후 고정)
7. 🚀 **실습**: preference 데이터 로드 → SFT 모델·reference 준비 → **DPO loss 직관 시각화 (margin)** → `DPOTrainer` 학습 → DPO 전·후 reward margin 비교
8. 📦 **등장 라이브러리** (`trl.DPOTrainer`·`DPOConfig` 첫 등장) / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)

> 📒 **사전 학습 자료**: Ch 28 (KoGPT2 SFT — 본 챕터의 *출발 모델*), Ch 29 (벤치마크 평가), Ch 27 (KoGPT2 토크나이저 함정). 본 챕터는 *alignment 의 두 thread 연장*: (1) `labels = -100` 의 *response-only* 가 DPO 의 log-prob 계산에서도 이어지고, (2) "파인튜닝" 의 의미가 *행동 정렬 (SFT)* 에서 *선호 정렬 (alignment)* 로 한 발 더 나아갑니다.

## 누적 추적표

| Ch | 모델 | 데이터 | 학습 신호 | Loss | Trainer |
|---|---|---|---|---|---|
| 27 | KoGPT2 (125M) | 한국어 TinyStories 30K | next-token | `CrossEntropyLoss` - continual pretraining | `Trainer` |
| 28 | KoGPT2 (125M, SFT) | KoAlpaca instruction-response 쌍 | response 토큰 (답변만) | `CrossEntropyLoss` (response-only) - SFT | `SFTTrainer` |
| 29 | Ch 28 SFT 모델 (평가) | 분야별 벤치마크 | - (평가만) | - (`lm-evaluation-harness`) | - |
| **30 ← 여기** | **SFT 모델 (policy) + frozen reference** | **preference 쌍 (chosen / rejected)** | **chosen 선호 ↑ / rejected 선호 ↓** | **DPO sigmoid loss (β=0.1)** | **`DPOTrainer`** |
| 31 (다음) | SFT 모델 + verifier | verifiable-reward prompts (수학·코드) | group relative advantage | `GRPO loss` | `GRPOTrainer` |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.

## GPT 시대 학습 4단계 — 본 챕터의 위치 (단계 4, Alignment / DPO)

Ch 24 에서 도입한 GPT 시대 학습 4단계 표. 본 챕터는 *단계 4 (Alignment)* — *사람의 선호에 맞춰 정렬* 하는 마지막 단계의 첫 방식 (DPO) 입니다.

| 단계 | 정확 용어 | 의미 | 학습 신호 | 본 커리큘럼 | 본 챕터? |
|---|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | random init 본체 + 일반 코퍼스 | next-token | Ch 24 (영어), Ch 26 (한국어) | |
| 2 | **Continual pretraining** (계속 사전학습) | 사전학습 본체 + 새 데이터 | next-token | Ch 25 (영어), Ch 27 (한국어) | |
| 3 | **SFT** (Supervised Fine-Tuning) | instruction-response 로 *행동 정렬* | response 토큰 | Ch 28 | |
| 4 | **Alignment** (DPO / RLHF / GRPO) | preference·verifier reward 로 *선호 정렬* | preference 쌍 / reward | **Ch 30 (DPO) ← 여기**, Ch 31 (GRPO) | ✅ |

### 단계 3 (SFT) → 단계 4 (Alignment) 의 결정적 변화

- **단계 3 (SFT)**: *"좋은 답변 하나"* 를 따라 학습 (정답 demonstration 모방). 모델은 *지시를 따르는 법* 을 배웁니다 — 하지만 *"여러 답변 중 어느 게 더 나은가"* 라는 *선호* 는 가르치지 못합니다
- **단계 4 (Alignment / DPO)**: *"같은 질문에 좋은 답 vs 나쁜 답"* 쌍으로 학습. 모델은 *사람이 선호하는 방향* 으로 정렬됩니다 — *더 도움되고, 더 안전하고, 더 품질 높은* 답 쪽으로

> SFT 가 *지시를 따르게* 했다면, alignment 는 *따르는 방식을 사람의 선호에 맞춥니다*. DPO 는 그 alignment 를 *reward model 없이, preference 쌍으로 직접* 해내는 방식입니다 — RLHF (PPO) 의 간소화. 그 핵심은 아래 §4 의 DPO loss 한 줄입니다.

## 변경점 (Diff from Ch 28 SFT)

| 축 | Ch 28 (KoGPT2 SFT) | Ch 30 (본 챕터, DPO) |
|---|---|---|
| 본체 | KoGPT2 `skt/kogpt2-base-v2` (125M) | **SFT 모델 (= KoGPT2 SFT 산출) 을 policy 로** ← 출발점이 SFT 모델 |
| 토크나이저 | `PreTrainedTokenizerFast` (KoGPT2 BBPE) | **(동일)** ← 고정 |
| **데이터** | instruction-response 쌍 (`prompt` / `completion`) | **preference 쌍 (`prompt` / `chosen` / `rejected`)** ← *변화 1* |
| **Trainer** | `trl.SFTTrainer` | **`trl.DPOTrainer`** ← *변화 2* (새 클래스, 첫 등장) |
| **Loss** | next-token `CrossEntropyLoss` (response-only) | **DPO sigmoid loss** ← *변화 3* (log-likelihood ratio) |
| **reference 모델** | 없음 (policy 하나) | **frozen reference 추가** ← *변화 4* (SFT 모델 복사 + freeze) |
| 학습 신호 | 좋은 답변 하나 *모방* | **chosen 선호 ↑, rejected 선호 ↓** (*비교*) |
| lr | 2e-5 | **5e-6 - 1e-5** ← DPO 는 SFT 보다 작은 lr (reference 에서 천천히 벗어남) |

> **핵심**: SFT 는 *하나의 좋은 답* 을 모방했다면, DPO 는 *(좋은 답, 나쁜 답) 쌍* 을 *비교* 합니다. 그러려면 *(1) preference 데이터*, *(2) 비교를 loss 로 바꾸는 DPOTrainer*, *(3) "원본에서 얼마나 벗어났나" 의 기준이 되는 frozen reference* 가 필요합니다. 네 가지가 한꺼번에 바뀌지만, *목적은 하나* — *모델을 사람이 선호하는 방향으로* 정렬.

## alignment 의 의미 — SFT(지시 따름) 에서 선호·안전성·품질 정렬로

**alignment (정렬)** 은 모델의 행동을 *사람이 원하는 방향* 에 맞추는 단계입니다. SFT 와의 차이를 한 줄로:

- **SFT**: *지시를 따르게* 만든다 (행동 정렬). "질문이 오면 답하라"
- **Alignment**: *따르는 방식을 사람의 선호에 맞춘다* (선호 정렬). "기왕 답할 거면, 더 도움되고·안전하고·품질 높게"

### RLHF 흐름 — 그리고 DPO 가 그 간소화인 이유

전통적인 **RLHF (Reinforcement Learning from Human Feedback)** 는 세 단계입니다:

```
1. SFT          : instruction-response 로 base 모델을 지시 따르게 (Ch 28)
2. Reward Model : (chosen, rejected) preference 로 '점수 매기는 모델' 을 별도 학습
3. PPO          : reward model 을 보상으로 policy 를 강화학습 (RL)
```

PPO 단계는 *네 개의 모델* 을 동시에 메모리에 올립니다:

| 모델 | 역할 |
|---|---|
| **actor** (policy) | 학습 대상 — 답변을 생성 |
| **critic** (value) | 각 상태의 가치 추정 (PPO advantage 용) |
| **reward model** | 생성된 답변에 점수 |
| **reference** | KL 제약 기준 (원본에서 벗어남 측정) |

**T4 (16GB) 에 네 모델은 무리** 입니다. 그래서 본 커리큘럼은 PPO 대신 **DPO** 를 채택합니다.

### DPO = reward model 없이 preference 로 *직접* 정책 최적화

DPO 의 통찰: *"reward model 을 따로 학습한 뒤 RL 로 최적화"* 하는 두 단계를, *"preference 쌍에서 곧바로 policy 를 최적화"* 하는 **한 단계로 합칠 수 있다**. 수학적으로 *최적 정책과 reward 의 관계* 를 닫힌 형태로 풀면, reward model 을 명시적으로 만들 필요 없이 *preference 만으로* policy 를 직접 학습할 수 있습니다 (아래 §4 의 loss).

| 방식 | 필요한 모델 | 단계 | T4 적합성 |
|---|---|---|---|
| **PPO (RLHF)** | actor + critic + reward + reference (**4개**) | SFT → RM → PPO | ✗ (메모리 초과) |
| **DPO (본 챕터)** | policy + frozen reference (**2개**) | SFT → DPO | ✓ (batch 작게 + grad accum) |

> **DPO 는 PPO 대비 간단·안정** 합니다 — reward model 학습도, RL 루프도, critic 도 없습니다. *policy + frozen reference 두 모델* 만으로, *preference 쌍* 에서 *지도학습처럼* (loss.backward()) 정렬합니다. 그래서 T4 한 장에서도 alignment 를 *직접 손으로* 돌려볼 수 있습니다.

## DPO Loss — preference 를 log-likelihood ratio 로

DPO 의 loss 는 *chosen 의 (정책 대비 reference) log-prob 우위* 를 *rejected 보다 크게* 만듭니다:

$$L_{\text{DPO}} = -\log \sigma\!\Big( \beta \cdot \big[\, (\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)) - (\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)) \,\big] \Big)$$

- $y_w$ = chosen (좋은 답), $y_l$ = rejected (나쁜 답), $x$ = prompt
- $\pi_\theta$ = policy (학습 대상), $\pi_{\text{ref}}$ = frozen reference (SFT 모델 복사·freeze)
- $\sigma$ = sigmoid, $\beta$ = reference 에서 벗어나는 정도 제어 (KL 제약 역할, 기본 0.1)

### 직관 — 두 개의 "정책 대비 reference 우위" 를 비교

각 답변에 대해 **"정책이 reference 보다 이 답변을 얼마나 더 좋아하나"** 를 측정합니다:

$$r_\theta(x, y) = \log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x) \qquad (\text{= implicit reward})$$

DPO 는 이 *implicit reward* 가 *chosen 에서 rejected 보다 크도록* 학습합니다. **margin** $= r_\theta(x, y_w) - r_\theta(x, y_l)$ 가 클수록 loss 가 작아집니다 (sigmoid → 1 → $-\log 1 = 0$).

### 수치 예시 — margin 이 loss 에 어떻게 (β=0.1)

implicit reward 차이 (margin) 가 커질수록 loss 가 어떻게 줄어드는지 (β·margin 을 sigmoid 에 넣고 $-\log$):

| 상황 | margin $= r_\theta(y_w) - r_\theta(y_l)$ | β·margin | $\sigma(\beta \cdot \text{margin})$ | $L = -\log \sigma$ |
|---|---|---|---|---|
| chosen 이 rejected 보다 *훨씬* 선호됨 | +20 | +2.0 | 0.881 | **0.127** (낮음 ✓) |
| chosen 이 rejected 보다 약간 선호됨 | +5 | +0.5 | 0.622 | **0.474** |
| 둘이 비슷 (정렬 안 됨) | 0 | 0.0 | 0.500 | **0.693** |
| rejected 가 *더* 선호됨 (틀림!) | −10 | −1.0 | 0.269 | **1.313** (높음 ✗) |

> *chosen 의 우위가 클수록 loss ↓*, *역전되면 loss 가 폭증* 합니다. 학습은 자연히 *chosen 의 implicit reward 를 올리고 rejected 를 내리는* 방향으로 흐릅니다. **§3 에서 실제 KoGPT2 로 margin 을 손으로 계산** 해 봅니다.

### β 의 역할 — reference 에서 벗어나는 정도

- **β 큼** (예: 0.5): reference 제약이 *강함* → policy 가 reference 근처에 묶여 *안전하지만 정렬이 느림* (trl 공식 문서: *Higher β means less deviation from the reference model*)
- **β 작음** (예: 0.05): reference 제약이 *느슨* → policy 가 preference 에 강하게 끌려가 *빨리 정렬되지만* reference 에서 멀어져 *collapse (degeneration)·reward hacking* 위험
- 기본값 **0.1** 이 무난한 출발점

### 왜 frozen reference 가 필요한가

reference 가 없으면 (또는 β=0), 모델은 *chosen 의 확률을 무한정 올리고 rejected 를 0 으로* 밀어붙입니다 — 그 과정에서 *원본 SFT 의 일반 능력이 collapse* 합니다 (한 패턴만 반복하거나, 문법이 무너지는 등). **frozen reference 는 "원본에서 너무 멀어지지 마라" 는 닻** 입니다:

- $\log \pi_\theta - \log \pi_{\text{ref}}$ 가 *상대적* 비교 → policy 가 reference 근처에 머물도록 KL 제약을 거는 효과
- *정렬하면서도 SFT 의 능력을 보존* — reward hacking·degeneration 방지

> reference 는 *SFT 모델을 복사해 freeze* 한 것입니다 (gradient 안 흐름). 학습 중 *policy 만 움직이고 reference 는 고정* 되어, 둘의 log-prob 차이가 *"얼마나 멀어졌나"* 의 기준이 됩니다.

## `labels = -100` thread 연결 — DPO 도 *response 부분만*

Ch 28 SFT 의 핵심은 *prompt 를 `-100` 으로 가리고 response 부분만* loss 를 계산하는 것이었습니다. **DPO 도 똑같이 response 부분만 봅니다.**

DPO loss 의 $\log \pi(y \mid x)$ 는 *답변 $y$ 의 토큰들에 대한 log-likelihood 합* 입니다 — **prompt $x$ 부분은 제외**. chosen 도, rejected 도 *각자의 response 토큰에서만* log-prob 을 더합니다 (prompt 는 양쪽 공통이라 비교에서 상쇄되기도 하고, 애초에 학습 대상이 아님).

| 단계 | 챕터 | response-only log-prob 계산 자리 |
|---|---|---|
| MLM | Ch 20·21·22 | 가린 약 15% 토큰 |
| CausalLM | Ch 24·25·26·27 | 거의 전 토큰 (pad 만 제외) |
| **SFT** | Ch 28 | **response 부분만** (prompt = `-100`) |
| **DPO (본 챕터)** | **Ch 30** | **chosen / rejected 각각의 response 부분만** (prompt 제외) |

```
prompt:   ### 명령어:\n건강한 식습관을 알려줘\n\n### 응답:\n   <- 양쪽 공통, log-prob 계산 제외
chosen:   규칙적인 식사와 채소 섭취가 중요합니다.            <- 이 부분의 log π_θ, log π_ref
rejected: ㄴㄴ 몰라 아무거나 먹어                          <- 이 부분의 log π_θ, log π_ref
```

> `labels = -100` thread 가 *alignment 단계까지* 이어집니다. SFT 에서 "답변 부분만 학습" 이었다면, DPO 에서는 "답변 부분의 log-prob 만 비교". **prompt 는 늘 조건 (given), 답변만 학습·비교 대상 (target)** 이라는 원리가 Phase 4 전체를 관통합니다. `DPOTrainer` 가 이 마스킹을 자동으로 처리하므로 우리가 직접 `-100` 을 찍을 필요는 없습니다 (§3 에서 그 효과를 *손으로 재현* 해 확인).

## 토크나이저 노트 — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 이후 고정)

본 챕터의 토크나이저는 *Ch 27·28 과 완전히 동일*. KoGPT2 BBPE (vocab 51,200) 를 그대로 가져옵니다. **KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정** 이 있어 (Ch 27 §토크나이저 노트), `PreTrainedTokenizerFast` + special token 명시로 로드합니다.

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
```

### preference 데이터의 토큰화 — prompt / chosen / rejected

DPO 데이터는 *세 개의 텍스트* 로 구성됩니다 (`prompt`, `chosen`, `rejected`). `DPOTrainer` 는 내부적으로:

1. `prompt + chosen` 과 `prompt + rejected` 를 *각각* 토큰화
2. 두 시퀀스의 *prompt 부분은 공통* (같은 토큰), *response 부분만 다름*
3. response 부분의 토큰에서 log-prob 을 계산 (위 §의 response-only)

> 같은 KoGPT2 토크나이저이므로 *Ch 28 SFT 에서 본 instruction 포맷 토큰화* 가 그대로 적용됩니다. chosen / rejected 는 *같은 prompt 에 대한 다른 답변* 이라 *prompt 토큰열은 완전히 동일*, 답변 토큰열만 갈립니다 — DPO 가 비교하는 건 정확히 그 *답변 토큰열의 log-prob* 입니다.

토크나이저는 Ch 27 이후 *Phase 4 내내 고정* — Ch 31 (GRPO) 에서도 같은 KoGPT2 토크나이저를 씁니다.

## 이 장의 구성

[[SubPages]]
