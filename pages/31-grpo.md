**목표**: Phase 4 의 *마지막 챕터*. Ch 30 에서 **DPO** 로 *사람/AI 가 비교한 preference 쌍 (chosen / rejected)* 으로 정렬했다면, 본 챕터는 alignment 의 *두 번째 방식* — **GRPO (Group Relative Policy Optimization)** 입니다. GRPO 는 정반대 접근으로, *정답을 자동 검증(verifier) 할 수 있는 task (수학·코드)* 에서 모델이 *여러 답을 생성(rollout)* 하고 *verifier 가 채점* 해 *잘한 답 방향* 으로 강화학습 합니다. **DeepSeek-R1 이 순수 RL 로 reasoning 능력을 끌어낸 방법** 이 바로 이것입니다. 바뀌는 건 **신호 출처 (preference 쌍 → verifier reward)** + **trainer (`DPOTrainer` → `trl.GRPOTrainer`)** + **데이터 (chosen/rejected → prompt + 정답)** + **rollout (한 prompt 에 여러 답 생성)** 입니다.

**환경**: Google Colab **T4 GPU 필수**. GRPO 는 *매 step 여러 답을 생성(rollout)* 하므로 무겁습니다 — group size 를 작게 (4) + 짧은 generation + 작은 step 으로 시간을 통제합니다.

**예상 소요 시간**: 약 22-30분 (verifiable 데이터 준비 약 1분 + SFT 모델 로드 약 2분 + verifier·group advantage 손계산 시연 약 2분 + `GRPOTrainer` 학습 약 15-22분 + GRPO 전·후 정확도(verifier pass rate) 비교 약 3분)


## 학습 흐름

1. 📊 **누적 추적표** (Ch 28/29/30 + **31 강조**) + GPT 학습 4단계 표 (Ch 30 DPO·Ch 31 GRPO = 단계 4 alignment 의 두 방식)
2. 🔄 **변경점 (Diff from Ch 30 DPO)** — *신호 출처 + trainer + 데이터 + rollout* 이 변함
3. 🎯 **PPO vs DPO vs GRPO 대비 표** — 신호·모델·데이터. *왜 GRPO 가 critic 도 reward model 도 없이 되나*
4. 📐 **GRPO 메커니즘** — rollout group → verifier reward → group relative advantage 수식 + 수치 예시
5. 🔬 **verifiable reward 의 의미** — 정답 있는 task 는 사람 채점 없이 무한 RL 신호. DeepSeek-R1 의 reasoning
6. 🔤 **토크나이저 노트** — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 이후 고정)
7. 🚀 **실습**: verifiable 데이터 → SFT 모델 → **verifier + group advantage 손계산** → `GRPOTrainer` 학습 → GRPO 전·후 정확도 비교
8. 📦 **등장 라이브러리** (`trl.GRPOTrainer`·`GRPOConfig`·`reward_funcs` 첫 등장) / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)
9. 🎓 **Phase 4 회고 + Phase 5 (Diffusion LM) 예고**


> 📒 **사전 학습 자료**: Ch 30 (DPO — alignment 의 첫 방식), Ch 28 (KoGPT2 SFT — 본 챕터의 *출발 모델*), Ch 29 (벤치마크 평가 — 특히 부록의 *pass@1·cons@64* 가 verifiable reward 와 직접 연결). 본 챕터는 *alignment 의 두 방식 비교* 를 완성합니다: **DPO (주관적 선호, 사람/AI 비교) vs GRPO (객관적 정답, 자동 검증)**.

## 누적 추적표

| Ch | 모델 | 데이터 | 학습 신호 | Loss | Trainer |
|---|---|---|---|---|---|
| 28 | KoGPT2 (125M, SFT) | KoAlpaca instruction-response 쌍 | response 토큰 (답변만) | `CrossEntropyLoss` (response-only) - SFT | `SFTTrainer` |
| 29 | Ch 28 SFT 모델 (평가) | 분야별 벤치마크 | - (평가만) | - (`lm-evaluation-harness`) | - |
| 30 | SFT 모델 (policy) + frozen reference | preference 쌍 (chosen / rejected) | chosen 선호 ↑ / rejected 선호 ↓ | DPO sigmoid loss (β=0.1) | `DPOTrainer` |
| **31 ← 여기** | **SFT 모델 (policy) + verifier** | **prompt + 정답 (검증 가능, 수학)** | **group relative advantage** | **GRPO loss (group baseline)** | **`GRPOTrainer`** |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.


## GPT 시대 학습 4단계 — 본 챕터의 위치 (단계 4, Alignment 의 두 번째 방식 / GRPO)

Ch 24 에서 도입한 GPT 시대 학습 4단계 표. 본 챕터는 *단계 4 (Alignment)* 의 *두 번째 방식* — Ch 30 DPO 와 Ch 31 GRPO 가 *alignment 의 두 방식* 입니다.

| 단계 | 정확 용어 | 의미 | 학습 신호 | 본 커리큘럼 | 본 챕터? |
|---|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | random init 본체 + 일반 코퍼스 | next-token | Ch 24 (영어), Ch 26 (한국어) | |
| 2 | **Continual pretraining** (계속 사전학습) | 사전학습 본체 + 새 데이터 | next-token | Ch 25 (영어), Ch 27 (한국어) | |
| 3 | **SFT** (Supervised Fine-Tuning) | instruction-response 로 *행동 정렬* | response 토큰 | Ch 28 | |
| 4 | **Alignment** (DPO / GRPO / RLHF) | preference·verifier reward 로 *선호·능력 정렬* | preference 쌍 / reward | Ch 30 (DPO), **Ch 31 (GRPO) ← 여기** | ✅ |

### alignment 의 두 방식 — DPO vs GRPO

- **DPO (Ch 30)**: *"같은 질문에 좋은 답 vs 나쁜 답"* preference 쌍으로 학습. 신호는 *사람/AI 가 비교* 한 *주관적 선호*. 열린 질문(글쓰기·대화·취향) 처럼 *정답이 없는* task 에 적합
- **GRPO (Ch 31, 본 챕터)**: *"이 답이 맞나"* 를 *자동 검증(verifier)* 해 reward. 신호는 *객관적 정답* (수학 답이 맞나, 코드가 테스트를 통과하나). *정답을 자동 확인할 수 있는* task 에 적합

> DPO 가 *사람의 선호* 를 따라간다면, GRPO 는 *정답이라는 객관 신호* 를 따라갑니다. 후자의 강점은 ***사람 채점 없이 무한히 RL 신호를 만들 수 있다*** 는 것 — 정답이 있는 task 라면 verifier 가 *공짜 reward model* 역할을 합니다. DeepSeek-R1 이 이걸로 *순수 RL 만으로 reasoning* 능력을 끌어냈습니다 (§6).

## 변경점 (Diff from Ch 30 DPO)

| 축 | Ch 30 (DPO) | Ch 31 (본 챕터, GRPO) |
|---|---|---|
| 본체 | SFT 모델 (policy) + frozen reference | **SFT 모델 (policy)** ← *reference 는 옵션* (β=0 이면 불필요) |
| 토크나이저 | `PreTrainedTokenizerFast` (KoGPT2 BBPE) | **(동일)** ← 고정 |
| **신호 출처** | preference 쌍 (사람/AI 가 비교) | **verifier reward (정답 자동 검증)** ← *변화 1* |
| **Trainer** | `trl.DPOTrainer` | **`trl.GRPOTrainer`** ← *변화 2* (새 클래스, 첫 등장) |
| **데이터** | `(prompt, chosen, rejected)` 쌍 | **`(prompt, 정답)`** ← *변화 3* (검증 가능한 task) |
| **rollout** | 없음 (주어진 쌍을 비교) | **한 prompt 에 여러 답 생성** ← *변화 4* (group rollout) |
| **Loss** | DPO sigmoid loss | **GRPO loss (group relative advantage)** ← *변화 5* |
| advantage baseline | - (쌍 비교) | **group 평균** (critic 대신) |

> **핵심**: DPO 는 *주어진 (좋은 답, 나쁜 답) 쌍* 을 *비교* 했다면, GRPO 는 *모델이 직접 여러 답을 생성* 하고 *verifier 가 채점* 해 *그룹 안에서 상대 비교* 합니다. 가장 큰 변화는 *신호의 출처* — *사람/AI 가 만든 preference* 에서 *정답 자동 검증* 으로. 그래서 *정답이 있는 task (수학·코드)* 라면 *사람 없이 무한히 RL 신호* 를 만들 수 있습니다.

## PPO vs DPO vs GRPO — alignment 의 세 갈래 (본 챕터의 뼈대)

alignment 의 세 방법을 *신호 출처·필요 모델·데이터* 로 정리합니다. GRPO 의 위치를 이 표 하나로 잡습니다.

| 방법 | 신호 출처 | 필요 모델 | 데이터 | T4 |
|---|---|---|---|---|
| **PPO** (전통 RLHF) | reward model 점수 | actor + critic + reward model + reference (**4개**) | prompt + 학습된 RM | ✗ (메모리 초과) |
| **DPO** (Ch 30) | preference 쌍 (사람/AI 비교) | policy + frozen reference (**2개**) | `(prompt, chosen, rejected)` | ✓ |
| **GRPO** (Ch 31, 본 챕터) | **verifier (정답 자동 검증)** | **policy 만** (+ 옵션 reference) | **`(prompt, 정답)` — 검증 가능** | ✓ |

### 왜 GRPO 는 critic 도 reward model 도 없이 되나 — *group 평균이 baseline*

전통 PPO 는 *advantage* 를 계산하려고 **critic (value model)** 을 따로 둡니다 — "이 상태에서 기대되는 reward 가 얼마인가" 의 *baseline* 을 추정하기 위해서입니다. advantage = (실제 reward) − (critic 이 예측한 baseline).

GRPO 의 통찰: **같은 prompt 에 답을 여러 개 (group) 생성하면, *그 group 의 평균 reward* 가 곧 baseline 이 된다.** critic 을 학습할 필요가 없습니다 — *그룹 동료들의 평균* 이 "이 prompt 에서 보통 어느 정도 받나" 를 알려주니까요.

| 항목 | PPO | GRPO |
|---|---|---|
| baseline (advantage 기준) | **critic (value model)** 이 예측 | **group 평균 reward** (동료 비교) |
| reward 출처 | **reward model** (별도 학습) | **verifier** (정답 자동 검증, 학습 불필요) |
| 필요 모델 | actor + critic + RM + ref (4) | **policy** (+ 옵션 ref) |

> **GRPO 는 PPO 의 또 다른 간소화** 입니다. DPO 가 *reward model + RL 루프* 를 *지도학습 한 단계* 로 줄였다면, GRPO 는 *critic 을 group 평균* 으로, *reward model 을 verifier* 로 대체합니다. 둘 다 *PPO 의 4 모델* 을 덜어내는 길이지만, GRPO 는 *RL 루프(rollout)는 유지* 하면서 *critic 과 RM 만* 없앤 점이 다릅니다 — 그래서 *정답이 있는 task* 에서 강력합니다.

## GRPO 메커니즘 — rollout group → verifier reward → group relative advantage

GRPO 의 한 step 은 네 단계입니다:

1. **rollout**: 한 prompt $x$ 에 대해 policy 가 **여러 답 (group)** $\{y_1, \dots, y_G\}$ 을 생성 (예: $G=4$)
2. **verifier reward**: 각 답을 verifier 로 채점 → reward $\{r_1, \dots, r_G\}$ (수학: 정답이면 1, 아니면 0)
3. **group relative advantage**: group 내에서 *평균 대비 상대 위치* 로 advantage 를 계산:

$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \varepsilon}$$

4. **정책 갱신**: advantage 가 *양수* 인 답 (group 평균보다 잘함) 의 확률은 ↑, *음수* 인 답은 ↓

여기서 **group 평균이 baseline** 역할을 합니다 — "이 prompt 에서 동료들은 평균 얼마나 받았나" 보다 *잘했으면* advantage 양수. 그래서 *critic (value model) 이 불필요* 합니다 (위 §의 PPO 대비 핵심 간소화).

### 수치 예시 — group 4개 답, reward → advantage

한 prompt 에 4개 답을 생성하고 verifier 로 채점한 reward 가 $[1, 0, 1, 0]$ 라고 합시다 (2개 정답, 2개 오답):

| 답 | reward $r_i$ | $r_i - \text{mean}$ | advantage $A_i = (r_i - \text{mean}) / \text{std}$ | 정책 갱신 |
|---|---|---|---|---|
| $y_1$ | 1 | +0.5 | **+1.0** | 확률 ↑ (잘함) |
| $y_2$ | 0 | −0.5 | **−1.0** | 확률 ↓ (못함) |
| $y_3$ | 1 | +0.5 | **+1.0** | 확률 ↑ (잘함) |
| $y_4$ | 0 | −0.5 | **−1.0** | 확률 ↓ (못함) |

(mean = 0.5, std = 0.5) → 정답인 답은 *advantage +1* 로 강화, 오답은 *−1* 로 억제. **reward 자체가 아니라 *그룹 평균 대비 상대값* 으로 학습** 한다는 점이 핵심입니다.

다른 group $[1, 1, 1, 0]$ (3개 정답, 1개 오답) 이라면: mean=0.75, std≈0.43 → 정답 advantage ≈ **+0.58**, 오답 ≈ **−1.73**. *동료 대부분이 맞힌 상황에서 혼자 틀린 답* 이 더 크게 억제됩니다.

### 모든 답이 같으면 — 학습 신호 0

group 전체가 정답 $[1,1,1,1]$ 이거나 전체 오답 $[0,0,0,0]$ 이면 std = 0 → **advantage 가 전부 0** → 그 prompt 에서는 학습 신호가 없습니다. *그룹 안에 잘한 답과 못한 답이 섞여 있어야* 비교가 생깁니다. (그래서 group size 와 temperature 로 *답의 다양성* 을 확보하는 게 중요 — §의 변형.)

> **§3 에서 실제 verifier 와 group advantage 를 손으로 계산** 해 위 표를 재현합니다. `GRPOTrainer` 가 매 step·매 prompt 내부에서 하는 일이 정확히 이것입니다.

## verifiable reward 의 의미 — 정답 있는 task 는 무한 RL 신호

GRPO 의 진짜 힘은 *알고리즘* 보다 **reward 의 출처** 에 있습니다.

### verifiable reward = *자동 채점 가능한* 신호

- **DPO 의 신호**: *사람/AI 가 비교* 한 preference 쌍. 만들려면 *사람 라벨링* 이나 *강한 judge 모델 (GPT-4)* 이 필요 → *비용·확장 한계*
- **GRPO 의 신호 (verifiable)**: *정답을 자동 검증* (수학 답 일치, 코드 테스트 통과). 한 번 verifier 를 만들면 *사람 없이 무한히* reward 를 생성 → *확장 자유*

| | DPO (preference) | GRPO (verifiable reward) |
|---|---|---|
| reward 만드는 주체 | 사람 / judge 모델 | **verifier (규칙·테스트)** |
| 비용 | 라벨당 비용 (사람·API) | **거의 0** (검증은 자동) |
| 확장성 | 라벨 수에 묶임 | **정답만 있으면 무한 rollout** |
| 적용 범위 | 모든 task (주관 포함) | **검증 가능한 task 만** (수학·코드·형식) |

### DeepSeek-R1 — 순수 RL 로 reasoning

DeepSeek-R1 (그리고 R1-Zero) 은 *수학·코드처럼 정답을 자동 검증* 할 수 있는 문제에 GRPO 를 대규모로 적용해, **사람의 reasoning 데모(SFT) 없이도 모델이 스스로 *긴 사고 과정(chain-of-thought)* 을 만들어내게** 했습니다. 정답이라는 *객관 신호* 만으로, 모델이 *"천천히 단계를 밟아 풀면 정답률이 오른다"* 를 *스스로 발견* 한 것입니다.

> Ch 29 부록에서 본 **pass@1 vs cons@64** (한 번 맞히기 vs 여러 번 생성해 다수결) 가 여기 직접 연결됩니다. verifiable task 는 *여러 답을 생성해 정답을 골라낼 수 있으니*, GRPO 의 *group rollout + verifier* 와 자연스럽게 맞물립니다. *생성을 여러 번 해 정답을 확인* 하는 평가(cons@64)가, *생성을 여러 번 해 정답 방향으로 학습* 하는 GRPO 와 같은 뿌리입니다.

### 한계 — 검증 가능한 task 에만

verifiable reward 의 강점은 *검증 가능한 task* 에서만 성립합니다:

- ✅ **잘 맞음**: 수학 (답 일치), 코드 (테스트 통과), 형식 준수 (정규식·파서), 게임 (승패)
- ✗ **안 맞음**: 글쓰기·대화·요약·취향 — *"무엇이 정답인지" 자동 판정이 어려움*. 이런 *열린 질문* 은 DPO (사람 선호) 나 *LLM-as-judge* (Ch 29 부록) 가 적합

> 실무에서는 **두 신호를 섞습니다** — *검증 가능한 부분은 verifier (GRPO)*, *주관적 품질은 preference/judge (DPO)*. 본 챕터는 *verifiable reward 의 원리* 를 *산술 task* 로 가장 깨끗하게 보입니다.

## 토크나이저 노트 — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 이후 고정)

본 챕터의 토크나이저는 *Ch 27·28·30 과 완전히 동일*. KoGPT2 BBPE (vocab 51,200) 를 그대로 가져옵니다. **KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정** 이 있어 (Ch 27 §토크나이저 노트), `PreTrainedTokenizerFast` + special token 명시로 로드합니다.

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
```

### GRPO 데이터의 토큰화 — prompt 만 입력, 답은 *생성*

DPO 데이터는 `(prompt, chosen, rejected)` *세 텍스트* 였습니다. **GRPO 데이터는 `prompt` 하나만** 토큰화해 모델에 넣고, *답(completion)은 모델이 직접 생성(rollout)* 합니다. 정답은 *토큰화 대상이 아니라 verifier 가 채점할 때만* 쓰는 *추가 컬럼* 입니다.

1. `prompt` 를 토큰화 → policy 에 입력
2. policy 가 `num_generations` 개의 *completion 을 생성* (rollout) — 각 completion 도 KoGPT2 토크나이저로 디코딩
3. 디코딩된 텍스트를 *verifier(reward 함수)* 가 채점 → reward

> 같은 KoGPT2 토크나이저이므로 *Ch 28 SFT·Ch 30 DPO 에서 본 instruction 포맷 토큰화* 가 그대로 적용됩니다. 차이는 *답이 데이터에 있느냐 (DPO) vs 모델이 생성하느냐 (GRPO)* 입니다. 토크나이저는 *Phase 4 내내 고정* — Ch 27 이후 한 번도 바뀌지 않았습니다.

## 환경 셋업

`trl` 의 **`GRPOTrainer`** 와 **`GRPOConfig`**, 그리고 **`reward_funcs`** (verifier 함수) 가 이번 챕터에 새로 등장합니다. `transformers` / `datasets` / `accelerate` 와 함께 설치합니다.

> ⚠️ `trl` 은 버전마다 `GRPOTrainer` / `GRPOConfig` API 변동이 큽니다 (인자 이름이 버전에 따라 바뀝니다 — 예: `max_completion_length` 는 있지만 `max_prompt_length` 는 버전에 따라 없음). 본 노트북은 설치된 `trl` 버전을 셋업 셀에서 출력하고, *버전 간 안정적인 핵심 경로* (`num_generations` + `reward_funcs` + `max_completion_length` + `prompt` 컬럼) 만 사용합니다.

## verifiable 데이터 — `prompt` + 정답 (산술)

GRPO 데이터의 핵심은 **정답을 자동 검증할 수 있어야** 한다는 것입니다. 코드(테스트 실행) 는 무겁고 환경 의존이 크니, 본 챕터는 *가장 깨끗한 verifiable task* 인 **산술(arithmetic)** 로 시작합니다 — 정답이 *정수 하나* 라 *문자열 매칭만으로 채점* 됩니다.

각 샘플은 `(prompt, answer)` 두 컬럼입니다:
- `prompt`: 풀어야 할 문제 (예: `"3 + 5 = ?"`) — 모델에 입력
- `answer`: 정답 (예: `"8"`) — *verifier 가 채점할 때만* 사용 (모델 입력 아님)

> 합성 산술이라 *정답을 우리가 알고* 있으니, *verifier (정답 매칭) 가 완벽* 합니다. 이것이 verifiable reward 의 이상적 형태 — *reward 가 잡음 없이 정확*. (GSM8K 같은 실제 수학 데이터셋도 같은 방식이지만, 답 추출이 더 까다롭습니다 — FAQ 참고.)

## SFT 모델 (policy) 로드

GRPO 는 *SFT 모델에서 출발* 합니다 (Ch 28 의 SFT 체크포인트가 정석). 노트북 단독 실행을 위해 **base KoGPT2 로 시작** 합니다 — 보통은 *이미 지시를 따르는 SFT 모델* 에서 GRPO 를 시작해야 *rollout 이 의미 있는 답* 을 내고 verifier 가 *섞인 reward* (잘한 답 + 못한 답) 를 줄 수 있습니다.

토크나이저는 Ch 27·28·30 과 동일 (`PreTrainedTokenizerFast` + special token 명시 — `AutoTokenizer` 함정 회피).

## verifier (reward function) 정의 + group advantage 손계산

여기가 본 챕터의 *개념 핵심*. **verifier 함수** 를 정의하고, 한 prompt 에 *여러 답* 을 채점한 뒤 *group relative advantage* 를 손으로 계산해 §의 표를 재현합니다. `GRPOTrainer` 가 매 step·매 prompt 내부에서 하는 일을 *축소판으로 재현* 하는 셈입니다.

### verifier — 생성 답에서 정답 추출 → 매칭 → reward

`trl` 의 reward 함수 시그니처는 **`reward_func(completions, **kwargs)`** 입니다:
- `completions`: policy 가 생성한 답들의 *리스트* (group)
- `**kwargs`: 데이터셋의 *나머지 컬럼* 이 *리스트로* 전달 (우리의 `answer` 컬럼이 `answer=[...]` 로 들어옴)
- 반환: 각 completion 의 **reward 리스트** (`list[float]`)

### group relative advantage 손계산 — reward → advantage

verifier 가 매긴 reward $[1, 0, 1, 0]$ 를 *group 평균 대비 상대값* 으로 바꿉니다 (§의 수식):

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \varepsilon}$$

이게 `GRPOTrainer` 가 *critic 없이* advantage 를 만드는 방법 — *group 평균이 baseline*.

**무엇을 보고 있나** — 위 두 출력은 `GRPOTrainer` 가 *매 step, 매 prompt* 내부에서 하는 계산입니다:

- **verifier** 가 *사람 없이 자동* 으로 reward 를 매깁니다 (정답 매칭). preference 라벨이 필요 없습니다
- **group advantage** 가 *critic 없이* 만들어집니다 — *그룹 동료들의 평균* 이 baseline. 평균보다 잘한 답은 +, 못한 답은 −
- **group 전체가 같으면 (전부 정답·전부 오답) advantage = 0** → 학습 신호 없음. *그룹 안에 다양성* (잘한 답 + 못한 답) 이 있어야 GRPO 가 작동합니다

> 이 두 부품 — *verifier (reward)* 와 *group advantage (baseline)* — 이 GRPO 의 전부입니다. 아래 §4 에서 `GRPOTrainer` 에 이 verifier 를 넘기면, 나머지 (rollout · advantage · 정책 갱신) 는 자동입니다.

## `GRPOTrainer` 로 GRPO 학습 — *새 trainer, verifier 로 정렬*

`trl.GRPOTrainer` 는 본 챕터에 처음 등장합니다. §3 에서 손으로 한 *verifier reward → group advantage* 를, *매 step* *rollout (여러 답 생성) → 채점 → advantage → 정책 갱신* 으로 자동 수행합니다. 설정은 `GRPOConfig` (`TrainingArguments` 상속) 로 주며, **`num_generations`** 가 group size 입니다.

> **rollout 주의 (T4 시간·메모리)**: GRPO 는 *매 step 여러 답을 생성* 하므로 무겁습니다 (DPO 보다 generation 비용이 큼). T4 + 30분 룰을 지키려면: **group size 작게 (`num_generations=4`) + 짧은 generation (`max_completion_length` 작게) + 작은 batch + 적은 step**. 시간이 빡빡하면 `N_TRAIN` 이나 step 을 더 줄이세요.

> **`trl` 버전 주의**: `GRPOConfig` 는 `max_completion_length` 를 받지만 `max_prompt_length` 는 버전에 따라 없습니다. `beta` 기본값은 *0.0 (reference 없이, ref-free)* — KL 제약을 켜려면 `beta>0` 으로 주고 reference 가 메모리에 추가됩니다. 본 노트북은 *ref-free (beta=0)* 로 메모리를 아낍니다.

## GRPO 전·후 정확도 비교 — *verifier pass rate 가 올랐는가*

본 챕터의 핵심 데모. *같은 eval 셋* (학습에 안 쓴 산술 문제) 에 대해 *GRPO 전* 과 *후* 의 **정확도 (verifier pass rate)** 를 비교합니다.

- **GRPO 전**: policy 가 산술을 잘 못 풀어 pass rate 낮음
- **GRPO 후**: *정답 방향* 으로 정책이 강화되어 pass rate ↑ (정답을 더 자주 생성)

정확도가 *올랐다면* verifiable reward 로 능력이 정렬된 직접 증거입니다.

**해석 가이드 — verifiable reward alignment 의 증거**

- **before (gray)**: policy 가 산술을 잘 못 풀어 pass rate 가 낮습니다 (base KoGPT2 는 산술에 약함)
- **after (green)**: *정답 방향* 으로 정책이 강화되어 pass rate 가 오릅니다 — 모델이 *정답을 더 자주 생성*

> **핵심**: GRPO 는 *preference 라벨 없이*, *verifier 가 자동 채점한 reward* 만으로 능력을 정렬합니다. group 안에서 *정답이 평균보다 잘한 답* 으로 강화되며, 그 효과가 *정확도(pass rate) 상승* 으로 나타납니다.

> ⚠️ KoGPT2 (125M) 는 작은 base 모델이고 (정석은 SFT 모델에서 출발), 학습 step·group size 도 작아 효과가 *미묘* 할 수 있습니다. 관전 포인트는 *극적 향상* 이 아니라 ***정확도가 정답 방향으로 올랐는가*** 입니다. 또한 *group 안에 정답·오답이 섞여야* (std>0) 학습 신호가 생기므로, base 모델이 *가끔이라도 정답을 내야* GRPO 가 작동합니다 — §6 의 reward 곡선에서 확인.

## 학습 곡선 — reward / reward std / completion 길이

`GRPOTrainer` 는 학습 중 *loss* 뿐 아니라 *reward (group 평균)·reward_std·completion 길이* 같은 GRPO 고유 지표를 로깅합니다 (`trainer.state.log_history`). reward 가 오르고, reward_std 가 *0 이 아닌* (= group 안에 다양성이 있는) 구간에서 학습이 일어났는지 확인합니다.

## 왜 reward 가 잘 안 올랐는가 — GRPO 의 전제조건

§5 의 정확도 막대와 §6 의 reward 곡선을 정직하게 보면, *극적인 상승은 보기 어려웠을* 것입니다. group reward 가 대부분 0 에 머물고, GRPO 전·후 정확도 차이도 미미했을 가능성이 큽니다. 이건 *버그가 아니라* GRPO 라는 알고리즘의 **전제조건** 을 정확히 드러내는 현상입니다. 이번 절에서 *왜 그런지* 를 짚고, *어떻게 하면 reward 가 실제로 오르는지* 를 부록으로 넘깁니다.

### 증상 — group reward 가 대부분 0

base KoGPT2 (125M) 는 산술을 거의 못 풉니다. 한 prompt 에 4개 답을 생성하면 *대부분 전부 오답* → group reward 가 `[0, 0, 0, 0]` 입니다. §3 의 손계산에서 봤듯이:

| group reward | mean | std | advantage | 학습 신호 |
|---|---|---|---|---|
| `[0, 0, 0, 0]` (전부 오답) | 0 | 0 | **전부 0** | **없음** |
| `[1, 1, 1, 1]` (전부 정답) | 1 | 0 | **전부 0** | **없음** |
| `[1, 0, 1, 0]` (섞임) | 0.5 | 0.5 | `[+1,-1,+1,-1]` | **있음** |

base KoGPT2 의 GRPO 는 거의 매번 첫 번째 줄 (`[0,0,0,0]`) 에 빠집니다.

### 근본 원인 — GRPO 는 "무에서 유" 를 만들지 못한다

GRPO 의 advantage 는 $A_i = (r_i - \text{mean}) / (\text{std} + \varepsilon)$ 입니다. group 안의 *모든 답이 같은 reward* 면 std = 0 → **advantage 0 → gradient 0 → 학습 신호 없음**. 즉 GRPO 가 작동하려면 *group 안에 잘한 답과 못한 답이 섞여 있어야* 합니다. 그런데 모델이 task 를 *아예 못 풀면* 모든 답이 똑같이 reward 0 이라 비교 자체가 불가능합니다.

> **핵심 교훈**: GRPO(RL)는 SFT 처럼 *없던 능력을 새로 가르치지* 못합니다. *모델이 이미 가끔이라도 성공하는 능력* 을 그 방향으로 **증폭** 하는 기법입니다. 그래서 *"가끔이라도 정답이 나와야"* GRPO 가 그 방향을 강화할 수 있습니다. base KoGPT2 처럼 *한 번도 성공하지 못하는* 모델에는 증폭할 신호 자체가 없습니다.

작은 base 모델 (KoGPT2 125M) + 어려운 task (산술) = **reward 가 sparse(희소)** = GRPO 가 *출발점* 을 잡지 못하는 전형적 상황입니다. DeepSeek-R1 이 *순수 RL* 로 reasoning 을 끌어낼 수 있었던 것도 *충분히 큰 base 모델* 에서 출발했기 때문입니다 — 큰 모델은 어려운 문제도 *가끔* 맞히므로 group 에 다양성이 생기고, GRPO 가 그 *가끔의 성공* 을 증폭할 수 있었습니다.

### reward 가 안 오를 때의 해결 레버 4가지

| 레버 | 무엇을 | 왜 도움이 되나 |
|---|---|---|
| **(1) SFT 먼저** | instruction-response 로 형식·기초를 먼저 가르침 (Ch 28) | base 가 *가끔이라도 정답·형식* 을 내게 만들어 group 다양성 확보 |
| **(2) 더 강한 base 모델** | KoGPT2 125M → 산술 가능한 더 큰/능력 있는 모델 | 어려운 문제도 *가끔 맞혀* group reward 에 차이 발생 |
| **(3) task 난이도 ↓** | 더 쉬운 문제부터 (한 자리 덧셈 등) | 성공 확률 ↑ → group 에 정답이 섞일 확률 ↑ |
| **(4) format reward + HPO** | 정답 형식을 따르면 *부분 보상* + group size↑·temperature↑ | reward 가 *0 만 나오는 것* 을 막아 *학습 신호를 확보*. 형식 준수만으로도 std>0 |

특히 **(4) format reward** 는 작은 모델에 강력합니다. 정답을 *못 맞혀도* 답을 *정해진 형식* (예: `"정답: N"`) 으로 내면 0.2 같은 부분 보상을 줍니다. 그러면 group 안에서 *형식을 지킨 답 vs 안 지킨 답* 의 reward 차이가 생겨 (예: `[0.2, 0.0, 0.2, 0.0]`) std>0 → **advantage 가 0 에서 벗어나 학습이 시작** 됩니다. 모델이 먼저 *형식* 을 배우고, 그 위에서 *정답* 으로 나아가는 사다리를 놓는 셈입니다.

### 부록에서 reward 가 실제로 오르는 모습 확인

이 레버들을 적용해 reward 가 *실제로 오르는* GRPO 는 부록 [`appendix_qwen_grpo_hpo.ipynb`](./appendix_qwen_grpo_hpo.ipynb) 에서 봅니다. 부록은:

- **(2) 더 강한 base** — `Qwen/Qwen2.5-0.5B-Instruct` (Ch 29 에서 쓴, 산술을 *가끔 맞히는* 모델) 로 교체
- **(4) format reward** — correctness reward + format reward 두 개를 조합해 *0 만 나오는 것* 을 방지
- **HPO** — `num_generations`·`temperature`·`beta`·`learning_rate` 가 reward·수렴에 주는 영향

을 적용해, *본 챕터의 KoGPT2 와 대비* 되도록 **reward 전·후 차이가 명확히 보이는** 셋업을 시연합니다. 본문은 *GRPO 의 전제조건을 (안 오르는 현상으로) 체감* 하는 챕터, 부록은 *그 전제조건을 충족시켜 reward 를 올리는* 챕터입니다.

> 한 문장 요약: ***GRPO 는 모델이 이미 가끔 성공하는 능력을 증폭할 뿐, 무에서 유를 만들지 못한다. 그래서 RL 전에 SFT·충분한 base·format reward 로 "출발점" 을 먼저 마련해야 한다.***

## 이 장의 구성

- [31-1. 실습](31-grpo-practice.md)
- [31-2. 변형 — group size / format reward / 코드 verifier / 다른 task](31-grpo-variation.md)
- [31-3. 정리와 FAQ](31-grpo-wrapup.md)
