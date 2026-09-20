**목표**: Phase 4 의 *마지막 챕터*. Ch 30 에서 **DPO** 로 *사람/AI 가 비교한 preference 쌍 (chosen / rejected)* 으로 정렬했다면, 본 챕터는 alignment 의 *두 번째 방식* — **GRPO (Group Relative Policy Optimization)** 입니다. GRPO 는 정반대 접근으로, *정답을 자동 검증(verifier) 할 수 있는 task* 에서 모델이 *여러 답을 생성(rollout)* 하고 *verifier 가 채점* 해 *잘한 답 방향* 으로 강화학습 합니다. **DeepSeek-R1 이 순수 RL 로 reasoning 능력을 끌어낸 방법** 이 바로 이것입니다. 바뀌는 건 **신호 출처 (preference 쌍 → verifier reward)** + **trainer (`DPOTrainer` → `trl.GRPOTrainer`)** + **데이터 (chosen/rejected → prompt + 정답)** + **rollout (한 prompt 에 여러 답 생성)** 입니다.

> ⚠️ **본 챕터는 base 모델도 바뀝니다 — KoGPT2(125M) → `Qwen2.5-0.5B-Instruct`.** 이건 임의의 교체가 아니라 *GRPO 의 전제조건* 때문입니다: **GRPO 는 없는 능력을 새로 만들지 못하고, 모델이 이미 *가끔이라도 성공하는* 능력을 그 방향으로 증폭** 합니다. KoGPT2 125M 은 글자 세기를 거의 못 해(가끔의 성공조차 없음) GRPO 가 증폭할 신호가 없습니다(부록에서 그 실패를 재현). 그래서 본편은 *지시를 따르고 세기를 가끔 성공* 하는 `Qwen2.5-0.5B-Instruct` 로 GRPO 가 *실제로* 정확도를 올리는 모습을 보입니다. **모델 교체 자체가 이 챕터의 핵심 교훈** 입니다.

**환경**: Google Colab **T4 GPU 필수**. GRPO 는 *매 step 여러 답을 생성(rollout)* 하므로 무겁습니다 — group size 를 작게 (8) + 짧은 generation + 작은 step 으로 시간을 통제합니다.

**예상 소요 시간**: 약 15-25분 (환경 셋업·모델 다운로드 약 3-5분 + 난이도 필터 약 3-5분 + `GRPOTrainer` 학습 약 6-9분 + GRPO 전·후 정확도 비교 약 3분). 0.5B 모델 + rollout 이라 KoGPT2 때보다 무겁습니다.


## 학습 흐름

1. 📊 **누적 추적표** (Ch 28/29/30 + **31 강조**) + GPT 학습 4단계 표 (Ch 30 DPO·Ch 31 GRPO = 단계 4 alignment 의 두 방식)
2. 🔄 **변경점 (Diff from Ch 30 DPO)** — *신호 출처 + trainer + 데이터 + rollout + 모델(→Instruct)* 이 변함
3. 🎯 **PPO vs DPO vs GRPO 대비 표** — 신호·모델·데이터. *왜 GRPO 가 critic 도 reward model 도 없이 되나*
4. 📐 **GRPO 메커니즘** — rollout group → verifier reward → group relative advantage 수식 + 수치 예시
5. 🔬 **verifiable reward 의 의미** — 정답 있는 task 는 사람 채점 없이 무한 RL 신호. DeepSeek-R1 의 reasoning
6. 🔤 **토크나이저 노트** — `Qwen2.5-0.5B-Instruct` 의 BBPE + chat template (KoGPT2 에서 바뀜)
7. 🚀 **실습**: verifiable 데이터(train/eval 분리) → Instruct 모델 로드(**SFT cold start 없이**) → verifier·group advantage 손계산 → **난이도 필터** → `GRPOTrainer` 학습 → GRPO 전·후 정확도 비교(**상승 확인**)
8. 🔍 **GRPO 는 왜 여기서 통했나** — GRPO 의 전제조건(능력 있는 base)과 그것을 충족한 이번 셋업
9. 📦 **등장 라이브러리** (`trl.GRPOTrainer`·`GRPOConfig`·`reward_funcs` 첫 등장) / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)
10. 🎓 **Phase 4 회고 + Phase 5 (Diffusion LM) 예고**


> 📒 **사전 학습 자료**: Ch 30 (DPO — alignment 의 첫 방식), Ch 29 (벤치마크 평가 — `Qwen2.5-0.5B-Instruct` 가 여기서 처음 등장, 특히 부록의 *pass@1·cons@64* 가 verifiable reward 와 직접 연결). 본 챕터는 *alignment 의 두 방식 비교* 를 완성합니다: **DPO (주관적 선호, 사람/AI 비교) vs GRPO (객관적 정답, 자동 검증)**.

## 누적 추적표

| Ch | 모델 | 데이터 | 학습 신호 | Loss | Trainer |
|---|---|---|---|---|---|
| 28 | KoGPT2 (125M, SFT) | KoAlpaca instruction-response 쌍 | response 토큰 (답변만) | `CrossEntropyLoss` (response-only) - SFT | `SFTTrainer` |
| 29 | Qwen2.5-0.5B-Instruct (+ Ch 28 SFT 모델 대조) | KoBEST/산술 subset | - (평가만) | - (`lm-evaluation-harness`) | - |
| 30 | KoGPT2 SFT 모델 (policy) + frozen reference | preference 쌍 (chosen / rejected) | chosen 선호 ↑ / rejected 선호 ↓ | DPO sigmoid loss (β=0.1) | `DPOTrainer` |
| **31 ← 여기** | **Qwen2.5-0.5B-Instruct (SFT 없이 policy)** | **prompt + 정답 (검증 가능, 글자 세기)** | **group relative advantage** | **GRPO loss (group baseline)** | **`GRPOTrainer`** |

> ⚠️ **모델이 KoGPT2 → `Qwen2.5-0.5B-Instruct` 로 바뀝니다** (Ch 28·30 은 KoGPT2). GRPO 는 *이미 가끔 성공하는 능력* 을 증폭하는 기법이라 *능력 있는 base* 가 전제입니다 — KoGPT2 125M 은 그 전제를 못 채워(부록에서 확인) 본편은 Ch 29 에서 만난 `Qwen2.5-0.5B-Instruct` 를 씁니다. 또 Instruct 모델은 이미 지시를 따르므로 **SFT cold start 없이** 바로 GRPO 를 겁니다.

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
- **GRPO (Ch 31, 본 챕터)**: *"이 답이 맞나"* 를 *자동 검증(verifier)* 해 reward. 신호는 *객관적 정답* (세기 결과가 맞나, 수학 답이 맞나, 코드가 테스트를 통과하나). *정답을 자동 확인할 수 있는* task 에 적합

> DPO 가 *사람의 선호* 를 따라간다면, GRPO 는 *정답이라는 객관 신호* 를 따라갑니다. 후자의 강점은 ***사람 채점 없이 무한히 RL 신호를 만들 수 있다*** 는 것 — 정답이 있는 task 라면 verifier 가 *공짜 reward model* 역할을 합니다. DeepSeek-R1 이 이걸로 *순수 RL 만으로 reasoning* 능력을 끌어냈습니다 (§5).

## 변경점 (Diff from Ch 30 DPO)

| 축 | Ch 30 (DPO) | Ch 31 (본 챕터, GRPO) |
|---|---|---|
| **모델** | KoGPT2 (125M) SFT 모델 | **`Qwen2.5-0.5B-Instruct`** ← *변화 0* (GRPO 전제조건 = 능력 있는 base) |
| 본체 구성 | policy + frozen reference | **policy + reference** (reference = *출발 Instruct 모델 자체*, β=0.04) |
| cold start | Ch 28 SFT 모델에서 출발 | **없음** ← Instruct 모델이라 SFT 워밍스타트 불필요 (chat template + 원샷으로 형식 유도) |
| 토크나이저 | `PreTrainedTokenizerFast` (KoGPT2 Character BPE) | **`Qwen2Tokenizer` (BBPE) + chat template** ← 모델과 함께 바뀜 |
| **신호 출처** | preference 쌍 (사람/AI 가 비교) | **verifier reward (정답 자동 검증)** ← *변화 1* |
| **Trainer** | `trl.DPOTrainer` | **`trl.GRPOTrainer`** ← *변화 2* (새 클래스, 첫 등장) |
| **데이터** | `(prompt, chosen, rejected)` 쌍 | **`(prompt, 정답)`** ← *변화 3* (검증 가능한 task = 글자 세기) |
| **rollout** | 없음 (주어진 쌍을 비교) | **한 prompt 에 여러 답 생성** ← *변화 4* (group rollout) |
| **Loss** | DPO sigmoid loss | **GRPO loss (group relative advantage)** ← *변화 5* |
| advantage baseline | - (쌍 비교) | **group 평균** (critic 대신) |

> **핵심 변화는 *신호의 출처*** — *사람/AI 가 만든 preference* 에서 *정답 자동 검증* 으로. 그래서 *정답이 있는 task* 라면 *사람 없이 무한히 RL 신호* 를 만들 수 있습니다.
>
> **모델이 바뀐 이유(변화 0)**: 이 챕터의 *한 가지 축* 은 원래 *정렬 방식(DPO→GRPO)* 입니다. 그런데 GRPO 는 *모델이 이미 가끔 성공하는 능력* 을 증폭할 뿐 *없는 능력을 만들지* 못합니다. KoGPT2 125M 은 글자 세기를 거의 못 해 증폭할 신호가 없어(부록 `31_grpo_appendix.ipynb` 에서 그 실패를 재현), *GRPO 가 실제로 작동하는 모습* 을 보이려면 *가끔이라도 성공하는* 모델이 필요합니다. 그래서 Ch 29 에서 만난 `Qwen2.5-0.5B-Instruct` 로 올라갑니다 — **모델 교체는 GRPO 의 전제조건을 만족시키기 위한 것이고, 그 전제조건 자체가 §8 의 핵심 교훈** 입니다.

## PPO vs DPO vs GRPO — alignment 의 세 갈래 (본 챕터의 뼈대)

alignment 의 세 방법을 *신호 출처·필요 모델·데이터* 로 정리합니다. GRPO 의 위치를 이 표 하나로 잡습니다.

| 방법 | 신호 출처 | 필요 모델 | 데이터 | T4 |
|---|---|---|---|---|
| **PPO** (전통 RLHF) | reward model 점수 | actor + critic + reward model + reference (**4개**) | prompt + 학습된 RM | ✗ (메모리 초과) |
| **DPO** (Ch 30) | preference 쌍 (사람/AI 비교) | policy + frozen reference (**2개**) | `(prompt, chosen, rejected)` | ✓ |
| **GRPO** (Ch 31, 본 챕터) | **verifier (정답 자동 검증)** | **policy** (+ reference; β>0 이면 필수) | **`(prompt, 정답)` — 검증 가능** | ✓ |

### 왜 GRPO 는 critic 도 reward model 도 없이 되나 — *group 평균이 baseline*

전통 PPO 는 *advantage* 를 계산하려고 **critic (value model)** 을 따로 둡니다 — "이 상태에서 기대되는 reward 가 얼마인가" 의 *baseline* 을 추정하기 위해서입니다. advantage = (실제 reward) − (critic 이 예측한 baseline).

GRPO 의 통찰: **같은 prompt 에 답을 여러 개 (group) 생성하면, *그 group 의 평균 reward* 가 곧 baseline 이 된다.** critic 을 학습할 필요가 없습니다 — *그룹 동료들의 평균* 이 "이 prompt 에서 보통 어느 정도 받나" 를 알려주니까요.

| 항목 | PPO | GRPO |
|---|---|---|
| baseline (advantage 기준) | **critic (value model)** 이 예측 | **group 평균 reward** (동료 비교) |
| reward 출처 | **reward model** (별도 학습) | **verifier** (정답 자동 검증, 학습 불필요) |
| 필요 모델 | actor + critic + RM + ref (4) | **policy** (+ reference; 본 노트북은 β=0.04 라 2개) |

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
| $y_1$ | 1 | +0.5 | **+0.87** | 확률 ↑ (잘함) |
| $y_2$ | 0 | −0.5 | **−0.87** | 확률 ↓ (못함) |
| $y_3$ | 1 | +0.5 | **+0.87** | 확률 ↑ (잘함) |
| $y_4$ | 0 | −0.5 | **−0.87** | 확률 ↓ (못함) |

(mean = 0.5, 표본std ≈ 0.58) → 정답인 답은 *advantage ≈ +0.87* 로 강화, 오답은 *≈ −0.87* 로 억제. **reward 자체가 아니라 *그룹 평균 대비 상대값* 으로 학습** 한다는 점이 핵심입니다. (std 는 *표본표준편차 ddof=1* — trl 이 쓰는 값과 같게 맞췄습니다.)

다른 group $[1, 1, 1, 0]$ (3개 정답, 1개 오답) 이라면: mean=0.75, 표본std=0.5 → 정답 advantage = **+0.5**, 오답 = **−1.5**. *동료 대부분이 맞힌 상황에서 혼자 틀린 답* 이 더 크게 억제됩니다.

### 모든 답이 같으면 — 학습 신호 0

group 전체가 정답 $[1,1,1,1]$ 이거나 전체 오답 $[0,0,0,0]$ 이면 std = 0 → **advantage 가 전부 0** → 그 prompt 에서는 학습 신호가 없습니다. *그룹 안에 잘한 답과 못한 답이 섞여 있어야* 비교가 생깁니다. (그래서 group size 와 temperature 로 *답의 다양성* 을 확보하는 게 중요 — §의 변형.)

> **§3 에서 실제 verifier 와 group advantage 를 손으로 계산** 해 위 표를 재현합니다. `GRPOTrainer` 가 매 step·매 prompt 내부에서 하는 일이 정확히 이것입니다 — 본 노트북은 `scale_rewards="group"`(std 정규화) + 표본std(ddof=1) 라 위 수식·수치가 trainer 내부 계산과 그대로 일치합니다.

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

> 실무에서는 **두 신호를 섞습니다** — *검증 가능한 부분은 verifier (GRPO)*, *주관적 품질은 preference/judge (DPO)*. 본 챕터는 *verifiable reward 의 원리* 를 *글자 세기 task* 로 가장 깨끗하게 보입니다.

## 토크나이저 노트 — `Qwen2.5-0.5B-Instruct` 의 BBPE + chat template (KoGPT2 에서 바뀜)

본 챕터는 모델이 바뀌면서 **토크나이저도 KoGPT2 Character BPE → `Qwen2Tokenizer` (Byte-level BPE, vocab 151,643)** 로 바뀝니다. Ch 27 이후 고정이던 KoGPT2 토크나이저 흐름이 여기서 끊깁니다 — GRPO 의 전제조건(능력 있는 base) 때문에 모델을 올린 결과입니다. Qwen 토크나이저는 **`AutoTokenizer` 함정이 없어** 그대로 로드합니다 (KoGPT2 는 영어 GPT2 로 잘못 fallback 해 `PreTrainedTokenizerFast` 가 필요했던 것과 대조 — Ch 27).

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token   # eos=<|endoftext|>
```

### 글자 세기 데이터의 토큰화 — 하이픈으로 글자를 *분리* 하는 이유

Qwen 은 **byte-level BPE** 라 자주 쳐지는 글자열은 *한 토큰으로 병합* 될 수 있습니다 (예: `"nine"` → 한두 토큰). 그러면 모델이 *개별 글자를 세기* 어렵습니다. 그래서 단어를 **하이픈으로 이어** `"n-i-n-e"` 처럼 만들어, 각 글자가 *토큰 경계로 드러나게* 합니다 — 이게 verl-recipe `char_count` 의 설계 포인트이자, *세기* 라는 task 를 토크나이저 수준에서 가능하게 하는 장치입니다.

### chat template — Instruct 모델은 이 형식으로 호출해야 한다

**Instruct 모델은 평문 프롬프트가 아니라 *chat template* 로 호출** 해야 지시를 제대로 따릅니다. Qwen 은 `<|im_start|>role ... <|im_end|>` 형식을 씁니다:

```python
msgs = [{"role": "system", "content": "...글자 개수를 세는 조수..."},
        {"role": "user", "content": '"a-b-a" 에서 문자 \'a\' 은 몇 개인가요?'},
        {"role": "assistant", "content": "a = a\nb != a\na = a\n\\boxed{2}"},   # 원샷 예시
        {"role": "user", "content": question}]
prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
```

> chat template + **원샷 예시** 가 *SFT cold start 를 대체* 합니다 (§2.5). Instruct 모델은 이미 지시를 따르므로, 예시 하나로 *`\boxed{개수}` 형식* 을 유도하면 SFT 없이도 곧바로 verifiable 한 답을 냅니다. 평문 프롬프트로 부르면 모델이 장황하게 답해 답이 잘려(\boxed 에 도달 못 함) 채점이 안 됩니다 — chat template 이 필수인 이유입니다.

## 이 장의 구성

[[SubPages]]
