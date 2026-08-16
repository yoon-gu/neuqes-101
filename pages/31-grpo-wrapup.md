## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | Ch 30 과 차이 |
|---|---|---|
| `trl.GRPOTrainer` | GRPO 특화 trainer (rollout → verifier 채점 → group advantage → 정책 갱신 자동) | **새로 등장** (Ch 30 은 `DPOTrainer`) |
| `trl.GRPOConfig` | `GRPOTrainer` 설정 (`TrainingArguments` 상속 + `num_generations`·`max_completion_length`·`beta` 등) | **새로 등장** |
| `reward_funcs` (verifier) | 생성 답을 채점하는 callable (또는 list). `(completions, **kwargs)` → `list[float]` | **새로 등장** (DPO 는 preference 데이터, reward 함수 없음) |
| `GRPOConfig(num_generations=4)` | group size — 한 prompt 당 생성 답 개수 (rollout) | **새로 등장** |
| `GRPOConfig(beta=0.04)` | KL 제약 강도. 작은 값으로 reference(=SFT 모델) 근처에 묶어 collapse·hacking 완화 (0 = ref-free) | **새로 등장** (DPO 의 beta 와 의미 비슷) |
| group relative advantage | `(r - mean) / (std + eps)` — group 평균이 baseline (critic 대체) | **새로 등장** (DPO 는 쌍 비교, advantage 없음) |
| `model.generate(num_return_sequences=k)` | rollout — 한 prompt 에 여러 답 생성 | **새로 등장** (DPO 는 생성 불필요) |
| `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` | KoGPT2 Character BPE (AutoTokenizer 함정 회피) | **공유** (Ch 27 이후 고정) |

> `trl` 은 버전마다 `GRPOTrainer` / `GRPOConfig` API 변동이 큽니다 (`max_prompt_length` 같은 인자가 버전에 따라 없음). 본 노트북은 *버전 간 안정적인 핵심 경로* (`num_generations` + `reward_funcs` + `max_completion_length` + `prompt` 컬럼) 만 사용합니다. 설치된 `trl` 버전은 셋업 셀 출력에서 확인하세요.

## 체크포인트 질문

1. GRPO 는 *PPO 와 달리 critic (value model) 이 없습니다*. 그런데도 advantage 를 계산할 수 있는 이유는 무엇인가요? (*group 평균* 이라는 단어를 써서 설명)
2. *DPO 와 GRPO* 는 둘 다 alignment (단계 4) 입니다. *신호의 출처* 가 어떻게 다른가요? 각각 *어떤 task* 에 적합한가요?
3. 한 prompt 의 group reward 가 `[1, 1, 1, 1]` (전부 정답) 이면 *advantage 가 전부 0* 이 됩니다. 이게 *왜 학습 신호가 없는* 상태인지, 그리고 *어떻게 다양성을 확보* 하는지 설명해 보세요.
4. *reward hacking* 이란 무엇인가요? verifier 가 *정답 매칭* 일 때 일어날 수 있는 reward hacking 의 예를 하나 들어 보세요.

## FAQ

### Q1. (이론) GRPO 와 DPO, 언제 무엇을 쓰나요?

*신호의 출처* 가 다르고, 그에 따라 *적합한 task* 가 갈립니다:

| | DPO (Ch 30) | GRPO (Ch 31) |
|---|---|---|
| 신호 | 사람/AI 가 *비교* 한 preference 쌍 | verifier 가 *자동 검증* 한 reward |
| 적합 task | *정답이 없는* 열린 질문 (글쓰기·대화·취향) | *정답을 자동 확인* 가능 (수학·코드·형식) |
| 데이터 비용 | 라벨당 비용 (사람·judge) | 거의 0 (정답만 있으면 무한 rollout) |
| 학습 방식 | 지도학습 (생성 불필요) | RL (rollout - 매 step 생성) |

```python
# 정답이 있는 task -> GRPO
trainer = GRPOTrainer(model, reward_funcs=verifier, ...)   # verifier 가 자동 채점
# 정답이 없는 주관적 품질 -> DPO
trainer = DPOTrainer(model, ref_model=None, ...)           # preference 쌍으로 비교
```

> 실무에서는 *섞어 씁니다* — 검증 가능한 능력(수학·코드)은 GRPO, 주관적 품질(말투·안전성)은 DPO/judge.

### Q2. (실무) verifier 가 없는 task 는 GRPO 를 못 쓰나요?

GRPO 의 전제는 *reward 를 자동으로 매길 수 있어야* 한다는 것입니다. *정답을 자동 판정할 수 없는* task (예: "이 시가 아름다운가") 는 GRPO 의 *깨끗한 신호* 를 얻기 어렵습니다. 대안:

- **LLM-as-judge 를 verifier 로** (Ch 29 부록): 강한 모델이 *점수* 를 매겨 reward 로. 단 judge 의 편향·비용·잡음이 reward 에 섞임 (RLAIF)
- **부분적 verifier**: 형식·길이·금칙어 같은 *검증 가능한 부분만* reward 로 (format reward)
- **DPO 로 전환**: 비교가 더 쉬운 task 면 preference 쌍이 나음

```python
# judge 모델을 reward 로 (예시 - 비용·잡음 주의)
def reward_judge(completions, **kwargs):
    return [judge_model.score(c) for c in completions]   # 0-1 점수
```

> 핵심: *reward 가 신뢰할 만한가* 가 GRPO 성패를 가릅니다. 잡음 많은 reward 는 *잘못된 방향* 으로 정렬합니다.

### Q3. (이론) group size (`num_generations`) 는 결과에 어떤 영향을 주나요?

group 평균이 *baseline (critic 대체)* 이므로, group size 가 *baseline 추정의 안정성* 을 좌우합니다:

- **group 작음** (예: 2): rollout 싸지만, *평균(baseline) 추정이 불안정*. group 안에 *정답·오답이 섞일 확률* 도 낮아져 *advantage 0 (학습 신호 없음)* 인 prompt 가 많아짐
- **group 큼** (예: 8-16): baseline 안정 + 다양성 확보 → advantage 정밀. 단 *rollout 비용 = group size 에 비례* (T4 시간 ↑)

```python
grpo_config.num_generations = 4   # T4 출발점. 시간 여유 있으면 8 로
```

> 직관: group 은 *"이 prompt 에서 동료 몇 명에게 물어볼까"* 입니다. 많을수록 *평균이 믿을 만* 하지만 *물어보는 비용* 이 듭니다.

### Q4. (이론·실무) reward hacking 이란? GRPO 에서 어떻게 막나요?

**reward hacking** = 모델이 *진짜 목표가 아니라 reward 의 허점* 을 찾아 점수만 올리는 현상입니다. verifier 가 *정답 매칭* 일 때의 예:

- verifier 가 *"문자열에 정답 숫자가 들어 있으면 1.0"* 이면, 모델이 *"답은 1 2 3 4 5 6 7 8 9 ..."* 처럼 *모든 숫자를 나열* 해 정답을 포함시킬 수 있음 (풀이 없이 reward 획득)
- *마지막 정수만* 본다면, *엉뚱한 풀이 뒤에 정답만 붙이는* 식으로 우회

막는 법:

```python
# verifier 를 엄격하게 - 정확한 형식 + 정답 둘 다 요구
def reward_strict(completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        m = re.search(r"####\s*(-?\d+)\s*$", c.strip())   # 정해진 형식 + 끝에 위치
        out.append(1.0 if (m and m.group(1) == str(a)) else 0.0)
    return out
# beta>0 으로 KL 제약 (reference 에서 멀어지면 페널티 - collapse/hacking 완화)
# format reward 와 정답 reward 를 분리해 reward_weights 로 균형
```

> verifier 설계가 GRPO 의 *가장 중요한 부분* 입니다. *허점 없는 reward* = *원하는 능력* 으로 정렬.

### Q5. (이론) GRPO 가 DeepSeek-R1 의 reasoning 과 무슨 관계인가요?

DeepSeek-R1(-Zero) 은 *수학·코드처럼 정답을 자동 검증* 할 수 있는 문제에 GRPO 를 *대규모로* 적용했습니다. 핵심 발견:

- *사람의 reasoning 데모(SFT) 없이도*, **정답이라는 객관 reward 만으로** 모델이 *스스로 긴 사고 과정(chain-of-thought)* 을 만들어냄
- *"단계를 천천히 밟으면 정답률이 오른다"* 를 모델이 *RL 로 스스로 발견* (생성이 길어지고 self-check 가 나타남)

> Ch 29 부록의 **cons@64** (여러 번 생성해 다수결) 와 같은 뿌리입니다 — *여러 답을 생성해 정답을 골라내는* 평가가, *여러 답을 생성해 정답 방향으로 학습* 하는 GRPO 와 맞물립니다. verifiable task 라서 *생성을 무한히* 할 수 있다는 점이 둘의 공통 전제입니다.

### Q6. (실무) 왜 PPO 대신 GRPO 인가요? (특히 T4)

PPO 는 *actor + critic + reward model + reference* **4 모델** 을 동시에 메모리에 올립니다 — T4 (16GB) 에 무리입니다. GRPO 는:

- **critic 제거** → group 평균이 baseline
- **reward model 제거** → verifier 가 자동 채점 (학습 불필요)
- 남는 건 **policy** (+ 옵션 reference). T4 한 장에서 *rollout 만 감당* 하면 됨

```python
# PPO: actor + critic + reward model + reference (4 모델) -> T4 초과
# GRPO: policy + 작은 KL 앵커(reference=SFT 모델) + verifier(함수) -> T4 가능
GRPOConfig(num_generations=4, beta=0.04, use_vllm=False)   # 작은 KL 앵커 + HF generate
```

> 단 GRPO 도 *rollout (매 step 생성)* 은 PPO 와 공유하므로, *생성 비용* 은 듭니다. T4 에서는 group·step·generation 길이를 작게 잡아 통제합니다.

### Q7. (실무) 작은 모델 (KoGPT2 125M) GRPO 의 한계는?

GRPO 효과는 *출발 모델이 가끔이라도 정답을 내는지* 에 달렸습니다:

- **base 에서 출발 (본 노트북)**: 모델이 산술을 거의 못 풀면 group 이 *전부 오답* → std=0 → *advantage 0 (학습 신호 없음)*. 정석은 *SFT 모델에서 출발* (이미 어느 정도 푸는 상태)
- **작은 모델**: reasoning 능력 자체가 약해 GRPO 로 끌어올릴 *상한* 이 낮음 (R1 은 큰 모델이라 가능)
- **짧은 학습**: 방향을 보기엔 충분하나 극적 변화는 어려움

> 본 챕터의 목표는 *완성된 reasoning 모델* 이 아니라 ***GRPO 가 무엇을 최적화하는가 (verifier reward + group advantage) 를 눈으로 확인*** 하는 것입니다. §3 의 손계산과 §5 의 정확도 변화가 핵심. 실전은 *SFT 모델 + 큰 모델 + 많은 rollout + 엄격한 verifier* 의 영역입니다.

## 왜 reward 가 잘 안 올랐는가 — GRPO 의 전제조건

§5 의 정확도 막대를 보면 GRPO 후 정확도가 SFT 베이스라인 위로 **소폭 올랐습니다**(0.875 → 0.891). 극적이진 않지만 분명한 **양(+)의 개선** 이고, 이건 *그냥 얻어진 게 아닙니다*. 사실 이 산술 task 에 GRPO 를 *순진하게* 돌리면 정확도가 **오히려 떨어지거나 collapse** 합니다 — group reward 가 전부 0 이거나 전부 1 로 쏠려 학습 신호가 사라지기 때문입니다. 우리가 §2.5 의 **SFT 워밍스타트**(능력 부여)와 §4.5 의 **난이도 필터**(group 분산 확보)로 *바로 이 전제조건* 을 먼저 충족시켰기에 GRPO 가 비로소 작동했습니다. 이번 절에서 그 전제조건의 정체 — *왜 group 안에 정답·오답이 섞여야 하는가* — 를 짚습니다.

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

이 레버들을 적용해 reward 가 *실제로 오르는* GRPO 는 부록 [`31_grpo_appendix.ipynb`](./31_grpo_appendix.ipynb) 에서 봅니다. 부록은:

- **(2) 더 강한 base** — `Qwen/Qwen2.5-0.5B-Instruct` (Ch 29 에서 쓴, 산술을 *가끔 맞히는* 모델) 로 교체
- **(4) format reward** — correctness reward + format reward 두 개를 조합해 *0 만 나오는 것* 을 방지
- **HPO** — `num_generations`·`temperature`·`beta`·`learning_rate` 가 reward·수렴에 주는 영향

을 적용해, *본 챕터의 KoGPT2 와 대비* 되도록 **reward 전·후 차이가 명확히 보이는** 셋업을 시연합니다. 본문은 *GRPO 의 전제조건을 (안 오르는 현상으로) 체감* 하는 챕터, 부록은 *그 전제조건을 충족시켜 reward 를 올리는* 챕터입니다.

> 한 문장 요약: ***GRPO 는 모델이 이미 가끔 성공하는 능력을 증폭할 뿐, 무에서 유를 만들지 못한다. 그래서 RL 전에 SFT·충분한 base·format reward 로 "출발점" 을 먼저 마련해야 한다.***

## Phase 4 회고 + Phase 5 예고

### Phase 4 완성 — encoder 에서 decoder 로, pretraining 에서 alignment 로

Ch 24-31 의 **Phase 4 (GPT 시대)** 를 마칩니다. Phase 1-3 이 *encoder (BERT)* 로 *이해(분류)* 를 다뤘다면, Phase 4 는 *decoder (GPT)* 로 *생성* 과 *학습 4단계 전체* 를 통과했습니다:

| 단계 | 챕터 | 무엇을 | 학습 신호 |
|---|---|---|---|
| 1 **Pretraining** | Ch 24 (영어), Ch 26 (한국어) | scratch GPT 를 일반 코퍼스로 | next-token |
| 2 **Continual pretraining** | Ch 25 (영어), Ch 27 (한국어) | 사전학습 모델에 새 데이터 | next-token |
| 3 **SFT** | Ch 28 | 지시를 따르게 (행동 정렬) | response 토큰 |
| 4 **Alignment** (DPO) | Ch 30 | 사람 선호로 (주관적 비교) | preference 쌍 |
| 4 **Alignment** (GRPO) | **Ch 31 ← 여기** | 정답으로 (객관적 검증) | verifier reward + group advantage |

**Phase 4 를 관통한 thread**:
- **`labels = -100` 의 response-only**: SFT(답변만 학습) → DPO(답변만 비교) 로 이어짐
- **영/한 대칭**: pretraining·continual pretraining 을 영어(Ch 24·25)와 한국어(Ch 26·27) 로 대칭 진행
- **alignment 의 두 방식**: DPO(주관적 선호) 와 GRPO(객관적 정답) — *신호의 출처* 가 정렬 방법을 나눔
- **PPO 의 두 갈래 간소화**: DPO(reward model + RL 루프 제거), GRPO(critic + reward model 제거) — 둘 다 *T4 에서 alignment 를 손으로* 돌려볼 수 있게 함

### Phase 5 예고 — Diffusion LM (Ch 32-34), 새 패러다임

Phase 1-4 의 모든 모델은 *autoregressive* 였습니다 — *왼쪽에서 오른쪽으로, 한 토큰씩* 생성 (MLM 도 결국 토큰 단위 예측). **Phase 5 는 완전히 다른 생성 패러다임 — Diffusion Language Model** 을 다룹니다:

- **autoregressive (지금까지)**: 토큰을 *순차적* 으로 하나씩. 이전 토큰이 다음 토큰의 조건
- **diffusion (Phase 5)**: *전체 시퀀스를 한꺼번에* 두고, *잡음(masked/noised) 상태에서 병렬로 denoise* 해 점진적으로 완성. 이미지 diffusion (노이즈에서 그림으로) 의 텍스트 버전

> Phase 5 (Ch 32-34) 에서는 *왜 텍스트에 diffusion 을 적용하는가*, *autoregressive 대비 무엇이 다른가* (병렬 생성·양방향 문맥·되돌리기), 그리고 *작은 diffusion LM 을 직접 학습* 해 봅니다. *한 토큰씩* 이라는 Phase 1-4 의 대전제를 깨는, 커리큘럼의 새 막입니다.

**다음 챕터: Chapter 32 — Diffusion Language Model 입문 (autoregressive 가 아닌 병렬 denoise).**
