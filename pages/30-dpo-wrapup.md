## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | Ch 28 과 차이 |
|---|---|---|
| `trl.DPOTrainer` | DPO 특화 trainer (response-only log-prob → margin → sigmoid loss 자동) | **새로 등장** (Ch 28 은 `SFTTrainer`) |
| `trl.DPOConfig` | `DPOTrainer` 설정 (`TrainingArguments` 상속 + `beta`·`max_length` 등) | **새로 등장** |
| `DPOConfig(beta=0.1)` | reference 제약 강도 (KL) — DPO 의 핵심 하이퍼파라미터 | **새로 등장** |
| `DPOTrainer(ref_model=None)` | reference 자동 복사·freeze (명시 지정도 가능) | **새로 등장** (frozen reference 개념) |
| `prompt` / `chosen` / `rejected` 데이터 형식 | preference 쌍 표준 형식 | **새로 등장** (Ch 28 은 `prompt`/`completion`) |
| `torch.nn.functional.log_softmax` + `gather` | response 토큰의 log-prob 합 (§3 손계산) | **공유** (개념은 CausalLM loss 와 동일) |
| `copy.deepcopy(policy)` + `requires_grad_(False)` | frozen reference 직접 생성 (§3 시연용) | **새로 등장** |
| `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` | KoGPT2 Character BPE (AutoTokenizer 함정 회피) | **공유** (Ch 27 이후 고정) |

> `trl` 은 버전마다 `DPOTrainer` / `DPOConfig` API 변동이 큽니다 (`max_prompt_length` 같은 인자가 버전에 따라 사라지기도). 본 노트북은 *버전 간 안정적인 핵심 경로* (`prompt`/`chosen`/`rejected` 데이터 + `beta` + `max_length` + `ref_model=None`) 만 사용합니다. 설치된 `trl` 버전은 셋업 셀 출력에서 확인하세요.

## 체크포인트 질문

1. DPO 에서 *왜 frozen reference 가 필요한가요?* reference 없이 (또는 β=0 으로) chosen 의 확률만 무한정 올리면 어떤 문제가 생기나요?
2. *DPO 와 PPO (RLHF)* 는 둘 다 preference 로 정렬합니다. 그런데 DPO 가 *T4 한 장에서 가능* 한 이유는 무엇인가요? (필요한 모델 개수로 설명)
3. DPO loss 의 **β** 가 *크면* / *작으면* 각각 어떤 trade-off 가 있나요? (정렬 속도 vs reference 에서 벗어남)
4. preference 데이터 `(prompt, chosen, rejected)` 는 어떻게 만드나요? 공개 데이터셋 외에, SFT 모델로 *직접* 만들려면 어떤 절차가 필요할까요?

## FAQ

### Q1. (이론) DPO 와 RLHF (PPO) 는 정확히 뭐가 다른가요?

둘 다 *preference 로 모델을 정렬* 하지만, *경로* 가 다릅니다:

| 항목 | PPO (RLHF) | DPO |
|---|---|---|
| reward model | *별도로 학습* (preference → 점수 모델) | **없음** (preference 에서 직접) |
| 학습 방식 | 강화학습 (rollout + advantage + PPO clip) | **지도학습** (loss.backward()) |
| 필요 모델 | actor + critic + reward + reference (4개) | **policy + reference (2개)** |
| 안정성 | RL 특유의 불안정 (튜닝 까다로움) | **상대적으로 안정** |

DPO 의 통찰은 *"reward model 의 최적 정책을 닫힌 형태로 풀면, reward model 을 명시적으로 만들 필요 없이 preference 만으로 policy 를 직접 최적화할 수 있다"* 는 것입니다. 그래서 *RM 학습 + RL 루프* 두 단계가 *지도학습 한 단계* 로 줄어듭니다.

```python
# PPO: SFT -> reward model 학습 -> PPO (rollout + RL)  ... 4 모델
# DPO: SFT -> DPOTrainer(model, ref_model=None, ...).train()  ... 2 모델, 지도학습
```

### Q2. (실무) reference 모델 없이 DPO 를 할 수 있나요?

`DPOTrainer(ref_model=None)` 은 *reference 가 없는 게 아니라*, *policy 의 복사본을 자동으로 reference 로 freeze* 하는 것입니다 (또는 PEFT 사용 시 adapter 를 끈 base 가 reference). 즉 reference 는 *항상* 있습니다.

*진짜로 reference 를 빼면* (`reference_free` 류 옵션 또는 ORPO):
- KL 제약이 사라져 *원본에서 멀어지는 것을 막을 닻이 없어집니다*
- chosen 확률만 무한정 키우다 *모델이 collapse (한 패턴 반복, 문법 collapse)* 할 위험
- ORPO 는 *reference 없이도* 작동하도록 *loss 를 다르게 설계* 한 변종 (SFT 와 preference 를 한 번에)

```python
# 보통은 자동 reference 로 충분:
trainer = DPOTrainer(model=policy, ref_model=None, args=cfg,
                     train_dataset=dpo_ds, processing_class=tokenizer)
# 메모리가 빠듯하면: PEFT(LoRA) 로 policy 를 학습 -> reference 는 adapter 끈 base (추가 메모리 거의 0)
```

### Q3. (이론) β 가 너무 크면 / 너무 작으면 어떻게 되나요?

β 는 *reference 에서 벗어나는 정도* 를 제어합니다 (KL 제약의 세기):

- **β 너무 큼** (예: 1.0): reference 제약이 *매우 강함* → policy 가 reference 근처에 묶여 *거의 안 움직임* → 정렬이 느리거나 안 됨 (trl 공식 문서: *Higher β means less deviation from the reference model*)
- **β 너무 작음** (예: 0.01): reference 제약이 *거의 없음* → policy 가 preference 에 강하게 끌려가 *빨리 정렬* 되지만, *원본 SFT 의 일반 능력이 collapse* (degeneration)·*reward hacking* 위험. margin 만 키우려고 *답변 품질을 희생* 할 수 있습니다

```python
# 1 에서 시작. reward accuracy 가 안 오르면 0.05 로 낮춰 보고 (제약 완화),
# 답변이 망가지면 (반복/collapse) 0.2-0.3 으로 올려 보세요 (제약 강화).
dpo_config.beta = 0.1
```

> 직관: β 는 *"preference 를 얼마나 공격적으로 따를 것인가 vs 원본을 얼마나 지킬 것인가"* 의 다이얼입니다.

### Q4. (실무) preference 데이터 `(chosen, rejected)` 는 어디서 / 어떻게 만드나요?

세 가지 경로:

1. **공개 데이터셋**: 본 챕터의 `maywell/ko_Ultrafeedback_binarized`, 영어는 `Anthropic/hh-rlhf`, `argilla/ultrafeedback-binarized-preferences` 등
2. **사람 라벨링**: 같은 prompt 에 *여러 답변* 을 생성 → 사람이 *더 나은 쪽을 chosen* 으로 표시 (RLHF 의 원형)
3. **AI 라벨링 (RLAIF)**: 강한 모델 (예: GPT-4) 이 *어느 답이 더 나은지 판정* → chosen/rejected 자동 생성

```python
# SFT 모델로 직접 만들기 (간이):
# 같은 prompt 에 답변 2개 생성 (temperature 다르게)
# 더 강한 모델/규칙/사람이 chosen 선택
# {"prompt":..., "chosen":..., "rejected":...} 로 저장
```

핵심은 *chosen 이 rejected 보다 "사람이 선호하는" 방향* 이면 된다는 점 — 정답일 필요는 없고 *상대적 선호* 만 있으면 DPO 가 작동합니다.

### Q5. (이론) DPO 변종 (IPO, KTO, ORPO) 은 무엇인가요?

모두 *preference 정렬* 의 변주입니다 — *loss 형태·데이터 요구* 만 다릅니다:

| 변종 | 핵심 차이 | 언제 |
|---|---|---|
| **IPO** | sigmoid 대신 *squared loss* | DPO 의 *overfitting* 완화 |
| **KTO** | *쌍이 아닌* 개별 좋음/나쁨 라벨 | preference *쌍을 만들기 어려울* 때 |
| **ORPO** | *reference 없이* SFT + preference 동시 | reference 메모리 절약 + 단계 합치기 |

```python
# trl 에서 loss_type 으로 변종 선택 (버전에 따라 지원 범위 다름)
dpo_config.loss_type = "ipo"        # IPO
# KTO 는 KTOTrainer, ORPO 는 ORPOTrainer 로 별도 클래스인 경우도
```

> 본 챕터는 *원조 DPO (sigmoid loss)* 로 *원리* 에 집중합니다. 변종들은 *같은 목표 (chosen 선호 ↑), 다른 수단*.

### Q6. (실무) 작은 모델 (KoGPT2 125M) DPO 의 한계는?

DPO 의 효과는 *출발 모델의 능력* 에 크게 의존합니다:

- **base 에서 출발 (본 노트북)**: 모델이 아직 *지시를 잘 못 따르므로* preference 정렬 효과가 *미묘*. 정석은 *SFT 모델에서 출발*
- **작은 모델**: chosen/rejected 의 log-prob 차이를 *섬세하게* 다루기 어려워 margin 이동 폭이 작음
- **짧은 학습**: 1 epoch / 1.5K 샘플은 *방향* 을 보기엔 충분하지만 *극적 변화* 는 어려움

> 본 챕터의 목표는 *완성된 정렬 모델* 이 아니라 ***DPO 가 무엇을 최적화하는가 (reward margin) 를 눈으로 확인*** 하는 것입니다. §3 의 손계산과 §5 의 margin 이동이 핵심. 실전 품질은 *SFT 모델 + 큰 모델 + 많은 preference + LoRA* 의 영역.

### Q7. (이론) 다음 단계 GRPO (Ch 31) 는 DPO 와 뭐가 다른가요?

둘 다 alignment (단계 4) 지만, *선호의 출처* 가 다릅니다:

| 단계 | 선호의 출처 | 데이터 |
|---|---|---|
| **DPO (Ch 30)** | *사람이 비교* 한 preference 쌍 | `(prompt, chosen, rejected)` |
| **GRPO (Ch 31)** | *verifier 가 자동 채점* 한 reward | verifiable-reward prompts (수학·코드) |

> DPO 는 *주관적 선호* (어느 답이 더 좋은가 — 사람 판단) 를, GRPO 는 *객관적 정답* (수학 답이 맞나, 코드가 돌아가나 — 자동 검증) 을 신호로 씁니다. GRPO 는 *같은 prompt 에 여러 답을 rollout* 해 *그룹 안에서 상대 비교* (group relative advantage) 합니다 — Ch 31 에서 본격.

```python
# Ch 31 미리보기 (GRPO)
# from trl import GRPOTrainer, GRPOConfig
# reward_funcs = [정답 검증 함수]  # 예: 수학 답 일치 여부 -> 1.0 / 0.0
# 같은 prompt 에 여러 답을 생성 -> 그룹 평균 대비 advantage 로 학습
```

## 다음 챕터 예고

**Chapter 31. GRPO — verifier reward 로 정렬 (Group Relative Policy Optimization)**

- DPO 는 *사람이 비교한 preference 쌍* 으로 정렬했다면, GRPO 는 *verifier 가 자동 채점한 reward* 로 정렬 — 수학·코드처럼 *정답을 자동 검증* 할 수 있는 영역
- *같은 prompt 에 여러 답을 rollout* → *그룹 안에서 상대 비교* (group relative advantage) → reward 높은 답 쪽으로 정책 강화
- reward model 도, critic 도 없이 *그룹 평균을 baseline* 으로 advantage 를 만드는 *PPO 의 또 다른 간소화*
- alignment 의 *두 방식 비교*: **DPO (주관적 선호, 사람 비교) vs GRPO (객관적 정답, 자동 검증)**

**Phase 4 GPT 시대 4단계 흐름 정리**:

| 챕터 | 단계 | 본체 | 데이터 | 학습 신호 |
|---|---|---|---|---|
| Ch 24·26 | 1 (pretraining) | 작은 GPT scratch | TinyStories (영/한) | next-token |
| Ch 25·27 | 2 (continual pretraining) | gpt2 / KoGPT2 | TinyStories (동일) | next-token |
| Ch 28 | 3 (SFT) | KoGPT2 | KoAlpaca instruction-response | response 토큰 |
| **Ch 30 ← 여기** | **4 (alignment, DPO)** | **SFT 모델 + frozen ref** | **preference 쌍 (chosen/rejected)** | **chosen 선호 ↑, rejected ↓** |
| Ch 31 | 4 (alignment, GRPO) | SFT 모델 + verifier | verifiable-reward prompts | group relative advantage |

> **변하는 축** (Ch 28 → Ch 30): *학습 단계* (SFT → alignment). 본체·토크나이저는 SFT 모델을 잇고, *데이터 (preference 쌍) + trainer (`DPOTrainer`) + loss (DPO sigmoid) + reference 모델* 이 바뀝니다. `labels = -100` 의 *response-only* 원리는 DPO 의 log-prob 계산에서도 이어집니다 — Phase 4 를 관통하는 thread.
