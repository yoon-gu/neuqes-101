본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래를 출발점으로:

### 변형 1. group size (`num_generations`)

group 크기는 baseline (group 평균) 추정의 안정성과 rollout 비용을 동시에 좌우합니다. 키우면 advantage 가 정밀해지지만 T4 시간이 그만큼 늘어납니다. 한 줄만 바꿔 가며 출발점을 잡아 보세요.

```python
# grpo_config.num_generations = 8   # group 키우면 baseline (group 평균) 추정이 안정 -> advantage 정밀
# # 단 rollout 비용 = group size 에 비례 (T4 시간 증가)
# 4 가 T4 출발점. group 안에 정답·오답이 섞이려면 너무 작지 않아야 함 (2 는 비교가 빈약).
```

### 변형 2. format reward 추가 — 여러 verifier 조합

`reward_funcs` 는 *리스트* 로 줄 수 있습니다. 정답 reward + *형식 reward* (예: 정해진 형식으로 답했나) 를 합칠 수 있습니다:

```python
def reward_format(completions, **kwargs):
    '''정답을 '#### 숫자' 형식으로 냈으면 보너스 (형식 준수도 verifiable).'''
    return [0.2 if re.search(r"####\s*-?\d+", c) else 0.0 for c in completions]

# 여러 verifier 를 리스트로 -> reward 가 합산됨 (reward_weights 로 가중치도 가능)
trainer = GRPOTrainer(model=policy, reward_funcs=[reward_correct, reward_format], ...)
```

### 변형 3. 코드 verifier

산술 대신 *코드 생성* task 면, verifier 가 *생성 코드를 실행해 테스트 통과 여부* 를 채점합니다:

```python
def reward_code(completions, test_cases, **kwargs):
    '''생성 코드를 샌드박스에서 실행 -> 테스트 통과하면 1.0 (주의: 샌드박스 필수).'''
    return [1.0 if run_tests_safely(c, t) else 0.0 for c, t in zip(completions, test_cases)]
```

> 코드 실행은 *보안 샌드박스* 가 필수이고 T4 + 30분 룰엔 무거워 본 챕터는 산술로 한정했습니다. 원리는 동일 — *자동 검증 → reward*.

### 변형 4. GSM8K 등 실제 수학 데이터

산술 대신 실제 수학 문제 데이터로 옮기면 알고리즘은 그대로지만 정답 추출이 까다로워집니다. GSM8K 는 정답이 `#### 42` 형식으로 들어 있어, verifier 의 정답 파싱을 그 형식에 맞춰야 합니다.

```python
# from datasets import load_dataset
# gsm = load_dataset("openai/gsm8k", "main", split="train")
# 정답 추출이 더 까다로움 (답이 '#### 42' 형식) -> verifier 의 정답 파싱을 맞춰야 함
```

> 모든 변형의 공통점: *verifier 를 어떻게 정의하나* 가 핵심입니다. *무엇을 reward 로 줄지* = *어떤 능력을 정렬할지*. GRPO 알고리즘 자체는 동일합니다.
