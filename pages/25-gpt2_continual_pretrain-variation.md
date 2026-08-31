본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래 코드를 출발점으로:

### 변형 1. epoch 수 늘리기 — *언제 catastrophic forgetting 이 시작되는가*

```python
# args.num_train_epochs = 3  # 또는 5
# 더 많은 epoch -> TinyStories 적응 강해짐, 다만 WebText 도메인 능력 손실 위험
```

### 변형 2. 더 큰 본체 (gpt2-medium, 355M)

```python
# model = AutoModelForCausalLM.from_pretrained("gpt2-medium")  # 355M
# T4 16GB 에서 fp16 + per_device_train_batch_size=2, gradient_accumulation_steps=8 권장
# 학습 시간 약 25-30분 — 30분 룰 한계
```

### 변형 3. 다른 도메인 데이터 — *continual pretraining 의 일반성*

```python
# TinyStories 대신 코드 (예: bigcode/the-stack-smol) / 의료 텍스트 / 법률 문서 등
# raw_train = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train[:5000]")
# 본체 + 토크나이저는 그대로, 데이터만 교체 -> 도메인 적응
```

### 변형 4. catastrophic forgetting 직접 확인

```python
# 학습 후 모델에 *비-동화 prompt* (예: "The quick brown fox", "Albert Einstein was") 를 넣어보면
# - 학습 전: WebText 풍 다양한 답
# - 학습 후: 동화 풍으로 끌려가는 경향 (TinyStories 도메인에 과적응)
# 이게 catastrophic forgetting 의 정성적 신호 — FAQ Q5 참고
```
