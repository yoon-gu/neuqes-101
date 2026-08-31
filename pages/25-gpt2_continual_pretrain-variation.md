본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래 코드를 출발점으로:

### 변형 1. epoch 수 늘리기 — *언제 catastrophic forgetting 이 시작되는가*

`num_train_epochs` 만 늘려 학습량을 키워 봅니다. TinyStories 적응은 강해지지만, 너무 많이 돌리면 gpt2 가 원래 갖고 있던 일반 도메인 능력이 덮어쓰이기 시작합니다 — 그 경계가 어디인지 직접 관찰해 보세요.

```python
# args.num_train_epochs = 3  # 또는 5
# 더 많은 epoch -> TinyStories 적응 강해짐, 다만 WebText 도메인 능력 손실 위험
```

### 변형 2. 더 큰 본체 (gpt2-medium, 355M)

본체만 `gpt2-medium`(355M)으로 키워 봅니다. 토크나이저·collator·loss 는 그대로지만, T4 16GB 에 맞추려면 batch 를 줄이고 gradient accumulation 을 늘려야 하며 학습 시간이 30분 룰 한계에 다가가는 점을 유의하세요.

```python
# model = AutoModelForCausalLM.from_pretrained("gpt2-medium")  # 355M
# T4 16GB 에서 fp16 + per_device_train_batch_size=2, gradient_accumulation_steps=8 권장
# 학습 시간 약 25-30분 — 30분 룰 한계
```

### 변형 3. 다른 도메인 데이터 — *continual pretraining 의 일반성*

데이터만 코드·의료·법률 등 다른 도메인으로 바꿔 봅니다. 본체와 토크나이저는 그대로 두고 데이터만 교체하면 그 도메인으로 적응한다는 점이 continual pretraining 의 일반성을 보여줍니다 — TinyStories 가 특별해서가 아닙니다.

```python
# TinyStories 대신 코드 (예: bigcode/the-stack-smol) / 의료 텍스트 / 법률 문서 등
# raw_train = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train[:5000]")
# 본체 + 토크나이저는 그대로, 데이터만 교체 -> 도메인 적응
```

### 변형 4. catastrophic forgetting 직접 확인

학습 후 모델에 *동화와 무관한 prompt* 를 넣어 봅니다. 학습 전이라면 다양하게 답했을 prompt 가 학습 후 동화 풍으로 끌려간다면, 그것이 catastrophic forgetting 의 정성적 신호입니다.

```python
# 학습 후 모델에 *비-동화 prompt* (예: "The quick brown fox", "Albert Einstein was") 를 넣어보면
# - 학습 전: WebText 풍 다양한 답
# - 학습 후: 동화 풍으로 끌려가는 경향 (TinyStories 도메인에 과적응)
# 이게 catastrophic forgetting 의 정성적 신호 — FAQ Q5 참고
```
