본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래 코드를 출발점으로:

### 변형 1. epoch 수 늘리기 — *언제 catastrophic forgetting 이 시작되는가*

```python
# args.num_train_epochs = 3  # 또는 5
# 더 많은 epoch -> TinyStories 적응 강해짐, 다만 일반 한국어 도메인 능력 손실 위험
```

### 변형 2. 다른 도메인 데이터 — *continual pretraining 의 일반성*

```python
# 한국어 TinyStories 대신 한국어 위키 / 한국어 뉴스 / 한국어 코드 주석 등
# 본체 + 토크나이저는 그대로, 데이터만 교체 -> 도메인 적응
```

### 변형 3. catastrophic forgetting 직접 확인

```python
# 학습 후 모델에 *비-동화 prompt* (예: "대한민국의 수도는", "인공지능은") 를 넣어보면
# - 학습 전: 일반 도메인 풍 다양한 답
# - 학습 후: 동화 풍으로 끌려가는 경향 (TinyStories 도메인에 과적응)
# 이게 catastrophic forgetting 의 정성적 신호 - FAQ Q5 참고
```

### 변형 4. lr 키워 보기 — *왜 작은 lr 가 표준인가*

```python
# args.learning_rate = 5e-4  # Ch 26 의 scratch lr
# 큰 lr -> 사전학습 표상이 빠르게 덮어쓰기됨 -> 학습 전 자연스러움 손실 위험
# 작은 lr (2e-5) 가 continual pretraining 표준인 이유를 직접 체감
```
