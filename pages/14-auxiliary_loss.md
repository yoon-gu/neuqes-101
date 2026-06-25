**목표**: Ch 13의 multi-label 항목 분류를 *메인 task* 로 그대로 두고, **별점 회귀 보조 헤드** 를 추가합니다. 손실은 가중합 형태:

$$L = L_\text{main}(\text{항목 BCE per-label}) + \lambda \cdot L_\text{aux}(\text{별점 MSE})$$

핵심 질문은 *"보조 task가 메인 task의 정확도를 끌어올리는가?"* — 같은 BERT 본체를 두 task가 *공유* 학습하면서, 별점이라는 *연속적이고 일관성 있는 신호* 가 항목 분류 표현에 도움이 되는지 직접 측정합니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 22분 (보조 ON 학습 ~9분 + 보조 OFF 비교용 학습 ~9분 + 평가/시각화)

## 학습 흐름

1. 🚀 **실습**: Ch 13의 데이터 + *별점 보조 회귀 라벨* (1★→0.0, 5★→1.0 스케일) 추가. `AutoModelForSequenceClassification` (`num_labels=5`, multi-label) 에 `aux_head = Linear(H, 1)` 한 줄 추가, `Trainer.compute_loss` 오버라이드.
2. 🔬 **해부**: 메인 metric (per-label F1, hamming, AUC) + 보조 metric (RMSE, Pearson r) 동시 측정.
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 **λ=0 baseline** (= Ch 13 재현)을 학습한 뒤 **λ=0.05 (sweet spot)** 결과와 비교 — *보조 loss가 메인 task를 끌어올렸는가?* 라벨별 F1 차이로 시각화. (sweet spot 은 부록 `14_auxiliary_loss_lambda_sweep` 의 λ 스윕에서 찾았습니다.)

> 📒 **사전 학습 자료**: Ch 9 (BERT 회귀 — MSELoss), Ch 13 (BERT multi-label — BCE per-label). 이번 챕터는 둘을 한 모델 안에 같이 넣습니다.

> ⚠️ **이번 챕터에 *처음* 등장**: `Trainer.compute_loss` 오버라이드 — `problem_type` 만으로 매핑할 수 없는 *복합 loss* 를 직접 계산하는 패턴.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 9 | DistilBERT 파인튜닝 | WordPiece | Yelp 별점 | `Linear(H, 1)` | 없음 | `MSELoss` |
| 13 | DistilBERT 파인튜닝 | WordPiece | Yelp + 항목 합성 | `Linear(H, 5)` | sigmoid (각각) | `BCEWithLogitsLoss` |
| **14 ← 여기** | DistilBERT + **보조 헤드** | WordPiece | Yelp + 항목 + **별점** | **메인(5) + 보조(1)** | 메인 sigmoid + 보조 없음 | **`BCE per-label + λ·MSE`** |
| 15 (다음 Phase 2) | klue/bert-base | WordPiece (한국어) | NSMC | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 13)

| 축 | Ch 13 (multi-label) | Ch 14 (multi-label + auxiliary) |
|---|---|---|
| Task (메인) | 5라벨 multi-label | (그대로) |
| `num_labels` | 5 | (그대로) |
| `problem_type` | `multi_label_classification` | (그대로 — 자동 매핑은 메인 loss 만 처리) |
| 메인 활성화 / loss | per-label sigmoid / BCE | (그대로) |
| **보조 head** | 없음 | **새로 추가**: `Linear(H, 1)` linear regressor |
| **보조 라벨** | 없음 | **새로 추가**: 별점 0-1 스케일 (`label / 4`, 1★→0.0, 5★→1.0) |
| **보조 loss** | — | **`MSELoss`** (Ch 9 그대로) |
| **결합 loss** | `outputs.loss` 자동 | **`L_main + λ·L_aux`** 직접 계산 |
| `Trainer.compute_loss` | 자동 (오버라이드 X) | **오버라이드 필수** |
| 데이터 콜레이터 | `DataCollatorWithPadding` 자동 | **커스텀** — `aux_labels` 도 같이 batching |
| 학습 hyperparams | (epoch=2, lr=2e-5, …) | (그대로) |

> **변하는 축 — Loss 축 끝**: 메인 task와 모델 본체는 *완전히 동일*, *Loss에 보조 항이 가중합으로 추가* 됩니다. 실무에서 *학습 데이터에 추가 신호가 있을 때* 이를 활용하는 정통 multi-task learning 패턴.

### 왜 보조 task가 메인 task에 도움이 되는가 (한 줄 요약)

BERT 본체(67M 파라미터)가 *공유* 되어 메인 헤드와 보조 헤드가 같은 768-dim CLS 표현을 입력으로 받습니다. 보조 task가 메인과 *부분적으로 상관* 이면 보조 학습이 본체를 *더 일반적인 표현* 으로 끌고 가서 메인 정확도까지 올라갑니다. 다음 섹션에서 *어떤 상황에서 왜 도움이 되는지* 다섯 갈래로 자세히 봅니다.

## 왜 Auxiliary Loss 인가 — 다섯 가지 동기

코드를 짜기 전에 *언제·왜* 보조 loss를 쓰는지 정리합니다. 이번 챕터의 셋업이 다섯 동기 중 어디에 해당하는지 의식적으로 짚어 두면 학습 결과를 *예측하고 해석* 하는 감각이 생깁니다.

### (1) 정규화 (Regularization) — 가장 흔한 목적

메인 task만 학습하면 모델이 *그 task 한정* 으로 과적합됩니다. 보조 task가 BERT 본체를 *더 일반적인* 표현으로 끌고 가서 메인에서도 일반화가 좋아집니다. Dropout과 비슷한 효과지만 **데이터 신호** 로 정규화하는 점이 다릅니다.

> **이번 챕터의 주된 목적이 이거.** 항목 라벨이 키워드 매칭으로 *합성* 되어 노이즈가 큰데, 별점은 사용자가 *직접* 매긴 깨끗한 ground truth. 별점 회귀를 보조로 두면 BERT가 *과도하게 키워드 매칭에 맞추는* 걸 막아 일반화에 도움.

### (2) 데이터 효율 — *공짜로 있는* 메타데이터 활용

메인 task의 라벨링 비용이 큰데 *부가 신호* (별점, 작성일, 길이, 작성자, 카테고리 …)가 데이터에 *공짜로 함께 있는* 시나리오. 메인 라벨이 부족해도 보조 신호를 *동시에* 학습하면 BERT 본체가 더 풍부하게 학습됩니다.

```python
# 일반 패턴
main_label = expensive_human_annotated_aspects   # 라벨링당 $0.50
aux_label  = star_rating                         # 데이터에 이미 있음 ($0)
# main 5,000건 + aux 5,000건 = "사실상" 두 배 데이터 효과
```

### (3) 도메인 지식 주입 (Inductive Bias)

알고 있는 *구조적 정보* 를 보조 task로 모델에 *강제* 학습시키는 방법.

| 메인 task | 보조 task | 주입되는 지식 |
|---|---|---|
| NER (개체명 추출) | POS tagging (품사) | 구문 구조 |
| 객체 검출 (위치+범주) | 세그멘테이션 (경계) | 위치-경계 일관성 |
| 한국어 분류 | 한자 음운 예측 | 어휘 형태소 구조 |
| 영화 리뷰 감성 | 영화 장르 분류 | 장르별 감성 표현 차이 |

연구·실무에서 *어떤 보조를 써야 메인이 좋아질지* 는 도메인 지식의 영역. 입문 수준에선 *데이터에 있는 신호 중 메인과 양의 상관* 인 걸 골라 시작.

### (4) 학습 안정화 — vanishing gradient 완화

깊은 네트워크에서 gradient가 *최하위 층까지 도달하지 못하는* 문제를 보조 분류 헤드를 *중간 층에* 달아 해결한 패턴. **GoogLeNet (2014)** 의 intermediate auxiliary classifiers 가 원조.

> BERT는 12-24 layer로 *그리* 깊지 않아 이 목적은 약합니다. CNN 100+ layer (ResNet-152) 시대의 잔재. 입문 수준에선 *왜 이런 패턴이 historically 등장했는지* 만 알아두면 충분.

### (5) 운영 시점에 둘 다 필요 — 진짜 multi-task

추론 시점에 *두 task 결과가 모두 필요* 한 경우 — 한 번의 forward로 항목 분류 + 별점 예측을 동시에 받기. 엄밀히 말하면 *auxiliary* 가 아니라 *joint training* 이 됩니다.

> Ch 14는 *auxiliary* 패턴이라 추론 시 보조 헤드를 *호출하지 않습니다*. 별점 회귀는 학습 정규화 용도로만 사용. 운영에서 별점 예측이 필요하면 *auxiliary* 가 아니라 *joint multi-task* 를 의도적으로 설계.

### Ch 14의 위치 — 동기 (1) + (2) 의 결합

이번 챕터의 셋업은 다섯 중 두 가지에 해당합니다:

- **(1) 정규화**: 합성 항목 라벨의 노이즈를 깨끗한 별점 신호로 정규화 — 주된 효과
- **(2) 데이터 효율**: 별점은 Yelp 데이터에 *이미* 있어 추가 라벨링 비용 0

학습 결과를 해석할 때 두 효과가 *같이* 나타날 거라 예측. 메인 F1이 1-3%p 올라가면 정규화 효과가 두드러진 것, 5%p 이상이면 데이터 효율 + 정규화가 합쳐진 것.

### 언제 *안* 쓰나 — 보조가 *해로운* 경우

| 조건 | 결과 | 진단 신호 |
|---|---|---|
| 보조가 메인과 *반대 신호* | gradient 충돌, 학습 발산 | 두 loss가 *둘 다 정체*, λ 키울수록 메인 metric 하락 |
| 보조 라벨이 *극도로 sparse* | 학습 신호 부족 | aux loss가 거의 0, 메인엔 영향 없음 |
| 보조가 메인보다 *훨씬 어려움* | 학습량 부족으로 둘 다 학습 안 됨 | 두 metric 모두 baseline 이하 |
| λ 튜닝 비용을 감당 못 함 | 잘못된 λ로 메인이 망가짐 | 그냥 단일 task가 더 안전 |

### 역사적 교훈 — RoBERTa의 NSP 제거

BERT 자체가 *MLM + NSP (Next Sentence Prediction)* 이라는 멀티태스크 사전학습으로 만들어졌습니다 — MLM이 메인, NSP가 보조였던 셈. 그런데 **RoBERTa (2019)** 가 *NSP를 빼면 오히려 성능이 좋아진다* 는 걸 발견하고 NSP를 제거. 이후 거의 모든 BERT 후속 모델 (DeBERTa, ELECTRA …)이 NSP 없이 학습됨.

> **시사점**: 보조 task가 *직관적으로* 도움 될 것 같아도 *실제로 도움이 되는지는 측정* 해야 합니다. 그래서 이번 챕터의 §8 클라이맥스가 λ=0 baseline과의 *직접 비교* 인 것 — "직관" 대신 "측정".

## Loss 노트 — Combined loss `L = L_main + λ · L_aux`

$$L = \underbrace{\frac{1}{N \cdot K}\sum_{i,k}\text{BCE}(z_{i,k}^{main}, y_{i,k}^{main})}_{L_\text{main}: \text{항목 BCE per-label}} + \lambda \cdot \underbrace{\frac{1}{N}\sum_{i}(z_{i}^{aux} - y_{i}^{aux})^2}_{L_\text{aux}: \text{별점 MSE}}$$

- $z^\text{main} \in \mathbb{R}^5$ — 항목 logit 5개, sigmoid 후 BCE.
- $z^\text{aux} \in \mathbb{R}$ — 별점 회귀 logit (활성화 없음, 직접 MSE).
- $\lambda$ — 보조 loss의 가중치 (hyperparameter, 보통 0.1-10 범위 탐색).

**λ 선택 가이드 — 숫자로 감 잡기**

| λ | 의미 | 메인 항목 분류 효과 (이 챕터 부록 스윕 실측) |
|---|---|---|
| 0.0 | 보조 loss 무시 | Ch 13과 동일한 baseline (micro-F1 0.840) |
| **0.05** | 보조 *약하게* 반영 | **sweet spot — 메인이 올라감 (micro-F1 0.847 ↑)** |
| 0.1 | 보조 약하게 | 여전히 메인 향상 (0.844 ↑) |
| 0.3 | 보조 강화 | 메인이 baseline 아래로 (0.803 ↓) |
| 1.0 | 보조와 메인 동등 가중 | 보조 과다 — 메인 폭락 (micro 0.66, macro 0.39) |

이번 챕터에선 **λ=0.05** 로 학습하고 λ=0 baseline 과 비교합니다 — 부록 `14_auxiliary_loss_lambda_sweep` 의 λ 스윕에서 **λ=0.05 가 메인을 가장 끌어올리는 sweet spot** 으로 나왔기 때문입니다. *분류 BCE + 회귀 MSE* 처럼 두 손실의 스케일이 다르면 최적 λ 는 1 보다 훨씬 작습니다.

**숫자로 감 잡기 (단일 샘플)** — 항목 multi-hot $\mathbf{y}^\text{main} = [1, 0, 1, 0, 1]$, 별점 보조 라벨 $y^\text{aux} = 0.75$ (4★/4):

| 단계 | 값 |
|---|---|
| $L_\text{main}$ (Ch 13의 BCE per-label 평균) | 0.45 |
| $L_\text{aux}$ ($(z^\text{aux} - 0.75)^2$, 가정 $z^\text{aux} = 0.55$) | $(0.55 - 0.75)^2 = 0.04$ |
| $L$ (λ=1) | $0.45 + 1 \cdot 0.04 = 0.49$ |
| $L$ (λ=10) | $0.45 + 10 \cdot 0.04 = 0.85$ ← 보조 항이 메인보다 큼 |

λ가 너무 크면 *메인 신호 자체가 보조에 묻힙니다*(이 챕터 λ=1 이 그 경우). 반대로 너무 작으면 *보조 신호가 학습에 영향 안 줌*. λ=1 은 *두 손실의 스케일이 비슷할 때* 의 출발점일 뿐, BCE+MSE 조합인 이 챕터의 sweet spot 은 부록 스윕이 보여주듯 **λ≈0.05** 입니다.

## 토크나이저 노트

Ch 13과 완전히 동일 — `distilbert-base-uncased` WordPiece, `max_length=128`. 토크나이저는 *라벨에 무관* 합니다. 보조 라벨이 추가됐다고 토큰화가 바뀌지는 않음 — 별점 정수가 *float 값으로 한 칸 더* 추가될 뿐.

> **다음 챕터(Ch 15)부터 Phase 2**: 같은 셋업을 한국어 BERT(`klue/bert-base`)에서 다시. 토크나이저가 *영어 WordPiece → 한국어 WordPiece* 로 바뀌는 게 변화의 본질.

## 이 장의 구성

[[SubPages]]
