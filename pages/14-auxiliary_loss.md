**목표**: Ch 13의 multi-label 항목 분류를 *메인 task* 로 그대로 두고, **별점 회귀 보조 헤드** 를 추가합니다. 손실은 가중합 형태:

$$L = L_\text{main}(\text{항목 BCE per-label}) + \lambda \cdot L_\text{aux}(\text{별점 MSE})$$

핵심 질문은 *"보조 task가 메인 task의 정확도를 끌어올리는가?"* — 같은 BERT 본체를 두 task가 *공유* 학습하면서, 별점이라는 *연속적이고 일관성 있는 신호* 가 항목 분류 표현에 도움이 되는지 직접 측정합니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 22분 (보조 ON 학습 ~9분 + 보조 OFF 비교용 학습 ~9분 + 평가/시각화)


## 학습 흐름

1. 🚀 **실습**: Ch 13의 데이터 + *별점 보조 회귀 라벨* (1★→0.0, 5★→1.0 스케일) 추가. `AutoModelForSequenceClassification` (`num_labels=5`, multi-label) 에 `aux_head = Linear(H, 1)` 한 줄 추가, `Trainer.compute_loss` 오버라이드.
2. 🔬 **해부**: 메인 metric (per-label F1, hamming, AUC) + 보조 metric (RMSE, Pearson r) 동시 측정.
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 **λ=0 baseline** (= Ch 13 재현)을 학습한 뒤 λ=1 결과와 비교 — *보조 loss가 메인 task에 도움이 됐는가?* 라벨별 F1 차이로 시각화.


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

| λ | 의미 | 효과 (예상) |
|---|---|---|
| 0.0 | 보조 loss 무시 | Ch 13과 *완전히 동일* (이번 챕터 §5 baseline) |
| 0.1 | 보조 *약하게* 반영 | 메인이 압도적, 보조는 살짝 정규화 효과 |
| 1.0 | **보조와 메인 *동등* 가중** | 표준 시작점. 두 task가 균형 잡혀 학습 |
| 5.0 | 보조 우세 | 보조 task가 학습 방향을 지배 — 메인 task 성능 떨어질 수 있음 |
| 10+ | 보조 절대 우세 | 메인 task 신호가 거의 묻힘. 권장 안 됨 |

이번 챕터에선 **λ=1** 로 학습하고 λ=0 baseline과 비교. 실무에선 validation set에서 λ를 grid search (0.1, 0.3, 1, 3, 10).

**숫자로 감 잡기 (단일 샘플)** — 항목 multi-hot $\mathbf{y}^\text{main} = [1, 0, 1, 0, 1]$, 별점 보조 라벨 $y^\text{aux} = 0.75$ (4★/4):

| 단계 | 값 |
|---|---|
| $L_\text{main}$ (Ch 13의 BCE per-label 평균) | 0.45 |
| $L_\text{aux}$ ($(z^\text{aux} - 0.75)^2$, 가정 $z^\text{aux} = 0.55$) | $(0.55 - 0.75)^2 = 0.04$ |
| $L$ (λ=1) | $0.45 + 1 \cdot 0.04 = 0.49$ |
| $L$ (λ=10) | $0.45 + 10 \cdot 0.04 = 0.85$ ← 보조 항이 메인보다 큼 |

λ가 너무 크면 *메인 신호 자체가 보조에 묻힙니다*. 반대로 너무 작으면 *보조 신호가 학습에 영향 안 줌*. λ=1이 균형의 기본 출발점.

## 토크나이저 노트

Ch 13과 완전히 동일 — `distilbert-base-uncased` WordPiece, `max_length=128`. 토크나이저는 *라벨에 무관* 합니다. 보조 라벨이 추가됐다고 토큰화가 바뀌지는 않음 — 별점 정수가 *float 값으로 한 칸 더* 추가될 뿐.

> **다음 챕터(Ch 15)부터 Phase 2**: 같은 셋업을 한국어 BERT(`klue/bert-base`)에서 다시. 토크나이저가 *영어 WordPiece → 한국어 WordPiece* 로 바뀌는 게 변화의 본질.

**baseline VRAM**:

## 데이터 — Yelp + 항목 (Ch 13) + 별점 보조 라벨

Ch 13의 항목 합성 라벨을 그대로 쓰고, **별점 보조 회귀 라벨** 을 추가합니다. 별점은 1-5 정수지만 회귀 헤드와 MSE를 자연스럽게 쓰기 위해 *0-1 스케일* 로 변환만 해 둡니다 (학습 정규화 효과를 위한 데이터 가공이 아니라, 그냥 단위만 맞추는 작업).

- 메인 라벨 $\mathbf{y}^\text{main} \in \{0, 1\}^5$ — 항목 multi-hot.
- 보조 라벨 $y^\text{aux} = \text{label} / 4 \in [0, 1]$ — 1★ → 0.0, 5★ → 1.0.

## 토큰화 — 메인 multi-hot + 보조 float 같이 부착

`tokenize_fn` 이 두 라벨을 모두 attach. 메인은 `labels` (multi-hot float), 보조는 `aux_labels` (float scalar).

## 커스텀 Data Collator — `aux_labels` 도 batch에 같이 담기

기본 `DataCollatorWithPadding` 은 input_ids·attention_mask·labels 만 알고 있어 *추가 라벨* 은 통과시키지 못합니다. 한 줄짜리 wrapper로 `aux_labels` 를 텐서로 만들어 batch에 추가합니다.

## 모델 셋업 — Ch 13 모델 + 보조 헤드 한 줄 추가

`AutoModelForSequenceClassification` (Ch 13과 *완전히 동일*) 을 로드한 뒤 `model.aux_head = nn.Linear(...)` 한 줄로 보조 헤드를 *모델 객체에 attach*. 이후 `Trainer.compute_loss` 가 메인 출력 + 보조 헤드를 동시에 사용해 결합 loss 를 계산합니다.

**보조 헤드는 ~770개 파라미터** — 768→1 Linear의 weight + bias. 전체 67M 의 *0.001%*. 이 *미세한 추가 자유도* 만으로 멀티태스크 학습이 동작합니다.

## 커스텀 Trainer — `compute_loss` 오버라이드

핵심 로직 (코드 한 줄로 요약):

```python
loss = outputs.loss + λ · MSE(aux_head(CLS), aux_labels)
```

- `outputs.loss` 는 `problem_type="multi_label_classification"` 자동 매핑으로 이미 BCE per-label 평균이 계산됨.
- 보조 loss는 우리가 *직접 계산* — `output_hidden_states=True` 로 받은 마지막 layer의 CLS 표현을 `aux_head` 에 통과.

**평가용 metric 함수** — 메인 (Ch 13과 동일) + 보조 (RMSE, R², Pearson r). 보조 logit 추출은 `Trainer.predict()` 가 메인 logits 만 반환하기 때문에 별도 단계로 빼서 처리.

## 학습 — λ=1 (보조 ON)

Ch 13과 동일한 hyperparams. `AuxTrainer` + `lambda_aux=1.0`.

**중요: `remove_unused_columns=False`** — Trainer는 기본으로 *model.forward 시그니처에 없는 컬럼* 을 제거합니다. `aux_labels` 는 모델 시그니처에 없어 자동 제거되면 우리 `compute_loss` 가 받을 수 없습니다. 이 옵션을 꺼야 함.

## 평가 — 메인 task + 보조 task

메인 metric 은 자동으로 계산됨 (`compute_metrics`). 보조 metric (RMSE, R², Pearson r) 은 별도 forward로 보조 logits 를 추출해 측정.

## 클라이맥스 — *λ=0 baseline* 학습 (= Ch 13 재현)

같은 코드를 `lambda_aux=0.0` 으로 한 번 더 돌립니다. 그러면 보조 loss 의 gradient 가 0이 되어 메인 task만 학습되는 상태 = **Ch 13과 정확히 동일한 학습 결과**. (보조 헤드는 학습되긴 하지만 메인 학습엔 영향 없음.)

> 의도적으로 *Ch 13 노트북을 따로 돌리지 않고* 이 셀에서 baseline을 다시 만듭니다 — 비교가 *같은 노트북·같은 환경* 안에서 self-contained 하도록.

### 8-1. 메인 metric 비교 — λ=0 baseline vs λ=1 aux

**해석 가이드**

- `delta` > 0 — 보조 loss 가 메인 task 에 *도움* 이 됨 (멀티태스크의 정통 효과).
- `delta` < 0 — 보조 loss 가 메인 task 를 *방해* 함 (λ가 너무 큼 / 보조 task 가 메인과 상관 약함).
- `delta` ≈ 0 — 별 영향 없음 (보조 신호가 메인 표현에 의미 없는 추가).

별점은 항목 분포와 *부분적으로* 상관 (긍정 항목 → 높은 별점) 이라 *작은 양의 delta* 가 자연스러운 결과. 0.5%p 정도면 노이즈일 수 있고, 1-2%p 면 의미 있는 효과.

### 8-2. 라벨별 F1 비교 — 어느 항목이 보조 loss로 가장 도움받았나

**해석**

- **별점과 상관이 강한 항목** (food, service): 보조 별점 회귀 학습이 *긍정/부정 신호* 를 잘 잡으면 도움이 됩니다. 작은 양의 delta 기대.
- **별점과 상관이 약한 항목** (location, price): 별점 신호가 *직접적 도움* 이 안 됨. delta가 0 근처거나 약간 음수일 수 있음.
- **분산이 큰 라벨** — eval 표본이 적어 F1 자체가 노이즈가 큼. delta 도 의미 해석 조심.

### 8-3. 보조 task 자체는 얼마나 잘 학습됐나

별점 회귀가 잘 됐다는 건 BERT 본체가 *별점 신호도 효율적으로 인코딩* 하고 있다는 뜻 — 메인 task 표현에도 그 신호가 들어가 있을 가능성.

**해석**

- 각 violin이 *해당 별점의 정답 위치* (점선 가이드: 1★=0.0, 2★=0.25, 3★=0.5, 4★=0.75, 5★=1.0) 에 *중심* 하면 보조 head 가 별점 신호를 잘 학습한 것.
- violin 의 *너비* = 그 별점 내 예측 분산. 너비가 좁을수록 모델이 자신 있게 회귀 — 모든 별점에서 좁으면 calibration 이 좋음.
- 가장 어려운 별점은 보통 *3★* (중간값) — 사람도 모호한 평가라 violin 이 길게 늘어지거나 인접 별점 위치까지 침범하면 자연스러움.
- 1★/5★ 양 끝 violin 이 정답 위치에서 *체계적으로 안쪽* 으로 치우치면 *극단값에 보수적* 인 회귀 — MSE 가 양 끝에서 손실이 작아지는 특성과 결부된 일반적 경향.

## 결과 해석 — 보조 loss 가 *항상 좋게 나오지는 않습니다*

이 챕터의 §8 비교가 다음과 같은 형태로 나오면 *그게 자연스러운 모습* 이라는 점을 먼저 짚습니다. 학습 데이터·에폭·λ 모두 *입문 수준 빠르게 검증* 에 맞춰져 있어 *드라마틱한 양의 delta* 가 항상 나오는 셋업이 아닙니다.

### 가능한 결과 시나리오 4가지

| delta 패턴 | 의미 | 권장 다음 단계 |
|---|---|---|
| **모든 항목 +1-3%p** | 정규화 효과가 고르게 작동 (이상적) | λ 를 0.3·1·3 으로 grid search 해 최적점 탐색 |
| **일부 항목만 향상, 나머지 0 또는 음수** ← *이번 quick 모드의 전형* | 별점과 *상관 강한* 항목 (food/service) 만 보조의 도움을 받음. 활성률이 낮은 항목 (price/ambiance/location) 은 *학습 데이터 자체가 부족* 해 둘 다 학습 안 됨 | 데이터·에폭 늘리기가 우선, λ 보다 더 큰 lever |
| **모든 항목 거의 0 변화** | 보조 신호와 메인 task 가 *서로 무관* — BERT 본체가 두 신호를 분리 표상으로 학습하면 결합 효과 없음 | 다른 보조 task 시도 (작성자, 길이 분포, 작성 시각 등) 또는 정규화 목적은 *dropout/weight decay* 강화로 대체 |
| **모든 항목 음수** | 보조가 메인 학습을 *방해* — 신호 충돌 가능성 (FAQ Q4) | λ 줄이거나(0.1, 0.01) 보조 task 제거. 이게 RoBERTa 가 NSP 를 뺀 이유 |

### 우리 결과를 *솔직하게* 읽기

§8-2 의 라벨별 F1 표 가 *어떤* 패턴인지 확인:

- **food, service 가 양의 delta** + **price·ambiance·location 이 0** → 두 번째 시나리오. *학습 데이터가 부족한 항목이 0 으로 수렴* 한 것이지 *보조가 해롭다* 는 신호가 아님. Colab T4 풀 모드 (5,000 샘플 + 2 epoch) 로 돌리면 활성률 낮은 항목도 학습이 시작돼 delta 가 더 고르게 나옵니다.
- **macro AUC 가 살짝 음수** + **macro F1 양수** → noise 영역. 평가 셋이 작으면 (100건 quick 모드) 두 metric 모두 ±0.01 정도 흔들리는 게 정상.
- **보조 task 자체** (별점 회귀 RMSE / Pearson) 는 *보통 잘 학습* 됨 (Pearson 0.7+). 보조 학습이 *작동* 하지 않아서 delta 가 작은 게 아니라, 보조 신호의 *유용성* 이 항목별로 다르기 때문.

### 그럼 auxiliary loss 의 가치는 어디에서?

quick 모드 결과만 보면 "굳이 쓸 필요 있나?" 싶을 수 있습니다. 본문 셋업이 *짧고 작아* 효과가 묻혀서 그렇습니다 — 다음과 같은 *다른 측면* 도 같이 봐 주세요:

1. **학습 안정화 / gradient noise 감소** — 메인 BCE per-label 의 gradient 가 sparse 한 multi-label 에서 보조 MSE 의 *밀도 있는* gradient 가 BERT 본체 학습을 부드럽게 만듭니다. 학습 곡선이 더 안정적 (epoch마다 loss 분산이 작아짐) — metric 한 개로는 안 보이지만 학습 동역학에선 큰 차이.
2. **다운스트림 transfer 에 유리한 표상** — 학습이 끝난 BERT 본체를 *다른 task로 재 fine-tune* 할 때, auxiliary 로 학습된 본체가 *더 일반적* 인 표상을 갖고 있어 새 task 적응이 빠릅니다. 단일 task 학습한 본체는 그 task 에 *과적합* 되어 transfer 성능이 떨어지는 경향.
3. **데이터 효율 누적 효과** — 별점 같은 *공짜 메타데이터* 신호를 추가 라벨링 비용 없이 활용. 데이터셋이 *크면 클수록* 효과 누적. quick 모드 500 건에선 미미해도 5,000건·50,000건에선 의미있는 격차.
4. **λ 그리드의 학습 자원 자체가 가치** — λ ∈ {0.1, 0.3, 1, 3, 10} 시도하면 그중 하나는 보통 *명확하게* 최적. 본문은 λ=1 한 점만 보여주는 *단일 실험*, 실무에선 *grid search* 의 best 가 보조 loss 의 진짜 가치.

> **원고 취지** — auxiliary loss 라는 *기법 자체* 와 *언제 도움이 되는지* 의 이해. quick 모드 한 번 실행으로 "보조 loss 는 좋다 / 나쁘다" 를 단정하기보다, *언제 효과가 나타나는 패턴인지* 의 감각을 잡는 게 이 챕터의 목적. *나쁘게 나온 항목* 도 진단할 수 있는 게 더 중요한 학습 결과입니다.

## 이 장의 구성

- [14-1. 실습](14-auxiliary_loss-practice.md)
- [14-2. 정리와 FAQ](14-auxiliary_loss-wrapup.md)
