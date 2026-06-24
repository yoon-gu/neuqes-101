**목표**: Ch 17의 한국어 multi-label 셋업을 *메인 task* 로 그대로 두고, **활성 라벨 *개수* 회귀 보조 헤드** 를 추가합니다. 손실은 가중합:

$$L = L_\text{main}(\text{카테고리 BCE per-label}) + \lambda \cdot L_\text{aux}(\text{활성 개수 MSE})$$

Ch 14(영어 auxiliary, 별점 회귀 보조)의 한국어 버전입니다. 보조 task만 *별점* → *활성 라벨 개수* (몇 개 카테고리가 동시에 등장하는가) 로 달라집니다. 모델 본체·토크나이저·hyperparams 는 Ch 17 과 *완전히 동일*.

핵심 질문은 Ch 14 와 같습니다 — *"보조 task가 메인 task의 정확도를 끌어올리는가?"* 같은 KLUE-BERT 본체를 두 task가 공유 학습하면서, "이 결합 헤드라인이 몇 개 주제를 다루는가" 라는 *밀도 있는 보조 신호* 가 multi-label 카테고리 예측 표현에 도움이 되는지 직접 측정합니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 22분 (보조 ON 학습 약 10분 + λ=0 baseline 학습 약 10분 + 평가/시각화)


## 학습 흐름

1. 🚀 **실습**: Ch 17 의 KLUE-YNAT 합성 multi-label 데이터에 *활성 라벨 개수* 보조 라벨(`n_active`, 1 또는 2 — 두 헤드라인 결합 시 같은 카테고리면 1) 을 추가. `AutoModel` 위에 메인 헤드 `Linear(H, 7)` 와 보조 헤드 `Linear(H, 1)` 를 직접 attach, `Trainer.compute_loss` 오버라이드.
2. 🔬 **해부**: 메인 metric (micro/macro F1, hamming, AUC) + 보조 metric (RMSE, Pearson r) 동시 측정.
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 **λ=0 baseline** (= Ch 17 재현) 을 학습한 뒤 λ=0.05 결과와 비교 — *보조 loss 가 메인 task 에 도움이 됐는가?* 카테고리별 F1 차이로 시각화.


> 📒 **사전 학습 자료**: Ch 14 (영어 auxiliary, 별점 회귀 보조), Ch 17 (한국어 multi-label, KLUE-YNAT 합성). 이번 챕터는 두 챕터의 *결합* — Ch 17 의 한국어 셋업 + Ch 14 의 multi-task 학습 패턴.

> ⚠️ **이번 챕터의 새로운 점**: 보조 라벨이 *데이터에 외부 신호로 있던* (Ch 14 의 별점) 게 아니라 *합성 과정에서 자연스럽게 얻어지는* (`n_active`) 메타데이터. 합성 multi-label 의 *구조적 정보* 를 그대로 보조로 활용하는 패턴.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 14 | DistilBERT + 보조 헤드 | WordPiece (영어) | Yelp + 항목 + 별점 | 메인(5) + 보조(1) | 메인 sigmoid + 보조 없음 | `BCE per-label + λ·MSE` |
| 16 | klue/bert-base | WordPiece (한국어) | KLUE-YNAT (뉴스 7분류) | `Linear(H, 7)` | softmax | `CrossEntropyLoss` |
| 17 | klue/bert-base | 같음 | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | sigmoid (per-label) | `BCEWithLogitsLoss` (per-label) |
| **18 ← 여기** | klue/bert-base + **보조 헤드** | 같음 | KLUE-YNAT 합성 multi-label + **활성 개수** | **메인(7) + 보조(1)** | 메인 sigmoid + 보조 없음 | **`BCE per-label + λ·MSE`** |
| 19 (다음 Phase 3) | (없음) — 토크나이저 학습 | **직접 학습한 워드레벨** | (코퍼스) | — | — | — |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 17)

| 축 | Ch 17 (한국어 multi-label) | Ch 18 (한국어 multi-label + auxiliary) |
|---|---|---|
| Task (메인) | 7라벨 multi-label | (그대로) |
| `num_labels` | 7 | (그대로) |
| 메인 활성화 / loss | per-label sigmoid / BCE | (그대로) |
| **보조 head** | 없음 | **새로 추가**: `Linear(H, 1)` linear regressor |
| **보조 라벨** | 없음 | **새로 추가**: `n_active` (활성 카테고리 *개수*, 합성 과정에서 얻음 — 1 또는 2) |
| **보조 loss** | — | **`MSELoss`** (Ch 9·14 와 같은 식) |
| **결합 loss** | `outputs.loss` 자동 (BCE) | **`L_main + λ·L_aux`** 직접 계산 |
| 모델 구조 | `AutoModelForSequenceClassification` (자동 매핑) | **`AutoModel` 본체 + 메인 헤드 + 보조 헤드 직접 부착** (자동 매핑 X) |
| `Trainer.compute_loss` | 자동 (오버라이드 X) | **오버라이드 필수** |
| 데이터 콜레이터 | `DataCollatorWithPadding` 자동 | **커스텀** — `n_active` 도 같이 batching |
| 학습 hyperparams | (epoch=2, lr=2e-5, bs=16, fp16) | (그대로) |

> **변하는 축 — Loss 축 끝 (한국어)**: 메인 task 와 모델 본체는 *완전히 동일*, *Loss 에 보조 항이 가중합으로 추가* 됩니다. Ch 14 (영어) 와 같은 패턴을 한국어 데이터·모델로 재현. Phase 2 (한국어) 의 마지막 단계 — 다음 Phase 3 에선 토크나이저 자체를 학습.

### 왜 *활성 라벨 개수* 가 좋은 보조 task 인가

합성 multi-label 데이터에서 `n_active` 는 두 가지 좋은 보조 task 조건을 만족합니다:

1. **공짜로 얻어짐** — `make_multilabel` 이 두 헤드라인을 결합할 때 활성 카테고리 개수가 자연히 부산물로 생김 (라벨링 비용 0). Ch 14 의 *별점* 이 데이터에 *원래* 있던 것과 비슷한 자연스러움.
2. **메인과 강한 상관** — multi-label 정답 벡터 $\mathbf{y}$ 와 그 *합* $\sum_k y_k = n_\text{active}$ 는 직접적으로 연결된 신호. 모델이 "이 헤드라인이 *몇 개* 카테고리에 걸치는가" 를 잘 추정하면 "*어느* 카테고리인가" 도 더 잘 맞힐 가능성 — 두 task 가 같은 BERT 표현을 공유.

> **Ch 14 의 별점 vs Ch 18 의 활성 개수** — Ch 14 별점은 메인 (항목) 과 *부분* 상관 (긍정 리뷰가 음식 라벨일 가능성 높음 정도). Ch 18 활성 개수는 메인 (multi-label 벡터) 의 *직접 함수* (합). 따라서 Ch 18 이 *원리적으로* auxiliary 효과가 더 명확하게 나타날 수 있는 셋업.

## Loss 노트 — Combined loss `L = L_main + λ · L_aux`

$$L = \underbrace{\frac{1}{N \cdot K}\sum_{i,k}\text{BCE}(z_{i,k}^\text{main}, y_{i,k}^\text{main})}_{L_\text{main}: \text{카테고리 BCE per-label}} + \lambda \cdot \underbrace{\frac{1}{N}\sum_{i}(z_{i}^\text{aux} - n_{i}^\text{active})^2}_{L_\text{aux}: \text{활성 개수 MSE}}$$

- $z^\text{main} \in \mathbb{R}^7$ — 카테고리 logit 7개, sigmoid 후 BCE per-label.
- $z^\text{aux} \in \mathbb{R}$ — 활성 개수 회귀 logit (활성화 없음, 직접 MSE).
- $n^\text{active} \in \{1, 2\}$ — 합성 시 두 헤드라인이 같은 카테고리면 1, 다르면 2 (이론상 1 또는 2 만 등장).
- $\lambda$ — 보조 loss 가중치. 본문 기본값은 스윕에서 확인한 sweet spot 인 **0.05**입니다. 보조 MSE 가 메인 BCE 보다 *크기 자체가 커서* λ 를 작게 잡아 균형을 맞춥니다.

**λ 스케일 감 잡기 — 보조 MSE 의 *크기* 부터**

활성 개수 정답은 1 또는 2 의 *정수*. 학습 초기 보조 헤드 예측이 평균 1.5 근처면 MSE 는 약 $0.25$, 무작위 예측이면 $1-4$까지 커질 수 있습니다. 메인 BCE 는 K=7 평균이라 학습 초반에도 $0.3-0.7$ 수준입니다. 따라서 *λ=1* 은 과하고, 부록 스윕에서는 **λ=0.05** 가 메인 F1 을 가장 끌어올렸습니다.

| λ | $L_\text{main}$ (가정 0.45) | $L_\text{aux}$ (가정 0.25) | $L$ | 보조 비중 |
|---|---|---|---|---|
| 0.0 | 0.45 | (무시) | 0.45 | 0% (= Ch 17) |
| 0.05 | 0.45 | 0.25 | 0.4625 | 2.7% ← **본문 기본, sweet spot** |
| 1.0 | 0.45 | 0.25 | 0.70 | 36% (보조가 메인의 절반 이상 영향) |
| 5.0 | 0.45 | 0.25 | 1.70 | 74% (보조 우세 — 메인 신호 묻힘) |

이번 챕터에선 **λ=0.05** 로 학습하고 λ=0 baseline 과 비교합니다. 부록 `18_ko_auxiliary_lambda_sweep` 의 공정 seed 스윕에서 λ=0.05 가 micro/macro-F1 을 가장 끌어올리는 sweet spot 으로 확인됐기 때문입니다.

> **Auxiliary 가 *새 task* 가 아니라 *loss 보조항* 인 이유** — `n_active` 회귀가 *추론 시 결과* 로 필요한 게 아닙니다. 운영에선 메인 multi-label 만 쓰고 보조 헤드는 *호출조차 하지 않음*. 학습 *과정* 에서 BERT 본체를 더 일반적인 표상으로 끌고 가려는 *정규화* 신호일 뿐입니다. 그래서 이 변화는 *task 축* 이 아니라 *loss 축* 에 보조 항을 더하는 변화로 분류됩니다 — 보조 헤드는 task 를 신설하는 게 아니라 손실에 항을 추가할 뿐입니다.

## 토크나이저 노트

Ch 17 과 *완전히 동일* — `klue/bert-base` 한국어 WordPiece, `max_length=128`, 두 헤드라인을 `" [SEP] "` 로 이어붙인 결합 텍스트. **토크나이저는 라벨에 무관** 하므로 보조 라벨 (`n_active`) 추가로 인한 변화 없음.

> **다음 챕터 (Ch 19, Phase 3 시작)**: 토크나이저를 *직접 학습*. 사전학습 모델에 의존하지 않고 코퍼스에서 워드레벨 어휘를 직접 만들어 봅니다 — Ch 1 부터 따라온 "토크나이저 시각" 의 클라이맥스. 본 챕터까지는 *기성품 KLUE WordPiece 를 그대로 썼지만* 다음 챕터부터는 *어휘 구성 자체가 학습 대상*.

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

## 데이터 — KLUE-YNAT 합성 multi-label + 활성 개수 보조 라벨

Ch 17 의 `make_multilabel` 을 *그대로* 가져옵니다. 함수 안에서 이미 `n_active` (활성 라벨 개수) 컬럼이 만들어지고 있어 보조 라벨로 그대로 사용 가능 — *합성 과정의 자연스러운 부산물*.

| 라벨 | 카테고리 |
|---|---|
| 0 | IT과학 |
| 1 | 경제 |
| 2 | 사회 |
| 3 | 생활문화 |
| 4 | 세계 |
| 5 | 스포츠 |
| 6 | 정치 |

> **합성 규칙 (Ch 17 동일)** — 두 헤드라인 A, B 를 `" [SEP] "` 로 연결, multi-hot 라벨에서 $c_A, c_B$ 위치를 1 로. 우연히 $c_A = c_B$ 면 활성 개수 1, 다르면 2. 7카테고리에서 무작위 결합이므로 $P(c_A = c_B) = 1/7$ → 평균 `n_active` 약 $2 \cdot 6/7 + 1 \cdot 1/7 \approx 1.86$.

### 1-1. 합성 함수 — Ch 17 의 `make_multilabel` 재사용

`n_active` (활성 개수) 컬럼이 합성 시 만들어집니다. Ch 18 의 보조 task 정답이 바로 이 값.

## 토큰화 — 메인 multi-hot + 보조 `n_active` 같이 부착

Ch 14 의 `aux_labels` 패턴 그대로 — `tokenize_fn` 이 두 라벨을 모두 attach. 메인은 `labels` (multi-hot 7차원 float), 보조는 `n_active` (float scalar).

## 커스텀 Data Collator — `n_active` 도 batch 에 같이 담기

Ch 14 의 `AuxCollator` 패턴 그대로. 기본 `DataCollatorWithPadding` 은 `input_ids`/`attention_mask`/`labels` 만 알고 있어 *추가 라벨* 은 통과시키지 못합니다. wrapper 로 `n_active` 를 텐서로 만들어 batch 에 추가.

## 모델 — `AutoModel` 본체 + 메인 헤드 + 보조 헤드 직접 부착

Ch 14 는 `AutoModelForSequenceClassification` 의 자동 매핑을 *그대로* 쓰면서 `model.aux_head = nn.Linear(...)` 한 줄로 보조 헤드를 attach 했습니다. Ch 18 도 같은 패턴이 가능하지만, *두 헤드를 명시적으로 한 클래스에서 관리* 하는 패턴이 multi-task 의 정통 — 이번엔 **`nn.Module` 을 직접 정의** 해 두 헤드를 같은 곳에 둡니다.

두 패턴 모두 결과는 같습니다. 명시 정의가 *디버깅·확장* (e.g. 헤드를 더 추가하거나 layer-wise lr 차등) 에 유리.

**보조 헤드는 약 769개 파라미터** — 768→1 Linear 의 weight + bias. 전체 약 110M 의 *0.0007%*. 이 미세한 추가 자유도만으로 multi-task 학습이 동작합니다 (Ch 14 와 동일한 직관).

## 커스텀 Trainer — `compute_loss` 오버라이드

핵심 로직 한 줄:

```python
loss = l_main + λ · l_aux       # l_main: BCE per-label, l_aux: MSE on n_active
```

Ch 14 와의 차이 — Ch 14 는 `outputs.loss` (자동 매핑 메인 BCE) 를 그대로 받고 보조만 직접 계산. Ch 18 은 모델 forward 가 *이미* combined loss 를 계산해 반환하므로 `compute_loss` 는 forward 결과를 그대로 돌려주기만 하면 됩니다. λ 만 trainer 에서 model forward 로 넘김.

**평가용 metric 함수** — 메인 (Ch 17 과 동일) 만 자동 계산. 보조 metric (RMSE, R², Pearson r) 은 별도 forward 로 `count_pred` 를 추출해 측정 (eval 후 별도 단계).

## 학습 — λ=0.05 (sweet spot, 보조 ON)

Ch 17 과 동일한 hyperparams. `AuxTrainer` + `lambda_aux=0.05`. 이 값은 부록 `18_ko_auxiliary_lambda_sweep` 의 λ 스윕에서 **메인 F1 을 가장 끌어올린 지점** 입니다 (λ≥0.2 부터는 §10 처럼 메인이 무너집니다).

**중요: `remove_unused_columns=False`** — Trainer 는 기본으로 *model.forward 시그니처에 없는 컬럼* 을 제거합니다. `n_active` 는 KoBertMultiTask.forward 에 있어 자동 인식되지만, 모델 클래스를 바꿔 끼울 때 위험할 수 있어 명시적으로 끕니다 (Ch 14 와 같은 보호 패턴).

## 평가 — 메인 task + 보조 task

메인 metric 은 자동 (`compute_metrics_main`). 보조 metric (RMSE, R², Pearson r) 은 별도 forward 로 `count_pred` 를 추출해 측정.

## 클라이맥스 — *λ=0 baseline* 학습 (= Ch 17 재현)

같은 코드를 `lambda_aux=0.0` 으로 한 번 더 돌립니다. 보조 loss 의 gradient 가 0 이 되어 메인 task 만 학습되는 상태 = **Ch 17 과 정확히 동일한 학습 결과** (보조 헤드는 학습되긴 하지만 메인 학습엔 영향 없음).

> 의도적으로 *Ch 17 노트북을 따로 돌리지 않고* 이 셀에서 baseline 을 다시 만듭니다 — 비교가 *같은 노트북·같은 환경* 안에서 self-contained 하도록 (Ch 14 와 같은 패턴).

### 8-1. 메인 metric 비교 — λ=0 baseline vs λ=0.05 aux

**해석 가이드**

- `delta` > 0 — 보조 loss 가 메인 task 에 *도움* 이 됨.
- `delta` < 0 — 보조 loss 가 메인 task 를 *방해* 함 (λ 가 너무 크거나 보조 task 상관이 약함).
- `delta` ≈ 0 — 별 영향 없음.

`n_active` 는 메인 multi-label 벡터의 *합* 이라 양의 상관이 매우 강합니다 — Ch 14 의 별점보다 상관이 직접적이므로 *작은 양의 delta* 가 자연스러운 결과. 단 보조 task 가 *너무 쉬워서* (1 vs 2 이항 회귀) 추가 정보량이 적을 수 있다는 점도 고려.

### 8-2. 카테고리별 F1 비교 — 어느 카테고리가 보조 loss 로 가장 도움받았나

**해석**

- **활성률 높은 카테고리** (스포츠/세계/사회 등): baseline 자체 F1 이 높음. delta 는 작거나 0 — 이미 신호가 충분.
- **활성률 낮은·헷갈리는 카테고리** (정치/IT과학 등): baseline F1 이 낮음. 보조 신호의 *정규화 효과* 가 상대적으로 도움이 될 가능성 — 그래도 5K 샘플·2 epoch quick 모드에선 delta 가 노이즈 영역 (±0.01) 안에 머무를 수 있음.
- **모든 카테고리 delta 가 ±0.005 이내** → quick 모드 표본의 노이즈 영역. 학습량 (epoch·데이터) 을 늘려야 보조 효과가 통계적으로 분리 가능.

### 8-3. 보조 task 자체는 얼마나 잘 학습됐나

`n_active` 는 1 또는 2 정수만 나오므로 *binary 같은 회귀* 입니다. RMSE 가 0 에 가까우면 모델이 두 경우를 잘 구분, 0.5 근처면 무작위 추측 (분산이 0.25 인 1-vs-2 분포).

**해석**

- 두 violin (n_active=1, n_active=2) 의 *중심* 이 점선 가이드 (1.0 / 2.0) 에 잘 맞으면 보조 헤드가 활성 개수를 잘 학습한 것.
- *분포 폭* — 모델이 두 경우를 *자신 있게* 구분하면 violin 이 좁고 점선 가이드에 집중. 폭이 넓고 두 violin 이 0.5 근처에서 겹치면 학습 부족.
- 1.5 근처에 한 데가 몰려 있으면 *상수 평균 예측* 으로 회귀 — 보조 신호가 메인 표상에 *반영되지 못한* 상태. 이 경우 λ 를 더 키우거나 데이터·epoch 를 늘려야 함.

## 결과 해석 — sweet spot 에서는 약한 보조도 메인을 (살짝) 돕는다

§8 비교에서 **λ=0.05 보조가 λ=0 baseline 을 micro·macro-F1 모두에서 앞섰습니다** (각 +0.003, +0.004). `n_active` 라는 *약한* 보조 신호도 작은 λ 에서는 공유 KLUE-BERT 본체에 가벼운 정규화로 작용해 메인 분류를 살짝 끌어올립니다.

다만 그 효과는 **Ch 14(영어, 별점 보조)보다 작습니다.** 부록 `18_ko_auxiliary_lambda_sweep` 의 λ 곡선과 Ch 14 를 나란히 두면:

| | 보조 task | 보조 R² | sweet spot Δ(micro) |
|---|---|---|---|
| Ch 14 | 별점 회귀 | 0.43 (강함) | +0.007 |
| **Ch 18** | **n_active 회귀** | **0.065 (약함)** | **+0.003** |

두 챕터의 sweet spot 은 똑같이 **λ=0.05** 인데 도움의 *크기* 가 다릅니다. 차이는 λ 가 아니라 **보조 신호의 정보량** 입니다.

### 왜 `n_active` 는 약한가

`n_active` 는 합성 규칙상 거의 항상 2 입니다 (train 분포 {1: 732, 2: 4268}). 분산이 작아 *예측할 게 별로 없어서*, 보조 헤드가 λ 를 0.5 까지 키워도 R² 가 0.08 에 머뭅니다. 보조가 입력을 깊이 들여다볼 동기가 약하니 공유 표현에 실어주는 추가 정보도 적습니다. 반면 Ch 14 의 별점은 *사용자가 직접 매긴* 입력 의존도 높은 신호라 R² 0.43 으로 잘 학습되고, 그만큼 메인에도 더 보탬이 됐습니다.

### λ 를 키우면

λ≥0.2 부터는 약한 보조가 오히려 메인을 깎습니다 (λ=0.5 에서 micro 0.80). 약한 보조일수록 *도움이 되는 작은 λ 구간이 더 좁습니다* — 본편이 λ=0.05 를 쓰는 이유입니다.

> *이 챕터의 메시지* — auxiliary loss 는 공짜 만병통치약이 아니라, **(1) λ 를 작게 잡고 (2) 입력 의존도 높은 보조 신호를 골라야** 메인을 돕습니다. `n_active` 는 데이터 합성의 자연 부산물이라 손쉽지만 약하고, 그래도 sweet spot 에서는 +가 납니다. 더 큰 도움을 원하면 헤드라인 길이·발행 메타데이터처럼 *입력 의존도 큰* 보조로 바꾸는 게 다음 수입니다. 전체 λ 곡선은 [18-4 부록 — λ 스윕으로 약한 보조 신호의 sweet spot 찾기](18-ko_auxiliary-lambda_sweep.md)에서 확인할 수 있습니다.

## 이 장의 구성

[[SubPages]]
