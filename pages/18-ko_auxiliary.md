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
- $\lambda$ — 보조 loss 가중치. 본문 기본값 **0.1** (보조 MSE 가 메인 BCE 보다 *크기 자체가 커서* — 1-4 vs 0.3-0.6 — λ 를 작게 잡아 균형).

**λ 스케일 감 잡기 — 보조 MSE 의 *크기* 부터**

활성 개수 정답은 1 또는 2 의 *정수*. 학습 초기 보조 헤드 예측이 평균 1.5 근처면 MSE 는 약 $0.25$, 무작위 예측이면 $1-4$. 메인 BCE 는 K=7 평균이라 학습 초반에도 $0.3-0.7$ 수준. *λ=1* 이면 보조가 메인보다 크게 잡힐 수 있어 **λ=0.1** 이 권장 기본값.

| λ | $L_\text{main}$ (가정 0.45) | $L_\text{aux}$ (가정 0.25) | $L$ | 보조 비중 |
|---|---|---|---|---|
| 0.0 | 0.45 | (무시) | 0.45 | 0% (= Ch 17) |
| 0.1 | 0.45 | 0.25 | 0.475 | 5% ← **본문 기본** |
| 1.0 | 0.45 | 0.25 | 0.70 | 36% (보조가 메인의 절반 이상 영향) |
| 5.0 | 0.45 | 0.25 | 1.70 | 74% (보조 우세 — 메인 신호 묻힘) |

이번 챕터에선 **λ=0.1** 로 학습하고 λ=0 baseline 과 비교, §10 의 변형 섹션에서 λ ∈ {0.0, 0.1, 1.0} 스윕으로 효과 분포를 봅니다.

> **Auxiliary 가 *새 task* 가 아니라 *loss 보조항* 인 이유** — `n_active` 회귀가 *추론 시 결과* 로 필요한 게 아닙니다. 운영에선 메인 multi-label 만 쓰고 보조 헤드는 *호출조차 하지 않음*. 학습 *과정* 에서 BERT 본체를 더 일반적인 표상으로 끌고 가려는 *정규화* 신호일 뿐입니다. 그래서 이 변화는 *task 축* 이 아니라 *loss 축* 에 보조 항을 더하는 변화로 분류됩니다 — 보조 헤드는 task 를 신설하는 게 아니라 손실에 항을 추가할 뿐입니다.

## 토크나이저 노트

Ch 17 과 *완전히 동일* — `klue/bert-base` 한국어 WordPiece, `max_length=128`, 두 헤드라인을 `" [SEP] "` 로 이어붙인 결합 텍스트. **토크나이저는 라벨에 무관** 하므로 보조 라벨 (`n_active`) 추가로 인한 변화 없음.

> **다음 챕터 (Ch 19, Phase 3 시작)**: 토크나이저를 *직접 학습*. 사전학습 모델에 의존하지 않고 코퍼스에서 워드레벨 어휘를 직접 만들어 봅니다 — Ch 1 부터 따라온 "토크나이저 시각" 의 클라이맥스. 본 챕터까지는 *기성품 KLUE WordPiece 를 그대로 썼지만* 다음 챕터부터는 *어휘 구성 자체가 학습 대상*.

## 이 장의 구성

[[SubPages]]
