**목표**: Ch 16 의 한국어 multi-class 셋업을 그대로 두고 **task 만 single-label → multi-label** 로 바꿉니다. 모델·토크나이저·hyperparams 가 *완전히 동일* 하고, 변하는 건 라벨이 *하나* 가 아니라 *여러 개 동시 활성* 될 수 있다는 점과 그에 따른 loss/activation.

KLUE-YNAT 에는 multi-label 정답이 없습니다. 그래서 *서로 다른 두 뉴스 헤드라인을 이어붙여* 두 카테고리가 동시에 활성된 합성 multi-label 샘플을 직접 만듭니다 — 한 헤드라인이 *여러 주제에 걸치는* 상황을 시뮬레이션.

이 챕터는 Ch 13 (영어 multi-label, Yelp 항목 합성) 의 한국어 버전입니다. *합성 방식* 만 다릅니다 — Ch 13 은 키워드 매칭으로 라벨을 붙였고, Ch 17 은 두 single-label 샘플을 *결합* 해 라벨을 union 합니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 13분 (모델 다운로드 캐시 -10s + 2 에폭 학습 -10분 + 평가/시각화)


## 학습 흐름

1. 🚀 **실습**: KLUE-YNAT 헤드라인 두 개를 결합해 multi-label 5,000건 합성 → klue/bert-base 를 `multi_label_classification` 으로 파인튜닝
2. 🔬 **해부**: 카테고리별 sigmoid 확률 분포 (7 패널 KDE) + 카테고리 간 공동 활성 패턴 (co-occurrence heatmap)
3. 🛠️ **변형**: 합성 샘플을 직접 읽어보며 모델이 *두 주제를 모두 잡는지* 확인 + threshold 를 옮기면 결과가 어떻게 바뀌는지


> 📒 **사전 학습 자료**: Ch 13 (영어 multi-label, per-label BCE), Ch 16 (한국어 multi-class, KLUE-YNAT). 이번 챕터는 두 챕터의 *결합* — Ch 16 의 한국어 셋업 + Ch 13 의 multi-label 처리.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 13 | DistilBERT | WordPiece (영어) | Yelp + 항목 키워드 합성 | `Linear(H, 5)` | sigmoid (per-label) | `BCEWithLogitsLoss` (per-label) |
| 15 | klue/bert-base | WordPiece (한국어) | NSMC binary | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| 16 | klue/bert-base | 같음 | KLUE-YNAT (뉴스 7분류) | `Linear(H, 7)` | softmax | `CrossEntropyLoss` |
| **17 ← 여기** | klue/bert-base | 같음 | **KLUE-YNAT 합성 multi-label (두 헤드라인 결합)** | `Linear(H, 7)` | **sigmoid (per-label)** | **`BCEWithLogitsLoss` (per-label)** |
| 18 (다음) | klue/bert-base | 같음 | KLUE-YNAT 합성 multi-label + 라벨 개수 보조 | `Linear(H, 7)` 메인 + `Linear(H, 1)` 보조 | sigmoid + 없음 | `BCE(per-label) + λ·MSE` |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 16)

| 축 | Ch 16 (한국어 multi-class) | Ch 17 (한국어 multi-label) |
|---|---|---|
| **Task** | 7-클래스 *single-label* (서로 배타적) | **7-라벨 *multi-label*** (동시 활성 가능) ← *유일한 변화* |
| `num_labels` | 7 | **7** (그대로!) |
| `problem_type` | `"single_label_classification"` | **`"multi_label_classification"`** ← BCE 자동 매핑 |
| Activation | softmax (합=1 강제) | **per-label sigmoid** (각각 독립 0-1) |
| Loss | `CrossEntropyLoss` | **`BCEWithLogitsLoss`** per-label |
| 라벨 형식 | int 스칼라 (0-6) | **multi-hot float 7차원 `[0, 1, 0, 0, 0, 1, 0]`** |
| 데이터 | KLUE-YNAT 원본 헤드라인 | **두 헤드라인 결합 → 두 카테고리 union** |
| 평가 metric | accuracy + macro F1 + AUC OvR | **hamming loss + micro/macro F1 + per-label F1 + macro AUC** |
| 모델 본체 / 토크나이저 / hyperparams | (모두 동일) | (모두 동일) |

> **변경점 한 가지 원칙** — Phase 2 안에선 *task 차원* (single-label → multi-label) 만 바뀝니다. 한국어 셋업·hyperparams 는 Ch 16 과 *완전히 같음*. 모델 아키텍처조차 `Linear(H, 7)` 그대로 — `problem_type` 한 줄과 라벨 *형식* 만 바뀝니다.

### 같은 7차원 출력을 두 가지로 해석

Ch 16 과 Ch 17 의 모델은 둘 다 `Linear(768, 7)` 헤드를 가집니다. 차이는 *그 7개 숫자를 어떻게 읽는가*:

- **Ch 16 (softmax)**: 7개를 *한꺼번에* 정규화해 합=1 로 만든 뒤 *가장 큰 하나* 를 고름. "이 헤드라인은 7개 카테고리 중 *정확히 하나*".
- **Ch 17 (per-label sigmoid)**: 7개 각각을 *독립적으로* 0-1 확률로 변환. "이 헤드라인에 각 카테고리가 *각자* 활성됐는가?" — 여러 개가 동시에 1 일 수 있음.

## Loss 함수의 변화 — `CrossEntropyLoss` → `BCEWithLogitsLoss` per-label

Ch 16 의 multi-class CE 는 *정답 클래스 하나* 의 로그확률만 봤습니다:

$$L_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\log \hat p_{i, y_i}$$

Ch 17 은 K=7 개 라벨 각각에 *독립적* BCE 를 적용한 뒤 평균합니다 (Ch 13 의 식 그대로, 한국어 맥락):

$$L_{\text{BCE}} = -\frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K}\left[ y_{i,k} \log \sigma(z_{i,k}) + (1-y_{i,k}) \log(1-\sigma(z_{i,k})) \right]$$

각 $z_{i,k}$ 는 *독립 logit* — 카테고리 k 가 *얼마나 활성될지* 의 점수, 다른 카테고리와 무관. PyTorch `BCEWithLogitsLoss` 가 7개 위치를 한 번에 처리하지만 수식적으론 7개의 binary BCE 평균입니다.

**숫자로 감 잡기 (K=7, 정답 multi-hot $\mathbf{y} = [0, 1, 0, 0, 0, 1, 0]$ — 경제+스포츠 동시 활성)** — logit 별 손실 분해:

| 라벨 | 카테고리 | $y_k$ | logit $z_k$ | $\sigma(z_k)$ | 손실 $-\log(\cdot)$ |
|---|---|---|---|---|---|
| 1 | 경제 | 1 | 3.0 | 0.953 | 0.048 |
| 5 | 스포츠 | 1 | 0.5 | 0.622 | 0.474 |
| 0 | IT과학 | 0 | -2.0 | 0.119 | 0.127 |
| 2 | 사회 | 0 | 1.5 | 0.818 | 1.704 ← 자신 있게 틀림 |
| 나머지 3개 | (음성) | 0 | -3.0 | 0.047 | 각 0.048 |

평균 loss = $(0.048 + 0.474 + 0.127 + 1.704 + 0.048 \times 3) / 7 \approx 0.387$.

**핵심 직관 — 라벨 사이엔 직접 신호가 없음**: BCE per-label 은 카테고리 k 의 logit 이 카테고리 j 의 정답에서 *직접* 학습 신호를 받지 않습니다. 모델이 카테고리 간 상관을 학습하는 건 *공유 BERT 본체* (모든 라벨이 같은 768-dim CLS 표현에서 옴) 덕분이지 loss 자체엔 라벨 간 결합 항이 없습니다.

**코드 한 줄 변화** — Ch 16 → Ch 17:

```python
# Ch 16: int 스칼라 라벨 + single_label_classification
out["labels"] = [int(l) for l in batch["label"]]
problem_type = "single_label_classification"   # → CrossEntropyLoss

# Ch 17: multi-hot 7차원 float 라벨 + multi_label_classification
out["labels"] = [list(map(float, mh)) for mh in batch["multi_hot"]]
problem_type = "multi_label_classification"     # → BCEWithLogitsLoss per-label
```

### 왜 multi-label 은 softmax 로 풀 수 없는가

softmax 는 출력의 *합 = 1* 을 강제합니다 ($\sum_k \mathrm{softmax}(z)_k = 1$). 이는 *서로 배타적* 클래스 (Ch 16) 에 자연스럽지만 multi-label 과 충돌합니다.

합성 샘플이 "경제 헤드라인 + 스포츠 헤드라인" 이라 정답이 경제=1, 스포츠=1 일 때:
- **softmax 모델은 표현 불가**: P(경제)=0.6 이면 나머지 6 라벨 합이 0.4 로 강제 → '스포츠=0.55 동시 활성' 이 *수학적으로 불가능*.
- **per-label sigmoid 모델은 표현 가능**: 각 라벨이 독립이라 P(경제)=0.9 와 P(스포츠)=0.85 가 동시에 자연스러움.

즉 task 가 *진짜 multi-label* 이라면 loss/activation 선택이 강제됩니다 (Ch 4·13 에서 본 동일한 논리).

## 토크나이저 노트

Ch 16 과 *완전히 동일* — `klue/bert-base` 한국어 WordPiece. 토크나이저는 라벨 *형식* (int 스칼라든 multi-hot 벡터든) 에 무관하므로 변화 없음.

> **Phase 2 안에서는 토크나이저 고정** — Ch 15·16·17·18 모두 같은 한국어 WordPiece. Phase 3 (Ch 19-20) 에서 비로소 *직접 학습한 워드레벨 토크나이저* 가 등장.

### 결합 헤드라인 토큰화 예시

Ch 16 은 헤드라인 *한 줄* (-25-30 토큰) 이었지만, Ch 17 은 두 헤드라인을 `[SEP]` 로 이어붙여 *길이가 약 2배* (-50-60 토큰) 가 됩니다. `max_length=128` 안에 충분히 들어갑니다.

두 문장을 잇는 `[SEP]` 토큰은 BERT 가 *문장 경계* 를 인식하는 특수 토큰입니다 — NSP (Next Sentence Prediction) 사전학습에서 쓰던 그 토큰. 결합 헤드라인에선 두 주제의 경계 역할을 합니다.

> **다음 챕터 (Ch 18)**: 토크나이저 그대로. 변하는 건 *모델에 보조 헤드* (활성 라벨 *개수* 회귀) 가 추가되고 *loss 에 보조 항* 이 가중합으로 더해지는 점.

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

## 데이터 — KLUE-YNAT 결합으로 multi-label 합성

**KLUE-YNAT** 은 single-label 데이터 (헤드라인 한 줄 → 카테고리 하나) 라 multi-label 정답이 없습니다. Ch 13 에서 Yelp 에 항목 키워드를 합성했듯, 여기선 *서로 다른 두 헤드라인을 결합* 해 두 카테고리가 동시에 활성된 샘플을 만듭니다.

| 라벨 | 카테고리 |
|---|---|
| 0 | IT과학 |
| 1 | 경제 |
| 2 | 사회 |
| 3 | 생활문화 |
| 4 | 세계 |
| 5 | 스포츠 |
| 6 | 정치 |

> **합성 방식**: 샘플 A (카테고리 $c_A$) 와 샘플 B (카테고리 $c_B$) 를 뽑아 (1) 텍스트를 `" [SEP] "` 로 이어붙이고 (2) multi-hot 라벨에서 $c_A, c_B$ 두 위치를 1 로. 우연히 $c_A = c_B$ 면 활성 라벨은 1개뿐 (자연스러운 single-label 케이스도 일부 섞임).

### 1-1. 두 헤드라인을 결합해 multi-label 샘플 합성

`make_multilabel` 이 single-label split 을 받아 *짝* 을 지어 합성 데이터셋을 만듭니다. seed 를 고정해 train/eval 이 재현 가능하게.

## 토큰화 — Ch 16 패턴, 라벨 형식만 multi-hot

**Ch 16 과의 한 줄 차이**: `out["labels"] = [int(l) for l in batch["label"]]` → `out["labels"] = [list(map(float, mh)) for mh in batch["multi_hot"]]`. 라벨이 *int 스칼라* 가 아니라 *길이 7 multi-hot float 벡터*. 이 형식 + `problem_type="multi_label_classification"` 두 가지가 BCE per-label 자동 매핑의 트리거입니다.

## 모델 로드 — `num_labels=7` 그대로, `problem_type` 만 전환

Ch 16 과 *모델 아키텍처는 완전히 동일* (`Linear(H, 7)` 분류 헤드). 변하는 한 가지 — `problem_type="multi_label_classification"` — 가 자동 매핑되는 loss 를 BCE per-label 로 바꿉니다.

**Ch 16 과 파라미터 수가 *완전히 동일*** — 둘 다 `Linear(768, 7)` 헤드. 차이는 `problem_type` 한 줄뿐입니다. 같은 모델이 *어떻게 해석되고 어떤 loss 로 학습되는가* 만 바뀝니다. 이게 Ch 16 ↔ Ch 17 변경이 "한 가지 축" 인 이유 — *task 의 의미* 만 single-label → multi-label 로 옮기고 나머지는 전부 고정.

## 학습 — Ch 16 과 동일한 hyperparams

Ch 16 과 *완전히 같은* learning rate, batch size, epoch 수, seed. 평가 metric 만 multi-label 용으로 새로 짭니다 (Ch 13 의 패턴 그대로).

## 평가 — 카테고리별 sigmoid 확률 + 공동 활성 패턴

Ch 16 의 평가가 *7개 클래스 중 하나 고르기* 였다면, Ch 17 은 *7개 카테고리 각각을 독립적으로 0/1 판정* 합니다. Ch 13 의 multi-label 평가 패턴을 한국어 환경에서 재현.

### 5-1. 메인 그림 — 카테고리별 sigmoid 확률 KDE (7 패널)

Ch 16 에선 *top-1 확률 하나* 만 봤지만, multi-label 에선 *각 카테고리* 가 독립이라 7개 확률 분포를 *각각* 그립니다. 카테고리마다 학습 난이도가 *다를 수* 있다는 multi-label 의 본질이 시각적으로 드러납니다.

**해석**

- **잘 학습된 카테고리** (예: 스포츠): label=0 곡선은 0 근처, label=1 곡선은 1 근처에 있고 둘이 거의 만나지 않음. *분리가 깨끗*.
- **헷갈리는 카테고리** (예: 사회 ↔ 생활문화 ↔ 정치): 두 곡선이 0.5 근처에서 크게 겹침. 결합 헤드라인 안에서 두 주제 신호가 *섞이는* 카테고리.
- **결합의 부작용** — 한 샘플에 *두 헤드라인* 이 들어가니 모델이 "둘 중 어느 쪽 신호가 어느 라벨인지" 분리해야 합니다. 이게 단일 헤드라인 (Ch 16) 보다 어려운 점이고, multi-label task 의 자연스러운 난이도.

### 5-2. 보조 그림 — 카테고리 간 공동 활성 패턴

Multi-label 의 핵심 질문: *어떤 카테고리 쌍이 같이 등장하는가?* 합성 방식이 *무작위 결합* 이라 true co-occurrence 는 거의 균등에 가까워야 하고, 모델 예측이 그 패턴을 따라가는지 확인합니다.

`true co-occurrence` (실제 합성 라벨의 동시 등장) 와 `predicted co-occurrence` (모델 예측의 동시 등장) 를 나란히 그립니다.

**해석**

- **대각선 = 1.0** — 자기 자신과는 항상 같이 등장 (정의상).
- **off-diagonal cell M[i, j]** = "카테고리 i 가 활성된 샘플 중 카테고리 j 도 활성된 비율".
- **합성이 무작위 결합** 이라 true 행렬의 off-diagonal 은 *대략 균등* (각 카테고리의 전체 활성률에 비례) — 특정 쌍이 *유난히 높지 않음*. 실제 사람-annotated 데이터라면 "정치+경제" 처럼 *자연스러운 상관* 이 드러나겠지만, 여기선 합성 방식 때문에 그런 구조가 약합니다.
- **predicted cell 이 true 보다 일관되게 높으면** → 모델이 라벨을 *너무 많이* 활성하는 경향 (over-prediction). threshold 를 0.5 보다 높게 두면 calibration 개선.

## 이 장의 구성

- [17-1. 실습](17-ko_multilabel-practice.md)
- [17-2. 변형 — 합성 샘플 직접 읽기 + threshold 옮겨보기](17-ko_multilabel-variation.md)
- [17-3. 정리와 FAQ](17-ko_multilabel-wrapup.md)
