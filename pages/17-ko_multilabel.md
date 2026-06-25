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

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

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

> **Phase 2 안에서는 토크나이저 고정** — Ch 15·16·17·18 모두 같은 한국어 WordPiece. Phase 3 (Ch 19-23) 에서 비로소 *직접 학습한 워드레벨 토크나이저* 가 등장.

### 결합 헤드라인 토큰화 예시

Ch 16 은 헤드라인 *한 줄* (평균 약 16 토큰) 이었지만, Ch 17 은 두 헤드라인을 `[SEP]` 로 이어붙여 *길이가 약 2배* (평균 약 30 토큰, 최대 41) 가 됩니다. `max_length=128` 안에 충분히 들어갑니다.

두 문장을 잇는 `[SEP]` 토큰은 BERT 가 *문장 경계* 를 인식하는 특수 토큰입니다 — NSP (Next Sentence Prediction) 사전학습에서 쓰던 그 토큰. 결합 헤드라인에선 두 주제의 경계 역할을 합니다.

> **다음 챕터 (Ch 18)**: 토크나이저 그대로. 변하는 건 *모델에 보조 헤드* (활성 라벨 *개수* 회귀) 가 추가되고 *loss 에 보조 항* 이 가중합으로 더해지는 점.

## 이 장의 구성

[[SubPages]]
