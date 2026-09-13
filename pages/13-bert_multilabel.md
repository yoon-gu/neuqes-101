**목표**: Ch 12(BERT 5클래스 분류) 셋업과 출력 헤드 크기를 *완전히 동일* 하게 둡니다 (`num_labels=5` 그대로). 변하는 건 **task의 의미** 입니다 — 5개 라벨이 *서로 배타적인 클래스* 가 아니라 *각각 독립적으로 활성될 수 있는 태그* 입니다 ("food" 와 "service" 가 동시에 1).

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 5분 (BERT 학습 약 1분 + sklearn 비교 약 30초 + 평가/시각화)


## 학습 흐름

1. 🚀 **실습**: Ch 6에서 만들었던 *항목 키워드 합성 라벨* (food/service/price/ambiance/location)을 그대로 BERT로 학습. `num_labels=5` + `problem_type="multi_label_classification"` 로 BCE per-label 자동 매핑.
2. 🔬 **해부**: 라벨별 sigmoid 확률 분포 (5 패널 KDE) + 라벨 간 공동 활성 패턴 (correlation heatmap).
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 Ch 6와 *같은 계열* 의 sklearn `OneVsRestClassifier(LogisticRegression)` baseline 을 다시 학습해 라벨별 metric 비교.


> 📒 **사전 학습 자료**: Ch 6 (sklearn multi-label, OvR), Ch 10 (BERT `num_labels=1` + `multi_label_classification` 트릭 — Ch 13은 그 트릭을 K=5 로 확장한 형태), Ch 12 (BERT multi-class). 이번 챕터는 self-contained.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 6 | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 항목 키워드 합성 | (5차원) | sigmoid (각각) | `BCEWithLogitsLoss` per-label |
| 10 | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp 이진화 | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| 12 | DistilBERT 파인튜닝 | 같음 | Yelp 5클래스 | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |
| **13 ← 여기** | DistilBERT 파인튜닝 | 같음 | **Yelp + 항목 키워드 합성** | **`Linear(H, 5)`** | **sigmoid (per-label)** | **`BCEWithLogitsLoss` (per-label)** |
| 14 (다음) | DistilBERT 파인튜닝 | 같음 | Yelp + 항목 + 별점 보조 | `Linear(H, 5)` 메인 + `Linear(H, 1)` 보조 | sigmoid + 없음 | `BCE(per-label) + λ·MSE` |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 12)

| 축 | Ch 12 (multi-class) | Ch 13 (multi-label) |
|---|---|---|
| **Task** | 5클래스 *single-label* (서로 배타적) | **5라벨 *multi-label*** (동시 활성 가능) ← 본질적 변화 |
| `num_labels` | 5 | **5** (그대로!) |
| `problem_type` | `"single_label_classification"` | **`"multi_label_classification"`** ← BCE 자동 매핑 |
| Activation | softmax (합=1 강제) | **per-label sigmoid** (각각 독립 0-1) |
| Loss | `CrossEntropyLoss` | **`BCEWithLogitsLoss`** per-label |
| 라벨 형식 | int 스칼라 (0-4) | **multi-hot float `[1, 0, 1, 0, 1]`** |
| 모델 출력 의미 | logits[k] = 클래스 k의 *상대적* 점수 | logits[k] = 라벨 k의 *독립적* 활성 점수 |
| 평가 metric | accuracy + macro F1 + AUC OvR | per-label precision/recall/F1 + micro/macro F1 + per-label AUC |

> **결정적 인사이트**: 모델 *아키텍처는 동일* — `Linear(H, 5)` 헤드. 변하는 건 *해석* 과 그에 따른 loss/activation입니다. 같은 5차원 출력을 *softmax로 모아서 한 클래스 고르기* 와 *5개 sigmoid로 각자 0/1 결정하기* 두 가지로 다르게 쓰는 것.

### 왜 multi-label은 softmax로 풀 수 없는가

softmax는 출력의 *합 = 1* 을 강제합니다 ($\sum_k \mathrm{softmax}(z)_k = 1$). 이는 *서로 배타적* 클래스에 자연스럽지만 multi-label과 충돌합니다.

리뷰가 "food=1 (음식 언급) 그리고 service=1 (서비스 언급)" 일 때:
- **softmax 모델은 표현 불가**: P(food)=0.9 면 나머지 4 라벨 합이 0.1 로 강제 → 'service=0.85 동시 활성' 이 *수학적으로 불가능*.
- **per-label sigmoid 모델은 표현 가능**: 각 라벨이 독립이라 P(food)=0.9 와 P(service)=0.85 가 동시에 자연스러움.

즉 task가 *진짜 multi-label* 이라면 loss/activation 선택이 강제됩니다 (Ch 6 에서 본 동일한 논리, BERT로 옮겨옴).

## Loss 노트 — `BCEWithLogitsLoss` per-label (Ch 6 그대로, BERT 맥락)

K개 라벨 각각에 *독립적* BCE를 적용한 뒤 평균:

$$L = \frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K}\left[ -y_{i,k} \log \sigma(z_{i,k}) - (1-y_{i,k}) \log(1-\sigma(z_{i,k})) \right]$$

각 $z_{i,k}$ 는 *독립 logit* — 라벨 k가 *얼마나 활성될지* 의 점수, 다른 라벨과 무관. PyTorch `BCEWithLogitsLoss` 가 5개 위치를 한 번에 처리하지만 수식적으론 K개의 binary BCE 평균.

**숫자로 감 잡기 (K=5, 정답 multi-hot $\mathbf{y} = [1, 0, 1, 0, 1]$)** — logits 별 손실 분해:

| 라벨 | $y_k$ | logit $z_k$ | $\sigma(z_k)$ | 정답일 때 손실 |
|---|---|---|---|---|
| food | 1 | 3.0 | 0.953 | $-\log 0.953 = 0.048$ |
| service | 0 | -2.0 | 0.119 | $-\log(1-0.119) = 0.127$ |
| price | 1 | 0.5 | 0.622 | $-\log 0.622 = 0.474$ |
| ambiance | 0 | 1.5 | 0.818 | $-\log(1-0.818) = 1.704$ ← 자신 있게 틀림 |
| location | 1 | -0.5 | 0.378 | $-\log 0.378 = 0.974$ |

평균 loss = $(0.048 + 0.127 + 0.474 + 1.704 + 0.974) / 5 \approx 0.665$.

**핵심 직관 — 라벨 사이엔 직접 신호가 없음**: BCE per-label은 라벨 k의 logit이 라벨 j의 정답에서 *직접* 학습 신호를 받지 않습니다. 모델이 라벨 간 상관을 학습하는 건 *공유 BERT 본체* (모든 라벨이 같은 768-dim CLS 표현에서 옴) 덕분이지 loss 자체에는 라벨 간 결합 항이 없습니다. 이 점이 multi-class softmax와 결정적 차이.

## 토크나이저 노트

Ch 12와 동일 — `distilbert-base-uncased` WordPiece, `max_length=128`. 토크나이저는 라벨 *형식* 에 무관하므로 single-label 이든 multi-label 이든 변화 없음.

> **다음 챕터(Ch 14)**: 토크나이저 그대로. 변하는 건 *모델에 보조 헤드* 가 추가되고 *loss에 보조 항* 이 가중합으로 더해지는 점.

## 이 장의 구성

[[SubPages]]
