**목표**: Ch 12(BERT 5클래스 분류) 셋업과 출력 헤드 크기를 *완전히 동일* 하게 둡니다 (`num_labels=5` 그대로). 변하는 건 **task의 의미** 입니다 — 5개 라벨이 *서로 배타적인 클래스* 가 아니라 *각각 독립적으로 활성될 수 있는 태그* 입니다 ("food" 와 "service" 가 동시에 1).

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 12분 (BERT 학습 ~10분 + sklearn 비교 ~30초 + 평가/시각화)


## 학습 흐름

1. 🚀 **실습**: Ch 6에서 만들었던 *항목 키워드 합성 라벨* (food/service/price/ambiance/location)을 그대로 BERT로 학습. `num_labels=5` + `problem_type="multi_label_classification"` 로 BCE per-label 자동 매핑.
2. 🔬 **해부**: 라벨별 sigmoid 확률 분포 (5 패널 KDE) + 라벨 간 공동 활성 패턴 (correlation heatmap).
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 Ch 6의 sklearn `OneVsRestClassifier(LogisticRegression)` baseline 재현 → 라벨별 metric 비교.


> 📒 **사전 학습 자료**: Ch 6 (sklearn multi-label, OvR), Ch 10 (BERT `num_labels=1` + `multi_label_classification` 트릭 — Ch 13은 그 트릭을 K=5 로 확장한 형태), Ch 12 (BERT multi-class). 이번 챕터는 self-contained.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 6 | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 항목 키워드 합성 | (5차원) | sigmoid (각각) | `BCEWithLogitsLoss` per-label |
| 10 | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp 이진화 | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| 12 | DistilBERT 파인튜닝 | 같음 | Yelp 5클래스 | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |
| **13 ← 여기** | DistilBERT 파인튜닝 | 같음 | **Yelp + 항목 키워드 합성** | **`Linear(H, 5)`** | **sigmoid (per-label)** | **`BCEWithLogitsLoss` (per-label)** |
| 14 (다음) | DistilBERT 파인튜닝 | 같음 | Yelp + 항목 + 별점 보조 | `Linear(H, 5)` 메인 + `Linear(H, 1)` 보조 | sigmoid + 없음 | `BCE(per-label) + λ·MSE` |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

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

$$L = \frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K}\left[ y_{i,k} \log \sigma(z_{i,k}) + (1-y_{i,k}) \log(1-\sigma(z_{i,k})) \right]$$

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

**baseline VRAM**:

## 데이터 — Yelp + 항목(aspect) 합성 라벨 (Ch 6과 동일)

Yelp 리뷰엔 multi-label 정답이 없습니다. Ch 6에서처럼 5개 항목(aspect)별 키워드 사전을 만들어 텍스트에서 매칭 — 어떤 키워드라도 등장하면 해당 항목을 1로 활성. 5차원 multi-hot 벡터가 합성됩니다.

| 항목 | 의미 | 키워드 예시 |
|---|---|---|
| `food` | 음식의 맛/메뉴 | food, meal, dish, taste, delicious, ... |
| `service` | 서비스/응대 | service, staff, waiter, friendly, rude, ... |
| `price` | 가격/가성비 | price, cheap, expensive, value, worth, ... |
| `ambiance` | 분위기/인테리어 | atmosphere, decor, music, vibe, cozy, ... |
| `location` | 위치/주차 | location, parking, area, neighborhood, ... |

> **합성의 한계** — 키워드 매칭은 *언급한 항목* 만 잡고 *언급한 항목이 긍정인지 부정인지* 는 알 수 없습니다. 또 *키워드 없이* 항목이 표현된 경우(예: "10 minutes wait" → service)도 놓칩니다. 이 한계는 챕터 끝에서 솔직히 짚습니다.

**Ch 12와의 한 줄 차이**: `out["labels"] = [int(l) for l in batch["label"]]` → `out["labels"] = [list(map(float, a)) for a in batch["aspects"]]`. 라벨이 *int 스칼라* 가 아니라 *길이 5 multi-hot float 벡터*. 이 형식 + `problem_type="multi_label_classification"` 두 가지가 BCE per-label 자동 매핑의 트리거.

## 모델 로드 — `num_labels=5` + `multi_label_classification`

Ch 12와 *모델 아키텍처는 동일* (`Linear(H, 5)` 분류 헤드). 변하는 한 가지 — `problem_type="multi_label_classification"` — 가 자동 매핑되는 loss를 BCE per-label 로 바꿉니다.

**Ch 12와 파라미터 수가 *완전히 동일*** — 차이는 `problem_type` 한 줄뿐. 같은 모델이 *어떻게 해석되고 어떤 loss로 학습되는가* 만 바뀝니다.

## 학습 — Ch 12와 동일한 hyperparams

Ch 12와 *완전히 같은* learning rate, batch size, epoch 수, seed. 평가 metric만 multi-label용으로 새로 짭니다.

## 평가 — 라벨별 sigmoid 확률 + 활성 패턴

Ch 10의 sigmoid+BCE 평가 패턴을 *5번 반복* 한 셈입니다 — 각 라벨에 대해 독립적으로 확률 분포·정확도·F1을 계산.

### 샘플 단위 해석 — 모델 출력을 읽어내는 법

평가 metric (F1·hamming·AUC) 은 *전체 평균* 이라 모델이 *한 리뷰를 보고 어떻게 판단했는지* 직관이 안 옵니다. 본격 시각화로 가기 전에, 5차원 출력을 *문장 단위* 로 어떻게 해석하는지 두 샘플로 짚어 보겠습니다.

**읽는 법 — 표를 한 줄씩**

1. **`true` 컬럼** — 키워드 합성으로 만든 *정답 multi-hot*. 1 이면 "이 리뷰 본문에 그 항목 키워드가 등장했다".
2. **`prob` 컬럼** — 모델이 출력한 *각 항목 sigmoid 확률* (독립). 합이 1 일 필요 없음 — multi-label 의 본질.
3. **`pred` 컬럼** — `prob ≥ 0.5` 이면 1, 아니면 0. *임계값 0.5* 는 사후 후처리 — 라벨별로 다른 값을 줄 수도 있음 (FAQ Q1).
4. **사람이 읽는 한 줄**: `predicted: [...]` 와 `true: [...]` 가 *얼마나 겹치는지* — 두 리스트가 같으면 완벽한 hit, 한 항목 차이면 *near miss*, 전혀 다르면 모델이 헛다리.

**이 표가 한 리뷰에 대해 보여주는 것**:

- 모델이 *어떤 항목에 자신* 있는지 (prob 0.9 이상)
- 어떤 항목에서 *망설이는지* (prob 0.4-0.6 부근 — threshold 살짝 옮기면 결과가 뒤집히는 자리)
- 키워드 합성 라벨의 한계가 드러나는 순간 — 예: 본문에 "10 minutes wait" 처럼 service 를 *키워드 없이* 묘사한 경우 정답은 `service=0` 인데 모델이 prob 0.7 로 활성할 수 있음. 이건 *모델 오답이 아니라 합성 라벨의 누락* 으로 봐야 함.

**전체 metric 의 micro/macro F1 해석**: 위 표 같은 *샘플별 (true vs pred) 비교* 를 평가 셋 1,000건에 대해 *집계* 한 게 §4 상단 metric. micro 는 모든 (샘플 × 라벨) 위치를 동등하게 세고, macro 는 항목 5개의 F1 을 평균. 활성률이 낮은 항목 (location 등) 의 정확도가 *전체* 에 묻히는 걸 막으려면 macro 를 봅니다.

### 4-1. 메인 그림 — 라벨별 sigmoid 확률 KDE (5 패널)

Ch 10에서 봤던 *확률 공간 KDE* 를 5개 라벨에 대해 *각각* 그립니다. 라벨이 *독립* 이라는 multi-label의 본질이 시각적으로 드러나는 그림입니다 — 라벨마다 학습 난이도와 분리도가 *다를 수* 있습니다.

**해석**

- **잘 학습된 라벨** (예: food): label=0 곡선은 0 근처, label=1 곡선은 1 근처에 있고 둘이 거의 만나지 않음. *분리가 깨끗*.
- **활성률이 낮은 라벨** (예: location): label=1 샘플이 적어 곡선이 노이즈가 큼. 그래도 분리는 보여야 함.
- **두 곡선이 0.5 근처에서 크게 겹치면** → 그 라벨은 모델이 잘 못 분리. 키워드 매칭이 *얕아서* 진짜 신호를 못 잡았거나, 학습 데이터가 부족한 상태.

### 4-2. 보조 그림 — 라벨 간 공동 활성 패턴

Multi-label 의 핵심 질문 중 하나: *어떤 라벨 쌍이 같이 등장하는가?* 모델이 라벨 *간 상관* 을 학습 데이터에서 흡수했는지 확인합니다.

`true co-occurrence` (실제 데이터의 라벨 동시 등장 빈도)와 `predicted co-occurrence` (모델 예측의 동시 등장 빈도)를 나란히 그려 두 행렬이 비슷하면 모델이 라벨 구조를 잘 잡고 있는 것.

**해석**

- **대각선 = 1.0** — 자기 자신과는 항상 같이 등장 (정의상).
- **off-diagonal cell M[i, j]** = "라벨 i가 활성된 샘플 중 라벨 j 도 활성된 비율". 비대칭 행렬.
- food와 service가 같이 자주 등장하면 두 모델 모두 0.5+ 값. *true* 와 *predicted* 가 거의 비슷한 패턴이면 모델이 라벨 구조를 잘 학습했다는 뜻.
- **predicted cell이 true cell 보다 일관되게 높으면** → 모델이 라벨을 *너무 많이* 활성하는 경향 (over-prediction). threshold를 0.5보다 높게 (예: 0.6) 두면 calibration 개선.

## 클라이맥스 — Ch 6 sklearn `OneVsRestClassifier(LogisticRegression)` 와 비교

Ch 6의 sklearn 셋업을 *이 노트북 안에서* 다시 학습해 라벨별로 BERT와 비교합니다. **multi-label에서도 BERT의 67M이 sklearn 대비 어디서 이기는가?**

### 5-1. 두 모델의 metric 비교

### 5-2. 라벨별 F1 비교 — 어디서 BERT가 이기나

**해석**

- 키워드 매칭으로 합성한 라벨은 *키워드 단어가 본질 신호* 라 sklearn TF-IDF가 의외로 강합니다 — 라벨 정의 자체가 단어 빈도와 일치하기 때문.
- BERT가 *큰 폭으로* 이기는 라벨이 있다면 → 그 라벨의 *합성 룰이 키워드만으로 안 잡히는 신호* 를 BERT가 추가로 학습한 것 (예: ambiance에서 "lighting was perfect" 같은 묘사).
- BERT가 *지는 라벨* 도 있을 수 있음 — 키워드가 *결정적* 인 라벨에서 sklearn은 *완벽* 한 매칭, BERT는 *근사* 라 약간의 noise가 들어감.

**합성 라벨의 본질적 한계** — 이 비교는 *키워드 매칭으로 만든 라벨* 위에서의 비교. 실제 사람-annotated multi-label 데이터에선 BERT 격차가 훨씬 큼 (단어 빈도로 안 잡히는 미묘한 항목 인식이 BERT의 강점).

## 이 장의 구성

- [13-1. 실습](13-bert_multilabel-practice.md)
- [13-2. 정리와 FAQ](13-bert_multilabel-wrapup.md)
