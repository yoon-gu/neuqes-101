**목표**: Ch 4(sklearn)에서 본 *두 방식 동등성* 의 BERT 버전을 시작합니다. 이번 챕터는 **방식 A**인 sigmoid + BCE 패턴을 BERT로 학습합니다 (`num_labels=1`, `problem_type="multi_label_classification"`). 다음 Ch 11에서 같은 데이터를 **방식 B**(softmax + CE)로 학습한 뒤 두 결과를 비교합니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 10분 (모델 다운로드 + 2 에폭 학습 + 평가)


## 학습 흐름

1. 🚀 **실습**: Ch 3과 같은 Yelp 이진화 데이터를 BERT로 학습 — `num_labels=1` + sigmoid + `BCEWithLogitsLoss`
2. 🔬 **해부**: 학습 후 sigmoid 확률 분포 직접 확인, 평가 지표(accuracy/precision/recall/F1/AUC) 계산
3. 🛠️ **다음 챕터(Ch 11) 예고**: 같은 task에 `num_labels=2` + softmax + `CrossEntropyLoss` 로 다시 학습해 두 방식 결과 비교


> 📒 **사전 학습 자료**: Ch 4 (sklearn binary on softmax) — 두 방식이 수학적으로 동등하다는 것을 식으로 본 챕터. Ch 9 (BERT regression) — `Trainer` 기본 골격.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` |
| 9 | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp (별점 1-5) | `Linear(H, 1)` | 없음 | `MSELoss` |
| **10 ← 여기** | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp 이진화 | **`Linear(H, 1)`** | **sigmoid** | **`BCEWithLogitsLoss`** |
| 11 (다음) | DistilBERT 파인튜닝 | 같음 | Yelp 이진화 | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 9)

| 축 | Ch 9 | Ch 10 |
|---|---|---|
| Task | 회귀 | **이진 분류 (방식 A)** |
| `num_labels` | 1 | **1** (그대로) |
| `problem_type` | `"regression"` | **`"multi_label_classification"`** ← BCE 자동 적용 트릭 |
| Activation | 없음 | **sigmoid** (output head는 1차원, 학습 시 logit이 sigmoid 통과 후 BCE) |
| Loss | `MSELoss` | **`BCEWithLogitsLoss`** |
| 라벨 | float (1-5) 별점 | **float [0.0 또는 1.0]** (multi-hot 1차원 벡터로 둠) |
| 데이터 | Yelp 별점 1-5 | **Yelp 이진화** (4-5 → 1, 1-2 → 0, 3 제외) |

### `num_labels=1` + `problem_type="multi_label_classification"` 의 트릭

`Trainer` 의 자동 loss 매핑은 이렇게 작동합니다 ([Ch 9에서 본 표](../09_bert_regression/09_bert_regression.ipynb)).

| `problem_type` | 자동 적용 loss | num_labels | 라벨 형식 |
|---|---|---|---|
| `"regression"` | `MSELoss` | 보통 1 | float |
| `"single_label_classification"` | `CrossEntropyLoss` | K (≥2) | int 인덱스 |
| `"multi_label_classification"` | **`BCEWithLogitsLoss`** | K (≥1) | **multi-hot float** |

방식 A는 *binary 분류이지만 num_labels=1* 형태를 유지해야 합니다. 그러려면 `multi_label_classification` 으로 두어 BCE를 적용시키되, *num_labels=1짜리 multi-label* 즉 라벨을 길이 1짜리 multi-hot 벡터(`[0.0]` 또는 `[1.0]`)로 만들면 됩니다. 이게 sklearn `LogisticRegression()` 의 sigmoid+BCE와 정확히 같은 셋업입니다.

## Loss 노트 — `BCEWithLogitsLoss` (Ch 3 그대로, BERT 맥락에서 다시)

수식과 직관은 Ch 3에서 봤습니다.

$$L = -\frac{1}{N}\sum_{i=1}^{N}\left[\,y_i \log \hat p_i + (1 - y_i)\log(1 - \hat p_i)\,\right]$$

이번 챕터에서 새로운 점:

1. **모델이 BERT** 라 logit $z = w^\top h_{[CLS]} + b$ 의 *분포 표현 $h_{[CLS]}$* 가 768차원 hidden state를 압축한 결과입니다 (sklearn TF-IDF 입력보다 풍부).
2. `BCEWithLogits` 의 *Logits* — 모델 마지막 단의 raw 점수에 sigmoid를 따로 통과시키지 않고 BCE 안에서 한꺼번에 처리하기 때문에 **수치적으로 안정** 합니다.
3. `Trainer` 가 `problem_type="multi_label_classification"` 만 보고 자동으로 BCE를 골라줍니다. 우리는 라벨을 `[0.0]` 또는 `[1.0]` float 형태로 두기만 하면 됩니다.

**숫자로 감 잡기** — Ch 3 표 그대로:

| 정답 $y$ | 예측 확률 $\hat p$ | 손실 $-\log \hat p$ |
|---|---|---|
| 1 | 0.9 | 0.105 |
| 1 | 0.5 | 0.693 |
| 1 | 0.1 | **2.303** |

확률이 0에 가까울수록 손실이 로그 스케일로 폭증한다는 BCE의 성격은 sklearn에서든 BERT에서든 동일합니다. 다른 점은 *어떻게 그 확률을 만드느냐* 입니다 — sklearn은 단어 빈도, BERT는 attention으로 압축한 문장 표현.

## 토크나이저 노트

Ch 7-9와 같은 `distilbert-base-uncased` WordPiece 토크나이저. 토크나이저·데이터 가공 파이프라인은 Ch 8에서 익힌 패턴을 그대로 적용합니다.

> **다음 챕터(Ch 11)**: 같은 토크나이저, 같은 데이터, 같은 BERT 본체. 변하는 건 출력 헤드가 1차원에서 2차원으로 늘어나고 sigmoid가 softmax로 바뀐다는 점뿐입니다.

**baseline VRAM**:

## 데이터 — Yelp 이진화 (Ch 3·4와 동일)

별점 4-5는 `1.0` (긍정), 1-2는 `0.0` (부정), 3은 제외. 라벨을 *float 1차원 multi-hot 벡터* (`[0.0]` 또는 `[1.0]`) 형태로 둡니다 — 이게 BCE를 자동 적용시키는 핵심 형식.

## 모델 로드 — 방식 A 셋업

`num_labels=1` + `problem_type="multi_label_classification"` 이 핵심.

## 학습 — Ch 9 골격 그대로

`compute_metrics` 만 binary 분류용으로 새로 짭니다 — sigmoid + threshold 0.5 로 0/1 예측을 만들고 accuracy/F1/AUC 계산.

## 평가 — sigmoid 확률 분포 직접 확인

`Trainer.predict()` 로 logit을 받아 sigmoid를 통과시킨 확률 분포를 살펴봅니다.

### 4-1. 메인 그림 — *확률 공간* 에서 라벨별 분포 (`seaborn.kdeplot`)

`seaborn.kdeplot` 으로 *부드러운* 분포를 그립니다. histogram이 막대로 끊기는 반면 KDE는 연속 곡선이라 두 분포가 어디서 만나는지(=오분류 영역)가 한눈에 들어옵니다.

이 그림에서 봐야 할 세 가지:

- **양 끝 봉우리**: 학습이 잘 되면 라벨 0의 확률은 0 근처에, 라벨 1의 확률은 1 근처에 몰립니다 — sigmoid가 큰 음수 logit을 0에, 큰 양수 logit을 1에 *압착* 시키기 때문 ($\sigma(z) = 1/(1+e^{-z})$ 의 양 극단 포화).
- **0.5 근처의 교차 영역**: 두 곡선이 만나는 부분이 모델이 헷갈려하는 샘플들. 면적이 작을수록 분리가 잘 된 것.
- **반대쪽 꼬리**: 라벨 0인데 확률 1쪽에, 라벨 1인데 확률 0쪽에 잡히는 작은 봉우리는 *오분류*. 이 두 꼬리가 학습 손실(BCE)이 가장 크게 잡히는 영역.

**설명 — 왜 양 끝이 솟아 있나?** sigmoid는 logit이 ±5만 넘어가도 거의 0 또는 1로 수렴합니다 ($\sigma(5) \approx 0.993$, $\sigma(-5) \approx 0.007$). BERT가 학습 후 어느 정도 자신감을 갖게 되면 logit이 ±5-10 범위로 뻗어 나가고, 결과적으로 확률 공간에서는 **양 끝에 압착된 U자 분포**가 나옵니다. 가운데(0.3-0.7)는 모델이 *판단을 망설이는* 샘플 — 진짜 어려운 케이스이거나 라벨 노이즈일 가능성이 큽니다.

**`common_norm=False` 의 의미**: 라벨별로 *각자* 적분이 1이 되도록 정규화. 이렇게 해야 라벨 0 샘플 수와 라벨 1 샘플 수가 다를 때도 *분포의 모양* 만 비교됩니다 (개수 차이는 빠짐).

### 4-2. 보조 그림 — *logit 공간* 에서 같은 분포 (`BCE가 실제로 동작하는 자리`)

방금 본 확률 공간 그림은 사용자 눈에 보이는 결과지만, **`BCEWithLogitsLoss` 가 실제로 손실을 계산하는 자리** 는 *logit 공간* 입니다 ($z$, sigmoid를 통과하기 *전*). 같은 데이터를 logit 축에서 다시 그려보면 사뭇 다른 풍경이 펼쳐집니다.

확률 공간(4-1)에서는 분포가 0과 1 양 끝에 *압착*되어 안쪽 모양을 알 수 없었는데, logit 공간에서는 **두 개의 정규분포-비슷한 봉우리**가 결정 경계 $z = 0$ 양옆에 깔끔하게 분리됩니다. 이게 BERT가 학습한 *진짜 표상*에 더 가깝습니다.

**두 그림을 함께 보는 법 — sigmoid가 한 일**

- 확률 공간(4-1)의 *양 끝 압착* 은 logit 공간(4-2)의 *바깥쪽 꼬리* 에서 옵니다. logit이 +6 이든 +10 이든 sigmoid 통과 후엔 모두 0.99 이상이라 구분이 안 됨 — 정보가 *압축* 되는 것.
- 결정 경계는 두 그림 모두 *같은 자리*: 확률에서 0.5, logit에서 0. 단지 좌표축이 다를 뿐.
- 두 봉우리의 **거리** 는 logit 공간에서만 의미가 있습니다. 거리가 멀수록 모델이 두 클래스를 자신 있게 구분하는 것. 확률 공간에서는 이 거리가 양 끝 압착 때문에 안 보입니다.

**왜 `BCEWithLogitsLoss` 인가** — BCE를 *확률* 위에서 계산하면 ($p = \sigma(z)$), $\log p$ 와 $\log(1-p)$ 가 양 극단에서 0에 매우 가까운 수가 되어 로그 안의 수치가 폭주합니다 (`log(0)` 발산). 반면 logit 위에서 직접 계산하면 ($\text{BCE}(z, y) = \max(z, 0) - z y + \log(1 + e^{-|z|})$) 로그-합-지수(log-sum-exp) 트릭으로 **수치적으로 안정**. 그래서 우리는 모델 출력 logit을 sigmoid 통과 *없이* 그대로 `BCEWithLogitsLoss` 에 넣습니다.

## 결과 저장 — Ch 11에서 비교용

다음 챕터 Ch 11에서 같은 데이터에 *방식 B* (softmax+CE)로 학습한 뒤 *이번 방식 A* 의 결과와 비교합니다. 평가 지표와 확률 예측을 디스크에 저장해 두면 비교가 깔끔해집니다.

**참고**: Colab은 세션이 끝나면 `./shared_binary_results/` 가 사라집니다. *Drive에 보존* 하려면 다음과 같이 마운트.

```python
from google.colab import drive
drive.mount("/content/drive")
import shutil
shutil.copytree("./shared_binary_results", "/content/drive/MyDrive/neuqes-101/shared_binary_results")
```

Ch 11 노트북은 같은 세션에서 이어 돌리거나, 같은 데이터·seed·모델로 다시 학습해서 결과를 만든 뒤 비교합니다.

## 이 장의 구성

- [10-1. 실습](10-bert_binary_sigmoid-practice.md)
- [10-2. 정리와 FAQ](10-bert_binary_sigmoid-wrapup.md)
