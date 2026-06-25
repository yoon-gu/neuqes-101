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

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

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

## 이 장의 구성

[[SubPages]]
