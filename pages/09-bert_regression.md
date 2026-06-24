**목표**: Phase 0의 별점 회귀(Ch 2)를 *DistilBERT 파인튜닝* 으로 다시 풉니다. sklearn `LinearRegression` 이 1초 만에 풀던 문제를, BERT는 GPU에서 수 분간 학습합니다. `Trainer` 가 처음 등장하고, 우리가 sklearn의 `fit()` 대신 *학습 과정 전체* 를 명시적으로 통제하기 시작합니다.

**환경**: Google Colab **T4 GPU 필수** (런타임 → 런타임 유형 변경 → T4 GPU). CPU에서도 동작은 하지만 학습이 한 시간 가까이 걸립니다.

**예상 소요 시간**: 약 10-15분 (T4 GPU 기준, 모델 다운로드 + 2 에폭 학습 + 평가)

## 학습 흐름

1. 🚀 **실습**: 데이터 준비 (Ch 8 패턴) → 모델 로드 → `Trainer` + `TrainingArguments` 한 묶음으로 학습
2. 🔬 **해부**: 학습 중·후 GPU 메모리(VRAM) 변화, `Trainer` 가 내부에서 하는 일, sklearn(Ch 2) 결과와 직접 비교
3. 🛠️ **변형**: 평가 지표 (`compute_metrics`) 직접 정의, 예측 분포 시각화

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 7-8 | DistilBERT (추론·데이터 파이프라인) | `AutoTokenizer.from_pretrained(...)` | Yelp / 영어 예시 | 사전학습 헤드 | softmax | — |
| **9 ← 여기** | **DistilBERT 파인튜닝** | `AutoTokenizer.from_pretrained(...)` | Yelp (별점 1-5, Ch 2와 동일) | **`Linear(H, 1)`** | 없음 | **`MSELoss`** |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 8)

| 축 | Ch 8 | Ch 9 |
|---|---|---|
| 모델 | 모델 로드 없음 | **`AutoModelForSequenceClassification` (`num_labels=1`, `problem_type="regression"`)** |
| 학습 | 없음 | **있음** — Trainer.train() |
| Loss | — | **`MSELoss`** (Ch 2와 같은 식, 최소화 방식만 SGD로 바뀜) |
| 데이터 | Yelp 토크나이저 옵션 실험 | Yelp 4,000 학습 + 1,000 평가 (별점 1-5 float 라벨) |
| GPU | 옵션 | **필수** — fp16, 옵티마이저+gradient가 VRAM에 추가 |
| 작업 시간 | 즉시 | **수 분** (T4에서 ~5-8분 학습) |

**핵심 변화**: 같은 MSELoss이지만 *어떻게 최소화하느냐* 가 다릅니다.

- Ch 2 `LinearRegression`: 정규방정식으로 *한 번에* 닫힌 해 도출. 1초 미만.
- Ch 9 BERT: SGD/Adam으로 *수천 번 step* 을 밟으며 점진적 최소화. fp16, 옵티마이저 모멘텀, gradient accumulation 등 도구가 한꺼번에 등장.

Ch 6 끝의 "sklearn vs HuggingFace 미리보기" 표가 이번 챕터에서 실제 코드로 펼쳐집니다.

## Loss 노트 — `MSELoss` 그대로, 최소화 방식만 바뀜

수식은 Ch 2와 동일합니다.

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat y_i)^2$$

다른 점은 *어떻게 이 $L$을 최소화하느냐* 입니다.

| 항목 | Ch 2 (`LinearRegression`) | Ch 9 (BERT) |
|---|---|---|
| 최소화 방법 | 정규방정식 $w = (X^\top X)^{-1} X^\top y$ — 한 번에 닫힌 해 | Adam optimizer — gradient descent step을 수천 번 |
| 학습 시간 | 1초 미만 | T4에서 5-8분 |
| 결정성 | 입력이 같으면 가중치가 정확히 같음 | random seed·batch 순서에 따라 매번 미세 차이 |
| 왜 BERT를 쓰나 | 단어 독립 가정의 한계 (`"not bad"` ≠ `"bad"` 구분 불가) | 문맥을 attention으로 학습해 더 정확한 회귀 |

Hugging Face `Trainer` 는 `problem_type="regression"` 을 보고 자동으로 `MSELoss` 를 적용합니다. 우리가 직접 `criterion = nn.MSELoss()` 같은 코드를 쓸 필요가 없습니다.

## 토크나이저 노트

Ch 7·8과 동일한 `distilbert-base-uncased` WordPiece 토크나이저를 그대로 사용합니다. 이번 챕터는 모델·loss·학습 루프에 집중하므로 토크나이저 파이프라인은 Ch 8에서 익힌 그대로 (`map(batched=True)` + `DataCollatorWithPadding`).

> **다음 챕터(Ch 10·11)**: 같은 토크나이저, 같은 데이터지만 task가 binary 분류로 바뀝니다. Ch 10에서 sigmoid+BCE 방식, Ch 11에서 softmax+CE 방식을 *별도 학습* 해 두 방식을 비교합니다.

## 이 장의 구성

[[SubPages]]
