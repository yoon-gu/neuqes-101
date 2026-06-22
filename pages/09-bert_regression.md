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

**baseline VRAM** — 모델 로드 전:

## 데이터 준비

Ch 8에서 익힌 `datasets` + 토크나이저 패턴을 그대로 적용합니다. 차이는 라벨을 *float* 형으로 바꾼다는 점입니다 — 회귀이므로 정답이 정수 클래스가 아닌 실수입니다.

별점 1-5를 그대로 학습 라벨로 사용합니다 (`label` 필드는 0-4로 저장돼 있어 +1).

## 모델 로드 — `num_labels=1`, `problem_type="regression"`

Ch 7에서는 사전학습된 분류 헤드(`distilbert-base-uncased-finetuned-sst-2-english`, num_labels=2)를 그대로 썼습니다. 이번엔 본체 모델만 받고 **분류 헤드를 새로** 만듭니다 — `num_labels=1` 이라 출력 차원이 1, `problem_type="regression"` 이라 `Trainer` 가 자동으로 MSELoss 사용.

**경고 메시지를 보셨을 겁니다** — `Some weights of DistilBertForSequenceClassification were not initialized ...`. 분류 헤드(`Linear(768, 1)`)가 새로 만들어지면서 *랜덤 초기화* 됐다는 알림입니다. 이 부분이 학습으로 채워지고, BERT 본체는 사전학습 가중치를 미세 조정합니다 (transfer learning의 본 모습).

### 학습되는 파라미터 vs 동결된 파라미터

`from_pretrained()` 직후엔 *모든* 파라미터가 학습 대상입니다 (`requires_grad=True`). 그러나 데이터가 작거나 빠른 학습이 필요하면 BERT 본체를 *동결(freeze)* 하고 분류 헤드만 학습하기도 합니다. 학습 시작 전에 *전체 vs 학습되는 파라미터* 를 한 번 확인하는 게 좋은 습관입니다.

### 시연: BERT 본체 동결 패턴

본 학습은 *모든 파라미터* 를 학습하지만, 동결 패턴이 어떻게 적용되는지 *별도 모델 인스턴스* 로 한 번 보여드립니다 (이 시연 모델은 학습에 사용하지 않습니다).

**언제 동결을 쓰나**

- **분류 헤드만 학습 (모든 본체 동결)**: 데이터 매우 작음 (수백 건), 빠른 baseline 필요.
- **하위 N개 layer 동결**: 일반 언어 표현은 BERT 그대로, 상위 layer만 task 적응.
- **모든 파라미터 학습 (default)**: 데이터 충분 (수천 건+), 본체도 task에 맞게 적응.

이번 챕터는 4,000건이라 default(전체 학습)이 가장 좋은 선택입니다.

모델 가중치(약 67M 파라미터, fp32 약 255 MB)가 GPU에 올라간 상태입니다. 학습이 시작되면 *옵티마이저 모멘텀(2배) + gradient(1배)* 가 추가되어 VRAM이 더 늘어납니다.

## `TrainingArguments` + `Trainer`

Ch 6 끝에서 미리 본 코드 형태가 이제 실제로 등장합니다. `TrainingArguments` 한 객체에 학습 하이퍼파라미터를 모두 모으고, `Trainer` 가 학습 루프·평가·로그·체크포인트를 자동화합니다.

학습이 진행되는 동안 step별 loss와 에폭별 평가 metric이 출력됩니다. **핵심 관찰**:

- `loss` 가 처음 수 step에서 큰 값(흔히 0.3-0.5)이었다가 학습이 진행되면 줄어들어야 정상입니다.
- 에폭 끝에서 출력되는 `eval_mse`, `eval_mae`, `eval_r2` 가 우리가 정의한 평가 지표입니다.
- `loss` 가 줄어들지 않거나 nan으로 가면 학습률을 낮추거나(`5e-6`), `fp16=False` 로 시도해 봅니다.

> 📒 **부록 노트북 두 편**
>
> 1. [`appendix_experiment_tracking.ipynb`](./appendix_experiment_tracking.ipynb) — `report_to` 인자로 **wandb · trackio · MLflow** 같은 experiment tracker를 붙이는 패턴. 학습 곡선·평가 metric을 dashboard에서 보고 여러 run을 한 화면에 비교. ([Colab으로](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/appendix_experiment_tracking.ipynb))
>
> 2. [`appendix_hpo.ipynb`](./appendix_hpo.ipynb) — **하이퍼파라미터 최적화(HPO)의 어려움**. `TrainingArguments` 인자 정리, HPO가 어려운 5가지 이유, `Trainer.hyperparameter_search` + Optuna 직접 시도, wandb sweeps · MLflow autolog 통합. ([Colab으로](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/appendix_hpo.ipynb))

학습 후 VRAM 상태입니다. 학습 *중* 에는 옵티마이저 모멘텀과 gradient가 추가되어 더 큰 VRAM을 잠시 쓰지만, 학습이 끝나면 일부가 해제됩니다 (단, PyTorch 캐시 할당자가 다음 사용을 위해 일부 메모리를 보유).

**학습 시 VRAM 구성 (fp16 기준)**:

| 구성 요소 | 크기 (DistilBERT 67M 기준) |
|---|---|
| 모델 가중치 (fp16) | ~128 MB |
| Adam 1차 모멘텀 (fp32 마스터) | ~255 MB |
| Adam 2차 모멘텀 (fp32 마스터) | ~255 MB |
| Gradient (fp16) | ~128 MB |
| Activation (배치 16, max_len 128) | ~수백 MB |
| 합계 | 약 1-1.5 GB |

큰 모델(BERT-large 340M)이나 큰 배치를 쓰면 한도(15.36 GB)에 빠르게 다가갑니다.

## 평가 — sklearn(Ch 2)과 직접 비교

학습된 BERT의 평가 지표를 같은 데이터에 sklearn `LinearRegression`(Ch 2 방식)으로 학습한 결과와 비교합니다. BERT가 더 정확하면 *문맥 정보가 단어 독립 가정을 깬다* 는 가설이 검증됩니다.

**해석 가이드** (실제 숫자는 random seed에 따라 조금씩 다릅니다):

- BERT의 MSE가 sklearn보다 작다면, *문맥을 활용한 회귀가 단어 독립 회귀보다 정확하다* 는 직관이 확인됩니다.
- BERT의 R²가 더 높다면 평균 예측이 데이터 분산을 더 잘 설명합니다.
- 차이가 크지 않다면? Yelp 별점은 단어 빈도(긍정 단어 vs 부정 단어)만으로도 꽤 잡히는 task라 그런 경우가 있습니다. *문맥 활용 효과* 가 크게 드러나는 task는 Ch 14 auxiliary나 Ch 15 한국어 NSMC 쪽이 더 명확할 수 있습니다.

### 시각 1 — 예측 분포 per actual class

각 actual class에 대해 BERT와 sklearn이 *어떤 값을 출력했는지* 의 분포를 split violin으로 좌우에 둡니다. 빨간 점선이 ideal (정답 = 예측). 분포 중심이 그 선 근처에 모이고 좌우 폭이 좁을수록 정확합니다.

**무엇이 보이나**

- BERT 쪽 violin이 더 가늘고 빨간 점선 근처에 모이면 같은 actual class 안에서 예측 일관성이 높다는 뜻.
- 두 끝(1점, 5점)에서 분포 중심이 안쪽으로 살짝 치우치는 모양이 자주 보입니다 — 모델이 *중앙 쪽으로 회귀(regression to the mean)* 하는 경향.
- sklearn 쪽 violin이 더 두텁고 길게 늘어진다면 outlier 예측이 많다는 신호.

이 그래프는 "모델이 무엇을 출력하나"의 *raw 분포* 를 봅니다. 다음 그래프는 *오차 자체* 에 집중합니다.

### 시각 2 — 잔차(Residual = Predicted − Actual) 분포 per actual class

`Predicted − Actual` 을 y축에 두고 0 기준선을 긋습니다. 잔차가 0 근처에 좁게 모일수록 정확하고, 양/음 한 쪽으로 치우치면 *bias* 가 있다는 뜻.

**무엇이 보이나**

- 잔차의 *중심* 이 0 위/아래 어디에 있는지가 *bias의 방향*. 1점 class에서 잔차 중심이 +쪽이면 모델이 "1점인데 1점보다 높게" 예측하는 경향이 있다는 뜻.
- 잔차의 *폭* 이 그 class에서의 일반적 오차 크기. BERT가 sklearn보다 좁다면 더 정확.
- 두 끝 class(1점, 5점)에서 잔차 중심이 *반대 방향* (1점은 +, 5점은 −)으로 치우치는 패턴이 자주 보입니다 — 위에서 본 *regression to the mean* 의 잔차 시각화 형태.
- 0 기준선에서 멀리 늘어진 꼬리는 큰 오차를 내는 outlier 샘플들. 어느 모델이 꼬리가 더 두꺼운지 비교.

**두 시각을 함께 보는 이유**: 시각 1은 *모델이 무엇을 출력하나* (raw 분포), 시각 2는 *얼마나 틀렸나* (오차 분포). 같은 데이터의 다른 시점이라 한쪽만 봐서는 놓치는 패턴이 있습니다. 정량 지표(MSE/MAE/R²) 표와 이 두 시각을 함께 읽으면 회귀 평가가 입체적이 됩니다.

## 이 장의 구성

[[SubPages]]
