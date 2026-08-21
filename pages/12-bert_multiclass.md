**목표**: Ch 11(BERT binary, softmax+CE) 셋업을 그대로 두고 **클래스 개수만 2 → 5** 로 늘립니다. 데이터는 Yelp 별점 1-5를 *그대로* 5클래스 분류로 사용 (Ch 3-4·10-11처럼 이진화하지 않음). 이번 챕터는 Ch 5(sklearn multinomial LogReg)의 BERT 버전입니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 3분 (BERT 학습 약 40초 + sklearn 비교 baseline 약 30초 + 다운로드·평가·시각화)

## 학습 흐름

1. 🚀 **실습**: Ch 11과 같은 `(num_labels=K, problem_type="single_label_classification")` 셋업, K만 5로. Yelp 별점 1-5를 라벨 0-4 int 인덱스로.
2. 🔬 **해부**: 학습 후 *혼동 행렬* 과 *top-1 확률 분포* 로 클래스별 패턴 확인. 별점 4 ↔ 5 같은 *인접 클래스 혼동* 이 자연스러운지 검증.
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 Ch 5의 sklearn baseline(TF-IDF + multinomial LogReg)을 *inline 재현* 해 BERT 67M 파라미터가 진짜 도움이 되는지 직접 비교. 격차가 *데이터 양에 어떻게 의존* 하는지는 부록 `12_bert_multiclass_data_scaling` 의 100-30K 곡선에서 봅니다.

> 📒 **사전 학습 자료**: Ch 5 (sklearn multi-class), Ch 11 (BERT binary 방식 B). 이번 챕터는 self-contained — 다른 챕터의 결과 파일에 의존하지 않습니다.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 5 | `LogisticRegression(multinomial)` | `TfidfVectorizer()` | Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| 11 | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp 이진화 | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| **12 ← 여기** | DistilBERT 파인튜닝 | 같음 | **Yelp 5클래스** | **`Linear(H, 5)`** | softmax | `CrossEntropyLoss` |
| 13 (다음) | DistilBERT 파인튜닝 | 같음 | Yelp + 항목 키워드 (5라벨 multi-label) | `Linear(H, 5)` | sigmoid (per-label) | `BCEWithLogitsLoss` (per-label) |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 11)

| 축 | Ch 11 (binary) | Ch 12 (multi-class) |
|---|---|---|
| **Task** | 이진 분류 | **5-클래스 분류** ← *유일한 변화* |
| `num_labels` | 2 | **5** |
| 데이터 | Yelp 이진화 (별점 3 제외) | **Yelp 별점 1-5 그대로** (제외 없음) |
| 라벨 형식 | int `0` / `1` | **int `0`-`4`** (별점-1) |
| `problem_type` | `single_label_classification` | (그대로) |
| Activation / Loss | softmax / CE | (그대로) |
| 평가 metric | binary precision/recall/F1 + AUC | **accuracy + macro precision/recall/F1 + multi-class AUC (OvR)** |
| 학습 hyperparams (lr, batch, epoch, seed) | 동일 | 동일 |

> **변경점 한 가지 원칙**: Loss·activation·문제 셋업이 그대로 유지되고 *task 차원만 K=2 → K=5* 로 일반화됩니다. Ch 5의 sklearn 챕터에서 본 K=5 셋업이 BERT에 그대로 옮겨오는 모습을 확인하는 것이 핵심.

## Loss 노트 — `CrossEntropyLoss` 가 K=5 에서 어떻게 보이나

수식은 Ch 4-5·11과 동일:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log \hat p_{i, y_i} \quad\text{where}\quad \hat p_{i,k} = \dfrac{e^{z_{i,k}}}{\sum_{j=1}^{K} e^{z_{i,j}}}$$

K가 늘어나면 *random baseline 손실* 도 같이 커집니다 — 학습 초반 모델이 logit을 거의 0으로 출력하면 softmax는 균등 $(1/K, \ldots, 1/K)$ 가 되고 정답 클래스의 손실은 $-\log(1/K) = \log K$.

| K | random baseline loss $-\log(1/K)$ | 의미 |
|---|---|---|
| 2 | $\log 2 = 0.693$ | Ch 11 학습 첫 step에서 흔히 보이는 값 |
| 5 | $\log 5 = 1.609$ | **이번 챕터 학습 첫 step의 baseline** |
| 10 | $\log 10 = 2.303$ | 일반적인 ImageNet 1000클래스 학습 비교 |
| 1000 | $\log 1000 = 6.908$ | 학습 시작 직후 손실이 ~7이면 정상 |

**숫자로 감 잡기 (K=5, 정답=클래스 4)** — logits에서 정답 클래스가 얼마나 커야 손실이 얼마인지:

| logits $(z_0, z_1, z_2, z_3, z_4)$ | softmax → $\hat p_4$ | 손실 $-\log \hat p_4$ |
|---|---|---|
| $(0, 0, 0, 0, 0)$ | $0.200$ | **1.609** ← random |
| $(0, 0, 0, 0, 2)$ | $0.541$ | 0.615 |
| $(0, 0, 0, 0, 5)$ | $0.985$ | 0.015 |
| $(5, 0, 0, 0, 0)$ | $0.005$ | **5.310** ← 자신 있게 틀린 케이스 |

**핵심 직감 — softmax는 *상대 logit* 만 본다**: 모든 logit에 같은 상수를 더해도 softmax는 변하지 않음 ($e^{z_k+c} / \sum e^{z_j+c} = e^{z_k}/\sum e^{z_j}$). 즉 K=5 모델이 학습할 때 의미 있는 신호는 *클래스 간 logit 차이* 뿐. *softmax의 4가지 자유도* (K=5에서 K-1=4)만 학습됨.

## 토크나이저 노트

Ch 11과 완전히 동일 — `distilbert-base-uncased` WordPiece, `max_length=128`. 토크나이저는 라벨 개수에 상관없이 *문장* 만 처리하므로 K가 2든 5든 변화 없습니다. 라벨 개수는 모델의 *분류 헤드* 와 *데이터 라벨 형식* 에서만 다릅니다.

> **다음 챕터(Ch 13)**: 토크나이저 동일. 변하는 건 *라벨 형식* (int 인덱스 → multi-hot 벡터)과 그에 따른 활성화·loss(softmax/CE → sigmoid/BCE per-label).

## 이 장의 구성

[[SubPages]]
