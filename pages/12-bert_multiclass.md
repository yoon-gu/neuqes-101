**목표**: Ch 11(BERT binary, softmax+CE) 셋업을 그대로 두고 **클래스 개수만 2 → 5** 로 늘립니다. 데이터는 Yelp 별점 1-5를 *그대로* 5클래스 분류로 사용 (Ch 3-4·10-11처럼 이진화하지 않음). 이번 챕터는 Ch 5(sklearn multinomial LogReg)의 BERT 버전입니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 12분 (BERT 학습 ~10분 + sklearn 비교 baseline ~30초 + 평가/시각화)


## 학습 흐름

1. 🚀 **실습**: Ch 11과 같은 `(num_labels=K, problem_type="single_label_classification")` 셋업, K만 5로. Yelp 별점 1-5를 라벨 0-4 int 인덱스로.
2. 🔬 **해부**: 학습 후 *혼동 행렬* 과 *top-1 확률 분포* 로 클래스별 패턴 확인. 별점 4 ↔ 5 같은 *인접 클래스 혼동* 이 자연스러운지 검증.
3. 🛠️ **클라이맥스**: 같은 노트북 안에서 Ch 5의 sklearn baseline(TF-IDF + multinomial LogReg)을 *inline 재현* 해 BERT 67M 파라미터가 진짜 도움이 되는지 직접 비교.


> 📒 **사전 학습 자료**: Ch 5 (sklearn multi-class), Ch 11 (BERT binary 방식 B). 이번 챕터는 self-contained — 다른 챕터의 결과 파일에 의존하지 않습니다.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 5 | `LogisticRegression(multinomial)` | `TfidfVectorizer()` | Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| 11 | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp 이진화 | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| **12 ← 여기** | DistilBERT 파인튜닝 | 같음 | **Yelp 5클래스** | **`Linear(H, 5)`** | softmax | `CrossEntropyLoss` |
| 13 (다음) | DistilBERT 파인튜닝 | 같음 | Yelp + 항목 키워드 (5라벨 multi-label) | `Linear(H, 5)` | sigmoid (per-label) | `BCEWithLogitsLoss` (per-label) |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

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

**baseline VRAM**:

## 데이터 — Yelp 별점 1-5 (Ch 5와 동일)

별점 3 제외 같은 전처리 *없이* 그대로 사용. 라벨은 `dataset["label"]` 가 이미 0-4 int 인덱스 (Yelp 데이터셋의 기본 형식).

**Ch 11 과의 한 줄 차이**: `out["labels"] = [int(b) for b in batch["binary"]]` → `out["labels"] = [int(l) for l in batch["label"]]`. 별점-1 인덱스를 그대로 라벨로 사용.

## 모델 로드 — `num_labels=5` 만 바뀜

Ch 11 셋업에서 K=2 → K=5 한 줄 변화.

**파라미터 수 비교 — K가 늘어나도 거의 변하지 않습니다**

| 부분 | Ch 11 (K=2) | Ch 12 (K=5) |
|---|---|---|
| DistilBERT body | 66,362,880 | 66,362,880 |
| pre_classifier (`Linear(768→768)`) | 590,592 | 590,592 |
| classifier (`Linear(768→K)`) | 1,538 | **3,845** |
| 합계 | 66,955,778 | **66,958,085** |

분류 헤드만 K에 비례해 늘어나지만 (768·K + K), DistilBERT body가 ~67M이라 K=2 ↔ K=5 전체 차이는 0.003%. **K가 늘어났다고 모델이 *훨씬* 무거워지지는 않는다** 는 점이 multi-class BERT의 매력 중 하나.

## 학습 — Ch 11과 동일한 hyperparams

Ch 11과 *완전히 같은* learning rate, batch size, epoch 수, seed. 변하는 건 모델의 출력 차원 (5)과 평가 metric의 average 방식 (`"macro"`, multi-class AUC는 `multi_class="ovr"`).

## 평가 — softmax 확률 분포와 혼동 패턴

Ch 11 패턴 그대로 — `Trainer.predict()` 로 logits를 받아 softmax → argmax. K=5에선 *클래스마다* 정밀도·재현율이 다를 수 있어서 *macro* 평균과 *클래스별* 분해를 같이 봅니다.

### 4-1. 메인 그림 — 혼동 행렬 (`seaborn.heatmap`)

5클래스 분류의 *어디에서 혼동이 일어나는지* 한눈에 보는 가장 강력한 도구입니다. 행은 정답 별점, 열은 예측 별점, 셀의 숫자는 해당 (정답, 예측) 조합의 샘플 수.

**봐야 할 패턴**

- **대각선** (정답=예측): 색이 진할수록 그 클래스가 잘 맞은 것.
- **인접 클래스 혼동** (`(2★, 3★)`, `(4★, 5★)` 등): 별점은 *순서가 있는* 라벨이라 인접 별점끼리 헷갈리는 건 자연스럽습니다.
- **먼 클래스 혼동** (`(1★, 5★)`): 이건 진짜 오류. 데이터에 라벨 노이즈가 있거나 모델 학습이 부족한 신호.

**해석 가이드**

- 색의 진하기는 *행 정규화* (정답 클래스 안에서의 비율) — 대각선 셀의 색이 그 클래스의 *재현율* 입니다.
- 숫자는 *원본 카운트* 라 클래스별 표본 크기도 같이 보입니다 — 어떤 클래스에 모델 평가 표본이 적으면 통계적 노이즈가 큼을 인지.
- `1★ → 2★` 또는 `4★ → 5★` 같은 *±1 이웃 오류* 가 가장 흔할 것 — 별점 회귀에 가까운 task의 자연스러운 양상. 별점 *3★* 이 가장 어려울 가능성이 큰데, 이는 사람도 1★/2★보다 헷갈리는 *중간* 평가이기 때문.

### 4-2. 보조 그림 — top-1 확률의 분포 (정답/오답 갈림)

K=5에서는 *어느 한 클래스에 압도적인 자신감* 이 있는 경우와 *2-3 클래스 사이에서 갈피를 못 잡는* 경우가 나뉩니다. 정답·오답을 구분해 그리면 모델 자신감이 *얼마나 calibration 됐는지* 가 드러납니다.

**해석**

- **잘 학습된 모델**은 *correct 곡선이 1.0 가까이* 몰리고 *wrong 곡선은 더 낮은 영역* (0.4-0.7)에 퍼져 있습니다. 모델이 틀릴 때는 *덜 자신 있게* 틀려야 calibration이 좋다는 뜻.
- **두 곡선이 1.0 근처에서 함께 압착** 되어 있으면 → 모델이 *틀린 답에도 매우 자신* 있는 *over-confident* 상태. 별점 ±1 이웃 오류가 많을수록 이 현상이 도드라짐.
- **correct 곡선이 0.5-0.8 근처에 머무르면** → 모델이 *정답을 알면서도 망설이는* 상태. 학습이 부족하거나 task가 본질적으로 모호한 경우.

## 클라이맥스 — sklearn TF-IDF + LogReg 와의 비교 (Ch 5의 BERT 검증)

같은 데이터에 Ch 5 셋업(TF-IDF + multinomial LogReg)을 *이 노트북 안에서* 다시 학습해 비교합니다. **BERT 67M 파라미터가 진짜로 도움이 되는가?** 가 이 비교의 핵심 질문 — sklearn은 GPU 없이도 몇 초 만에 끝나기 때문에 self-contained로 부담 없이 포함됩니다.

### 5-1. 두 모델의 metric 표 비교

### 5-2. 두 모델의 혼동 행렬 비교

같은 평가 데이터에 sklearn은 어디서, BERT는 어디서 헷갈리는지 *나란히* 봅니다.

**해석 가이드**

- *대각선이 더 진하면* 그 모델이 더 잘 맞춘 것.
- *인접 클래스 혼동(±1)* 은 두 모델 모두에서 가장 흔할 것 — 별점이 *순서형* 라벨이라 자연스럽습니다.
- BERT가 sklearn 대비 가장 크게 개선되는 영역은 보통 **3★ (중간 별점)**: 단어 빈도만으로는 *애매한 칭찬·비판이 섞인* 리뷰를 구분하기 어렵지만, BERT는 attention으로 문맥을 보기 때문.
- 만약 BERT가 sklearn보다 *모든 셀에서* 비슷하거나 더 나쁘다면 → 학습량 부족 신호. epoch을 늘리거나 lr을 조정.

## 이 장의 구성

- [12-1. 실습](12-bert_multiclass-practice.md)
- [12-2. 정리와 FAQ](12-bert_multiclass-wrapup.md)
