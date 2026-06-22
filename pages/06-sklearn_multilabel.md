**목표**: 한 샘플에 *여러* 라벨이 동시에 붙는 multi-label 문제로 확장합니다. softmax의 클래스 *상호배타* 가정이 깨지고, K개 sigmoid가 라벨마다 독립적으로 작동합니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분


## 학습 흐름

1. 🚀 **실습**: Yelp 리뷰에 항목(aspect) 키워드를 매칭해 5개 라벨(food/service/price/ambiance/location) multi-hot 합성 → `OneVsRestClassifier`로 학습
2. 📐 **Loss 분해**: 학습된 모델의 실제 예측으로 BCE 5개를 직접 합산해 본다 — multinomial CE를 못 쓰는 이유를 숫자로
3. 🔬 **해부**: multi-label 평가 지표 — subset accuracy, hamming loss, micro/macro F1
4. 🛠️ **변형**: 임계값(threshold)을 옮기면 micro/macro F1이 어떻게 움직이나
5. ⚠️ **합성의 한계** — 키워드 매칭으로 만든 라벨이 실제 라벨링과 어떻게 다른지 솔직히 짚기

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` |
| 5 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| **6 ← 여기** | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 항목 키워드 합성 | (5차원) | **sigmoid (각각 독립)** | **`BCEWithLogitsLoss` per-label** |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 5)

가장 큰 변화는 **데이터 가정** 이고, 그게 모델 선택을 강제합니다.

| 축 | Ch 5 (multi-class) | Ch 6 (multi-label) |
|---|---|---|
| 데이터 가정 | 클래스 *상호배타* (한 샘플 = 한 라벨) | **라벨 *독립* (한 샘플에 여러 라벨 가능)** |
| 라벨 형식 | int 한 개 (0-4) | **multi-hot 벡터** (예: `[1, 0, 1, 0, 1]`) |
| 모델 패러다임 | **multinomial 기본** + OvR 대안 (Ch 5 후반에서 두 방식 비교) | **OvR 만** (multinomial은 데이터 가정과 충돌) |
| Activation | softmax 한 번 (합 = 1 강제) | **per-label sigmoid** (라벨끼리 독립) |
| Loss | `CrossEntropyLoss` | **per-label `BCEWithLogitsLoss` 평균** |
| OvR 사용 방식 | K개 sigmoid → **argmax로 한 라벨 선택** | K개 sigmoid **그대로** (argmax 없음, 각자 임계값 0.5와 비교) |
| 데이터 | 별점 5클래스 | **Yelp + 항목 키워드 합성** (5개 항목) |
| 토크나이저 | TF-IDF | TF-IDF (그대로) |

### 왜 OvR이 multi-label의 자연스러운 선택인가

Ch 5에선 두 패러다임이 모두 가능했습니다.

- **multinomial**: softmax 한 번으로 K개 logit을 묶어 합=1 강제. "K개 중 정확히 하나"라는 데이터 가정과 정합.
- **OvR (대안)**: K개 *독립* sigmoid + argmax 후처리. 각 binary 모델은 독립이지만 마지막에 강제로 하나만 고르므로 결과는 상호배타.

Ch 6의 multi-label은 이 가정 자체를 깹니다 — 한 리뷰가 "음식 + 서비스 + 가격"을 동시에 다룰 수 있어요. 그러면:

- **multinomial은 부적합**: softmax가 합=1을 강제하므로 'food=0.9, service=0.85 동시 활성' 같은 분포를 *표현할 수가 없습니다*. P(food)=0.9면 나머지 합이 0.1로 강제돼 동시 활성이 수학적으로 불가능.
- **OvR은 자연스럽게 들어맞음**: K개 sigmoid가 *각자* 0/1을 결정 → 어떤 조합이든 표현 가능. argmax 후처리 단계만 빼면 그대로 multi-label.

요약: Ch 5에서 *대안* 이었던 OvR이 Ch 6에서는 *유일한 자연스러운 선택* 이 됩니다. 알고리즘(`OneVsRestClassifier`)은 그대로, 사용 방식만 "argmax로 한 라벨 선택" → "K개 출력 그대로" 로 바뀝니다.

## Loss 함수의 변화 — `BCEWithLogitsLoss` per-label

K개 라벨에 대해 BCE를 각각 계산하고 평균을 냅니다.

$$L = \frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K}\bigl[-y_{ik}\log \hat p_{ik} - (1 - y_{ik})\log(1 - \hat p_{ik})\bigr]$$

각 (샘플, 라벨) 쌍이 독립적으로 손실에 기여합니다 — 한 라벨에서 틀렸다고 다른 라벨의 손실이 변하지 않습니다. CE의 클래스 경쟁 구조와 정반대.

**숫자로 감 잡기** (K=5, 정답 multi-hot $\mathbf{y} = [1, 0, 1, 0, 1]$):

| 시나리오 | 예측 확률 $\hat{\mathbf{p}}$ | 라벨별 손실 | 평균 BCE |
|---|---|---|---|
| 잘 맞춤 | `[0.9, 0.1, 0.8, 0.2, 0.6]` | 0.105 / 0.105 / 0.223 / 0.223 / 0.511 | **0.233** |
| 균등 (baseline) | `[0.5, 0.5, 0.5, 0.5, 0.5]` | 0.693 × 5 | **0.693** |
| 정반대로 자신감 | `[0.1, 0.9, 0.1, 0.9, 0.1]` | 2.303 × 5 | **2.303** |

baseline = $\log 2 = 0.693$ — 모든 라벨에 0.5를 줄 때 (BCE에서 K=2 분포의 균등 추측과 같은 값). 학습된 모델은 이 값보다 작아야 정상.

```python
# PyTorch (Ch 12 이후, multi-label)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets.float())   # logits: (N, K), targets: (N, K) multi-hot

# sklearn (이번 챕터)
from sklearn.multiclass import OneVsRestClassifier
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X, Y_multilabel)   # Y_multilabel shape: (N, K) 0/1
```

## 토크나이저 노트

이번 챕터의 토크나이저는 **Ch 1-5와 동일한 `TfidfVectorizer`**. 입력 표현은 그대로고, 변화는 라벨 구조와 출력 헤드의 형태에 있습니다.

> **다음 챕터(Ch 7)** — Phase 1 시작: 사전학습된 **DistilBERT WordPiece** 가 처음 등장합니다. TF-IDF의 단어 단위 어휘 학습과 어떻게 다른지 비교 시작.

## Loss 한 단계 더: 학습된 모델의 실제 예측으로 BCE 분해

방금 fit한 `model_ml`이 한 샘플에 대해 어떤 손실을 만들어내는지 직접 분해합니다. 변경점 표에서 본 "**per-label BCE 평균**" 이 단순한 수식이 아니라 **실제 5개 숫자의 산수** 라는 걸 확인합니다 — 그리고 그 위에서 "왜 multinomial CE는 여기 못 쓰나"를 진짜 값으로 짚습니다.

**관찰**

- 5개 라벨이 *각자 독립적으로* 손실을 기여합니다 — 한 라벨에서 잘 맞춰도 다른 라벨에서 못 맞추면 그 영향이 그대로 평균에 더해집니다.
- 정답이 1인 라벨은 $-\log(p)$ — 예측 확률이 1에 가까울수록 손실 0에 수렴.
- 정답이 0인 라벨은 $-\log(1-p)$ — 예측 확률이 0에 가까울수록 손실 0에 수렴.
- 같은 sigmoid 출력에 대해 정답이 0이냐 1이냐에 따라 **정반대 방향** 으로 페널티가 커집니다 (대칭 구조).

### 같은 샘플을 multinomial CE로 풀려고 하면

위 샘플의 정답은 multi-hot — 여러 라벨이 동시에 1입니다. 만약 multinomial CE를 *억지로* 적용하려면 다음 두 가지 *임의 결정* 이 필요합니다.

1. **5개 활성 라벨 중 *하나만* 정답으로 골라야 함**: argmax? 첫 활성? 어느 기준이든 *임의*.
2. **그러면 나머지 활성 라벨들은 *틀린* 클래스로 학습됨**: 모델이 그 라벨에 강한 확률을 줄수록 손실이 *커짐*.

결과: 모델이 "동시 활성 패턴을 *피하려고*" 학습됩니다 — 실제 정답에서는 동시 활성이 정답인데도. 이건 *데이터 가정과 정반대 방향* 으로 학습 신호가 작동하는 셈입니다.

per-label BCE는 위 표처럼 5개 손실을 *독립적으로* 합산하므로 각 라벨이 자기 정답에만 책임을 집니다. **multi-label 데이터의 본래 구조와 정합한 유일한 선택** 인 이유가 이 산수에 있습니다.

## 합성의 한계 — 솔직한 한계 짚기

이 챕터의 학습 결과가 너무 좋아 보일 수 있습니다 (subset accuracy, micro F1 모두 매우 높음). 그 이유는 **모델이 *키워드 매칭 규칙* 자체를 학습** 하기 때문입니다 — 우리가 정한 사전을 다시 거꾸로 풀어내고 있을 뿐, 진짜 항목 추출 능력을 입증한 게 아닙니다.

실제 multi-label 문제에서 부딪히는 것들:

1. **부정·반어 무시** — `"this place is not noisy at all"` 의 'noisy'를 ambiance 활성으로 잡는 게 우리 사전의 한계. 사람이 읽으면 ambiance가 *아닌* 데도.
2. **사전 협소** — 'food'에 'sushi', 'ramen', 'pasta' 같은 구체 음식명이 빠져 있으면 그 리뷰는 food=0이 되어 버림.
3. **정답이 노이지** — 우리 라벨 자체가 진짜 정답이 아닌 휴리스틱이라, 모델 성능을 이 정답에 비교하는 건 결국 "모델이 휴리스틱을 얼마나 따라 했나"를 잴 뿐.
4. **빈 라벨**: 모든 항목이 0인 샘플도 있음 (`{n_labels_per_sample == 0).sum()` 건). 실제 multi-label 데이터에선 보통 최소 한 라벨은 보장.

**그럼 왜 합성을 쓰나?** — 학습 코드의 *형태* 와 *평가 지표 해석* 을 익히는 게 이 챕터의 목적이기 때문입니다. Ch 12 BERT multi-label에서 **같은 합성 라벨** 을 그대로 사용하므로 비교가 깔끔하게 됩니다. 진짜 multi-label 데이터(예: GoEmotions, Reuters)는 라벨이 사람 손으로 만들어져 있어 비용이 큽니다.

## Phase 0 마무리 — sklearn vs HuggingFace 미리보기

이 챕터로 sklearn 시대가 끝납니다. 다음 챕터(Ch 7)부터 등장하는 `transformers` (Hugging Face)는 *loss를 최소화한다* 는 목적은 같지만 **그 방식과 철학** 이 다릅니다. 큰 그림을 미리 잡아두면 Phase 1의 학습 코드가 낯설지 않습니다.

### 핵심 차이 한 문장

> **sklearn은 *수학 문제를 풀어준다*.**
> **HuggingFace는 *수학 문제를 푸는 과정* 을 우리가 통제한다.**

이 챕터에서 쓴 `LogisticRegression(max_iter=1000)` 은 lbfgs solver가 알아서 BCE를 최소화하는 가중치를 찾아 돌려줬습니다 — 우리는 `fit()` 한 줄과 결과만 봤어요.

HF의 `Trainer`는 학습 *과정* 을 명시합니다 — 학습률, 배치 크기, 에폭 수, 스케줄러, 평가 빈도. loss는 매 step마다 계산되어 backprop으로 가중치를 *조금씩* 옮깁니다. 같은 데이터로 학습해도 random seed가 바뀌면 결과가 미세하게 달라지는 이유.

### 한 표로 정리

| 축 | sklearn (Phase 0) | HuggingFace / PyTorch (Phase 1+) |
|---|---|---|
| **최적화 방식** | 수렴 보장 solver (lbfgs 등) 한 번 호출 → 전역 최적해 | 미니배치 SGD/Adam — 학습자가 epoch·step 통제 |
| **언제 끝나나** | 수렴 기준(`tol`) 도달 시 자동 | 사용자 지정 epoch 수 (멈출 시점 직접 결정) |
| **결정성** | convex 문제라 같은 입력엔 같은 출력 | non-convex — random seed·batch 순서에 따라 매번 미세 차이 |
| **에폭/배치 개념** | 보통 없음 — 전체 데이터 한 번에 | **핵심** — `num_train_epochs`, `batch_size` 명시 필수 |
| **loss를 직접 보나** | 거의 안 봄 (fit 후 평가만) | 매 step마다 loss 출력 + 곡선 추적 (학습이 망가지면 즉시 보임) |
| **하드웨어** | CPU, 단일 스레드 위주 | **GPU 필수** (fp16, gradient accumulation 등) |
| **loss 함수 지정** | 모델 클래스에 내장 (LogReg = log loss) | `problem_type` 자동 매핑 또는 `compute_loss` 오버라이드 |
| **모델 크기** | 수만 ~ 수십만 파라미터 | 사전학습 BERT — 6천만 ~ 수억 파라미터 |
| **학습 시간 (Yelp 5,000)** | 1초 미만 | T4 GPU에서 2-5분 |

### 코드 형태 미리보기 (Ch 9 BERT 회귀에서 본격 등장)

```python
# Phase 0 (sklearn — 우리가 한 형태)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)   # 한 줄. 수렴 기준까지 알아서 풀어줌.

# Phase 1 이후 (HuggingFace — 다음부터)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2,
)
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,             # 학습 반복 횟수
    per_device_train_batch_size=16, # 미니배치 크기
    learning_rate=2e-5,             # 학습률
    fp16=True,                      # T4에서 GPU 효율
    logging_steps=20,               # loss 곡선 출력
)
trainer = Trainer(model=model, args=args, train_dataset=..., ...)
trainer.train()   # 매 step마다 loss → backward → optimizer step
```

각 인자가 무엇을 하는지는 Ch 9에서 본격적으로 펼쳐 봅니다. 지금은 "fit 한 줄이 수십 개 인자로 펼쳐진다" 는 감만 가지면 됩니다.

### 변하지 않는 것

Loss 자체 — `BCEWithLogitsLoss`, `CrossEntropyLoss`, `MSELoss` — 는 sklearn에서 익힌 그대로 Phase 1+에서도 등장합니다. **달라지는 건 *어떻게 최소화하느냐* 의 도구뿐**. Phase 0의 직관(BCE 수치 표, OvR fit 분해, softmax 동등성 등)이 Phase 1의 BERT 학습에서도 그대로 살아 있습니다.

## 이 장의 구성

[[SubPages]]
