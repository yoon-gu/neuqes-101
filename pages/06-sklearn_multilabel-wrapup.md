## 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `sklearn.multiclass.OneVsRestClassifier` | K개 독립 binary 분류기를 묶고 multi-label 모드 자동 감지 | Ch 13 BERT multi-label에서 같은 패러다임을 BERT로 |
| `sklearn.metrics.hamming_loss` | 라벨별 평균 오답 비율 | Ch 13에서도 평가 지표로 |
| `sklearn.metrics.f1_score(average="micro" / "macro")` | multi-label F1 집계 방식 | — |
| `sklearn.metrics.classification_report` | 라벨별 precision/recall/F1 한 번에 | — |

## 체크포인트 질문

1. multi-class와 multi-label은 라벨 구조와 활성화 함수에서 어떻게 다른가요?
2. multi-label에서 subset accuracy가 너무 엄격한 지표가 되는 이유는?
3. multi-label baseline BCE는 왜 $\log 2 = 0.693$ 인가요?
4. 임계값을 0.5에서 0.3으로 낮추면 micro F1과 macro F1은 어느 방향으로 움직이는 게 일반적이고, 그 이유는 무엇인가요?

## FAQ

### Q1. (이론) Multi-class와 Multi-label은 정확히 어떻게 다른가요?

| 항목 | Multi-class | Multi-label |
|---|---|---|
| 한 샘플의 라벨 수 | 정확히 1개 | 0개 이상 (K개 가능) |
| 라벨 구조 | 정수 인덱스 (예: 3) | multi-hot 벡터 (예: [1, 0, 1, 0, 1]) |
| 활성화 | softmax (합 = 1) | per-label sigmoid (라벨끼리 독립) |
| Loss | CrossEntropy | per-label BCE 평균 |
| 가정 | 클래스 *상호배타* | 라벨 *독립* |
| 예시 | 별점 1-5 중 하나, 뉴스 7카테고리 | 영화 장르(로맨스+코미디), 영화 태그 |

### Q2. (이론) micro F1과 macro F1 중 뭘 봐야 하나요?

데이터 분포에 따라 다릅니다.

- **micro F1**: 라벨 빈도와 무관하게 *전체 예측 풀* 의 정확도를 봄. 빈도 큰 라벨이 점수를 끌고감 — 우리 데이터에서 food가 매우 자주 활성된다면 food 성능이 좋으면 micro F1도 좋아 보임.
- **macro F1**: 라벨별 F1을 단순 평균. *드문 라벨* 도 동등하게 평가 — location 같은 빈도 작은 라벨에서 못하면 점수가 깎임.

**언제 무엇:**
- 라벨 빈도가 비슷 + 모든 라벨 동등 중요 → 둘이 비슷, 그냥 macro 보면 됨.
- 빈도 차이가 크고 *드문 라벨도 잘 잡고 싶다* → macro F1 (소수 클래스 보호).
- 전체 시스템의 평균적 정확도가 핵심 → micro F1.

실무에선 둘 다 보고 차이를 해석하는 게 표준입니다.

### Q3. (실무) 모든 라벨이 0인 샘플은 어떻게 처리하나요?

세 가지 접근.

1. **그대로 둔다**: BCE가 처리 가능. 각 라벨이 0이라는 신호를 학습. 가장 흔한 방식.
2. **버린다**: "라벨이 없으면 학습 신호가 없다"는 가정. 합성 데이터에서 빈 라벨 비율이 너무 크면 고려.
3. **"기타" 라벨 추가**: K+1번째 라벨을 만들어 "어느 항목도 안 맞음"을 명시적으로 표현. 데이터셋 설계 결정.

```python
# 옵션 (b): 빈 라벨 샘플 제거
mask_nonempty = Y.sum(axis=1) > 0
Y_clean = Y[mask_nonempty]
df_clean = df[mask_nonempty]
```

이번 챕터는 (a) 그대로 두기 — 합성 데이터에서 "어느 키워드도 없는 짧은 리뷰"는 자연스러운 부분이라.

### Q4. (실무) 키워드 매칭이 너무 단순한 것 같은데 실무에선 어떻게 하나요?

세 가지 접근이 일반적.

1. **사람 라벨링**: 가장 정확하지만 비용 큼. crowdsourcing, 도메인 전문가, 사내 라벨러.
2. **약지도 학습 (weak supervision)**: 우리 챕터처럼 휴리스틱 라벨로 *부트스트랩* → 학습된 모델로 *재라벨링* → 사람이 검수. Snorkel 같은 프레임워크.
3. **거대 모델 보조**: GPT-4/Claude 같은 LLM에게 라벨 지시문을 줘서 자동 태깅. 비용은 사람 라벨링보다 훨씬 저렴, 정확도는 도메인에 따라 다름.

이번 합성은 (2)의 가장 단순한 형태입니다 — 한 번 매칭하고 끝. 실무에선 매칭 결과를 한 번 학습한 모델로 다시 라벨을 매겨 사전을 보강하는 사이클을 돌립니다.

### Q5. (이론) hamming loss는 정확히 무엇을 측정하나요?

평균 라벨별 오답 비율입니다.

$$\text{Hamming loss} = \frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K} \mathbb{1}[\hat y_{ik} \neq y_{ik}]$$

직관: K개 라벨 중 평균적으로 몇 개가 틀렸나. K=5에서 hamming loss 0.1 = 평균 0.5개 라벨 틀림 = 한 샘플의 5개 라벨 중 평균 0.5개 잘못됨.

**Subset accuracy와의 관계**: subset accuracy는 "5개 라벨 *전부* 맞아야 1점"이라 매우 보수적, hamming은 *부분 점수* 를 줌. 일반적으로 hamming이 더 너그럽게 나옵니다.

### Q6. (실무) 임계값을 라벨마다 다르게 줄 수 있나요?

네, 자주 합니다. 라벨 빈도가 매우 다를 때 (예: food는 활성률 60%, location은 10%) 한 임계값으로는 둘 다 잘 잡기 어렵습니다.

```python
from sklearn.metrics import f1_score

# 라벨별 임계값 sweep으로 F1 최대화
best_thresholds = []
for k in range(K):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 17):
        f1 = f1_score(Y_test[:, k], (proba_ml[:, k] >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    best_thresholds.append(best_t)

print(f"라벨별 최적 임계값: {dict(zip(ASPECTS, best_thresholds))}")
```

⚠️ 주의: 위 코드는 *test set* 으로 임계값을 정해 보여준 것 — 실무에선 *별도 검증 데이터셋* 으로 임계값을 정하고 test에는 적용만 해야 합니다 (안 그러면 test에 누설).

### Q7. (실무) sklearn 에서 OvR 을 어떻게 구현하나요?

**한 줄 답**: `OneVsRestClassifier(LogisticRegression())` 로 wrap. multi-class·multi-label 양쪽에 통하는 *표준 패턴* 입니다.

`OneVsRestClassifier` wrapper 의 특징:

| 비교 항목 | `OneVsRestClassifier(LogisticRegression())` |
|---|---|
| 받는 Y 형식 | **1D + 2D 둘 다** (multi-class 또는 multi-label) |
| 내부 저장 | K개 *별도* estimator 를 `.estimators_` 리스트로, 각자 `coef_` shape `(1, V)` |
| base classifier | **임의 binary 분류기** 가능 (`SVC`, `RandomForestClassifier`, ...) |
| multi-label 지원 | ✅ |

```python
# multi-class 와 multi-label 모두 같은 패턴
OneVsRestClassifier(LogisticRegression()).fit(X, y_1d)        # ✅ multi-class
OneVsRestClassifier(LogisticRegression()).fit(X, y_multihot)  # ✅ multi-label

# 보너스: base 분류기를 바꾸는 것도 자유
from sklearn.svm import SVC
OneVsRestClassifier(SVC(probability=True)).fit(X, y_multihot)  # ✅
```

이 커리큘럼은 Ch 5 에선 *모던 `LogisticRegression()` (multinomial 자동)* 으로 multi-class softmax 를 하고, Ch 6 의 multi-label 에서는 `OneVsRestClassifier` 로 *명시적* 으로 K개 binary 학습을 표현합니다.

## 삽질 코너 (선택)

같은 데이터를 multi-class로 잘못 풀면 어떻게 될까요? *가장 강하게 활성된 항목 하나* 만 정답으로 골라 (argmax) multinomial LogReg를 학습합니다.

```python
# 항목이 하나라도 활성된 샘플만 사용
mask_nonempty = Y.sum(axis=1) > 0
y_pseudo_class = Y[mask_nonempty].argmax(axis=1)   # 0-4 사이 정수, "가장 강한 항목"
texts = df["text"][mask_nonempty]

X_pseudo = tfidf.transform(texts)
# 모던 sklearn — multi-class 데이터면 multinomial(softmax) 자동
model_pseudo = LogisticRegression(max_iter=1000)
model_pseudo.fit(X_pseudo[:4000], y_pseudo_class[:4000])
acc_pseudo = (model_pseudo.predict(X_pseudo[4000:]) == y_pseudo_class[4000:]).mean()
print(f"강제로 multi-class화 한 경우 accuracy: {acc_pseudo:.4f}")
```

힌트: 한 리뷰에 여러 항목이 동시에 있을 때 *하나만* 정답으로 골라 학습하면 정보가 사라집니다. 정답이 임의 선택이라 모델이 어느 라벨을 골라야 할지 모호해지고, multi-label 결과보다 정보가 적은 모델이 됩니다.

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
| **모델 크기** | 수만-수십만 파라미터 | 사전학습 BERT — 6천만-수억 파라미터 |
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

## 다음 챕터 예고

**Phase 1 시작 — Chapter 7. BERT 첫 만남 (`pipeline`)**

- sklearn 시대 끝, **`transformers` 라이브러리** 가 처음 등장
- `pipeline("sentiment-analysis")` 한 줄로 사전학습된 DistilBERT 추론
- `pipeline` 안에서 일어나는 3단계 (tokenizer / model / post-processing) 직접 풀어보기
- **첫 WordPiece 등장** — 같은 문장이 TF-IDF와 어떻게 다른 단위로 쪼개지는지 비교
- 학습 없음 (추론만) — `Trainer`는 Ch 9에서 본격 등장
