## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `load_dataset("klue/klue", "ynat")` | KLUE 벤치마크 YNAT (한국어 뉴스 7분류) | Ch 17 에서 같은 데이터로 multi-label 합성 |
| `ds["train"].features["label"].names` | datasets.ClassLabel 의 사람-읽는 이름 | id2label 자동 매핑에 사용 |
| `seaborn.heatmap(..., xticklabels=한국어)` | 혼동 행렬 한국어 라벨 표시 | Ch 17 도 사용 |
| `sklearn.metrics.precision_recall_fscore_support(..., average="macro")` | 클래스별 평균 metric (불균형에 강함) | 분류 챕터마다 |
| `roc_auc_score(..., multi_class="ovr")` | multi-class AUC (One-vs-Rest) | Ch 17 multi-label 평가 |

## 체크포인트 질문

1. K=7 학습 첫 step 의 loss 가 약 1.95 정도라면 모델이 무엇을 학습한 상태인가요?
2. macro F1 이 weighted F1 보다 *훨씬* 낮으면 무엇을 의심해야 하나요?
3. 혼동 행렬에서 *정치 ↔ 경제* 혼동이 많은데 *스포츠 ↔ 정치* 혼동은 거의 없는 이유는?
4. 같은 klue/bert-base 가 NSMC binary (Ch 15) 와 KLUE-YNAT 7분류 (Ch 16) 에서 *분류 헤드만* 다른데, 왜 두 task 모두 잘 동작하나요?

## FAQ

### Q1. (실무) 한국어 multi-class 데이터셋이 KLUE-YNAT 외에 어떤 게 있나요?

| 데이터셋 | 도메인 | 클래스 수 | 크기 |
|---|---|---|---|
| **KLUE-YNAT** (이번 챕터) | 뉴스 헤드라인 | 7 | 45K train |
| AI Hub 뉴스 분류 | 뉴스 본문 | 50+ | 100K+ (가입 필요) |
| 모두의 말뭉치 신문 코퍼스 | 신문 기사 | 다양 | 매우 큼 (라이선스 확인 필요) |
| Naver shopping 카테고리 분류 | 상품 설명 | 100+ | 1M+ (실무에선 흔히 다룸) |

KLUE 벤치마크의 TC(Topic Classification) task 가 곧 YNAT 입니다 — 별도 데이터셋이 아닙니다. KLUE-YNAT 가 *입문에 편한 이유* — datasets.load_dataset 한 줄, 깔끔한 라벨, 균형 분포에 가까움, 헤드라인 한 줄이라 max_length 짧음.

### Q2. (이론) 혼동 행렬에서 *대칭* 인 혼동과 *비대칭* 인 혼동은 무슨 차이인가요?

- **대칭 혼동** (예: 정치↔경제 양방향 비슷): 두 카테고리 *경계가 모호* 하다는 신호. 헤드라인 한 줄로는 *사람도 헷갈리는* 데이터.
- **비대칭 혼동** (예: 정치 → 경제 는 흔한데 경제 → 정치 는 드뭄): 모델이 *한쪽으로 편향* 됨. 가능한 원인:
  - 학습 데이터 *클래스 불균형* (한쪽이 많아 그쪽으로 답을 미는 경향)
  - *시그널 단어* 가 한쪽에 더 강하게 학습됨 (예: "정부", "정책" 이 정치 카테고리에 너무 강하게 매핑)

비대칭 혼동이 보이면 `class_weight` 적용 또는 *under-represented 클래스 oversampling* 으로 처치.

### Q3. (실무) `class_weight` 를 multi-class CE 에 적용하려면?

```python
import torch
from torch import nn

# 클래스별 빈도 → 가중치 (적은 클래스에 큰 가중)
class_counts = np.bincount(train_full["label"], minlength=len(LABEL_NAMES))
class_weights = len(train_full) / (len(LABEL_NAMES) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

KLUE-YNAT 는 *심한* 불균형이 아니라 (5K-8K 범위) class_weight 효과가 *작은 폭* — 완전 균형이 아닌 데이터에서도 보통 +1-2%p macro F1 정도 개선. 사용 결정은 *macro F1 vs accuracy* 의 격차로.

### Q4. (이론) 헤드라인 한 줄 (~30 토큰) 이 분류에 *충분* 한가요?

대부분의 경우 충분합니다 — *카테고리* 라는 task 가 *키워드 + 표현 스타일* 로 풀리고, 그 신호가 헤드라인에 *압축* 되어 들어 있어요. 예: "월드컵 한국 vs 일본 16강전 시작" → 스포츠 신호 명확.

부족한 경우:
- *제목과 본문이 정반대* 인 풍자 기사 (드뭄)
- *정치/경제 처럼 경계가 모호* 한 카테고리 (위 Q2)
- *지나치게 일반적* 인 헤드라인 ("어제 뉴스 정리")

이런 케이스엔 *기사 본문* 까지 같이 입력하면 정확도 +5-8%p 가능. 단 max_length 가 길어져 학습 비용 ↑.

### Q5. (실무) 한국어 BERT 가 *짧은 헤드라인* 에 *왜 그렇게* 잘 동작하나요?

`klue/bert-base` 의 사전학습 코퍼스에 *뉴스 + 위키* 가 큰 비중을 차지합니다. 이미 모델 weight 가 한국어 뉴스 텍스트의 *언어 분포* 를 학습한 상태에서 fine-tune 하니, *적은 데이터·짧은 학습* 으로도 좋은 성능.

영어 BERT 가 NLI 같은 *추론 task* 에서 사전학습 효과가 큰 것과 같은 원리 — *task 의 도메인이 사전학습 코퍼스와 가까울수록* fine-tune 효과가 큼.

### Q6. (실무) 클래스 수가 50, 100 으로 늘면 같은 코드가 동작하나요?

코드는 그대로 동작합니다 (`num_labels=100` 만 바꾸면 됨). 하지만 *학습 동역학* 이 달라집니다:

| K | random baseline loss | 학습 도전 |
|---|---|---|
| 7 (이번 챕터) | $\log 7 = 1.95$ | 무난 |
| 50 | $\log 50 = 3.91$ | 학습 데이터가 *클래스당 충분* 해야 |
| 100 | $\log 100 = 4.60$ | 클래스 불균형이 심하면 자주 collapse |
| 1000 | $\log 1000 = 6.91$ | hierarchical softmax 등 특수 기법 고려 |

K 가 50+ 가 되면 *클래스당 학습 샘플* 이 핵심 — 클래스당 100 개 미만이면 BERT 정확도 떨어짐. KLUE-YNAT 는 클래스당 5K-8K 라 *풍족한* 셋업.

## 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 결과가 나올까요?

```python
# K=7 인데 num_labels=2 로 모델 만들기 (Ch 15 그대로 복붙)
model_wrong = AutoModelForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=2,   # ← 잘못 (실제는 K=7)
)
# ... 같은 학습 코드 ...
```

힌트: 학습 시 `CrossEntropyLoss(logits.shape=(B, 2), targets in 0-6)` 가 호출되는데 target 값이 logits 의 클래스 차원 범위를 *벗어남* → IndexError 또는 *runtime cuda assert*. 모델 헤드가 라벨 범위와 *일치해야* 한다는 단순하지만 흔한 실수.

## 다음 챕터 예고

**Chapter 17. 한국어 BERT Multi-label — KLUE-YNAT 합성 multi-label**

- 같은 데이터·같은 모델, 단 *task 만* single-label → multi-label
- KLUE-YNAT 헤드라인 *두 개를 결합* 해 인공 multi-label 데이터 합성 (Ch 13 의 측면 합성과 비슷한 패턴)
- `num_labels=7` 그대로, 단 `problem_type="multi_label_classification"` 으로 전환
- Activation: per-label sigmoid, Loss: per-label `BCEWithLogitsLoss`
- Ch 13 의 한국어 버전. 한 헤드라인이 *여러 카테고리에 걸칠 수 있는* 상황을 시뮬레이션

> **변하는 축**: Phase 2 안에서 *task 차원* (single-label → multi-label). 모델·토크나이저·hyperparams 는 그대로.
