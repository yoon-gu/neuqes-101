## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModelForSequenceClassification(num_labels=5, problem_type="multi_label_classification")` | Ch 12 셋업에서 problem_type만 변경 → BCE per-label 자동 매핑 | Ch 14에서 메인 헤드로 그대로 |
| `sklearn.metrics.hamming_loss` | 전체 (sample × label) 위치에서 틀린 비율 | multi-label 챕터마다 |
| `precision_recall_fscore_support(..., average="micro"/"macro")` | multi-label용 F1 — micro는 라벨 합산, macro는 라벨 평균 | Ch 14·17·18 |
| `roc_auc_score(..., average="macro")` | 라벨별 AUC를 평균 | Ch 17·18 |
| `seaborn.FacetGrid + map_dataframe` | 5개 라벨에 같은 KDE를 facet으로 | 라벨이 많은 시각화에 재등장 |
| `OneVsRestClassifier(LogisticRegression())` | sklearn multi-label baseline | 비교용 |

## 체크포인트 질문

1. multi-label 문제를 *softmax + CrossEntropyLoss* 로 풀려고 하면 무엇이 잘못되나요? 수식으로 한 줄 설명할 수 있나요?
2. `num_labels=5` 가 Ch 12와 Ch 13에서 *같은 숫자* 인데 *모델이 학습하는 의미* 는 어떻게 다른가요?
3. Macro F1과 Micro F1의 차이는 무엇이고, 어느 한쪽이 *훨씬* 낮으면 무엇을 의심해야 하나요?
4. 라벨별 *공동 활성 행렬* 에서 모델 예측이 실제보다 일관되게 높은 행을 보였다면 어떤 처치가 필요한가요?

## FAQ

### Q1. (실무) Multi-label에서 threshold 0.5는 항상 옳은가요?

아닙니다. 0.5는 *기본값* 일 뿐 라벨마다 최적 threshold가 다를 수 있습니다.

- **클래스 불균형**: 라벨 활성률이 5% 인 라벨에선 0.5가 너무 보수적 — 모델이 거의 안 활성. threshold를 0.2-0.3으로 낮추면 recall이 크게 올라감.
- **F1 최적 threshold 탐색**: validation set에서 *라벨별로* 0.1-0.9 grid search → F1 최대 지점 선택.

```python
def best_threshold(probs_k, labels_k):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.1, 0.91, 0.05):
        preds_k = (probs_k >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels_k, preds_k, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1

# 라벨별로 따로
thresholds = [best_threshold(probs[:, k], labels[:, k])[0] for k in range(K)]
```

운영 환경에선 *라벨별 threshold* 를 저장해 두고 추론 시 적용.

### Q2. (이론) Multi-class를 multi-label로 풀어도 되나요? (single-label 데이터를 multi-hot 형식으로 변환)

기술적으론 됩니다. 단점이 큽니다.

| | multi-class (CE) | single-label을 multi-label (BCE) 로 |
|---|---|---|
| 라벨 간 *경쟁* 학습 | softmax가 자동으로 강제 | BCE per-label은 라벨 간 직접 신호 없음 |
| confidence 의미 | 항상 합 = 1, 명확 | 5개 라벨의 sigmoid 확률, *합 ≠ 1* |
| 추론 후처리 | `argmax` 한 줄 | `argmax(probs)` 또는 `np.where(probs > th)` |
| 학습 수렴 속도 | 빠름 (라벨 경쟁 신호 있음) | 약간 느림 |

multi-class가 *명확히 single-label* 이면 CE가 정답. multi-label로 푸는 건 *어떤 이유로 두 task를 통합해야 할 때* (예: Ch 14 보조 헤드처럼).

### Q3. (실무) Class weights를 multi-label BCE에 적용하려면?

`pos_weight` 파라미터 — `BCEWithLogitsLoss(pos_weight=...)` 가 라벨별 양성 가중치를 받습니다.

```python
import torch
from torch import nn

# 라벨별 양성 비율 → pos_weight = (negative count / positive count)
pos_count = Y_train_bin.sum(axis=0)
neg_count = len(Y_train_bin) - pos_count
pos_weight = torch.tensor(neg_count / np.maximum(pos_count, 1), dtype=torch.float).to("cuda")
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

`Trainer.compute_loss` 를 오버라이드해서 위 `loss_fn` 으로 바꾸면 됩니다. 활성률 5%인 라벨은 pos_weight ≈ 19 → 양성 샘플의 손실이 19배 가중되어 모델이 양성 예측을 더 자주 하도록 강제.

### Q4. (이론) 모델이 라벨 *간* 상관을 학습하는 메커니즘은? (loss엔 라벨 결합 항이 없는데)

핵심은 **공유 BERT 본체** 입니다. 5개 라벨의 logit이 같은 768-dim CLS hidden state $h$ 에서 *별도의 5개 가중치 행* 을 통해 나옵니다.

$$z_k = w_k^\top h + b_k, \quad k = 1, \ldots, K$$

학습 중:

- 라벨 1이 활성된 샘플에서 $w_1$ 이 $h$ 의 특정 차원을 강조하도록 학습됩니다.
- 같은 샘플에서 라벨 2도 활성됐다면 $w_2$ 도 *비슷한 차원* 을 강조하게 됩니다.
- 결과적으로 라벨 1과 2가 같이 활성되는 *입력 패턴* 에 대해 두 logit이 동시에 커지는 *간접* 결합이 생깁니다.

Loss에는 결합 항이 없지만 *gradient가 BERT 본체를 거쳐 흐를 때* 결합이 자연스럽게 학습됩니다. 이게 BERT 같은 *공유 표현* 모델이 multi-label에서 sklearn OvR 보다 강한 이유 — sklearn은 K개 LogReg가 *완전 분리* 학습되므로 이런 간접 결합이 없습니다.

### Q5. (실무) 라벨이 100개 이상인 multi-label은 BERT로 어떻게 푸나요?

**일반적 패턴**:

1. **라벨이 약 50개 이하**: `num_labels=K` + per-label sigmoid + BCE 그대로. 본 챕터 패턴.
2. **라벨이 수백~수천 개**: 헤드는 아직 가볍지만(768·K — K=1,000이면 3MB) **라벨 희소성** 이 문제. 라벨당 양성 샘플이 적어 학습 신호가 부족. 해결책:
   - **Hierarchical labels**: 라벨을 트리로 구조화해 *상위 → 하위* 단계적 분류 (예: "음식 > 한식 > 김치찌개"). 상위 노드가 하위 라벨의 양성을 합쳐 보므로 신호가 늘어남
   - **Knowledge distillation**: 큰 multi-label 모델에서 작은 모델로 distill
3. **라벨이 수만 개 이상**: extreme multi-label (XML). 헤드도 이때부터 부담 (K=10만이면 본체보다 큼). 별도 분야 — `XML-CNN`, `BERT-XMC` 등 특화 모델 사용.

영화 장르 분류 (약 30개), 기사 토픽 (약 50개) 정도면 본 챕터 패턴으로 충분.

### Q6. (실무) Multi-label에서 *클래스가 추가되면* 모델을 처음부터 다시 학습해야 하나요?

기본적으론 그렇습니다. 분류 헤드의 weight shape이 `(K, 768)` 이라 K가 바뀌면 헤드가 호환 안 됨. 단, BERT 본체는 그대로 재사용 가능.

```python
# 이 노트북은 save_strategy="no" 이므로 먼저 저장합니다
trainer.save_model("./ch13_output")

# 새 라벨 1개 추가 (K=5 → K=6)
old_model = AutoModelForSequenceClassification.from_pretrained("./ch13_output")
new_model = AutoModelForSequenceClassification.from_pretrained(
    "./ch13_output", num_labels=6, problem_type="multi_label_classification",
    ignore_mismatched_sizes=True,   # 크기가 달라진 classifier 만 새로 초기화
)
# 기존 5라벨 weight를 새 모델에 복사
new_model.classifier.weight.data[:5] = old_model.classifier.weight.data
new_model.classifier.bias.data[:5]   = old_model.classifier.bias.data
# 6번째 라벨은 random init 그대로 → fine-tune 시작
```

이러면 기존 5라벨은 *학습된 상태로 시작*, 6번째만 *처음부터 학습* . 실무에서 자주 쓰는 incremental learning 패턴.

## 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# multi-label 모델에 int 스칼라 라벨 (Ch 12 형식) 을 넣어보기
def tokenize_wrong(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # multi-hot vector 대신 첫 번째 활성 라벨의 인덱스만 (single-label 형식)
    out["labels"] = [
        next((i for i, v in enumerate(a) if v > 0), 0)
        for a in batch["aspects"]
    ]
    return out
```

힌트: `BCEWithLogitsLoss` 는 *logits 와 같은 shape의 float 텐서* 를 라벨로 받는데, 위 코드는 *(B,) int* 를 넘깁니다. shape mismatch + dtype mismatch 두 가지 에러가 동시에 날 수 있어 메시지가 길어집니다.

## 다음 챕터 예고

**Chapter 14. BERT Auxiliary Loss — 항목 분류 + 별점 보조 회귀**

- 메인 task: Ch 13의 multi-label 항목 분류 (`num_labels=5` + BCE per-label) — *완전히 동일*
- 추가: *보조 헤드* `Linear(H, 1)` 로 별점 점수 회귀 (별점 정규화 0-1)
- 손실: `L = BCE_per_label(메인) + λ · MSE(보조)` 가중합 (λ는 hyperparameter)
- `Trainer.compute_loss` 오버라이드로 두 헤드를 동시 학습
- 보조 task가 메인 task의 정확도를 *얼마나 끌어올리는지* 측정 (auxiliary loss의 정통 활용)

> **변하는 축**: 메인 task와 모델 본체는 그대로, *Loss에 보조 항이 추가* 됩니다 — Loss 축의 마지막 단계 ("BCE per-label → +Auxiliary").
