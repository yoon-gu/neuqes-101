## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModelForSequenceClassification(num_labels=7, problem_type="multi_label_classification")` | Ch 16 셋업에서 problem_type 만 변경 → BCE per-label 자동 매핑 | Ch 18 에서 메인 헤드로 그대로 |
| `datasets.Dataset.from_dict(...)` | 합성한 텍스트·라벨로 새 데이터셋 생성 | 합성 데이터 챕터에서 재등장 |
| `numpy.random.default_rng(seed)` | 재현 가능한 난수 생성기 (결합 짝짓기) | 합성 챕터마다 |
| `sklearn.metrics.hamming_loss` | 전체 (sample × label) 위치 중 틀린 비율 | multi-label 챕터마다 |
| `precision_recall_fscore_support(..., average="micro"/"macro")` | multi-label F1 — micro 는 라벨 합산, macro 는 라벨 평균 | Ch 18 |
| `roc_auc_score(..., average="macro")` | 카테고리별 AUC 의 macro 평균 | Ch 18 |
| `seaborn.FacetGrid + map_dataframe` | 7개 카테고리에 같은 KDE 를 facet 으로 | 라벨이 많은 시각화에 재등장 |

## 체크포인트 질문

1. Ch 16 과 Ch 17 의 모델이 *둘 다* `Linear(H, 7)` 헤드인데, 같은 7개 출력을 두 챕터가 *어떻게 다르게 해석* 하나요?
2. multi-label 문제를 *softmax + CrossEntropyLoss* 로 풀려고 하면 무엇이 잘못되나요? 합성 샘플 (경제+스포츠) 을 예로 한 줄 설명할 수 있나요?
3. 두 헤드라인을 `[SEP]` 로 잇는 합성 방식에서, 모델이 *두 주제를 모두 잡지 못하고 하나만 잡는* 경우는 왜 생기나요?
4. 카테고리 간 공동 활성 행렬에서 *무작위 결합* 합성이라 true co-occurrence 가 거의 균등에 가까운데, 실제 사람-annotated multi-label 데이터라면 이 행렬이 어떻게 달라질까요?

## FAQ

### Q1. (실무) Multi-label 에서 threshold 0.5 는 항상 옳은가요?

아닙니다. 0.5 는 *기본값* 일 뿐 카테고리마다 최적 threshold 가 다를 수 있습니다 (§6 의 threshold sweep 에서 확인했듯).

- **활성률 차이**: 활성률이 낮은 카테고리에선 0.5 가 너무 보수적 — threshold 를 0.3-0.4 로 낮추면 recall 이 크게 올라갑니다.
- **F1 최적 threshold 탐색**: validation set 에서 *카테고리별로* grid search → F1 최대 지점 선택.

```python
def best_threshold(probs_k, labels_k):
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.1, 0.91, 0.05):
        preds_k = (probs_k >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels_k, preds_k, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1

thresholds = [best_threshold(probs[:, k], labels[:, k])[0] for k in range(K)]
```

운영 환경에선 *카테고리별 threshold* 를 저장해 두고 추론 시 적용합니다.

### Q2. (이론) 두 헤드라인을 결합하는 합성 방식의 한계는 무엇인가요?

합성이 *무작위 결합* 이라 두 가지 한계가 있습니다.

1. **자연스러운 카테고리 상관이 사라짐** — 실제 뉴스에선 "정치+경제" 가 "스포츠+정치" 보다 훨씬 자주 같이 등장하지만, 무작위 결합은 모든 쌍을 *비슷한 확률* 로 만듭니다. 그래서 §5-2 의 true co-occurrence 가 거의 균등.
2. **두 주제가 한 문장에 *섞이지* 않고 그냥 *이어붙음*** — 진짜 multi-topic 헤드라인 (예: "삼성전자 스포츠단 창단 발표" — 경제+스포츠가 *한 문장에서 융합*) 과 달리, 결합 샘플은 `[SEP]` 로 나뉜 *두 독립 문장*. 모델이 학습하기엔 오히려 *쉬운* 편 (각 절반이 명확한 단일 카테고리 신호).

더 현실적인 합성은 *문장 수준 paraphrase* 나 *LLM 으로 multi-topic 헤드라인 생성* 이지만, 그건 입문 범위를 넘어섭니다. 핵심 — *합성 데이터의 통계적 특성은 합성 방식이 결정* 한다는 감각.

### Q3. (실무) `pos_weight` 로 카테고리 불균형을 다루려면?

`BCEWithLogitsLoss(pos_weight=...)` 가 라벨별 양성 가중치를 받습니다. 무작위 결합이라 카테고리 활성률이 KLUE-YNAT 원본 분포를 따라가 *약간 불균형* (스포츠/세계가 정치/IT 보다 많음).

```python
import torch
from torch import nn

# 라벨별 양성 비율 → pos_weight = (negative count / positive count)
pos_count = Y_train.sum(axis=0)
neg_count = len(Y_train) - pos_count
pos_weight = torch.tensor(neg_count / np.maximum(pos_count, 1), dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

활성률이 낮은 카테고리는 pos_weight 가 커져 양성 샘플의 손실이 더 가중됩니다 → 모델이 그 카테고리를 더 자주 활성하도록.

### Q4. (이론) 모델이 카테고리 *간* 상관을 학습하는 메커니즘은? (loss 엔 결합 항이 없는데)

핵심은 **공유 BERT 본체** 입니다. 7개 카테고리의 logit 이 같은 768-dim CLS hidden state $h$ 에서 *별도의 7개 가중치 행* 을 통해 나옵니다.

$$z_k = w_k^\top h + b_k, \quad k = 1, \ldots, K$$

학습 중 한 합성 샘플에서 경제+스포츠가 동시 활성됐다면 $w_{\text{경제}}$ 와 $w_{\text{스포츠}}$ 가 각자 $h$ 의 관련 차원을 강조하도록 학습됩니다. Loss 에 결합 항은 없지만 *gradient 가 BERT 본체를 거쳐 흐를 때* 결합이 간접적으로 학습됩니다. 다만 Q2 에서 봤듯 무작위 결합이라 *학습할 만한 상관 구조 자체가 약합니다*.

### Q5. (실무) 헤드라인 두 개 대신 *세 개* 를 결합하면 어떻게 되나요?

활성 라벨 개수가 평균 약 2.7개 (충돌 고려) 로 늘어 *더 어려운* multi-label 이 됩니다. 코드 변경은 작습니다 — `make_multilabel` 에서 3개 인덱스를 뽑아 3개 위치를 1 로:

```python
idx = rng.integers(0, n_src, size=3 * n_samples).tolist()
# ... 3개 짝 ca, cb, cc 의 위치를 multi-hot 에서 1 로
```

단 텍스트가 더 길어져 (`max_length` 압박) 모델이 세 신호를 분리하기 더 어렵고, F1 이 떨어집니다. 이게 *Ch 18 의 보조 task* (활성 라벨 *개수* 회귀) 가 의미 있는 이유 — "몇 개 카테고리가 활성됐는가" 를 보조로 학습하면 메인 분류가 도움을 받습니다.

### Q6. (이론) micro F1 과 macro F1 중 multi-label 에선 어느 쪽을 봐야 하나요?

둘 다 봐야 하지만 보는 *이유* 가 다릅니다.

- **micro F1** — 모든 (샘플 × 카테고리) 위치를 동등하게 세서 합산. *활성률 높은 카테고리* 의 영향이 큼. "전체적으로 라벨을 얼마나 잘 맞히나" 의 종합 점수.
- **macro F1** — 카테고리 7개의 F1 을 *단순 평균*. 활성률이 낮은 카테고리도 *동등한 가중치*. "소수 카테고리도 챙기나" 의 공정성 점수.

micro 가 높은데 macro 가 *훨씬* 낮으면 → 모델이 *다수 카테고리만 잘 맞히고 소수 카테고리를 버리는* 상태. 이때 Q3 의 `pos_weight` 나 Q1 의 카테고리별 threshold 로 처치.

### Q7. (실무) 합성 multi-label 로 학습한 모델을 실제 multi-topic 뉴스에 써도 되나요?

부분적으로 됩니다. 합성 데이터로도 모델은 *각 카테고리의 단어·표현 신호* 를 학습하므로, 진짜 multi-topic 헤드라인에서도 *기본적인 카테고리 인식* 은 작동합니다. 단 Q2 에서 짚은 한계 때문에:

- *자연스러운 카테고리 상관* 을 못 배웠으니 "정치+경제 가 흔하다" 같은 사전 지식이 없음.
- *한 문장에 융합된* multi-topic (이어붙임이 아닌) 에는 약함.

실무에선 *소량의 사람-annotated multi-label 데이터* 로 fine-tune 을 한 번 더 하면 (합성 → 진짜 데이터 2단계) 격차가 크게 줄어듭니다. 합성 데이터의 가치는 *없는 라벨을 0 에서 만드는 것* 이 아니라 *모델을 task 형태에 적응시키는 워밍업* 에 있습니다.

## 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# multi-label 모델에 int 스칼라 라벨 (Ch 16 형식) 을 넣어보기
def tokenize_wrong(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    # multi-hot 벡터 대신 첫 번째 활성 카테고리의 인덱스만 (single-label 형식)
    out["labels"] = [
        next((i for i, v in enumerate(mh) if v > 0), 0)
        for mh in batch["multi_hot"]
    ]
    return out
```

힌트: `BCEWithLogitsLoss` 는 *logits 와 같은 shape 의 float 텐서* 를 라벨로 받는데, 위 코드는 *(B,) int* 를 넘깁니다. shape mismatch + dtype mismatch 두 가지 에러가 동시에 날 수 있어 메시지가 길어집니다 (Ch 13 의 삽질과 같은 함정 — 라벨 *형식* 이 problem_type 과 일치해야 함).

## 다음 챕터 예고

**Chapter 18. 한국어 BERT Auxiliary Loss — multi-label 분류 + 활성 라벨 개수 보조 회귀**

- 메인 task: Ch 17 의 multi-label 카테고리 분류 (`num_labels=7` + BCE per-label) — *완전히 동일*
- 추가: *보조 헤드* `Linear(H, 1)` 로 *활성 라벨 개수* 를 회귀 (결합한 헤드라인이 몇 개 카테고리에 걸치는가, 정규화 0-1)
- 손실: `L = BCE_per_label(메인) + λ · MSE(보조)` 가중합 (λ 는 hyperparameter)
- `Trainer.compute_loss` 오버라이드로 두 헤드를 동시 학습
- Ch 14 (영어 auxiliary) 의 한국어 버전 — 보조 task 만 별점 회귀 → 라벨 개수 회귀로 달라짐

> **변하는 축**: 메인 task 와 모델 본체는 그대로, *Loss 에 보조 항이 추가* 됩니다 — Loss 축의 마지막 단계 ("BCE per-label → +Auxiliary").
