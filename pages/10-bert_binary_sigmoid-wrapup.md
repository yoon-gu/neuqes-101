## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModelForSequenceClassification(num_labels=1, problem_type="multi_label_classification")` | num_labels=1 + multi_label로 BCE 자동 매핑 | Ch 12-13에서 multi-hot 라벨로 재사용 |
| `sklearn.metrics.precision_recall_fscore_support` | 이진 분류 지표 한 묶음 | Ch 11·15·17에서 동일 |
| `sklearn.metrics.roc_auc_score` | AUC 계산 (확률 임계값 무관) | 분류 챕터마다 사용 |
| `numpy 1/(1+exp(-x))` | sigmoid 직접 구현 (모델 logit → 확률) | Ch 7에서 본 동일 패턴 |
| `seaborn.kdeplot(..., hue=, fill=True, common_norm=False)` | 라벨별 부드러운 분포를 *각자* 정규화해 모양만 비교 | 분류 챕터마다 동일 패턴으로 재등장 |

## 체크포인트 질문

1. `num_labels=1` 인데 왜 `problem_type="single_label_classification"` 이 아닌 `"multi_label_classification"` 으로 두나요?
2. `BCEWithLogitsLoss` 의 *Logits* 가 의미하는 바는? sigmoid를 따로 적용하지 않는 이유는?
3. 학습 후 *확률 공간* 에서 라벨별 분포가 양 끝에 압착되는 이유는? 같은 분포를 *logit 공간* 에서 보면 무엇이 달라지나요? (4-1, 4-2 그래프 비교)
4. 같은 binary 분류를 sklearn `LogisticRegression()` 으로 풀 때(Ch 3)와 BERT로 풀 때(Ch 10), accuracy 차이가 어디서 오나요?

## FAQ

### Q1. (실무) `num_labels=1` 일 때 `problem_type="single_label_classification"` 으로 두면 안 되나요?

안 됩니다. `single_label_classification` 은 CrossEntropyLoss를 적용하는데, CE는 `num_labels >= 2` 를 가정합니다. `num_labels=1` 로 두고 CE를 적용하면 logit이 단 하나라 softmax가 항상 1을 출력하고, loss가 항상 0이 되어 학습이 안 됩니다.

방식 A의 정수 라벨이 0/1 두 개이지만 *출력은 1차원 logit* 이므로 — 이 형태를 `Trainer` 가 이해하게 하려면 `multi_label_classification` 으로 두어야 BCE가 자동 적용됩니다.

### Q2. (이론) 라벨을 길이 1 multi-hot 벡터(`[0.0]`, `[1.0]`)로 두는 이유는?

`BCEWithLogitsLoss` 가 *logits 와 같은 shape의 float 텐서* 를 라벨로 받기 때문입니다.

```python
# logits.shape = (batch, num_labels=1)
# labels.shape = (batch, num_labels=1)  ← 같은 shape여야 함
loss = BCEWithLogitsLoss()(logits, labels)
```

라벨을 scalar(`float(b)`)로 두면 batching 시 shape가 (batch,)가 되어 (batch, 1) logits와 안 맞습니다. `[float(b)]` 한 번 감싸 length-1 list로 두면 (batch, 1) shape이 자동으로 만들어집니다.

### Q3. (실무) `compute_metrics` 의 `eval_pred.predictions` 는 어떤 형태인가요?

`AutoModelForSequenceClassification` 의 출력은 *logits* 입니다 (sigmoid 적용 전).

- 방식 A (이번 챕터, num_labels=1): shape `(batch, 1)`. flatten 후 sigmoid 적용 → 확률.
- 방식 B (Ch 11, num_labels=2): shape `(batch, 2)`. softmax 적용 → 두 확률.

`compute_metrics` 안에서 우리가 sigmoid·argmax·threshold 같은 후처리를 직접 합니다.

### Q4. (이론) sklearn LogReg(Ch 3)와 BERT(Ch 10)의 accuracy 차이는 어디서?

세 가지 layer에서 차이가 옵니다.

1. **단어 표현**: TF-IDF는 단어 독립 벡터, BERT는 문맥 attention. `"not bad"` vs `"bad"` 구분이 BERT만 가능.
2. **모델 capacity**: TF-IDF + LogReg는 ~10K개 가중치, BERT는 67M개. 표현력 6,700배 차이.
3. **사전학습**: BERT는 위키피디아·BookCorpus로 미리 학습돼 일반 언어 지식이 모델에 인코딩됨.

다만 Yelp 별점 같은 *단어 빈도* 가 강한 신호인 task는 sklearn도 90% 이상 잘 맞춰서, 차이가 *극적이지 않은* 경우도 있습니다. 차이가 명확히 드러나는 task는 부정·반어가 많은 sentiment 데이터(SST-2)나 NLI(자연어 추론)입니다.

### Q5. (실무) 같은 데이터로 두 번 학습하면 결과가 똑같나요?

**거의 같지만 미세한 차이가 있습니다**. 이유:

- random seed (`seed=42`) 고정해도 *CUDA 비결정성* 이 남음 (cuDNN의 일부 알고리즘이 floating point 연산 순서를 비결정적으로 처리).
- DataLoader의 shuffle, dropout, layer init은 seed로 통제되지만 GPU 부동소수 연산 자체가 결정적이지 않음.

완전히 결정적으로 만들려면 `torch.use_deterministic_algorithms(True)` 같은 옵션이 있지만 속도가 느려집니다. 실무에선 *seed 여러 개로 학습 후 평균* 이 일반적.

### Q6. (실무) 학습 도중 학습률을 어떻게 조정하나요?

`Trainer` 는 기본으로 *linear warmup → linear decay* 스케줄러를 적용합니다. `lr_scheduler_type` 으로 바꿀 수 있습니다.

```python
TrainingArguments(
    ...,
    lr_scheduler_type="cosine",   # cosine decay (BERT 큰 모델에서 흔함)
    warmup_ratio=0.1,             # 첫 10%를 warmup
)
```

작은 데이터·짧은 학습이면 default linear가 무난. 학습 중 LR을 직접 모니터링하려면 `report_to="wandb"` 같은 트래커 필수.

## 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# 라벨을 scalar로 두고 학습 시도
def tokenize_wrong(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [float(b) for b in batch["binary"]]   # scalar (감싸지 않음!)
    return out
```

힌트: `BCEWithLogitsLoss(logits, labels)` 에서 logits.shape는 `(B, 1)` 인데 labels.shape가 `(B,)` 가 되어 broadcasting 또는 shape mismatch 에러가 납니다. 정확히 어떤 메시지가 뜨는지 확인해 보세요.

## 다음 챕터 예고

**Chapter 11. BERT Binary 방식 B — softmax + CrossEntropyLoss**

- 같은 Yelp 이진화 데이터, 같은 BERT 본체
- `num_labels=2` + `problem_type="single_label_classification"` (BERT 표준)
- Activation은 softmax, Loss는 `CrossEntropyLoss`
- 학습 후 *방식 A의 저장된 결과* 와 직접 비교 — 두 방식이 거의 같은 확률 분포를 만들어내는지 확인 (Ch 4의 sklearn 동등성을 BERT에서 다시)
- 학습된 가중치 비교, 두 방식의 prediction agreement 측정
