## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `Trainer.compute_loss` 오버라이드 | 자동 매핑이 못 다루는 *복합 loss* 직접 계산 | Ch 18 한국어 auxiliary 에서 다시 |
| `model.aux_head = nn.Linear(...)` 한 줄 추가 | 표준 BERT 모델에 보조 헤드 동적 부착 | Ch 18 |
| `output_hidden_states=True` | 마지막 layer hidden 까지 받아 보조 헤드 입력으로 사용 | 보조 헤드 패턴마다 |
| `remove_unused_columns=False` | model.forward 시그니처에 없는 컬럼(aux_labels)을 자동 제거 안 함 | custom collator 패턴마다 |
| 커스텀 `DataCollator` | input_ids 외 *추가 라벨* 도 batch에 같이 담기 | Ch 18 |
| `r2_score`, `np.corrcoef` | 회귀 task 보조 metric (R², Pearson r) | 회귀 결합 task 마다 |

## 체크포인트 질문

1. λ를 0.01에서 1까지 grid search한다고 할 때 *작은 λ* 와 *큰 λ* 가 각각 어떤 학습 양상을 만드는지 한 줄로 설명하세요.
2. `remove_unused_columns=False` 를 빠뜨리면 어떤 에러가 나나요? `aux_labels` 가 어디로 사라지는지 추적해 보세요.
3. 보조 헤드의 파라미터가 ~770개에 불과한데도 모델이 *공유 본체* 학습에 영향을 주는 이유는?
4. 메인 metric의 delta가 *음수* 로 나왔다면 무엇을 의심해야 하고 어떻게 처치하나요?

## FAQ

### Q1. (실무) λ는 어떻게 정하나요? 그냥 1.0으로 두면 되나요?

1.0 은 흔한 출발점으로 알려져 있지만, 이 챕터 셋업에서는 λ=1 에서 메인이 무너집니다 (§9).
손실 종류가 다르면 **1 보다 훨씬 작은 쪽부터** 훑어야 합니다.

```python
# 권장 grid search (validation set 위에서 수행)
for lam in [0.01, 0.03, 0.05, 0.1, 0.3, 1.0]:
    trainer = AuxTrainer(..., lambda_aux=lam)
    trainer.train()
    metrics = trainer.evaluate()
    print(f"lambda={lam}: macro_f1={metrics['eval_macro_f1']:.4f}")
```

흔한 패턴:
- 메인 task가 분류, 보조가 회귀 → λ=0.05-0.3 (이 챕터 부록 스윕의 sweet spot 은 **λ=0.05** — BCE+MSE 조합은 1 보다 작은 λ 에서 최적인 경우가 많음)
- 메인이 복잡하고 보조가 *미세* 보조 → λ=0.01-0.1
- 메인과 보조가 비슷한 중요도이고 **두 손실의 스케일도 비슷할 때** → λ=1.0
  (이 챕터처럼 BCE + MSE 로 스케일이 다르면 해당하지 않습니다 — §9 참고)

또 다른 패턴: **uncertainty weighting** — λ를 *학습 가능한 파라미터* 로 두고 모델이 직접 결정하게 함 (Kendall et al. 2018).

### Q2. (이론) `outputs.loss` 는 BCE per-label 평균인데 왜 *우리가 따로* 계산 안 하나요?

`AutoModelForSequenceClassification` 의 forward 가 `problem_type="multi_label_classification"` 와 `labels` 를 보고 *자동으로* `BCEWithLogitsLoss(logits, labels)` 를 계산해 `outputs.loss` 에 담기 때문입니다.

```python
# transformers 내부 (개념적)
class DistilBertForSequenceClassification:
    def forward(self, ..., labels=None):
        logits = self.classifier(self.pre_classifier(hidden))
        loss = None
        if labels is not None:
            if self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits, labels)
            elif self.config.problem_type == "regression":
                loss = F.mse_loss(logits.squeeze(), labels.float())
        return SequenceClassifierOutput(loss=loss, logits=logits, ...)
```

우리는 *메인 loss는 자동으로 받고, 보조 loss만 직접 계산해 더하면* 됩니다. 깔끔.

### Q3. (실무) 보조 헤드의 가중치는 사전학습이 안 되어 있는데 그래도 잘 학습되나요?

네. 이유:

1. *작은 분류 헤드* 는 보통 사전학습 없이 random init 부터 시작해도 fine-tune 동안 충분히 빠르게 학습됩니다 (몇백 step 안에).
2. 보조 헤드 위에 가중치가 ~770개라 데이터에 *과적합* 할 위험도 적습니다.
3. BERT 본체(67M)는 이미 표상이 풍부해 작은 head가 *읽어내기만* 하면 됨.

대조적으로 *BERT 본체를 random init 부터* 학습하려면 수백 GB 데이터가 필요합니다. 사전학습 + 작은 head fine-tune 패턴이 강력한 이유.

### Q4. (이론) 보조 task가 메인과 *반대 방향* 신호면 학습이 망가지나요?

네, 정확히 그렇습니다. 예시:

- 메인: 항목 분류 (Ch 13 의 5개 항목 multi-label 셋업 그대로)
- 보조: *반대로* 라벨링된 별점 (1★=좋음, 5★=나쁨 — 라벨링 실수 시나리오)

이 경우 두 task의 gradient가 BERT 본체에서 *반대 방향* 으로 끌어당겨 학습이 *발산* 하거나 *느려집니다*. 진단 신호:

- 두 task의 train loss 가 *둘 다 정체* (서로 상쇄)
- λ 를 키울수록 메인 metric 이 *떨어짐*

해결: 보조 task가 메인과 같은 방향 신호인지 *간단한 sklearn baseline* 부터 확인. 별점이 항목 분류와 양의 상관이면 보조로 쓸 만함.

### Q5. (실무) 보조 task로 어떤 신호를 쓰는 게 좋나요?

좋은 보조 task의 조건:

| 좋은 조건 | 예시 |
|---|---|
| *메인과 양의 상관* | 항목 분류 ↔ 별점, 감성 분류 ↔ 이모지 사용 |
| *학습 데이터가 *공짜로* 있음* | 메타데이터(별점, 작성일, 길이) — 라벨링 비용 0 |
| *연속적이고 안정적인 신호* | float regression, 순서형 점수 |
| *메인보다 *덜 복잡* 한 task* | 회귀 < 분류, 단어 vs 문장 |

피해야 할 보조:

- 메인보다 *복잡한* task (예: 분류에 *생성* 보조) — 학습 비용 폭증
- *드물게 등장하는* 라벨 — 학습 신호 부족
- *중복* 라벨 (메인과 거의 같은 정보) — 추가 정보 없음

### Q6. (실무) 보조 task가 *test 환경에서는 정답이 없는* 라벨이면 어떻게 활용하나요?

학습 시에만 보조 라벨 사용, 추론 시에는 메인 head만 사용 — *학습 정규화* 용. 이게 auxiliary loss 의 *정통 사용 패턴* 입니다.

```python
# 학습 시: 별점이 학습 데이터에 있음 → 보조 학습
trainer.train()   # main + aux 둘 다 학습

# 추론 시: 별점 없음 → 메인 head만 사용
with torch.no_grad():
    out = model(input_ids=..., output_hidden_states=False)
    main_probs = torch.sigmoid(out.logits)   # 항목 예측만
    # aux_head 는 *호출 안 함*
```

훈련 데이터에만 있는 *부가 신호* (운영 시엔 사라지는 메타데이터)를 학습 시 활용하는 좋은 트릭.

## 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# remove_unused_columns 를 True (기본값) 로 두기
training_args = TrainingArguments(
    ...,
    remove_unused_columns=True,   # ← 잘못 (default 가 True 임)
)
trainer = AuxTrainer(...)
trainer.train()
```

힌트: Trainer 가 `aux_labels` 를 *자동 제거* 한 뒤 우리 `compute_loss` 안에서 `inputs.pop("aux_labels")` 가 KeyError. `aux_labels` 가 어디서 사라지는지 추적해 보면 학습 inputs 가 model.forward 시그니처와 맞춰지는 구조를 이해하게 됩니다.

## 다음 챕터 예고 — Phase 2 시작

**Chapter 15. BERT 한국어 Binary — NSMC**

- Phase 1 영어(DistilBERT) → Phase 2 한국어(klue/bert-base) 전환
- 데이터: 네이버 영화 리뷰 (NSMC) 이진 분류 (긍정/부정)
- 셋업: `num_labels=2` + `problem_type="single_label_classification"` (Ch 11과 같은 표준 binary 셋업)
- 변하는 축: *언어 + 데이터 + 토크나이저* (영어 WordPiece → 한국어 WordPiece). 모델 크기는 비슷, 셋업도 같음.
- 회귀 챕터는 *생략* (영어 Ch 9 에서 다뤘으므로). Phase 2는 Binary 부터 시작.

> **Phase 1 마무리** — Ch 7-14를 통해 BERT 분류·회귀·multi-label·auxiliary loss의 기본 5가지를 다 익혔습니다. Phase 2는 같은 패턴을 *한국어 데이터* 위에서 압축적으로 재방문 — 토크나이저가 어떻게 다른지, 한국어 특유의 학습 어려움이 무엇인지 확인.
