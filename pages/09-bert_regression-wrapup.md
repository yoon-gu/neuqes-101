## 이번 챕터에 등장한 라이브러리·함수

### `transformers` 학습 도구

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModelForSequenceClassification.from_pretrained(..., num_labels=1, problem_type="regression")` | 회귀 헤드를 가진 모델 자동 구성 | Ch 10-13에서 num_labels·problem_type만 바꾸어 재사용 |
| `Trainer` | 학습 루프 + 평가 + 로깅 + 체크포인트 자동화 | 모든 Phase 1·2 학습 챕터의 기본 |
| `TrainingArguments` | 학습 하이퍼파라미터 묶음 | num_epochs / batch / lr / fp16 등 인자 동일 |
| `Trainer.train()`, `.evaluate()`, `.predict()` | 학습 / 평가 / 추론 호출 | 모든 학습 챕터 |

### Trainer가 자동으로 해주는 일

- **DataCollatorWithPadding** 생성 (`tokenizer` 인자 보고)
- 학습 루프 (epoch 반복, batch 단위 forward/backward/optimizer step)
- 평가 (`eval_strategy` 에 따라)
- fp16 mixed precision (`fp16=True` 옵션 보고)
- 로깅 (`logging_steps`)
- 체크포인트 (`save_strategy`)
- gradient clipping (기본 1.0)
- learning rate scheduler (기본 linear warmup → linear decay)

### `compute_metrics` 함수 시그니처

```python
def compute_metrics(eval_pred) -> dict:
    preds, labels = eval_pred       # numpy arrays
    return {"metric_name": value, ...}   # 결과는 eval_metric_name 으로 출력
```

## 체크포인트 질문

1. `num_labels=1` 과 `problem_type="regression"` 두 인자가 `Trainer` 동작에 어떤 영향을 주나요?
2. 학습 중 VRAM이 모델 가중치만 있는 상태보다 더 큰 이유는 무엇인가요? (옵티마이저·gradient·activation)
3. sklearn `LinearRegression` 의 정규방정식과 BERT의 Adam optimizer는 같은 MSE를 최소화하는데도 학습 시간·결정성이 왜 그렇게 다른가요?
4. `compute_metrics` 함수의 입력 `eval_pred` 는 어떤 형태인가요? 회귀와 분류에서 어떻게 달라지나요?

## FAQ

### Q1. (실무) 학습 중 loss가 nan이 됩니다. 어떻게 하나요?

가장 흔한 원인이 fp16 수치 오버플로우입니다. 해결 순서:

1. `fp16=False` 로 두고 다시 시도. nan이 사라지면 fp16이 원인.
2. 학습률을 낮춥니다 (`learning_rate=1e-5` 또는 `5e-6`).
3. 그래도 안 되면 `gradient_clipping` 을 더 작게 (`max_grad_norm=0.5`).
4. 입력 데이터에 비정상값(빈 문자열, 너무 긴 시퀀스)이 있는지 확인.

T4는 fp16만 지원하고 bf16이 안 됩니다. fp16이 자주 nan을 일으키면 fp32로 가는 게 가장 안전한 fallback입니다.

### Q2. (실무) `Trainer.train()` 한 줄로 GPU 사용률이 낮은데 어떻게 올리나요?

세 가지 흔한 원인.

1. **batch_size가 작음**: T4에서 DistilBERT는 max_length=128, batch_size=32까지 무난. `per_device_train_batch_size` 를 키워보세요.
2. **DataLoader 워커 부족**: `dataloader_num_workers=2` 또는 4로 늘려 CPU에서 토크나이저 처리가 GPU를 기다리지 않게.
3. **`fp16=True` 로 메모리 여유 확보**: 같은 VRAM에 더 큰 batch를 담을 수 있음.

```python
training_args = TrainingArguments(
    ...,
    per_device_train_batch_size=32,
    fp16=True,
    dataloader_num_workers=2,
)
```

### Q3. (실무) 학습 중간에 끊겼는데 이어서 학습하려면?

`save_strategy="epoch"` 으로 체크포인트를 저장해 두면 다음과 같이 이어 학습할 수 있습니다.

```python
trainer.train(resume_from_checkpoint="./output/checkpoint-500")
# 또는 가장 최근 체크포인트 자동 탐지:
trainer.train(resume_from_checkpoint=True)
```

이번 챕터는 학습이 짧아 `save_strategy="no"` 로 두었습니다. Ch 14 같은 긴 학습이나 실무 프로젝트에선 `save_strategy="epoch"` + `save_total_limit=2` 가 표준 패턴입니다.

### Q4. (이론) BERT의 `[CLS]` 토큰이 회귀 출력을 어떻게 만들어내나요?

DistilBERT의 마지막 layer hidden state는 shape `(batch, seq_len, 768)` 입니다. 분류·회귀 헤드는 그중 *첫 번째 토큰* 의 hidden state(`[CLS]` 위치)를 가져와 `Linear(768, num_labels)` 에 통과시킵니다.

```python
# AutoModelForSequenceClassification 내부 (단순화)
hidden_states = self.distilbert(input_ids).last_hidden_state  # (B, L, 768)
cls_hidden = hidden_states[:, 0]                              # (B, 768)
logits = self.classifier(cls_hidden)                          # (B, num_labels)
```

`[CLS]` 위치의 hidden state는 사전학습 단계부터 *전체 문장의 의미를 모으는 자리* 로 학습됐습니다 (attention을 통해). 그래서 분류·회귀 헤드를 `[CLS]` 위치에 붙이는 게 자연스럽고, BERT 표준 관행이 됐습니다.

### Q5. (이론) 사전학습된 BERT를 가져다 쓰는 게 sklearn보다 왜 더 잘 되나요?

세 가지 이유.

1. **단어 독립 가정 탈피**: TF-IDF는 `"not bad"` 와 `"bad"` 를 구분 못 합니다. BERT는 attention으로 `"not"` 과 `"bad"` 의 *조합* 을 학습합니다.
2. **사전학습으로 얻은 일반 지식**: 위키피디아·BookCorpus로 학습한 단어 의미·문법·상식이 본체에 인코딩돼 있고, 우리는 그 위에 task-specific 헤드만 미세 조정.
3. **분포 표현(distributed representation)**: 768차원 hidden state로 단어·문장을 벡터로 표현하니 미묘한 의미 차이도 거리로 잡힘.

다만 *모든* 데이터에서 BERT가 sklearn을 압도하는 건 아닙니다. Yelp 별점은 단어 빈도가 강한 신호라 sklearn도 꽤 잘하고, 데이터가 작거나(수백 건) feature가 명확한 task(분야 키워드 분류)는 sklearn이 BERT보다 빠르고 가성비 좋습니다.

### Q6. (실무) `output_dir` 에 뭐가 저장되나요?

`save_strategy` 가 `"no"` 가 아니면:

- `checkpoint-{step}/` — 모델 가중치, 옵티마이저 상태, scheduler 상태, RNG state (재현 가능한 재개)
- `pytorch_model.bin` 또는 `model.safetensors` — 모델 가중치
- `config.json` — 모델 config (Ch 7 부록 참고)
- `tokenizer.json`, `vocab.txt` 등 — 토크나이저 (`tokenizer=...` 인자가 있을 때)
- `training_args.bin` — 학습 인자 dump
- `trainer_state.json` — loss/metric 로그

Colab에서 `output_dir` 를 Drive 경로로 잡으면 세션이 끊겨도 보존됩니다 (`output_dir="/content/drive/MyDrive/runs/ch09"`).

## 삽질 코너 (선택)

다음 코드를 돌려보고 어떤 에러가 나는지 보세요.

```python
# label dtype 실수 — 정수로 두고 학습 시도
def tokenize_int(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = batch["label"]   # float이 아닌 int!
    return out

train_int = train_ds.select(range(100)).map(tokenize_int, batched=True).remove_columns(["text", "label"])
```

힌트: 회귀 task에서 라벨이 int면 `MSELoss` 가 squared error 계산은 하지만 dtype 경고나 미묘한 문제가 생길 수 있습니다. `problem_type="regression"` 은 라벨이 *float* 이라고 가정합니다. `label2id`/`id2label` 도 자동 안 만들어집니다.

회귀 라벨은 *항상 float* 으로 변환하는 게 안전합니다.

## 다음 챕터 예고

**Chapter 10. BERT Binary 방식 A — sigmoid+BCE**

- Ch 4 (sklearn) 에서 본 동등성의 BERT 버전 시작
- `num_labels=1`, sigmoid, `BCEWithLogitsLoss` 로 Yelp 이진화 학습
- Ch 9의 `Trainer` 골격을 그대로 재사용

**Chapter 11. BERT Binary 방식 B — softmax+CE** 가 그 다음에 이어집니다 (BERT 표준). 두 챕터의 `predict_proba` 가 거의 일치하는지 비교가 핵심.
