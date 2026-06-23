## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModel.from_pretrained(...)` | 분류 헤드 없이 BERT 본체만 로드 — 메인·보조 헤드를 직접 부착 | Phase 3 토크나이저 학습엔 등장 안 함 (Ch 19 부터는 본체보다 어휘 자체에 집중) |
| 커스텀 `nn.Module` (KoBertMultiTask) | 본체 공유 + 두 헤드 명시 정의 — multi-task 정통 패턴 | Ch 24 이후 GPT 챕터 의 task-specific head 패턴과 연결 |
| `Trainer.compute_loss` 오버라이드 + `lambda_aux` 인자 | 자동 매핑이 못 다루는 *복합 loss* + λ 동적 주입 | λ grid search 패턴 |
| 커스텀 `AuxCollator` | input_ids 외 *추가 라벨* (n_active) 도 batch 에 같이 담기 | Ch 14 와 같은 패턴, 보조 신호 변형마다 재사용 |
| `remove_unused_columns=False` | 모델 시그니처와 무관하게 모든 컬럼 통과 | custom collator 패턴마다 |
| `SequenceClassifierOutput` | Trainer 호환 출력 형식 (loss + logits) — 커스텀 모델에서도 표준 dict 반환 | 커스텀 모델 패턴마다 |
| `r2_score`, `np.corrcoef` | 회귀 보조 metric (R², Pearson r) | 회귀 결합 task 마다 |

## 체크포인트 질문

1. Ch 14 (영어 별점 보조) 와 Ch 18 (한국어 활성 개수 보조) 의 *변경된 축* 은 무엇인가요? *한 가지 축* 원칙 관점에서 어느 쪽이 더 "loss 축 변화" 에 가까운가요?
2. `n_active` 가 메인 multi-hot 벡터의 *합* 이라는 점이 보조 task 로서 *유리한 점* 과 *불리한 점* 을 각각 한 줄로.
3. `AutoModelForSequenceClassification` 대신 `AutoModel + 커스텀 nn.Module` 로 간 이유는? 어떤 상황에서 자동 매핑이 부족한가요?
4. λ=0.1 을 기본값으로 잡은 근거는? (메인 BCE 와 보조 MSE 의 *크기 자체* 가 어떻게 다른가)

## FAQ

### Q1. (실무) `n_active` 외에 어떤 보조 task 를 시도할 수 있나요?

KLUE-YNAT 합성 데이터에서 *공짜로* 얻을 수 있는 보조 신호:

| 보조 task | 라벨 형식 | 예상 효과 |
|---|---|---|
| 헤드라인 길이 (토큰 수) | float (정규화) | 약함 — 길이는 카테고리와 상관 약함 |
| 두 원본 헤드라인의 *주제 유사도* | float [0, 1] | 중간 — 유사 주제 결합 vs 이질 결합 구분 |
| 두 카테고리의 *id 차이* `abs(c_A - c_B)` | int | 약함 — id 자체가 의미 없음 |
| **원본 단일 카테고리 예측** (combined 가 아닌 *각 절반* 의 카테고리) | int | 강함 — 메인과 직접 관련, 추천 |

원본 카테고리 예측을 보조로 쓰려면 `make_multilabel` 에서 `(c_A, c_B)` 를 *순서 라벨* 로 보존하고 보조 헤드를 `Linear(H, K) × 2` 로 두 개 만들면 됩니다 (multi-task 가 *3-task* 가 됨 — multi-label + 2개 single-label).

### Q2. (이론) `n_active` 가 메인의 *합* 이라는 게 왜 *보조로 쓰면 의외로 약한* 신호인가요?

직관적으로는 "메인을 잘 풀면 합도 잘 풂" 인데, 역방향 ("합을 잘 풀면 메인도 잘 풂") 의 정보량이 작기 때문입니다.

- 메인이 잘 풀린 모델 → `n_active` 도 잘 추정 (정의상 합이니까).
- 그러나 *보조만 잘 학습* 된 모델 → "이 헤드라인이 1개 vs 2개 카테고리에 걸친다" 만 알지 *어느* 카테고리인지의 정보는 없음. 표현 공간이 카테고리 *식별* 방향이 아니라 *개수* 방향으로 발전.
- BCE per-label 의 gradient 가 *어느* 카테고리를 활성할지 직접 학습 신호. MSE on n_active 의 gradient 는 *몇 개* 인지만 학습 신호.

따라서 `n_active` 가 메인의 *함수* 이긴 하지만 *전사 가능한 inverse 가 없는 함수* 라 보조로서의 추가 정보량이 제한적. Ch 14 의 별점은 항목 분류와 *부분* 상관이지만 *완전히 다른 방향* 의 정보 (감성) 라 BERT 표상에 *추가* 차원을 만듭니다 — 그래서 보조 효과 자체는 별점 쪽이 클 가능성.

### Q3. (실무) `AutoModel + nn.Module` 패턴이 `model.aux_head = nn.Linear(...)` 한 줄 부착보다 권장되는 경우는?

| 상황 | 권장 패턴 |
|---|---|
| 보조 헤드 1개, 메인은 표준 분류 | 한 줄 부착 (Ch 14 패턴) — 간결 |
| 보조 헤드 2개+, 또는 메인 헤드도 비표준 | `nn.Module` 정의 (Ch 18 패턴) — 명시성 |
| layer-wise lr 차등 (BERT 본체는 작은 lr, 헤드는 큰 lr) | `nn.Module` — `named_parameters` 로 그룹 분리 쉬움 |
| 헤드별 dropout 등 미세 조정 | `nn.Module` — `__init__` 에서 한 곳에 정의 |
| BERT 본체 일부 layer freeze | 둘 다 가능, 명시 정의가 가독성 더 좋음 |

```python
# nn.Module 패턴이면 named_parameters 분리가 깔끔
optimizer_grouped = [
    {"params": [p for n, p in model.named_parameters() if n.startswith("bert.")],   "lr": 2e-5},
    {"params": [p for n, p in model.named_parameters() if n.startswith("cls_head")],"lr": 1e-4},
    {"params": [p for n, p in model.named_parameters() if n.startswith("count_head")],"lr": 1e-4},
]
optimizer = torch.optim.AdamW(optimizer_grouped)
```

### Q4. (이론) 보조 loss 가 *MSE* 대신 *MAE (L1)* 면 학습 양상이 어떻게 달라지나요?

`n_active` 는 1 또는 2 정수라 분포가 *binary 같음*. MAE 와 MSE 의 차이:

- **MSE**: gradient 가 잔차에 *비례* — 큰 오차에 큰 gradient. 평균값 ($\sim$ 1.86) 으로 수렴.
- **MAE (L1)**: gradient 가 잔차의 *부호* 만 (크기 일정 ±1). 중앙값 (= 2, 빈도 6/7) 으로 수렴.

따라서 MAE 로 바꾸면 보조 헤드가 *항상 2 를 예측* 하게 되어 (중앙값 = 2) 학습 신호가 거의 없습니다. binary-같은 회귀에선 MSE 가 더 적절.

또 다른 선택: `n_active - 1` 을 *binary 분류* 로 풀어 `BCEWithLogitsLoss` 적용 — 1 vs 2 를 0 vs 1 로 매핑. 회귀보다 분류가 더 자연스러운 형식이지만, *연속적 보조 신호* 라는 multi-task 의 전통적 셋업에선 회귀가 일반적.

```python
# 보조를 binary 분류로 — n_active in {1, 2} → {0, 1}
aux_logits = self.count_head(cls).squeeze(-1)   # (B,)
aux_target = (n_active - 1).float()             # 1→0, 2→1
l_aux = F.binary_cross_entropy_with_logits(aux_logits, aux_target)
```

### Q5. (실무) 학습 중 보조 loss 가 *증가* 하면 어떻게 진단하나요?

`logging_steps=50` 으로 학습 곡선 (train loss) 만 보면 *combined loss 만* 출력됩니다. 보조 loss 만 분리해 추적하려면 `compute_loss` 안에서 직접 log:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    inputs = {**inputs, "lambda_aux": self.lambda_aux}
    outputs = model(**inputs)
    loss = outputs.loss
    # 학습 중 보조 loss 따로 로깅 (커스텀 추가)
    if self.state.global_step % 50 == 0 and not return_outputs:
        # 모델 forward 안에서 self.last_count_pred 가 저장됨 — 따로 다시 계산
        with torch.no_grad():
            cls = model.bert(**{k: v for k, v in inputs.items()
                                if k in ("input_ids", "attention_mask", "token_type_ids")}
                            ).last_hidden_state[:, 0, :]
            count_pred = model.count_head(cls).squeeze(-1)
            l_aux_only = F.mse_loss(count_pred, inputs["n_active"].float()).item()
        print(f"step {self.state.global_step}  main+aux={loss.item():.4f}  aux_only={l_aux_only:.4f}")
    return (loss, outputs) if return_outputs else loss
```

보조 loss 가 *상승* 한다면:
- λ 가 너무 작아 보조 학습이 *메인 방향에 휘둘림* — λ 키워 보조 학습 강화.
- 또는 메인과 보조가 *충돌 방향* — λ 줄이거나 보조 제거.

### Q6. (이론) Ch 14 와 Ch 18 의 결과 (delta) 패턴이 *다를* 거라 예상되는 이유는?

| 측면 | Ch 14 (영어, 별점) | Ch 18 (한국어, n_active) |
|---|---|---|
| 보조 신호의 *입력 의존도* | 높음 — 사용자가 *직접* 매긴 감성 평가 | 낮음 — 합성 규칙의 기계적 함수 (1/7 확률) |
| 보조 신호의 *메인과의 독립 차원* | 다른 차원 (감성 vs 항목) | 같은 차원의 함수 (합) |
| 보조 정답의 분포 | 1-5 별점 (5 카테고리) | 1 또는 2 (이항 같음) |
| 보조 정답 예측 난이도 | 중간 — 진짜 회귀 | 매우 쉬움 — 상수 평균만 예측해도 RMSE 작음 |
| 예상 메인 delta | 약 양수 (별점 신호가 항목 표상 보강) | 약 0 (n_active 신호가 추가 정보 적음) |

따라서 quick 모드에선 Ch 18 delta 가 *더 작게* 나올 가능성. 그래도 셋업 자체는 *멀티태스크 학습의 정통 패턴* 을 한국어 환경에서 검증하는 가치가 있고, FAQ Q1 처럼 *더 유용한 보조 task* 로 확장하는 출발점.

### Q7. (실무) Phase 2 (한국어, Ch 15-18) 가 끝났습니다. Phase 3 에서 토크나이저를 *직접 학습* 하는 이유는?

Ch 1-18 모두 *사전학습 토크나이저* (sklearn TF-IDF 토큰화, BERT WordPiece) 에 의존했습니다. Phase 3 (Ch 19-23) 는 이 의존을 끊고 *어휘 자체를 코퍼스에서 학습*:

- **Ch 19**: BPE / WordPiece / Unigram 알고리즘을 직접 돌려 어휘 만들기 → 토큰화가 *데이터에 따라 어떻게 달라지는지* 직관.
- **Ch 20**: 학습한 토크나이저로 *작은 BERT 를 처음부터* 사전학습 → 사전학습 의존 없는 경험.

> Phase 3 가 클라이맥스인 이유 — Ch 1 부터 따라온 "🔤 토크나이저 노트" 가 *외부 도구의 사용법* 이었다면 Phase 3 는 *그 도구 자체를 만드는 단계*. 토크나이저를 직접 만들고 나면 Ch 1-18 의 모든 토큰화 노트를 *다시 읽었을 때* 보이는 풍경이 달라집니다.

## 삽질 코너 (선택)

다음 두 가지 흔한 함정:

**1. `remove_unused_columns=True` (기본값) 로 두기**

```python
training_args = TrainingArguments(
    ...,
    remove_unused_columns=True,   # ← 잘못 (default)
)
```

Trainer 가 model.forward 시그니처를 검사해 *맞지 않는 컬럼은 제거*. `n_active` 가 시그니처에 있긴 하지만 자동 검사가 실패할 때 (e.g. 커스텀 모델 시그니처 변경 시) `n_active` 가 사라져 `compute_loss` 안에서 None 이 됩니다. 안전상 `False` 권장.

**2. `count_pred` 를 모델 attribute 에 저장 안 하기**

```python
# KoBertMultiTask.forward 안에서
return SequenceClassifierOutput(loss=loss, logits=main_logits)
# count_pred 를 어디에도 저장 안 함 → eval 단계에서 보조 metric 측정 불가
```

`SequenceClassifierOutput` 은 `loss`/`logits`/`hidden_states`/`attentions` 만 표준 필드. `count_pred` 를 추가하려면 dataclass 를 직접 정의하거나, 본문처럼 `self.last_count_pred = ...` 에 저장해 eval 단계에서 꺼냅니다. 후자가 간단하지만 *멀티 GPU 학습 시 race condition* 가능 — 운영 코드는 정식 dataclass 정의 권장.

## 다음 챕터 예고 — Phase 3 시작 (클라이맥스)

**Chapter 19. 토크나이저 직접 학습 — BPE / WordPiece / Unigram**

- Phase 1-2 영어·한국어 모두 *사전학습 토크나이저* 를 그대로 썼습니다. Ch 19 는 그 의존을 끊고 *어휘를 코퍼스에서 직접 학습*.
- `tokenizers` 라이브러리로 BPE, WordPiece, Unigram 세 알고리즘을 같은 코퍼스에 적용해 *어휘 차이* 비교.
- 한국어 vs 영어 코퍼스에서 학습한 토크나이저의 *토큰 길이 분포* 가 어떻게 다른지 — Ch 1 부터 추적해 온 토크나이저 시각의 완성.

> **Phase 2 마무리** — Ch 15-18 을 통해 한국어 BERT 의 binary·multi-class·multi-label·auxiliary 4 가지를 다 익혔습니다. Phase 3 는 한 발 더 내려가 *어휘 구성* 자체에 도전 — 사전학습 모델에 *완전히 의존하지 않는* 경험.

> **변하는 축**: 모델·loss·task 가 아니라 *토크나이저 그 자체* 가 학습 대상. 입력 표현이 만들어지는 *가장 아래 단계* 로 내려갑니다.
