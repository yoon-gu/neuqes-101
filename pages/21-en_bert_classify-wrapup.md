## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.BertForSequenceClassification` | encoder + 분류 head, 분류 fine-tune 전용 | Ch 23 한국어 분류 |
| `BertForSequenceClassification(config)` (random init) | pretrained weight 없이 모델 생성 | Ch 23 같음 (random baseline) |
| `model.bert.load_state_dict(other.bert.state_dict())` | 본체만 통째로 옮기는 in-memory 헤드 교체 | Ch 23 같음 |
| `transformers.BertForMaskedLM` (재등장) | MLM 사전학습 (Ch 20 압축 재현) | Ch 22 한국어 MLM |
| `load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")` | Wikitext-103 일반 도메인 코퍼스 로드 (MLM 용) | Ch 22 한국어 Wikipedia 패턴과 대칭 |
| `load_dataset("fancyzhx/yelp_polarity")` | Yelp 이진 분류 데이터 (분류 fine-tune 용) | Ch 23 NSMC 와 대칭 |
| `sklearn.metrics.precision_recall_fscore_support(..., average="binary")` | 이진 분류 metric 한 묶음 | Ch 23 동일 |
| `sklearn.metrics.roc_auc_score` | AUC | Ch 23 동일 |

## 체크포인트 질문

1. `BertForMaskedLM` 과 `BertForSequenceClassification` 둘 다 *내부에 같은 `BertModel`* 을 갖습니다. 두 모델 사이에서 *어떤 파라미터* 가 이어지고 *어떤 파라미터* 가 새로 학습되나요?
2. MLM 학습 첫 step 의 loss 가 약 10.33 인 반면, 분류 fine-tune 첫 step 의 loss 는 약 0.693 입니다. 이 *4 배 차이* 가 모델의 학습 어려움 차이를 의미하나요? (힌트: K=vocab_size vs K=2)
3. Ch 21 의 작은 BERT 가 Ch 10 의 DistilBERT 보다 *낮은 정확도* 를 보입니다. 이 격차가 (a) *모델 크기* 차이 (약 10M vs 약 66M), (b) *사전학습 데이터 양* 차이 (약 70만-100만 토큰 vs 약 33억 토큰) 중 어느 쪽 영향이 클까요? 둘 다 *위키 일반 도메인 → Yelp transfer* 의 같은 패턴이라 *도메인 정합* 변수는 통제됨. 추가 실험으로 어떻게 (a) 와 (b) 를 분리할 수 있나요?
4. *MLM 3 epoch* 와 *random init* baseline 의 정확도 차이가 매우 작거나 (예: 1-2%p) 거꾸로 *random 이 더 높게* 나올 가능성이 있나요? 어떤 상황에서 그럴 수 있을까요? (힌트: 한국어 Ch 23 부록 참조)

## FAQ

### Q1. (실무) Ch 20 의 체크포인트를 디스크에 저장해두고 Ch 21 에서 `from_pretrained` 로 로드하면 안 되나요?

가능합니다 — 그게 *프로덕션 흐름* 입니다. Colab 의 단일 세션에서 Ch 20 → Ch 21 이어 돌리거나, 또는:

```python
# Ch 20 마지막 셀
mlm_model.save_pretrained("./ch20_small_bert_mlm")
tokenizer.save_pretrained("./ch20_small_bert_mlm")

# Ch 21 첫 셀
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "./ch20_small_bert_mlm",
    num_labels=2,
)
```

`from_pretrained` 가 `BertForMaskedLM` 체크포인트를 자동으로 분류 모델로 *헤드 변환* 합니다 — MLM head 는 버려지고 분류 head 가 random init 으로 부착됨 (warning 메시지가 *그 일이 일어났음* 을 알려줌).

이번 챕터가 *MLM 학습 코드를 직접 재현* 한 이유는 **노트북 self-contained** — Colab 세션이 끊겨도 노트북 하나만으로 끝까지 돌릴 수 있게.

### Q2. (이론) 왜 *task 도메인 (Yelp) 자체로* 사전학습하지 않고 일반 위키 (Wikitext-103) 로 사전학습하나요? 같은 도메인에서 학습한 게 transfer 가 더 유리하지 않나요?

같은 도메인 사전학습 → 같은 도메인 fine-tune 은 *domain-adaptive pretraining* (DAPT) 에 더 가깝습니다 — *사전학습이 이미 task 도메인의 표현* 을 학습한 상태에서 분류만 얹는 셈. *정직한 transfer 측정* 이 아닙니다.

```python
# 본 챕터 (Ch 21) 흐름 — 원본 BERT 와 같은 정신
# 일반 위키 (Wikitext-103) 로 MLM 사전학습  ← 다른 도메인
# Yelp 리뷰(식당·업체) 로 분류 fine-tune    ← task 도메인
# -> "일반 표상이 다른 도메인에도 적용되는가?" 의 시험 (= 원본 BERT 의 GLUE/SQuAD 흐름)

# 만약 Yelp text MLM 사전학습 → Yelp 분류였다면
# -> DAPT 우위 때문에 사전학습 효과가 *부풀려* 측정됨
# -> Ch 10 (DistilBERT, Wiki+BookCorpus 사전학습 → Yelp 분류) 과의 비교가 unfair
```

본 챕터의 *Ch 10 vs Ch 21* 비교가 *공정* 한 이유 — 둘 다 *일반 도메인 위키 사전학습 → Yelp 분류* 의 같은 패턴이라 *사전학습 규모* (3000-5000배) 와 *모델 크기* (6배) 만 변수. 만약 Ch 21 이 Yelp 사전학습이었다면 *어느 효과가 격차를 만들었는지* 분리할 수 없었을 것.

> Ch 22-23 (한국어 위키 → NSMC 영화 리뷰) 도 *같은 대칭 패턴*. *일반 도메인 → 다른 도메인 transfer* 가 사전학습-fine-tune 패러다임의 *진짜 메시지*.

### Q3. (이론) MLM 본체 가중치를 *완전히 같은* hyperparams 에 옮겼는데 왜 분류 정확도가 *작은 폭* 만 개선되나요?

**작은 데이터의 한계** — 사전학습 코퍼스 (Wikitext-103 paragraphs 5K = 약 70만-100만 토큰) 자체가 *학습할 언어 분포* 가 좁습니다. DistilBERT 의 사전학습 코퍼스 (약 33억 토큰) 와 비교하면 약 3000-5000배 작은 데이터로 같은 일을 한 것.

```python
# 더 많은 사전학습으로 격차 줄이기 (T4 30분 룰 안에서)
MLM_EPOCHS = 3                     # 1 -> 3
# 또는 데이터 늘리기 — N_MLM_TRAIN 만 늘려도 효과 큼
mlm_train_raw = (
    raw_train.filter(is_good).shuffle(seed=SEED).select(range(20000))
    .remove_columns([c for c in raw_train.column_names if c != "text"])
)
```

`N_MLM_TRAIN = 20000` + `MLM_EPOCHS = 1` 정도가 T4 30분 룰 안에서 최대치. 그래도 *대규모 사전학습* 의 격차는 메우기 어렵습니다 — *데이터 규모 자체의 가치* 가 진짜 BERT 의 비밀.

### Q4. (실무) MLM 사전학습이 fine-tune 정확도에 *해가 되는* 경우가 있나요?

드물지만 있습니다. 두 가지 시나리오:

```python
# (1) 사전학습이 *과도* — 작은 데이터에 너무 오래 학습해 본체가 overfitting
MLM_EPOCHS = 20   # 5K paragraphs 에 20 epoch -> 데이터에 과적합

# (2) downstream 과 *분포가 너무 동떨어진* 사전학습 — 본 챕터는 위키 -> Yelp 라 어느 정도 차이는 있어도 같은 영어
# 다른 경우: Wikipedia 영어로 MLM 사전학습한 모델로 한국어 분류 fine-tune (Q5 참고)
```

(1) 의 경우 본체가 *특정 문장 패턴에 과적합* 되어 fine-tune 일반화가 떨어질 수 있습니다. *MLM eval loss* 와 *MLM train loss* 의 격차로 진단 — 격차가 커지면 overfitting.

(2) 의 경우 본체가 *downstream 도메인에 무관한 표상* 을 학습. 토크나이저까지 안 맞으면 거의 동작 안 함 (Ch 19 의 cross-language 실험). 본 챕터처럼 *위키 영어 → Yelp 영어* 는 *도메인은 달라도 같은 언어 + 어휘 일부 공유* 라 transfer 가 작동.

### Q5. (이론) Ch 20 의 *작은 사전학습 모델* 을 Ch 21 의 *한국어 분류* (Ch 23 예고) 에 쓰면 어떻게 되나요?

거의 동작 안 합니다 — 토크나이저가 *영어 WordPiece* 라 한국어 문장의 대부분이 `[UNK]` 가 되거나 자모 단위로 쪼개집니다. *임베딩 자체* 가 한국어 의미를 모르는 상태.

```python
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
korean_text = "이 영화 정말 재밌었어요"
print(en_tokenizer.tokenize(korean_text))
# 예상: ['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]'] 또는 자모 분해된 형태
```

그래서 Ch 22-23 에서는 *한국어 토크나이저* (`klue/bert-base` WordPiece) 로 처음부터 다시 사전학습 + fine-tune. 토크나이저와 사전학습은 *언어 단위로 매칭* 되어야 함.

### Q6. (실무) 학습 곡선을 보면 fine-tune 후반에 loss 가 *다시 올라가는* 경우가 있는데 정상인가요?

자주 보이는 현상입니다. 원인 셋:

```python
# (1) 학습률 스케줄 — Trainer 기본은 linear warmup → linear decay
# epoch 후반의 LR 이 *너무 작아져* 미세 진동만 남음. loss 가 0.2-0.4 진동.

# (2) overfitting — train loss 는 떨어지는데 eval loss 가 올라감
# 일찍 멈추거나 weight_decay 늘리거나 데이터 늘리기

# (3) batch sample 의 불운 — 어려운 sample 이 몰린 batch 가 loss 를 튀게 만듦
# logging_steps 가 작으면 (50 step 등) 잘 보임. logging_steps=200 으로 평탄화도 가능
```

이번 챕터의 짧은 학습 (2 epoch ≈ 600 step) 에서는 (1) 이 흔합니다. *eval_loss* 가 *epoch 별로* 어떻게 움직이는지 (Trainer 가 `eval_strategy="epoch"` 으로 자동 측정) 가 진짜 신호.

### Q7. (이론) DistilBERT (Ch 10) 가 BERT 의 *distilled* 버전인데 왜 Ch 21 의 *작은 BERT* (scratch) 와 정확도가 차이 나나요? 둘 다 작지 않나요?

DistilBERT 와 Ch 21 의 작은 BERT 는 *축약 방법론* 이 전혀 다릅니다.

| 차원 | DistilBERT | Ch 21 small BERT |
|---|---|---|
| 출발점 | *이미 학습된* BERT-base 의 *지식 증류* (teacher → student) | random init 부터 시작 |
| 사전학습 | MLM + *teacher 의 soft label* + *hidden state 정합* | MLM only (이번 챕터 3 epoch) |
| 학습 코퍼스 | BERT-base 와 같음 (약 33억 토큰, 일반 도메인) | Wikitext-103 2K paragraphs × 3 epoch (약 30만-50만 토큰 효과적, 일반 도메인) |
| 파라미터 | 66M (BERT-base 110M 의 *60%*) | 10M (BERT-base 의 *9%*) |
| 사전학습 시간 | TPU 수일 | T4 10분 |

DistilBERT 가 *이미 똑똑한 큰 BERT 가 만든 답* 을 학습 신호로 받기 때문에 *훨씬 작은 데이터로도 같은 수준* 으로 학습됩니다. 우리는 *teacher 없이 처음부터* 학습하는 셋업 — *맨바닥에서 작은 모델로 사전학습이 어디까지 가능한가* 의 한계 실험.

### Q8. (실무) Ch 21 의 모델을 더 키우면 (예: hidden=512, layer=8) 어떻게 되나요?

T4 메모리 안에서는 가능합니다. 정확도 변화 추정:

| 모델 크기 | 파라미터 | T4 학습 시간 (MLM 3 epoch + cls 2 epoch) | 예상 accuracy |
|---|---|---|---|
| hidden=128, layer=2 | 약 5M | 약 5분 | 65-72% |
| **hidden=256, layer=4 (이번 챕터)** | **약 10M** | **약 1분** | **약 65% (실측 0.6490)** |
| hidden=384, layer=6 | 약 20M | 약 30분 | 78-88% (T4 30분 한계) |
| hidden=512, layer=8 | 약 35M | 약 45분 | 80-90% (T4 30분 룰 위반) |
| hidden=768, layer=12 (BERT-base) | 약 110M | 수일 | 90%+ (대규모 사전학습 데이터 필요) |

데이터 양을 안 늘리면 모델만 키워도 *정확도 한계* 가 빨리 옵니다. *모델 키움 + 데이터 키움* 이 같이 가야 하고, 그 정점이 *DistilBERT/BERT* 의 *대규모 사전학습*. Ch 21 의 *작은 모델 + 작은 데이터* 는 *원리 학습용 toy 셋업* 의 정의.

## 다음 챕터 예고

**Chapter 22. 작은 BERT 직접 사전학습 — 한국어 MLM (scratch, 한국어 Wikipedia)**

- Ch 20 의 영어 패턴을 한국어로 그대로: 작은 BertConfig + `klue/bert-base` 토크나이저 (가져옴) + **한국어 Wikipedia paragraphs MLM** (일반 도메인)
- 토크나이저·데이터만 한국어로 바뀜. 본체 구조·MLM 셋업은 *완전히 같음*
- Ch 22 → Ch 23 (한국어 NSMC 분류) 흐름은 본 챕터 (Ch 20 → Ch 21, 영어) 와 *대칭* — 둘 다 *일반 위키 사전학습 → 다른 도메인 분류 transfer* (영어는 Yelp 리뷰(식당·업체), 한국어는 NSMC 영화 리뷰)

> **변하는 축**: Phase 3 안에서 *언어* (영어 → 한국어). 모델 구조·학습 셋업·*일반 도메인 → task 도메인 transfer 패턴* 은 동일.

본 챕터 (Ch 21) 에서 본 *일반 위키 사전학습 + 다른 도메인 분류 transfer* 의 격차 패턴이 Ch 23 에서 *한국어 환경* 에서도 같은 결을 그리는지 확인하는 게 Phase 3 의 마지막 메시지.
