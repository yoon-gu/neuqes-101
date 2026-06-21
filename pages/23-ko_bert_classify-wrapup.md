## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 비고 |
|---|---|---|
| `transformers.BertForSequenceClassification` | encoder + 분류 head, 분류 fine-tune 전용 | Ch 15 / Ch 21 동일 |
| `model.bert.load_state_dict(other.bert.state_dict())` | 본체만 통째로 옮기는 in-memory 헤드 교체 | Ch 21 동일 |
| `transformers.BertForMaskedLM` (재등장) | MLM 사전학습 (Ch 22 압축 재현) | Ch 20 / Ch 22 동일 |
| `load_dataset("wikimedia/wikipedia", "20231101.ko")` | 한국어 Wikipedia 일반 도메인 코퍼스 로드 (MLM 용) | Ch 22 와 동일 |
| `pandas.read_csv(NSMC_URL, sep="\t")` | GitHub raw TSV 직접 다운로드 (분류 fine-tune 용) | Ch 15 와 동일 NSMC 패턴 |
| `sklearn.metrics.precision_recall_fscore_support(..., average="binary")` | 이진 분류 metric 한 묶음 | Ch 15 / Ch 21 동일 |
| `sklearn.metrics.roc_auc_score` | AUC | Ch 15 / Ch 21 동일 |

## 체크포인트 질문

1. `BertForMaskedLM` 과 `BertForSequenceClassification` 둘 다 *내부에 같은 `BertModel`* 을 갖습니다. 두 모델 사이에서 *어떤 파라미터* 가 이어지고 *어떤 파라미터* 가 새로 학습되나요? (Ch 21 의 같은 질문을 한국어 환경에서 재확인)
2. MLM 학습 첫 step 의 loss 가 약 10.37 인 반면, 분류 fine-tune 첫 step 의 loss 는 약 0.693 입니다. 이 차이가 모델의 학습 어려움 차이를 의미하나요? (힌트: K=vocab_size 약 32,000 vs K=2)
3. Ch 23 ours 가 Ch 15 보다 *낮은 정확도* 를 보입니다. 이 격차가 (a) *모델 크기* 차이 (약 10M vs 약 110M), (b) *사전학습 데이터 양·도메인 다양성* 차이 (약 50만-80만 토큰 위키 vs 약 8.4B 토큰 위키+뉴스+댓글) 중 어느 쪽 영향이 클까요? 둘 다 *한국어 일반 도메인 → NSMC transfer* 의 같은 패턴이라 *도메인 정합* 변수는 통제됨. 추가 실험으로 어떻게 (a) 와 (b) 를 분리할 수 있나요?
4. *사전학습 효과의 순 측정* — 한국어 환경에서는 *얕은 일반 도메인 사전학습* 이 *random init* 보다 우위인지 *비슷한지* 또는 *역전 (negative transfer)* 인지를 직접 확인하려면 어떤 실험이 필요할까요? 영어 Ch 21 의 패턴 (위키 → Yelp transfer) 과 비교해 한국어 NSMC 환경의 특수성은 무엇인가요? (부록 [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb) 에서 직접 측정·분석)

## FAQ

### Q1. (실무) 한국어 NSMC 가 영어 Yelp (Ch 21) 보다 어려운가요? 같은 5K/1K 인데 정확도가 더 낮게 나옵니다.

여러 요인이 겹쳐 *살짝 더 어렵습니다*.

| 차원 | Yelp polarity (Ch 21) | NSMC (Ch 23) |
|---|---|---|
| 평균 문장 길이 | 약 100-200자 (여러 문장 묶음) | **약 10-50자** (한 줄 리뷰) |
| 문장 구조 | 격식·반격식 혼합 | **구어체·이모티콘·맞춤법 흔들림** |
| 라벨 노이즈 | 적음 (별점 기반 자동화) | **약 3-5%** 추정 (수기 라벨링) |
| 데이터 양 (원본) | 56만 train | 15만 train |

```python
# Yelp(Ch 21) vs NSMC(이 챕터) 평균 토큰 길이 비교
yelp_lens = [len(tokenizer(t)["input_ids"]) for t in ds_train_full[:1000]["text"]]      # Ch 21 Yelp 텍스트
nsmc_lens = [len(tokenizer(t)["input_ids"]) for t in df_train["document"][:1000]]       # 이 챕터 NSMC 리뷰
# Yelp: 약 100-150 토큰, NSMC: 약 20-30 토큰
```

NSMC 의 *짧은 한 줄* 은 분류 신호가 *한두 단어에 집중* 됩니다 (`명작`, `시간 낭비`, `감동`). 모델이 *문맥 이해* 보다 *키워드 매칭* 에 가까워져, 사전학습이 얕은 작은 BERT 에는 더 불리합니다 — Ch 15 의 `klue/bert-base` 같은 *대규모 사전학습 + 비격식 코퍼스 포함* 모델이 진가를 발휘하는 영역.

### Q2. (이론) `klue/bert-base` 가 약 110M params 인데 우리 작은 BERT 약 10M 으로 따라잡을 수 있나요? 격차가 얼마나 본질적인가요?

**완전히 따라잡기는 어렵습니다** — 본 챕터의 2-way 비교 (+ 부록의 random init baseline) 가 그 정량입니다.

| 차원 | klue/bert-base | 우리 작은 BERT |
|---|---|---|
| 본체 파라미터 | 약 110M | 약 10M (11x 작음) |
| 사전학습 코퍼스 | 약 8.4B 토큰 (한국어 위키 + 모두의 말뭉치 + 뉴스 + 댓글) | 약 20만-30만 토큰 (위키 2K paragraphs × 3 epoch) — *약 30,000x 격차* |
| 사전학습 시간 | TPU 수일 | T4 약 8-12분 |

본 챕터의 *T4 30분 룰* 안에서 가능한 최대치는 *MLM 데이터 약 10K-15K paragraphs + 2 epoch* 정도. 그래도 *대규모 사전학습* 의 격차는 메우기 어렵습니다 — *데이터 규모 자체의 가치* 가 진짜 BERT 의 비밀.

```python
# T4 30분 룰 안에서 격차 줄여 보기
N_MLM_TRAIN = 10000          # 2K -> 10K (5x)
MLM_EPOCHS = 3               # 그대로 유지 (3 epoch 가 본체 정렬 적정선)
# 또는 모델 키우기
HIDDEN_SIZE = 384            # 256 -> 384, 약 18M params, T4 안 가능
```

*실무 결론* — 한국어 분류 task 에 대해서는 *`klue/bert-base` 또는 그 이상의 사전학습 모델을 가져다 fine-tune* 하는 게 답. 본 챕터의 목적은 *그 격차의 의미* 를 정량으로 보는 교육.

### Q3. (실무) Ch 22 본체를 디스크에 저장 안 하고 in-memory `load_state_dict` 로 옮기는 게 안전한가요? 디스크 경유와 무엇이 다르나요?

*완전히 동일* 합니다 — `load_state_dict` 는 디스크 경유든 in-memory 든 *같은 PyTorch state_dict* 를 그대로 옮기는 연산.

```python
# 디스크 경유 (Ch 22 → Ch 23 정석 흐름)
mlm_model.save_pretrained("./ch22_ckpt")          # state_dict + config 저장
cls_model = AutoModelForSequenceClassification.from_pretrained(
    "./ch22_ckpt", num_labels=2,                  # 자동 헤드 교체 (MLM 버리고 분류 head 부착)
)

# in-memory (본 챕터 self-contained 흐름)
cls_model = BertForSequenceClassification(cls_config)
cls_model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)
# -> 본체만 통째로 옮김, classifier 는 새 random init 그대로
```

본 챕터가 in-memory 흐름을 쓴 이유는 **노트북 self-contained** — Colab 세션이 끊겨도 노트북 하나만으로 끝까지 돌릴 수 있게. 디스크 경유 흐름이 *프로덕션 표준* 입니다. `from_pretrained` 가 *MLM head 는 버려지고 분류 head 가 random init* 으로 부착됨을 warning 메시지로 알려줍니다.

### Q4. (이론) `labels = -100` 이 MLM 압축 재현 셀에서는 쓰이지만 분류 fine-tune 에서는 안 쓰입니다. 왜인가요?

**분류 task 는 모든 sample 에 *정답 라벨* 이 있기 때문** — 가릴 자리가 없습니다.

| 단계 | `labels = ?` | loss 계산 자리 | 학습되는 것 |
|---|---|---|---|
| **MLM 압축 재현** (셀 3) | 선택된 약 15% 만 원본 token id, 나머지 = `-100` | 가려진 자리 | 주변 문맥으로 *가려진 토큰 복원* |
| **NSMC 분류 fine-tune** (셀 4) | 모든 sample 에 `0` 또는 `1` | sample 전체 (배치 차원) | *문장 → 긍정/부정* 분류 |
| **GPT CausalLM 사전학습** (Ch 24-26) | `input_ids.clone()` — *거의 모든 토큰* | (pad 만 `-100`) 사실상 *전 자리* | 모든 자리에서 *다음 토큰 예측* |
| **SFT / Instruction Tuning** (Ch 27) | **prompt 부분 = `-100`**, *답변 토큰만* 원본 id | *답변 부분만* | "질문 외우지 말고 답변하는 법" |

```python
# 분류 task — 모든 sample 에 라벨, -100 사용 안 함
def cls_tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(l) for l in batch["label"]]   # 전부 0 또는 1
    return out
```

> *같은 `-100` 트릭, 적용 자리만 task 별로 다름.* MLM 은 *대부분을 가리고 일부만 학습*, 분류는 *전부 학습*, GPT 사전학습은 *거의 안 가림*, SFT 는 *prompt 만 가림*. 풀버전 표는 Ch 21 §3 *labels = -100 thread* 마크다운에 정리.

### Q5. (이론) 파인튜닝의 의미가 BERT 시대와 GPT 시대 사이에 어떻게 변하나요? 본 챕터는 *마지막 BERT 파인튜닝* 인가요?

본 챕터는 **Phase 3 의 마지막 챕터** 이자 *마지막 BERT 파인튜닝 (task head 부착 패러다임)* 챕터입니다. Phase 4 (Ch 24-) 부터는 같은 단어 "파인튜닝" 이 *다른 의미* 로 쓰입니다.

| 축 | **BERT 파인튜닝** (Ch 9-18, Ch 23) | **GPT 파인튜닝 = SFT** (Ch 25, Ch 27) |
|---|---|---|
| 무엇을 바꾸나 | 본체 + **새 head** (task별 부착) | 본체 + **기존 LM head 그대로** |
| 출력 형식 | task별 다름 (class id / score / multi-hot) | *항상 토큰 시퀀스* — 형식 통일 |
| 학습 신호 | task별 loss (CE/BCE/MSE) | *항상 next-token CE*, 단 자리 마스킹만 다름 |
| 학습되는 것 | *task 의 출력 분포* (긍정/부정 결정 경계 등) | *행동 = "이런 입력엔 이런 형식으로 답하라"* |
| 라벨 | 정답 카테고리/값 | *모범 답안 토큰 시퀀스* |

> **BERT 파인튜닝은 *task 적응*, GPT 파인튜닝은 *행동 정렬*.** Ch 24 부터 시작되는 Phase 4 에서 이 의미 변화를 직접 경험합니다. 풀버전 표는 Ch 21 §3 *파인튜닝 의미 변화* 마크다운 참조.

### Q6. (실무) NSMC 라벨 노이즈가 약 3-5% 라는데, 작은 모델 학습에 더 큰 영향을 주나요?

**그렇습니다** — 작은 모델·작은 데이터일수록 *노이즈 비율* 이 학습 신호를 흐립니다.

```python
# 노이즈 진단 — eval 셋에서 모델이 *자신 있게 틀린* sample 찾기
wrong_idx = np.where(cls_preds != cls_labels)[0]
confident_wrong = wrong_idx[cls_probs_full.max(axis=1)[wrong_idx] > 0.9]
print(f"confidently wrong (prob > 0.9): {len(confident_wrong)} / {len(cls_labels)}")
for i in confident_wrong[:5]:
    print(f"  pred={cls_preds[i]} true={cls_labels[i]} text={tokenizer.decode(cls_eval[int(i)]['input_ids'], skip_special_tokens=True)[:80]}")
```

자신 있게 틀린 sample 중 일부는 *진짜 라벨 노이즈* (반어법, 이중 의미, 라벨러 실수). 분류 정확도가 *천정 100%* 가 안 되는 본질적 이유 중 하나. NSMC 의 *알려진 한계* — `klue/bert-base` 같은 대규모 사전학습 모델도 NSMC 에서 약 86% 안팎이 천장이라는 걸 알면 *과적합 의심* 을 피할 수 있습니다.

### Q7. (이론) Phase 3 가 끝났는데, *원본 BERT 정신* 의 핵심 메시지를 한 줄로 정리하면 뭐가 남나요?

**일반 도메인 사전학습 + 다른 도메인 fine-tune transfer 가 *task 별 from-scratch 학습 보다 압도적으로 효율적*** 이라는 것이 Phase 3 의 한 줄 결론. 본 챕터의 2-way 비교와 부록의 random init baseline 이 그 직접 증거:

- *random init* 만 가지고 NSMC fine-tune: accuracy 약 0.50-0.55 (부록에서 측정 — 한국어 환경에선 negative transfer 로 *작은 사전학습 모델* 과 비등하거나 역전될 수도)
- *작은 일반 도메인 사전학습 + fine-tune* (본 챕터): accuracy 약 0.54 (실측 — MLM 약 0.2분의 짧은 사전학습이라 동전 던지기에 가까움)
- *대규모 일반 도메인 사전학습 + fine-tune* (Ch 15): accuracy 약 0.86 (실무 baseline)

세 셋업의 격차가 *사전학습 데이터 양·도메인 다양성 + 모델 크기* 에 거의 비례 (한국어는 도메인 다양성 변수가 특히 중요). *task 도메인으로 직접 사전학습* 하지 않고 *일반 위키* 만으로도 충분한 transfer 가 일어난다는 게 *원본 BERT 의 진짜 통찰*.

Phase 4 (Ch 24-) 부터는 같은 *사전학습 → fine-tune* 패러다임이 *decoder-only GPT* 환경에서 어떻게 *SFT / behavior alignment* 로 변하는지 봅니다. 본체 구조 (encoder → decoder), task (masked → causal), fine-tune 의미 (head 부착 → 행동 정렬) 셋 다 바뀝니다.

## 다음 챕터 예고

**Chapter 24. GPT scratch — 영어 TinyStories (Phase 4 시작)**

- *encoder* (BERT) → ***decoder-only* (GPT)** — attention 구조가 *causal mask* 로 바뀜
- *MLM* (가려진 토큰 양방향 예측) → ***Causal LM*** (앞 토큰만으로 다음 토큰 예측)
- *task별 head 부착* → ***LM head 그대로 next-token CE*** — Ch 23 까지의 분류 head 부착 패러다임은 여기서 막을 내림
- 데이터: TinyStories 영어 동화 — GPT-4 가 4세 어린이 어휘로 생성한 짧은 영문 동화 약 2.1M 편
- 모델: `GPT2LMHeadModel(config)` 약 3M params, *완전 무작위 초기화 from scratch*
- BPE 토크나이저 직접 학습 (vocab 2048) — Ch 19 의 토크나이저 학습 패턴 재등장

> **Phase 구조 전환** — Phase 3 (Ch 19-23) 가 *BERT 본체 + 토크나이저를 직접 학습하는 영어/한국어 두 갈래* 였다면, Phase 4 는 *GPT 본체를 from-scratch 로 학습 → SFT → behavior alignment* 흐름. 영어 BERT scratch (Ch 20-21) → 한국어 BERT scratch (Ch 22-23) → 영어 GPT scratch (Ch 24-) 의 대칭 구조. 사전학습-fine-tune 패러다임의 *의미 자체가* 바뀌는 자리입니다.

본 챕터 (Ch 23) 는 *Phase 3 의 마지막* 이자 *BERT 시대의 마지막 파인튜닝 챕터*. 한국어 NSMC 라는 *실무에서 자주 쓰이는 한국어 task* 로 마무리하는 게 Phase 3 의 의도된 마침표입니다.
