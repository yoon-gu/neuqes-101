## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.BertConfig` | BERT 구조 hyperparam 컨테이너 (hidden, layer, head 등) | Ch 22 에서 한국어 작은 BERT 설계 |
| `transformers.BertForMaskedLM` | encoder + MLM head, MLM 사전학습 전용 모델 클래스 | Ch 22 에서 한국어 MLM |
| `transformers.DataCollatorForLanguageModeling` | 매 batch 마다 자동 masking (15% rule) | Ch 22 같음 |
| `BertForMaskedLM(config)` (random init) | pretrained weight 없이 모델 생성 | Ch 22 같음 |
| `load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")` | Wikitext-103 HF 정제본 로드 (line 단위) | Ch 22 한국어 Wikipedia (`wikimedia/wikipedia`) 와 같은 패턴 |
| `group_texts` 패턴 (HF run_mlm.py 표준) | 가변 길이 텍스트를 고정 길이 블록 스트림으로 | Ch 22 같음 |
| `model.save_pretrained()` / `from_pretrained()` | HF 표준 체크포인트 인터페이스 | Ch 21 에서 분류 fine-tune 로드 |
| `math.log(vocab_size)` | MLM 의 random baseline loss | 사전학습 챕터 공통 진단 도구 |

## 체크포인트 질문

1. MLM 사전학습의 random baseline loss 가 `ln(30522) ≈ 10.33` 인 이유는 무엇이고, 학습 첫 step 의 loss 가 이 값과 *크게* 다르다면 무엇을 의심해야 하나요?
2. `DataCollatorForLanguageModeling` 이 매 batch 마다 *다른* 위치를 mask 합니다. 같은 위치를 *고정해서* mask 하면 어떤 문제가 생길까요?
3. `BertForMaskedLM` 의 MLM head 가 *입력 임베딩과 tied* 됩니다. 왜 이렇게 묶으면 파라미터 절약 + 학습 안정에 모두 도움이 되나요?
4. Ch 21 에서 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 를 호출하면, *이번 챕터 모델의 어떤 부분* 이 이어지고 *어떤 부분* 이 버려지나요?

## FAQ

### Q1. (이론) 왜 Yelp text 가 아니라 일반 위키 (Wikitext-103) 로 사전학습하나요? Ch 21 의 분류가 Yelp 인데 같은 도메인으로 학습하는 게 더 유리하지 않나요?

**일반 도메인 → 다른 도메인 transfer** 가 *진짜 사전학습-fine-tune 패러다임* 이기 때문입니다. Yelp text 로 MLM 사전학습 → Yelp 분류 fine-tune 의 흐름은 사실 *domain-adaptive pretraining* (DAPT) 에 더 가깝습니다 — 사전학습이 *이미 task 도메인의 표현* 을 학습한 상태에서 분류만 얹는 셈.

원본 BERT (Devlin et al., 2018) 가 *Wikipedia + BookCorpus* 라는 일반 도메인 코퍼스로 사전학습한 뒤 *완전히 다른* GLUE/SQuAD task 로 fine-tune 한 게 *transfer 의 본질* — 일반 표상이 다른 도메인에도 적용 가능한가의 시험.

```python
# 이번 챕터 (Ch 20 → Ch 21) 흐름 — 원본 BERT 와 같은 정신
# 일반 위키 (Wikitext-103) 로 MLM 사전학습
# 영화 리뷰 (Yelp) 로 분류 fine-tune  ← 다른 도메인 transfer

# 만약 Yelp 로 사전학습 → Yelp 분류 fine-tune 이었다면
# domain-adaptive pretraining 에 가까워져 transfer 메시지가 약해짐
```

Ch 22-23 의 한국어 흐름 (한국어 Wikipedia → NSMC 영화 리뷰) 도 *대칭* 패턴. 일반 위키 학습이 task 도메인 fine-tune 에 *얼마나 효율적으로 transfer 되는가* 가 본 챕터의 진짜 측정 대상.

> **참고** — Ch 10 (DistilBERT) 도 Wikipedia + BookCorpus 사전학습 → Yelp 분류 fine-tune 의 *같은 패턴*. 본 챕터의 작은 BERT 와 Ch 10 의 DistilBERT 가 *같은 일반 도메인 사전학습 → 같은 task fine-tune* 이라 *공정한 비교* 가 됩니다. Yelp 로 사전학습했다면 Ch 10 과 비교 자체가 unfair.

### Q2. (실무) `bert-base-uncased` 의 모델 weight 도 같이 가져오면 더 빠르지 않나요?

맞습니다 — 그게 *fine-tuning* 흐름 (Ch 7-18). 이번 챕터는 그 흐름을 *뒤집어*, "사전학습이 어떻게 이뤄지는지" 를 보는 게 목적입니다.

`BertForMaskedLM.from_pretrained("bert-base-uncased")` 라고 쓰면 110M 사전학습 모델이 로드되어 *이미 잘 작동* 함. 반면 이번 챕터는:

```python
config = BertConfig(hidden_size=256, num_hidden_layers=4, ...)
model = BertForMaskedLM(config)   # random init, weight 없음
```

`from_pretrained` 가 아니라 `BertForMaskedLM(config)` 라는 생성자 호출로 *비어 있는* 모델을 만듭니다. 학습이 *0 에서 시작* 하는 것 자체가 핵심.

### Q3. (이론) `mlm_probability=0.15` 의 15% 는 어디서 나온 숫자인가요?

BERT 원논문 (Devlin et al., 2018, arXiv:1810.04805) 의 sweet spot:

- 너무 작으면 (5% 정도): 한 batch 의 *학습 신호* (loss 가 계산되는 위치) 가 너무 적어 학습 효율 ↓
- 너무 크면 (40%+): 모델이 *문맥* 으로 볼 수 있는 토큰이 너무 적어 mask 추측이 *불가능에 가까워짐*. loss 가 크지만 학습 가치는 작음
- 15% 가 *학습 신호 양 + 추측 가능성* 의 균형점

후속 연구 (RoBERTa, ELECTRA) 는 *동적 masking* (매 batch 다른 위치, 이미 본 챕터에서 collator 가 자동 처리) 또는 *교체-기반 학습* (ELECTRA) 같은 변형을 시도했지만, *15% mask 비율* 자체는 거의 표준으로 정착.

### Q4. (실무) MLM 학습 중 loss 가 갑자기 *발산* 하면 어떻게 해야 하나요?

작은 BERT scratch 학습은 fine-tune 보다 *학습률에 민감* 합니다. 발산 (loss → NaN 또는 100+) 의 흔한 원인:

```python
# 학습률 낮추기 (5e-4 → 1e-4 → 5e-5 순서로)
training_args = TrainingArguments(learning_rate=1e-4, ...)

# warmup_ratio 늘리기 (0.06 → 0.1)
training_args = TrainingArguments(warmup_ratio=0.1, ...)

# gradient clipping (Trainer 기본 1.0, 더 빡빡하게)
training_args = TrainingArguments(max_grad_norm=0.5, ...)

# fp16 끄고 fp32 로 시도 (loss scale overflow 가능성)
training_args = TrainingArguments(fp16=False, ...)
```

이번 챕터의 `lr=5e-4, warmup=0.06, fp16=True` 셋업은 *작은 BERT + 5K 데이터* 에 맞춰 보수적으로 잡았습니다. 모델 키우거나 데이터 늘리면 위 옵션을 조정.

### Q5. (실무) 사전학습이 *얼마나* 도움 되는지 어떻게 확인하나요?

Ch 21 에서 두 모델을 *같은 분류 task* 로 fine-tune 해 비교하는 게 가장 직접적입니다:

```python
# A. 이번 챕터 사전학습 모델 (Ch 20 산출물)
model_pretrained = AutoModelForSequenceClassification.from_pretrained(
    "./ch20_small_bert_mlm", num_labels=2
)

# B. 같은 구조 + random init (사전학습 안 한 baseline)
config = BertConfig(hidden_size=256, num_hidden_layers=4, ...)
model_scratch = BertForSequenceClassification(config)
```

두 모델을 *같은 Yelp 이진 학습 데이터* 로 fine-tune → eval accuracy 비교. 사전학습이 도움 됐다면 (A) 가 (B) 보다 *빨리* 그리고 *높이* 도달. Ch 21 의 핵심 실험.

> **참고**: 이번 챕터의 *작은 사전학습* (5K 문장, 1-2 epoch) 은 *큰 효과* 를 기대하기 어렵습니다. 그러나 *방향성* (random 보다 시작점이 낫다) 은 분명히 나옵니다. 큰 효과를 보려면 데이터 100K+, epoch 5+, BERT 표준 크기 — 이건 T4 30분 룰 밖.

### Q6. (이론) `group_texts` 가 문장 경계를 무시하는데, BERT 가 잘 학습되나요?

원논문의 BERT 는 NSP (Next Sentence Prediction) 같은 *문장 쌍* task 도 같이 학습했지만, 후속 연구 (RoBERTa, 2019) 가 *NSP 를 빼고 그냥 토큰 스트림으로 학습* 해도 성능이 *더 좋다* 는 걸 보였습니다. 이번 챕터는 그 단순화된 흐름 (MLM only).

토큰 스트림이 *문장 경계 정보를 잃지만* 얻는 게 더 큽니다:
- 짧은 문장이 PAD 로 가득 차지 않음 → GPU 활용도 ↑
- 긴 문장이 잘리지 않음 → 정보 손실 ↓
- 학습 신호 (mask 위치) 가 *균등하게 분포*

문장 경계는 분류·NLI 같은 downstream 에서 다시 명시적으로 입력됩니다 ([CLS] 토큰).

### Q7. (실무) 저장된 체크포인트가 너무 무거우면 어떻게 가볍게 하나요?

이 챕터의 작은 BERT 는 약 40MB 정도라 무겁지 않지만, 큰 모델의 경우:

```python
# safetensors 형식 강제 (bin 보다 약간 작음 + 안전)
model.save_pretrained("./ch20_small_bert_mlm", safe_serialization=True)

# fp16 으로 저장 (weight 자체를 half 로)
model.half().save_pretrained("./ch20_small_bert_mlm")

# 양자화 (advanced — bitsandbytes 8-bit/4-bit)
# from transformers import BitsAndBytesConfig
# config = BitsAndBytesConfig(load_in_8bit=True)
```

이번 챕터는 *학습용 체크포인트* 이므로 fp32 그대로 저장 (Ch 21 fine-tune 시 정밀도 유지). 배포용이면 inference 단계에서 quantize 고려.

### Q8. (이론) 큰 BERT (110M) 와 비교해 이번 작은 BERT (10M) 의 *근본 한계* 는?

| 차원 | 작은 BERT (이번 챕터) | bert-base-uncased | 차이의 영향 |
|---|---|---|---|
| hidden_size | 256 | 768 | 표현 공간 차원이 1/3 → 미세한 의미 구분 어려움 |
| num_layers | 4 | 12 | *깊은* 추론 (구문 → 의미 → 문맥) 단계 부족 |
| 학습 데이터 | 5K paragraphs (약 700K-1M 토큰, Wikitext-103) | 33억 토큰 (BERT-base) | 어휘 다양성·문맥 풍부함 격차 약 5000배 |
| 학습 시간 | 20분 | 4 일 (TPU v3-256) | 압축한 *정보량* 자체가 다름 |

**결론**: 이번 챕터의 산출물은 *fine-tune 출발점으로는 random 보다 나음* 정도. *zero-shot* 또는 *복잡한 downstream* 에선 표준 BERT 와 비교 불가. *작은 모델 + 작은 데이터로도 일반 도메인 사전학습이 가능하다는 메커니즘* 을 *경험* 하는 게 이 챕터의 목적이고, *실용 모델* 은 표준 사전학습품을 가져다 쓰는 게 정답.

## 다음 챕터 예고

**Chapter 21. 영어 BERT 분류 (Ch 20 사전학습 모델 fine-tune) — *일반 도메인 → 다른 도메인 transfer***

- 이번 챕터의 `./ch20_small_bert_mlm` 체크포인트를 `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)` 로 로드 → MLM head 떼고 분류 헤드 부착
- **Yelp 이진 분류 fine-tune** (Ch 10·11 과 같은 데이터·셋업) — *완전히 다른 도메인 transfer*. 일반 위키로 사전학습한 본체가 *영화 리뷰 도메인* 에 얼마나 잘 적응하는가 측정
- **핵심 비교**: 이번 작은 사전학습 BERT (약 10M params, Wikitext-103 5K paragraphs MLM) vs Ch 10 의 DistilBERT (약 66M params, 대규모 Wikipedia + BookCorpus 사전학습). 둘 다 *일반 도메인 → Yelp transfer* 라 비교가 *fair* — *사전학습 규모* 차이만 측정됨
- 작은 모델 + 작은 데이터 사전학습이 *얼마나 도움 되는가* 의 정량 측정 — fine-tune 학습 곡선·최종 accuracy·confusion matrix 모두 나란히
- 일부러 *random init* baseline (사전학습 없이 분류 직접) 도 함께 학습해 *사전학습의 순 효과* 분리

> **변하는 축**: Phase 3 안에서 *task 가 사전학습 (MLM) → 분류 (fine-tune)* 로 전환. 모델 구조·토크나이저는 그대로, 데이터 도메인이 *위키 → Yelp* 로 바뀌고 loss·평가 metric 이 분류 표준으로 돌아옴.
