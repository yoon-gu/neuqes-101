## 이번 챕터에 등장한 라이브러리·함수 (Ch 20 과의 차이만)

| 이름 | 한 줄 설명 | Ch 20 과 차이 |
|---|---|---|
| `AutoTokenizer.from_pretrained("klue/bert-base")` | 한국어 WordPiece (vocab 약 32,000) | 영어 → 한국어 |
| `load_dataset("wikimedia/wikipedia", "20231101.ko")` | 한국어 Wikipedia HF 정제본 로드 | `load_dataset("Salesforce/wikitext", ...)` (Ch 20) — 같은 패턴, 언어만 변경 |
| `Dataset.from_pandas(df[["document"]]).rename_column(...)` | pandas → HF Dataset 변환 | Ch 15 와 같은 패턴 |
| `transformers.BertConfig` (동일) | 작은 BERT hyperparam | (Ch 20 동일) |
| `transformers.BertForMaskedLM(config)` (동일) | random init MLM 모델 | (Ch 20 동일) |
| `DataCollatorForLanguageModeling(mlm_probability=0.15)` (동일) | 매 batch 동적 80/10/10 masking | (Ch 20 동일) |
| `group_texts` 패턴 (동일) | 가변 길이 → 고정 블록 스트림 | (Ch 20 동일) |
| `model.save_pretrained()` / `tokenizer.save_pretrained()` (동일) | HF 표준 체크포인트 | (Ch 20 동일) |

## 체크포인트 질문

1. Ch 19 §5-4 에서 *영어 토크나이저로 한국어를 토큰화하면 UNK 가 폭증* 한다는 걸 봤습니다. 이번 챕터의 토크나이저 비교 표 (셀 2 하단) 가 그 결론과 정확히 일치하나요? `bert-base-uncased` 가 한국어 문장을 *자모 단위* 로 분해한 결과를 어떻게 해석해야 할까요?
2. MLM random baseline 이 Ch 20 (vocab 30,522) 의 약 10.33 에서 Ch 22 (vocab 32,000) 의 약 10.37 로 *미세하게* 바뀝니다. 이 0.04 차이가 학습 동역학에 의미 있는 영향을 주나요? (힌트: 학습 곡선의 절대값 vs 상대 변화)
3. 한국어 위키 paragraph 는 *제한 50-2000자 필터* 로 평균 길이가 일정합니다. NSMC 한 줄 리뷰보다는 깁니다. 같은 5K 샘플이라도 *총 토큰 양* 이 Ch 20 (Wikitext-103) 와 어떻게 다른지, 같은 epoch 수에서 *생성 블록 수* 가 어떻게 달라지는지 확인해 보세요.
4. `DataCollatorForLanguageModeling` 이 토큰 id 만 보고 동작한다는 게 이번 챕터의 결론 중 하나입니다. 그렇다면 *한국어 모델 학습 시 mlm_probability 를 0.15 가 아닌 다른 값으로 바꿔야 할 이유* 가 있을까요?

## FAQ

### Q1. (실무) `bert-base-uncased` 토크나이저를 그대로 쓰면 안 되나요? Ch 20 의 코드를 *언어만 데이터로* 바꾸는 게 더 단순한데.

쓰면 *거의 학습 안 됨* 입니다. 이번 챕터 셀 2 하단의 비교 표가 그 답:

```python
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
sent = "이 영화 정말 재미있어요!"
toks = tokenizer_en.tokenize(sent)
# ['ᄋ', '##ᅵ', 'ᄋ', '##ᅧ', '##ᆼ', '##ᄒ', '##ᅪ', 'ᄌ', '##ᅥ', '##ᆼ', '##ᄆ', '##ᅡ', '##ᆯ', '[UNK]', '!']
```

한국어 문장이 *자모 단위* 로 분해되거나 `[UNK]` 가 섞입니다. 임베딩 테이블이 *vocab 30,522 영어 단어 위주* 라 자모·UNK 자리의 임베딩이 *의미 없는 random vector*. 그 위에서 MLM 학습을 해도 *언어 정보를 압축할 자리* 가 없습니다. Ch 19 §5-4 의 결론을 그대로 재확인 — *토크나이저는 모델의 언어를 물리적으로 결정* 합니다.

### Q2. (이론) 한국어는 형태소가 풍부한데 `mlm_probability` 를 0.20-0.25 로 올리면 학습이 더 잘 되나요?

**일부 연구는 그런 시도를 했고 결과는 trade-off** 입니다. 한국어는 한 어절 안에 *어간 + 어미 + 조사* 가 결합되어 형태소 정보가 풍부 → 가릴 자리가 많아 *학습 신호 양* 은 늘 수 있습니다. 그러나:

```python
# 15 → 0.25 로 올렸을 때
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.25,
)
```

- **장점**: 한 batch 당 학습되는 토큰 수가 약 1.7배 증가 → loss 가 같은 step 수에서 더 빨리 떨어질 수 있음
- **단점**: 가려진 비율이 높으면 *주변 문맥* 자체가 줄어들어 *추측 가능성* 이 떨어짐. 모델이 *의미 없는 random guess* 만 학습할 위험

BERT 논문의 15% 는 *모든 언어에서 합리적 sweet spot* 으로 정착했고, 한국어 BERT (`klue/bert-base`) 도 15% 로 사전학습. *0.20-0.25 시도는 가능* 하지만 *명확한 개선 보장은 없음*. 작은 모델 + 작은 데이터일수록 *기본 셋업 안정* 이 더 가치 있습니다.

### Q3. (이론) `klue/bert-base` 가 이미 학습된 모델인데 왜 같은 구조의 mini 버전을 처음부터 다시 학습하나요?

**교육 목적** 입니다. 사전학습이 본체에 무엇을 새기는지 *직접 경험* 하기 위해.

```python
# 실무 흐름 (Ch 15)
model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)
# -> 이미 한국어 위키·뉴스·댓글로 학습된 110M 본체. 분류 헤드만 fine-tune.

# 이번 챕터 흐름
config = BertConfig(hidden_size=256, num_hidden_layers=4, ...)
model = BertForMaskedLM(config)   # random init, weight 없음
# -> 한국어 위키 5K paragraphs 로 MLM 직접 학습 (일반 도메인). Ch 23 에서 NSMC 분류 fine-tune.
```

실무에서는 *클루 본체 그대로 가져다 쓰는 게* 답입니다 — 데이터·연산이 *5000배 이상* 격차. 본 챕터의 목적은 *그 격차의 의미* 를 Ch 23 에서 정량 비교하기 위함이고, *작은 모델 + 작은 데이터로도 사전학습 동역학을 재현* 할 수 있음을 확인하는 것. Ch 20 (영어) 의 한국어 대칭본.

### Q4. (실무) 한국어 텍스트에 영어가 섞여 있으면 `klue/bert-base` 토크나이저가 잘 처리하나요?

대체로 잘 처리합니다 — `klue/bert-base` 의 vocab 약 32,000 안에 영어 단어·서브워드도 일부 포함되어 있어 *자주 등장하는 영어* 는 자연스럽게 토큰화됩니다. 다만:

```python
mixed = "이 movie 정말 amazing 했어요!"
print(tokenizer.tokenize(mixed))
# 예시: ['이', 'movie', '정말', 'am', '##azing', '했', '##어요', '!']
```

- *자주 쓰는 영단어* (`movie`, `OK` 등): 1 토큰 또는 짧은 WordPiece 로 처리
- *드문 영단어* 나 *고유명사*: 자모 단위 분해 또는 `[UNK]` 위험
- *한자, 일본어, 특수문자*: vocab 안에 일부만 있어 *부분 UNK* 가능

한국어 위키 본문은 *순한국어 + 인명·지명·과학 용어 등 영문 표기* 가 자주 섞입니다. `klue/bert-base` vocab 에 자주 쓰는 영단어 일부가 있어 큰 문제는 없습니다. 다국어 환경이라면 *multilingual BERT* (`bert-base-multilingual-cased`) 또는 *byte-level BPE* (XLM-R) 같은 *공통 vocab* 모델을 고려.

### Q5. (이론) 셀 5-1 에서 본 `label_id = -100` 이 정확히 어떻게 *loss 무시* 로 이어지나요?

PyTorch `CrossEntropyLoss` 의 기본 `ignore_index=-100` 동작:

```python
import torch
loss_fn = torch.nn.CrossEntropyLoss()   # ignore_index=-100 (default)
logits = torch.randn(10, 32000)         # (seq_len, vocab)
labels = torch.tensor([5, 9, -100, -100, 12, -100, -100, 7, -100, -100])
loss = loss_fn(logits, labels)
# -> 위치 0, 1, 4, 7 의 CE 만 평균. -100 자리 4개는 *완전 무시*
```

`DataCollatorForLanguageModeling` 이 가려지지 않은 자리에 `-100` 을 채우는 게 *전 자리에서 CE 계산 후 마스킹* 보다 효율적입니다. 같은 트릭이:

- **GPT 사전학습** (Ch 24-26): `labels = input_ids.clone()` → 사실상 *모든 자리* 학습 (pad 만 -100)
- **SFT / Instruction Tuning** (Ch 27): `labels[prompt_mask] = -100` → *답변 부분만* 학습

세 곳 모두 같은 `-100` 트릭, 적용 자리만 다릅니다. Ch 21 §3 의 *labels = -100 thread* 마크다운에 풀버전 설명.

### Q6. (실무) MLM eval loss 가 4-6 부근에서 *더 떨어지지 않으면* 어떻게 진단하나요?

작은 BERT scratch + 5K 문장의 *자연스러운 수렴 영역* 입니다. 추가로 떨어뜨리려면:

```python
# (1) 데이터 늘리기 — 가장 큰 효과
N_TRAIN_TEXT = 30000   # 5K -> 30K, T4 30분 안에 1 epoch 가능

# (2) epoch 늘리기 (단, 작은 데이터에 과적합 위험)
NUM_EPOCHS = 3

# (3) 모델 키우기 — T4 메모리 안에서
HIDDEN_SIZE = 384   # 256 -> 384, layer 4 유지 시 약 18M params
```

데이터·epoch·모델 크기를 *함께* 늘려야 효과가 큽니다. 단, T4 30분 룰 한계 — *원리 확인용 toy 셋업* 임을 잊지 말 것. 실제 한국어 사전학습은 `klue/bert-base` (한국어 위키 + 뉴스 + 댓글, 약 수억 토큰, GPU·TPU 수일) 가 한 결과물.

### Q7. (이론) Ch 20 (영어) 와 Ch 22 (한국어) 의 학습 곡선을 *비교* 하려면 어떻게 해야 하나요?

같은 hyperparam·같은 BLOCK_SIZE·같은 epoch 으로 학습된 두 모델의 *상대* 비교가 의미 있습니다.

```python
# 비교 차원
metrics = {
    "language":           ["EN (Ch 20)",    "KO (Ch 22)"],
    "vocab_size":         [30522,           32000],
    "random_baseline":    [10.33,           10.37],
    "epoch1_eval_loss":   ["measure",       "measure"],
    "epoch2_eval_loss":   ["measure",       "measure"],
    "epoch2_perplexity":  ["measure",       "measure"],
    "train_tokens":       ["approx 700K",   "approx 500K"],   # 한국어 위키 paragraphs 5K
}
```

한국어 위키 paragraphs 는 평균 길이가 NSMC 한 줄 리뷰보다는 깁니다. 같은 5K 샘플이라도 *토큰 총량* 이 Ch 20 의 Wikitext-103 paragraphs 와 살짝 다릅니다. 같은 step 수에 *실제 본 토큰 수* 가 다르고, eval loss 도 영향을 받습니다. *언어 자체의 어려움 차이* 가 아니라 *데이터 크기 차이* 가 더 큰 영향. 공정한 언어 비교는 *토큰 총량 매칭* 이 필요.

## 다음 챕터 예고

**Chapter 23. 작은 BERT 분류 — 한국어 NSMC 이진 (일반 도메인 사전학습 → 다른 도메인 fine-tune)**

- 이번 챕터의 `./ch22_small_bert_mlm_ko` 체크포인트를 `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)` 로 로드 → MLM head 떼고 분류 헤드 부착
- NSMC 이진 분류 fine-tune (Ch 15 와 같은 데이터·셋업) — *완전히 다른 도메인 transfer*
- **핵심 비교**: 이번 작은 사전학습 BERT (약 10M params, 위키 5K paragraphs MLM) vs Ch 15 의 `klue/bert-base` (약 110M params, 대규모 일반 한국어 사전학습) — 2-way
- 영어 Ch 20 → Ch 21 흐름의 *한국어 대칭본* — 같은 격차 패턴이 한국어 환경에서도 나오는지 검증
- 추가로 *random init baseline* 비교 + *위키 → NSMC 의 negative transfer 분석* 은 Ch 23 부록 [`appendix_random_baseline.ipynb`](../23_ko_bert_classify/appendix_random_baseline.ipynb)

> **변하는 축**: Phase 3 안에서 *task* 가 사전학습 (MLM) → 분류 (fine-tune) 로 전환. *파인튜닝* 의 의미는 **BERT 시대 = task 별 head 부착**. 본체는 그대로, downstream task 마다 새로 random init 된 작은 head 가 붙어 적응. Ch 23 에서 본격적으로 다시 짚어 봅니다.
