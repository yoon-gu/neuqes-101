**목표**: Phase 3 의 네 번째 챕터. Ch 20 에서 *영어 작은 BERT* 를 random init 해 MLM 사전학습 했다면, 이번엔 *완전히 같은 본체 구조* 로 **한국어 MLM 사전학습** 합니다. 변하는 축은 **언어** — 토크나이저는 `klue/bert-base` (한국어 WordPiece, vocab 약 32,000), 데이터는 **한국어 Wikipedia** (`wikimedia/wikipedia`, `20231101.ko`) paragraphs. 본체 hyperparam, loss, training args 는 Ch 20 과 동일. *Ch 23 의 분류 fine-tune (NSMC 영화 리뷰) 은 완전히 다른 도메인* — 일반 도메인 사전학습 → task 도메인 fine-tune 의 정직한 transfer 메시지.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 2-4분 (토크나이저 로드 + ko 위키 다운로드·paragraph split·토큰화가 대부분을 차지 + MLM 2 epoch 약 0.3분 + 평가/저장 — 전체 실측 약 2분, 네트워크·VM 상태에 따라 늘어날 수 있음). 전체 소요는 데이터 다운로드가 지배합니다.

## 학습 흐름

1. 🔤 **토크나이저**: `klue/bert-base` WordPiece (vocab 약 32,000) 그대로 로드
2. 📥 **데이터**: 한국어 Wikipedia (`wikimedia/wikipedia`, `20231101.ko`), paragraph 단위 5,000 sample (라벨 없음 — Wikipedia 본문)
3. 🚀 **토큰화 + `group_texts`**: Ch 20 과 같은 패턴 — 모든 텍스트를 이어붙여 `block_size=128` 블록 스트림
4. 🏗️ **모델 구성**: Ch 20 과 같은 `BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024)` + `BertForMaskedLM(config)` random init
5. 🚀 **학습**: `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` + Trainer, fp16, 2 epoch
6. 🔬 **평가**: MLM loss 학습 곡선, perplexity, 한국어 [MASK] 토큰 예측 시연
7. 💾 **저장**: `model.save_pretrained("./ch22_small_bert_mlm_ko")` — Ch 23 에서 분류 fine-tune

> 📒 **사전 학습 자료**: Ch 19 §5-4 (cross-language UNK) — *영어 토크나이저로 한국어를 처리하면 UNK·자모 폭증* 을 봤습니다. 이번 챕터는 그 결론을 *한국어 데이터엔 한국어 토크나이저 + 한국어 사전학습이 자연스럽다* 로 잇습니다. Ch 20 (영어 MLM scratch) 의 본체·셋업을 그대로 가져와 *언어 한 축만* 바꿉니다.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — | — |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Wikitext-103 paragraphs (일반 도메인) | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) |
| 21 | Ch 20 사전학습 BERT + 분류 헤드 | (Ch 20과 동일) | Yelp 이진화 | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| **22 ← 여기** | **작은 BERT (직접, scratch) — 한국어** | **`klue/bert-base` 토크나이저 (가져옴)** | **한국어 Wikipedia paragraphs** | **MLM head** | softmax (MLM) | **`CrossEntropyLoss` (masked token)** |
| 23 (다음) | Ch 22 사전학습 BERT + 분류 헤드 | (Ch 22와 동일) | NSMC 이진 | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

**Phase 3 안에서의 위치** — Ch 19 (토크나이저 학습) → Ch 20 (영어 모델 사전학습) → Ch 21 (영어 분류) → **Ch 22 (한국어 모델 사전학습)** → Ch 23 (한국어 분류). Ch 22 → Ch 23 흐름이 Ch 20 → Ch 21 흐름의 *한국어 대칭본*. 클라이맥스는 Ch 23 — *우리가 직접 사전학습한 작은 한국어 BERT* 와 *기존 `klue/bert-base` 사전학습 모델* (Ch 15) 의 정량 비교.

## 변경점 (Diff from Ch 20)

| 축 | Ch 20 (영어 MLM scratch) | Ch 22 (한국어 MLM scratch) |
|---|---|---|
| **언어** | 영어 | **한국어** ← *유일한 변화* |
| 토크나이저 | `bert-base-uncased` (vocab 30,522, 영어 WordPiece) | **`klue/bert-base` (vocab 약 32,000, 한국어 WordPiece)** |
| 데이터 | Wikitext-103 paragraphs 5K (일반 도메인, `Salesforce/wikitext`) | **한국어 Wikipedia paragraphs 5K (`wikimedia/wikipedia` 20231101.ko)** |
| 본체 hyperparam | `BertConfig(hidden=256, layer=4, head=4, intermediate=1024)` | (그대로) |
| 모델 클래스 | `BertForMaskedLM` (random init) | (그대로) |
| Collator | `DataCollatorForLanguageModeling(mlm_probability=0.15)` | (그대로) |
| Training args | epoch=2, batch=32, lr=5e-4, warmup=0.06, fp16 | (그대로) |
| Loss | `CrossEntropyLoss` (masked token, vocab 30,522 logits) | **`CrossEntropyLoss`** (masked token, vocab 약 32,000 logits) |
| 산출물 | `./ch20_small_bert_mlm` | **`./ch22_small_bert_mlm_ko`** — Ch 23 재사용 |

> **변경점 한 가지 원칙** — Phase 3 안에서 *언어 축* 만 변합니다. 본체 구조도 학습 셋업도 동일. *같은 코드를 한국어 토크나이저 + 한국어 데이터로 돌렸을 때 같은 결이 나오는가* 가 본 챕터의 검증 포인트.

### 왜 한국어엔 한국어 토크나이저인가 — Ch 19 §5-4 결론 잇기

Ch 19 의 cross-language 실험에서 *영어 토크나이저로 한국어를 토큰화하면 UNK 비율이 폭증* 한다는 걸 봤습니다. 그 위에서 모델을 사전학습하면 *언어 정보 자체가 사라진 [UNK] 자리* 가 대부분이라 학습 신호가 거의 없습니다.

이번 챕터는 그 결론의 자연스러운 다음 단계: **언어 데이터에 맞는 vocab 으로 토크나이저를 가져온 뒤 모델을 사전학습**. 토크나이저까지 *직접 학습* 하지 않은 이유는 Ch 20 과 같습니다 — `klue/bert-base` 의 vocab 은 한국어 위키 + 뉴스 + 댓글 등 대규모 코퍼스로 학습되어 *어휘 커버리지가 검증된* 출발점.

## Loss 함수의 변화 — *없음*. Ch 20 과 같은 MLM CE

이번 챕터는 *언어만 바뀌고* loss 함수는 Ch 20 과 동일한 MLM CrossEntropyLoss. 가려진 위치의 *원래 토큰* 을 vocab 차원 softmax 로 예측. 다만 vocab 크기가 살짝 달라 random baseline 이 미세하게 이동합니다.

### 수식 (Ch 20 과 동일)

$$L_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid x_{\setminus M})$$

- $M$: 가려진 위치 집합 (전체 토큰의 약 15%)
- $P(x_i \mid x_{\setminus M})$: 모델이 $i$ 번 위치에 *원래 토큰* 을 예측할 확률 (vocab 약 32,000 차원 softmax)

### vocab 차이가 random baseline 에 주는 미세한 영향

| 토크나이저 | vocab size $V$ | random baseline $\log V$ | random PPL $= V$ |
|---|---|---|---|
| `bert-base-uncased` (Ch 20) | 30,522 | **10.33** | 30,522 |
| `klue/bert-base` (Ch 22) | 32,000 | **10.37** | 32,000 |

차이는 약 0.04 정도로 *거의 무시할 수준*. 학습 첫 step 의 loss 가 약 10.37 부근이면 random init 직후 *균등 추측* 상태. 첫 100 step 안에 빠르게 떨어지면 vocab 정상 작동.

> 분류 챕터에서 K (클래스 수) 가 늘 때 random baseline `log K` 가 커지듯, MLM 도 vocab 이 커지면 random baseline 이 커집니다. 하지만 vocab 30K vs 32K 정도의 차이는 *학습 동역학에 영향 없음* — 학습 종료 loss 의 절대값을 비교할 때만 미세 보정.

### 숫자로 감 잡기 (vocab 32,000 — Ch 20 과 같은 척도)

| 모델 상태 | $-\log p$ | 해석 |
|---|---|---|
| 균등 추측 (random init 직후) | 10.37 | random baseline |
| **이번 챕터 도달점** (위키 5K paragraphs × 2 epoch) | **7.49 - 7.50** | ← **실측** (eval loss, `set_seed` 로 재현) |
| 약하게 학습 (정답 확률 0.01) | 4.61 | |
| 잘 학습된 작은 BERT (정답 확률 0.05-0.1) | 2.3 - 3.0 | **이 셋업의 사정거리 밖** |
| 큰 사전학습 BERT (정답 확률 0.3+) | 1.20 | `klue/bert-base` 본체 수준 |

**이번 챕터가 도달하는 곳은 약 7.5** — *어떤 토큰이 흔한가* 를 막 새긴 단계입니다. 2.3-3.0 구간은 데이터·모델 크기가 몇 자릿수 더 필요해 이 셋업으로는 닿지 않습니다. 7.5 가 나왔다면 학습이 실패한 게 아니라 *정상* 입니다.

**관전 포인트** — Ch 20 의 영어 MLM 과 *비슷한 수렴 곡선* 이 나오는지가 본 챕터의 핵심 관찰. Ch 20 은 같은 셋업에서 eval loss 약 7.06-7.13 (`executed/20_en_bert_pretrain.ipynb`) — 한국어도 비슷한 자리에 멈춥니다. *언어가 달라도 작은 BERT + 5K 문장 MLM 의 학습 동역학은 비슷하다* 가 검증 가설.

## 토크나이저 노트 — 본 챕터의 핵심 한 자리

Ch 19 §5-4 의 cross-language 결론을 *실측* 으로 다시 확인합니다. 같은 한국어 문장을 *영어 토크나이저 (`bert-base-uncased`)* 와 *한국어 토크나이저 (`klue/bert-base`)* 에 통과시켜 토큰 리스트·UNK 개수를 비교 — 왜 한국어 데이터엔 한국어 토크나이저가 필요한가 의 직접 답.

> 이 비교 표는 *코드 셀 2 - 토크나이저 로드* 에서 직접 실행합니다. 여기서는 결론만 한 줄: **한국어 토크나이저는 한국어를 어절·형태소 단위로 자연스럽게 쪼개고 UNK 가 거의 없음. 영어 토크나이저는 한국어를 *자모 단위* 또는 *UNK 폭증* 으로 잘못 쪼갬.**

### 한국어 BERT 의 표준 토크나이저 — `klue/bert-base`

- `AutoTokenizer.from_pretrained("klue/bert-base")`
- vocab_size = 약 32,000 (한국어 WordPiece)
- 학습 코퍼스 = 한국어 위키 + 모두의 말뭉치 + 뉴스 + 댓글 → 한국어 일반 분포 + 비격식 텍스트 모두 커버
- 특수 토큰: `[PAD]=0`, `[UNK]=1`, `[CLS]=2`, `[SEP]=3`, `[MASK]=4`

> Ch 21 에서 `bert-base-uncased` 토크나이저를 *사전학습-fine-tune 전 구간* 에서 동일하게 썼듯, 이번 챕터의 `klue/bert-base` 토크나이저는 Ch 23 분류 fine-tune 까지 *그대로* 이어집니다. 토크나이저와 모델 본체는 *함께 가야 의미가 유지* 됩니다.

### `labels = -100` 한 줄 환기

`DataCollatorForLanguageModeling` 이 가려지지 않은 자리에 `labels = -100` 을 채워 *해당 위치의 CE loss 를 무시* 합니다 (PyTorch `CrossEntropyLoss` 의 `ignore_index` 기본값). 같은 트릭이 Phase 4 의 SFT (Ch 28) 에서 *prompt 자리를 가리는* 방식으로 다시 등장합니다 — *적용 자리만 정반대*. 한국어 MLM 에서도 트릭 자체는 *완전히 동일*.

## 이 장의 구성

[[SubPages]]
