**목표**: Phase 3 의 네 번째 챕터. Ch 20 에서 *영어 작은 BERT* 를 random init 해 MLM 사전학습 했다면, 이번엔 *완전히 같은 본체 구조* 로 **한국어 MLM 사전학습** 합니다. 변하는 축은 **언어** — 토크나이저는 `klue/bert-base` (한국어 WordPiece, vocab 약 32,000), 데이터는 **한국어 Wikipedia** (`wikimedia/wikipedia`, `20231101.ko`) paragraphs. 본체 hyperparam, loss, training args 는 Ch 20 과 동일. *Ch 23 의 분류 fine-tune (NSMC 영화 리뷰) 은 완전히 다른 도메인* — 일반 도메인 사전학습 → task 도메인 fine-tune 의 정직한 transfer 메시지.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 5-8분 (토크나이저 로드 + ko 위키 다운로드·paragraph split·토큰화가 대부분을 차지 + MLM 2 epoch 약 0.3분 + 평가/저장). 전체 소요는 데이터 다운로드가 지배합니다.


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

### 학습 목표 영역 (Ch 20 과 같음)

| 모델 상태 | $-\log p$ | 해석 |
|---|---|---|
| 균등 추측 (random init 직후) | 10.37 | random baseline |
| 약하게 학습 (정답 확률 0.01) | 4.61 | |
| 잘 학습된 작은 BERT (정답 확률 0.05-0.1) | 2.3 - 3.0 | 이번 챕터 목표 영역 |
| 큰 사전학습 BERT (정답 확률 0.3+) | 1.20 | `klue/bert-base` 본체 수준 |

**관전 포인트** — Ch 20 의 영어 MLM 과 *비슷한 수렴 곡선* 이 나오는지가 본 챕터의 핵심 관찰. *언어가 달라도 작은 BERT + 5K 문장 MLM 의 학습 동역학은 비슷하다* 가 검증 가설.

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

`DataCollatorForLanguageModeling` 이 가려지지 않은 자리에 `labels = -100` 을 채워 *해당 위치의 CE loss 를 무시* 합니다 (PyTorch `CrossEntropyLoss` 의 `ignore_index` 기본값). 같은 트릭이 Phase 4 의 SFT (Ch 27) 에서 *prompt 자리를 가리는* 방식으로 다시 등장합니다 — *적용 자리만 정반대*. 한국어 MLM 에서도 트릭 자체는 *완전히 동일*.

## 환경 셋업

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

## 한국어 Wikipedia 데이터 로드 — 일반 도메인 사전학습 코퍼스

원본 BERT 가 영어 Wikipedia + BookCorpus 라는 *일반 도메인* 코퍼스로 사전학습한 정신을 따라, 본 챕터도 **한국어 Wikipedia 본문** 으로 MLM 사전학습합니다 — *task 도메인 (NSMC 영화 리뷰) 으로 사전학습하면 domain-adaptive pretraining 에 가까워져 사전학습의 진짜 메시지 (일반 표상 학습 → 다른 task 로 transfer) 가 흐려지기 때문*.

**원본**: `wikimedia/wikipedia`, config `20231101.ko`. CC-BY-SA, HF Hub 정제본. article 단위 다운로드 후 paragraph 단위로 split 해 NSMC 5K 문장과 비슷한 토큰 양으로 맞춤. Ch 23 의 분류 fine-tune (NSMC 이진) 은 *완전히 다른 도메인* — 사전학습 → fine-tune transfer 메시지가 정직해집니다.

## 토크나이저 — `klue/bert-base` 로드 + 영어 토크나이저와 한국어 비교

`klue/bert-base` 의 한국어 WordPiece (vocab 약 32,000) 를 그대로 가져옵니다. *모델은 random init* 이지만 토크나이저는 *완성품* — Ch 20 의 영어 패턴과 동일.

이어서 같은 한국어 문장을 *영어 토크나이저* (`bert-base-uncased`, Ch 20 에서 사용) 와 비교해 Ch 19 §5-4 의 cross-language 결론을 *직접* 확인합니다.

### 같은 한국어 문장을 두 토크나이저로 — Ch 19 §5-4 cross-language 검증

영어 토크나이저 (`bert-base-uncased`) 와 한국어 토크나이저 (`klue/bert-base`) 에 같은 한국어 문장을 통과시켜 토큰 리스트와 UNK 개수를 비교합니다.

**관찰 — Ch 19 §5-4 결론의 실측 확인**

- **`bert-base-uncased` (영어)**: 한국어 문장이 *자모 단위* (`ᄋ`, `##ᅵ`, `##ᅧ` ...) 로 분해되거나 `[UNK]` 가 섞임. 토큰 수가 길게 폭증, *의미 단위* 가 사라짐. 모델이 이 표현으로 학습해도 *한국어 어휘 정보* 가 거의 없음.
- **`klue/bert-base` (한국어)**: 한국어 문장이 *어절·형태소* 단위 (`이`, `영화`, `정말`, `재미있`, `##어요`) 로 자연스럽게 쪼개짐. UNK 0개, 토큰 수가 짧고 *의미 단위* 가 보존.

> **결론** — 한국어 데이터로 BERT 를 사전학습하려면 한국어 토크나이저가 필수. Ch 20 의 영어 패턴을 한국어로 옮길 때 *토크나이저만 바꿔도* 같은 학습 동역학이 가능합니다. Ch 19 §5-4 가 *문제 제기* 였다면, 이번 챕터는 *해결책의 첫 단계*.

## 토큰화 + `group_texts` — Ch 20 패턴 그대로

MLM 사전학습 표준 입력 포맷. 모든 문서를 *이어 붙여 토큰 스트림* 으로 만든 뒤 `block_size=128` 단위로 자릅니다. 문장 경계가 사라지는 trade-off 는 있지만 BERT 사전학습은 *임의 위치의 토큰 예측* 이라 문장 경계가 중요하지 않습니다.

한국어 Wikipedia paragraphs 는 *제한 50-2000자 필터링* 으로 평균 문장 길이가 일정 (수십 자-수백 자). 5,000 paragraphs 이 `block_size=128` 로 잘리면 약 500-1,500 블록 정도로 정리됩니다. NSMC 한 줄 리뷰보다 길고 Yelp 보다는 짧은 중간 수준 — 일반 도메인 코퍼스다운 균형.

## 작은 `BertConfig` + `BertForMaskedLM` — random init (Ch 20 과 동일)

본체 구조는 Ch 20 과 *완전히 동일* — hidden=256, layer=4, head=4, intermediate=1024. vocab 만 한국어 토크나이저 (32,000) 에 맞춤.

**관찰** — vocab 이 약 32,000 (Ch 20 의 30,522 보다 약간 큼) 이라 임베딩 테이블이 살짝 더 큽니다. 그래도 본체 구조는 동일 — encoder body 2M + 임베딩 8M 수준의 작은 BERT.

> Ch 20 과 마찬가지로 MLM head 의 weight 는 입력 임베딩과 *tied* (공유). vocab 차원 출력 layer 가 임베딩 테이블과 같아 파라미터 절약.

## `DataCollatorForLanguageModeling` + Trainer 학습

collator 가 매 batch 마다 *무작위로 약 15% 토큰을 [MASK]* 로 바꾸고, 그 위치의 정답 토큰을 `labels` 로 표시. 나머지 위치는 `-100` → CrossEntropyLoss 가 무시.

**MLM masking 규칙** (BERT 원논문) — Ch 20 / Ch 21 과 동일:
- 선택된 약 15% 중 80%: 실제로 `[MASK]` 로 교체
- 10%: 무작위 다른 토큰으로 교체
- 10%: 원래 토큰 유지

이 규칙은 *언어와 무관* — collator 코드가 토큰 id 만 보고 처리합니다.

### 5-1. 🔍 [MASK] 가 들어가는 원리 — 한 눈에 보는 80/10/10 (한국어 풀버전)

`DataCollatorForLanguageModeling` 은 매 step 마다 *입력 토큰의 약 15%* 를 *무작위로* 선택하고, 선택된 위치마다 세 가지 중 하나를 적용합니다.

| 선택된 토큰 운명 | 비율 | 의도 |
| --- | --- | --- |
| `[MASK]` 로 교체 | **80%** | 표준 마스킹 — 모델이 *주변 문맥만으로* 원래 토큰을 맞추도록 |
| **다른 random 토큰** 으로 교체 | 10% | inference 때는 `[MASK]` 가 없으니, 모델이 *항상* 자기 입력을 *의심* 하게 만듦 |
| **원본 그대로** 유지 | 10% | 동일 — 입력과 정답이 일치하는 케이스도 학습 데이터에 포함 |

**나머지 85%** 의 토큰은 `labels = -100` 으로 두어 *loss 계산에서 제외* 됩니다 (PyTorch CE 의 `ignore_index` 기본값). 즉 한 step 의 MLM loss 는 *선택된 15% 자리만* 모아 평균한 값.

> 이 `labels = -100` 트릭은 BERT-만의 것이 아닙니다 — Phase 4 GPT 사전학습은 *거의 모든 토큰* 을 학습 (`labels = input_ids`), SFT (Ch 27) 는 *prompt 만 -100, 답변만 학습*. 같은 트릭, 정반대 자리. Ch 21 / 영어 짝과 동일한 풀버전 시각화로 한국어 환경에서도 직접 확인.

**관전 포인트**

- `what_happened` 가 `-` 인 자리 (약 85%) 는 *입력과 정답이 그대로* — loss 에 기여하지 않습니다. 모델은 *문맥을 만들어 주는* 역할만.
- `[MASK]` 자리 (약 12%) 가 본 task 의 *진짜 학습 신호*. 주변 한국어 토큰들의 attention 결과로 *가려진 자리* 의 vocab 분포를 예측.
- `random` (약 1.5%) 과 `kept` (약 1.5%) 는 *inference 분포 일치* 를 위한 정규화. 추론 시에는 `[MASK]` 가 없으므로 *입력을 절대 신뢰하면 안 된다* 는 신호를 학습에 섞어 줌. 영어 (Ch 20·21) 와 같은 규칙.
- 매 epoch · 매 batch 마다 마스킹은 *새로 무작위* — 같은 한국어 문장이 epoch 마다 다른 자리에서 가려져 학습됨 (data augmentation 효과).

> **결론 한 줄** — *`[MASK]` 트릭은 언어와 무관, 본체만 한국어를 학습.* `DataCollatorForLanguageModeling` 코드는 한국어든 영어든 *토큰 id 위에서만* 동작합니다. 언어 차이는 *학습된 임베딩의 의미* 에 반영될 뿐, masking 메커니즘 자체는 동일.

### 5-2. 학습 시작

Ch 20 과 같은 hyperparams — epoch 2, batch 32, lr 5e-4 (scratch 사전학습 표준), warmup 0.06, fp16 (T4).

### 학습 직전 baseline — 사전학습 전·후 비교 준비

`trainer.train()` 을 호출하기 *전* 의 모델 상태 (`BertForMaskedLM(config)` random init) 로 두 가지를 측정해 둡니다 — *학습 후와 나란히* 보면 *사전학습이 본체에 무엇을 새겼는지* 가 한 화면에 드러납니다.

1. **`eval_loss` / `perplexity`** — random init 이므로 vocab 32,000 균등 분포 (`ln V` ≈ 10.37) 근처가 기대치.
2. **같은 문장의 `[MASK]` top-5** — random init 의 logits 는 거의 균등이라 *문맥과 무관한 토큰* (자주 등장하는 조사·어미·특수문자 등) 이 뽑힙니다.

학습이 끝난 뒤 7번 셀에서 *완전히 같은 문장* 으로 다시 측정해 *직접 비교* 합니다.

## 학습 결과 — Loss / Perplexity 곡선

학습이 *실제로 진행* 됐는지 세 각도로 확인:
1. step-by-step train loss 곡선 — 빠르게 약 10.37 (random baseline) 에서 5 이하로 떨어졌는지
2. eval set 의 perplexity — 외부 텍스트에서도 일관된 수준인지
3. 임의 한국어 문장에 `[MASK]` 를 끼워 top-5 후보 출력 — *어떤 한국어 토큰을 예측하는지* 정성 평가

## 사전학습 전·후 비교 — random init 본체 vs 2 epoch 학습 후

학습 직전 (5-2 마지막 셀에서 측정해 둔 `pre_eval_loss` / `pre_top5_records`) 와 *완전히 같은 문장·같은 평가 셋* 에 학습 후 모델을 적용해 두 결과를 나란히 봅니다. *사전학습이 본체에 무엇을 새겼는가* 의 가장 직접적인 증거.

### 7-1. eval_loss / perplexity — 수치 비교

두 측정치를 한 표·한 막대 그래프로.

### 7-2. 🏆 학습이 *충분히 잘 된 경우* 의 기준점 — 표준 `klue/bert-base` 비교

우리 작은 BERT (10M, 한국어 위키 5K paragraphs × 2 epoch) 의 top-5 가 *방향성은 맞지만 정답이 잘 안 보이는* 이유는 단순합니다 — **학습 데이터·모델 크기·학습 시간 모두 부족**. *그럼 학습이 충분히 잘 되면 어떤 결과가 나오나?* 의 답을 같은 한국어 문장에 표준 `klue/bert-base` (110M, 약 8.4B 토큰 대규모 한국어 코퍼스) 를 적용해 직접 봅니다.

같은 토크나이저 (`klue/bert-base`) 를 쓰고 있으므로 *모델만 바꿔* 두 결과를 나란히.

### 7-3. [MASK] top-5 — 3-way 비교 (before / ours / reference klue/bert-base)

같은 한국어 문장 4개의 [MASK] 자리 top-5 후보를 *사전학습 전 → 우리 작은 BERT 학습 후 → 표준 klue/bert-base* 셋으로 나란히.

**해석 가이드 — 사전학습이 만든 차이**

- **`eval_loss`**: random baseline `ln V ≈ 10.37` 에서 약 5-7 부근까지 떨어졌으면 본체가 *언어 구조 일부* 를 학습. *완전한* 한국어 표상은 아니어도 `klue/bert-base` 가 학습한 것의 *방향* 은 맞춤.
- **`perplexity`**: 32,000 (vocab 전체) 에서 수십-수백 부근으로. *마스크 자리마다 후보를 약 50-500 개로 좁힌 상태* 라는 직관적 해석.
- **top-5 토큰** (3-way 비교):
  - *before (random)*: 자주 등장하는 *조사·어미·특수문자* (`##요`, `##어`, `.`, `는`, `이`) — random init 이지만 logits 가 미세하게 흔들려 *통계적 빈도* 높은 토큰만 뽑힘.
  - *ours (small BERT, 위키 5K paragraphs × 2 epoch)*: 한국어 어미·내용어 일부가 섞이기 시작 — 위키 도메인은 *방향성이 보이지만* 정답 (`서울`, `8` 등) 이 top-5 안에 *안정적으로* 들어오지는 못함. **데이터·모델 크기 부족의 한계**.
  - *ref (klue/bert-base, 약 8.4B 토큰)*: 위키 도메인은 *정답이 top-1* — `서울`, `여덟` 같은 자연스러운 답. NSMC 도메인 (다른 도메인) 도 *감성 형용사·부사* (`재미있`, `정말`, `너무`) 가 자연스럽게 top-5 에 들어옴. **이게 사전학습이 충분히 잘 됐을 때의 모습**.

> **세 모델의 격차가 정확히 *데이터 규모 + 모델 크기 + 학습 시간* 의 격차** — 우리 작은 BERT (10M, 위키 5K paragraphs, 2 epoch) → reference (110M, 약 8.4B tokens) 사이에 *데이터 수천 배, 파라미터 11배*. 그 격차가 top-5 의 *질적 차이* 로 정확히 드러납니다.

이번 챕터의 작은 BERT 는 *한국어 위키 paragraphs 5K × 2 epoch* 로 학습한 *일반 도메인 mini BERT*. 위키 도메인은 직접 본 분포라 향상이 빠르지만, NSMC 영화 리뷰는 *다른 도메인* 이라 fine-tune 단계에서 적응이 필요합니다 — 이게 *진짜 사전학습 → fine-tune 패러다임* 의 핵심. Ch 23 에서 NSMC 이진 분류로 fine-tune 할 때 진짜 비교 — *우리가 직접 만든 작은 한국어 BERT (일반 도메인 5K, 약 10M)* vs *Ch 15 의 `klue/bert-base` (대규모 일반 코퍼스, 약 110M)*.

## 모델 저장 — Ch 23 에서 재사용

`model.save_pretrained()` 와 `tokenizer.save_pretrained()` 를 *같은 폴더* 에 저장. Ch 23 에서는 `AutoModelForSequenceClassification.from_pretrained("./ch22_small_bert_mlm_ko", num_labels=2)` 한 줄로 *이 BERT body* 를 가져와 분류 헤드를 새로 얹습니다.

**저장된 파일 구조** — Ch 20 과 동일한 HF 표준 레이아웃:

| 파일 | 역할 |
|---|---|
| `config.json` | `BertConfig` 직렬화 (hidden, layer, head, vocab 등) |
| `model.safetensors` (또는 `pytorch_model.bin`) | 모델 weight |
| `tokenizer.json` / `vocab.txt` | 한국어 토크나이저 (Ch 23 fine-tune 에서 동일 사용) |
| `special_tokens_map.json`, `tokenizer_config.json` | 특수 토큰 메타 |

> Ch 23 에서 `AutoModelForSequenceClassification.from_pretrained("./ch22_small_bert_mlm_ko", num_labels=2)` 호출 시, `BertForMaskedLM` 의 *MLM head 는 버려지고* encoder body 만 가져옴. 그 위에 새 `Linear(256, 2)` 분류 헤드를 random init 으로 부착. Ch 15 의 `klue/bert-base` fine-tune 과 *같은 구조* — 본체 출발점 (사전학습 규모) 만 다름.

## 이 장의 구성

- [22-1. 실습](22-ko_bert_pretrain-practice.md)
- [22-2. 변형 — 데이터 / 학습량 / 다른 한국어 코퍼스](22-ko_bert_pretrain-variation.md)
- [22-3. 정리와 FAQ](22-ko_bert_pretrain-wrapup.md)
