**목표**: Phase 3 의 두 번째 챕터. Ch 19 에서 *토크나이저를 직접 학습* 해 봤다면, 이번엔 **모델 본체를 직접 random init 해 사전학습** 합니다. 표준 BERT 보다 *훨씬 작은* (약 10M params) BERT 를 짜서 **일반 도메인 Wikitext-103** paragraphs 로 **Masked Language Modeling (MLM)** 사전학습. 원본 BERT 의 Wikipedia + BookCorpus 정신을 따라 *task 도메인이 아닌* 일반 위키 본문 사용 — Ch 21 의 분류 fine-tune (Yelp 리뷰(식당·업체)) 은 *완전히 다른 도메인* 으로 *일반 표상 → 다른 task* transfer 메시지가 정직해집니다. 토크나이저는 학습 안정성을 위해 표준 `bert-base-uncased` 를 그대로 가져옵니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 5-8분 (`bert-base-uncased` 토크나이저 로드 + Wikitext-103 다운로드·필터링·토큰화가 대부분을 차지 + MLM 2 epoch 약 0.4분 + 평가/저장). 전체 소요는 데이터 다운로드가 지배합니다.


## 학습 흐름

1. 🔤 **토크나이저**: `bert-base-uncased` WordPiece (vocab 30,522) 그대로 로드
2. 📥 **데이터**: `Salesforce/wikitext`, config `wikitext-103-raw-v1` paragraphs 5,000 (일반 도메인, 라벨 없음)
3. 🚀 **토큰화 + `group_texts`**: HF `run_mlm.py` 표준 — 모든 텍스트를 이어붙여 토큰 스트림으로 만든 뒤 `block_size=128` 단위로 자름
4. 🏗️ **모델 구성**: `BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024)` + `BertForMaskedLM(config)` random init
5. 🚀 **학습**: `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` + Trainer, fp16, 2 epoch
6. 🔬 **평가**: MLM loss 학습 곡선, perplexity, masked token 예측 시연 ([MASK] top-5 후보 — 위키 도메인 + Yelp 도메인 혼합)
7. 💾 **저장**: `model.save_pretrained("./ch20_small_bert_mlm")` — Ch 21 에서 `from_pretrained` 로 재사용


> 📒 **사전 학습 자료**: Ch 19 (토크나이저 직접 학습) — 토크나이저가 "어떻게 만들어지는지" 를 본 뒤, 이번 챕터는 *모델이 어떻게 사전학습되는지* 를 봅니다. 둘이 합쳐져 "사전학습된 BERT 를 가져다 쓰는" 흐름 (Ch 7-18) 의 *안쪽* 이 드러납니다.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 17 | klue/bert-base | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | sigmoid (각각) | `BCEWithLogitsLoss` |
| 18 | klue/bert-base + 보조 | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | sigmoid + 태스크별 | `BCEWithLogitsLoss + λ·L_aux` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — | — |
| **20 ← 여기** | **작은 BERT (직접, scratch)** | **`bert-base-uncased` 토크나이저 (가져옴)** | **Wikitext-103 paragraphs (일반 도메인)** | **MLM head** | softmax (MLM) | **`CrossEntropyLoss` (masked token)** |
| 21 (다음) | Ch 20 사전학습 BERT + 분류 헤드 | (Ch 20과 동일) | Yelp 이진화 (다른 도메인 transfer) | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

**Phase 3 안에서의 위치** — Ch 19 (토크나이저 학습) → Ch 20 (모델 사전학습) → Ch 21 (분류 fine-tune). 사전학습 모델을 *받아서 fine-tune* 하던 Phase 1·2 의 흐름을 이번엔 *직접 만들어* 봅니다.

## 변경점 (Diff from Ch 19)

| 축 | Ch 19 (토크나이저 학습 전용) | Ch 20 (작은 BERT scratch MLM) |
|---|---|---|
| **이 챕터의 task** | 토크나이저 학습 (모델 없음) | **모델 사전학습 (MLM)** ← *유일한 변화* |
| 모델 | 없음 | **작은 `BertForMaskedLM` (random init, 약 10M params)** |
| 토크나이저 | WordPiece + WordLevel *직접 학습* | **`bert-base-uncased` *가져옴*** (vocab 30,522) |
| 데이터 | Yelp text + NSMC text (vocab 학습용) | **Wikitext-103 paragraphs (일반 도메인 MLM 학습용)** |
| Loss | 없음 (vocab + merge rules 가 산출물) | **`CrossEntropyLoss` (masked token 위치만)** |
| 산출물 | tokenizer json 파일 4 종 | **모델 체크포인트** (`./ch20_small_bert_mlm`) — Ch 21 재사용 |

> **변경점 한 가지 원칙** — Phase 3 안에서 *모델 축* 이 변합니다 (없음 → 작은 BERT scratch). 토크나이저는 *직접 학습이 가능함을 본 뒤* 표준으로 돌아옵니다 — `bert-base-uncased` 가 *공개되어 검증된* vocab 이라 사전학습 안정성이 높음. 토크나이저 *학습 절차* 와 *모델 학습 절차* 가 두 챕터로 분리되어 각각의 메커니즘이 또렷이 보이게 합니다.

### 왜 토크나이저는 가져오고 모델만 직접 학습하나

(1) **vocab 신뢰성** — Ch 19 의 vocab 8K 토크나이저는 5K 문장 학습이라 어휘 커버리지가 좁음. `bert-base-uncased` 의 30,522 vocab 은 Wikipedia + BookCorpus 로 학습되어 *영어 일반 분포* 를 잘 표현. 모델 학습이 vocab 노이즈에 영향받지 않음. (2) **다음 챕터 호환** — Ch 21 에서 같은 토크나이저로 분류 fine-tune 하면, *문체가 다른* downstream 입력에도 안정. (3) **표준 패턴** — 실무에서도 보통 *모델은 직접 사전학습하지만 vocab 은 검증된 것* 을 쓰는 패턴 (예: HF `roberta-base` 도 GPT-2 의 BPE 그대로 가져옴).

### 왜 task corpus (Yelp) 가 아니라 일반 위키인가 — 원본 BERT 의 정신

원본 BERT (Devlin et al., 2018) 는 *영어 Wikipedia + BookCorpus* 라는 **일반 도메인** 코퍼스로 사전학습한 뒤, *완전히 다른 downstream task* (GLUE, SQuAD 등) 로 fine-tune 했습니다. 본 챕터도 같은 패턴 — Wikitext-103 *일반 위키 paragraphs* 로 MLM 사전학습 → Ch 21 에서 *Yelp 리뷰(식당·업체)* 라는 *다른 도메인* 으로 transfer.

만약 Yelp text 로 MLM 사전학습한 뒤 Yelp 분류로 fine-tune 하면 *domain-adaptive pretraining* (DAPT) 에 가까워져 *일반 표상 학습 → 다른 task transfer* 의 본질이 흐려집니다. *일반 도메인 → 다른 도메인 transfer* 가 *진짜 사전학습-fine-tune 패러다임*. Ch 22 (한국어 위키 → NSMC 분류) 가 같은 패턴.

## Loss 함수의 변화 — Masked Language Modeling (MLM)

이전 분류 챕터들 (Ch 11-18) 의 loss 는 *문장 한 개에 라벨 하나*. 이번 챕터는 *문장 안의 가려진 토큰들* 을 맞춰야 합니다 — 토큰 위치 하나하나가 *분류 task* 가 됩니다.

### 수식

입력 토큰 시퀀스 $x = (x_1, \dots, x_n)$ 의 일부를 무작위로 `[MASK]` 로 가린 뒤, 모델이 *원래 토큰* 을 예측:

$$L_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid x_{\setminus M})$$

- $M$: 가려진 위치 집합 (전체 토큰의 15%)
- $P(x_i \mid x_{\setminus M})$: 모델이 $i$ 번 위치에 *원래 토큰* 을 예측할 확률 (vocab 30,522 차원 softmax)
- $|M|$: 가려진 토큰 수로 평균

각 가려진 위치에서 *vocab 전체에 대한 `CrossEntropyLoss`*. 분류 헤드의 K (이전 챕터들의 2, 5, 7) 가 이번엔 **V = 30,522** 로 폭증.

### 숫자로 감 잡기 (vocab=30,522)

| 모델 상태 | 정답 토큰 확률 | $-\log p$ |
|---|---|---|
| 균등 추측 (random init 초기) | $1/30522 \approx 3.28 \times 10^{-5}$ | **10.33** ← random baseline |
| 약하게 학습 (정답 확률 0.01) | $0.01$ | 4.61 |
| 잘 학습된 작은 BERT (정답 확률 0.05-0.1) | $0.05$ - $0.1$ | **2.3 - 3.0** ← 이번 챕터 목표 영역 |
| 큰 사전학습 BERT (정답 확률 0.3+) | $0.3$ | 1.20 |
| 완벽 (정답 확률 1.0) | $1.0$ | 0.00 |

**관전 포인트**:
- 학습 첫 step 의 loss 가 약 10 부근이면 random init 직후 *균등 추측* 상태. 첫 100 step 안에 빠르게 떨어지면 vocab 정상.
- 목표는 *vocab 의 일부 후보를 추려내는* 단계 (약 2.5-4.0). 작은 모델 + 5K paragraphs + 2 epoch 으로 *완벽* 은 불가능 — 그러나 Ch 21 의 fine-tune 출발점으로는 충분.

### Perplexity (PPL)

언어 모델 표준 metric. $\text{PPL} = e^{L}$ — *모델이 다음 토큰을 평균 몇 후보 중에서 고민하는가* 의 직관:

| MLM loss | PPL | 해석 |
|---|---|---|
| 10.33 | 30,522 | 균등 (전체 vocab) |
| 5.0 | 148 | vocab 의 일부로 좁혀짐 |
| 3.0 | 20 | 20 개 후보 중에서 결정 |
| 1.0 | 2.7 | 거의 결정적 |

> 이전 분류 챕터의 `random baseline = log K` (K=2 → 0.69, K=7 → 1.95) 와 같은 직관을 *vocab 차원에 확장* 한 게 MLM. `ln(30522) ≈ 10.33` 이 그 random baseline.

## 토크나이저 노트

이번 챕터부터 토크나이저는 *표준 사전학습 모델 것* 을 가져옵니다.

- `AutoTokenizer.from_pretrained("bert-base-uncased")` — 영어 BERT 의 표준 WordPiece.
- vocab_size = 30,522 (Ch 19 의 8K 와 비교해 약 4 배).
- 학습 코퍼스 = Wikipedia (영어) + BookCorpus → 영어 일반 분포가 잘 반영.
- 특수 토큰: `[PAD]=0`, `[UNK]=100`, `[CLS]=101`, `[SEP]=102`, `[MASK]=103`.

### 같은 문장의 토큰화 — Ch 19 직접 학습 vs Ch 20 가져옴

`"The capital of France is Paris, located on the Seine river."` 같은 *일반 위키풍* 문장이:

- **Ch 19 의 8K WordPiece (Yelp 5K 학습)**: 학습 코퍼스 (Yelp) 분포에 *최적화* 되어 있어 *위키 도메인 단어* (`Seine`, `capital`, `located` 등) 가 작은 조각으로 쪼개짐 — Yelp 에 적게 등장하는 어휘일수록 더 잘게 분할.
- **Ch 20 의 `bert-base-uncased` 30K WordPiece (Wiki + BookCorpus 학습)**: 위키 어휘를 *덜 쪼개짐*. 30K vocab + 일반 영어 학습이라 일반 도메인 + Yelp 같은 task 도메인 *둘 다* 폭넓게 커버.

본 챕터의 사전학습 데이터 (Wikitext-103) 와 토크나이저 (`bert-base-uncased`) 가 *둘 다 일반 위키 분포* 라 *vocab 미스매치* 가 작습니다. Ch 21 에서 Yelp 분류로 fine-tune 할 때도 같은 토크나이저로 *문체 다른* 도메인을 처리 — 일반 영어 어휘는 거의 덜 쪼개지고, *Yelp 리뷰 특유 표현* 만 약간 더 분할되는 정도.

### "토크나이저는 모델과 운명공동체"

Ch 19 §5-4 의 cross-language 실험에서 봤듯, *학습 언어가 다른* 토크나이저를 모델에 끼우면 거의 100% UNK 가 됩니다. 모델 weight 와 vocab 은 *함께 학습되어* 그 vocab 의 토큰 임베딩 공간에서 의미를 형성합니다.

이번 챕터에서 *토크나이저는 vocab 만 빌려오고 모델은 random init* 입니다 — 즉 *vocab 구조* 와 *토큰 임베딩 의미* 가 분리됩니다. 학습 초기에는 임베딩이 random 이라 vocab 구조의 가치가 안 보이지만, MLM 으로 학습이 진행되면 임베딩이 vocab 구조에 *맞춰 정렬* 됩니다 — *이 챕터의 본질이 바로 그 정렬 과정*.

> Ch 21 부터는 *이 챕터의 모델 + 토크나이저 쌍* 을 통째로 가져가 fine-tune. 둘은 *함께 가야* 의미가 유지됩니다.

## 환경 셋업

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

## 토크나이저 — `bert-base-uncased` 그대로 로드

vocab 30,522 의 영어 WordPiece. *모델은 random init* 이지만 토크나이저는 *완성품* 을 가져옵니다.

## 데이터 — Wikitext-103 paragraphs (일반 도메인 사전학습 코퍼스)

원본 BERT 가 영어 Wikipedia + BookCorpus 라는 *일반 도메인* 코퍼스로 사전학습한 정신을 따라, 본 챕터도 **Wikitext-103** 본문으로 MLM 사전학습합니다 — *task 도메인 (Yelp 리뷰(식당·업체)) 으로 사전학습하면 domain-adaptive pretraining 에 가까워져 사전학습의 진짜 메시지 (일반 표상 학습 → 다른 task 로 transfer) 가 흐려지기 때문*.

**원본**: `Salesforce/wikitext`, config `wikitext-103-raw-v1` (CC-BY-SA, 정제된 영문 위키 paragraphs). HF Hub 정제본 — line 단위로 이미 정리되어 있어 빈 줄 / 너무 짧은 줄 / 너무 긴 줄만 제외하면 바로 사용 가능. Ch 21 의 분류 fine-tune (Yelp 이진) 은 *완전히 다른 도메인* — 사전학습 → fine-tune transfer 메시지가 정직해집니다. Ch 22 의 한국어 (한국어 Wikipedia paragraphs) 와 *대칭* 패턴.

## 토큰화 + `group_texts` — HF `run_mlm.py` 표준 패턴

MLM 사전학습의 표준 입력 포맷은 *고정 길이 블록*. 변동 길이 문장에 그대로 padding 하면 *손실*: (a) 짧은 문장이 많으면 PAD 비율이 높아 GPU 시간 낭비, (b) 긴 문장은 truncation 으로 정보 손실.

**해결책**: 모든 문서를 *이어 붙여 토큰 스트림* 으로 만든 뒤, `block_size=128` 단위로 자름. 문장 경계가 사라지는 trade-off 가 있지만, BERT 사전학습은 *임의 위치의 토큰 예측* 이라 문장 경계가 중요하지 않음.

Wikitext-103 paragraphs 는 *제한 50-2000자 필터링* 으로 평균 문장 길이가 일정 (수백 자 위주). 5,000 paragraphs 가 `block_size=128` 로 잘리면 약 5,352 블록 (약 68만 토큰) 으로 정리됩니다. 코드는 HF `examples/pytorch/language-modeling/run_mlm.py` 의 `group_texts` 함수를 그대로 따른 표준 패턴.

## 작은 `BertConfig` + `BertForMaskedLM` — random init

표준 `bert-base-uncased` 는 hidden=768, layer=12, head=12, intermediate=3072 = **110M params** — T4 에서 scratch 학습은 *수일* 필요.

이번 챕터는 *입문용 작은 BERT* 로 축소:

| hyperparam | 표준 `bert-base-uncased` | 이번 챕터 (작은 BERT) |
|---|---|---|
| `hidden_size` | 768 | **256** |
| `num_hidden_layers` | 12 | **4** |
| `num_attention_heads` | 12 | **4** |
| `intermediate_size` | 3072 | **1024** |
| `max_position_embeddings` | 512 | **128** (BLOCK_SIZE 와 같음) |
| 총 파라미터 | 약 110M | **약 10M** (toy 규모) |

크기는 1/10 이지만 *MLM 학습이 진행되는지* 보기에는 충분. Ch 21 에서 분류 fine-tune 할 때 성능 비교가 진짜 결과.

**관찰** — 작은 BERT 의 파라미터는 *임베딩 테이블이 절반 이상* 차지합니다 (vocab 30522 × hidden 256 ≈ 7.8M). encoder body 자체는 약 2M. 이게 *vocab 큰데 모델이 작은* 셋업의 특징 — 표준 BERT (vocab 30K × hidden 768 ≈ 23M / 110M = 21%) 와 비율이 매우 다릅니다.

> MLM head 의 weight 는 입력 임베딩과 *tied* (공유) — `BertForMaskedLM` 기본 동작. vocab 차원 출력 layer 가 임베딩 테이블과 같아 파라미터 절약.

## `DataCollatorForLanguageModeling` + Trainer 학습

collator 가 매 batch 마다 *무작위로 15% 토큰을 [MASK]* 로 바꾸고, 그 위치의 정답 토큰을 `labels` 로 표시 (나머지 위치는 -100 → CrossEntropyLoss 무시).

**MLM masking 규칙** (BERT 원논문):
- 선택된 15% 중 80%: 실제로 `[MASK]` 로 교체
- 10%: 무작위 다른 토큰으로 교체
- 10%: 원래 토큰 유지

이 비율은 *모델이 [MASK] 토큰 자체에 과도하게 의존하지 않게* 하는 트릭. `DataCollatorForLanguageModeling` 이 자동 처리.

### [MASK] 가 들어가는 원리 — 한 눈에 보는 80/10/10

`DataCollatorForLanguageModeling` 은 매 step 마다 *입력 토큰의 약 15%* 를 *무작위로* 선택하고, 선택된 위치마다 세 가지 중 하나를 적용합니다.

| 선택된 토큰 운명 | 비율 | 의도 |
| --- | --- | --- |
| `[MASK]` 로 교체 | **80%** | 표준 마스킹 — 모델이 *주변 문맥만으로* 원래 토큰을 맞추도록 |
| **다른 random 토큰** 으로 교체 | 10% | inference 때는 `[MASK]` 가 없으니, 모델이 *항상* 자기 입력을 *의심* 하게 만듦 |
| **원본 그대로** 유지 | 10% | 동일 — 입력과 정답이 일치하는 케이스도 학습 데이터에 포함 |

**나머지 85%** 의 토큰은 `labels = -100` 으로 두어 *loss 계산에서 제외* 됩니다 (PyTorch CE 의 `ignore_index` 기본값). 즉 한 step 의 MLM loss 는 *선택된 15% 자리만* 모아 평균한 값.

> 이 `labels = -100` 트릭은 BERT-만의 것이 아닙니다 — Phase 4 GPT 사전학습은 *거의 모든 토큰* 을 학습 (`labels = input_ids`), SFT (Ch 27) 는 *prompt 만 -100, 답변만 학습*. 같은 트릭, 정반대 자리. Ch 21 에서 더 자세히.

**관전 포인트**

- `what_happened` 가 `—` 인 자리 (약 85%) 는 *입력과 정답이 그대로* — loss 에 기여하지 않음. 모델은 *문맥을 만들어 주는* 역할만.
- `[MASK]` 자리 (약 12%) 가 본 task 의 *진짜 학습 신호*. 주변 토큰들의 attention 결과로 *가려진 자리* 의 vocab 분포를 예측.
- `random` (약 1.5%) 와 `kept` (약 1.5%) 는 *inference 분포 일치* 를 위한 정규화. 추론 시에는 `[MASK]` 가 없으므로 *입력을 절대 신뢰하면 안 된다* 는 신호를 학습에 섞어 줌.
- 매 epoch · 매 batch 마다 마스킹은 *새로 무작위* — 같은 문장이 epoch 마다 다른 자리에서 가려져 학습됨 (data augmentation 효과).

### 학습 직전 baseline — 사전학습 전·후 비교 준비

`trainer.train()` 을 호출하기 *전* 의 모델 상태 (`BertForMaskedLM(config)` random init) 로 두 가지를 측정해 둡니다 — *학습 후와 나란히* 보면 *사전학습이 본체에 무엇을 새겼는지* 가 한 화면에 드러납니다.

1. **`eval_loss` / `perplexity`** — random init 이므로 vocab 30,522 균등 분포 (`ln V` ≈ 10.33) 근처가 기대치.
2. **같은 문장의 `[MASK]` top-5** — random init 의 logits 는 거의 균등이라 *문맥과 무관한 토큰* (자주 등장하는 관사·전치사·특수문자 등) 이 뽑힙니다.

학습이 끝난 뒤 6-1 셀에서 *완전히 같은 문장* 으로 다시 측정해 *직접 비교* 합니다.

## 평가 — MLM loss 곡선 + perplexity + masked token 예측

학습이 *실제로 진행* 됐는지 세 각도로 확인:
1. step-by-step train loss 곡선 — 빠르게 10.33 (random baseline) → 약 7 부근으로 떨어졌는지
2. eval set 의 perplexity — 외부 텍스트에서도 일관된 수준인지
3. 임의 문장에 `[MASK]` 를 끼워 top-5 후보 출력 — *어떤 단어를 예측하는지* 정성 평가

### 6-1. 🔬 사전학습 전·후 비교 — random init 본체 vs 2 epoch 학습 후

학습 직전 (5번 마지막 셀에서 측정해 둔 `pre_eval_loss` / `pre_top5_records`) 와 *완전히 같은 문장·같은 평가 셋* 에 학습 후 모델을 적용해 두 결과를 나란히 봅니다. *사전학습이 본체에 무엇을 새겼는가* 의 가장 직접적인 증거.

### 6-2. eval_loss / perplexity — 수치 비교

두 측정치를 한 표·한 막대 그래프로.

### 6-3. 🏆 학습이 *충분히 잘 된 경우* 의 기준점 — 표준 `bert-base-uncased` 비교

우리 작은 BERT (10M, 5K paragraphs × 2 epoch) 의 top-5 가 *방향성은 맞지만 정답이 잘 안 보이는* 이유는 단순합니다 — **학습 데이터·모델 크기·학습 시간 모두 부족**. *그럼 학습이 충분히 잘 되면 어떤 결과가 나오나?* 의 답을 같은 문장에 표준 `bert-base-uncased` (110M, 위키+BookCorpus 약 33억 토큰) 를 적용해 직접 봅니다.

같은 토크나이저 (`bert-base-uncased`) 를 쓰고 있으므로 *모델만 바꿔* 두 결과를 나란히.

### 6-4. [MASK] top-5 — 3-way 비교 (before / ours / reference BERT)

같은 문장 4개의 [MASK] 자리 top-5 후보를 *사전학습 전 → 우리 작은 BERT 학습 후 → 표준 bert-base-uncased* 셋으로 나란히.

**해석 가이드 — 사전학습이 만든 차이**

- **`eval_loss`**: random baseline `ln V ≈ 10.33` 에서 약 7 부근까지 떨어졌으면 본체가 *언어 구조 일부* 를 학습. *완전한* BERT 수준은 아니어도 표준 BERT 가 학습한 것의 *방향* 은 맞춤.
- **`perplexity`**: 30,522 (vocab 전체) 에서 약 1,200 부근으로. *마스크 자리마다 후보를 약 1,200 개로 좁힌 상태* 라는 직관적 해석.
- **top-5 토큰** (3-way 비교):
  - *before (random)*: 자주 등장하는 *관사·전치사·특수문자* (`the`, `a`, `,`, `.`, `of`) — random init 이지만 logits 가 미세하게 흔들려 *통계적 빈도* 높은 토큰만 뽑힘.
  - *ours (small BERT, 5K paragraphs × 2 epoch)*: 위키 도메인은 *방향성이 보이기 시작* — 일반 부사·형용사, 위키 어휘 일부. 다만 정답 토큰 (`paris`, `0` 등) 이 top-5 안에 *안정적으로* 들어오지는 못함. **데이터·모델 크기 부족의 한계**.
  - *ref (bert-base-uncased, 약 33억 토큰 × 40 epoch)*: 위키 도메인은 *정답이 top-1* — `paris`, `zero` 같은 자연스러운 답. Yelp 도메인 (다른 도메인) 도 *감성 형용사* (`amazing`, `delicious`, `highly`) 가 자연스럽게 top-5 에 들어옴. **이게 사전학습이 충분히 잘 됐을 때의 모습**.

> **세 모델의 격차가 정확히 *데이터 규모 + 모델 크기 + 학습 시간* 의 격차** — 우리 작은 BERT (10M, 5K paragraphs, 2 epoch) → reference (110M, 3.3B tokens, 40 epoch) 사이에 *데이터 약 5,000배, 파라미터 11배, epoch 20배*. 그 격차가 top-5 의 *질적 차이* 로 정확히 드러납니다.

이번 챕터의 작은 BERT 는 *Wikitext-103 5K paragraphs × 2 epoch* 로 학습한 *일반 도메인 mini BERT*. 위키 도메인은 직접 본 분포라 향상이 빠르지만, Yelp 리뷰(식당·업체) 영역은 *다른 도메인* 이라 fine-tune 단계에서 적응이 필요합니다 — 이게 *진짜 사전학습 → fine-tune 패러다임* 의 핵심. Ch 21 에서 Yelp 이진 분류로 fine-tune 할 때 진짜 transfer 비교 — *우리가 직접 만든 작은 영어 BERT (일반 위키 5K, 약 10M)* vs *Ch 10 의 DistilBERT (대규모 Wikipedia+BookCorpus, 약 66M)* vs *random init baseline*.

## 모델 저장 — Ch 21 에서 재사용

`model.save_pretrained()` 와 `tokenizer.save_pretrained()` 를 *같은 폴더* 에 저장. Ch 21 에서는 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 한 줄로 *이 BERT body* 를 가져와 분류 헤드를 새로 얹습니다.

**저장된 파일 구조** — `from_pretrained` 가 인식하는 HF 표준 레이아웃:

| 파일 | 역할 |
|---|---|
| `config.json` | `BertConfig` 직렬화 (hidden, layer, head, vocab 등) |
| `model.safetensors` (또는 `pytorch_model.bin`) | 모델 weight |
| `tokenizer.json` / `vocab.txt` | 토크나이저 (Ch 21 fine-tune 에서 같은 토크나이저 사용) |
| `special_tokens_map.json`, `tokenizer_config.json` | 특수 토큰 메타 |

> Ch 21 에서 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 호출 시, `BertForMaskedLM` 의 *MLM head 는 버려지고* encoder body 만 가져옴. 그 위에 새 `Linear(256, 2)` 분류 헤드를 random init 으로 부착 — Ch 7-18 의 fine-tune 셋업과 *동일한 구조*. 이번 챕터의 사전학습이 *얼마나 도움 됐는지* 가 Ch 21 의 학습 곡선에서 직접 비교됩니다.

## 이 장의 구성

[[SubPages]]
