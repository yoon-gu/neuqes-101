**목표**: Phase 3 의 두 번째 챕터. Ch 19 에서 *토크나이저를 직접 학습* 해 봤다면, 이번엔 **모델 본체를 직접 random init 해 사전학습** 합니다. 표준 BERT 보다 *훨씬 작은* (약 11M params) BERT 를 짜서 **일반 도메인 Wikitext-103** paragraphs 로 **Masked Language Modeling (MLM)** 사전학습. 원본 BERT 의 Wikipedia + BookCorpus 정신을 따라 *task 도메인이 아닌* 일반 위키 본문 사용 — Ch 21 의 분류 fine-tune (Yelp 리뷰(식당·업체)) 은 *완전히 다른 도메인* 으로 *일반 표상 → 다른 task* transfer 메시지가 정직해집니다. 토크나이저는 학습 안정성을 위해 표준 `bert-base-uncased` 를 그대로 가져옵니다.

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
| 모델 | 없음 | **작은 `BertForMaskedLM` (random init, 약 11M params)** |
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
| unigram — 빈도만 아는 단계 | 코퍼스 빈도 그대로 | **7.25** ← 이번 챕터가 넘어서야 할 기준선 |
| **이번 챕터 도달점** (5K paragraphs × 2 epoch) | — | **7.06 - 7.13** ← 실측 (train 7.07 / eval 7.06-7.13) |
| 약하게 학습 (정답 확률 0.01) | $0.01$ | 4.61 |
| 잘 학습된 작은 BERT (정답 확률 0.05-0.1) | $0.05$ - $0.1$ | 2.3 - 3.0 (이 셋업의 사정거리 밖) |
| 큰 사전학습 BERT (정답 확률 0.3+) | $0.3$ | 1.20 |
| 완벽 (정답 확률 1.0) | $1.0$ | 0.00 |

**관전 포인트**:
- 학습 첫 step 의 loss 가 약 10 부근이면 random init 직후 *균등 추측* 상태. 첫 100 step 안에 빠르게 떨어지면 vocab 정상.
- **이번 챕터의 목표는 `unigram 기준선 (7.25)` 을 넘어서는 데까지** 입니다. 실제로 2 epoch 뒤 train loss 약 7.07, eval loss 약 7.06-7.13 에 도달합니다 — *"어떤 토큰이 흔한가"* 를 막 새긴 단계. 표의 loss 는 train/eval 공통 척도이고, 이 챕터에선 두 값이 거의 붙어 있습니다. 모델 생성 직전에 `set_seed(SEED)` 를 걸어 두었으므로, 직접 돌려도 위 값이 소수점 둘째 자리까지 그대로 재현됩니다.
- *vocab 의 일부 후보를 추려내는* 단계 (2.5-4.0) 는 **이 셋업으로는 도달하지 않습니다.** 부록 `20_en_bert_pretrain_scaling` 에서 16 epoch 까지 늘려도 loss 는 약 6.5 (ppl 697) 에서 평탄해집니다 — 더 내리려면 epoch 이 아니라 **데이터** 를 늘려야 합니다 (🛠️ 변형 참조).
- 작은 모델 + 5K paragraphs + 2 epoch 으로 *완벽* 은 불가능 — 그러나 Ch 21 의 fine-tune 출발점으로는 충분.

### Perplexity (PPL)

언어 모델 표준 metric. $\text{PPL} = e^{L}$ — *모델이 다음 토큰을 평균 몇 후보 중에서 고민하는가* 의 직관:

| MLM loss | PPL | 해석 |
|---|---|---|
| 10.33 | 30,522 | 균등 (전체 vocab) |
| 7.25 | 1,412 | unigram — 빈도만 아는 단계 |
| **7.13** | **1,253** | ← **이번 챕터 2 epoch 실측** |
| 6.55 | 697 | 부록에서 16 epoch 까지 늘린 결과 |
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

## 이 장의 구성

[[SubPages]]
