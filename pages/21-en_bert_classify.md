**목표**: Phase 3 의 세 번째 챕터. Ch 20 에서 *작은 BERT 를 일반 도메인 (Wikitext-103) 으로 직접 MLM 사전학습* 했다면, 이번엔 그 위에 **분류 헤드를 얹어 *완전히 다른 도메인 (Yelp 리뷰(식당·업체))* 이진 분류로 fine-tune** 합니다. Ch 10 (DistilBERT, 약 66M params, 대규모 Wikipedia + BookCorpus 사전학습) 과 같은 Yelp 이진 분류 셋업에 *우리가 만든 작은 BERT* (약 10M params, Wikitext-103 2K paragraphs × 3 epoch MLM) 를 붙여 두 결과를 나란히 비교 — 둘 다 *일반 도메인 → Yelp transfer* 라 비교가 *fair*, *사전학습 규모* 차이만 측정됨.

본 챕터의 강점: *위키 사전학습 → Yelp 분류 transfer* 가 **진짜 transfer**. *task corpus 로 사전학습 → 같은 task fine-tune* 의 domain-adaptive pretraining 함정을 피해 원본 BERT 의 *일반 표상 학습 → downstream 전이* 메시지를 그대로 재현합니다. **두 데이터셋이 노트북 안에 공존** — MLM 용 Wikitext-103 (2K paragraphs × 3 epoch) + 분류용 Yelp 이진 (5K/1K).

self-contained 노트북: Ch 20 의 MLM 학습을 압축 (2K × 3 epoch) 재현 → 같은 본체로 분류 fine-tune → Ch 10 결과와 비교. **한국어 Ch 23 self-contained 와 동일한 hyperparams** 로 영어/한국어 챕터 짝의 일관성 유지. 본문은 *사전학습 → 분류 fine-tune* 메인 흐름에 집중. *사전학습 없이 같은 GPU compute 로 분류 fine-tune* 만 했을 때의 fair-compute 비교는 부록 노트북 [`appendix_compute_budget.ipynb`](./appendix_compute_budget.ipynb) 에서 분리해 다룹니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 25-28분 (Wikitext-103 다운로드·필터링 약 2분 + MLM 3 epoch 약 8-10분 + 분류 fine-tune 2 epoch 약 8-10분 + 평가 약 2분)


## 학습 흐름

1. 🚀 **분류 데이터**: `fancyzhx/yelp_polarity` 이진 분류 (Ch 10 과 같은 5K/1K split, seed 42)
2. 🔤 **토크나이저**: `bert-base-uncased` (Ch 20 과 동일)
3. 📥 **MLM 사전학습 데이터**: `Salesforce/wikitext` config `wikitext-103-raw-v1` paragraphs 5K (일반 도메인 — *분류용 Yelp 와 별도*)
4. 🏗️ **MLM 사전학습 재현 (Ch 20 압축본)**: 같은 작은 BertConfig 로 2K paragraphs × 3 epoch (한국어 Ch 23 와 동일)
5. 🔀 **헤드 교체**: `BertForMaskedLM` → `BertForSequenceClassification(num_labels=2)`. 본체는 그대로, MLM head 떼고 분류 head 부착
6. 🚀 **분류 fine-tune**: Trainer fp16, 2 epoch
7. 🔬 **평가**: accuracy / precision / recall / F1 / AUC (Ch 10 과 같은 5종)
8. 🆚 **Ch 10 vs Ch 21 비교 표**: 정확도, 모델 크기, 사전학습 토큰량

📒 **부록**: [`appendix_compute_budget.ipynb`](./appendix_compute_budget.ipynb) — 같은 GPU compute budget 으로 *사전학습 없이* 분류 fine-tune 만 했을 때의 fair-compute 비교


> 📒 **사전 학습 자료**: Ch 20 (작은 BERT scratch MLM, Wikitext-103), Ch 10 (DistilBERT 사전학습 + Yelp 이진 분류). Ch 21 은 두 챕터를 *합쳐서* — Ch 20 의 일반 도메인 사전학습 흐름 그대로 + Ch 10 의 fine-tune 평가 그대로. Ch 22-23 (한국어 위키 → NSMC) 의 *대칭 패턴*.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 10 | DistilBERT 파인튜닝 (약 66M) | `bert-base-uncased` WordPiece | Yelp 이진 (4-5 → 1, 1-2 → 0) | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| 18 | klue/bert-base + 보조 | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | sigmoid + 태스크별 | `BCEWithLogitsLoss + λ·L_aux` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — | — |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Wikitext-103 paragraphs (일반 도메인) | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) |
| **21 ← 여기** | **Ch 20 사전학습 BERT + 분류 헤드 (약 10M)** | (Ch 20과 동일) | **Yelp 이진화 (다른 도메인 transfer)** | **`Linear(H, 2)`** | **softmax** | **`CrossEntropyLoss`** |
| 22 (다음) | 작은 BERT (직접, scratch) — 한국어 | `klue/bert-base` 토크나이저 (가져옴) | 한국어 Wikipedia paragraphs (일반 도메인) | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

**Phase 3 안에서의 위치** — Ch 19 (토크나이저 학습) → Ch 20 (모델 사전학습) → **Ch 21 (분류 fine-tune)** → Ch 22 (한국어 사전학습) → Ch 23 (한국어 분류). Ch 21 이 Phase 3 의 *영어 절반 마무리* — Ch 10 비교가 클라이맥스.

## 변경점 (Diff from Ch 20)

| 축 | Ch 20 (작은 BERT scratch MLM, 위키) | Ch 21 (작은 BERT 분류 fine-tune, Yelp) |
|---|---|---|
| **이 챕터의 task** | MLM 사전학습 (masked token 예측) | **이진 분류 (긍정/부정)** ← *task 축 변화* |
| 모델 클래스 | `BertForMaskedLM` | **`BertForSequenceClassification(num_labels=2)`** |
| 본체 (embedding + encoder) | random init → MLM 학습 | **Ch 20 사전학습 본체 그대로 이어받음** |
| 출력 헤드 | MLM head (vocab 30,522 차원) | **분류 head (`Linear(256, 2)`)** ← 새 random init |
| 토크나이저 | `bert-base-uncased` (vocab 30,522) | (그대로) |
| **데이터** | **Wikitext-103 paragraphs (일반 도메인, 라벨 없음)** | **Yelp 이진 (다른 도메인, 라벨 사용)** ← *사전학습과 fine-tune 도메인이 다름* |
| Loss | `CrossEntropyLoss` (vocab 30,522 logits) | **`CrossEntropyLoss` (2 logits)** ← K 만 큰 변화 |
| 학습률 | 5e-4 (scratch 사전학습) | **2e-5** (fine-tune, 표준) |

> **변경점 한 가지 원칙** — Phase 3 안에서 *task 축* 이 변합니다 (MLM → 분류). 데이터 *도메인* 도 같이 변합니다 (위키 → Yelp) — 이게 *진짜 transfer 의 본질*. 모델 본체·토크나이저는 그대로, 헤드와 라벨 형식·데이터 도메인이 바뀝니다. 이게 *사전학습-fine-tune 패러다임* 의 핵심: 본체는 한 번 학습한 *일반 표상* 을 재사용, downstream task 도메인마다 *작은 헤드 + 작은 학습률* 로 적응.

### 두 데이터셋이 노트북 안에 공존

본 챕터의 특수성 — 한 노트북에 두 데이터셋이 함께 들어갑니다.

| 단계 | 데이터셋 | 용도 |
|---|---|---|
| 3 §MLM 사전학습 | `Salesforce/wikitext`, `wikitext-103-raw-v1` 2K paragraphs × 3 epoch | self-supervised MLM (라벨 없음, 일반 위키 본문) |
| 4-5 §분류 fine-tune | `fancyzhx/yelp_polarity` 5K/1K | supervised 이진 분류 (긍정/부정 라벨) |

같은 토크나이저 (`bert-base-uncased`) 가 두 데이터셋의 모든 텍스트를 처리. 본체가 *위키 일반 어휘* 로 사전학습된 표상이 *Yelp 리뷰(식당·업체) 도메인 토큰* 에 얼마나 잘 전이되는가가 본 챕터의 측정 대상.

### Ch 10 (DistilBERT) 과의 비교가 본 챕터의 메인 메시지 — 이제 *fair*

| 차원 | Ch 10 (DistilBERT) | Ch 21 (이 챕터) | 비고 |
|---|---|---|---|
| 본체 파라미터 | 약 66M | **약 10M** | Ch 21 은 1/6 작음 |
| 사전학습 코퍼스 | Wikipedia + BookCorpus (약 33억 토큰, 일반 도메인) | **Wikitext-103 paragraphs 5K (약 70만-100만 토큰, 일반 도메인)** | 약 3000-5000배 격차, **둘 다 일반 위키** |
| 사전학습 시간 | TPU 수일 (대규모 인프라) | **T4 약 10분** | |
| Fine-tune 도메인 | Yelp 이진 (사전학습과 다른 도메인) | Yelp 이진 (사전학습과 다른 도메인) | **둘 다 일반 → Yelp transfer 라 fair** |
| 분류 fine-tune 셋업 | Ch 10 = 이번 챕터 동일 (같은 데이터, 같은 hyperparams) | | 변하는 건 *본체 출발점* 뿐 |
| 실측 accuracy | 약 0.90 | **약 0.65** | 실행본 기준 (Ch 10=0.9030, Ch 21=0.6490) |

비교가 *공정* 한 이유 — Ch 10 도 본 챕터도 둘 다 *일반 도메인 위키 사전학습 → Yelp 분류 transfer* 의 같은 패턴. *사전학습 규모* (3000-5000배) 와 *모델 크기* (6배) 만 차이. 만약 Ch 21 이 Yelp text 로 사전학습했다면 비교가 unfair 했을 것 — domain-adaptive pretraining 우위 때문.

이 격차가 *사전학습 규모의 가치* 를 정량으로 보여줍니다. *작은 일반 도메인 사전학습도 random init 보다는 낫다* 는 것, 그리고 *같은 GPU compute 를 fine-tune 에 모두 쏟아도 사전학습 효과를 메우기 어렵다* 는 것은 부록 노트북 [`appendix_compute_budget.ipynb`](./appendix_compute_budget.ipynb) 에서 fair-compute 관점으로 다룹니다.

## Loss 함수의 변화 — MLM CE (vocab=30,522) → 분류 CE (K=2)

Ch 20 의 MLM 도 본질은 *vocab 위에서의 다중 분류* 였습니다. 다만 K = vocab_size = 30,522 라 어려운 task. 이번 챕터는 K = 2 의 *훨씬 쉬운* 분류 task.

### 수식

분류 task 의 CE 는 Ch 11 과 같습니다 (K=2):

$$L_{\text{cls}} = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i, y_i}$$

- $\hat p_{i, k} = \mathrm{softmax}(z_i)_k$ — K=2 차원 softmax
- $y_i \in \{0, 1\}$ — 정수 라벨

### 두 CE 비교 (random baseline)

| task | K | random baseline loss $\log K$ | 학습 어려움 |
|---|---|---|---|
| MLM (Ch 20) | 30,522 | **10.33** | 매우 어려움 — 가려진 토큰 자리에 *vocab 전체 후보* 중 정답을 |
| 분류 (Ch 21) | 2 | **0.693** | 상대적으로 쉬움 — 긍정/부정 둘 중 하나 |

학습 첫 step 의 loss 가 약 0.693 부근이면 모델이 *균등 추측* 단계. fine-tune 첫 step 에서 분류 헤드만 새로 init 됐으므로 *이 정도* 가 정상.

### 사전학습 효과가 *loss 곡선* 에 어떻게 드러나나

| 셋업 | 학습 첫 step loss | 학습 종료 loss (epoch 2) | 메모 |
|---|---|---|---|
| random init + 분류 (부록) | 약 0.693 | 약 0.5-0.6 | 본체도 분류 헤드도 random — 학습이 *느림* |
| Wikitext-103 MLM 사전학습 본체 + 분류 (메인) | 약 0.693 | **약 0.3-0.5** | 본체에 *일반 위키 어휘·문맥 구조* 가 들어 있어 헤드가 Yelp 분류로 비교적 빠르게 적응 |
| Ch 10 DistilBERT 사전학습 본체 + 분류 | 약 0.693 | **약 0.15-0.25** | 대규모 일반 도메인 사전학습이 만든 표상의 위력 |

random baseline 은 *세 셋업 모두 같음* — 사전학습이 *학습 속도* 와 *수렴점* 에 영향. 학습 첫 step loss 가 같다고 사전학습이 의미 없는 게 아닙니다. *위키 사전학습 본체* 가 Yelp 도메인에서 *완벽한 성능* 을 내지는 못해도, random 보다 일관되게 빠르고 낮게 수렴.

> **숫자로 감 잡기** (K=2, 정답 = 클래스 1):
> | logits $(z_0, z_1)$ | softmax → $\hat p_1$ | 손실 |
> |---|---|---|
> | (0, 0) | 0.5 | **0.693** ← random |
> | (-1, +1) | 0.881 | 0.127 |
> | (-2, +2) | 0.982 | 0.018 |
> | (+2, -2) | 0.018 | **4.018** ← 자신 있게 틀림 |

## 토크나이저 노트

Ch 20 과 *완전히 동일* — `AutoTokenizer.from_pretrained("bert-base-uncased")`, vocab 30,522 영어 WordPiece. 사전학습-fine-tune 패러다임의 핵심: **토크나이저는 사전학습부터 분류까지 전 구간에서 동일** 해야 함. 그래야 본체가 학습한 토큰 임베딩이 그대로 의미를 유지.

### 두 도메인의 어휘 — 위키 vs Yelp

본 챕터의 두 데이터셋이 *같은 토크나이저* 를 공유하지만 *어휘 분포* 는 꽤 다릅니다.

- **Wikitext-103 (MLM 사전학습)**: 일반 위키 어휘 — 지명·인명·과학·역사 용어 (`capital`, `theorem`, `dynasty`, `proton` ...) 가 풍부. 격식 있는 문장 구조.
- **Yelp polarity (분류 fine-tune)**: 영화 리뷰 어휘 — 감성 형용사·구어체 (`amazing`, `terrible`, `loved`, `awful` ...) 가 풍부. 비격식 구어체.

같은 `bert-base-uncased` vocab (Wiki + BookCorpus 학습) 이 두 도메인을 *모두* 합리적으로 커버 — *위키 본문* 의 격식 어휘는 본 챕터 사전학습이 직접 본 분포, *Yelp 감성 어휘* 는 fine-tune 단계에서 본체가 적응. *토크나이저는 운명공동체* 라 vocab 미스매치가 없습니다.

### 분류 task 에서 [CLS] 토큰의 의미

MLM 사전학습 (Ch 20) 에서는 `group_texts` 패턴으로 *특수 토큰 없이* 토큰 스트림을 잘랐습니다. 분류 fine-tune 에서는 *문장 단위* 입력이라 표준 BERT 포맷:

```
[CLS] the food was excellent and the service was great [SEP]
```

- `[CLS]` 의 최종 hidden state $h_{[CLS]} \in \mathbb{R}^{256}$ 가 *문장 표상*. 분류 헤드 `Linear(256, 2)` 가 이 위에 얹힘.
- MLM 학습 중에는 `[CLS]` 의 hidden 이 *암묵적* 으로만 학습됨 (옆 토큰들과 attention 공유). 분류 fine-tune 단계에서 *이 자리* 가 본격 활용.

### 헤드 교체 시 어떤 파라미터가 어떻게 이어지나

| 모델 부분 | Ch 20 학습 끝 → Ch 21 시작 | 운명 |
|---|---|---|
| 임베딩 (vocab 30,522 × hidden 256) | Wikitext-103 사전학습으로 *일반 위키 어휘 표상* 학습 | **그대로 이어받음** (Yelp 어휘도 같은 vocab 안에 있어 호환) |
| Encoder 4 layer (attention + FFN) | MLM 으로 *문맥 의존 표상* 학습 | **그대로 이어받음** |
| MLM head (`cls.predictions`) | vocab 위 분류 헤드 | **버려짐** |
| 분류 head (`classifier`, `Linear(256, 2)`) | (없었음) | **새로 random init** ← Yelp fine-tune 으로 학습 |

> Ch 10 의 DistilBERT 가 같은 흐름 (일반 도메인 MLM 사전학습 → Yelp 분류 fine-tune) 을 *큰 규모* 로 거친 결과. 우리도 같은 흐름을 *작은 규모* 로 직접 거칩니다 — 둘 다 *위키 → Yelp transfer* 라 비교가 fair.

## 환경 셋업

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

## Yelp 이진 분류 데이터 로드 — Ch 10 과 같은 split

`fancyzhx/yelp_polarity` 는 *이미 이진화된* (긍정/부정) 5점 척도 yelp 리뷰 데이터셋. Ch 10 의 `Yelp/yelp_review_full` + 별점 이진화 와 *완전히 같은 형태* 의 결과가 나오도록 같은 seed·같은 sample 수를 적용. **5,000 train / 1,000 eval, seed 42**.

## 토크나이저 — `bert-base-uncased` (Ch 20 과 동일)

vocab 30,522 의 영어 WordPiece. MLM 사전학습과 분류 fine-tune 전 구간에서 *같은 토크나이저* 를 써야 본체가 학습한 임베딩의 의미가 유지됩니다.

## MLM 사전학습 — Ch 20 패턴 압축 재현 (Wikitext-103, 2K × 3 epoch)

이 노트북을 *self-contained* 로 만들기 위해 Ch 20 의 MLM 사전학습을 여기서 압축 재현합니다. Ch 20 (5K × 2 epoch) 보다 *데이터를 줄이고 (2K) epoch 를 늘려 (3)* 시간을 보존 — 한국어 Ch 23 self-contained 와 동일한 hyperparams. 같은 도메인 (위키) 표상의 *정렬 깊이* 가 충분해 fine-tune 시 random init 보다 분명히 우위.

**MLM 사전학습 데이터는 *분류용 Yelp 와 별도*** — `Salesforce/wikitext`, config `wikitext-103-raw-v1` paragraphs 5K 를 *새로 로드*. 본 챕터의 *진짜 transfer 메시지* — *일반 위키 사전학습 → Yelp 분류 transfer* 가 노트북 한 구조에 자연스럽게 들어맞도록 *두 데이터셋이 공존*. 같은 토크나이저 (`bert-base-uncased`) 가 두 도메인을 모두 처리.

같은 작은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024) → `BertForMaskedLM(config)` random init → Wikitext-103 paragraphs 2K MLM 3 epoch.

### [MASK] 가 들어가는 원리 — 한 눈에 보는 80/10/10

`DataCollatorForLanguageModeling` 은 매 step 마다 *입력 토큰의 약 15%* 를 *무작위로* 선택하고, 선택된 위치마다 세 가지 중 하나를 적용합니다.

| 선택된 토큰 운명 | 비율 | 의도 |
| --- | --- | --- |
| `[MASK]` 로 교체 | **80%** | 표준 마스킹 — 모델이 *주변 문맥만으로* 원래 토큰을 맞추도록 |
| **다른 random 토큰** 으로 교체 | 10% | inference 때는 `[MASK]` 가 없으니, 모델이 *항상* 자기 입력을 *의심* 하게 만듦 |
| **원본 그대로** 유지 | 10% | 동일 — 입력과 정답이 일치하는 케이스도 학습 데이터에 포함 |

**나머지 85%** 의 토큰은 `labels = -100` 으로 두어 *loss 계산에서 제외* 됩니다 (PyTorch CE 의 `ignore_index` 기본값). 즉 한 step 의 MLM loss 는 *선택된 15% 자리만* 모아 평균한 값.

> 이 80/10/10 비율은 BERT 논문 (Devlin et al., 2018) 의 원안 그대로. `mlm_probability=0.15` 만 바꾸면 *선택률* 이 바뀌고, 80/10/10 자체는 collator 내부에 고정.

**관전 포인트**

- `what_happened` 가 `—` 인 자리(85%) 는 *입력과 정답이 그대로* — loss 에 기여하지 않습니다. 모델은 *문맥을 만들어 주는* 역할만.
- `[MASK]` 자리(약 12%) 가 본 task 의 *진짜 학습 신호*. 주변 토큰들의 attention 결과로 *가려진 자리* 의 vocab 분포를 예측.
- `random` (약 1.5%) 와 `kept` (약 1.5%) 는 *inference 분포 일치* 를 위한 정규화. 추론 시에는 `[MASK]` 가 없으므로 *입력을 절대 신뢰하면 안 된다* 는 신호를 학습에 섞어 줌.
- 매 epoch · 매 batch 마다 마스킹은 *새로 무작위* — 같은 문장이 epoch 마다 다른 자리에서 가려져 학습됩니다 (data augmentation 효과).


### `labels = -100` ignore_index 는 BERT-만의 트릭이 아닙니다 — Phase 4 (GPT) 의 핵심으로 다시

PyTorch `CrossEntropyLoss` 의 `ignore_index=-100` 은 *어느 토큰 자리의 loss 를 학습 신호로 쓸지* 고르는 범용 스위치입니다. 같은 트릭이 Phase 4 GPT 챕터에서 **사전학습 vs Instruction Tuning(SFT) 의 가장 큰 차이** 를 만듭니다.

| 단계 | `labels = ?` | loss 계산 자리 | 학습되는 것 |
|---|---|---|---|
| **MLM 사전학습** (이 챕터·Ch 20) | 선택된 약 15% 만 원본 token id, 나머지 = `-100` | 가려진 자리 | 주변 문맥으로 *가려진 토큰 복원* |
| **GPT CausalLM 사전학습** (Ch 24-26) | `input_ids.clone()` — *거의 모든 토큰* | (pad 만 `-100`) 사실상 *전 자리* | 모든 자리에서 *다음 토큰 예측* — 언어 분포 자체 |
| **SFT / Instruction Tuning** (Ch 27) | **prompt 부분 = `-100`**, *답변 토큰만* 원본 id | *답변 부분만* | "질문을 외우지 말고 답변하는 법" 만 학습 |

> **세 곳 모두 같은 `-100` 트릭, 적용 자리만 정반대.** MLM 은 *대부분을 가리고 일부만 학습*, GPT 사전학습은 *거의 가리지 않음*, SFT 는 *prompt 만 가림*. Phase 4 (특히 Ch 27 SFT, `SFTTrainer` 의 `response-only mask` 옵션) 에서 이 차이를 *코드 라인 한 줄 — `labels[prompt_mask] = -100`* 으로 직접 보게 될 겁니다.

지금 위 셀에서 본 `label_id = -100` 의 의미를 기억해 두면, Ch 27 의 *왜 모델이 instruction 을 따라가게 되는가* 가 한 줄로 이해됩니다.

### 같은 단어 "파인튜닝", BERT 시대와 GPT 시대의 의미가 살짝 다릅니다

이 챕터의 *fine-tune* 은 **BERT 시대 의미** — *사전학습된 본체 + 새 task-specific head (`Linear(H, 2)`)* 를 붙여 *downstream task* 마다 다른 모델로 분기. 본체는 *일반 표상*, head 는 *task 별 특화*. 분류·회귀·NER·QA 각각 다른 head 가 붙고 라벨 포맷도 다릅니다.

GPT 시대 (Phase 4 Ch 24 이후) 부터는 같은 단어가 *살짝 다른 의미* 를 가집니다.

| 축 | **BERT 파인튜닝** (이 챕터, Ch 9-18, Ch 23) | **GPT 파인튜닝 = SFT** (Ch 25, Ch 27) |
|---|---|---|
| 무엇을 바꾸나 | 본체 + **새 head** (task별 부착) | 본체 + **기존 LM head 그대로** |
| 출력 형식 | task별 다름 (class id / score / multi-hot) | *항상 토큰 시퀀스* — 형식 통일 |
| 학습 신호 | task별 loss (CE/BCE/MSE) | *항상 next-token CE*, 단 자리 마스킹만 다름 |
| 학습되는 것 | *task 의 출력 분포* (긍정/부정 결정 경계 등) | *행동 = "이런 입력엔 이런 형식으로 답하라"* |
| 라벨 | 정답 카테고리/값 | *모범 답안 토큰 시퀀스* |

> **BERT 파인튜닝은 *task 적응*, GPT 파인튜닝은 *행동 정렬*.** GPT 는 head 가 바뀌지 않으므로 "파인튜닝" 이 *동일한 next-token 예측 task 안에서 데이터만 바뀌는* 일이 됩니다 (사전학습 = 웹 텍스트, SFT = 모범 응답 쌍). 그래서 Phase 4 부터는 "fine-tuning ≈ SFT ≈ instruction tuning ≈ behavior alignment" 가 거의 동의어로 섞여 쓰입니다.

이 의미 차이는 *왜 GPT 모델 하나가 모든 task 를 해내는가* 의 핵심 이유 — head 가 task 별로 분기하지 않으니 *입력 프롬프트* 만 바꾸면 *같은 모델* 이 다른 일을 합니다. Ch 27 에서 직접 확인.

**관전 포인트** — Wikitext-103 paragraphs 에서 MLM loss 가 *random baseline 10.33* 에서 시작해 약 7 부근까지 떨어졌다면 본체가 *일반 위키 어휘·문맥 구조의 일부* 를 학습한 상태. perplexity 로 환산하면 vocab 30,522 중 *약 1,300 개 후보* 로 좁혀진 정도. Ch 20 의 2 epoch 와 비슷한 수준이지만, *Yelp 분류 fine-tune 출발점* 으로는 충분합니다 — 본체가 *일반 영어 구조* 를 가지면 *Yelp 리뷰(식당·업체) 도메인* 도 fine-tune 으로 빠르게 적응.

> **체크포인트 저장은 생략** — 노트북 안에서 바로 본체 가중치를 분류 모델로 옮기기 때문. Ch 20 처럼 디스크에 저장하려면 `mlm_model.save_pretrained("./ch21_mlm_ckpt")` 한 줄.

## 헤드 교체 — MLM → 분류 + Fine-tune

이제 *방금 학습된 작은 BERT 본체* 를 분류 모델로 옮깁니다. 두 가지 흐름:

1. `BertForMaskedLM.bert` (embedding + encoder) 를 그대로 가져옴
2. 새 `BertForSequenceClassification(config)` 을 만들고, 1 의 본체를 *복사*. 분류 헤드는 새로 random init

이렇게 만든 모델을 같은 Yelp 데이터의 *라벨* 까지 사용해 분류 fine-tune. Ch 10 의 hyperparams 와 *완전히 같이* (`lr=2e-5, batch=16, epoch=2, fp16=True`) 둬서 *본체 출발점* 외 모든 조건을 통제.

**`bert.load_state_dict` 가 한 일** — `BertForMaskedLM` 과 `BertForSequenceClassification` 둘 다 *내부에 같은 `BertModel`* (이름 `self.bert`) 을 갖습니다. 그 본체만 통째로 옮긴 셈. MLM head (`cls.predictions`) 와 분류 head (`classifier`) 는 *모델 객체의 다른 자리* 라 자동으로 분리됩니다.

> Ch 7-18 의 `AutoModelForSequenceClassification.from_pretrained(...)` 가 디스크에서 같은 일을 합니다. 우리는 *방금 학습한 본체* 를 디스크 없이 in-memory 로 옮긴 셈.

## 평가 — Ch 10 과 같은 5종 metric + 학습 곡선

`accuracy / precision / recall / F1 / AUC` 전부 같은 정의. 마지막에 confusion matrix 와 학습 곡선을 같이 그려 *본체 출발점 변화가 학습 동역학에 어떻게 드러나는지* 시각화.

### 5-1. 학습 곡선 — MLM 사전학습 효과가 보이는 자리

분류 fine-tune 의 step-by-step train loss 를 그려, *시작점* 과 *수렴점* 을 같이 확인.

### 5-2. Confusion matrix

## Ch 10 (DistilBERT) vs Ch 21 (작은 BERT scratch) — 본 챕터의 핵심 결과

*같은 데이터·같은 hyperparams* 에 *본체 출발점만 다른* 두 셋업의 정확도 비교. 둘 다 *일반 도메인 위키 사전학습 → Yelp 분류 transfer* 의 같은 패턴이라 비교가 *fair*. Ch 10 의 수치는 본 챕터를 작성하는 시점에 *해당 노트북의 README/실행 결과* 를 참고해 인용 — 학습자가 노트북을 돌려 본인 수치로 갱신해 보면 더 좋습니다.

| 차원 | Ch 10 (DistilBERT pretrained) | Ch 21 (작은 BERT scratch + 2K × 3 epoch MLM) | 비고 |
|---|---|---|---|
| 본체 파라미터 | 약 66M | 약 10M | Ch 21 은 1/6 크기 |
| 사전학습 코퍼스 | Wikipedia + BookCorpus (약 33억 토큰, 일반 도메인) | Wikitext-103 paragraphs 5K (약 70만-100만 토큰, 일반 도메인) | 약 3000-5000배 격차, **둘 다 일반 위키** |
| 사전학습 시간 | TPU 수일 | T4 약 10-12분 | |
| Fine-tune 도메인 | Yelp 이진 (사전학습과 다른 도메인) | Yelp 이진 (사전학습과 다른 도메인) | **둘 다 일반 → Yelp transfer** |
| 분류 fine-tune 셋업 | (같음 — 5K/1K, batch 16, lr 2e-5, 2 epoch, fp16) | | 본체 외 통제 |

**관찰 — *동일 transfer 패턴 안에서 3000-5000배 사전학습 격차* 가 분류 정확도에 어떻게 드러나나**

실측 (실행본 기준):
- Ch 10 (DistilBERT, 대규모 Wiki+BookCorpus 사전학습): accuracy 약 0.90, AUC 약 0.97
- Ch 21 (작은 BERT, Wikitext-103 2K paragraphs × 3 epoch 사전학습): accuracy 약 0.65, AUC 약 0.71

**accuracy 약 25%p 격차** 가 나옵니다. 두 모델이 *같은 transfer 패턴* (일반 위키 → Yelp) 을 따르므로 이 격차의 거의 전부가 *사전학습 규모의 가치* — Wikipedia + BookCorpus 약 33억 토큰의 *일반 영어 지식* 이 DistilBERT 본체에 압축되어 있어, Yelp 같은 *다른 도메인* 에도 빠르게 적응합니다.

> 한편 Ch 21 의 accuracy 가 *random (50%) 보다 훨씬 높다* 는 것도 중요한 결과입니다. 작은 일반 도메인 사전학습 + 작은 모델로도 *기본 위키 어휘·문맥 구조* 가 본체에 들어가 Yelp 분류의 *기본 신호* (긍정/부정 단어들의 통계) 가 잡힙니다.

## 부록 — fair-compute 비교 (사전학습 없이 같은 GPU compute 로 분류만)

*MLM 사전학습 없이 random init 으로 바로 분류 fine-tune*, 그리고 *같은 GPU compute budget (MLM 시간 + fine-tune 시간 합)* 으로 *분류 fine-tune 만 더 길게* 돌렸을 때 어떻게 되는지는 부록 노트북 [`appendix_compute_budget.ipynb`](./appendix_compute_budget.ipynb) 에서 다룹니다.

> 부록의 핵심 질문 — *"사전학습에 쓰는 compute 를 그냥 fine-tune 에 더 쓰면 안 되나?"* 에 대한 정량 답. 작은 모델·작은 데이터 환경에서 사전학습이 *compute 등가물 보다도* 가치 있는지 확인.

## 이 장의 구성

[[SubPages]]
