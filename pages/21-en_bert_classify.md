**목표**: Phase 3 의 세 번째 챕터. Ch 20 에서 *작은 BERT 를 일반 도메인 (Wikitext-103) 으로 직접 MLM 사전학습* 했다면, 이번엔 그 위에 **분류 헤드를 얹어 *완전히 다른 도메인 (Yelp 리뷰(식당·업체))* 이진 분류로 fine-tune** 합니다. Ch 10 (DistilBERT, 약 66M params, 대규모 Wikipedia + BookCorpus 사전학습) 과 같은 Yelp 이진 분류 셋업에 *우리가 만든 작은 BERT* (약 10M params, Wikitext-103 2K paragraphs × 3 epoch MLM) 를 붙여 두 결과를 나란히 비교 — 둘 다 *일반 도메인 → Yelp transfer* 라 비교가 *fair*, *사전학습 규모* 차이만 측정됨.

본 챕터의 강점: *위키 사전학습 → Yelp 분류 transfer* 가 **진짜 transfer**. *task corpus 로 사전학습 → 같은 task fine-tune* 의 domain-adaptive pretraining 함정을 피해 원본 BERT 의 *일반 표상 학습 → downstream 전이* 메시지를 그대로 재현합니다. **두 데이터셋이 노트북 안에 공존** — MLM 용 Wikitext-103 (2K paragraphs × 3 epoch) + 분류용 Yelp 이진 (5K/1K).

self-contained 노트북: Ch 20 의 MLM 학습을 압축 (2K × 3 epoch) 재현 → 같은 본체로 분류 fine-tune → Ch 10 결과와 비교. **한국어 Ch 23 self-contained 와 동일한 hyperparams** 로 영어/한국어 챕터 짝의 일관성 유지. 본문은 *사전학습 → 분류 fine-tune* 메인 흐름에 집중. *사전학습 없이 같은 GPU compute 로 분류 fine-tune* 만 했을 때의 fair-compute 비교는 부록 노트북 [`appendix_compute_budget.ipynb`](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/appendix_compute_budget.ipynb) 에서 분리해 다룹니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 3-5분 — 대부분이 데이터 다운로드입니다 (실행본 `executed/21_en_bert_classify.ipynb` 기준 전체 2분 13초: Wikitext-103·Yelp 다운로드·전처리 약 1분 40초 + MLM 3 epoch 약 15초 + 분류 fine-tune 2 epoch 약 15초 + 평가·시각화 수 초). 다운로드 속도에 따라 달라집니다.

## 학습 흐름

1. 🚀 **분류 데이터**: `fancyzhx/yelp_polarity` 이진 분류 (Ch 10 과 같은 5K/1K split, seed 42)
2. 🔤 **토크나이저**: `bert-base-uncased` (Ch 20 과 동일)
3. 📥 **MLM 사전학습 데이터**: `Salesforce/wikitext` config `wikitext-103-raw-v1` paragraphs 2K (일반 도메인 — *분류용 Yelp 와 별도*)
4. 🏗️ **MLM 사전학습 재현 (Ch 20 압축본)**: 같은 작은 BertConfig 로 2K paragraphs × 3 epoch (한국어 Ch 23 와 동일)
5. 🔀 **헤드 교체**: `BertForMaskedLM` → `BertForSequenceClassification(num_labels=2)`. 본체는 그대로, MLM head 떼고 분류 head 부착
6. 🚀 **분류 fine-tune**: Trainer fp16, 2 epoch
7. 🔬 **평가**: accuracy / precision / recall / F1 / AUC (Ch 10 과 같은 5종)
8. 🆚 **Ch 10 vs Ch 21 비교 표**: 정확도, 모델 크기, 사전학습 토큰량

📒 **부록**: [`appendix_compute_budget.ipynb`](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/appendix_compute_budget.ipynb) — 같은 GPU compute budget 으로 *사전학습 없이* 분류 fine-tune 만 했을 때의 fair-compute 비교

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
| 사전학습 코퍼스 | Wikipedia + BookCorpus (약 33억 토큰, 일반 도메인) | **Wikitext-103 paragraphs 2K (약 27만 토큰, 일반 도메인)** | 약 1.2만배 격차, **둘 다 일반 위키** |
| 사전학습 시간 | TPU 수일 (대규모 인프라) | **T4 약 15초** (2K × 3 epoch = 198 step) | |
| Fine-tune 도메인 | Yelp 이진 (사전학습과 다른 도메인) | Yelp 이진 (사전학습과 다른 도메인) | **둘 다 일반 → Yelp transfer 라 fair** |
| 분류 fine-tune 셋업 | Ch 10 = 이번 챕터 동일 (같은 데이터, 같은 hyperparams) | | 변하는 건 *본체 출발점* 뿐 |
| 실측 accuracy | 약 0.90 | **random (0.50) 과 Ch 10 의 중간쯤** | 실행마다 흔들려 본문에 못 박지 않습니다 — 값은 §6 의 비교 셀 출력 |

비교가 *공정* 한 이유 — Ch 10 도 본 챕터도 둘 다 *일반 도메인 위키 사전학습 → Yelp 분류 transfer* 의 같은 패턴. *사전학습 규모* (약 1.2만배) 와 *모델 크기* (약 6배) 만 차이. 만약 Ch 21 이 Yelp text 로 사전학습했다면 비교가 unfair 했을 것 — domain-adaptive pretraining 우위 때문.

이 격차가 *사전학습 규모의 가치* 를 정량으로 보여줍니다. 한편 *작은 일반 도메인 사전학습이 random init 보다 나은가*, *같은 GPU compute 를 fine-tune 에 모두 쏟으면 그 차이가 메워지는가* 는 이 노트북에서 측정하지 않습니다 — 부록 노트북 [`appendix_compute_budget.ipynb`](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/appendix_compute_budget.ipynb) 이 그 비교를 위한 셋업입니다. **부록의 답은 이 규모에서 '메워진다' 입니다** — random init 대비 순 효과는 수 %p 로 실재하지만, 같은 GPU 예산을 분류 fine-tune 에 쓰는 쪽이 더 크게 이깁니다. *사전학습의 가치는 규모에서 나온다* 는 이 챕터의 메시지를 반대쪽에서 받쳐 줍니다.

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
| random init + 분류 (부록) | 약 0.693 | **약 0.69** (`executed/appendix_compute_budget.ipynb` 셋업 C 기록) | 본체도 분류 헤드도 random — 2 epoch 으로는 기준선을 거의 못 벗어남 |
| Wikitext-103 MLM 사전학습 본체 + 분류 (메인) | 약 0.693 | **랜덤 기준선 바로 아래** — 값은 §5-1 학습 곡선 셀 출력 | 2K × 3 epoch 은 아주 얕은 사전학습이라, 2 epoch fine-tune 으로 기준선에서 크게 내려가지 못합니다 |
| Ch 10 DistilBERT 사전학습 본체 + 분류 | 약 0.693 | **약 0.16** (`executed/10_bert_binary_sigmoid.ipynb`, 커밋 `d9fc77c` 기록) | 대규모 일반 도메인 사전학습이 만든 표상의 위력 |

random baseline 은 *세 셋업 모두 같음* — 사전학습은 *학습 속도* 와 *수렴점* 에 영향을 줍니다. 다만 이 챕터의 사전학습은 *2K paragraphs × 3 epoch* 으로 아주 얕아서, 종료 loss 가 랜덤 기준선(0.693)을 크게 밑돌지는 못합니다. *얼마나 내려가는지* 는 실행마다 흔들리므로 본문에서 숫자를 못 박지 않고 §5-1 의 곡선과 §6 의 셀 출력을 단일 출처로 둡니다. random init 과의 직접 비교는 부록에서 다룹니다 — 부록 실측에서 메인 조건(A)이 random init(C)보다 조금 앞섰지만, 같은 GPU 예산을 fine-tune 에 더 쓴 셋업(B)에는 뒤졌습니다.

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

## 이 장의 구성

[[SubPages]]
