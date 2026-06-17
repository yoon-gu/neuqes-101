**목표**: Phase 3 의 마지막 챕터. Ch 22 에서 *작은 한국어 BERT 를 일반 도메인 (한국어 Wikipedia) 으로 직접 MLM 사전학습* 했다면, 이번엔 그 위에 **분류 헤드를 얹어 *완전히 다른 도메인 (NSMC 영화 리뷰)* 이진 분류로 fine-tune** 합니다. Ch 15 (`klue/bert-base`, 약 110M params, 약 8.4B 토큰 대규모 한국어 사전학습) 와 같은 NSMC 분류 셋업에 *우리가 만든 작은 BERT* (약 10M params, 한국어 위키 2K paragraphs × 3 epoch MLM) 를 붙여 두 결과를 나란히 비교 — 둘 다 *일반 한국어 사전학습 → NSMC transfer* 라 비교가 *fair*, *사전학습 규모* 차이만 측정됩니다.

본 챕터의 강점: *위키 사전학습 → NSMC 분류 transfer* 가 **진짜 transfer**. 사전학습이 *task 도메인 (영화 리뷰) 자체* 를 본 적이 없는 일반 위키 본문으로 진행되어, *일반 표상 학습 → 다른 도메인 fine-tune* 의 정직한 메시지가 나옵니다. **두 데이터셋이 노트북 안에 공존** — MLM 용 한국어 Wikipedia (2K paragraphs × 3 epoch) + 분류용 NSMC (5K/1K).

self-contained 노트북: Ch 22 의 MLM 학습을 짧게 재현 → 같은 본체로 분류 fine-tune → Ch 15 결과와 2-way 비교. **random init baseline 비교 + negative transfer 분석** 은 부록 노트북 [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb) 으로 분리.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 20-23분 (한국어 위키 다운로드·필터링 약 2분 + MLM 3 epoch 약 8-10분 + 분류 fine-tune 2 epoch 약 8-10분 + 평가 약 2분)


## 학습 흐름

1. 🚀 **분류 데이터**: NSMC 이진 (e9t/nsmc, GitHub raw TSV, Ch 15 와 같은 5K/1K split, seed 42)
2. 🔤 **토크나이저**: `klue/bert-base` (Ch 22 와 동일)
3. 📥 **MLM 사전학습 데이터**: `wikimedia/wikipedia` `20231101.ko` paragraphs 5K (일반 도메인 — *분류용 NSMC 와 별도*)
4. 🏗️ **MLM 사전학습 재현 (Ch 22 압축본)**: 같은 작은 BertConfig 로 1 epoch 만 짧게
5. 🔀 **헤드 교체**: `BertForMaskedLM` → `BertForSequenceClassification(num_labels=2)`. 본체는 그대로, MLM head 떼고 분류 head 부착
6. 🚀 **분류 fine-tune**: Trainer fp16, 2 epoch
7. 🔬 **평가**: accuracy / precision / recall / F1 / AUC (Ch 15 / Ch 21 과 같은 5종) + confusion matrix
8. 🆚 **Ch 15 vs Ch 23 ours** 2-way 비교 — 정확도, 모델 크기, 사전학습 토큰량
9. 📒 **부록**: random init baseline + negative transfer 분석 → [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb)


> 📒 **사전 학습 자료**: Ch 22 (한국어 작은 BERT scratch MLM, 한국어 Wikipedia), Ch 15 (`klue/bert-base` 한국어 사전학습 + NSMC 이진 분류), Ch 21 (영어 작은 BERT 분류 — 본 챕터의 영어 대칭본). Ch 23 은 세 챕터를 *합쳐서* — Ch 22 의 한국어 일반 도메인 사전학습 흐름 그대로 + Ch 15 의 한국어 분류 fine-tune 평가 그대로 + Ch 21 의 transfer 메시지를 한국어 환경에서 재확인. **Phase 3 의 마지막 챕터** — Phase 4 (Ch 24, GPT scratch) 부터는 *decoder-only* 와 *SFT 의미의 파인튜닝* 으로 흐름이 바뀝니다.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 15 | `klue/bert-base` 파인튜닝 (약 110M) | WordPiece (한국어, 사전학습) | NSMC (네이버 영화 리뷰, 이진) | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Wikitext-103 paragraphs (일반 도메인) | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) |
| 21 | Ch 20 사전학습 BERT + 분류 헤드 (약 10M) | (Ch 20과 동일) | Yelp 이진화 (다른 도메인 transfer) | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| 22 | 작은 BERT (직접, scratch) — 한국어 | `klue/bert-base` 토크나이저 (가져옴) | 한국어 Wikipedia paragraphs (일반 도메인) | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) |
| **23 ← 여기** | **Ch 22 사전학습 BERT + 분류 헤드 (약 10M)** | **(Ch 22와 동일)** | **NSMC 이진 (다른 도메인 transfer)** | **`Linear(H, 2)`** | **softmax** | **`CrossEntropyLoss`** |
| 24 (다음, Phase 4) | GPT-2 (직접, scratch) | BPE 토크나이저 (직접 학습) | TinyStories 영어 동화 | LM head | softmax (next-token) | `CrossEntropyLoss` (causal LM) |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

**Phase 3 안에서의 위치** — Ch 19 (토크나이저 학습) → Ch 20 (영어 모델 사전학습) → Ch 21 (영어 분류) → Ch 22 (한국어 모델 사전학습) → **Ch 23 (한국어 분류, Phase 3 종료)**. Ch 22 → Ch 23 흐름이 Ch 20 → Ch 21 흐름의 *한국어 대칭본*. 본문의 클라이맥스는 *2-way 비교* — Ch 15 의 대규모 사전학습 모델과 본 챕터의 작은 자체 사전학습 모델의 격차로 *사전학습 규모의 가치* 를 정량 측정. 사전학습 자체의 *순* 효과 (random init baseline 대비) 와 *한국어 환경 특유의 negative transfer 가능성* 은 부록에서 다룹니다.

## 변경점 (Diff from Ch 22)

| 축 | Ch 22 (한국어 MLM scratch) | Ch 23 (한국어 분류 fine-tune) |
|---|---|---|
| **이 챕터의 task** | MLM 사전학습 (masked token 예측) | **이진 분류 (NSMC 긍정/부정)** ← *task 축 변화* |
| 모델 클래스 | `BertForMaskedLM` | **`BertForSequenceClassification(num_labels=2)`** |
| 본체 (embedding + encoder) | random init → 한국어 위키 MLM 학습 | **Ch 22 사전학습 본체 그대로 이어받음** |
| 출력 헤드 | MLM head (vocab 약 32,000 차원) | **분류 head (`Linear(256, 2)`)** ← 새 random init |
| 토크나이저 | `klue/bert-base` (vocab 약 32,000) | (그대로) |
| **데이터** | **한국어 Wikipedia paragraphs (일반 도메인, 라벨 없음)** | **NSMC 영화 리뷰 (다른 도메인, 라벨 사용)** ← *사전학습과 fine-tune 도메인이 다름* |
| Loss | `CrossEntropyLoss` (vocab 약 32,000 logits) | **`CrossEntropyLoss` (2 logits)** ← K 만 큰 변화 |
| 학습률 | 5e-4 (scratch MLM) | **2e-5** (fine-tune 표준) |

> **변경점 한 가지 원칙** — Phase 3 안에서 *task 축* 이 변합니다 (MLM → 분류). 데이터 *도메인* 도 같이 변합니다 (위키 → NSMC) — 이게 *진짜 transfer 의 본질*. 모델 본체·토크나이저는 그대로, 헤드와 라벨 형식·데이터 도메인이 바뀝니다. 이게 *사전학습-fine-tune 패러다임* 의 핵심: 본체는 한 번 학습한 *일반 표상* 을 재사용, downstream task 도메인마다 *작은 헤드 + 작은 학습률* 로 적응.

### 두 데이터셋이 노트북 안에 공존

본 챕터의 특수성 — 한 노트북에 두 데이터셋이 함께 들어갑니다.

| 단계 | 데이터셋 | 용도 |
|---|---|---|
| 3 §MLM 사전학습 | `wikimedia/wikipedia`, `20231101.ko` 2K paragraphs × 3 epoch | self-supervised MLM (라벨 없음, 일반 위키 본문) |
| 4-5 §분류 fine-tune | NSMC (e9t/nsmc GitHub raw TSV) 5K/1K | supervised 이진 분류 (긍정/부정 라벨) |

같은 토크나이저 (`klue/bert-base`) 가 두 데이터셋의 모든 텍스트를 처리. 본체가 *일반 위키 어휘* 로 사전학습된 표상이 *영화 리뷰 비격식 구어체 토큰* 에 얼마나 잘 전이되는가가 본 챕터의 측정 대상.

### Ch 15 (klue/bert-base) 와의 비교가 본 챕터의 메인 메시지 — 이제 *fair*

| 차원 | Ch 15 (klue/bert-base) | Ch 23 (이 챕터) | 비고 |
|---|---|---|---|
| 본체 파라미터 | 약 110M | **약 10M** | Ch 23 은 1/11 작음 |
| 사전학습 코퍼스 | 한국어 위키 + 모두의 말뭉치 + 뉴스 + 댓글 (약 8.4B 토큰, 일반 도메인) | **한국어 Wikipedia paragraphs 5K (약 50만-80만 토큰, 일반 도메인)** | 약 10,000배 격차, **둘 다 일반 한국어 코퍼스** |
| 사전학습 시간 | TPU 수일 | **T4 약 8-10분** | |
| Fine-tune 도메인 | NSMC 이진 (사전학습과 다른 도메인) | NSMC 이진 (사전학습과 다른 도메인) | **둘 다 일반 한국어 → NSMC transfer 라 fair** |
| 분류 fine-tune 셋업 | Ch 15 = 이번 챕터 동일 (같은 데이터, 같은 hyperparams) | | 변하는 건 *본체 출발점* 뿐 |
| 기대 accuracy | 약 85-88% | **약 65-75% 예상** | 비교는 실측치로 확인 |

비교가 *공정* 한 이유 — Ch 15 도 본 챕터도 둘 다 *일반 도메인 한국어 사전학습 → NSMC 분류 transfer* 의 같은 패턴. *사전학습 규모* (약 10,000배) 와 *모델 크기* (11배) 만 차이. 만약 Ch 23 이 NSMC text 로 사전학습했다면 비교가 unfair 했을 것 — domain-adaptive pretraining 우위 때문.

### Ch 21 (영어) → Ch 23 (한국어) 대칭

| 항목 | Ch 21 (영어) | Ch 23 (한국어, 이번 챕터) |
|---|---|---|
| 사전학습 코퍼스 (일반 도메인) | Wikitext-103 paragraphs 5K | 한국어 Wikipedia paragraphs 5K |
| 분류 데이터 (다른 도메인) | Yelp polarity 5K/1K | NSMC 5K/1K |
| 비교 대상 (대규모 사전학습) | Ch 10 (DistilBERT, 약 66M, 약 33억 토큰) | Ch 15 (`klue/bert-base`, 약 110M, 약 8.4B 토큰) |
| 토크나이저 | `bert-base-uncased` | `klue/bert-base` |
| 메시지 | *일반 위키 사전학습 → 영화 리뷰 transfer* | *일반 위키 사전학습 → 영화 리뷰 transfer* |

같은 결을 한국어 환경에서 재확인 — Phase 3 의 마지막 검증.

## Loss 함수의 변화 — MLM CE (vocab 약 32,000) → 분류 CE (K=2)

Ch 22 의 MLM 도 본질은 *vocab 위에서의 다중 분류* 였습니다. 다만 K = vocab_size 약 32,000 이라 어려운 task. 이번 챕터는 K = 2 의 *훨씬 쉬운* 분류 task.

### 수식

분류 task 의 CE 는 Ch 15 / Ch 21 과 같습니다 (K=2):

$$L_{\text{cls}} = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i, y_i}$$

- $\hat p_{i, k} = \mathrm{softmax}(z_i)_k$ — K=2 차원 softmax
- $y_i \in \{0, 1\}$ — 정수 라벨 (NSMC: 0=negative, 1=positive)

### 두 CE 비교 (random baseline)

| task | K | random baseline loss $\log K$ | 학습 어려움 |
|---|---|---|---|
| MLM (Ch 22) | 약 32,000 | **10.37** | 매우 어려움 — 가려진 토큰 자리에 *vocab 전체 후보* 중 정답을 |
| 분류 (Ch 23) | 2 | **0.693** | 상대적으로 쉬움 — 긍정/부정 둘 중 하나 |

학습 첫 step 의 loss 가 약 0.693 부근이면 모델이 *균등 추측* 단계. fine-tune 첫 step 에서 분류 헤드만 새로 init 됐으므로 *이 정도* 가 정상.

### 사전학습 효과가 *loss 곡선* 에 어떻게 드러나나

| 셋업 | 학습 첫 step loss | 학습 종료 loss (epoch 2) | 메모 |
|---|---|---|---|
| 한국어 Wikipedia MLM 사전학습 본체 + 분류 (본 챕터) | 약 0.693 | **약 0.45-0.6** | 본체에 *일반 한국어 어휘·문맥 구조* 가 들어 있어 헤드가 NSMC 분류로 비교적 빠르게 적응 |
| Ch 15 `klue/bert-base` 사전학습 본체 + 분류 | 약 0.693 | **약 0.25-0.4** | 대규모 일반 한국어 사전학습이 만든 표상의 위력 |

random baseline 은 *두 셋업 모두 같음* — 사전학습이 *학습 속도* 와 *수렴점* 에 영향. 학습 첫 step loss 가 같다고 사전학습이 의미 없는 게 아닙니다. *위키 사전학습 본체* 가 NSMC 도메인에서 *완벽한 성능* 을 내지는 못해도, random 보다 일관되게 빠르고 낮게 수렴 (사전학습 없는 random init 과의 직접 비교는 부록 [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb)).

> **숫자로 감 잡기** (K=2, 정답 = 클래스 1):
> | logits $(z_0, z_1)$ | softmax → $\hat p_1$ | 손실 |
> |---|---|---|
> | (0, 0) | 0.5 | **0.693** ← random |
> | (-1, +1) | 0.881 | 0.127 |
> | (-2, +2) | 0.982 | 0.018 |
> | (+2, -2) | 0.018 | **4.018** ← 자신 있게 틀림 |

## 토크나이저 노트

Ch 22 와 *완전히 동일* — `AutoTokenizer.from_pretrained("klue/bert-base")`, vocab 약 32,000 한국어 WordPiece. 사전학습-fine-tune 패러다임의 핵심: **토크나이저는 사전학습부터 분류까지 전 구간에서 동일** 해야 함. 그래야 본체가 학습한 토큰 임베딩이 그대로 의미를 유지.

### 두 도메인의 어휘 — 위키 vs NSMC

본 챕터의 두 데이터셋이 *같은 토크나이저* 를 공유하지만 *어휘 분포* 는 꽤 다릅니다.

- **한국어 Wikipedia (MLM 사전학습)**: 일반 위키 어휘 — 지명·인명·역사·과학 용어 (`수도`, `행성`, `왕조`, `이론` ...) 가 풍부. 격식 있는 문장 구조, 평균 길이 수십-수백 자.
- **NSMC (분류 fine-tune)**: 영화 리뷰 비격식 구어체 — 감성 형용사·구어체 어미·이모티콘·맞춤법 흔들림 (`재밌`, `노잼`, `ㅋㅋ`, `최고`, `별로` ...) 가 풍부. 평균 길이 매우 짧음 (한 줄, 보통 10-50자).

같은 `klue/bert-base` vocab (한국어 위키 + 모두의 말뭉치 + 뉴스 + 댓글 학습) 이 두 도메인을 *모두* 합리적으로 커버 — *위키 본문* 의 격식 어휘는 본 챕터 사전학습이 직접 본 분포, *NSMC 구어체 감성 어휘* 는 fine-tune 단계에서 본체가 적응. *토크나이저는 운명공동체* 라 vocab 미스매치가 없습니다.

### 분류 task 에서 [CLS] 토큰의 의미

MLM 사전학습 (Ch 22) 에서는 `group_texts` 패턴으로 *특수 토큰 없이* 토큰 스트림을 잘랐습니다. 분류 fine-tune 에서는 *문장 단위* 입력이라 표준 BERT 포맷:

```
[CLS] 이 영화 정말 재미있었어요 [SEP]
```

- `[CLS]` 의 최종 hidden state $h_{[\text{CLS}]} \in \mathbb{R}^{256}$ 가 *문장 표상*. 분류 헤드 `Linear(256, 2)` 가 이 위에 얹힘.
- MLM 학습 중에는 `[CLS]` 의 hidden 이 *암묵적* 으로만 학습됨 (옆 토큰들과 attention 공유). 분류 fine-tune 단계에서 *이 자리* 가 본격 활용.

### 헤드 교체 시 어떤 파라미터가 어떻게 이어지나

| 모델 부분 | Ch 22 학습 끝 → Ch 23 시작 | 운명 |
|---|---|---|
| 임베딩 (vocab 약 32,000 x hidden 256) | 한국어 Wikipedia 사전학습으로 *일반 위키 어휘 표상* 학습 | **그대로 이어받음** (NSMC 어휘도 같은 vocab 안에 있어 호환) |
| Encoder 4 layer (attention + FFN) | MLM 으로 *문맥 의존 표상* 학습 | **그대로 이어받음** |
| MLM head (`cls.predictions`) | vocab 위 분류 헤드 | **버려짐** |
| 분류 head (`classifier`, `Linear(256, 2)`) | (없었음) | **새로 random init** ← NSMC fine-tune 으로 학습 |

> Ch 15 의 `klue/bert-base` 가 같은 흐름 (한국어 일반 도메인 MLM 사전학습 → NSMC 분류 fine-tune) 을 *큰 규모* 로 거친 결과. 우리도 같은 흐름을 *작은 규모* 로 직접 거칩니다 — 둘 다 *위키 → NSMC transfer* 라 비교가 fair.

## 환경 셋업

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

## NSMC 이진 분류 데이터 로드 — Ch 15 와 같은 split

NSMC = Naver Sentiment Movie Corpus. 한국어 *binary* 감성 분류의 표준 벤치마크. 한 줄짜리 짧은 리뷰 + 긍정(1) / 부정(0) 라벨. **5,000 train / 1,000 eval, seed 42** — Ch 15 와 *완전히 같은* 셋업.

**원본**: `e9t/nsmc` GitHub 의 `ratings_train.txt` / `ratings_test.txt` TSV. Hugging Face datasets hub 의 nsmc 레포는 *로더 스크립트* 기반이라 최신 datasets 라이브러리에서 deprecated — 그래서 GitHub raw URL 에서 직접 받습니다 (Ch 15 와 동일 패턴).

## 토크나이저 — `klue/bert-base` (Ch 22 와 동일)

vocab 약 32,000 의 한국어 WordPiece. MLM 사전학습과 분류 fine-tune 전 구간에서 *같은 토크나이저* 를 써야 본체가 학습한 임베딩의 의미가 유지됩니다.

## MLM 사전학습 — Ch 22 패턴 압축 재현 (한국어 Wikipedia, 1 epoch)

이 노트북을 *self-contained* 로 만들기 위해 Ch 22 의 MLM 사전학습을 여기서 짧게 재현합니다. Ch 22 보다 *짧은 1 epoch* (시간 단축) 라 사전학습 깊이는 얕지만, *random init 보다는 낫다* 는 차이를 만들기에는 충분합니다.

**MLM 사전학습 데이터는 *분류용 NSMC 와 별도*** — `wikimedia/wikipedia` config `20231101.ko` paragraphs 5K 를 *새로 로드*. 본 챕터의 *진짜 transfer 메시지* — *일반 한국어 위키 사전학습 → NSMC 영화 리뷰 분류 transfer* 가 노트북 한 구조에 자연스럽게 들어맞도록 *두 데이터셋이 공존*. 같은 토크나이저 (`klue/bert-base`) 가 두 도메인을 모두 처리.

같은 작은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024) → `BertForMaskedLM(config)` random init → 한국어 Wikipedia paragraphs 2K MLM **3 epoch** (1 epoch 만으로는 본체 정렬이 약해 *random init 보다 못한* 결과가 나올 수 있음 — Ch 21 부록 패턴 따라 3 epoch).

**`labels = -100` 한 줄 환기** — `DataCollatorForLanguageModeling` 이 가려지지 않은 자리에 `labels = -100` 을 채워 *해당 위치의 CE loss 를 무시*. 분류 fine-tune (다음 섹션) 에서는 -100 을 *전혀 사용하지 않습니다* — 모든 sample 에 *정답 라벨* (0/1) 이 명시되어 있기 때문. 같은 `-100` 트릭이 Phase 4 의 SFT (Ch 27) 에서 *prompt 자리* 를 가리는 정반대 자리로 다시 등장합니다 — 풀버전 표는 Ch 21 §3 *labels = -100 thread* 참조.

**관전 포인트** — 한국어 Wikipedia paragraphs 에서 MLM loss 가 *random baseline 10.37* 에서 시작해 5-7 부근까지 떨어졌다면 본체가 *일반 한국어 어휘·문맥 구조의 일부* 를 학습한 상태. perplexity 로 환산하면 vocab 약 32,000 중 *수백-수천 개 후보* 로 좁혀진 정도. Ch 22 의 2 epoch 보다는 약간 얕지만, *NSMC 분류 fine-tune 출발점* 으로는 충분합니다 — 본체가 *일반 한국어 구조* 를 가지면 *영화 리뷰 비격식 도메인* 도 fine-tune 으로 빠르게 적응.

> **체크포인트 저장은 생략** — 노트북 안에서 바로 본체 가중치를 분류 모델로 옮기기 때문. Ch 22 처럼 디스크에 저장하려면 `mlm_model.save_pretrained("./ch23_mlm_ckpt")` 한 줄.

## 헤드 교체 — MLM → 분류 + Fine-tune

이제 *방금 학습된 작은 한국어 BERT 본체* 를 분류 모델로 옮깁니다. 두 가지 흐름:

1. `BertForMaskedLM.bert` (embedding + encoder) 를 그대로 가져옴
2. 새 `BertForSequenceClassification(config)` 을 만들고, 1 의 본체를 *복사*. 분류 헤드는 새로 random init

이렇게 만든 모델을 같은 NSMC 데이터의 *라벨* 까지 사용해 분류 fine-tune. Ch 15 의 hyperparams 와 *완전히 같이* (`lr=2e-5, batch=16, epoch=2, fp16=True`) 둬서 *본체 출발점* 외 모든 조건을 통제.

**`bert.load_state_dict` 가 한 일** — `BertForMaskedLM` 과 `BertForSequenceClassification` 둘 다 *내부에 같은 `BertModel`* (이름 `self.bert`) 을 갖습니다. 그 본체만 통째로 옮긴 셈. MLM head (`cls.predictions`) 와 분류 head (`classifier`) 는 *모델 객체의 다른 자리* 라 자동으로 분리됩니다.

> Ch 7-18 의 `AutoModelForSequenceClassification.from_pretrained(...)` 가 디스크에서 같은 일을 합니다. 우리는 *방금 학습한 본체* 를 디스크 없이 in-memory 로 옮긴 셈. 디스크 경유는 부록이 아닌 FAQ Q3 에서 짧게 다룹니다.

## 평가 — Ch 15 / Ch 21 과 같은 5종 metric + 학습 곡선

`accuracy / precision / recall / F1 / AUC` 전부 같은 정의. 마지막에 confusion matrix 와 학습 곡선을 같이 그려 *본체 출발점 변화가 학습 동역학에 어떻게 드러나는지* 시각화.

### 5-1. 학습 곡선 — MLM 사전학습 효과가 보이는 자리

분류 fine-tune 의 step-by-step train loss 를 그려, *시작점* 과 *수렴점* 을 같이 확인.

### 5-2. Confusion matrix

## 2-way 비교 — Ch 15 (klue/bert-base) vs Ch 23 ours (small BERT + ko wiki MLM)

두 셋업을 한 표·한 막대 그래프로. Ch 15 의 수치는 *해당 노트북의 검증된 결과* 를 인용 — 학습자가 직접 Ch 15 노트북을 돌려 본인 수치로 갱신해 보면 더 좋습니다.

| 차원 | Ch 15 (klue/bert-base) | Ch 23 ours (small + MLM) |
|---|---|---|
| 본체 파라미터 | 약 110M | 약 10M |
| 사전학습 코퍼스 | 한국어 위키 + 모두의 말뭉치 + 뉴스 + 댓글 (약 8.4B 토큰) | 한국어 Wikipedia paragraphs 5K (약 50만-80만 토큰) |
| 사전학습 시간 | TPU 수일 | T4 약 8-10분 |
| Fine-tune 도메인 | NSMC 이진 (다른 도메인) | NSMC 이진 (다른 도메인) |
| 분류 fine-tune 셋업 | (둘 다 같음 — 5K/1K, batch 16, lr 2e-5, 2 epoch, fp16) | |

**관찰 — *동일 transfer 패턴 안에서 사전학습 규모 격차* 가 NSMC 정확도에 어떻게 드러나나**

전형적으로:
- **Ch 15** (`klue/bert-base`, 약 110M, 약 8.4B 토큰 대규모 한국어 사전학습): accuracy 약 85-88%, AUC 약 0.92-0.94
- **Ch 23 ours** (small BERT, 한국어 Wikipedia 2K paragraphs × 3 epoch 사전학습): accuracy 약 65-75%, AUC 약 0.75-0.85

**Ch 15 vs Ch 23 ours**: accuracy 약 10-20%p 격차. 두 모델이 *같은 transfer 패턴* (일반 한국어 위키 → NSMC) 을 따르므로 격차의 거의 전부가 *사전학습 규모의 가치* — 약 8.4B 토큰의 *일반 한국어 지식* 이 `klue/bert-base` 본체에 압축되어 있어, NSMC 같은 *비격식 구어체 도메인* 에도 빠르게 적응합니다. *작은 일반 사전학습 < 대규모 일반 사전학습* 의 본질은 단순 양적 차이가 아니라 *도메인 다양성의 질적 차이* — `klue/bert-base` 는 위키 + 뉴스 + 블로그·댓글까지 포함한 약 8.4B 토큰으로 비격식 한국어 도메인도 *이미 본 적이 있어* transfer 가 자연스럽습니다.

> NSMC 는 *짧은 한 줄 리뷰* 이고 *반어·맞춤법 흔들림·라벨 노이즈* 가 섞여 있어 영어 Yelp (Ch 21) 보다 *살짝 더 어려운* 데이터. 작은 모델 + 작은 사전학습 환경에서는 그 어려움이 더 두드러집니다 — 한국어 환경 특유의 *negative transfer 가능성* 까지 포함한 정량 비교는 부록을 참조하세요.

## 부록 — random init baseline + negative transfer 분석

*MLM 사전학습 없이 random init 으로 바로 분류 fine-tune* 한 결과와의 비교, 그리고 *한국어 위키 → NSMC 의 큰 도메인 gap* 에서 발생할 수 있는 **negative transfer** 현상의 분석은 부록 노트북 [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb) 에서 다룹니다. 영어 Ch 21 (transfer 양성) 과 한국어 Ch 23 (transfer 음성 가능) 의 비대칭 메커니즘이 핵심.

## 이 장의 구성

- [23-1. 실습](23-ko_bert_classify-practice.md)
- [23-2. 정리와 FAQ](23-ko_bert_classify-wrapup.md)
