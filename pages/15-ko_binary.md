**목표**: Phase 1 의 영어 DistilBERT 셋업을 *한국어 BERT* 로 옮깁니다. 모델 본체는 `klue/bert-base`, 데이터는 NSMC (네이버 영화 리뷰), task 와 loss 셋업은 Ch 11 과 *완전히 동일* — softmax + CrossEntropyLoss. 변하는 축은 **언어 + 토크나이저 + 데이터** 한 묶음.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 3분 (모델 다운로드 ~30s + 2 에폭 학습 약 1분 + 평가/시각화)

## 학습 흐름

1. 🔤 **토크나이저 비교**: 영어 WordPiece (`distilbert-base-uncased`) vs 한국어 WordPiece (`klue/bert-base`). 같은 한국어 문장이 양쪽에서 어떻게 *완전히 다르게* 쪼개지는지 직접 확인.
2. 🚀 **실습**: NSMC 5,000건 → klue/bert-base 파인튜닝 → 영화 리뷰 긍정/부정 분류
3. 🔬 **해부**: 학습 결과 — accuracy / F1 / AUC + 확률·logit 분포 KDE (Ch 10·11 의 한국어판)
4. 🛠️ **샘플 단위 해석**: 짧은 한국어 리뷰 몇 개를 골라 모델이 어떻게 판단했는지 읽어보기

> 📒 **사전 학습 자료**: Ch 11 (BERT Binary 방식 B — softmax+CE). 이번 챕터는 Ch 11 셋업의 *언어 swap* 버전이라 모델·loss·코드 골격은 동일. **Phase 2 핵심 학습 포인트는 토크나이저** 입니다.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 10 | DistilBERT | WordPiece (영어) | Yelp 이진화 | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| 11 | DistilBERT | WordPiece (영어) | Yelp 이진화 | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| **15 ← 여기 (Phase 2 시작)** | **`klue/bert-base`** | **WordPiece (한국어)** | **NSMC (네이버 영화 리뷰)** | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| 16 (다음) | klue/bert-base | WordPiece (한국어) | KLUE-YNAT (뉴스 7분류) | `Linear(H, 7)` | softmax | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 11)

| 축 | Ch 11 (영어 binary) | Ch 15 (한국어 binary) |
|---|---|---|
| **언어** | 영어 | **한국어** |
| 모델 본체 | `distilbert-base-uncased` (약 66M) | **`klue/bert-base`** (110M, BERT-base full size) |
| 토크나이저 | 영어 WordPiece (vocab 30K) | **한국어 WordPiece** (vocab 32K) |
| 데이터 | Yelp 이진화 (5K샘플 / max_len 128) | **NSMC** 5K샘플 / max_len 128 |
| `num_labels` | 2 | 2 (그대로) |
| `problem_type` | `single_label_classification` | (그대로) |
| Activation / Loss | softmax / CE | (그대로) |
| 라벨 형식 | int 0/1 | int 0/1 (그대로) |
| 학습 hyperparams | (epoch=2, lr=2e-5, batch=16, seed=42) | (그대로) |

> **Phase 2 의 `변경점 한 가지 원칙` 변형**: 입문 수준에선 *언어 + 토크나이저 + 데이터* 가 한 묶음으로 같이 변합니다. *모델·loss·셋업이 그대로* 라 가르침은 Phase 1 과 분리해 *한국어 자체의 학습 어려움* 에만 집중할 수 있습니다.

### 왜 한국어가 영어 BERT 와 *그렇게* 다른가

영어 distilbert-base-uncased 토크나이저로 한국어를 처리하면 *처참한* 토큰화가 나옵니다. 영어 vocab 에 한국어 글자가 없어서 *바이트 단위* (`[UNK]` 또는 `##` prefix 부스러기) 로 깨집니다. 모델이 학습한 단어 임베딩이 한국어를 못 받아내요.

**해결**: *한국어 텍스트로 사전학습된* BERT 와 그 토크나이저를 사용. 이번 챕터는 KLUE 연구팀의 `klue/bert-base` — 한국어 위키 + 뉴스 + 댓글 등으로 사전학습.

## Loss 노트 — Ch 11 그대로

`CrossEntropyLoss`. 새로운 점은 없습니다 — *binary 분류 셋업* 이라 K=2, softmax 후 정답 클래스 확률에 -log.

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log \hat p_{i, y_i} \quad\text{where}\quad \hat p_{i,k} = \dfrac{e^{z_{i,k}}}{e^{z_{i,0}} + e^{z_{i,1}}}$$

데이터 분포는 *NSMC 가 거의 완벽 균형* (긍정 ~50%, 부정 ~50%) 이라 random baseline loss = $\log 2 = 0.693$. 학습 첫 step 에서 loss 가 이 근처면 정상.

## 토크나이저 노트 — Phase 2 의 핵심

**Ch 11 까지는** `distilbert-base-uncased` (영어 WordPiece) 를 그대로 썼습니다. **Ch 15 부터는** `klue/bert-base` (한국어 WordPiece). 두 토크나이저가 *같은 한국어 문장* 을 어떻게 다르게 쪼개는지가 이번 챕터의 *교훈* 의 절반입니다.

### 직관 — 같은 문장, 두 토크나이저

문장: `"이 영화 정말 재미있었어요"`

| 토크나이저 | 결과 토큰 | 토큰 수 |
|---|---|---|
| `distilbert-base-uncased` (영어) | `['ᄋ', '##ᅵ', 'ᄋ', '##ᅧ', '##ᆼ', '##ᄒ', '##ᅪ', 'ᄌ', '##ᅥ', '##ᆼ', '##ᄆ', '##ᅡ', '##ᆯ', '[UNK]']` 같이 *자모(초·중·종성) 단위* 로 산산조각 (또는 [UNK] 가득) | 14 |
| `klue/bert-base` (한국어) | `['이', '영화', '정말', '재미있', '##었', '##어요']` — *의미 있는 어휘* 단위 | 6 |

영어 토크나이저는 한국어를 *낯선 문자열* 로 보고 글자 단위까지 쪼갭니다. 한국어 토크나이저는 *재미있·었·어요* 를 어휘적 의미 단위로 분할 — 모델이 임베딩을 통해 *의미* 를 잡을 수 있는 형태.

이 비교는 §실습 직전에 *코드로 직접* 확인합니다.

### 한국어 WordPiece 의 특징

- vocab 32,000 (영어 30K 와 비슷한 규모)
- *어절 단위* 가 아니라 *형태소-비슷한* 서브워드 단위. 예: "재미있었어요" → "재미있" + "##었" + "##어요" (어간 + 어미)
- 영어처럼 `##` prefix 가 *이전 토큰에 이어지는 부분* 을 표시
- 한자·숫자·영어 단어도 vocab 에 포함 (한국어 텍스트엔 흔히 섞여 있음)

### Phase 2 의 토크나이저는 *이번 챕터부터 끝까지* `klue/bert-base` 고정

Ch 16, 17, 18 도 같은 토크나이저. 변하는 건 *데이터·task* 만. 영어 → 한국어 전환은 *이 챕터에서 한 번* 일어나고, 이후엔 한국어 셋업이 표준.

## 이 장의 구성

[[SubPages]]
