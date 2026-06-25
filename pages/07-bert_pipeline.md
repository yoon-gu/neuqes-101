**목표**: sklearn 시대를 마치고 `transformers` 라이브러리를 만납니다. **5줄짜리 코드** 로 사전학습된 DistilBERT를 돌려보고, 그 한 줄 뒤에 어떤 일이 일어났는지 단계별로 풀어 헤칩니다.

**환경**: Google Colab — **T4 GPU 권장** (런타임 → 런타임 유형 변경 → T4 GPU). 이번 챕터부터 GPU 메모리(VRAM) 추적이 등장하니 GPU 런타임에서 돌리면 모델 로드 → VRAM 증가가 한눈에 보입니다. CPU 런타임도 추론 자체는 동작하지만 `!nvidia-smi` 셀은 에러납니다.

**예상 소요 시간**: 약 10분 (학습 없음, 추론만)

## 학습 흐름

1. 🚀 **실습**: `pipeline("sentiment-analysis")` 한 줄로 감성 분석 돌리기
2. 🔬 **해부**: `pipeline` 안에서 일어나는 3단계 (tokenizer / model / post-processing)
3. 🛠️ **변형**: `pipeline` 없이 같은 일을 4단계로 직접 재현

## 변화추적표

**Phase 1 시작** — sklearn 시대 끝, `transformers` 등장.

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 이진화 | (2차원) | softmax | `CrossEntropyLoss` |
| 5 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| 6 | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 항목 합성 | (5차원) | per-label sigmoid | per-label `BCEWithLogitsLoss` |
| **7 ← 여기** | `pipeline("sentiment-analysis")` | `AutoTokenizer.from_pretrained(...)` | 간단 영어 예시 | **사전학습 헤드** | softmax | — (추론만) |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 6)

| 축 | Ch 6 | Ch 7 |
|---|---|---|
| 라이브러리 | `sklearn` | **`transformers`** (Hugging Face) |
| 모델 | `OneVsRestClassifier(LogisticRegression())` (학습) | **`pipeline("sentiment-analysis")`** (사전학습 + 추론) |
| 토크나이저 | `TfidfVectorizer()` (단어 단위 어휘 학습) | **`AutoTokenizer`** (사전학습된 WordPiece) |
| 학습 단계 | sklearn `fit()` 한 번에 학습 | **학습 없음** — 사전학습 가중치 로드 후 추론만 |
| 데이터 | Yelp 5,000건 | 간단 영어 예시 문장 (분해 시연용) |
| 하드웨어 | CPU | CPU 또는 T4 GPU (이번 챕터는 추론만이라 어느 쪽도 OK) |

**왜 학습 없이 시작하나?** Phase 1 첫 챕터는 `transformers` 의 *추상화 계층* 을 익히는 데 집중합니다. `pipeline` 한 줄 뒤에 토크나이저·모델·후처리 3단계가 어떻게 굴러가는지 손에 잡히면, Ch 8(Tokenizer/Datasets 해부)와 Ch 9(BERT 회귀 첫 학습)에서 `Trainer` 가 등장할 때 코드를 *읽을* 수 있습니다.

## 토크나이저 노트 — 첫 WordPiece 등장

이번 챕터의 토크나이저는 **사전학습된 WordPiece**. Phase 0의 `TfidfVectorizer` 와 *완전히 다른 패러다임* 입니다.

| 비교 | TF-IDF (Phase 0) | WordPiece (Phase 1+) |
|---|---|---|
| 분리 단위 | 단어 (whitespace + 정규식) | **서브워드** (자주 등장하는 문자 시퀀스) |
| 어휘 출처 | 학습 데이터에서 그때그때 학습 | **사전학습된 30,522개 어휘** (BERT 학습 시 정해짐) |
| OOV 처리 | 그냥 무시 | `[UNK]` 또는 더 작은 서브워드로 분해 |
| 특수 토큰 | 없음 | `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]` 등 |
| 출력 | sparse vector (V차원, 거의 0) | 정수 ID 시퀀스 + attention mask |

같은 문장 `"I love using Hugging Face!"` 가 어떻게 토큰화되는지 곧 직접 확인합니다 (Step 2). `##` 접두사가 보이는 단어는 어디고, 왜 그렇게 쪼개졌는지도 같이 봅니다.

> **다음 챕터(Ch 8)**: 같은 WordPiece 토크나이저를 *깊게* — `padding`, `truncation`, `max_length` 옵션과 `datasets` 라이브러리 메모리 효율까지.

## 이 장의 구성

[[SubPages]]
