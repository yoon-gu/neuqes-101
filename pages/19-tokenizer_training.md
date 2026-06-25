**목표**: Phase 3 의 첫 챕터. 지금까지 (Ch 7-18) 우리는 *사전학습된* 토크나이저 (`distilbert-base-uncased`, `klue/bert-base`) 를 그저 *불러* 썼습니다. 이번 챕터에서 **토크나이저 자체를 직접 학습** 해 봅니다 — *어떻게* 만들어지는지, 알고리즘이 다르면 *어떻게 다른 결과* 가 나오는지를 두 언어 × 두 알고리즘 = 네 가지 조합으로 비교.

**환경**: Google Colab T4 (모델 학습이 없어 GPU 거의 안 씀, 일관성 위해 T4 metadata 유지).

**예상 소요 시간**: 약 5-7분 (데이터 다운로드 -2분 + 토크나이저 4종 학습 -2분 + 시각화·비교)

## 학습 흐름

1. 🔤 **이론**: WordPiece (subword) vs WordLevel (어절 단위). BERT 가 WordPiece 를 쓰는 이유.
2. 📥 **코퍼스 준비**: 영어 Yelp text 5,000건 + 한국어 NSMC text 5,000건.
3. 🚀 **실습** — 4개 토크나이저 학습:
   - 영어 Yelp + WordPiece (vocab 8000)
   - 한국어 NSMC + WordPiece (vocab 8000)
   - 영어 Yelp + WordLevel (vocab 8000)
   - 한국어 NSMC + WordLevel (vocab 8000)
4. 🔬 **해부**: 같은 문장을 4 토크나이저에 통과 → 토큰 시퀀스 비교.
5. 📊 **비교 시각화**: 토큰 길이 분포 (4 토크나이저 × 같은 텍스트), unknown 비율, 2×2 요약 표.
6. 💾 **저장·로드**: `tokenizer.save()` / `Tokenizer.from_file()` + `PreTrainedTokenizerFast` 로 HF 인터페이스 변환.
7. 🛠️ **변형**: vocab 크기 sweep (1K / 4K / 8K / 16K).

> 📒 **사전 학습 자료**: Ch 7 (DistilBERT WordPiece 첫 사용), Ch 15 (한국어 WordPiece). 토크나이저는 이전 챕터들에서 *결과* 만 사용했고, 이번 챕터는 그 결과가 *어떻게 만들어지는지* 를 직접 봅니다.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 17 | klue/bert-base | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | `BCEWithLogitsLoss` |
| 18 | klue/bert-base + 보조 | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | `BCEWithLogitsLoss + λ·L_aux` |
| **19 ← 여기 (Phase 3 시작)** | — (토크나이저 학습 전용) | **WordPiece + WordLevel** (둘 다 *직접 학습*) | **Yelp text + NSMC text** | — | — |
| 20 (다음) | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Wikitext-103 (일반 도메인 위키) | MLM head | `CrossEntropyLoss` (masked) |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

**Phase 3 의 위치** — Ch 19 가 토크나이저 토대, Ch 20-23 은 그 위에 올린 작은 BERT 의 사전학습 + 분류. 단, Ch 20 부터는 *학습 안정성* 을 위해 표준 사전학습 토크나이저 (`bert-base-uncased`, `klue/bert-base`) 를 그대로 가져와 씁니다. Ch 19 는 "토크나이저가 어떻게 학습되나" 를 *경험* 하는 챕터.

## 변경점 (Diff from Ch 18)

| 축 | Ch 18 (한국어 auxiliary) | Ch 19 (토크나이저 학습 전용) |
|---|---|---|
| **이 챕터의 task** | 분류 (메인 multi-label + 보조) | **분류 아예 없음 — 토크나이저 학습 그 자체가 결과물** ← *유일한 변화* |
| 모델 | `klue/bert-base` 파인튜닝 | 없음 |
| 토크나이저 | 사전학습된 것 *로드만* | **WordPiece + WordLevel 둘 다 *직접 학습*** |
| 데이터 | KLUE-YNAT 합성 multi-label + 보조 라벨 | Yelp text + NSMC text (라벨 무시, *문장만* 사용) |
| Output / Loss | 메인 + 보조 + λ | 없음 (vocab + merge rules 가 산출물) |
| 평가 metric | F1 / AUC / MAE | 토큰 길이 분포 / UNK 비율 / vocab 커버리지 |

> **변경점 한 가지 원칙** — Phase 2 (분류 task) 의 종착인 Ch 18 에서 *task 자체* 가 토크나이저 학습으로 바뀝니다. 모델·loss·평가가 모두 사라지는 큰 전환이지만, 이건 *축 자체* 가 바뀌는 Phase 경계라 자연스러움. Phase 3 의 다음 챕터부터 다시 모델 학습으로 돌아갑니다.

### 왜 토크나이저를 *직접* 학습해야 하나

새 도메인 (의료·법률·코드 등) 이나 새 언어로 BERT 를 사전학습할 때, 기존 토크나이저는 *그 도메인의 어휘* 를 제대로 못 쪼갭니다. 예: 영어 토크나이저로 한국어를 처리하면 거의 모든 토큰이 `[UNK]` 가 됩니다. 도메인·언어에 맞춘 토크나이저를 직접 학습해야 모델이 *압축된* 정보를 받을 수 있어요.

## 토크나이저 알고리즘 노트 — WordPiece vs WordLevel

이번 챕터의 핵심 비교는 *Loss 변화* 가 아니라 **토크나이저 알고리즘 변화** 입니다. 두 알고리즘은 분할 단위 자체가 다릅니다.

### WordLevel — 단순 어절 (whole-word)

공백 단위로 단어를 잘라 그대로 vocab 에 등록. *변형* 처리 없음.

- 학습: 코퍼스에서 *빈도 상위 N 단어* 를 골라 vocab 에 넣음. 나머지는 모두 `[UNK]`.
- 장점: 토큰화가 *즉시* 이해 가능 (한 단어 = 한 토큰).
- 단점: vocab 밖 단어 = 전부 `[UNK]`. 영어처럼 굴절·합성이 많은 언어에서 OOV 가 폭증.

### WordPiece — subword (BERT 표준)

자주 등장하는 *조각* 을 vocab 에 등록하고, 모르는 단어는 더 작은 조각으로 *재귀적으로 쪼갬*. BERT 시리즈 (BERT / DistilBERT / KLUE-BERT) 가 이 알고리즘을 씁니다.

- 학습: 처음에는 *문자 단위 vocab* 부터 시작 → likelihood 가 가장 많이 오르는 *문자 쌍* 을 vocab 에 병합 → vocab 목표 크기까지 반복.
- 처음 등장 토큰은 `playing`, 그 뒤를 잇는 조각은 `##ing` 처럼 `##` prefix.
- 장점: 모르는 단어도 *작은 조각* 으로 쪼개 표현 가능 → `[UNK]` 거의 안 남음.
- 단점: 짧은 단어 하나가 여러 조각으로 쪼개져 토큰 수가 늘어남.

### 수치 예시 (같은 문장이 두 알고리즘에서 몇 토큰?)

학습된 두 토크나이저가 *같은 문장* 을 어떻게 처리하나를 미리 감 잡기 — 실제 결과는 §4 에서 직접 확인합니다.

| 문장 | WordPiece (vocab 8K) | WordLevel (vocab 8K) |
|---|---|---|
| `"the food was great"` (4 단어, 흔한 어휘) | 4 토큰 (1 단어=1 piece) | 4 토큰 (모두 vocab) |
| `"unforgettable experience"` (드문 어휘) | 5 토큰 (`un`, `##forget`, `##table`, `experience`) | 2 토큰 — 단, `unforgettable` 이 vocab 밖이면 `[UNK]` |
| `"이 영화 정말 재미있어요"` (한국어) | 5-7 토큰 (조사·어미 분리) | 4 토큰 (어절 단위) — 단, `재미있어요` vocab 밖이면 `[UNK]` |

핵심 관찰: **WordLevel 은 "단어가 vocab 에 있느냐 없느냐" 의 binary** — 있으면 1 토큰, 없으면 `[UNK]`. **WordPiece 는 "얼마나 잘게 쪼갤지" 의 spectrum** — 흔한 단어는 1 토큰, 드문 단어는 여러 조각.

### BERT 가 WordPiece 를 쓰는 이유

(1) `[UNK]` 토큰이 거의 안 생겨 모든 단어가 *학습 가능* 한 표현으로 인코딩됨. (2) vocab 크기가 작아도 (BERT 30K, KLUE 32K) *어휘 커버리지* 가 좋음. (3) 미세한 형태 차이 (`run`/`running`/`runs`) 가 *공통 조각* 으로 묶여 표현 학습이 효율적.

## 토크나이저 노트 — 이 챕터의 *주제* 자체

지금까지의 모든 챕터에서 마지막 자리를 차지하던 이 섹션이, 이번엔 챕터의 *전체* 입니다.

- **Ch 7-14**: 영어 WordPiece (`distilbert-base-uncased`, vocab 30K, 사전학습) 받아 쓰기.
- **Ch 15-18**: 한국어 WordPiece (`klue/bert-base`, vocab 32K, 사전학습) 받아 쓰기.
- **Ch 19 (지금)**: *같은 WordPiece 알고리즘* + 단순 비교용 WordLevel 을 두 언어 코퍼스에서 *직접* 학습.
- **Ch 20+**: 다시 표준 사전학습 토크나이저로 — 단, *이제는 그 안에서 무엇이 벌어지는지 알고* 사용.

같은 알고리즘이 *언어가 바뀌면 어떻게 다른 결과* 를 내는지 (영어 vs 한국어), *알고리즘이 바뀌면 어떻게 다른 결과* 를 내는지 (WordPiece vs WordLevel) — 2×2 비교가 이 챕터의 핵심.

## 이 장의 구성

[[SubPages]]
