# 07_bert_pipeline — BERT 첫 만남

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/07_bert_pipeline/07_bert_pipeline.ipynb)

## 한 줄 목표
**Phase 1 시작.** sklearn 시대를 마치고 `transformers` 라이브러리를 만납니다. `pipeline("sentiment-analysis")` 한 줄로 사전학습된 DistilBERT를 돌려보고, 그 한 줄 뒤의 3단계(tokenizer / model / post-processing)를 4단계로 직접 재현합니다.

## 다루는 핵심 개념
- `transformers.pipeline` — 추론 원스톱 함수
- 사전학습된 DistilBERT (SST-2 영화 리뷰 감성 모델)
- `AutoTokenizer` / `AutoModelForSequenceClassification` 의 `Auto` 패턴
- **첫 WordPiece 만남** — `##` 접두사, `[CLS]`/`[SEP]` 특수 토큰, 사전학습 어휘
- TF-IDF(Phase 0) vs WordPiece(Phase 1+) 패러다임 비교
- `pipeline` 의 4단계 수동 재현: `from_pretrained` → `tokenize` → `model.forward` → `softmax + argmax`

## 데이터
간단한 영어 예시 문장들 (분해 시연용). 학습 없이 추론만이라 데이터셋 로드는 없습니다.

## 환경
Google Colab — **CPU도 OK** (이번 챕터는 추론만이라 학습이 없음). T4 GPU면 더 빠릅니다 (런타임 → 런타임 유형 변경 → T4 GPU). 약 10분.

> 첫 실행 시 모델 다운로드(~250MB)에 30초~1분 걸림. 두 번째부터는 캐시.

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Loss |
|---|---|---|---|---|
| 6 | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 측면 합성 | per-label `BCE` |
| **7** | **`pipeline("sentiment-analysis")`** | **`AutoTokenizer.from_pretrained(...)`** | 간단 예시 | — (추론만) |

전체 19챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[08_tokenizer_datasets](../08_tokenizer_datasets/) — WordPiece 토크나이저의 `padding`/`truncation`/`max_length` 옵션과 `datasets` 라이브러리(`load_dataset`/`map`/Apache Arrow). Ch 9 학습의 데이터 파이프라인 준비.
