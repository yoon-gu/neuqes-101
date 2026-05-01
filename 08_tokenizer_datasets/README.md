# 08_tokenizer_datasets — Tokenizer 옵션 깊이 + `datasets` 라이브러리

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/08_tokenizer_datasets/08_tokenizer_datasets.ipynb)

## 한 줄 목표
Ch 7에서 만난 WordPiece 토크나이저와 사전학습 모델로 **Phase 0의 Yelp 데이터를 다시** 만납니다. `padding`/`truncation`/`max_length` 옵션을 손에 익히고, `datasets` 라이브러리로 65만 건 코퍼스를 메모리 걱정 없이 다룹니다 — Ch 9 학습의 *입력 파이프라인* 이 이 챕터에서 완성됩니다.

## 다루는 핵심 개념
- `datasets.load_dataset` — Apache Arrow + memory-mapped 캐시
- `Dataset.shuffle / select / map / filter / with_format` 패턴
- `tokenizer(...)` 옵션 3종: `padding=True/="max_length"`, `truncation=True`, `max_length=N`
- `attention_mask` 가 self-attention에서 패딩을 무시하는 원리 (`-inf` softmax 트릭)
- 토큰 길이 분포 → `max_length` 결정 (Yelp 데이터 95th percentile 기반)
- `DataLoader` + `DataCollatorWithPadding` — Ch 9 `Trainer` 입력 형태 미리보기

## 데이터
`yelp_review_full` (Hugging Face Hub) — 학습 65만 건 / 테스트 5만 건. **5,000건 subsample** 사용 (Phase 0과 동일).

## 환경
Google Colab — **CPU도 OK** (이번 챕터도 학습·추론 없음). T4 권장. 약 10분.

> 모델 가중치 로드 없음 → VRAM 거의 변화 없음. 데이터 로드 + 토크나이저 캐시만.

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Loss |
|---|---|---|---|---|
| 7 | `pipeline("sentiment-analysis")` | `AutoTokenizer.from_pretrained(...)` | 간단 영어 예시 | — |
| **8** | (모델 로드 없음) | `AutoTokenizer.from_pretrained(...)` + 옵션 | **Yelp 5,000 (Phase 0과 동일)** | — |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[09_bert_regression](../09_bert_regression/) — BERT 첫 파인튜닝. `Trainer` 본격 등장, `MSELoss` 회귀 (Ch 2와 같은 task를 BERT로 다시), GPU 필수, fp16.
