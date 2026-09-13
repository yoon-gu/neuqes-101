# 15_ko_binary — 한국어 BERT Binary (NSMC) — Phase 2 시작

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/15_ko_binary/15_ko_binary.ipynb)

## 한 줄 목표
Ch 11 의 BERT binary 셋업을 한국어로 옮깁니다. 모델·loss·셋업은 그대로 (softmax + CE), 변하는 건 **언어 + 토크나이저 + 데이터** 한 묶음 — `distilbert-base-uncased` → `klue/bert-base`, Yelp → NSMC.

## 다루는 핵심 개념
- 한국어 WordPiece 토크나이저 vs 영어 WordPiece 토크나이저 — 같은 한국어 문장을 둘이 어떻게 *전혀 다르게* 쪼개는지 직접 비교
- `klue/bert-base` (BERT-base full size, 110M) 와 영어 DistilBERT (67M) 파라미터 비교
- NSMC 데이터셋 — datasets hub 의 로더 스크립트가 deprecated 라 GitHub raw TSV 직접 다운로드
- Ch 11 의 모든 평가·시각화 패턴 (확률 KDE, logit KDE, 분류 리포트) 를 한국어 환경에서 재사용
- 샘플 단위 해석 — 모델이 가장 자신있는/망설이는 한국어 리뷰 직접 읽어보기
- 한국어 sentiment task 의 어려움 — 짧은 리뷰, 반어, 라벨 노이즈

## Loss
**`CrossEntropyLoss`** — Ch 11 그대로. K=2 binary 셋업.

## 데이터
NSMC (네이버 영화 리뷰) — GitHub raw TSV 직접 다운로드 (`https://raw.githubusercontent.com/e9t/nsmc/master/`). 5K train / 1K eval 로 subsample. 거의 완벽 균형 (긍정/부정 ~50/50).

## 환경
Google Colab **T4 GPU 필수**. 약 3분 (모델 다운로드 ~30s + BERT-base 110M × 2 epoch 약 1분 + 평가/시각화).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 11 | DistilBERT | 영어 WordPiece | Yelp 이진화 | `Linear(H, 2)` | `CrossEntropyLoss` |
| **15** | **`klue/bert-base`** | **한국어 WordPiece** | **NSMC** | `Linear(H, 2)` (그대로) | (그대로) |
| 16 (다음) | klue/bert-base | 같음 | KLUE-YNAT | `Linear(H, 7)` | (그대로) |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[16_ko_multiclass](../16_ko_multiclass/) — 같은 토크나이저·모델, K=2 → K=7 (KLUE-YNAT 뉴스 분류).
