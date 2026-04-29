# 01_tfidf — TF-IDF로 만나는 첫 벡터

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/01_tfidf/01_tfidf.ipynb)

## 한 줄 목표
자연어 모델을 만나기 전에 텍스트를 숫자 벡터로 바꾸는 가장 단순한 방법 — `CountVectorizer` / `TfidfVectorizer` — 을 손에 익힙니다.

## 다루는 핵심 개념
- 데이터 로딩과 탐색 (Yelp 리뷰 5,000건 샘플)
- 단어 단위 토큰화 + 어휘 학습 (sklearn 벡터라이저가 토크나이저 역할도 겸함)
- sparse 행렬 / sparsity / vocabulary
- TF-IDF의 IDF 가중치 직관 ("그 문서를 특징짓는 단어"를 골라내는 원리)
- "BERT 임베딩과 어떻게 다를까?" 떡밥

## 데이터
`yelp_review_full` (Hugging Face Hub) — 별점 1-5점 식당 리뷰. 학습 흐름을 가볍게 유지하기 위해 5,000건만 무작위 샘플링.

## 환경
Google Colab CPU 런타임으로 충분합니다 (GPU 불필요). 약 5분 소요.

## 변화 추적

| Ch | 모델 | 토크나이저 | Output Head | Activation | Loss |
|---|---|---|---|---|---|
| **1** | (TF-IDF) | `TfidfVectorizer` (학습) | — | — | — |

전체 18챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[02_sklearn_regression](../02_sklearn_regression/) — `LinearRegression`으로 별점을 회귀하면서 첫 Loss(`MSELoss`)가 등장합니다.
