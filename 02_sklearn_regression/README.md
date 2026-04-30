# 02_sklearn_regression — 첫 모델, 첫 Loss

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/02_sklearn_regression/02_sklearn_regression.ipynb)

## 한 줄 목표
Ch 1의 TF-IDF 입력 위에 **`LinearRegression`** 을 얹어 별점을 회귀합니다 — 활성화 함수도 출력 범위 제약도 없는, "그냥 숫자를 뱉는" 가장 단순한 모델 형태부터 시작합니다.

## 다루는 핵심 개념
- `LinearRegression` 닫힌 형태 풀이 (정규방정식)
- 첫 Loss로 **`MSELoss`** 등장 — 수식과 직관, sklearn `mean_squared_error`로 직접 재현
- "출력은 그냥 숫자" — 예측이 1-5 범위를 벗어나는 게 자연스러운 이유
- 라벨을 [0, 1]로 정규화해도 출력이 [0, 1]을 못 지키는 한계 → 다음 챕터의 sigmoid 떡밥
- MSE vs MAE, ordinal label에서 회귀 vs 분류 선택

## 데이터
`yelp_review_full` 5,000건 (Ch 1과 동일 샘플), 라벨은 별점 1-5 (`label + 1`).

## 환경
Google Colab CPU 런타임으로 충분합니다 (GPU 불필요). 약 5-10분 소요 — 학습 자체는 1초 내.

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Loss |
|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer` | Yelp 5,000 샘플 | — |
| **2** | `LinearRegression()` | TF-IDF | Yelp (별점 1-5) | `MSELoss` |

전체 18챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[03_sklearn_binary](../03_sklearn_binary/) — 출력에 sigmoid가 붙고 loss가 `BCEWithLogitsLoss`로 바뀝니다. 별점은 4-5 → 1, 1-2 → 0으로 이진화.
