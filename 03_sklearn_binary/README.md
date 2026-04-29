# 03_sklearn_binary — 출력에 sigmoid가 붙다

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/03_sklearn_binary/03_sklearn_binary.ipynb)

## 한 줄 목표
Ch 2 회귀의 한계 — 출력이 [0, 1]을 못 지킨다 — 를 모델 단계에서 강제로 해결합니다. 출력 직전에 **sigmoid** 가 붙고, loss는 **BCE** 로, 라벨은 정수 0/1로 일제히 바뀝니다.

## 다루는 핵심 개념
- `LogisticRegression`의 두 단계: logit($w^\top x + b$) → sigmoid → 확률
- 첫 분류 Loss로 **`BCEWithLogitsLoss`** (sklearn: log loss) 등장 — 정답 확률에 가까울수록 0, 멀수록 로그 스케일로 폭증
- `predict_proba`로 확률을 직접 보고 sigmoid 적용 결과를 **수동으로 재현**
- 임계값(threshold)을 0.5에서 옮기면 precision/recall이 정반대로 움직이는 trade-off
- accuracy / precision / recall / F1을 언제 보는지

## 데이터
`yelp_review_full` 5,000건에서 **별점 3 제외**, 4-5 → 1(positive), 1-2 → 0(negative)으로 이진화 (약 4,000건 남음, 긍정 약 60%).

## 환경
Google Colab CPU 런타임으로 충분 (GPU 불필요). 약 5-10분.

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Activation | Loss |
|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer` | Yelp 5,000 | — | — |
| 2 | LinearReg | TF-IDF | Yelp (별점 1-5) | 없음 | `MSELoss` |
| **3** | LogReg | TF-IDF | Yelp 이진화 | **sigmoid** | **`BCEWithLogitsLoss`** |

전체 18챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[04_sklearn_multiclass](../04_sklearn_multiclass/) — sigmoid가 softmax로 바뀌고 출력이 1차원에서 5차원으로 늘어납니다. Loss는 BCE에서 `CrossEntropyLoss`로 일반화.
