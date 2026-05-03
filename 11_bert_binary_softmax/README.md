# 11_bert_binary_softmax — BERT Binary 방식 B (softmax + CE)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/11_bert_binary_softmax/11_bert_binary_softmax.ipynb)

## 한 줄 목표
Ch 10에서 학습한 *방식 A* (sigmoid+BCE, `num_labels=1`)와 짝을 이루는 **방식 B** (softmax+CE, `num_labels=2`)를 같은 BERT·같은 Yelp 이진화 데이터로 학습한 뒤, 두 방식의 결과를 직접 비교해 *두 방식 동등성* 이 BERT에서도 성립함을 확인합니다.

## 다루는 핵심 개념
- `num_labels=2` + `problem_type="single_label_classification"` — BERT 분류의 *표준* 셋업
- 라벨은 int 스칼라 (Ch 10의 `[float(b)]` ↔ Ch 11의 `int(b)` 한 줄 차이)
- 2차원 logit `(z_0, z_1)` 에서 *방식 A 호환* 1차원 logit 만들기: $z = z_1 - z_0$
- 직접 구현하는 안정 softmax (`exp(x - max) / sum`)
- Ch 10 저장 결과와의 직접 비교: metric 표 / scatter plot ($p_A$ vs $p_B$) / 예측 일치율 / 4분면 분석
- `id2label` / `label2id` 등록의 실무적 가치

## Loss
**`CrossEntropyLoss`** — Ch 4 sklearn `LogisticRegression()` (multinomial 자동) 과 같은 식. 다른 점은 logit 출처가 BERT 768-dim hidden state.

## 두 방식 동등성 핵심
$\sigma(z) = \mathrm{softmax}(z_0, z_1)[1]$ when $z = z_1 - z_0$. 수학은 똑같고, 모델 출력 shape과 라벨 형식만 다른 *동일 함수의 두 가지 표현* 입니다.

## 데이터
Ch 10과 *완전히 동일* — Yelp 5,000건 → 별점 3 제외 + 이진화 (4-5 → 1, 1-2 → 0). 같은 seed=42, 같은 5,000/1,000 split. 마지막 비교가 의미를 가지려면 데이터·모델 본체·hyperparams가 모두 같아야 합니다.

## 환경
Google Colab **T4 GPU 필수**. 약 20분 (방식 B ~8분 + 비교용 방식 A ~8분 + 평가/시각화).

**Self-contained**: Ch 10 노트북이나 다른 세션 결과에 의존하지 않습니다. 5장 비교 단계에서 *이 노트북 안* 에서 방식 A를 한 번 더 학습합니다 — 다른 날 / 다른 세션에서 열어도 처음부터 끝까지 그대로 돕니다.

## 변화 추적

| Ch | 모델 | 데이터 | Output | Activation | Loss |
|---|---|---|---|---|---|
| 10 | DistilBERT 파인튜닝 | Yelp 이진화 | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| **11** | 같음 | **같음** | **`Linear(H, 2)`** | **softmax** | **`CrossEntropyLoss`** |
| 12 (다음) | 같음 | Yelp 5클래스 | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[12_bert_multiclass](../12_bert_multiclass/) — Yelp 별점 1-5 5클래스 분류. 셋업은 Ch 11과 동일, `num_labels` 만 2 → 5.
