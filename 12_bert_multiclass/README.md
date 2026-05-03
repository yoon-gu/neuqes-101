# 12_bert_multiclass — BERT Multi-class (Yelp 5클래스)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/12_bert_multiclass/12_bert_multiclass.ipynb)

## 한 줄 목표
Ch 11(BERT binary) 셋업을 그대로 두고 **클래스 개수만 K=2 → K=5** 로 일반화. Yelp 별점 1-5를 5클래스 분류로 풀고, Ch 5(sklearn multinomial LogReg)의 BERT 버전과 직접 비교합니다.

## 다루는 핵심 개념
- `num_labels=5` + `problem_type="single_label_classification"` — Ch 11 셋업의 자연스러운 확장
- multi-class 평가 지표: macro precision/recall/F1, multi-class AUC (`multi_class="ovr"`)
- *혼동 행렬* (`seaborn.heatmap` + 행 정규화) 으로 클래스별 패턴 진단
- top-1 확률 분포를 정답/오답으로 갈라 보는 calibration 진단
- Random baseline loss = $\log K$ (K=5 → 1.609) — 학습이 시작됐는지 즉시 검증
- *상대 logit* 만 의미 있다 — softmax의 K-1 자유도

## Loss
**`CrossEntropyLoss`** — Ch 11과 동일. K가 2에서 5로 늘었을 뿐.

## 데이터
Yelp 5,000건 → 별점 1-5를 *그대로* 5클래스로 (Ch 3-4·10-11처럼 이진화 안 함). 라벨은 0-4 int 인덱스.

## 환경
Google Colab **T4 GPU 필수**. 약 12분 (BERT 학습 ~10분 + sklearn 비교 ~30초 + 평가/시각화).

**Self-contained**: 다른 챕터 / 다른 세션 결과에 의존하지 않습니다. 5장 비교용 sklearn baseline은 같은 노트북에서 inline 학습.

## 변화 추적

| Ch | 모델 | 데이터 | Output | Activation | Loss |
|---|---|---|---|---|---|
| 5 | LogReg(multinomial) | TF-IDF Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| 11 | DistilBERT | Yelp 이진화 | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| **12** | DistilBERT | **Yelp 5클래스** | **`Linear(H, 5)`** | softmax | `CrossEntropyLoss` |
| 13 (다음) | DistilBERT | Yelp + 측면 라벨 | `Linear(H, 5)` | sigmoid (per-label) | `BCEWithLogitsLoss` (per-label) |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[13_bert_multilabel](../13_bert_multilabel/) — 같은 num_labels=5, 같은 BERT, *task 만 single-label → multi-label*. 한 리뷰가 *여러 측면* 라벨을 동시에 가질 수 있는 케이스.
