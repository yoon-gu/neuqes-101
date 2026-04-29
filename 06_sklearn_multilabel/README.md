# 06_sklearn_multilabel — softmax 합=1 제약을 푼다

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/06_sklearn_multilabel/06_sklearn_multilabel.ipynb)

## 한 줄 목표
한 샘플에 *여러* 라벨이 동시에 붙는 multi-label 문제로 확장. softmax의 *상호배타* 가정을 풀고 K개 sigmoid가 라벨마다 독립적으로 0/1을 결정합니다.

## 다루는 핵심 개념
- 측면(aspect) 키워드 합성으로 5차원 multi-hot 라벨 만들기 (food/service/price/ambiance/location)
- `OneVsRestClassifier(LogisticRegression())` + multi-hot Y 자동 인식
- per-label sigmoid + per-label BCE 평균 — 라벨끼리 독립
- 평가 지표: **subset accuracy / hamming loss / micro F1 / macro F1** (각각 언제 보는지)
- 라벨별 임계값 조정 (단일 임계값 sweep + 라벨별 최적 임계값 FAQ)
- **합성의 한계** 솔직히 짚기 — 부정·반어 무시, 사전 협소, 모델이 휴리스틱을 다시 학습

## Loss 수치 예시 (K=5, 정답 [1, 0, 1, 0, 1])
| 시나리오 | 예측 확률 | 평균 BCE |
|---|---|---|
| 잘 맞춤 | `[0.9, 0.1, 0.8, 0.2, 0.6]` | 0.233 |
| 균등 (baseline) | `[0.5, 0.5, 0.5, 0.5, 0.5]` | 0.693 |
| 정반대로 자신감 | `[0.1, 0.9, 0.1, 0.9, 0.1]` | 2.303 |

baseline = $\log 2 \approx 0.693$ — 모든 라벨에 0.5를 줄 때.

## 데이터
Yelp 5,000건 + **측면 키워드 매칭** 으로 5차원 multi-hot 합성. 활성률: food 56% / service 50% / price 29% / ambiance 18% / location 22%. 샘플당 평균 1.75개 라벨. 빈 라벨 샘플 약 15%.

## 환경
Google Colab CPU 런타임으로 충분 (GPU 불필요). 약 5-10분.

## 변화 추적

| Ch | 모델 | 데이터 | Activation | Loss | 라벨 |
|---|---|---|---|---|---|
| 5 | LogReg(multinomial) | Yelp 5클래스 | softmax | `CrossEntropyLoss` | int (0-4) |
| **6** | OneVsRest LogReg | Yelp + 측면 합성 | **per-label sigmoid** | **per-label `BCE`** | **multi-hot** |

전체 19챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
**Phase 1 시작** — [07_bert_pipeline](../07_bert_pipeline/) — `transformers.pipeline` 으로 첫 BERT 추론, WordPiece 토크나이저 첫 만남.
