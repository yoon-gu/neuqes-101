# 09_bert_regression — 첫 BERT 파인튜닝, 첫 `Trainer`

[![Open In Colab — 본 노트북](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/09_bert_regression.ipynb)
[![Open In Colab — 부록: 학습 실험 관리](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/appendix_experiment_tracking.ipynb)
[![Open In Colab — 부록: HPO](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/appendix_hpo.ipynb)

## 한 줄 목표
Phase 0의 별점 회귀(Ch 2)를 *DistilBERT 파인튜닝* 으로 다시 풉니다. sklearn `LinearRegression` 이 1초 만에 풀던 문제를 BERT는 GPU에서 수 분간 학습합니다. `Trainer` 가 처음 등장하고, 학습 과정 전체를 명시적으로 통제하기 시작합니다.

## 다루는 핵심 개념
- `AutoModelForSequenceClassification(num_labels=1, problem_type="regression")` — 분류 헤드를 회귀 헤드로
- **`Trainer` + `TrainingArguments` 첫 등장** — Phase 1·2 모든 학습 챕터의 골격
- `compute_metrics` 직접 정의 (sklearn 헬퍼 그대로 활용)
- `fp16=True` — T4 GPU에서 메모리·속도 효율 (bf16 미지원)
- 학습 중·후 nvidia-smi VRAM 추적 — 옵티마이저·gradient·activation의 메모리 비중
- sklearn(Ch 2) vs BERT 결과 직접 비교 — 같은 MSE, 다른 최소화 방식
- 학습 하이퍼파라미터(lr, batch_size, epochs, max_length)가 어디서 망가지는지 안전대 정리

## Loss
**`MSELoss`** — Ch 2와 식은 동일하지만 최소화 방식이 정규방정식에서 Adam SGD로 바뀝니다.

## 데이터
`yelp_review_full` — 학습 4,000건 + 평가 1,000건 (T4 30분 안에 학습 끝나도록 작게). 라벨은 별점 1-5 float.

## 환경
Google Colab **T4 GPU 필수**. CPU에선 학습이 한 시간 이상 걸립니다. 약 10-15분 (모델 다운로드 + 2 에폭 학습 + 평가).

## 변화 추적

| Ch | 모델 | 데이터 | Loss | 학습 |
|---|---|---|---|---|
| 2 | `LinearRegression()` | Yelp (별점 1-5) | `MSELoss` | 정규방정식 1초 |
| 8 | (모델 로드 없음) | Yelp 토크나이저 옵션 | — | — |
| **9** | **DistilBERT 파인튜닝** | Yelp (별점 1-5) | **`MSELoss`** | **Adam, T4 GPU 5-8분** |

전체 19챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 부록 노트북

📒 **[appendix_experiment_tracking.ipynb](./appendix_experiment_tracking.ipynb)** — `report_to` 인자로 학습 metric을 외부 트래커에 보내는 패턴. **wandb · trackio · MLflow** 세 도구를 직접 돌려보고 비교.

📒 **[appendix_hpo.ipynb](./appendix_hpo.ipynb)** — 하이퍼파라미터 최적화의 어려움. `TrainingArguments` 의 핵심 인자 정리, HPO가 어려운 5가지 이유, `Trainer.hyperparameter_search` + Optuna 직접 시도, wandb sweeps · MLflow autolog 통합, 실용 가이드.

## 다음 챕터
[10_bert_binary](../10_bert_binary/) — sigmoid+BCE vs softmax+CE 두 방식을 BERT로 다시 비교 (Ch 4 sklearn 동등성의 BERT 버전). 같은 `Trainer` 골격으로 두 번 학습.
