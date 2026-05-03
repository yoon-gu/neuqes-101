# 14_auxiliary_loss — BERT Auxiliary Loss (측면 + 별점 멀티태스크) — Phase 1 클라이맥스

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/14_auxiliary_loss/14_auxiliary_loss.ipynb)

## 한 줄 목표
Ch 13(BERT multi-label 측면 분류)에 **별점 회귀 보조 헤드** 를 추가, 결합 loss `L = L_main + λ · L_aux` 로 학습. *보조 task가 메인 task의 정확도를 끌어올리는가?* 를 같은 노트북 안에서 λ=0 baseline과 직접 비교해 측정.

## 다루는 핵심 개념
- 결합 loss 수식: BCE per-label (메인) + λ · MSE (보조 별점 회귀)
- 한 모델에 *두 헤드* — `model.aux_head = nn.Linear(H, 1)` 한 줄로 표준 BERT 모델에 보조 헤드 attach
- `Trainer.compute_loss` 오버라이드 — `problem_type` 만으로 매핑할 수 없는 복합 loss 를 직접 계산하는 패턴
- 커스텀 `DataCollator` — `aux_labels` 같은 *비표준 라벨* 도 batch에 같이 담는 패턴
- `remove_unused_columns=False` — model.forward 시그니처에 없는 컬럼 자동 제거 방지
- λ 선택 가이드 (0.1 - 10 grid search) + uncertainty weighting 언급
- 보조 task가 메인을 *돕는 조건* (양의 상관, 메타데이터, 단순함) vs *방해하는 조건* (반대 신호)

## Loss
**`BCEWithLogitsLoss + λ·MSELoss`** — 자동 매핑은 메인 BCE만 처리, 보조 MSE는 우리가 `compute_loss` 안에서 직접 계산해 가중합.

## 데이터
Ch 13의 측면 합성 라벨(food/service/price/ambiance/location, multi-hot 5차원) **+** 별점 보조 회귀 라벨(0-1 스케일, `label / 4`, 1★→0.0, 5★→1.0).

## 환경
Google Colab **T4 GPU 필수**. 약 22분 (보조 ON 학습 ~9분 + λ=0 baseline 학습 ~9분 + 평가/시각화).

**Self-contained**: 다른 챕터 결과에 의존하지 않습니다. 비교용 baseline (λ=0) 도 같은 노트북 안에서 inline 학습.

## 변화 추적

| Ch | 모델 | 데이터 | Output | Loss |
|---|---|---|---|---|
| 9 | DistilBERT | Yelp 별점 | `Linear(H, 1)` | `MSELoss` |
| 13 | DistilBERT | Yelp + 측면 | `Linear(H, 5)` | `BCEWithLogitsLoss` (per-label) |
| **14** | **DistilBERT + 보조 헤드** | **Yelp + 측면 + 별점** | **메인(5) + 보조(1)** | **`BCE per-label + λ·MSE`** |
| 15 (Phase 2 시작) | klue/bert-base | NSMC | `Linear(H, 2)` | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[15_ko_binary](../15_ko_binary/) — Phase 2 시작. 영어 DistilBERT → 한국어 klue/bert-base 전환, NSMC 이진 분류.
