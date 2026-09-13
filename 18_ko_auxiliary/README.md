# 18_ko_auxiliary — 한국어 BERT Auxiliary Loss (KLUE-YNAT 합성 multi-label + 활성 개수 보조)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/18_ko_auxiliary/18_ko_auxiliary.ipynb)

## 한 줄 목표
Ch 17(한국어 multi-label, KLUE-YNAT 합성)에 **활성 라벨 개수 회귀 보조 헤드** 를 추가, 결합 loss `L = L_main + λ · L_aux` 로 학습. *보조 task가 메인 task의 정확도를 끌어올리는가?* 를 같은 노트북 안에서 λ=0 baseline 과 직접 비교해 측정. Ch 14(영어 별점 보조)의 한국어 버전 — 보조 task 만 *별점* → *활성 개수* 로 달라짐.

## 다루는 핵심 개념
- 결합 loss 수식: BCE per-label (메인 카테고리) + λ · MSE (보조 활성 개수)
- 한 모델에 *두 헤드* — `AutoModel` 본체 + `cls_head` (Linear(H, 7)) + `count_head` (Linear(H, 1)) 를 `nn.Module` 로 명시 정의
- `Trainer.compute_loss` 오버라이드 — 자동 매핑이 못 다루는 복합 loss 를 직접 계산하는 패턴 (Ch 14 와 같음)
- 커스텀 `DataCollator` — `n_active` 같은 *비표준 라벨* 도 batch 에 같이 담는 패턴
- `remove_unused_columns=False` — model.forward 시그니처와 무관하게 모든 컬럼 통과
- λ 스케일 가이드 — 보조 MSE 가 메인 BCE 보다 크기 자체가 커서 λ=0.1 기본
- 보조 task 가 *메인의 함수* (합) 일 때의 한계 — 입력 의존도가 낮으면 추가 정보량 작음
- λ 스윕 선택 셀 (RUN_LAMBDA_SWEEP=False 기본) + 결과 해석 4 시나리오

## Loss
**`BCEWithLogitsLoss + λ·MSELoss`** — 자동 매핑은 쓰지 않고 모델 forward 가 두 loss 를 모두 계산해 가중합 반환. λ 기본 0.1 (보조 MSE 가 메인 BCE 보다 크기가 커서 작게 잡음).

## 데이터
Ch 17 의 KLUE-YNAT 합성 multi-label (두 헤드라인 결합, multi-hot 7차원) **+** 활성 개수 보조 라벨 `n_active` ∈ {1, 2} (합성 시 같은 카테고리면 1, 다르면 2). 5K train / 1K eval, seed 고정(42).

## 환경
Google Colab **T4 GPU 필수**. 약 3분 (보조 ON 학습 약 1분 + λ=0 baseline 학습 약 1분 + 평가/시각화).

**Self-contained**: 다른 챕터 결과에 의존하지 않습니다. 비교용 baseline (λ=0) 도 같은 노트북 안에서 inline 학습 (Ch 14 와 같은 패턴).

## 변화 추적

| Ch | 모델 | 데이터 | Output | Loss |
|---|---|---|---|---|
| 14 | DistilBERT + 보조 헤드 | Yelp + 항목 + 별점 | 메인(5) + 보조(1) | `BCE per-label + λ·MSE` |
| 16 | klue/bert-base | KLUE-YNAT 7분류 | `Linear(H, 7)` | `CrossEntropyLoss` |
| 17 | klue/bert-base | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | `BCEWithLogitsLoss` (per-label) |
| **18** | **klue/bert-base + 보조 헤드** | **합성 multi-label + 활성 개수** | **메인(7) + 보조(1)** | **`BCE per-label + λ·MSE`** |
| 19 (Phase 3 시작) | (없음) — 토크나이저 학습 | (코퍼스) | — | — |

전체 20챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[19_tokenizer_training](../19_tokenizer_training/) (Phase 3 시작) — 토크나이저를 직접 학습. Ch 1 부터 따라온 "🔤 토크나이저 노트" 의 클라이맥스.
