**목표**: Ch 15 의 한국어 binary 셋업을 그대로 두고 **클래스 수만 K=2 → K=7** 로 늘립니다. 모델·토크나이저·hyperparams 가 *완전히 동일* 하고, 변하는 건 출력 헤드 차원과 데이터.

이 챕터는 Ch 12 (영어 multi-class, Yelp 5클래스) 의 한국어 버전이고, *Phase 1 → Phase 2* 의 task 일반화가 어떻게 자연스럽게 이어지는지 보여줍니다 — softmax+CE 셋업은 K 가 무엇이든 같은 코드.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 3분 (모델 다운로드 ~30s + 2 에폭 학습 약 40초 + 평가/시각화)

## 학습 흐름

1. 🚀 **실습**: KLUE-YNAT 5,000건으로 klue/bert-base 파인튜닝 → 뉴스 헤드라인 7카테고리 분류
2. 🔬 **해부**: 7×7 혼동 행렬, top-1 확률 분포, 카테고리별 precision/recall/F1
3. 🛠️ **샘플 단위 해석**: 자신있는 / 망설이는 헤드라인 직접 읽어보고 모델이 어디서 헷갈리는지 확인

> 📒 **사전 학습 자료**: Ch 12 (영어 multi-class, Yelp 5클래스), Ch 15 (한국어 binary, NSMC). 이번 챕터는 두 챕터의 *결합* — Ch 15 의 한국어 셋업 + Ch 12 의 multi-class 처리.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 12 | DistilBERT | WordPiece (영어) | Yelp 5클래스 | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |
| 15 | klue/bert-base | WordPiece (한국어) | NSMC binary | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| **16 ← 여기** | klue/bert-base | 같음 | **KLUE-YNAT (뉴스 7분류)** | **`Linear(H, 7)`** | softmax | `CrossEntropyLoss` |
| 17 (다음) | klue/bert-base | 같음 | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | sigmoid (per-label) | `BCEWithLogitsLoss` (per-label) |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 15)

| 축 | Ch 15 (한국어 binary) | Ch 16 (한국어 multi-class) |
|---|---|---|
| **Task** | 이진 분류 (K=2) | **7-클래스 분류 (K=7)** ← *유일한 변화* |
| `num_labels` | 2 | **7** |
| 데이터 | NSMC (영화 리뷰) | **KLUE-YNAT (뉴스 헤드라인)** |
| 라벨 형식 | int 0/1 | **int 0-6** |
| 평가 metric | binary precision/recall/F1 + AUC | **accuracy + macro precision/recall/F1 + multi-class AUC (OvR)** |
| 모델 / 토크나이저 / problem_type / Activation / Loss / hyperparams | (모두 동일) | (모두 동일) |

> **변경점 한 가지 원칙** — Phase 2 안에선 *task 차원* (K=2 → K=7) 만 바뀝니다. 한국어 셋업·hyperparams 는 Ch 15 와 *완전히 같음*. 새 챕터의 학습 부담은 *7클래스 평가 metric 의 해석* 에만 집중.

### 한국어 환경에서도 multi-class 일반화는 *그대로* 작동

Ch 11 → Ch 12 (영어 binary → multi-class) 에서 `num_labels` 를 2 → 5 로만 바꿔도 모든 셋업이 그대로 동작했습니다. Ch 15 → Ch 16 도 정확히 같은 패턴 — *softmax+CE 의 진짜 강점*: K 가 무엇이든 같은 식으로 일반화됩니다.

## Loss 노트 — `CrossEntropyLoss` 가 K=7 에서 보이는 모습

수식은 Ch 12 와 동일:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log \hat p_{i, y_i}$$

K=7 의 random baseline loss = $\log 7 = 1.946$. 학습 첫 step 에서 loss 가 ~1.9 정도 보이면 모델이 *균등 추측 단계* — 이후 loss 가 떨어지는 곡선이 학습이 *실제로 진행되는지* 진단 신호.

**숫자로 감 잡기 (K=7, 정답 = 클래스 5)**:

| logits | softmax → $\hat p_5$ | 손실 |
|---|---|---|
| 모두 0 (균등) | $1/7 \approx 0.143$ | **1.946** ← random |
| 정답 클래스만 +2 | $\approx 0.471$ | 0.752 |
| 정답 클래스만 +5 | $\approx 0.985$ | 0.015 |
| 다른 클래스 +5 (정답 0) | $\approx 0.005$ | **5.302** ← 자신 있게 틀림 |

**클래스 균형 영향** — KLUE-YNAT 의 클래스 분포는 *완벽 균형이 아닙니다* (스포츠/세계가 정치/IT보다 많음). class_weight 없이 학습하면 *다수 클래스* 에 편향될 가능성. 평가 단계에서 *macro F1* 을 같이 봐야 소수 클래스 정확도가 묻히지 않음.

## 토크나이저 노트

Ch 15 와 *완전히 동일* — `klue/bert-base` 한국어 WordPiece. 토크나이저는 K 변화에 무관 (라벨 처리는 모델 측 일).

> **Phase 2 안에서는 토크나이저 고정** — Ch 15·16·17·18 모두 같은 한국어 WordPiece. Phase 3 (Ch 19-23) 에서 비로소 *직접 학습한 워드레벨 토크나이저* 가 등장.

### 헤드라인 토큰화 예시

NSMC 영화 리뷰는 보통 *짧은 한 줄* (약 20 토큰), KLUE-YNAT 뉴스 헤드라인도 비슷하게 짧아 *평균 약 16 토큰 (최대 27)*. 같은 한국어지만 *문체가 다른* 두 도메인 — 도메인 적응이 어떻게 이뤄지는지 확인할 좋은 비교.

## 이 장의 구성

[[SubPages]]
