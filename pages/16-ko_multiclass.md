**목표**: Ch 15 의 한국어 binary 셋업을 그대로 두고 **클래스 수만 K=2 → K=7** 로 늘립니다. 모델·토크나이저·hyperparams 가 *완전히 동일* 하고, 변하는 건 출력 헤드 차원과 데이터.

이 챕터는 Ch 12 (영어 multi-class, Yelp 5클래스) 의 한국어 버전이고, *Phase 1 → Phase 2* 의 task 일반화가 어떻게 자연스럽게 이어지는지 보여줍니다 — softmax+CE 셋업은 K 가 무엇이든 같은 코드.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 13분 (모델 다운로드 캐시 ~10s + 2 에폭 학습 ~10분 + 평가/시각화)


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

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

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

> **Phase 2 안에서는 토크나이저 고정** — Ch 15·16·17·18 모두 같은 한국어 WordPiece. Phase 3 (Ch 19-20) 에서 비로소 *직접 학습한 워드레벨 토크나이저* 가 등장.

### 헤드라인 토큰화 예시

NSMC 영화 리뷰는 보통 *짧은 한 줄* (~20 토큰), KLUE-YNAT 뉴스 헤드라인은 *조금 더 정형* 된 한국어 (~25-30 토큰). 같은 한국어지만 *문체가 다른* 두 도메인 — 도메인 적응이 어떻게 이뤄지는지 확인할 좋은 비교.

**baseline VRAM**:

## 데이터 — KLUE-YNAT (뉴스 헤드라인 7분류)

**KLUE** = Korean Language Understanding Evaluation 벤치마크. **YNAT** = Yonhap News Agency Topic. 연합뉴스 헤드라인 한 줄 + 7카테고리 라벨.

| 라벨 | 카테고리 |
|---|---|
| 0 | IT과학 |
| 1 | 경제 |
| 2 | 사회 |
| 3 | 생활문화 |
| 4 | 세계 |
| 5 | 스포츠 |
| 6 | 정치 |

`datasets.load_dataset("klue/klue", "ynat")` 로 정상 로드 (parquet 기반).

## 토큰화 — Ch 15 패턴 그대로

라벨 형식만 binary int → 0-6 int. 한 줄 차이.

## 모델 로드 — `num_labels=7` 만 바뀜

Ch 15 셋업에서 K=2 → K=7 한 줄 변화.

**Ch 15 와의 파라미터 수 비교** — 7클래스로 늘어났는데도 모델은 *거의 안 무거워짐*:

| 부분 | Ch 15 (K=2) | Ch 16 (K=7) |
|---|---|---|
| BERT body (12 layer) | 110,617,344 | 110,617,344 |
| classifier `Linear(768, K)` | 1,538 | **5,383** |
| 합계 | 110,618,882 | **110,622,727** |

분류 헤드만 K 에 비례해 늘어나지만 BERT body 가 ~110M 이라 K 가 5 늘어도 전체 차이는 0.003%. **K 가 늘어났다고 학습이 *훨씬* 무거워지지는 않는다** — multi-class BERT 의 매력.

## 학습 — Ch 15 와 동일한 hyperparams

`compute_metrics` 만 multi-class 용으로 (Ch 12 의 패턴 그대로).

## 평가 — softmax 확률 분포 + 혼동 패턴

Ch 12 의 평가 패턴을 한국어 환경에서 재현. 7클래스라 혼동 행렬이 7×7 — *어떤 카테고리가 어떤 카테고리와 헷갈리는지* 보는 데 핵심.

### 5-1. 혼동 행렬 — 어디서 헷갈리는가

행은 정답 카테고리, 열은 예측. 색은 *행 정규화 (recall)*, 숫자는 *원본 카운트*. 대각선이 진할수록 그 카테고리 재현율이 좋음.

**해석 가이드**

- **대각선 셀** = 그 카테고리의 재현율. 모든 셀이 0.85+ 면 잘 학습된 것.
- **오답 패턴**:
  - 정치 ↔ 경제: 둘 다 정책·법안·국제 이슈 다뤄 *경계가 모호* — 자연스러운 혼동
  - 생활문화 ↔ 사회: 사회 이슈 vs 일상·문화 보도 — 헤드라인 한 줄로는 사람도 헷갈리는 경계
  - IT과학 ↔ 경제: 기업·산업 뉴스가 양쪽에 걸침 (예: "삼성전자 4분기 실적 발표")
- **먼 클래스 혼동** (스포츠 ↔ 정치 등) 이 자주 보이면 라벨 노이즈나 학습 부족 신호.

### 5-2. Top-1 확률 분포 — 모델 자신감 진단

K=7 에선 *어느 한 클래스에 압도적 자신* 있는 경우 vs *2-3 후보 사이에서 갈등* 하는 경우가 나뉩니다. correct/wrong 으로 갈라 그려 calibration 확인.

**해석**

- 잘 학습된 모델은 *correct* 곡선이 1.0 가까이 몰림. *wrong* 은 더 낮은 영역 (0.4-0.7) 에 분산.
- correct/wrong 둘 다 1.0 근처에 압축돼 있으면 *over-confident* — 틀린 답에도 자신만만한 위험 신호. K 가 클수록 (7클래스) 이런 경향이 더 잘 드러남.
- *random baseline* 인 1/K = 0.143 근처 봉우리가 보이면 모델이 *판단 자체를 못 하는* 샘플 — 학습 데이터 부족 또는 헤드라인이 너무 짧은 경우.

### 5-3. 샘플 단위 해석 — 실제 헤드라인이 어떻게 분류되나

가장 자신있는 샘플 / 망설이는 샘플 / 자신있게 틀린 샘플 세 종류를 골라 직접 읽어 봅니다. 헤드라인 한 줄 만으로 모델이 어떤 카테고리 신호를 잡는지 감각.

**관찰 포인트**

- *가장 자신있는* 샘플은 보통 카테고리 *시그널 단어* 가 명확 (예: "주가" → 경제, "월드컵" → 스포츠).
- *망설이는 샘플* 의 top-3 분포를 보면 모델이 *어느 카테고리 사이에서 갈팡질팡* 하는지 보임. 정치/경제/사회 셋이 비슷한 확률이면 헤드라인 자체가 다중 카테고리에 걸침.
- *자신있게 틀린* 샘플은 보통 *반어*, *비유*, *카테고리 간 경계 사례* — 학습 데이터에 비슷한 패턴이 없었거나 라벨 자체가 모호. 이걸 보면 "모델이 *바보* 라서 틀린 게 아니라 *데이터가 어렵다* " 는 감각 잡힘.

## 이 장의 구성

- [16-1. 실습](16-ko_multiclass-practice.md)
- [16-2. 정리와 FAQ](16-ko_multiclass-wrapup.md)
