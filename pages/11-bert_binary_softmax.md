**목표**: Ch 10에서 본 *방식 A* (sigmoid + BCE, `num_labels=1`)와 짝을 이루는 **방식 B** (softmax + CE, `num_labels=2`)를 같은 BERT·같은 Yelp 이진화 데이터로 학습합니다. 마지막에 두 방식의 결과를 **직접 비교** 해서 Ch 4(sklearn)에서 식으로 봤던 *두 방식 동등성* 이 BERT에서도 그대로 성립함을 확인합니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 20분 (방식 B 학습 ~8분 + 방식 A 비교용 학습 ~8분 + 평가/시각화/비교)

## 학습 흐름

1. 🚀 **실습**: Ch 10과 같은 데이터를 BERT로 다시 학습 — 이번엔 `num_labels=2` + softmax + `CrossEntropyLoss`
2. 🔬 **해부**: 학습 후 softmax 확률 분포를 sigmoid (방식 A) 와 같은 KDE 그림으로 비교. logit 공간에서는 $z = z_1 - z_0$ 으로 변환.
3. 🛠️ **클라이맥스**: *이 노트북 안에서* 방식 A도 한 번 더 학습한 뒤 *샘플 단위로* 비교 — scatter plot과 agreement metric. 노트북이 self-contained라 Ch 10 세션이 살아 있을 필요가 없습니다.

> 📒 **사전 학습 자료**: Ch 4 (sklearn 두 방식 동등성), Ch 10 (BERT 방식 A — 방식 A의 단독 학습/시각화는 거기서 자세히). 이번 챕터는 Ch 10에 *의존하지 않습니다* — 5장 비교를 위해 같은 노트북 안에서 방식 A를 한 번 더 학습합니다.

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` |
| 9 | DistilBERT 파인튜닝 | `AutoTokenizer.from_pretrained(...)` | Yelp (별점 1-5) | `Linear(H, 1)` | 없음 | `MSELoss` |
| 10 | DistilBERT 파인튜닝 | 같음 | Yelp 이진화 | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| **11 ← 여기** | DistilBERT 파인튜닝 | 같음 | Yelp 이진화 (Ch 10과 동일) | **`Linear(H, 2)`** | **softmax** | **`CrossEntropyLoss`** |
| 12 (다음) | DistilBERT 파인튜닝 | 같음 | Yelp 5클래스 | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 10)

| 축 | Ch 10 (방식 A) | Ch 11 (방식 B) |
|---|---|---|
| `num_labels` | 1 | **2** |
| `problem_type` | `"multi_label_classification"` | **`"single_label_classification"`** ← BERT 표준 분류 |
| Activation | sigmoid | **softmax** |
| Loss | `BCEWithLogitsLoss` | **`CrossEntropyLoss`** |
| 라벨 형식 | float `[0.0]` / `[1.0]` (multi-hot 1차원) | **int `0` / `1`** (스칼라 인덱스) |
| 모델 출력 shape | `(B, 1)` | **`(B, 2)`** |
| 데이터 / 모델 본체 / hyperparams | (변동 없음) | (그대로) |

> **딱 하나의 축만 바뀝니다** — 이번 챕터에서 변하는 건 *Loss 축* (BCE → CE)이고, 그에 따라가는 부수적 변화(num_labels, activation, 라벨 형식)는 모두 *같은 축의 일관된 표현 변경* 입니다. 데이터·모델·학습 hyperparams는 Ch 10과 *완전히 동일* 하게 유지해야 마지막 비교가 의미가 있습니다.

### 왜 두 방식이 거의 같은 결과를 내야 하는가 (수식 한 줄)

방식 A의 확률: $\hat p_A = \sigma(z)$ — *1차원 logit* $z$.

방식 B의 확률: $\hat p_B = \mathrm{softmax}(z_0, z_1)[1] = \dfrac{e^{z_1}}{e^{z_0} + e^{z_1}} = \dfrac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)$.

→ **$z_A \equiv z_1 - z_0$** 으로 두면 두 방식이 수학적으로 같은 함수입니다. 학습된 가중치는 다른 경로로 수렴하지만, *최종 확률* 은 거의 같아야 합니다 (Ch 4에서 sklearn으로 봤던 그 동등성).

## Loss 노트 — `CrossEntropyLoss` (Ch 4 그대로, BERT 맥락)

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log \hat p_{i, y_i} \quad\text{where}\quad \hat p_{i,k} = \dfrac{e^{z_{i,k}}}{\sum_{j} e^{z_{i,j}}}$$

이번 챕터에서 새로운 점은 없고 — Ch 4·5에서 이미 다 익혔습니다. *BERT* 라는 컨텍스트로 옮겨 쓸 뿐입니다.

**숫자로 감 잡기 (binary, K=2)** — 정답이 클래스 1, logits를 $(z_0, z_1)$ 로 두면:

| logits $(z_0, z_1)$ | softmax → $(p_0, p_1)$ | 정답=1일 때 손실 $-\log p_1$ |
|---|---|---|
| $(0, 0)$ | $(0.5, 0.5)$ | 0.693 |
| $(0, 2)$ | $(0.119, 0.881)$ | 0.127 |
| $(0, 5)$ | $(0.007, 0.993)$ | 0.007 |
| $(0, -2)$ | $(0.881, 0.119)$ | **2.127** |

$z_1 - z_0$ 의 크기가 정답 클래스 쪽으로 클수록 손실이 작아집니다. **방식 A의 $z$ 와 정확히 같은 신호**: $\sigma(z_1 - z_0) = p_1$ 가 1에 가까우면 손실이 작음.

**`CrossEntropyLoss` 의 안정성** — PyTorch `nn.CrossEntropyLoss` 는 내부적으로 *log-softmax + NLL* 로 구현되어 있어 `BCEWithLogitsLoss` 와 동일하게 **logit에서 직접 계산** (softmax를 따로 적용하지 않음). 두 loss 모두 "raw logit 받아서 안정적인 log-sum-exp 트릭으로 처리" 라는 점이 같습니다.

## 토크나이저 노트

Ch 10과 동일한 `distilbert-base-uncased` WordPiece 토크나이저, 동일한 max_length=128. 토크나이저 단계는 그대로지만 **`tokenize_fn` 안에서 라벨을 `[float(b)]` 가 아닌 `int(b)` 로 둡니다** — 이게 이번 챕터의 데이터 측 유일한 변경.

## 이 장의 구성

[[SubPages]]
