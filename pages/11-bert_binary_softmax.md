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

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

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

**baseline VRAM**:

## 데이터 — Yelp 이진화 (Ch 10과 정확히 동일)

같은 seed, 같은 5,000/1,000 샘플, 같은 별점 3 제외 + 4-5 → 1, 1-2 → 0 룰. **마지막 비교가 의미를 가지려면 데이터가 동일해야 합니다.**

**Ch 10과의 한 줄 차이**: `out["labels"] = [[float(b)] for b in batch["binary"]]` → `out["labels"] = [int(b) for b in batch["binary"]]`. 라벨이 *길이 1짜리 float 리스트* 가 아니라 *int 스칼라* 입니다.

## 모델 로드 — 방식 B 셋업

`num_labels=2` + `problem_type="single_label_classification"` (BERT 분류의 *기본값*).

**파라미터 수 비교 — 방식 A vs 방식 B**

| 부분 | 방식 A (`num_labels=1`) | 방식 B (`num_labels=2`) |
|---|---|---|
| DistilBERT body | 66,362,880 | 66,362,880 |
| pre_classifier (`Linear(768→768)`) | 590,592 | 590,592 |
| classifier (`Linear(768→K)`) | **769** (=768+1) | **1,538** (=768·2+2) |
| 합계 | 66,955,010 | 66,955,778 |

방식 B의 분류 헤드 파라미터가 정확히 *2배* 입니다. 차이는 **769개** — 전체 67M 중 0.001%. 이 미세한 자유도 차이가 두 방식의 *최종 확률* 을 거의 같게, *학습된 가중치* 는 미묘하게 다르게 만듭니다.

## 학습 — Ch 10과 동일한 hyperparams

Ch 10과 *완전히 같은* learning rate, batch size, epoch 수, seed. **변하는 건 모델 출력 shape와 loss 종류뿐**.

## 평가 — softmax 확률 분포

Ch 10과 같은 패턴 — Ch 10에서는 sigmoid로 1차원 logit을 확률로 바꿨다면, 여기서는 *2차원 logit에 softmax* 를 적용해 클래스 1의 확률을 뽑습니다.

### 4-1. 메인 그림 — *확률 공간* 분포 (Ch 10과 같은 KDE)

Ch 10에서 봤던 것과 같은 형태의 KDE. 이번엔 확률이 *softmax 출력 1번째 원소* ($p_1$)이라는 점만 다릅니다. 그림 자체는 거의 같은 모양이어야 합니다 — 두 방식이 동등하다는 직관의 첫 번째 증거.

### 4-2. 보조 그림 — $z = z_1 - z_0$ 의 logit 공간 분포

방식 B는 logit이 2차원 $(z_0, z_1)$ 이라 단순한 logit 공간 그림이 안 그려집니다. 그래서 **방식 A와 같은 1차원 logit 좌표로 환산** ($z = z_1 - z_0$) 해서 그립니다 — 이러면 결정 경계는 $z=0$, 의미는 $\sigma(z)=p_1$ 로 방식 A와 정확히 같아집니다.

**여기까지 정리** — 4-1과 4-2의 그림은 Ch 10의 것과 *모양* 이 거의 같아야 합니다. 봉우리 높이나 위치가 미세하게 다를 순 있어도, 양 끝 압착 / 가운데 헷갈림 영역 / 결정 경계 자리 같은 *큰 그림* 은 동일. 이게 두 방식 동등성의 *시각적* 증거.

## 클라이맥스 — 방식 A 를 *이 노트북 안에서* 다시 학습해 비교

이전 챕터(Ch 10)의 결과 파일에 의존하지 않도록, 같은 데이터·같은 hyperparams·같은 seed로 방식 A를 *바로 여기서* 한 번 더 학습합니다. 변하는 것은 **모델 셋업과 라벨 형식뿐** (Ch 10에서 본 그대로):

| 셋업 | 방식 B (§3-4에서 학습) | 방식 A (지금 inline 재학습) |
|---|---|---|
| `num_labels` | 2 | **1** |
| `problem_type` | `single_label_classification` | **`multi_label_classification`** |
| 라벨 형식 | int 스칼라 (`0` / `1`) | **길이 1 multi-hot float (`[0.0]` / `[1.0]`)** |
| 학습 hyperparams | (epoch=2, lr=2e-5, seed=42 …) | **그대로** |

T4 기준 추가 ~8분. 학습이 끝나면 같은 eval 셋의 $p_A^{(i)}$ 와 §4에서 구한 $p_B^{(i)}$ 를 1,000개 점으로 비교할 수 있게 됩니다.

### 5-1. 두 방식의 metric 표 비교

같은 데이터에 같은 모델 본체로 학습했고 hyperparams도 같으니, accuracy/F1/AUC 같은 평가 지표가 *거의 같은* 값이어야 합니다. 차이가 있다면 random init과 dropout 같은 *학습 경로* 차이에서 옵니다.

### 5-2. 샘플 단위 확률 비교 — scatter plot

x축 = 방식 A의 $p_A$, y축 = 방식 B의 $p_B$. 점 색은 정답 라벨.

**완전히 동등하면 모든 점이 $y = x$ 직선 위**. 실제로는 random init·dropout·optimizer 비결정성 때문에 약간 흩어지지만, 직선에서 크게 벗어나면 안 됩니다.

**해석**

- **상관계수가 0.99 이상**, **평균 절대 차가 0.05 이하** 면 두 방식이 사실상 같은 함수를 학습했다고 봐도 됩니다.
- 만약 점들이 *체계적으로* 직선 한쪽으로 치우친다면 → 한 방식이 다른 방식보다 일관되게 더 자신 있게 / 더 보수적으로 예측하고 있다는 뜻. seed를 여러 개 시도해 평균내면 보통 사라집니다.
- 점들이 직선 *주변에 무작위로* 흩어져 있으면 → 단순 학습 경로 차이. 학습량을 늘리거나 더 큰 데이터에서 학습하면 줄어듭니다.

### 5-3. 예측 일치율 (threshold 0.5)

확률을 0/1 예측으로 떨어뜨린 뒤 두 방식의 예측이 얼마나 일치하는지 봅니다. 일치율이 95% 이상이면 *실질적으로* 같은 분류기로 봐도 됩니다.

**여기까지 보고 결론** — 식으로 본 동등성 ($\sigma(z) = \mathrm{softmax}(z_0, z_1)[1]$ when $z = z_1 - z_0$)이 BERT에서도 그대로 성립합니다. 차이가 있어 봐야 random init / dropout 같은 *학습 경로 차이* 정도. 두 방식은 **수식이 다른 같은 모델**, 라이브러리·코드 컨벤션이 강요하는 표현 차이일 뿐입니다.

> **현장 가이드**: 새 BERT 분류 task를 시작할 때는 *방식 B (softmax+CE)* 가 표준 — `num_labels=K`, `problem_type="single_label_classification"` 만 두면 끝. 방식 A는 *binary 라벨이 multi-label 형식으로 들어오는 시나리오* (예: 이진 라벨이 여러 헤드 중 하나로 끼어 있는 경우)에서만 의식적으로 사용합니다.

## 이 장의 구성

- [11-1. 실습](11-bert_binary_softmax-practice.md)
- [11-2. 정리와 FAQ](11-bert_binary_softmax-wrapup.md)
