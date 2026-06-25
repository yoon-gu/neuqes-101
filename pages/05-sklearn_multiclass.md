**목표**: Ch 4에서 본 softmax+CE를 K=2에서 K=5로 그대로 확장합니다. 모델·loss·코드 변화는 거의 없습니다 — 데이터의 클래스 수만 늘어납니다. 회귀(Ch 2)와 5클래스 분류가 같은 데이터를 어떻게 다르게 해석하는지도 확인합니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분

## 학습 흐름

1. 🚀 **실습**: 별점 1-5를 5개 독립 클래스로 보고 multinomial LogReg로 분류
2. 🔬 **해부**: 5×5 confusion matrix가 대각선 근처에 몰리는 ordinal 흔적 관찰
3. 🛠️ **변형**: 모던 `LogisticRegression()` (multinomial 자동) vs `OneVsRestClassifier(LogisticRegression())` (OvR) 비교 — Ch 6 multi-label 로 가는 다리

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` |
| **5 ← 여기** | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 5클래스 (별점 0-4) | **(5차원)** | softmax | `CrossEntropyLoss` |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 4)

| 축 | Ch 4 | Ch 5 |
|---|---|---|
| 데이터 | Yelp 이진화 (K=2) | **Yelp 5클래스 (K=5)** |
| Output Head | (2차원) | **(5차원)** |
| 활성화·Loss | softmax + CE | softmax + CE (그대로) |
| 모델·토크나이저 | LogReg(multinomial) + TF-IDF | 그대로 |

**한 가지 변화** — K=2가 K=5로 늘어난 것. softmax/CE는 K가 무엇이든 같은 식이라 코드 변화도 거의 없습니다. 이게 softmax/CE의 진짜 가치 — sigmoid+BCE는 K=2 전용이지만 softmax+CE는 자연스럽게 다중 클래스로 확장됩니다.

## Loss 노트 — 같은 CE, K=5 수치 예시

Loss는 Ch 4와 동일한 `CrossEntropyLoss`. K가 늘어나도 식은 그대로:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i,\, y_i}$$

**숫자로 감 잡기** (K=5, 정답 클래스 = 2 인 한 샘플):

| 예측 분포 $[\hat p_0, \hat p_1, \hat p_2, \hat p_3, \hat p_4]$ | 정답 확률 $\hat p_2$ | 손실 $-\log \hat p_2$ |
|---|---|---|
| **정답에 집중**: `[0.05, 0.05, 0.80, 0.05, 0.05]` | 0.80 | 0.223 |
| **균등(uniform)**: `[0.20, 0.20, 0.20, 0.20, 0.20]` | 0.20 | 1.609 |
| **틀린 클래스에 집중**: `[0.05, 0.05, 0.05, 0.05, 0.80]` | 0.05 | **2.996** |

**baseline = $\log K = \log 5 \approx 1.609$**: 모델이 아무 정보 없이 균등 추측만 할 때의 손실. 학습된 모델은 이보다 작아야 하고, baseline을 초과하면 "정답이 *아닌* 곳에 자신 있다"는 신호 — gradient가 모델을 강하게 끌어당깁니다.

## 토크나이저 노트

**Ch 1-4와 동일한 `TfidfVectorizer`**. 입력 표현, 모델, loss 모두 그대로 — 변하는 건 데이터의 클래스 수와 출력 차원뿐.

> **다음 챕터(Ch 6)**: 같은 TF-IDF. 변화는 활성화 함수가 softmax(합=1)에서 **K개 독립 sigmoid**(라벨 간 독립)로 일반화되어 multi-label로 확장.

## 이 장의 구성

[[SubPages]]
