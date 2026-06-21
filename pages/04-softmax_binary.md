**목표**: Ch 3과 **완전히 같은 binary 데이터** 를 출력 차원 2로 늘리고 softmax + CrossEntropy로 다시 풀어봅니다. 두 방식이 수학적으로 동등하다는 것을 식과 코드로 직접 확인합니다 — 이 직관은 Ch 10·11에서 BERT binary로 옮길 때 곧장 재활용됩니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분


## 학습 흐름

1. 🚀 **실습**: 같은 Yelp 이진화 데이터에 두 방식을 학습 — sigmoid+BCE(Ch 3 그대로) vs softmax+CE(이번 챕터)
2. 🔬 **해부**: $\sigma(z) = \text{softmax}([z_0, z_1])_1 = \sigma(z_1 - z_0)$ — 동등성을 식으로 보이고 코드로 검증
3. 🛠️ **변형**: 두 모델의 coefficient 자유도 차이 — softmax+2차원에는 잉여 자유도가 있다

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| **4 ← 여기** | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 이진화 (Ch 3과 동일) | **(2차원)** | **softmax** | **`CrossEntropyLoss`** |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 3)

| 축 | Ch 3 | Ch 4 |
|---|---|---|
| Output Head | (1차원) | **(2차원)** |
| Activation | sigmoid | **softmax** |
| Loss | `BCEWithLogitsLoss` | **`CrossEntropyLoss`** |
| 라벨 | int (0/1) | int (0/1) (그대로) |
| 데이터 | Yelp 이진화 | Yelp 이진화 (그대로) |
| 토크나이저 | TF-IDF | TF-IDF (그대로) |

**왜 같은 데이터에 같은 task인데 따로 챕터로 다루나?** 두 방식은 출력 차원·activation·loss가 모두 바뀌어 보이지만 *수학적으로 동등* 합니다. 이 동등성을 가장 단순한 sklearn 환경에서 미리 체험해두면, Ch 10·11에서 BERT binary가 두 방식 중 어느 쪽을 골라도 같다는 사실을 자연스럽게 받아들일 수 있습니다.

또 K=2를 통과하면 같은 식이 K=5(다음 챕터)로 자연스럽게 일반화됩니다 — softmax/CE는 K가 무엇이든 작동합니다.

## Loss 함수의 변화 — `CrossEntropyLoss` 등장

**Cross Entropy** 는 모델 예측 분포 $\hat{\mathbf{p}}$ 와 정답 one-hot $\mathbf{y}$ 사이의 차이를 잽니다.

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log \hat p_{ik}$$

원-핫 정답이라 정답 클래스 항만 살아남습니다.

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i,\, y_i}$$

**숫자로 감 잡기** (K=2, 정답 $y = 1$인 한 샘플):

| 정답 $y$ | 예측 분포 $[\hat p_0, \hat p_1]$ | 정답 확률 $\hat p_1$ | 손실 $-\log \hat p_1$ |
|---|---|---|---|
| 1 | `[0.1, 0.9]` | 0.9 | 0.105 |
| 1 | `[0.5, 0.5]` | 0.5 | 0.693 |
| 1 | `[0.9, 0.1]` | 0.1 | **2.303** |

**Ch 3 BCE 표와 비교해보면** 손실값이 *완전히 같습니다* (0.105 / 0.693 / 2.303). K=2에서 BCE와 CE가 동등하다는 첫 단서.

```python
# PyTorch (Ch 11, 방식 B)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)   # logits: (N, K), targets: (N,) 정수 인덱스

# sklearn (이번 챕터) — 데이터의 클래스 수에 따라 자동 매핑
LogisticRegression(max_iter=1000)
```

## 토크나이저 노트

이번 챕터의 토크나이저는 **Ch 1-3과 동일한 `TfidfVectorizer`** 입니다. 모델·loss·데이터까지 Ch 3과 같고, 변하는 건 출력 차원과 활성화 함수뿐.

> **다음 챕터(Ch 5)**: 같은 TF-IDF, 같은 multinomial LogReg. 변하는 건 데이터가 binary에서 5클래스로, K가 2에서 5로.

## 이 장의 구성

- [04-1. 실습: 두 방식을 나란히 학습](04-softmax_binary-practice.md)
- [04-2. 해부: 수학적 동등성](04-softmax_binary-anatomy.md)
- [04-3. 변형: sklearn은 왜 K=2 multinomial에서 `(2, V)` coef를 안 만드나?](04-softmax_binary-variation.md)
- [04-4. 정리와 FAQ](04-softmax_binary-wrapup.md)
