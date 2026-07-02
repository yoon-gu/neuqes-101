**목표**: 한 샘플에 *여러* 라벨이 동시에 붙는 multi-label 문제로 확장합니다. softmax의 클래스 *상호배타* 가정이 깨지고, K개 sigmoid가 라벨마다 독립적으로 작동합니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5-10분

## 학습 흐름

1. 🚀 **실습**: Yelp 리뷰에 항목(aspect) 키워드를 매칭해 5개 라벨(food/service/price/ambiance/location) multi-hot 합성 → `OneVsRestClassifier`로 학습
2. 📐 **Loss 분해**: 학습된 모델의 실제 예측으로 BCE 5개를 직접 합산해 본다 — multinomial CE를 못 쓰는 이유를 숫자로
3. 🔬 **해부**: multi-label 평가 지표 — subset accuracy, hamming loss, micro/macro F1
4. 🛠️ **변형**: 임계값(threshold)을 옮기면 micro/macro F1이 어떻게 움직이나
5. ⚠️ **합성의 한계** — 키워드 매칭으로 만든 라벨이 실제 라벨링과 어떻게 다른지 솔직히 짚기

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` |
| 5 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| **6 ← 여기** | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 항목 키워드 합성 | (5차원) | **sigmoid (각각 독립)** | **`BCEWithLogitsLoss` per-label** |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 5)

가장 큰 변화는 **데이터 가정** 이고, 그게 모델 선택을 강제합니다.

| 축 | Ch 5 (multi-class) | Ch 6 (multi-label) |
|---|---|---|
| 데이터 가정 | 클래스 *상호배타* (한 샘플 = 한 라벨) | **라벨 *독립* (한 샘플에 여러 라벨 가능)** |
| 라벨 형식 | int 한 개 (0-4) | **multi-hot 벡터** (예: `[1, 0, 1, 0, 1]`) |
| 모델 패러다임 | **multinomial 기본** + OvR 대안 (Ch 5 후반에서 두 방식 비교) | **OvR 만** (multinomial은 데이터 가정과 충돌) |
| Activation | softmax 한 번 (합 = 1 강제) | **per-label sigmoid** (라벨끼리 독립) |
| Loss | `CrossEntropyLoss` | **per-label `BCEWithLogitsLoss` 평균** |
| OvR 사용 방식 | K개 sigmoid → **argmax로 한 라벨 선택** | K개 sigmoid **그대로** (argmax 없음, 각자 임계값 0.5와 비교) |
| 데이터 | 별점 5클래스 | **Yelp + 항목 키워드 합성** (5개 항목) |
| 토크나이저 | TF-IDF | TF-IDF (그대로) |

### 왜 OvR이 multi-label의 자연스러운 선택인가

Ch 5에선 두 패러다임이 모두 가능했습니다.

- **multinomial**: softmax 한 번으로 K개 logit을 묶어 합=1 강제. "K개 중 정확히 하나"라는 데이터 가정과 정합.
- **OvR (대안)**: K개 *독립* sigmoid + argmax 후처리. 각 binary 모델은 독립이지만 마지막에 강제로 하나만 고르므로 결과는 상호배타.

Ch 6의 multi-label은 이 가정 자체를 깹니다 — 한 리뷰가 "음식 + 서비스 + 가격"을 동시에 다룰 수 있어요. 그러면:

- **multinomial은 부적합**: softmax가 합=1을 강제하므로 'food=0.9, service=0.85 동시 활성' 같은 분포를 *표현할 수가 없습니다*. P(food)=0.9면 나머지 합이 0.1로 강제돼 동시 활성이 수학적으로 불가능.
- **OvR은 자연스럽게 들어맞음**: K개 sigmoid가 *각자* 0/1을 결정 → 어떤 조합이든 표현 가능. argmax 후처리 단계만 빼면 그대로 multi-label.

요약: Ch 5에서 *대안* 이었던 OvR이 Ch 6에서는 *유일한 자연스러운 선택* 이 됩니다. 알고리즘(`OneVsRestClassifier`)은 그대로, 사용 방식만 "argmax로 한 라벨 선택" → "K개 출력 그대로" 로 바뀝니다.

## Loss 함수의 변화 — `BCEWithLogitsLoss` per-label

K개 라벨에 대해 BCE를 각각 계산하고 평균을 냅니다.

$$L = \frac{1}{N \cdot K}\sum_{i=1}^{N}\sum_{k=1}^{K}\bigl[-y_{ik}\log \hat p_{ik} - (1 - y_{ik})\log(1 - \hat p_{ik})\bigr]$$

각 (샘플, 라벨) 쌍이 독립적으로 손실에 기여합니다 — 한 라벨에서 틀렸다고 다른 라벨의 손실이 변하지 않습니다. CE의 클래스 경쟁 구조와 정반대.

**숫자로 감 잡기** (K=5, 정답 multi-hot $\mathbf{y} = [1, 0, 1, 0, 1]$):

| 시나리오 | 예측 확률 $\hat{\mathbf{p}}$ | 라벨별 손실 | 평균 BCE |
|---|---|---|---|
| 잘 맞춤 | `[0.9, 0.1, 0.8, 0.2, 0.6]` | 0.105 / 0.105 / 0.223 / 0.223 / 0.511 | **0.233** |
| 균등 (baseline) | `[0.5, 0.5, 0.5, 0.5, 0.5]` | 0.693 × 5 | **0.693** |
| 정반대로 자신감 | `[0.1, 0.9, 0.1, 0.9, 0.1]` | 2.303 × 5 | **2.303** |

baseline = $\log 2 = 0.693$ — 모든 라벨에 0.5를 줄 때 (BCE에서 K=2 분포의 균등 추측과 같은 값). 학습된 모델은 이 값보다 작아야 정상.

```python
# PyTorch (Ch 13 이후, multi-label)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets.float())   # logits: (N, K), targets: (N, K) multi-hot

# sklearn (이번 챕터)
from sklearn.multiclass import OneVsRestClassifier
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X, Y_multilabel)   # Y_multilabel shape: (N, K) 0/1
```

## 토크나이저 노트

이번 챕터의 토크나이저는 **Ch 1-5와 동일한 `TfidfVectorizer`**. 입력 표현은 그대로고, 변화는 라벨 구조와 출력 헤드의 형태에 있습니다.

> **다음 챕터(Ch 7)** — Phase 1 시작: 사전학습된 **DistilBERT WordPiece** 가 처음 등장합니다. TF-IDF의 단어 단위 어휘 학습과 어떻게 다른지 비교 시작.

## 이 장의 구성

[[SubPages]]
