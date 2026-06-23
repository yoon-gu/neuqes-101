## 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `LogisticRegression()` | 데이터가 multi-class 면 multinomial(softmax+CE) 자동, binary 면 sigmoid+BCE (이번 챕터엔 K=2) | Ch 5 에서 K=5 로 확장, Ch 12 BERT multi-class 에서 같은 패러다임 |
| `sklearn.metrics.log_loss` | CE/BCE 평가 함수 (multi-class 호환) | — |

## 체크포인트 질문

1. softmax 함수 정의를 적고, $\text{softmax}([z_0, z_1])_1$ 이 $\sigma(z_1 - z_0)$ 와 같음을 증명해보세요.
2. Cross Entropy를 K=2에 적용하면 정확히 BCE가 되는 과정을 식으로 보일 수 있나요? ($y_1 = y$, $y_0 = 1-y$ 대입)
3. 방식 B의 두 coefficient 벡터 사이에 어떤 관계가 학습되는 경향이 있나요? 그 이유는?
4. 같은 binary 데이터에 두 방식의 accuracy가 거의 같다면, 실무에서 어느 쪽을 택해야 하나요?

## FAQ

### Q1. (이론) K=2일 때 sigmoid+BCE와 softmax+CE가 정확히 동등하다는 걸 식으로 어떻게 보이나요?

두 가지를 따로 보여야 합니다.

**확률 동등성:**

$$\text{softmax}([z_0, z_1])_1 = \frac{e^{z_1}}{e^{z_0}+e^{z_1}} = \frac{1}{1+e^{-(z_1-z_0)}} = \sigma(z_1 - z_0)$$

방식 A의 logit을 $z = z_1 - z_0$ 로 두면 두 모델의 P(y=1)이 일치합니다.

**Loss 동등성** ($K=2$, one-hot 정답에서 $y_1 = y$, $y_0 = 1-y$):

$$\text{CE} = -\sum_{k=0}^{1} y_k \log \hat p_k = -[y \log \hat p_1 + (1-y) \log \hat p_0] = -[y \log \hat p_1 + (1-y) \log(1 - \hat p_1)] = \text{BCE}$$

확률과 loss가 모두 같으니 학습된 결정 경계도 같고, gradient도 같습니다.

### Q2. (실무) 실제로 둘 중 어느 방식이 더 널리 쓰이나요?

두 가지 관행이 공존합니다.

- **sigmoid+BCE (방식 A, num_labels=1)**: sklearn 기본, 통계학·의학 분야 표준. 출력 1개라 "확률 하나"라는 해석이 단순.
- **softmax+CE (방식 B, num_labels=2)**: BERT/PyTorch 기본, 딥러닝 표준. 다중 클래스로 일반화하기 자연스럽고 라이브러리 코드가 단순(같은 헤드/같은 loss로 K가 2이든 N이든 호환).

이 커리큘럼의 BERT 챕터(Ch 9-14)는 방식 B가 기본이라 이번 챕터에서 미리 익숙해지는 게 의미 있습니다. Ch 10·11에서 두 방식을 BERT로 별도 학습해 비교합니다.

### Q3. (이론) softmax 합=1 제약은 어디서 오나요?

수식 자체가 정규화를 강제합니다.

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

분모가 모든 클래스 $e^{z_j}$의 합이라 분자들이 그 합을 정확히 분할 — 합이 1.

**왜 이런 구조?** 모델 출력을 "확률 분포"로 해석하고 싶어서입니다. 확률 분포는 정의상 $\sum_k p_k = 1$ 이고 각 항이 [0, 1]. softmax는 임의 실수 logit 벡터를 그런 분포로 보내는 가장 자연스러운 변환 중 하나(지수 함수 = 단조증가 + 양수 보장 → 정규화).

### Q4. (실무) sklearn binary 에서 `coef_.shape` 가 왜 `(1, V)` 인가요? — 방식 A·B 가 왜 같은 결과로 수렴하는가

위 본문에서 확인했듯 K=2 softmax 는 두 logit 중 하나가 잉여(redundant)입니다 — $z_1 - z_0$ 만 의미가 있어요. sklearn 은 이걸 알고 **K=2 multinomial 을 자동으로 binary form 으로 collapse** 시킵니다. 그래서 `coef_` 가 `(2, V)` 가 아니라 `(1, V)`, `intercept_` 도 `(1,)` 로 저장됩니다.

```python
LogisticRegression().fit(X, y_binary).coef_.shape  # (1, V)
LogisticRegression().fit(X, y_3class).coef_.shape  # (3, V) — K≥3 에선 (K, V)
```

그래서 방식 A와 방식 B가 sklearn 안에서는 사실상 같은 모델이고, predict_proba 도 미세한 수치 오차 빼고 일치합니다. 진짜 *두 별개의 logit head* 가 살아 있는 형태는 프레임워크가 collapse 하지 않는 환경 — PyTorch 에서 `nn.Linear(H, 2)` 를 직접 만들 때 — 비로소 등장합니다 (Ch 10·11).

### Q5. (이론) sklearn 에서 softmax 와 OvR 을 어떻게 구분하나요?

`LogisticRegression()` 의 동작은 **학습 데이터의 클래스 개수** 가 결정합니다.

| 데이터 | 자동 학습 형태 | logit head |
|---|---|---|
| binary (K=2) | sigmoid + BCE | `coef_` shape `(1, V)` |
| multi-class (K≥3) | softmax + CE (multinomial) | `coef_` shape `(K, V)` |

**K개 독립 binary** (multi-label 또는 명시적 OvR) 가 필요하면 별도 wrapper:

```python
from sklearn.multiclass import OneVsRestClassifier
OneVsRestClassifier(LogisticRegression()).fit(X, Y)   # Y 가 1D 면 OvR multi-class, 2D 면 multi-label
```

이번 챕터의 방식 A·B 둘 다 `LogisticRegression()` 한 줄로 학습합니다 — binary 데이터라 sklearn 이 같은 collapse 형태로 만들어 둘이 *수치적으로 동일* (FAQ Q4). 두 방식이 *진짜로 분리된* 학습이 되려면 PyTorch nn.Linear 를 직접 짜야 합니다 (Ch 10·11).

### Q6. (실무) Hugging Face `Trainer`도 두 방식이 가능한가요?

가능하고, `AutoModelForSequenceClassification` 인자 한 줄 차이입니다.

```python
# 방식 A: num_labels=1, BCE 자동 적용
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1,
    problem_type="single_label_classification",  # 또는 자동
)

# 방식 B: num_labels=2, CE 자동 적용 (BERT 표준)
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)
```

이 커리큘럼의 Ch 10·11이 두 방식을 BERT로 *별도 학습* 해 비교하는 챕터입니다. 이번 챕터에서 익힌 동등성이 그 비교의 출발점이 됩니다.

## 삽질 코너 (선택)

다음 코드는 sklearn 없이 numpy만으로 softmax의 **shift invariance** 를 보여줍니다 — 모든 logit에 같은 상수를 더해도 출력 분포가 변하지 않는다는 성질입니다. 이게 K=2일 때 sklearn이 collapse를 하는 *정확한 이유* 입니다.

```python
z = np.array([1.0, 2.0])

p_original  = np.exp(z)         / np.sum(np.exp(z))
p_shift5    = np.exp(z + 5)     / np.sum(np.exp(z + 5))
p_zero_z0   = np.exp(z - z[0])  / np.sum(np.exp(z - z[0]))   # z_0 = 0 으로 정규화한 형태

print(f"softmax([1, 2]):    {p_original.round(6)}")
print(f"softmax([6, 7]):    {p_shift5.round(6)}")
print(f"softmax([0, 1]):    {p_zero_z0.round(6)}")
```

힌트: 세 결과가 모두 같습니다. 두 logit 중 *하나는 자유롭게 정할 수 있고* 의미 있는 정보는 차이 $z_1 - z_0$ 뿐 — 만약 $z_0 = 0$ 으로 고정하면 $z_1$ 만 학습하면 됩니다. 이게 sklearn이 K=2 multinomial을 binary form으로 collapse하는 형식적 근거이고, K=2 softmax가 sigmoid의 리파라미터화에 불과한 이유입니다.

## 다음 챕터 예고

**Chapter 5. sklearn Multi-class — K=5로 진짜 일반화**

- 같은 multinomial LogReg를 별점 1-5(5클래스)로 그대로 확장
- 수식·코드 변화 거의 없음 — softmax/CE는 K가 무엇이든 같은 형태
- 5×5 confusion matrix가 대각선 근처에 몰리는 ordinal 흔적 관찰
- multinomial vs OvR 비교 (Ch 6 multi-label로 가는 다리)
