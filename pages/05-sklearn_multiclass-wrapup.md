## 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `sklearn.multiclass.OneVsRestClassifier` | K개 독립 binary 분류기를 묶어 estimators_ 로 노출 | Ch 6 multi-label의 핵심 도구로 그대로 재등장 |
| `sklearn.metrics.confusion_matrix` | 다중 클래스도 K×K로 일반화 | 분류 챕터마다 사용 |
| `sklearn.metrics.classification_report` | per-class precision/recall/F1 한 번에 | — |

## 체크포인트 질문

1. K=5 균등 분포일 때 CE 손실값은 얼마인가요? 학습된 모델은 이보다 작아야 하는 이유는?
2. 5×5 confusion matrix에서 오답이 대각선 근처에 몰리는 현상은 모델의 어떤 가정과 데이터의 어떤 성질 사이에서 나오나요?
3. multinomial과 OvR의 핵심 차이를 한 문장으로 요약해보세요.
4. 같은 별점 데이터를 회귀(Ch 2)로 풀 때와 5클래스 분류(Ch 5)로 풀 때 어떤 종류의 실수에 더 큰 페널티를 주나요?

## FAQ

### Q1. (이론) multinomial과 OvR은 어떻게 다른가요?

핵심 차이는 **클래스 간 의존성**.

- **multinomial (softmax)**: 한 모델이 K개 logit을 동시에 학습. softmax → 확률 합 = 1, 클래스 *상호배타*.
- **OvR (One-vs-Rest)**: K개 *독립* binary 분류기 (`OneVsRestClassifier`로 명시적으로 구성). 클래스 0의 logit은 다른 클래스 학습에 영향 없음. 정규화 전 확률 합이 1이 아닐 수 있음.

| 상황 | 적합한 방식 |
|---|---|
| 한 샘플에 정확히 한 라벨 (별점, 뉴스 카테고리) | multinomial |
| 한 샘플에 여러 라벨 가능 (영화 장르: 로맨스+코미디) | OvR (Ch 6) |
| 클래스 수백 개 + 빠른 학습 필요 | OvR (binary들이 병렬 학습 쉬움) |

### Q2. (실무) 클래스 수가 100개를 넘어가면 학습이 느려지는데 어떻게 하나요?

세 가지 흔한 처리법.

1. **Hierarchical classification**: 큰 그룹 → 세부 분류. 예: 의류 → 상의/하의 → 셔츠/티셔츠.
2. **희귀 클래스 묶기**: 빈도 1-2회짜리는 "기타"로. long tail 80%는 어차피 못 배움.
3. **계산 트릭**: hierarchical softmax, sampled softmax (학습 시 일부 클래스만 negative). PyTorch 모델에서 자주, sklearn 기본엔 없음.

K=5 정도는 위 트릭이 필요 없는 작은 규모.

### Q3. (실무) 클래스 불균형이 심한데 `class_weight`를 어떻게 적용하나요?

```python
model = LogisticRegression(
    class_weight="balanced",   # 빈도의 역수로 자동 가중치
    max_iter=1000,
)
```

`balanced`는 빈도 적은 클래스에 더 큰 가중치를 줍니다. 또는 `class_weight={0: 0.5, 1: 1.0, ...}` 으로 직접 지정 가능. 효과가 미미하면 데이터 레벨 처리(SMOTE 등) 또는 임계값 후처리.

### Q4. (실무) confusion matrix를 시각화하는 추천 방법은?

`seaborn.heatmap`이 깔끔합니다.

```python
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"{i+1}★" for i in range(5)],
            yticklabels=[f"{i+1}★" for i in range(5)])
plt.xlabel("Predicted"); plt.ylabel("True")
```

**행 정규화**(recall 관점)도 자주 봅니다.

```python
cm_norm = cm / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
```

### Q5. (이론) macro F1과 weighted F1 중 어떤 걸 봐야 하나요?

- **macro F1**: 클래스별 F1을 단순 평균. 클래스 불균형이 있으면 소수 클래스 성능을 가리지 않음.
- **weighted F1**: 클래스 빈도로 가중 평균. 다수 클래스 성능을 더 반영.

**언제 무엇:**
- 모든 클래스가 똑같이 중요 → macro F1.
- 다수 클래스 위주로 평가하고 싶음 → weighted F1.
- 보고에는 보통 둘 다 함께 적습니다.

### Q6. (이론) baseline = $\log K$의 의미는 무엇이고 왜 챙겨봐야 하나요?

CE 식에 균등 분포 $\hat p_k = 1/K$ 를 대입하면:

$$L = -\log(1/K) = \log K$$

**의미**: 모델이 *아무것도 안 학습한 상태* 의 손실. 학습 손실이 baseline보다 크면 "정답이 *아닌* 곳에 자신 있다"는 뜻 — 모델이 잘못된 방향으로 가고 있습니다.

K=5에서 baseline ≈ 1.609. 학습 도중 loss가 이 값 근처에서 정체되면 "모델이 데이터에서 신호를 못 잡고 있다"는 진단.

## 삽질 코너 (선택)

같은 5클래스 데이터에서 회귀 모델(Ch 2 방식)과 분류 모델(이번 챕터)이 4★를 3★로 예측하는 비율과 4★를 1★로 예측하는 비율의 비를 각각 계산해보세요. 회귀가 더 큰 실수를 더 강하게 회피하는지 확인할 수 있습니다.

```python
from sklearn.linear_model import LinearRegression

# 회귀 모델 (Ch 2 그대로)
model_reg = LinearRegression()
model_reg.fit(X_train, y_train.astype(float))
pred_reg = np.clip(np.round(model_reg.predict(X_test)), 0, 4).astype(int)

# 분류 모델 (이번 챕터)
pred_clf = y_pred

for name, pred in [("regression+round/clip", pred_reg), ("classification", pred_clf)]:
    mask4 = (y_test == 3)   # 라벨 3 = 4★
    n_4to3 = ((pred == 2) & mask4).sum()  # 4★ → 3★ 오답
    n_4to1 = ((pred == 0) & mask4).sum()  # 4★ → 1★ 오답
    print(f"{name}: 4-star->3-star {n_4to3}, 4-star->1-star {n_4to1}")
```

힌트: 회귀는 큰 실수일수록 손실이 제곱으로 커지므로 4★을 1★로 예측하는 큰 실수가 더 드물어야 합니다. 분류는 둘을 동등하게 처벌하므로 그런 경향이 약할 수 있습니다.

## 다음 챕터 예고

**Chapter 6. sklearn Multi-label — softmax 합=1 제약을 푼다**

- 한 샘플에 *여러* 라벨이 동시에 붙는 multi-label 문제로 확장
- 새 데이터: Yelp 리뷰 + **항목(aspect) 키워드 합성** (food/service/price/ambiance/location 5개)
- softmax 한 개 대신 **5개 독립 sigmoid** — 각 라벨이 다른 라벨에 영향받지 않음
- Loss는 CrossEntropyLoss에서 **per-label `BCEWithLogitsLoss`** 로
- `OneVsRestClassifier(LogisticRegression())` + `MultiLabelBinarizer`로 구현
