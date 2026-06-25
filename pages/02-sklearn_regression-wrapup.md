## 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `sklearn.linear_model.LinearRegression` | MSE 최소화 1차원 회귀 | sklearn 모델 라인업의 시작, BERT는 Ch 8부터 같은 역할 |
| `sklearn.model_selection.train_test_split` | 훈련/평가 분할 | Ch 3-5에서 계속 사용 |
| `sklearn.metrics.mean_squared_error` | MSE 평가 | Ch 8 BERT 회귀에서도 평가 지표로 등장 |
| `sklearn.metrics.mean_absolute_error` | MAE 평가 (참고용) | — |
| `sklearn.metrics.r2_score` | 결정계수 R² | — |

## 체크포인트 질문

1. `LinearRegression`은 왜 활성화 함수가 없나요? 회귀에서 활성화 함수가 빠지면 어떤 자유도가 생기나요?
2. MSE를 수식으로 적어보세요. 큰 오차에 큰 페널티를 주는 이유는 어느 항에서 오나요?
3. 별점을 [0, 1]로 정규화한 모델의 예측값이 여전히 그 범위를 벗어나는 이유는 무엇인가요?
4. 같은 데이터를 정규화 없이 학습한 모델과 정규화 후 학습한 모델은 (후처리로 되돌렸을 때) Test MSE가 거의 같습니다. 왜 그런가요?

## FAQ

### Q1. (이론) MSE 대신 MAE를 쓰면 어떻게 다른가요?

MSE는 오차를 제곱하지만 MAE(Mean Absolute Error)는 절댓값만 취합니다.

$$\text{MSE} = \frac{1}{N}\sum (y_i - \hat y_i)^2 \qquad \text{MAE} = \frac{1}{N}\sum |y_i - \hat y_i|$$

차이는 두 가지입니다.

1. **outlier 민감도**: MSE는 큰 오차를 제곱해 압도적으로 큰 페널티를 주므로 outlier 한 개가 모델 전체를 끌어당깁니다. MAE는 선형이라 outlier에 덜 민감합니다 (robust).
2. **gradient 형태**: MSE의 gradient는 $\hat y - y$ (오차에 비례), MAE는 $\pm 1$ (오차와 무관한 상수). 그래서 SGD 기반 학습에서 MAE는 미세 조정이 어려울 수 있고 0 근처에서 비미분(non-differentiable)이라 변형(Huber loss 등)이 자주 쓰입니다.

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred_test)
```

### Q2. (이론) 별점을 [0, 1]로 정규화하면 학습이 더 잘 되나요?

`LinearRegression`은 정규방정식(closed-form)으로 풀기 때문에 라벨 스케일이 결과에 영향을 거의 주지 않습니다. 위 셀에서 보았듯 정규화 공간에서 학습한 모델을 원래 별점 공간으로 되돌리면 정규화 없이 학습한 모델과 거의 같은 MSE가 나옵니다.

라벨 정규화가 의미 있는 경우는 따로 있습니다.

- **SGD 기반 학습 (BERT 포함)**: 라벨 스케일이 크면 gradient도 커져 학습률 조정이 까다로워집니다. [0, 1] 정규화가 안정성에 도움.
- **다른 loss와 호환**: BCE/sigmoid는 [0, 1] 라벨이 필요. 이번 챕터의 정규화는 다음 챕터를 위한 다리.
- **해석 편의**: [0, 1]은 "확률"처럼 읽혀 다른 점수와 비교할 때 편함.

### Q3. (실무) 학습이 너무 빨리 끝나는데 정상인가요?

네, 정상입니다. `LinearRegression`은 SGD가 아니라 **정규방정식(normal equation)** 으로 한 번에 답을 내는 닫힌 형태 풀이입니다.

$$w = (X^\top X)^{-1} X^\top y$$

5,000 샘플 × 10,000 feature(sparse) 정도는 1초 안에 끝납니다. BERT 파인튜닝은 같은 데이터로 5-10분 걸리는 것과 비교하면 수백 배 빠릅니다 — 모델 표현력이 다르기 때문입니다.

### Q4. (이론) `LinearRegression`은 왜 활성화 함수가 없나요?

회귀의 정의가 "임의의 실수를 예측한다"이기 때문입니다. 활성화 함수는 출력 범위에 제약을 거는 도구라(sigmoid → [0,1], tanh → [-1,1], softmax → 확률 분포), 범위 제약이 필요 없는 회귀에서는 자연스럽게 빠집니다.

오히려 헷갈리는 건 분류 쪽입니다. 분류는 출력이 "확률"이어야 하므로 sigmoid/softmax가 강제로 들어가는 거지, 활성화 함수가 더 "원본" 형태인 게 아닙니다. **회귀의 비활성 출력이 가장 단순한 형태** 라고 보면 됩니다.

### Q5. (실무) 예측값이 5보다 크거나 1보다 작게 나오는데 어떻게 하나요?

세 가지 선택지가 있습니다.

1. **그대로 둔다 (가장 흔함)**: 평가 지표(MSE, MAE)는 범위 밖 값도 그대로 받습니다. 회귀에서는 범위 이탈을 굳이 막지 않는 게 일반적.
2. **clip으로 잘라낸다**: 후처리로 [1, 5] 안으로 강제.

```python
y_pred_clipped = np.clip(y_pred_test, 1, 5)
print(f"Test MSE (clip 후): {mean_squared_error(y_test, y_pred_clipped):.4f}")
```

3. **출력에 sigmoid를 붙인다**: 다음 챕터의 방법. 활성화 함수로 모델 단계에서 [0, 1]을 강제.

선택은 목적에 따릅니다 — UI에 보여줄 점수면 clip이 깔끔하고, 다른 모델과 합쳐 학습 신호로 쓸 거면 그대로 둡니다.

### Q6. (이론) 별점은 1, 2, 3, 4, 5의 정수인데 회귀로 다루는 게 맞나요? 분류가 더 낫지 않나요?

둘 다 가능하고, 절대 정답은 없습니다.

- **회귀가 자연스러운 이유**: 별점은 **순서(ordinal)** 가 있습니다. 4점과 5점이 가깝다는 정보를 회귀는 보존합니다 (loss가 거리 기반).
- **분류가 자연스러운 이유**: 클래스 사이 간격이 균등하지 않을 수 있습니다. 1점과 2점의 차이, 4점과 5점의 차이가 같은 의미일까요? 사람마다 다릅니다. 분류는 클래스 간 거리를 가정하지 않습니다.

이 커리큘럼에서는 두 관점을 모두 보여줍니다 — Ch 2 (회귀) → Ch 5 (5클래스 분류). 같은 데이터를 두 방식으로 다룰 때 결과가 어떻게 달라지는지 비교하면 회귀와 분류의 차이가 손에 잡힙니다.

## 삽질 코너 (선택)

다음 코드를 돌려보고, 두 모델의 결과가 같은지 다른지 예측해보세요.

```python
# 모델 A: 별점 1-5 그대로 학습
model_a = LinearRegression().fit(X_train, y_train)

# 모델 B: 정답에 100을 곱해서 학습 (별점 100-500처럼)
model_b = LinearRegression().fit(X_train, y_train * 100)

pred_a = model_a.predict(X_test)
pred_b = model_b.predict(X_test) / 100   # 다시 1-5 스케일로

print(f"A vs B 예측 차이 최대: {np.abs(pred_a - pred_b).max():.2e}")
```

힌트: 닫힌 형태 풀이에서 라벨에 상수를 곱하면 가중치도 같은 비율로 바뀔 뿐, 예측을 같은 스케일로 되돌리면 정확히 같은 값이 나와야 합니다.

## 다음 챕터 예고

**Chapter 3. sklearn Binary — 출력에 sigmoid가 붙다**

- 별점을 4-5점 → 1, 1-2점 → 0으로 이진화 (3점은 제외)
- `LogisticRegression`: 출력에 **sigmoid** 가 붙고, loss는 **`BCEWithLogitsLoss`** (sklearn: log loss)로 바뀜
- 처음으로 "확률"을 출력하는 모델 — `predict_proba`로 확인
- 변경점은 **출력 형태 + Loss** 두 가지 (모델·토크나이저는 그대로, 데이터는 이진화로 가공)
