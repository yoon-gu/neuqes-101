## 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `sklearn.linear_model.LogisticRegression` | sigmoid + BCE 내장 이진/다중 분류 | Ch 5에서 multi-class로 확장 |
| `sklearn.metrics.classification_report` | accuracy/precision/recall/F1 한 번에 | 분류 챕터마다 계속 사용 |
| `sklearn.metrics.confusion_matrix` | 혼동 행렬 | 다중 분류에서도 활용 |
| `sklearn.metrics.log_loss` | BCE 평가 | Ch 9 이후 BERT binary에서도 |
| `sklearn.metrics.precision_recall_fscore_support` | 임계값별 지표 추적 | — |

## 체크포인트 질문

1. `LogisticRegression`이 내부에서 하는 두 단계(logit 계산, sigmoid)를 식으로 적어보세요.
2. BCE에서 정답이 $y = 1$이면 어느 항이 살아남고, 그 항이 예측 확률에 따라 어떻게 변하나요?
3. `predict_proba`의 출력 shape가 $(N, 2)$인 이유는? 두 열의 합은 항상 얼마여야 하나요?
4. 임계값을 0.5에서 0.7로 올리면 precision과 recall은 어떤 방향으로 움직이고, 그 이유는 무엇인가요?

## FAQ

### Q1. (이론) 왜 binary classification에서 MSE를 쓰면 안 되나요?

작동은 하지만 두 가지 문제가 있습니다.

1. **sigmoid + MSE는 비볼록(non-convex)**: 손실면이 매끈한 그릇 모양이 아니라 평탄한 구간이 생깁니다. local minima에 빠질 수 있고 학습이 불안정합니다. 반면 sigmoid + BCE는 볼록(convex)이라 전역 최적해로 수렴이 보장됩니다.

2. **gradient 소실(saturation)**: sigmoid 양 극단(출력이 0이나 1 근처)에서 도함수가 거의 0이 됩니다. MSE의 gradient는 그 도함수에 곱해지므로 함께 0으로 죽습니다. BCE는 sigmoid와 결합되었을 때 gradient가 단순히 $\hat p - y$로 떨어져 saturation이 사라집니다.

참고로 통계학에선 0/1 라벨을 그대로 `LinearRegression`으로 푸는 **Linear Probability Model(LPM)** 이 있긴 합니다. 단순 분석엔 쓰지만 출력이 [0, 1]을 안 지키고 BCE보다 학습 안정성이 떨어져 ML에선 거의 안 씁니다.

### Q2. (이론) sigmoid 대신 다른 활성화 함수(tanh, softmax)를 쓰면 어떻게 되나요?

- **tanh**: 출력 범위가 $[-1, 1]$입니다. 라벨을 -1/+1로 매핑하고 hinge loss 등을 쓰면 SVM과 비슷한 모델이 됩니다. 수학적으로는 $\tanh(z) = 2\sigma(2z) - 1$이라 sigmoid의 단순 변환이지만 라벨 컨벤션이 다릅니다.
- **softmax**: 다중 클래스용 일반화. binary에 softmax를 쓰려면 출력 차원을 2로 늘리고 라벨도 one-hot으로 바꿔야 합니다 → 이게 정확히 Ch 11에서 다룰 "방식 B"입니다 (방식 A=sigmoid 1차원, 방식 B=softmax 2차원이 수학적으로 동등).
- **ReLU/identity**: 출력이 [0, 1] 보장이 안 됩니다 — 음수도 1 초과도 가능 → BCE의 $\log$가 정의되지 않습니다.

### Q3. (실무) 클래스 불균형이 있으면 어떻게 하나요?

세 가지 접근이 있습니다.

1. **`class_weight='balanced'`**: sklearn에 한 줄로 적용. 소수 클래스의 손실에 더 큰 가중치.

```python
model = LogisticRegression(class_weight="balanced", max_iter=1000)
```

2. **임계값 조정**: 확률 분포가 한쪽으로 쏠려 있어 0.5 임계값이 의미 없을 때, 위 셀의 threshold sweep으로 F1이 최대인 점을 찾습니다.

3. **데이터 레벨 처리**: SMOTE(소수 클래스 합성), undersampling(다수 클래스 줄이기). `imbalanced-learn` 라이브러리.

### Q4. (실무) 임계값을 0.5 외에 다른 값으로 바꾸려면?

`predict_proba`로 확률을 얻은 뒤 직접 자르면 됩니다.

```python
proba_pos = model.predict_proba(X_test)[:, 1]
y_pred_custom = (proba_pos >= 0.3).astype(int)
```

최적 임계값은 보통 검증 데이터에서 F1이 최대가 되는 지점, 또는 ROC 곡선의 Youden's J 통계량(`tpr - fpr`이 최대인 지점)으로 잡습니다.

```python
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(y_test, proba_pos)
best_thr = thr[(tpr - fpr).argmax()]
print(f"Youden's J 기준 최적 임계값: {best_thr:.3f}")
```

### Q5. (이론) accuracy, precision, recall, F1 중 뭘 봐야 하나요?

데이터 분포와 도메인 비용에 따라 다릅니다.

- **accuracy** (전체 맞춘 비율): 클래스가 균형 잡혔을 때만 의미 있음. 95% 음성/5% 양성 데이터에서 모두 음성이라 찍어도 95%.
- **precision** (positive 예측 중 진짜 positive 비율): **오탐 비용** 이 클 때. 스팸 필터, 광고 추천.
- **recall** (실제 positive 중 잡아낸 비율): **놓치는 비용** 이 클 때. 암 진단, 사기 탐지.
- **F1** (precision·recall의 조화 평균): 둘 다 중요할 때의 균형 지표. 어느 한 쪽이 0이면 F1도 0.

이 챕터의 Yelp 이진화는 클래스 불균형이 크지 않아(긍정 ≈ 49%, 거의 반반) accuracy도 의미 있지만, 실무에선 항상 classification_report 전체를 보는 습관이 안전합니다.

### Q6. (실무) sklearn `LogisticRegression`은 정규화(L2)가 기본인데 끄려면?

`penalty=None`(0.22 이상) 또는 `C` 값을 매우 크게 설정합니다.

```python
LogisticRegression(penalty=None, max_iter=1000)        # 정규화 없음
LogisticRegression(C=1e10, max_iter=1000)              # 거의 정규화 없음 (이전 버전 호환)
```

기본값은 `C=1.0`(L2 정규화)이고, 텍스트 분류처럼 feature가 많은 경우 정규화가 있어야 일반화 성능이 안정적입니다. 실무에선 거의 끄지 않습니다.

## 삽질 코너 (선택)

다음 코드를 돌려보고 결과를 예측해보세요. 0/1 라벨에 `LinearRegression`을 그대로 적용하면 무슨 일이 일어날까요?

```python
from sklearn.linear_model import LinearRegression

lpm = LinearRegression()
lpm.fit(X_train, y_train)
pred_lpm = lpm.predict(X_test)

print(f"LPM 예측 범위: [{pred_lpm.min():.3f}, {pred_lpm.max():.3f}]")
print(f"0 미만 예측: {(pred_lpm < 0).sum()}개")
print(f"1 초과 예측: {(pred_lpm > 1).sum()}개")

# 임계값 0.5로 잘라서 정확도 비교
acc_lpm = accuracy_score(y_test, (pred_lpm >= 0.5).astype(int))
print(f"\nLPM accuracy (threshold 0.5): {acc_lpm:.4f}")
print(f"LogReg accuracy:               {accuracy_score(y_test, y_pred):.4f}")
```

힌트: LPM은 출력이 [0, 1]을 안 지키지만, 임계값 0.5로 자르면 분류 성능 자체는 LogReg와 비슷할 수도 있습니다. 다만 "출력이 확률"이라는 해석을 잃습니다.

## 다음 챕터 예고

**Chapter 4. sklearn Softmax Binary — 같은 이진 데이터를 2차원 softmax+CE로**

- Ch 3과 **완전히 같은 이진화 데이터** 를 출력 차원 2로 늘리고 softmax + `CrossEntropyLoss` 로 다시 풉니다.
- 출력이 1차원에서 **2차원** 으로 늘어나고, sigmoid 대신 **softmax** 가 붙음 (두 열의 합 = 1 강제)
- $\sigma(z) = \text{softmax}([z_0, z_1])_1 = \sigma(z_1 - z_0)$ — 방식 A(sigmoid+BCE)와 방식 B(softmax+CE)가 **수학적으로 동등** 함을 식과 코드로 직접 확인
- 이 동등성 직관은 Ch 10·11에서 BERT binary의 두 방식을 비교할 때 그대로 재활용됩니다.
