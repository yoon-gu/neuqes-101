`LogisticRegression`의 `coef_`와 `intercept_`는 Ch 2 `LinearRegression`과 동일한 선형 가중치입니다. 차이는 그 위에 sigmoid가 한 번 더 붙는다는 것뿐.

직접 재현해 봅시다 — sklearn의 `predict_proba`가 정말 sigmoid에 logit을 넣은 결과인지 확인합니다.

```python
# 모델의 logit 직접 계산: z = X · w + b
# (sparse 행렬이라 .toarray() 안 쓰고 sparse dot product 사용)
logits = X_test @ model.coef_.T + model.intercept_   # shape: (N, 1)
logits = logits.flatten()

# sigmoid 적용
proba_manual = 1 / (1 + np.exp(-logits))
proba_sklearn = y_proba[:, 1]   # P(y=1)
```

**위 코드 읽기** — `X_test @ model.coef_.T + model.intercept_`가 바로 logit $z = w^\top x + b$입니다(Ch 2 선형 결합과 동일). 여기에 `1 / (1 + np.exp(-logits))` 곧 sigmoid를 직접 씌운 `proba_manual`을, sklearn이 내부에서 계산한 `predict_proba`의 양성 확률 `proba_sklearn`과 비교할 준비를 합니다.

```python
# 둘이 같은가?
diff = np.abs(proba_manual - proba_sklearn).max()
print(f"Max diff (manual vs sklearn): {diff:.2e}")

print(f"\nManual first 5:  {proba_manual[:5].round(4)}")
print(f"sklearn first 5: {proba_sklearn[:5].round(4)}")
```

**▶ 실행 결과**

```text
Max diff (manual vs sklearn): 1.11e-16

Manual first 5:  [0.4477 0.1001 0.1591 0.6653 0.758 ]
sklearn first 5: [0.4477 0.1001 0.1591 0.6653 0.758 ]
```

**결과 해석**

최대 차이가 `1.11e-16`, 즉 부동소수점 오차 수준이라 사실상 완전히 같습니다. `predict_proba`가 마법이 아니라 "선형 결합 → sigmoid"라는 두 줄로 재현된다는 것이 확인됩니다.

```python
# BCE(log loss)도 직접 계산 가능
# 정답이 1이면 -log(p), 0이면 -log(1-p)
y_test_arr = y_test.values
p = proba_sklearn
manual_bce = -(y_test_arr * np.log(p) + (1 - y_test_arr) * np.log(1 - p)).mean()
sklearn_bce = log_loss(y_test, y_proba)

print(f"Manual BCE:  {manual_bce:.6f}")
print(f"sklearn BCE: {sklearn_bce:.6f}")
print(f"Diff:        {abs(manual_bce - sklearn_bce):.2e}")
```

**▶ 실행 결과**

```text
Manual BCE:  0.383569
sklearn BCE: 0.383569
Diff:        0.00e+00
```

**결과 해석**

식 그대로 손으로 더한 BCE와 `log_loss`가 소수점 6자리까지 정확히 일치합니다(차이 0). 위 수식의 "정답이 1이면 $-\log p$, 0이면 $-\log(1-p)$"가 sklearn의 log loss와 같은 것임을 코드로 확인한 셈입니다.

확률 대신 0/1 예측을 클래스별로 뜯어봅니다. `classification_report`로 precision·recall·F1을, `confusion_matrix`로 어떤 방향으로 틀렸는지(FP/FN)를 함께 확인합니다.

```python
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

cm = confusion_matrix(y_test, y_pred)
print(f"\nconfusion matrix:\n{cm}")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}\n  FN={cm[1,0]}, TP={cm[1,1]}")
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

    negative       0.87      0.86      0.86       409
    positive       0.86      0.87      0.86       399

    accuracy                           0.86       808
   macro avg       0.86      0.86      0.86       808
weighted avg       0.86      0.86      0.86       808


confusion matrix:
[[352  57]
 [ 53 346]]
  TN=352, FP=57
  FN=53, TP=346
```

**결과 해석**

두 클래스의 precision·recall·F1이 모두 0.86 안팎으로 고르게 나옵니다. 혼동 행렬을 보면 오탐(FP=57)과 미탐(FN=53)도 비슷한 규모라, 모델이 한쪽 클래스에 치우치지 않고 균형 있게 맞히고 있습니다.
