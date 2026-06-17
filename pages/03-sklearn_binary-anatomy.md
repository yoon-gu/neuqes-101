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

# 둘이 같은가?
diff = np.abs(proba_manual - proba_sklearn).max()
print(f"Max diff (manual vs sklearn): {diff:.2e}")

print(f"\nManual first 5:  {proba_manual[:5].round(4)}")
print(f"sklearn first 5: {proba_sklearn[:5].round(4)}")
```

**▶ 실행 결과**

```text
Max diff (manual vs sklearn): 0.00e+00

Manual first 5:  [0.4485 0.0999 0.1638 0.6661 0.7578]
sklearn first 5: [0.4485 0.0999 0.1638 0.6661 0.7578]
```

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
Manual BCE:  0.383475
sklearn BCE: 0.383475
Diff:        0.00e+00
```

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

precision과 recall이 두 클래스 모두 0.86 언저리로 고르고, 오분류도 FP 57건과 FN 53건으로 거의 대칭입니다. 한쪽으로 치우쳐 찍는 모델이 아니라 positive·negative를 비슷한 신뢰도로 가른다는 뜻입니다.
