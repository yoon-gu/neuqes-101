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

**위 코드 읽기.** `X_test @ model.coef_.T + model.intercept_`가 바로 Ch 2와 같은 선형 결합 $z = w^\top x + b$이고, 그 결과를 `1 / (1 + np.exp(-logits))`에 통과시켜 직접 sigmoid를 계산합니다. 이 손수 계산한 확률을 sklearn의 `predict_proba`의 positive 열 `y_proba[:, 1]`과 비교할 준비를 합니다.

```python
# 둘이 같은가?
diff = np.abs(proba_manual - proba_sklearn).max()
print(f"Max diff (manual vs sklearn): {diff:.2e}")

print(f"\nManual first 5:  {proba_manual[:5].round(4)}")
print(f"sklearn first 5: {proba_sklearn[:5].round(4)}")
```

**위 코드 읽기.** 두 확률 배열의 최대 절대 차이를 구해 sklearn이 정말 "logit에 sigmoid를 씌운 값"을 내놓는지 검증합니다. 차이가 0에 가까우면 `predict_proba`가 블랙박스가 아니라 우리가 손으로 잰 식과 똑같다는 뜻입니다.

**▶ 실행 결과**

```text
Max diff (manual vs sklearn): 1.11e-16

Manual first 5:  [0.4477 0.1001 0.1591 0.6653 0.758 ]
sklearn first 5: [0.4477 0.1001 0.1591 0.6653 0.758 ]
```

**결과 해석**

최대 차이가 `1.11e-16`로 부동소수점 오차 수준이고, 앞 5개 값도 소수점 4자리까지 완전히 일치합니다. `predict_proba`가 정확히 sigmoid에 logit을 넣은 결과임이 수치로 확인됩니다.

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

수식 $-[y\log p + (1-y)\log(1-p)]$를 직접 평균 낸 값이 sklearn의 `log_loss`와 소수점 6자리까지 정확히 같습니다. BCE(sklearn: log loss)가 별도 개념이 아니라 바로 이 한 줄 수식임을 보여줍니다.

정확도 한 숫자로는 클래스별 성능 차이가 가려지므로, 클래스별 precision/recall/F1과 혼동 행렬을 함께 봅니다. 혼동 행렬은 `[[TN, FP], [FN, TP]]` 배치라 어느 방향으로 오분류가 쏠리는지까지 드러납니다.

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

두 클래스의 precision/recall/F1이 모두 0.86 부근으로 대칭적이라, 한쪽으로 치우치지 않고 고르게 맞히고 있습니다. 혼동 행렬에서도 오분류가 FP=57, FN=53으로 비슷해 특정 방향 편향이 없습니다.
