회귀의 출력이 어차피 임의 범위를 갖는다면, 정답 라벨을 일부러 [0, 1]로 정규화하면 어떻게 될까요? 별점 1-5를 4로 나눠 [0, 1] 사이로 압축합니다.

$$y' = \frac{y - 1}{4} \in [0, 1]$$

이게 의미 있는 이유는 **다음 챕터의 다리** 가 되기 때문입니다. Ch 3에서 sigmoid를 출력에 붙여 강제로 [0, 1]로 누를 텐데, 그때 정답 라벨도 [0, 1] 형식이어야 호환됩니다.

```python
y_train_norm = (y_train - 1) / 4   # 1-5 → 0-1
y_test_norm = (y_test - 1) / 4

model_norm = LinearRegression()
model_norm.fit(X_train, y_train_norm)

y_pred_norm = model_norm.predict(X_test)

# 정규화 공간에서의 MSE
print(f"Test MSE (normalized space): {mean_squared_error(y_test_norm, y_pred_norm):.4f}")
```

**위 코드 읽기** 라벨을 `(y - 1) / 4` 로 [0, 1] 공간으로 옮긴 뒤 같은 `LinearRegression` 을 학습합니다. 정규화 공간에서 잰 MSE 0.0973은 별점 공간 값과 직접 비교할 수 없는데, 라벨 스케일이 1/4로 줄면 오차 제곱은 그 제곱인 1/16 배가 되기 때문입니다($1.5565 / 16 \approx 0.097$).

```python
# 원래 별점 공간으로 되돌렸을 때 MSE 비교
y_pred_back = y_pred_norm * 4 + 1
print(f"Test MSE (back to star space): {mean_squared_error(y_test, y_pred_back):.4f}")
print(f"Test MSE (no normalization):    {mean_squared_error(y_test, y_pred_test):.4f}")
```

**위 코드 읽기** 예측을 `* 4 + 1` 로 별점 공간에 되돌려 정규화 없이 학습한 모델과 같은 척도로 비교합니다. 두 값이 같다면 라벨 정규화가 닫힌 형태 풀이의 결과를 바꾸지 않았다는 뜻입니다.

**▶ 실행 결과**

```text
Test MSE (normalized space): 0.0973
Test MSE (back to star space): 1.5565
Test MSE (no normalization):    1.5565
```

**결과 해석**

별점 공간으로 되돌린 MSE가 정규화 없이 학습한 1.5565와 정확히 같습니다. `LinearRegression` 은 정규방정식으로 풀기에 라벨 스케일링이 최종 예측에 영향을 주지 않음을 보여줍니다.

라벨을 [0, 1]로 압축했으니 예측도 [0, 1]에 머무를 것 같지만, 정말 그런지 확인합니다. 범위를 벗어난 예측이 몇 개나 되는지 세어, 라벨 스케일링만으로는 출력을 가둘 수 없음을 수치로 드러냅니다.

```python
# 정규화한 모델도 여전히 [0, 1]을 벗어나는 값을 뱉는가?
print(f"Normalized model pred range: [{y_pred_norm.min():.3f}, {y_pred_norm.max():.3f}]")
print(f"Ideal range: [0, 1]")

n_below = (y_pred_norm < 0).sum()
n_above = (y_pred_norm > 1).sum()
print(f"\nPredictions < 0: {n_below} ({n_below / len(y_pred_norm):.1%})")
print(f"Predictions > 1: {n_above} ({n_above / len(y_pred_norm):.1%})")
```

**▶ 실행 결과**

```text
Normalized model pred range: [-0.638, 1.538]
Ideal range: [0, 1]

Predictions < 0: 85 (8.5%)
Predictions > 1: 92 (9.2%)
```

**관찰**: 정답 라벨을 [0, 1]로 압축해도 모델 출력은 여전히 그 범위를 벗어납니다. 가중합을 그대로 뱉는 한 어떤 라벨 스케일링으로도 [0, 1] 안에 가둘 수 없습니다.

그렇다면 출력을 강제로 [0, 1] 사이로 누르려면 **활성화 함수** 가 필요합니다.

> **다음 챕터(Ch 3) 예고**: `LogisticRegression`이 등장합니다. 모델 구조는 거의 그대로지만 출력 직전에 **sigmoid** 가 붙어 [0, 1]을 강제하고, loss는 MSE 대신 **`BCEWithLogitsLoss`** (sklearn: log loss)로 바뀝니다. 라벨도 [0, 1] 정규화 대신 **0/1 이진** 으로 바뀝니다.
