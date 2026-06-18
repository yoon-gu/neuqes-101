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

# 원래 별점 공간으로 되돌렸을 때 MSE 비교
y_pred_back = y_pred_norm * 4 + 1
print(f"Test MSE (back to star space): {mean_squared_error(y_test, y_pred_back):.4f}")
print(f"Test MSE (no normalization):    {mean_squared_error(y_test, y_pred_test):.4f}")
```

**▶ 실행 결과**

```text
Test MSE (normalized space): 0.0970
Test MSE (back to star space): 1.5522
Test MSE (no normalization):    1.5522
```

**결과 해석**

라벨을 [0, 1]로 압축한 모델도 별점 공간으로 되돌리면 MSE가 1.5522로, 정규화하지 않은 모델과 정확히 같습니다. 라벨 스케일을 바꾸는 건 단위 환산일 뿐이라 예측 품질 자체는 그대로라는 뜻이고, 출력을 범위 안에 가두려면 라벨이 아니라 모델 쪽(활성화 함수)을 손봐야 함을 확인시켜 줍니다.

정규화한 모델의 예측 범위를 출력하고, [0, 1]을 벗어난 예측이 몇 개나 되는지 아래위로 세어 봅니다. 정답 라벨을 [0, 1]로 눌렀는데도 출력은 여전히 그 밖으로 새는지 수치로 확인하려는 것입니다. 8% 안팎이 경계를 벗어난다는 점에 주목하세요.

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
Normalized model pred range: [-0.568, 1.612]
Ideal range: [0, 1]

Predictions < 0: 84 (8.4%)
Predictions > 1: 88 (8.8%)
```

**관찰**: 정답 라벨을 [0, 1]로 압축해도 모델 출력은 여전히 그 범위를 벗어납니다. 가중합을 그대로 뱉는 한 어떤 라벨 스케일링으로도 [0, 1] 안에 가둘 수 없습니다.

그렇다면 출력을 강제로 [0, 1] 사이로 누르려면 **활성화 함수** 가 필요합니다.

> **다음 챕터(Ch 3) 예고**: `LogisticRegression`이 등장합니다. 모델 구조는 거의 그대로지만 출력 직전에 **sigmoid** 가 붙어 [0, 1]을 강제하고, loss는 MSE 대신 **`BCEWithLogitsLoss`** (sklearn: log loss)로 바뀝니다. 라벨도 [0, 1] 정규화 대신 **0/1 이진** 으로 바뀝니다.
