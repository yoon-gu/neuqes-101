위 분포를 보면 모델이 0.4점이나 5.7점 같은 **별점 범위 밖** 의 값도 뱉습니다. 이상해 보이지만 자연스러운 결과입니다.

`LinearRegression`이 학습한 것은 단지 "MSE를 최소화하는 가중합"이지, "출력값이 1과 5 사이여야 한다"는 제약을 듣지 않습니다. 모델은 활성화 함수 없이 $w^\top x + b$를 그대로 뱉을 뿐이라 음수도 5 초과도 모두 가능한 결과입니다.

이게 **회귀의 본질** 입니다. 출력 범위 제약은 모델이 아니라 사람이 따로 입혀야 합니다 — clipping 같은 후처리, 혹은 sigmoid 같은 활성화 함수로요.

`mean_squared_error` 가 블랙박스가 아님을 보이기 위해, MSE 정의 $\frac{1}{N}\sum(y_i - \hat y_i)^2$ 를 NumPy로 직접 계산해 sklearn 결과와 맞춰봅니다.

```python
# sklearn의 mean_squared_error가 내부에서 뭘 계산하는지 직접 재현
manual_mse = ((y_test - y_pred_test) ** 2).mean()
sklearn_mse = mean_squared_error(y_test, y_pred_test)
```

**위 코드 읽기.** `(y_test - y_pred_test) ** 2` 가 샘플마다 제곱 오차를 만들고 `.mean()` 이 평균을 냅니다 — 이 두 연산이 곧 MSE 수식 그대로입니다. 라벨이 float 이라 뺄셈이 연속값으로 이뤄지고, 그 결과가 sklearn 내부 계산과 한 치도 다르지 않음을 다음 출력에서 확인합니다.

```python
print(f"Manual MSE: {manual_mse:.6f}")
print(f"sklearn MSE: {sklearn_mse:.6f}")
print(f"Diff:        {abs(manual_mse - sklearn_mse):.2e}")
```

**▶ 실행 결과**

```text
Manual MSE: 1.556471
sklearn MSE: 1.556471
Diff:        0.00e+00
```

**결과 해석**

직접 계산한 MSE와 `mean_squared_error` 의 결과가 소수점 끝까지 일치하고 차이가 정확히 0입니다. MSE는 "제곱 오차의 평균"이라는 정의 한 줄이 전부임을 보여줍니다.
