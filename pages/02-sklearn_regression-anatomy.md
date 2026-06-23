위 분포를 보면 모델이 0.4점이나 5.7점 같은 **별점 범위 밖** 의 값도 뱉습니다. 이상해 보이지만 자연스러운 결과입니다.

`LinearRegression`이 학습한 것은 단지 "MSE를 최소화하는 가중합"이지, "출력값이 1과 5 사이여야 한다"는 제약을 듣지 않습니다. 모델은 활성화 함수 없이 $w^\top x + b$를 그대로 뱉을 뿐이라 음수도 5 초과도 모두 가능한 결과입니다.

이게 **회귀의 본질** 입니다. 출력 범위 제약은 모델이 아니라 사람이 따로 입혀야 합니다 — clipping 같은 후처리, 혹은 sigmoid 같은 활성화 함수로요.

`mean_squared_error`가 마법이 아니라 단순히 "오차 제곱의 평균"임을 직접 손으로 계산해 확인합니다. 수식 $\frac{1}{N}\sum(y-\hat y)^2$ 을 그대로 코드로 옮긴 값이 sklearn 결과와 일치하는지 봅니다.

```python
# sklearn의 mean_squared_error가 내부에서 뭘 계산하는지 직접 재현
manual_mse = ((y_test - y_pred_test) ** 2).mean()
sklearn_mse = mean_squared_error(y_test, y_pred_test)

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

두 값이 소수점 여섯 자리까지 똑같고 차이가 정확히 0입니다. MSE는 특별한 블랙박스가 아니라 "오차를 제곱해 평균낸 것"일 뿐임이 확인됩니다.
