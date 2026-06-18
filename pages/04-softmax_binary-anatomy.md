방식 B는 *개념적으로* logit을 두 개 ($z_0, z_1$) 학습합니다. softmax의 두 번째 성분을 풀어보면:

$$\text{softmax}([z_0, z_1])_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \frac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)$$

즉 두 logit에서 **의미 있는 정보는 $z_1 - z_0$ 뿐** 입니다 — 두 logit에 같은 상수를 더해도 softmax 결과가 안 바뀌니까요 ($e^{z+c}/\sum e^{z+c} = e^{z}/\sum e^{z}$). softmax+2는 sigmoid+1의 **리파라미터화** 일 뿐입니다.

CE 쪽도 K=2에서 BCE와 같은 식이 됩니다 (one-hot이라 $y_1 = y$, $y_0 = 1-y$ 대입):

$$\text{CE} = -[y_1 \log \hat p_1 + y_0 \log \hat p_0] = -[y \log \hat p_1 + (1-y)\log(1 - \hat p_1)] = \text{BCE}$$

확률·loss가 같으니 학습된 결정 경계도, gradient도 같습니다.

먼저 식이 정말 일치하는지 임의의 logit 쌍으로 직접 확인합니다. logit 쌍 4개를 만들어 softmax의 두 번째 성분과 sigmoid(z1-z0)를 각각 계산해 표로 나란히 출력합니다. 두 열의 값이 자리까지 같은지, 마지막 줄의 최대 차이가 부동소수점 한계 수준으로 작은지 눈여겨보세요.

```python
# 임의의 logit 쌍 4개를 만들어 softmax([z0,z1])_1 == sigmoid(z1 - z0) 인지 확인
z0_arr = np.array([-2.0, 0.0, 1.5, 3.0])
z1_arr = np.array([ 1.0, 0.5, -0.5, 2.0])

softmax_p1  = np.exp(z1_arr) / (np.exp(z0_arr) + np.exp(z1_arr))
sigmoid_diff = 1.0 / (1.0 + np.exp(-(z1_arr - z0_arr)))

print(f"{'z_0':>6} {'z_1':>6}    {'softmax([z0,z1])_1':>22}    {'sigmoid(z1-z0)':>16}")
print("-" * 60)
for i in range(len(z0_arr)):
    print(f"{z0_arr[i]:>6.1f} {z1_arr[i]:>6.1f}    {softmax_p1[i]:>22.8f}    {sigmoid_diff[i]:>16.8f}")

print(f"\nMax diff: {np.abs(softmax_p1 - sigmoid_diff).max():.2e}  (numerical noise)")
```

**▶ 실행 결과**

```text
   z_0    z_1        softmax([z0,z1])_1      sigmoid(z1-z0)
------------------------------------------------------------
  -2.0    1.0                0.95257413          0.95257413
   0.0    0.5                0.62245933          0.62245933
   1.5   -0.5                0.11920292          0.11920292
   3.0    2.0                0.26894142          0.26894142

Max diff: 2.22e-16  (numerical noise)
```

**결과 해석**

네 쌍 모두 softmax의 두 번째 성분과 sigmoid(z1-z0)가 8자리까지 일치하고, 차이는 부동소수점 한계인 2e-16뿐입니다. 식으로 보인 동등성이 숫자로도 그대로 성립합니다.
