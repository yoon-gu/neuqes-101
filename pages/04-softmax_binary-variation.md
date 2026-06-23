위 동등성 덕분에 K=2에서 두 logit 중 하나는 잉여입니다. sklearn은 이 사실을 알고 **K=2 multinomial을 자동으로 binary form으로 collapse** 시킵니다 — `coef_` 를 `(2, V)` 가 아니라 `(1, V)` 로만 저장합니다. 두 방식이 그래서 사실상 같은 모델이 되어 `predict_proba`도 거의 일치하는 거였죠.

직접 두 모델의 `coef_` 모양을 확인합니다.

```python
print(f"Method A coef_ shape:      {model_a.coef_.shape}")
print(f"Method B coef_ shape:      {model_b.coef_.shape}")
print(f"Method A intercept_ shape: {model_a.intercept_.shape}")
print(f"Method B intercept_ shape: {model_b.intercept_.shape}")
print()
print("→ both (1, V) — sklearn collapses K=2 multinomial to binary form")
print()
print(f"coef_ max diff:      {np.abs(model_a.coef_ - model_b.coef_).max():.2e}")
print(f"intercept_ max diff: {np.abs(model_a.intercept_ - model_b.intercept_).max():.2e}")
print()
print("(small difference is only solver convergence noise; same model essentially)")
print()
print("True (2, V) two-logit head appears in PyTorch (Ch 10/11 BERT binary).")
```

**▶ 실행 결과**

```text
Method A coef_ shape:      (1, 10000)
Method B coef_ shape:      (1, 10000)
Method A intercept_ shape: (1,)
Method B intercept_ shape: (1,)

→ both (1, V) — sklearn collapses K=2 multinomial to binary form

coef_ max diff:      0.00e+00
intercept_ max diff: 0.00e+00

(small difference is only solver convergence noise; same model essentially)

True (2, V) two-logit head appears in PyTorch (Ch 10/11 BERT binary).
```

**결과 해석**

두 모델 모두 `coef_` 가 `(1, 10000)`, `intercept_` 가 `(1,)` 입니다 — 방식 B를 의도했어도 sklearn이 K=2를 binary form으로 collapse해 1개 logit만 저장했습니다. 두 계수의 최대 차이가 `0.00e+00` 이라 사실상 동일한 모델입니다. 진짜 `(2, V)` 두 logit 헤드는 프레임워크가 collapse하지 않는 PyTorch(Ch 10·11)에서야 등장합니다.
