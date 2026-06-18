위 동등성 덕분에 K=2에서 두 logit 중 하나는 잉여입니다. sklearn은 이 사실을 알고 **K=2 multinomial을 자동으로 binary form으로 collapse** 시킵니다 — `coef_` 를 `(2, V)` 가 아니라 `(1, V)` 로만 저장합니다. 두 방식이 그래서 사실상 같은 모델이 되어 `predict_proba`도 거의 일치하는 거였죠.

직접 두 모델의 `coef_` 모양을 확인합니다. 두 방식의 가중치 행렬과 절편 모양을 출력하고, 그 값들의 최대 차이까지 계산합니다. 모양이 (2, V)가 아니라 (1, V)로 나오는지, 가중치 차이가 0에 가까운지를 보면 sklearn이 K=2를 binary form으로 접었다는 사실을 직접 확인할 수 있습니다.

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

두 모델 모두 `coef_`가 (2, V)가 아니라 (1, V)이고 가중치 차이가 0입니다. sklearn이 K=2를 binary form으로 접어 logit 하나만 저장하기 때문이며, 진짜 2-logit 헤드는 PyTorch BERT(Ch 10/11)에서 처음 등장합니다.
