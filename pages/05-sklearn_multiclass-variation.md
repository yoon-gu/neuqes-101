**multinomial** (`LogisticRegression()` 의 모던 sklearn 기본 동작) 은 한 모델이 K개 logit을 **동시에** 학습합니다. softmax 한 번이라 합 = 1이 강제 — "K개 클래스 중 정확히 하나"라는 *상호배타* 가정.

또 다른 방식 **OvR (One-vs-Rest)** 은 K개의 *독립* binary 분류기. 각 분류기는 "이 클래스 vs 나머지 모든 클래스"만 학습합니다.

### 두 방식의 구조 비교

**multinomial (softmax)**:

```
[입력 x] ──→ Linear(V → K) ──→ logits [z_1, ..., z_K]
                            ──→ softmax 한 번
                            ──→ [p_1, ..., p_K]   (합 = 1, 클래스끼리 경쟁)
```

**OvR** (이 챕터의 K=5 예시):

```
             ┌──→ 분류기 1:  "1★ vs 나머지"  ──→ sigmoid ──→ P_1
             ├──→ 분류기 2:  "2★ vs 나머지"  ──→ sigmoid ──→ P_2
[입력 x] ──→ ├──→ 분류기 3:  "3★ vs 나머지"  ──→ sigmoid ──→ P_3
             ├──→ 분류기 4:  "4★ vs 나머지"  ──→ sigmoid ──→ P_4
             └──→ 분류기 5:  "5★ vs 나머지"  ──→ sigmoid ──→ P_5

   각 P_k 는 다른 P_j 와 무관한 독립 sigmoid 출력 (raw 합 ≠ 1)
   예측: argmax(P_k) — sklearn이 표시할 땐 행을 정규화해 합 1로 보여줌
```

핵심 차이는 **클래스가 서로 경쟁하느냐** 입니다. multinomial은 한 logit이 커지면 다른 logit의 softmax 확률이 자동으로 줄어듭니다 (분모 공유). OvR의 각 sigmoid는 다른 클래스 학습과 독립적이라 P_k 가 모두 동시에 0.8이 될 수도, 모두 0.1이 될 수도 있습니다.

```python
# OvR은 sklearn.multiclass.OneVsRestClassifier로 만듭니다.
# 내부적으로 K개 binary LogisticRegression이 따로 학습되어 model_ovr.estimators_ 에 들어갑니다.
model_ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_ovr.fit(X_train, y_train)

print(f"OvR estimators count: {len(model_ovr.estimators_)}")
print(f"Each estimator is a separate LogisticRegression for 'class k vs rest'")
print(f"  estimator 0 coef_ shape: {model_ovr.estimators_[0].coef_.shape}  (1, V)")
```

**위 코드 읽기** `OneVsRestClassifier(LogisticRegression(...))` 는 K=5개의 별도 binary LogReg 를 만들어 `model_ovr.estimators_` 리스트에 담습니다. 각 estimator 의 `coef_` 모양이 `(1, V)` 인 것에서 보듯, 하나하나가 "클래스 k vs 나머지"만 푸는 독립 분류기입니다.

```python
# 5개 binary 모델의 coef를 (5, V)로 쌓아 한 번에 행렬 곱
ovr_coef = np.vstack([est.coef_[0] for est in model_ovr.estimators_])         # (5, V)
ovr_intercept = np.array([est.intercept_[0] for est in model_ovr.estimators_]) # (5,)

ovr_logits_all = np.asarray(X_test @ ovr_coef.T) + ovr_intercept
ovr_sigmoid_all = 1.0 / (1.0 + np.exp(-ovr_logits_all))    # (N, 5) 독립 sigmoid
```

**위 코드 읽기** 5개 binary 모델의 계수를 `(5, V)` 로 쌓아 한 번에 행렬곱해 logit 을 구한 뒤, softmax 가 아니라 **각 클래스마다 따로 sigmoid** 를 적용합니다. 이 `ovr_sigmoid_all` 이 OvR 의 본래 출력으로, 클래스끼리 분모를 공유하지 않아 행 합이 1이 될 이유가 없습니다.

```python
# 한 test 샘플을 골라 두 방식의 K=5 출력을 나란히 비교
sample_idx = 0
sample_text = X_text_test.iloc[sample_idx]
true_label = y_test.iloc[sample_idx]

p_multi = proba_5[sample_idx]                                  # multinomial softmax (합 = 1)
p_ovr_raw = ovr_sigmoid_all[sample_idx]                        # OvR 5개 독립 sigmoid (정규화 전)
p_ovr_norm = model_ovr.predict_proba(X_test)[sample_idx]       # OvR 정규화 후 (sklearn 표시용)

print("\nReview preview (200 chars):")
print(f"{sample_text[:200]}...")
print(f"True star:    {true_label + 1}★\n")

print(f"{'class':>8}  {'multinomial':>14}  {'OvR raw':>10}  {'OvR normalized':>16}")
print("-" * 56)
for k in range(5):
    print(f"  {k+1}★    {p_multi[k]:>14.4f}  {p_ovr_raw[k]:>10.4f}  {p_ovr_norm[k]:>16.4f}")
print("-" * 56)
print(f"  sum    {p_multi.sum():>14.4f}  {p_ovr_raw.sum():>10.4f}  {p_ovr_norm.sum():>16.4f}")
```

**위 코드 읽기** 한 샘플을 골라 세 열을 나란히 둡니다 — multinomial softmax(`p_multi`), OvR 의 raw sigmoid(`p_ovr_raw`), 그리고 sklearn 이 표시용으로 행 합을 1로 맞춘 `p_ovr_norm`. `p_ovr_raw` 의 합만 1에서 벗어난다는 점이 두 방식의 본질적 차이를 한눈에 보여줍니다.

**▶ 실행 결과**

```text
OvR estimators count: 5
Each estimator is a separate LogisticRegression for 'class k vs rest'
  estimator 0 coef_ shape: (1, 10000)  (1, V)

Review preview (200 chars):
A friend suggested this.. and we met up here with 2 other friends on Christmas day. Another friend also suggested I get wide noodles with an …(뒤 63자 생략)
True star:    2★

   class     multinomial     OvR raw    OvR normalized
--------------------------------------------------------
  1★            0.3570      0.3374            0.3625
  2★            0.3071      0.2680            0.2880
  3★            0.1634      0.1543            0.1658
  4★            0.0806      0.0827            0.0889
  5★            0.0919      0.0882            0.0948
--------------------------------------------------------
  sum            1.0000      0.9307            1.0000
```

**관찰**

- **multinomial 열**: 깨끗한 분포, 합 = 1. "이 문서는 K개 별점 중 어느 *한* 별점일 확률"을 나타냄.
- **OvR raw 열**: 5개 sigmoid가 서로 독립적으로 작동한 결과. 합이 1이 아닙니다 (보통 1보다 크거나 작음).
- **OvR 정규화 후 열**: sklearn이 raw 값을 행 합으로 나눠 합=1을 만들어준 것. 표시용일 뿐 모델의 본래 출력은 아닙니다.

**왜 이 차이가 중요한가** (Ch 6 떡밥)

- multinomial의 합=1 제약은 "이 문서의 별점은 정확히 *하나* 다"라는 데이터 가정에 잘 맞습니다.
- 그러나 한 문서가 여러 라벨을 가질 수 있는 *multi-label* 문제에서는 이 가정이 깨집니다 — 영화는 "로맨스 + 코미디"일 수 있고, 식당 리뷰는 "음식 + 서비스 + 가격"을 동시에 다룰 수 있습니다.
- multi-label은 **OvR의 사고방식을 그대로** 가져갑니다: K개 독립 sigmoid를 별도로 학습하고, **정규화하지 않습니다**. 각 라벨이 독립적으로 0/1을 결정하는 것이 그 모델의 본래 모습.
- 그래서 OvR을 multi-class에서 미리 만나두는 게 다음 챕터의 다리가 됩니다.

마지막으로 두 방식의 전체 test 정확도를 비교하고, OvR raw 확률의 행 합이 실제로 1을 벗어나는 정도를 분포로 확인합니다. 정확도는 비슷하더라도 출력의 성질(합=1 제약 여부)이 다르다는 점이 핵심입니다.

```python
# 전체 test set 정확도 비교
acc_ovr = accuracy_score(y_test, model_ovr.predict(X_test))
print(f"multinomial accuracy: {acc:.4f}")
print(f"OvR accuracy:         {acc_ovr:.4f}")
print(f"Diff: {abs(acc - acc_ovr):.4f}")

# OvR raw 확률 행 합 분포 (정규화 전)
raw_sums = ovr_sigmoid_all.sum(axis=1)
print(f"\nOvR raw row sum distribution (pre-normalization):")
print(f"  min:  {raw_sums.min():.3f}")
print(f"  max:  {raw_sums.max():.3f}")
print(f"  mean: {raw_sums.mean():.3f}")
print(f"  → 5 independent sigmoids; rows do not sum to exactly 1")
```

**▶ 실행 결과**

```text
multinomial accuracy: 0.5110
OvR accuracy:         0.5070
Diff: 0.0040

OvR raw row sum distribution (pre-normalization):
  min:  0.778
  max:  1.387
  mean: 0.982
  → 5 independent sigmoids; rows do not sum to exactly 1
```

**결과 해석**

정확도는 0.5110 대 0.5070으로 거의 같아 — multi-class 에서는 두 방식의 성능 차이가 미미합니다. 반면 OvR raw 행 합은 0.778~1.387까지 흩어지고 평균도 0.982로 1이 아닌데, 이 "정규화하지 않는" 성질이 한 샘플에 여러 라벨이 붙는 Ch 6 multi-label 로 그대로 이어집니다.
