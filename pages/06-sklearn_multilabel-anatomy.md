multi-class에서 자주 쓰던 accuracy는 multi-label에서 의미가 미묘하게 다릅니다.

- **subset accuracy** (`accuracy_score`): "K개 라벨이 *전부* 일치한 샘플 비율" — 라벨 하나만 틀려도 0점. 매우 엄격.
- **hamming loss**: 평균 라벨별 오답 비율. 5개 중 1개 틀리면 0.2 기여. 가장 직관적.
- **micro F1**: 모든 라벨의 TP/FP/FN를 한 풀에 모아 계산 — 빈도 큰 라벨이 영향력 큼.
- **macro F1**: 라벨별 F1을 단순 평균 — 모든 라벨 동등 가중.

```python
print(f"Subset accuracy (all match): {accuracy_score(Y_test, Y_pred):.4f}")
print(f"Hamming loss (mean per-label error): {hamming_loss(Y_test, Y_pred):.4f}")
print(f"micro F1: {f1_score(Y_test, Y_pred, average='micro', zero_division=0):.4f}")
print(f"macro F1: {f1_score(Y_test, Y_pred, average='macro', zero_division=0):.4f}")
```

**▶ 실행 결과**

```text
Subset accuracy (all match): 0.4930
Hamming loss (mean per-label error): 0.1372
micro F1: 0.7749
macro F1: 0.6467
```

**결과 해석**

subset accuracy 0.4930은 "5개 라벨이 전부 일치한 샘플"만 세는 가장 엄격한 지표라 가장 낮게 나옵니다. hamming loss 0.1372는 5개 라벨 중 평균 0.69개가 틀렸다는 뜻이고, micro F1(0.7749)이 macro F1(0.6467)보다 높은 건 빈도 큰 food·service가 점수를 끌어올리기 때문입니다.

전체 점수만으로는 어느 라벨이 발목을 잡는지 알 수 없으니, 항목별 precision/recall/F1을 한 번에 출력합니다.

```python
# 항목별 precision/recall/F1
print(classification_report(
    Y_test, Y_pred,
    target_names=ASPECTS,
    zero_division=0,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

        food       0.90      0.90      0.90       564
     service       0.90      0.86      0.88       519
       price       0.93      0.43      0.59       276
    ambiance       1.00      0.27      0.43       185
    location       0.94      0.29      0.44       203

   micro avg       0.91      0.68      0.77      1747
   macro avg       0.93      0.55      0.65      1747
weighted avg       0.92      0.68      0.74      1747
 samples avg       0.69      0.58      0.61      1747
```

**결과 해석**

모든 라벨에서 precision은 높지만(0.90 이상) recall은 빈도 낮은 라벨로 갈수록 무너집니다 — ambiance·location은 recall 0.27/0.29로 절반 넘게 놓칩니다. 모델이 확신할 때만 1로 찍고 애매하면 0으로 두는 보수적 경향이라, 임계값 0.5가 드문 라벨에는 너무 높다는 신호입니다.
