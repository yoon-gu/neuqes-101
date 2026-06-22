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
