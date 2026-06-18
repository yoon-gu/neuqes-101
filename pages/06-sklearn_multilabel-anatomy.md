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
Hamming loss (mean per-label error): 0.1382
micro F1: 0.7737
macro F1: 0.6470
```

**결과 해석**

지표를 어떻게 세느냐에 따라 점수가 크게 달라집니다. 라벨 5개가 전부 맞아야 하는 subset accuracy는 49%로 엄격하지만, 라벨별 오답률인 hamming loss는 0.14(평균 86% 정답)로 너그럽습니다. 빈도 큰 라벨에 가중되는 micro F1(0.77)이 라벨을 동등 가중하는 macro F1(0.65)보다 높은 건 드문 라벨의 성능이 낮기 때문입니다.

전체 지표만으로는 어떤 라벨이 약한지 보이지 않으므로, 항목별로 쪼개 봅니다. `classification_report`로 5개 항목 각각의 precision·recall·F1과 support(정답 개수)를 한 표에 출력합니다. 빈도가 낮은 라벨에서 recall이 어떻게 달라지는지 눈여겨보세요.

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
     service       0.89      0.86      0.88       519
       price       0.93      0.43      0.59       276
    ambiance       1.00      0.28      0.43       185
    location       0.94      0.29      0.44       203

   micro avg       0.90      0.68      0.77      1747
   macro avg       0.93      0.55      0.65      1747
weighted avg       0.92      0.68      0.74      1747
 samples avg       0.68      0.58      0.61      1747
```

**결과 해석**

food·service는 precision·recall이 모두 0.9 안팎이지만, 드문 라벨(price·ambiance·location)은 precision은 높아도 recall이 0.3 내외로 무너집니다. 키워드가 적게 등장하는 항목일수록 모델이 확실할 때만 켜서 놓치는 게 많다는 뜻입니다.
