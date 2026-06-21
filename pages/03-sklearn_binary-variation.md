`predict()`는 기본적으로 $\hat p \geq 0.5$를 기준으로 0/1을 자릅니다. 이 임계값을 다른 값으로 옮기면 precision과 recall이 정반대로 움직입니다.

- 임계값 ↑ (예: 0.7): "확실해야만 positive" → **precision 상승**, recall 하락
- 임계값 ↓ (예: 0.3): "조금만 의심돼도 positive" → **recall 상승**, precision 하락

스팸 필터(precision 중시), 암 진단(recall 중시) 같은 도메인 요구에 따라 임계값을 조정합니다.

```python
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
proba_pos = y_proba[:, 1]

rows = []
for t in thresholds:
    y_pred_t = (proba_pos >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_t, average="binary", zero_division=0
    )
    acc = accuracy_score(y_test, y_pred_t)
    rows.append({"threshold": t, "accuracy": acc, "precision": p, "recall": r, "f1": f1})
```

**위 코드 읽기.** 모델을 다시 학습하지 않고, 고정된 `proba_pos`(positive 확률)에 임계값 `t`만 바꿔 가며 `(proba_pos >= t)`로 0/1을 새로 자릅니다. 임계값마다 precision/recall/F1/accuracy를 모아 임계값이 지표에 미치는 영향만 분리해서 봅니다.

```python
df_t = pd.DataFrame(rows).round(4)
print(df_t.to_string(index=False))
```

**위 코드 읽기.** 모은 결과를 `DataFrame`으로 만들어 임계값별 지표를 한 표에 나란히 보여 줍니다. 행을 위아래로 훑으면 임계값이 오를 때 precision과 recall이 어떻게 반대로 움직이는지 한눈에 드러납니다.

**▶ 실행 결과**

```text
 threshold  accuracy  precision  recall     f1
       0.3    0.7686     0.6840  0.9875 0.8082
       0.4    0.8527     0.7929  0.9499 0.8643
       0.5    0.8639     0.8586  0.8672 0.8628
       0.6    0.8366     0.9159  0.7368 0.8167
       0.7    0.7772     0.9700  0.5664 0.7152
```

**결과 해석**

임계값을 0.3 → 0.7로 올리면 precision은 0.6840 → 0.9700으로 오르고 recall은 0.9875 → 0.5664로 떨어져, 예고한 trade-off가 그대로 나타납니다. 정확도와 F1은 0.5 근처에서 가장 높아, 균형 데이터에서는 기본 임계값 0.5가 합리적인 출발점임을 보여 줍니다.
