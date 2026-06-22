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

df_t = pd.DataFrame(rows).round(4)
print(df_t.to_string(index=False))
```

**▶ 실행 결과**

```text
 threshold  accuracy  precision  recall     f1
       0.3    0.7686     0.6840  0.9875 0.8082
       0.4    0.8527     0.7929  0.9499 0.8643
       0.5    0.8639     0.8586  0.8672 0.8628
       0.6    0.8366     0.9159  0.7368 0.8167
       0.7    0.7772     0.9700  0.5664 0.7152
```
