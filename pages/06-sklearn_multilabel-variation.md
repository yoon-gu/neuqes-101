multi-label에서는 K개 sigmoid 출력에 대해 임계값 0.5로 자르는 게 기본입니다. 이 임계값을 옮기면 모든 라벨이 함께 반응합니다 — 임계값을 낮추면 더 많은 라벨이 활성되어 recall은 오르고 precision은 내려갑니다.

(고급 트릭: 라벨마다 *별도* 임계값을 정해 검증 F1을 최대화할 수도 있음. 여기서는 하나로 통일.)

```python
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
rows = []
for t in thresholds:
    Y_pred_t = (proba_ml >= t).astype(int)
    rows.append({
        "threshold": t,
        "subset_acc": accuracy_score(Y_test, Y_pred_t),
        "hamming": hamming_loss(Y_test, Y_pred_t),
        "micro_F1": f1_score(Y_test, Y_pred_t, average="micro", zero_division=0),
        "macro_F1": f1_score(Y_test, Y_pred_t, average="macro", zero_division=0),
    })
df_t = pd.DataFrame(rows).round(4)
print(df_t.to_string(index=False))
```

**▶ 실행 결과**

```text
 threshold  subset_acc  hamming  micro_F1  macro_F1
       0.2       0.230   0.2706    0.7170    0.6978
       0.3       0.468   0.1478    0.8127    0.7889
       0.4       0.561   0.1146    0.8320    0.7657
       0.5       0.493   0.1372    0.7749    0.6467
       0.6       0.415   0.1658    0.6989    0.5172
       0.7       0.342   0.2036    0.5928    0.3877
```

**결과 해석**

기본값 0.5가 최적이 아닙니다 — 임계값을 0.4로 낮추면 micro F1(0.8320)·subset accuracy(0.561)가 최대가 되고, 0.3에서는 macro F1(0.7889)이 가장 높습니다. 앞 해부에서 본 "드문 라벨의 낮은 recall"을 임계값을 내려 더 많은 라벨을 활성화함으로써 보완한 결과이고, 더 내려 0.2까지 가면 거짓 양성이 늘어 다시 나빠집니다.
