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

## 합성의 한계 — 솔직한 한계 짚기

이 챕터의 학습 결과가 너무 좋아 보일 수 있습니다 (subset accuracy, micro F1 모두 매우 높음). 그 이유는 **모델이 *키워드 매칭 규칙* 자체를 학습** 하기 때문입니다 — 우리가 정한 사전을 다시 거꾸로 풀어내고 있을 뿐, 진짜 항목 추출 능력을 입증한 게 아닙니다.

실제 multi-label 문제에서 부딪히는 것들:

1. **부정·반어 무시** — `"this place is not noisy at all"` 의 'noisy'를 ambiance 활성으로 잡는 게 우리 사전의 한계. 사람이 읽으면 ambiance가 *아닌* 데도.
2. **사전 협소** — 'food'에 'sushi', 'ramen', 'pasta' 같은 구체 음식명이 빠져 있으면 그 리뷰는 food=0이 되어 버림.
3. **정답이 노이지** — 우리 라벨 자체가 진짜 정답이 아닌 휴리스틱이라, 모델 성능을 이 정답에 비교하는 건 결국 "모델이 휴리스틱을 얼마나 따라 했나"를 잴 뿐.
4. **빈 라벨**: 모든 항목이 0인 샘플도 있음 (`{n_labels_per_sample == 0).sum()` 건). 실제 multi-label 데이터에선 보통 최소 한 라벨은 보장.

**그럼 왜 합성을 쓰나?** — 학습 코드의 *형태* 와 *평가 지표 해석* 을 익히는 게 이 챕터의 목적이기 때문입니다. Ch 13 BERT multi-label에서 **같은 합성 라벨** 을 그대로 사용하므로 비교가 깔끔하게 됩니다. 진짜 multi-label 데이터(예: GoEmotions, Reuters)는 라벨이 사람 손으로 만들어져 있어 비용이 큽니다.
