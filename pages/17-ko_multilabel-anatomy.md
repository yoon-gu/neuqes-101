## 평가 — 카테고리별 sigmoid 확률 + 공동 활성 패턴

Ch 16 의 평가가 *7개 클래스 중 하나 고르기* 였다면, Ch 17 은 *7개 카테고리 각각을 독립적으로 0/1 판정* 합니다. Ch 13 의 multi-label 평가 패턴을 한국어 환경에서 재현.

학습된 모델을 eval set 으로 평가해 multi-label 지표를 한꺼번에 확인합니다.

```python
eval_metrics = trainer.evaluate()
print("klue/bert-base KLUE-YNAT multi-label — evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
klue/bert-base KLUE-YNAT multi-label — evaluation:
               eval_loss: 0.2166
       eval_hamming_loss: 0.0741
           eval_micro_f1: 0.8500
    eval_micro_precision: 0.8638
       eval_micro_recall: 0.8367
           eval_macro_f1: 0.8487
    eval_macro_precision: 0.8445
       eval_macro_recall: 0.8556
          eval_macro_auc: 0.9623
            eval_runtime: 0.6885
  eval_samples_per_second: 1452.5010
   eval_steps_per_second: 46.4800
```

**결과 해석**

micro F1 0.8500 과 macro F1 0.8487 이 거의 같습니다 — 카테고리별 활성률이 크게 치우치지 않아 다수·소수 카테고리 사이 격차가 작다는 뜻입니다. macro AUC 0.9623 은 임계값과 무관한 순위 분리력으로, 라벨별 sigmoid 가 양성·음성을 잘 갈라놓고 있음을 보여줍니다. hamming loss 0.0741 은 전체 라벨 위치 중 약 7%만 틀렸다는 의미입니다.

eval set 전체에 대해 예측을 뽑아 카테고리별 sigmoid 확률 범위와 실제·예측 활성률을 비교합니다. 이후 시각화·해부에서 쓸 `probs`, `preds`, `labels` 를 여기서 준비합니다.

```python
# logits → per-label sigmoid → multi-hot 예측
preds_output = trainer.predict(eval_tok)
logits = preds_output.predictions                   # (N, 7)
labels = preds_output.label_ids.astype(int)         # (N, 7) multi-hot
probs  = 1.0 / (1.0 + np.exp(-logits))              # (N, 7) per-label prob
preds  = (probs >= 0.5).astype(int)                 # (N, 7) multi-hot prediction

print(f"logits shape: {logits.shape}")
print(f"prob ranges per category:")
for k in range(K):
    print(f"  {LABEL_NAMES_EN[k]:>9}: [{probs[:, k].min():.4f}, {probs[:, k].max():.4f}]  "
          f"true rate={labels[:, k].mean():.1%}, pred rate={preds[:, k].mean():.1%}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
logits shape: (1000, 7)
prob ranges per category:
  IT/Science: [0.0113, 0.9731]  true rate=12.6%, pred rate=13.1%
    Economy: [0.0165, 0.9889]  true rate=23.8%, pred rate=22.9%
    Society: [0.0430, 0.9647]  true rate=68.4%, pred rate=58.7%
  Life&Culture: [0.0148, 0.9864]  true rate=27.8%, pred rate=29.9%
      World: [0.0109, 0.9884]  true rate=15.9%, pred rate=16.2%
     Sports: [0.0103, 0.9898]  true rate=11.7%, pred rate=12.5%
   Politics: [0.0093, 0.9884]  true rate=15.6%, pred rate=17.0%
```

**결과 해석**

대부분 카테고리에서 예측 활성률이 실제 활성률과 거의 일치합니다 — 라벨별 sigmoid 가 0.5 임계값 기준으로 잘 보정돼 있다는 신호입니다. 다만 활성률이 가장 높은 Society 는 실제 68.4% 대비 예측 58.7% 로 과소 활성하는 경향이 보이는데, 결합 헤드라인에서 사회 신호가 다른 주제와 섞여 0.5 를 넘기지 못한 경우가 그만큼 있다는 뜻입니다. 모든 카테고리에서 확률이 0.01~0.99 양극단까지 퍼져 있어 모델이 자신 있게 판정하고 있습니다.

카테고리별 precision·recall·F1 을 한 표로 봅니다. 어느 카테고리가 잘 분리되고 어느 카테고리가 헷갈리는지 진단할 수 있습니다.

```python
# Per-category classification report
print(classification_report(
    labels, preds,
    target_names=LABEL_NAMES_EN,
    digits=4, zero_division=0,
))
```

**▶ 실행 결과**

```text
              precision    recall  f1-score   support

  IT/Science     0.7634    0.7937    0.7782       126
     Economy     0.8428    0.8109    0.8266       238
     Society     0.9267    0.7953    0.8560       684
Life&Culture     0.8261    0.8885    0.8562       278
       World     0.8704    0.8868    0.8785       159
      Sports     0.8880    0.9487    0.9174       117
    Politics     0.7941    0.8654    0.8282       156

   micro avg     0.8638    0.8367    0.8500      1758
   macro avg     0.8445    0.8556    0.8487      1758
weighted avg     0.8683    0.8367    0.8501      1758
 samples avg     0.8820    0.8535    0.8485      1758
```

**결과 해석**

카테고리별로 보면 Sports 가 F1 0.9174 로 가장 깨끗하게 분리되고, IT/Science 가 0.7782 로 가장 낮습니다. 활성률이 높은 Society 는 precision 0.9267 로 매우 정확하지만 recall 0.7953 으로 놓치는 양성이 많아 — 위 prob range 에서 본 과소 활성과 일치합니다. 반대로 Life&Culture 와 Politics 는 recall 이 precision 보다 높아 약간 과활성 쪽입니다. 카테고리마다 precision·recall 균형이 다르다는 점이 카테고리별 임계값 조정의 동기가 됩니다.

### 5-1. 메인 그림 — 카테고리별 sigmoid 확률 KDE (7 패널)

Ch 16 에선 *top-1 확률 하나* 만 봤지만, multi-label 에선 *각 카테고리* 가 독립이라 7개 확률 분포를 *각각* 그립니다. 카테고리마다 학습 난이도가 *다를 수* 있다는 multi-label 의 본질이 시각적으로 드러납니다.

7개 카테고리 각각의 sigmoid 확률 분포를 정답(label=0/1) 기준으로 facet KDE 로 그립니다. 카테고리마다 두 곡선이 얼마나 깨끗이 갈라지는지가 학습 난이도를 보여줍니다.

```python
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})

# Long-form DataFrame
records = []
for k in range(K):
    name = LABEL_NAMES_EN[k]
    for i in range(len(probs)):
        records.append({"category": name, "prob": probs[i, k], "label": int(labels[i, k])})
df_long = pd.DataFrame(records)

g = sns.FacetGrid(
    df_long, col="category", col_wrap=4, height=2.8, aspect=1.3,
    sharex=True, sharey=False,
)
g.map_dataframe(
    sns.kdeplot, x="prob", hue="label",
    fill=True, common_norm=False, alpha=0.5,
    palette={0: "#5B8DEF", 1: "#F47272"}, clip=(0, 1),
)
for ax in g.axes.flat:
    ax.axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("sigmoid 확률")
g.add_legend(title="label")
g.fig.suptitle("카테고리별 sigmoid 확률 분포 (정답 기준)", y=1.03)
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/17-ko_multilabel-out1-1.png)

**해석**

- **잘 학습된 카테고리** (예: 스포츠): label=0 곡선은 0 근처, label=1 곡선은 1 근처에 있고 둘이 거의 만나지 않음. *분리가 깨끗*.
- **헷갈리는 카테고리** (예: 사회 ↔ 생활문화 ↔ 정치): 두 곡선이 0.5 근처에서 크게 겹침. 결합 헤드라인 안에서 두 주제 신호가 *섞이는* 카테고리.
- **결합의 부작용** — 한 샘플에 *두 헤드라인* 이 들어가니 모델이 "둘 중 어느 쪽 신호가 어느 라벨인지" 분리해야 합니다. 이게 단일 헤드라인 (Ch 16) 보다 어려운 점이고, multi-label task 의 자연스러운 난이도.

### 5-2. 보조 그림 — 카테고리 간 공동 활성 패턴

Multi-label 의 핵심 질문: *어떤 카테고리 쌍이 같이 등장하는가?* 합성 방식이 *무작위 결합* 이라 true co-occurrence 는 거의 균등에 가까워야 하고, 모델 예측이 그 패턴을 따라가는지 확인합니다.

`true co-occurrence` (실제 합성 라벨의 동시 등장) 와 `predicted co-occurrence` (모델 예측의 동시 등장) 를 나란히 그립니다.

```python
def cooccurrence_matrix(Y):
    # Y: (N, K) multi-hot. Returns (K, K) where M[i, j] = P(label_j=1 | label_i=1).
    Y = Y.astype(float)
    K_ = Y.shape[1]
    M = np.zeros((K_, K_))
    for i in range(K_):
        row_i = Y[:, i]
        n_i = row_i.sum()
        if n_i == 0:
            continue
        for j in range(K_):
            M[i, j] = (row_i * Y[:, j]).sum() / n_i
    return M

cooc_true = cooccurrence_matrix(labels)
cooc_pred = cooccurrence_matrix(preds)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, M, title in [
    (axes[0], cooc_true, "실제 동시출현  P(j | i)"),
    (axes[1], cooc_pred, "예측 동시출현  P(j | i)"),
]:
    sns.heatmap(
        M, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=LABEL_NAMES_EN, yticklabels=LABEL_NAMES_EN,
        cbar_kws={"label": "조건부 확률"}, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("카테고리 j")
    ax.set_ylabel("조건 카테고리 i")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/17-ko_multilabel-out2-1.png)

**해석**

- **대각선 = 1.0** — 자기 자신과는 항상 같이 등장 (정의상).
- **off-diagonal cell M[i, j]** = "카테고리 i 가 활성된 샘플 중 카테고리 j 도 활성된 비율".
- **합성이 무작위 결합** 이라 true 행렬의 off-diagonal 은 *대략 균등* (각 카테고리의 전체 활성률에 비례) — 특정 쌍이 *유난히 높지 않음*. 실제 사람-annotated 데이터라면 "정치+경제" 처럼 *자연스러운 상관* 이 드러나겠지만, 여기선 합성 방식 때문에 그런 구조가 약합니다.
- **predicted cell 이 true 보다 일관되게 높으면** → 모델이 라벨을 *너무 많이* 활성하는 경향 (over-prediction). threshold 를 0.5 보다 높게 두면 calibration 개선.
