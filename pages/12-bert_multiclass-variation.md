## 클라이맥스 — sklearn TF-IDF + LogReg 와의 비교 (Ch 5의 BERT 검증)

같은 데이터에 Ch 5 셋업(TF-IDF + multinomial LogReg)을 *이 노트북 안에서* 다시 학습해 비교합니다. **BERT 67M 파라미터가 진짜로 도움이 되는가?** 가 이 비교의 핵심 질문 — sklearn은 GPU 없이도 몇 초 만에 끝나기 때문에 self-contained로 부담 없이 포함됩니다.

```python
# Ch 5 셋업 재현 — TF-IDF + multinomial LogReg
texts_train  = list(train_full["text"])
labels_train = list(train_full["label"])
texts_eval   = list(eval_full["text"])
labels_eval  = list(eval_full["label"])

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train = vec.fit_transform(texts_train)
X_eval  = vec.transform(texts_eval)

clf = LogisticRegression(max_iter=2000, n_jobs=-1)   # 최신 sklearn은 multinomial이 default for multi-class
clf.fit(X_train, labels_train)

probs_sk = clf.predict_proba(X_eval)                 # (B, 5)
preds_sk = probs_sk.argmax(axis=1)                   # (B,)

acc_sk = float(accuracy_score(labels_eval, preds_sk))
ps, rs, f1s, _ = precision_recall_fscore_support(labels_eval, preds_sk, average="macro", zero_division=0)
auc_sk = float(roc_auc_score(labels_eval, probs_sk, multi_class="ovr"))

print(f"sklearn TF-IDF + LogReg:")
print(f"  vocabulary size:    {len(vec.vocabulary_):,}")
print(f"  trained parameters: {clf.coef_.size + clf.intercept_.size:,}  (~{clf.coef_.size/1e3:.0f} K)")
print(f"  accuracy:           {acc_sk:.4f}")
print(f"  macro F1:           {f1s:.4f}")
print(f"  AUC (OvR):          {auc_sk:.4f}")
```

**▶ 실행 결과**

```text
sklearn TF-IDF + LogReg:
  vocabulary size:    20,000
  trained parameters: 100,005  (~100 K)
  accuracy:           0.5420
  macro F1:           0.5380
  AUC (OvR):          0.8420
```

### 5-1. 두 모델의 metric 표 비교

```python
metrics_bert = {
    k.replace("eval_", ""): v for k, v in eval_metrics.items()
    if k.startswith("eval_") and isinstance(v, float)
}
metrics_sk = {
    "accuracy":        acc_sk,
    "macro_precision": float(ps),
    "macro_recall":    float(rs),
    "macro_f1":        float(f1s),
    "auc_ovr":         auc_sk,
}

common = [k for k in metrics_bert if k in metrics_sk]
cmp = pd.DataFrame({
    "metric":             common,
    "sklearn (TF-IDF)":   [metrics_sk[k]   for k in common],
    "BERT":               [metrics_bert[k] for k in common],
})
cmp["BERT - sklearn"] = cmp["BERT"] - cmp["sklearn (TF-IDF)"]
print(cmp.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
         metric  sklearn (TF-IDF)   BERT  BERT - sklearn
       accuracy            0.5420 0.5580          0.0160
macro_precision            0.5377 0.5555          0.0177
   macro_recall            0.5410 0.5595          0.0185
       macro_f1            0.5380 0.5561          0.0181
        auc_ovr            0.8420 0.8657          0.0236
```

### 5-2. 두 모델의 혼동 행렬 비교

같은 평가 데이터에 sklearn은 어디서, BERT는 어디서 헷갈리는지 *나란히* 봅니다.

```python
cm_bert = confusion_matrix(labels, preds, labels=list(range(5)))
cm_sk   = confusion_matrix(labels_eval, preds_sk, labels=list(range(5)))

cm_bert_n = cm_bert / cm_bert.sum(axis=1, keepdims=True)
cm_sk_n   = cm_sk   / cm_sk.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, cm_n, cm_raw, title in [
    (axes[0], cm_sk_n,   cm_sk,   "sklearn TF-IDF + LogReg"),
    (axes[1], cm_bert_n, cm_bert, "BERT"),
]:
    sns.heatmap(
        cm_n, annot=cm_raw, fmt="d", cmap="Blues", vmin=0, vmax=1,
        xticklabels=[STAR_LABELS[k] for k in range(5)],
        yticklabels=[STAR_LABELS[k] for k in range(5)],
        cbar=False, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("예측 별점")
    ax.set_ylabel("실제 별점")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/12-bert_multiclass-out3-1.png)

**해석 가이드**

- *대각선이 더 진하면* 그 모델이 더 잘 맞춘 것.
- *인접 클래스 혼동(±1)* 은 두 모델 모두에서 가장 흔할 것 — 별점이 *순서형* 라벨이라 자연스럽습니다.
- BERT가 sklearn 대비 가장 크게 개선되는 영역은 보통 **3★ (중간 별점)**: 단어 빈도만으로는 *애매한 칭찬·비판이 섞인* 리뷰를 구분하기 어렵지만, BERT는 attention으로 문맥을 보기 때문.
- 만약 BERT가 sklearn보다 *모든 셀에서* 비슷하거나 더 나쁘다면 → 학습량 부족 신호. epoch을 늘리거나 lr을 조정.

## 결과 해석 — '근소 우위'의 정체: 데이터가 정한다

§5 에서 BERT 67M 파라미터가 sklearn 을 **근소하게만**(약 +0.02-0.04) 앞섰습니다. 이진(Ch 10, BERT 약 0.90)·회귀(Ch 9, R² 약 0.66)에서 BERT 가 압도하던 것과 비교하면 5클래스에서 유독 격차가 작아 *실망스럽게* 느껴질 수 있습니다. 하지만 이건 버그가 아니라 **데이터 양과 task 난이도의 함수** 입니다.

부록 `12_bert_multiclass_data_scaling` 에서 학습 데이터를 100 → 30,000 으로 키우며 두 곡선을 그리면 (고정 eval · nested subsample · epoch 2 고정):

| 학습 수 | sklearn | BERT | gap |
|---|---|---|---|
| 100 | 0.373 | 0.229 | **-0.144** ← sklearn 압승 |
| 300 | 0.396 | 0.262 | -0.134 |
| 1,000 | 0.471 | 0.500 | +0.029 ← 교차 |
| 3,000 | 0.524 | 0.557 | +0.033 |
| 10,000 | 0.569 | 0.588 | +0.019 |
| 30,000 | 0.563 | **0.600** | +0.037 |

- **작은 데이터(100-300)선 sklearn 이 크게 이깁니다.** DistilBERT 는 6,700만 파라미터를 100 샘플로 적응시킬 수 없어 분류 헤드가 거의 random(0.229 ≈ 1/5). *큰 모델이 늘 이기는 게 아니라, 적응할 데이터가 있어야* 이깁니다.
- **교차점은 N≈1,000.** 그 위로 BERT 가 줄곧 앞섭니다.
- **30K 까지 키워도 격차는 +0.04 안팎.** 본편(5,000)의 근소 우위는 *이 데이터 규모에서 정상* 이고, 이진만큼 극적이지 않은 건 **별점 5클래스가 본질적으로 어려운**(인접 별점 경계가 모호한 ordinal) task 이기 때문입니다.

> 데이터를 더 부으면 BERT 가 더 벌리지만, 정확도를 크게 끌어올리려면 데이터 외에 *더 강한 모델·HPO·LLM distill* 같은 다른 lever 도 필요합니다 — 부록 §6 의 5가지 lever 카드 참조.
