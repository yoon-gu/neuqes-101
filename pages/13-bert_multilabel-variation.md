## 클라이맥스 — Ch 6 sklearn `OneVsRestClassifier(LogisticRegression)` 와 비교

Ch 6의 sklearn 셋업을 *이 노트북 안에서* 다시 학습해 라벨별로 BERT와 비교합니다. **multi-label에서도 BERT의 67M이 sklearn 대비 어디서 이기는가?**

Ch 6의 sklearn 셋업(TF-IDF + 라벨마다 독립 LogisticRegression)을 같은 데이터로 재현해 BERT와 비교할 baseline을 만듭니다. `OneVsRestClassifier`는 5개 라벨을 *완전히 분리된* 5개 이진 분류기로 학습합니다.

```python
# Ch 6 셋업 재현 — TF-IDF + OneVsRestClassifier(LogisticRegression)
texts_train = list(train_full["text"])
texts_eval  = list(eval_full["text"])
Y_train_bin = np.array(train_full["aspects"]).astype(int)
Y_eval_bin  = np.array(eval_full["aspects"]).astype(int)

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train = vec.fit_transform(texts_train)
X_eval  = vec.transform(texts_eval)

clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, n_jobs=-1))
clf.fit(X_train, Y_train_bin)

probs_sk = clf.predict_proba(X_eval)        # (N, 5)
preds_sk = (probs_sk >= 0.5).astype(int)    # (N, 5)

p_mi_sk, r_mi_sk, f1_mi_sk, _ = precision_recall_fscore_support(
    Y_eval_bin, preds_sk, average="micro", zero_division=0,
)
p_ma_sk, r_ma_sk, f1_ma_sk, _ = precision_recall_fscore_support(
    Y_eval_bin, preds_sk, average="macro", zero_division=0,
)
auc_sk = float(roc_auc_score(Y_eval_bin, probs_sk, average="macro"))

print(f"sklearn TF-IDF + OvR LogReg:")
print(f"  vocabulary size:    {len(vec.vocabulary_):,}")
print(f"  micro F1:           {f1_mi_sk:.4f}")
print(f"  macro F1:           {f1_ma_sk:.4f}")
print(f"  macro AUC:          {auc_sk:.4f}")
print(f"  hamming loss:       {hamming_loss(Y_eval_bin, preds_sk):.4f}")
```

**▶ 실행 결과**

```text
sklearn TF-IDF + OvR LogReg:
  vocabulary size:    20,000
  micro F1:           0.7634
  macro F1:           0.6141
  macro AUC:          0.9387
  hamming loss:       0.1426
```

### 5-1. 두 모델의 metric 비교

```python
metrics_bert = {
    k.replace("eval_", ""): v for k, v in eval_metrics.items()
    if k.startswith("eval_") and isinstance(v, float)
}
metrics_sk = {
    "hamming_loss":    float(hamming_loss(Y_eval_bin, preds_sk)),
    "micro_f1":        float(f1_mi_sk),
    "micro_precision": float(p_mi_sk),
    "micro_recall":    float(r_mi_sk),
    "macro_f1":        float(f1_ma_sk),
    "macro_precision": float(p_ma_sk),
    "macro_recall":    float(r_ma_sk),
    "macro_auc":       auc_sk,
}

common = [k for k in metrics_bert if k in metrics_sk]
cmp = pd.DataFrame({
    "metric":             common,
    "sklearn (OvR)":      [metrics_sk[k]   for k in common],
    "BERT (this chapter)":[metrics_bert[k] for k in common],
})
cmp["BERT - sklearn"] = cmp["BERT (this chapter)"] - cmp["sklearn (OvR)"]
print(cmp.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
         metric  sklearn (OvR)  BERT (this chapter)  BERT - sklearn
   hamming_loss         0.1426               0.1246         -0.0180
       micro_f1         0.7634               0.7977          0.0343
micro_precision         0.8915               0.9056          0.0141
   micro_recall         0.6674               0.7127          0.0453
       macro_f1         0.6141               0.7239          0.1099
macro_precision         0.9036               0.9155          0.0119
   macro_recall         0.5307               0.6447          0.1140
      macro_auc         0.9387               0.8994         -0.0393
```

**결과 해석**

BERT가 macro F1에서 +0.11(0.6141 → 0.7239)로 가장 크게 앞서고, 이득의 대부분은 macro recall(+0.114)에서 옵니다 — 드문 라벨을 BERT가 더 잘 잡습니다. 다만 macro AUC는 sklearn이 +0.039 높아, 키워드가 본질 신호인 합성 라벨에서는 TF-IDF의 확률 정렬력도 만만치 않음을 보여줍니다.

### 5-2. 라벨별 F1 비교 — 어디서 BERT가 이기나

라벨별 F1을 두 모델에 대해 따로 계산해 표와 막대그래프로 비교합니다. 어떤 항목에서 BERT가 이기고 어디서 sklearn이 이기는지가 합성 라벨의 성격을 드러냅니다.

```python
def per_label_f1(Y_true, Y_pred):
    f1s = []
    for k in range(K):
        _, _, f1, _ = precision_recall_fscore_support(
            Y_true[:, k], Y_pred[:, k], average="binary", zero_division=0,
        )
        f1s.append(float(f1))
    return f1s

f1_bert = per_label_f1(labels, preds)
f1_sk   = per_label_f1(Y_eval_bin, preds_sk)

label_cmp = pd.DataFrame({
    "aspect":     ASPECTS,
    "sklearn F1": f1_sk,
    "BERT F1":    f1_bert,
})
label_cmp["BERT - sklearn"] = label_cmp["BERT F1"] - label_cmp["sklearn F1"]
print(label_cmp.round(4).to_string(index=False))

# 막대 그래프
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(K)
width = 0.38
ax.bar(x_pos - width/2, f1_sk,   width, label="sklearn (OvR)",     color="#5B8DEF")
ax.bar(x_pos + width/2, f1_bert, width, label="BERT (이번 챕터)", color="#F47272")
ax.set_xticks(x_pos)
ax.set_xticklabels(ASPECTS)
ax.set_ylim(0, 1)
ax.set_ylabel("라벨별 F1")
ax.set_title("라벨별 F1 — sklearn OvR vs BERT")
ax.legend()
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

```text
  aspect  sklearn F1  BERT F1  BERT - sklearn
    food      0.9057   0.9203          0.0146
 service      0.8833   0.8545         -0.0288
   price      0.5271   0.3708         -0.1563
ambiance      0.3333   0.6510          0.3176
location      0.4211   0.8232          0.4022
```

![output](../assets/13-bert_multilabel-out3.png)

**결과 해석**

BERT는 드문 라벨에서 압도적입니다 — location +0.40, ambiance +0.32로, 키워드만으로 안 잡히는 신호까지 학습한 결과입니다. 반대로 price는 sklearn이 +0.16 앞서는데, 0.5 임계값에서 BERT의 recall이 무너진 탓이라 임계값 조정으로 회복 가능한 손실입니다.

**해석**

- 키워드 매칭으로 합성한 라벨은 *키워드 단어가 본질 신호* 라 sklearn TF-IDF가 의외로 강합니다 — 라벨 정의 자체가 단어 빈도와 일치하기 때문.
- BERT가 *큰 폭으로* 이기는 라벨이 있다면 → 그 라벨의 *합성 룰이 키워드만으로 안 잡히는 신호* 를 BERT가 추가로 학습한 것 (예: ambiance에서 "lighting was perfect" 같은 묘사).
- BERT가 *지는 라벨* 도 있을 수 있음 — 키워드가 *결정적* 인 라벨에서 sklearn은 *완벽* 한 매칭, BERT는 *근사* 라 약간의 noise가 들어감.

**합성 라벨의 본질적 한계** — 이 비교는 *키워드 매칭으로 만든 라벨* 위에서의 비교. 실제 사람-annotated multi-label 데이터에선 BERT 격차가 훨씬 큼 (단어 빈도로 안 잡히는 미묘한 항목 인식이 BERT의 강점).
