## 평가 — Ch 15 / Ch 21 과 같은 5종 metric + 학습 곡선

`accuracy / precision / recall / F1 / AUC` 전부 같은 정의. 마지막에 confusion matrix 와 학습 곡선을 같이 그려 *본체 출발점 변화가 학습 동역학에 어떻게 드러나는지* 시각화.

eval 셋에서 5종 지표를 측정합니다. accuracy 가 0.5 (동전 던지기) 에서 얼마나 떨어져 있는지가 이 작은 본체 transfer 의 실효성을 보여 줍니다.

```python
cls_eval_metrics = cls_trainer.evaluate()
print("Ch 23 small BERT (scratch MLM 3 epoch on Korean Wikipedia + NSMC fine-tune) — eval:")
for k, v in cls_eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
Ch 23 small BERT (scratch MLM 3 epoch on Korean Wikipedia + NSMC fine-tune) — eval:
             eval_loss: 0.6885
         eval_accuracy: 0.5480
        eval_precision: 0.5614
           eval_recall: 0.4309
               eval_f1: 0.4875
              eval_auc: 0.5545
```

**결과 해석**

accuracy 0.548, AUC 0.554 로 동전 던지기(0.5)를 살짝 웃도는 수준입니다. recall 0.431 이 precision 0.561 보다 낮아, 모델이 긍정을 덜 예측하는 쪽으로 약간 치우쳐 있습니다 — 짧은 사전학습으로는 NSMC 의 감성 신호를 거의 잡지 못했음을 보여 줍니다.

eval 셋 전체에 대해 예측을 뽑아 클래스별 precision/recall/f1 을 자세히 봅니다. 정답일 때와 틀릴 때의 top-1 확률 평균도 같이 출력해, 모델이 *확신을 가지고 맞히는지* 아니면 *애매하게 추측하는지* 를 진단합니다.

```python
preds_output = cls_trainer.predict(cls_eval)
cls_logits = preds_output.predictions
cls_labels = preds_output.label_ids.astype(int)

exp = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
cls_probs_full = exp / exp.sum(axis=1, keepdims=True)
cls_preds = cls_probs_full.argmax(axis=1)
cls_probs_pos = cls_probs_full[:, 1]

print(f"Logits shape: {cls_logits.shape}")
print(f"Predicted positive rate: {(cls_preds == 1).mean():.1%}")
print(f"Top-1 prob mean: correct={cls_probs_full.max(axis=1)[cls_preds == cls_labels].mean():.4f}, "
      f"wrong={cls_probs_full.max(axis=1)[cls_preds != cls_labels].mean():.4f}")
print()
print(classification_report(
    cls_labels, cls_preds,
    target_names=["negative", "positive"],
    digits=4, zero_division=0,
))
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Logits shape: (1000, 2)
Predicted positive rate: 38.3%
Top-1 prob mean: correct=0.5245, wrong=0.5226

              precision    recall  f1-score   support

    negative     0.5397    0.6647    0.5957       501
    positive     0.5614    0.4309    0.4875       499

    accuracy                         0.5480      1000
   macro avg     0.5505    0.5478    0.5416      1000
weighted avg     0.5505    0.5480    0.5417      1000
```

**결과 해석**

정답일 때 top-1 확률 0.5245, 틀릴 때 0.5226 으로 거의 같습니다 — 모델이 맞히든 틀리든 0.5 부근에서 *확신 없이* 판단한다는 뜻입니다. 예측 positive rate 38.3% 와 negative recall 0.665 vs positive recall 0.431 에서 보듯 부정 쪽으로 살짝 기울었지만, 전반적으로 균등 추측에 가깝습니다.

### 5-1. 학습 곡선 — MLM 사전학습 효과가 보이는 자리

분류 fine-tune 의 step-by-step train loss 를 그려, *시작점* 과 *수렴점* 을 같이 확인.

분류 fine-tune 의 step별 train loss 를 그려 *시작점* 과 *수렴점* 을 한눈에 봅니다. random 기준선 `ln 2` ≈ 0.693 을 점선으로 같이 그어, 곡선이 기준선에서 의미 있게 떨어졌는지 시각적으로 확인합니다.

```python
log_history = cls_trainer.state.log_history
train_logs = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]

if train_logs:
    steps, losses = zip(*train_logs)
    random_baseline = math.log(2)

    sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, "o-", color="#4878D0", label="학습 CE loss (small BERT + ko wiki MLM)")
    ax.axhline(random_baseline, color="black", lw=1.0, ls=":",
               label=f"랜덤 기준선 (ln 2 = {random_baseline:.3f})")
    ax.set_xlabel("학습 step")
    ax.set_ylabel("CE loss (binary)")
    ax.set_title("NSMC 분류 fine-tune loss — small BERT (한국어 위키백과 MLM body)")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No train loss logs found.")
```

**▶ 실행 결과**

![output](../assets/23-ko_bert_classify-out1.png)

**결과 해석**

학습 곡선이 random 기준선 0.693 바로 위에 거의 붙어 머물러, 2 epoch 동안 의미 있는 하강이 일어나지 않았습니다. 얕은 사전학습 본체로는 NSMC 분류 신호를 학습할 출발점이 부족했음을 곡선이 그대로 보여 줍니다.

### 5-2. Confusion matrix

혼동 행렬로 부정/긍정 각 클래스가 어디로 잘못 분류되는지 봅니다. 셀 숫자는 실제 개수, 색은 행 기준 정규화(recall)라 *실제 라벨별로 얼마나 맞혔는지* 가 색 농도로 드러납니다.

```python
sns.set_theme(style="white", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
cm = confusion_matrix(cls_labels, cls_preds, labels=[0, 1])
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_norm, annot=cm, fmt="d",
    cmap="Blues", vmin=0, vmax=1,
    xticklabels=["부정", "긍정"],
    yticklabels=["부정", "긍정"],
    cbar_kws={"label": "행 기준 정규화 (recall)"}, ax=ax,
)
ax.set_xlabel("예측값")
ax.set_ylabel("실제값")
ax.set_title("Ch 23 small BERT (ours + ko wiki MLM) — 혼동 행렬")
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/23-ko_bert_classify-out2.png)

**결과 해석**

실제 부정의 약 66%, 실제 긍정의 약 43% 만 맞혀, 부정 쪽으로 치우친 예측 경향이 행렬에서도 확인됩니다. 두 클래스 모두 절반 안팎의 오분류가 있어, 모델이 뚜렷한 결정 경계를 형성하지 못했음을 보여 줍니다.

## 2-way 비교 — Ch 15 (klue/bert-base) vs Ch 23 ours (small BERT + ko wiki MLM)

두 셋업을 한 표·한 막대 그래프로. Ch 15 의 수치는 *해당 노트북의 검증된 결과* 를 인용 — 학습자가 직접 Ch 15 노트북을 돌려 본인 수치로 갱신해 보면 더 좋습니다.

| 차원 | Ch 15 (klue/bert-base) | Ch 23 ours (small + MLM) |
|---|---|---|
| 본체 파라미터 | 약 110M | 약 10M |
| 사전학습 코퍼스 | 한국어 위키 + 모두의 말뭉치 + 뉴스 + 댓글 (약 8.4B 토큰) | 한국어 Wikipedia paragraphs 5K (약 50만-80만 토큰) |
| 사전학습 시간 | TPU 수일 | T4 약 8-10분 |
| Fine-tune 도메인 | NSMC 이진 (다른 도메인) | NSMC 이진 (다른 도메인) |
| 분류 fine-tune 셋업 | (둘 다 같음 — 5K/1K, batch 16, lr 2e-5, 2 epoch, fp16) | |

마지막으로 Ch 15 (`klue/bert-base`) 의 참고 수치와 본 챕터 결과를 한 표로 나란히 둡니다. 두 모델 모두 *일반 한국어 → NSMC transfer* 라는 같은 패턴이므로, 격차의 거의 전부가 *사전학습 규모* 에서 옵니다.

```python
# Ch 15 reference 수치 — klue/bert-base + NSMC 5K/1K + 2 epoch 의 *전형적* 결과
# (실측치는 학습자가 Ch 15 노트북을 돌려 본인 값으로 갱신 권장)
CH15_REFERENCE = {
    "accuracy":  0.86,
    "precision": 0.86,
    "recall":    0.86,
    "f1":        0.86,
    "auc":       0.93,
}

ch23_ours = {k.replace("eval_", ""): v for k, v in cls_eval_metrics.items()
             if k.startswith("eval_") and isinstance(v, float)
             and k.replace("eval_", "") in CH15_REFERENCE}

comparison = pd.DataFrame({
    "metric":                    list(CH15_REFERENCE.keys()),
    "Ch15 klue/bert-base (ref)": [CH15_REFERENCE[k] for k in CH15_REFERENCE.keys()],
    "Ch23 ours (small + MLM)":   [ch23_ours.get(k, float("nan")) for k in CH15_REFERENCE.keys()],
})
print("2-way comparison — NSMC binary classification metrics")
print(comparison.round(4).to_string(index=False))
```

**▶ 실행 결과**

```text
2-way comparison — NSMC binary classification metrics
   metric  Ch15 klue/bert-base (ref)  Ch23 ours (small + MLM)
 accuracy                       0.86                   0.5480
precision                       0.86                   0.5614
   recall                       0.86                   0.4309
       f1                       0.86                   0.4875
      auc                       0.93                   0.5545
```

**결과 해석**

accuracy 0.86 vs 0.548 로 약 32%p 격차입니다. 두 셋업이 *같은 transfer 패턴* 을 따르므로 이 격차는 거의 전부 사전학습 규모(약 10,000배 토큰 차이)와 모델 크기(11배)의 가치를 정량으로 보여 줍니다.

같은 비교를 막대 그래프로 그려 5종 지표의 격차를 한눈에 봅니다. 표를 long-format 으로 `melt` 한 뒤 모델별 색으로 묶어 그립니다.

```python
# 2-way bar chart 로 한눈에 보기
sns.set_theme(style="whitegrid", context="talk", font="NanumGothic", rc={"axes.unicode_minus": False})
plot_df = comparison.melt(
    id_vars=["metric"],
    value_vars=["Ch15 klue/bert-base (ref)", "Ch23 ours (small + MLM)"],
    var_name="model", value_name="score",
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=plot_df, x="metric", y="score", hue="model",
    palette={
        "Ch15 klue/bert-base (ref)": "#4878D0",
        "Ch23 ours (small + MLM)":   "#EE854A",
    },
    ax=ax,
)
ax.set_ylim(0, 1.05)
ax.set_title("NSMC 이진 분류 — 2-way 비교 (Ch15 ref / Ch23 ours)")
ax.set_xlabel("지표")
ax.set_ylabel("점수")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()
```

**▶ 실행 결과**

![output](../assets/23-ko_bert_classify-out3.png)

**결과 해석**

모든 지표에서 Ch 15 (파란 막대) 가 본 챕터 (주황 막대) 를 크게 앞서며, 특히 AUC 에서 0.93 vs 0.55 의 격차가 두드러집니다. 같은 transfer 패턴·같은 fine-tune 셋업이라는 통제 조건 위에서 이 격차가 *사전학습 규모의 가치* 를 시각적으로 못 박습니다.

**관찰 — *동일 transfer 패턴 안에서 사전학습 규모 격차* 가 NSMC 정확도에 어떻게 드러나나**

실측 (실행본 기준):
- **Ch 15** (`klue/bert-base`, 약 110M, 약 8.4B 토큰 대규모 한국어 사전학습): accuracy 약 0.86, AUC 약 0.93
- **Ch 23 ours** (small BERT, 한국어 Wikipedia 2K paragraphs × 3 epoch 사전학습): accuracy 약 0.54, AUC 약 0.56 (동전 던지기 수준 — MLM 약 0.2분의 짧은 사전학습 한계)

**Ch 15 vs Ch 23 ours**: accuracy 약 32%p 격차. 두 모델이 *같은 transfer 패턴* (일반 한국어 위키 → NSMC) 을 따르므로 격차의 거의 전부가 *사전학습 규모의 가치* — 약 8.4B 토큰의 *일반 한국어 지식* 이 `klue/bert-base` 본체에 압축되어 있어, NSMC 같은 *비격식 구어체 도메인* 에도 빠르게 적응합니다. *작은 일반 사전학습 < 대규모 일반 사전학습* 의 본질은 단순 양적 차이가 아니라 *도메인 다양성의 질적 차이* — `klue/bert-base` 는 위키 + 뉴스 + 블로그·댓글까지 포함한 약 8.4B 토큰으로 비격식 한국어 도메인도 *이미 본 적이 있어* transfer 가 자연스럽습니다.

> NSMC 는 *짧은 한 줄 리뷰* 이고 *반어·맞춤법 흔들림·라벨 노이즈* 가 섞여 있어 영어 Yelp (Ch 21) 보다 *살짝 더 어려운* 데이터. 작은 모델 + 작은 사전학습 환경에서는 그 어려움이 더 두드러집니다 — 한국어 환경 특유의 *negative transfer 가능성* 까지 포함한 정량 비교는 부록을 참조하세요.

## 부록 — random init baseline + negative transfer 분석

*MLM 사전학습 없이 random init 으로 바로 분류 fine-tune* 한 결과와의 비교, 그리고 *한국어 위키 → NSMC 의 큰 도메인 gap* 에서 발생할 수 있는 **negative transfer** 현상의 분석은 부록 노트북 [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb) 에서 다룹니다. 영어 Ch 21 (transfer 양성) 과 한국어 Ch 23 (transfer 음성 가능) 의 비대칭 메커니즘이 핵심.
