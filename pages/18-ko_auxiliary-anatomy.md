## 평가 — 메인 task + 보조 task

메인 metric 은 자동 (`compute_metrics_main`). 보조 metric (RMSE, R², Pearson r) 은 별도 forward 로 `count_pred` 를 추출해 측정.

```python
# 메인 metric
eval_metrics_aux = trainer_aux.evaluate()
print("With-aux (lambda=0.05) — main task metrics:")
for k, v in eval_metrics_aux.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
```

**▶ 실행 결과**

```text
Training Loss  Validation Loss  Epoch  Hamming Loss  Micro F1  Micro Precision  Micro Recall  Macro F1  Macro Precision  Macro Recall  Macro Auc  Runtime   Samples Per Second  Steps Per Second
0.154351       0.200851         2      0.073857      0.852328  0.855995         0.848692      0.849294  0.840769         0.859991      0.963988   0.670000  1492.643000         47.765000
With-aux (lambda=0.05) — main task metrics:
               eval_loss: 0.2009
       eval_hamming_loss: 0.0739
           eval_micro_f1: 0.8523
    eval_micro_precision: 0.8560
       eval_micro_recall: 0.8487
           eval_macro_f1: 0.8493
    eval_macro_precision: 0.8408
       eval_macro_recall: 0.8600
          eval_macro_auc: 0.9640
            eval_runtime: 0.6700
  eval_samples_per_second: 1492.6430
   eval_steps_per_second: 47.7650
```

```python
# 보조 metric — eval 전체에 대해 수동 forward
@torch.no_grad()
def aux_predictions(trainer, dataset, batch_size=64):
    trainer.model.eval()
    device = trainer.model.bert.device
    aux_preds, aux_true = [], []
    for i in range(0, len(dataset), batch_size):
        batch_features = [dict(dataset[j]) for j in range(i, min(i + batch_size, len(dataset)))]
        batch = trainer.data_collator(batch_features)
        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        n_act_true = batch_on_device.pop("n_active").cpu().numpy()
        # 메인 labels 도 잠시 제거 (forward 에서 loss 계산 안 하도록)
        batch_on_device.pop("labels", None)
        _ = trainer.model(**batch_on_device, labels=None, n_active=None)
        count_pred = trainer.model.last_count_pred.cpu().numpy()
        aux_preds.extend(count_pred.tolist())
        aux_true.extend(n_act_true.tolist())
    return np.array(aux_preds), np.array(aux_true)


aux_preds_aux, aux_true = aux_predictions(trainer_aux, eval_tok)
rmse_aux = float(np.sqrt(mean_squared_error(aux_true, aux_preds_aux)))
r2_aux   = float(r2_score(aux_true, aux_preds_aux))
pear_aux = float(np.corrcoef(aux_true, aux_preds_aux)[0, 1])

print("\nWith-aux (lambda=0.05) — aux task metrics (n_active regression):")
print(f"  RMSE:    {rmse_aux:.4f}")
print(f"  R^2:     {r2_aux:.4f}")
print(f"  Pearson: {pear_aux:.4f}")
print(f"\n  Aux pred range: [{aux_preds_aux.min():.3f}, {aux_preds_aux.max():.3f}]")
print(f"  Aux true range: [{aux_true.min():.1f}, {aux_true.max():.1f}]")
```

**▶ 실행 결과**

```text
With-aux (lambda=0.05) — aux task metrics (n_active regression):
  RMSE:    0.4141
  R^2:     0.0652
  Pearson: 0.4895

  Aux pred range: [1.188, 2.795]
  Aux true range: [1.0, 2.0]
```

```python
# 메인 task per-sample 예측 (다음 비교 단계에서 사용)
preds_output_aux = trainer_aux.predict(eval_tok)
logits_aux = preds_output_aux.predictions
if isinstance(logits_aux, tuple):
    logits_aux = logits_aux[0]
labels_eval = preds_output_aux.label_ids.astype(int)
probs_aux = 1.0 / (1.0 + np.exp(-logits_aux))
preds_main_aux = (probs_aux >= 0.5).astype(int)

print(f"Main logits shape: {logits_aux.shape}")
print(f"Eval samples:      {len(labels_eval)}")
```

**▶ 실행 결과**

```text
Main logits shape: (1000, 7)
Eval samples:      1000
```

```python
# Per-category classification report (with-aux)
print("Per-category report — with aux (lambda=0.05):")
print(classification_report(
    labels_eval, preds_main_aux,
    target_names=LABEL_NAMES_EN,
    digits=4, zero_division=0,
))
```

**▶ 실행 결과**

```text
Per-category report — with aux (lambda=0.05):
              precision    recall  f1-score   support

  IT/Science     0.7769    0.8016    0.7891       126
     Economy     0.8451    0.8025    0.8233       238
     Society     0.9095    0.8231    0.8642       684
Life&Culture     0.8071    0.9029    0.8523       278
       World     0.8521    0.9057    0.8780       159
      Sports     0.8934    0.9316    0.9121       117
    Politics     0.8012    0.8526    0.8261       156

   micro avg     0.8560    0.8487    0.8523      1758
   macro avg     0.8408    0.8600    0.8493      1758
weighted avg     0.8592    0.8487    0.8524      1758
 samples avg     0.8803    0.8650    0.8534      1758
```
