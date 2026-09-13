## 평가 — 메인 task + 보조 task

메인 metric 은 자동으로 계산됨 (`compute_metrics`). 보조 metric (RMSE, R², Pearson r) 은 별도 forward로 보조 logits 를 추출해 측정.

```python
# 메인 metric
eval_metrics_aux = trainer_aux.evaluate()
print("With-aux (λ=0.05) — main task metrics:")
for k, v in eval_metrics_aux.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
```

**▶ 실행 결과**

```text
Training Loss  Validation Loss  Epoch  Hamming Loss  Micro F1  Micro Precision  Micro Recall  Macro F1  Macro Precision  Macro Recall  Macro Auc  Runtime   Samples Per Second  Steps Per Second
0.289381       0.296347         2      0.097800      0.846948  0.919158         0.785258      0.810893  0.932765         0.731567      0.918232   0.988200  1011.891000         32.381000
With-aux (λ=0.05) — main task metrics:
               eval_loss: 0.2963
       eval_hamming_loss: 0.0978
           eval_micro_f1: 0.8469
    eval_micro_precision: 0.9192
       eval_micro_recall: 0.7853
           eval_macro_f1: 0.8109
    eval_macro_precision: 0.9328
       eval_macro_recall: 0.7316
          eval_macro_auc: 0.9182
            eval_runtime: 0.9882
  eval_samples_per_second: 1011.8910
   eval_steps_per_second: 32.3810
```

```python
# 보조 metric — eval 전체에 대해 수동 forward (작아서 빠름)
@torch.no_grad()
def aux_predictions(trainer, dataset, batch_size=64):
    trainer.model.eval()
    device = trainer.model.device
    aux_preds, aux_true = [], []
    for i in range(0, len(dataset), batch_size):
        batch_features = [dict(dataset[j]) for j in range(i, min(i + batch_size, len(dataset)))]
        batch = trainer.data_collator(batch_features)
        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        aux_lbl = batch_on_device.pop("aux_labels").cpu().numpy()
        out = trainer.model(**{k: v for k, v in batch_on_device.items() if k != "labels"},
                            output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0, :]
        aux_logits = trainer.model.aux_head(cls).squeeze(-1).cpu().numpy()
        aux_preds.extend(aux_logits.tolist())
        aux_true.extend(aux_lbl.tolist())
    return np.array(aux_preds), np.array(aux_true)


aux_preds_aux, aux_true = aux_predictions(trainer_aux, eval_tok)
rmse_aux = float(np.sqrt(mean_squared_error(aux_true, aux_preds_aux)))
r2_aux   = float(r2_score(aux_true, aux_preds_aux))
pear_aux = float(np.corrcoef(aux_true, aux_preds_aux)[0, 1])

print("\nWith-aux (λ=0.05) — aux task metrics (star regression, 0-1 scale):")
print(f"  RMSE:    {rmse_aux:.4f}")
print(f"  R^2:     {r2_aux:.4f}")
print(f"  Pearson: {pear_aux:.4f}")
```

**▶ 실행 결과**

```text
With-aux (λ=0.05) — aux task metrics (star regression, 0-1 scale):
  RMSE:    0.2640
  R^2:     0.4277
  Pearson: 0.6603
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
Main logits shape: (1000, 5)
Eval samples:      1000
```
