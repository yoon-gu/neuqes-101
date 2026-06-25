§8 은 λ=0 vs λ=0.05 *두 점* 만 비교했습니다. **λ 전체 곡선(0 → 0.5)은 부록 `18_ko_auxiliary_lambda_sweep` 에서 실측** 으로 그립니다 — sweet spot 이 λ=0.05 이고, λ≥0.2 부터 메인이 무너지는 모습, 그리고 Ch 14(강한 보조)와의 대조를 거기서 봅니다.

아래는 이 노트북 안에서 *빠르게* 몇 점만 직접 돌려보고 싶을 때의 선택 코드입니다 (각 λ 마다 처음부터 재학습 — 시간 여유 있을 때만).

```python
# 시간 여유 있을 때만 실행 — 각 lambda 마다 처음부터 다시 학습
LAMBDA_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
RUN_LAMBDA_SWEEP = False        # ← True 로 바꿔 실행

sweep_results = []

if RUN_LAMBDA_SWEEP:
    for lam in LAMBDA_GRID:
        print(f"\n{'='*60}")
        print(f"Training with lambda_aux = {lam}")
        print(f"{'='*60}")
        m = make_model()
        args = TrainingArguments(
            output_dir=f"./ch18_sweep_lam{lam}",
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            fp16=True,
            eval_strategy="no",
            logging_steps=200,
            save_strategy="no",
            report_to="none",
            seed=SEED,
            remove_unused_columns=False,
        )
        tr = AuxTrainer(
            model=m, args=args,
            train_dataset=train_tok, eval_dataset=eval_tok,
            data_collator=collator, processing_class=tokenizer,
            compute_metrics=compute_metrics_main, lambda_aux=lam,
        )
        tr.train()
        ev = tr.evaluate()
        sweep_results.append({
            "lambda": lam,
            "macro_f1": float(ev.get("eval_macro_f1", float("nan"))),
            "micro_f1": float(ev.get("eval_micro_f1", float("nan"))),
            "macro_auc": float(ev.get("eval_macro_auc", float("nan"))),
        })
        del m, tr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sweep_df = pd.DataFrame(sweep_results)
    print("\nLambda sweep result:")
    print(sweep_df.round(4).to_string(index=False))
else:
    print("Lambda sweep skipped. Set RUN_LAMBDA_SWEEP=True to run (~30 min extra on T4).")
```

**▶ 실행 결과**

```text
Lambda sweep skipped. Set RUN_LAMBDA_SWEEP=True to run (~30 min extra on T4).
```

**해석 가이드 — 결과를 직접 보면**

- macro_f1 이 λ=0.05 근처에서 최대 → 약한 보조 신호가 작은 가중치에서 정규화 효과로 메인에 도움.
- λ≥0.2 에서 macro_f1 이 하락 → 보조 신호가 메인 학습을 방해하기 시작.
- λ=0 이 최대 (baseline 이 가장 좋음) → 보조 task 가 이 셋업에선 도움 안 됨. quick 모드 노이즈일 수 있어 시드 바꿔 재실행 권장.

전체 실행 결과와 그래프는 [18-4 부록 — λ 스윕으로 약한 보조 신호의 sweet spot 찾기](18-ko_auxiliary-lambda_sweep.md)에 정리했습니다.
