"""Build 09_bert_regression/appendix_hpo.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "09_bert_regression" / "appendix_hpo.ipynb"

cells = []
_counter = 0


def _cid():
    global _counter
    _counter += 1
    return f"cell{_counter:03d}"


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": text,
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ----- 1. Title -----
md(r"""# Ch 9 부록 — Hyperparameter Optimization (HPO)의 어려움

본 챕터(Ch 9)에서 `learning_rate=2e-5`, `batch_size=16`, `epochs=2` 같은 값을 *경험적으로* 골라 학습했습니다. 그런데 다른 task·다른 데이터로 가면 같은 값이 항상 잘 동작하지는 않습니다. **하이퍼파라미터 최적화(HPO)** 는 이런 값들을 자동으로 찾는 절차이고, 그 자체가 어려운 문제입니다.

이 부록에서 다루는 것:

1. `TrainingArguments` 가 어떤 인자를 받고 각각이 의미하는 바
2. HPO가 *왜* 어려운가 (5가지 이유)
3. 검색 전략 비교 (Grid / Random / Bayesian)
4. 실제 도구로 HPO 시도: **`Trainer.hyperparameter_search` + Optuna** 와 **wandb sweeps**
5. MLflow autolog 통합 (자동 metric 추적)
6. 흔한 함정과 실용 가이드

**환경**: T4 GPU 권장. 작은 sweep을 직접 돌리는 셀이 있어서 ~10분 소요.

> Ch 9 본 흐름으로 돌아가기: [09_bert_regression.ipynb](./09_bert_regression.ipynb)
> 관련 부록: [`appendix_experiment_tracking.ipynb`](./appendix_experiment_tracking.ipynb) — wandb / trackio / MLflow 사용법""")

# ----- 2. TrainingArguments 도입 -----
md(r"""## 1. `TrainingArguments` — 무엇을 받고, 무엇을 의미하나

HPO는 결국 *이 인자들의 값* 을 자동으로 탐색하는 일이라, 어떤 인자가 있고 무엇을 의미하는지부터 정리합니다.

### 핵심 인자 (학습 동역학)

| 인자 | 기본값 | 의미·영향 |
|---|---|---|
| `learning_rate` | 5e-5 | **HPO의 1순위**. BERT 파인튜닝은 보통 1e-5 ~ 5e-5. 너무 크면 발산, 너무 작으면 학습 정체. |
| `num_train_epochs` | 3 | 전체 학습 데이터를 몇 번 돌릴지. 작으면 underfit, 크면 overfit. 1-5 범위가 흔함. |
| `per_device_train_batch_size` | 8 | 한 GPU에서 처리하는 배치 크기. T4에서 BERT-base + max_len=128이면 16-32가 안전. |
| `per_device_eval_batch_size` | 8 | 평가용 배치. 학습보다 크게 두어도 됨 (gradient 없음 → 메모리 여유). |
| `weight_decay` | 0.0 | L2 정규화 강도. BERT 파인튜닝엔 보통 0.01. |
| `warmup_ratio` | 0.0 | 학습 초반 lr을 0에서 천천히 올리는 비율. 0.1 (10%)이 흔함. |
| `lr_scheduler_type` | "linear" | linear/cosine/constant. 기본 linear가 BERT엔 무난. |
| `gradient_accumulation_steps` | 1 | 메모리가 부족할 때 *배치 시뮬레이션*. 1로 두고 batch_size 우선 조정. |
| `max_grad_norm` | 1.0 | gradient clipping. nan 방지. |

### 메모리·속도 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `fp16` | False | mixed precision. T4는 True 권장 (bf16은 불가). |
| `bf16` | False | bf16 precision. A100·H100·TPU에서 권장 (T4 미지원). |
| `gradient_checkpointing` | False | True면 메모리 절감 (속도 ~30% 손해). 큰 모델·긴 시퀀스에 유용. |
| `dataloader_num_workers` | 0 | DataLoader 워커 수. 2~4면 GPU 대기 시간 감소. |

### 평가·로깅·저장 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `eval_strategy` | "no" | "no" / "steps" / "epoch". 평가 빈도. |
| `eval_steps` | None | "steps" 모드일 때 평가 간격. |
| `logging_steps` | 500 | step별 loss/lr 로그 빈도. |
| `save_strategy` | "steps" | "no" / "steps" / "epoch". 체크포인트 저장 빈도. |
| `save_total_limit` | None | 보존할 체크포인트 최대 개수 (오래된 것 삭제). |
| `load_best_model_at_end` | False | True면 학습 끝에 *최고 eval metric 체크포인트* 로 모델 복구. |
| `metric_for_best_model` | None | 어떤 metric을 기준으로 best 판단할지 (예: "eval_mse"). |
| `greater_is_better` | None | metric이 클수록 좋은지 (mse면 False, accuracy면 True). |
| `report_to` | "all" | wandb/trackio/mlflow 등 트래커 연결. 다른 부록에서 다룸. |

### 재현성 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `seed` | 42 | 데이터 셔플·layer init·dropout 마스크에 영향. 같은 seed면 결과 같음 (실용상). |
| `data_seed` | None | 데이터 샘플링 seed (별도로 두면 모델 init seed와 분리). |

### `TrainingArguments` 의 인자 수

전체 100개 이상의 인자가 있습니다. 위는 *HPO에서 자주 건드리는* 핵심만 추렸습니다. 전체 목록은 [공식 문서](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)를 참고하세요.""")

# ----- 3. install -----
code(r"""!pip install -q transformers datasets optuna mlflow wandb""")

# ----- 4. 데이터 준비 (Ch 9 압축) -----
code(r"""import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("yelp_review_full")
# HPO는 trial을 여러 번 돌리므로 데이터를 더 작게 (1 trial이 1-2분에 끝나도록)
train_ds = ds["train"].shuffle(seed=42).select(range(800))
eval_ds  = ds["test"].shuffle(seed=42).select(range(200))

def tok(b):
    out = tokenizer(b["text"], truncation=True, max_length=128)
    out["labels"] = [float(l) + 1.0 for l in b["label"]]
    return out

train_tok = train_ds.map(tok, batched=True).remove_columns(["text", "label"])
eval_tok  = eval_ds.map(tok,  batched=True).remove_columns(["text", "label"])

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.flatten()
    return {
        "mse": float(mean_squared_error(labels, preds)),
        "mae": float(mean_absolute_error(labels, preds)),
        "r2":  float(r2_score(labels, preds)),
    }

print(f"train: {len(train_tok)} / eval: {len(eval_tok)}")""")

# ----- 5. 왜 어려운가 -----
md(r"""## 2. HPO가 왜 어려운가 — 5가지 이유

### (1) 검색 공간이 너무 큼
`learning_rate` (실수, 보통 1e-7 ~ 1e-3 사이의 *log scale*) × `batch_size` (8/16/32/64) × `weight_decay` (0 ~ 0.1) × `warmup_ratio` (0 ~ 0.2) × `lr_scheduler_type` (3종) ... 인자 5개만 잡아도 가능한 조합이 *수만 개* 입니다.

### (2) 한 번 평가가 비싸다
학습 한 번이 5분 걸리면 100개 조합을 다 돌려보는 데 *500분 (8시간 이상)* 입니다. 큰 모델·큰 데이터로 가면 trial 하나가 *수 시간* 걸리기도 합니다.

### (3) Loss surface가 비볼록(non-convex)
sklearn `LinearRegression` 의 정규방정식과 달리 BERT 학습의 loss landscape는 매끄럽지 않습니다. 같은 하이퍼파라미터로 같은 데이터를 학습해도 *random seed에 따라* 결과가 미세하게 달라집니다.

### (4) Validation set에 overfit
검증 데이터로 HPO를 반복하면 모델이 *그 검증 셋에 맞춰 미세 조정* 됩니다 (data snooping). 진짜 일반화 성능을 알려면 *별도 test set* 이 필요한데, 이게 작으면 평가 자체에 잡음이 큽니다.

### (5) 인자 간 상호작용
`learning_rate` 와 `batch_size` 는 독립이 아닙니다. 보통 batch_size를 2배 늘리면 lr도 1.4배 (sqrt 스케일) ~ 2배 (linear 스케일)로 늘려야 학습이 비슷하게 진행됩니다. *한 인자씩 바꿔서 최적화* 하는 방식 (one-at-a-time)은 이런 상호작용을 못 잡습니다.""")

# ----- 6. 검색 전략 -----
md(r"""## 3. 검색 전략 비교

| 전략 | 동작 | 장점 | 단점 |
|---|---|---|---|
| **Grid Search** | 미리 정한 격자의 모든 조합 시도 | 단순, 결과 해석 쉬움 | 인자 수에 *지수적* — 5인자 × 4값씩이면 1024 trial |
| **Random Search** | 정해진 분포에서 무작위 샘플링 | 비싼 인자에 budget 효율적 분배, 구현 단순 | 운에 의존 |
| **Bayesian Optimization** (Optuna, HyperOpt) | 이전 trial 결과로 다음 인자 예측 | 적은 trial로 좋은 결과 | 구현 복잡, 시작이 무거움 |
| **Population-Based** (PBT) | 여러 모델을 동시에 학습하며 인자를 진화 | 학습 도중 인자 변경 가능 | 자원 많이 소비 |

**일반적인 권장**: 처음엔 **Random Search 10-20 trial** 로 대략적인 좋은 영역을 찾고, 그 영역에서 Bayesian으로 좁혀 들어가는 *2단계* 접근.""")

# ----- 7. Trainer.hyperparameter_search -----
md(r"""## 4. `Trainer.hyperparameter_search` + Optuna

Hugging Face `Trainer` 는 `hyperparameter_search` 메서드를 통해 Optuna·Ray Tune·sigopt와 통합돼 있습니다. 가장 적은 코드로 HPO를 붙이는 방법.

### `model_init` 패턴

HPO는 매 trial마다 *새 모델을 처음부터* 학습해야 하므로, 미리 만든 모델 객체 대신 **모델을 만드는 함수** 를 넘깁니다.""")

code(r"""def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=1, problem_type="regression",
    )""")

# ----- 8. hp_space -----
md(r"""### `hp_space` — 탐색 공간 정의

각 trial이 시도할 하이퍼파라미터의 분포를 정합니다. Optuna의 `trial.suggest_*` API.""")

code(r"""def hp_space(trial):
    return {
        # learning_rate: 1e-6 ~ 1e-4, log scale (10배 단위로 분포가 균등)
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        # batch_size: 정해진 후보 중 선택
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
        # weight_decay: 0 ~ 0.1, 균등
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }""")

# ----- 9. compute_objective -----
md(r"""### `compute_objective` — 무엇을 최소화/최대화할지

`compute_metrics` 가 dict를 반환하므로, 그중 어떤 값을 기준으로 trial을 비교할지 정합니다.""")

code(r"""# 회귀 → MSE 최소화
def compute_objective(metrics):
    return metrics["eval_mse"]""")

# ----- 10. 실제 sweep 돌리기 -----
md(r"""### 실제 HPO 돌리기 (n_trials=3)

학습 시간 절약을 위해 trial 3개·각 trial 1 epoch만 돌립니다. 실무에선 보통 20-50 trial.""")

code(r"""# 기본 TrainingArguments — hp_space에서 덮어쓸 인자만 비워둠
default_args = TrainingArguments(
    output_dir="./hpo_out",
    num_train_epochs=1,
    per_device_eval_batch_size=32,
    fp16=True,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=42,
    disable_tqdm=True,    # trial 출력이 너무 많아지지 않게
)

trainer = Trainer(
    model=None,                    # ← model_init 사용시 None
    model_init=model_init,
    args=default_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

best_run = trainer.hyperparameter_search(
    direction="minimize",          # mse 최소화
    backend="optuna",
    n_trials=3,                    # 시간 절약 위해 3개만
    hp_space=hp_space,
    compute_objective=compute_objective,
)

print(f"\nBest trial:")
print(f"  objective(eval_mse): {best_run.objective:.4f}")
print(f"  hyperparameters:     {best_run.hyperparameters}")""")

# ----- 11. 결과 해석 -----
md(r"""**관찰 포인트** — 출력에서 trial 3개의 결과를 봅니다.

- `learning_rate` 가 너무 크거나(1e-4 근처) 너무 작은 경우(1e-6 근처) loss가 정체될 수 있습니다.
- `weight_decay` 영향은 보통 작은 데이터에선 미미합니다 (실험적으로 큰 차이가 안 보일 수 있음).
- `batch_size` 16 vs 32 차이는 epoch 수가 같으면 큰 차이가 없습니다 (step 수가 절반인 게 단점).

**제대로 된 HPO는** trial 30-50개에 더 다양한 하이퍼파라미터를 두지만, 이 부록에서는 *패턴* 만 봅니다.""")

# ----- 12. wandb sweeps -----
md(r"""## 5. wandb Sweeps — 다른 접근

wandb는 자체 sweep 시스템을 갖고 있어 여러 머신에서 *분산 HPO* 를 돌리기 좋습니다. 코드 형태만 보여드리고 직접 실행은 wandb 계정이 있어야 의미가 있어서 생략합니다.

```python
# wandb sweep 정의
sweep_config = {
    "method": "bayes",                              # random / grid / bayes
    "metric": {"name": "eval_mse", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "min": 1e-6, "max": 1e-4,
            "distribution": "log_uniform_values",
        },
        "per_device_train_batch_size": {"values": [8, 16, 32]},
        "weight_decay": {"min": 0.0, "max": 0.1},
    },
}

def train_with_sweep():
    import wandb
    wandb.init()
    config = wandb.config

    args = TrainingArguments(
        output_dir="./sweep_out",
        num_train_epochs=1,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        report_to="wandb",
        fp16=True,
    )
    trainer = Trainer(
        model_init=model_init, args=args,
        train_dataset=train_tok, eval_dataset=eval_tok,
        processing_class=tokenizer, compute_metrics=compute_metrics,
    )
    trainer.train()

# CLI에서:
# wandb sweep sweep.yaml         → sweep ID 발급
# wandb agent <sweep_id>          → 실제 trial 실행
```

wandb sweeps의 장점은 *여러 머신* 에서 같은 sweep_id로 agent를 띄우면 자동으로 분산 실행된다는 것입니다.""")

# ----- 13. MLflow autolog -----
md(r"""## 6. MLflow autolog — 자동 metric 추적

MLflow에는 `autolog()` 라는 편의 기능이 있어, *Trainer 내부의 모든 학습 metric을 자동으로* MLflow에 기록합니다. HPO trial마다 별도 run으로 자동 분리.""")

code(r"""import mlflow

mlflow.set_experiment("ch09-hpo")
mlflow.transformers.autolog()    # transformers Trainer 자동 로깅

# 위에서 만든 trainer를 다시 한 번 돌려도 (또는 hyperparameter_search 다시 돌려도)
# 학습 metric이 ./mlruns/ 에 자동 저장됩니다.
# autolog가 켜져 있으면 manual log 호출 없이 작동.

print("autolog enabled — subsequent trainer.train() calls auto-log to ./mlruns/")""")

# ----- 14. 함정 -----
md(r"""## 7. 흔한 함정

### 함정 1 — Validation에 overfit
HPO를 반복하면 모델이 검증 셋에 맞춰 미세 조정됩니다. 해결: *별도 test set* 으로 최종 평가, HPO 결과를 너무 신뢰하지 말기.

### 함정 2 — Trial 예산 부족
trial 3개로 좋은 인자를 찾을 수는 없습니다. 결과 해석 시 *통계적 유의미함이 있는지* 봐야 합니다 (run 여러 번 돌려 평균 + 표준편차).

### 함정 3 — Random seed 변동성
같은 하이퍼파라미터로 seed만 바꿔도 eval_mse가 0.05 정도 흔들리는 경우가 흔합니다. trial 간 차이가 그 수준이면 의미 없는 noise일 수 있습니다.

### 함정 4 — Early stopping을 너무 공격적으로
trial 비용 절감을 위해 1 epoch만 돌리면, *원래는 좋은* 인자가 첫 epoch에서 나쁘게 보여 버려질 수 있습니다 (느린 학습률은 처음에 약하지만 길게 가면 좋음).

### 함정 5 — 인자 분포가 잘못
`learning_rate` 를 `[1e-6, 1e-3]` 균등 분포에서 뽑으면, 99%의 trial이 `>1e-4` 영역(거의 발산)에 떨어집니다. **반드시 `log=True` (log scale)** 로.""")

# ----- 15. 실용 가이드 -----
md(r"""## 8. Ch 9 BERT 회귀에 적용한다면 — 실용 가이드

### 1단계 — 한 인자씩 좁히기 (sanity check)
- `learning_rate` 만 [5e-6, 1e-5, 2e-5, 5e-5, 1e-4] 5개로 grid 학습. 가장 좋은 영역 찾기.
- 그 다음 `weight_decay`, `warmup_ratio` 등을 조정.

### 2단계 — Random/Bayesian으로 동시 최적화
- `Trainer.hyperparameter_search(backend="optuna", n_trials=20)` 패턴.
- `learning_rate` (log scale), `per_device_train_batch_size`, `weight_decay`, `warmup_ratio` 4-5개 인자.

### 3단계 — 최종 모델 학습
- 찾은 인자 + 더 큰 데이터·더 많은 epoch로 *한 번* 학습.
- 별도 test set으로 평가.

### Ch 9 본 노트북에서는 왜 HPO를 안 했나
교육용으로 *한 번에 끝나는 학습 흐름* 을 보여주는 게 우선이라, 검증된 BERT 파인튜닝 표준값(`lr=2e-5`, `batch=16`, `epochs=2`)을 사용했습니다. 이 값들이 Yelp 별점 회귀에서 충분히 동작합니다.

실무에서는 데이터·태스크가 새로 등장할 때마다 위 1-3단계를 거치는 게 표준입니다.""")

# ----- 16. 정리 -----
md(r"""## 정리

- `TrainingArguments` 는 100개 이상 인자가 있고, HPO에서 자주 건드리는 건 *5-10개* 정도.
- HPO가 어려운 이유: 큰 검색 공간 × 비싼 평가 × non-convex landscape × validation overfit × 인자 간 상호작용.
- 도구는 셋 다 같은 `Trainer` 골격에 통합됩니다.
  - **Optuna + `Trainer.hyperparameter_search`**: 가장 가벼운 시작.
  - **wandb sweeps**: 분산 실행, dashboard 풍부.
  - **MLflow autolog**: 자동 metric 추적.
- 실무에선 Random Search 20개 → Bayesian으로 좁혀 들어가는 *2단계* 접근이 흔합니다.

**Ch 9 본 흐름으로 돌아가기**: [09_bert_regression.ipynb](./09_bert_regression.ipynb)

**관련 부록**: [`appendix_experiment_tracking.ipynb`](./appendix_experiment_tracking.ipynb) — wandb / trackio / MLflow 트래킹 통합

**다음 챕터**: [Ch 10 — BERT Binary 두 방식 비교](../10_bert_binary/)""")


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT.relative_to(REPO)}  ({len(cells)} cells)")
