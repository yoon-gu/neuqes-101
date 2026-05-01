"""Build 09_bert_regression/appendix_experiment_tracking.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "09_bert_regression" / "appendix_experiment_tracking.ipynb"

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
md(r"""# Ch 9 부록 — 학습 실험 관리 (wandb / trackio / MLflow)

본 챕터(Ch 9)에서는 `report_to="none"` 으로 두어 학습 metric이 콘솔에만 출력됐습니다. 실무에서는 **여러 학습 실행을 비교** 하거나 **하이퍼파라미터를 바꿔가며 실험** 하는 일이 많아서, 결과를 한 곳에 모으는 *experiment tracker* 가 거의 필수입니다.

이 부록은 세 가지 도구를 직접 돌려보는 자리입니다.

- **wandb (Weights & Biases)** — 가장 널리 쓰이는 SaaS 도구. 무료 tier로 시작 가능, 공유 가능한 dashboard.
- **trackio** — Hugging Face가 제공하는 가벼운 자체 호스팅 트래커. 외부 계정 없이 로컬·HF Spaces에서 동작.
- **MLflow** — 회사 내부 인프라에서 자주 쓰는 표준. 자체 서버 + model registry까지 묶음.

`Trainer` 와의 통합은 세 도구 모두 같은 패턴이므로, `TrainingArguments(report_to="...")` 한 줄만 바꾸면 됩니다.

**환경**: T4 GPU 권장 (Ch 9와 동일한 짧은 학습을 도구별로 한 번씩, 총 세 번)
**예상 시간**: 약 25분 (wandb 로그인 시간 포함)

> Ch 9 본 흐름으로 돌아가기: [09_bert_regression.ipynb](./09_bert_regression.ipynb)""")

# ----- 2. report_to 도입 -----
md(r"""## 1. `report_to` — Trainer가 실험 정보를 어디로 보낼까

`TrainingArguments` 의 `report_to` 인자가 **학습 step별 loss·learning rate·평가 metric을 어떤 트래커에 보낼지** 결정합니다.

| 값 | 동작 |
|---|---|
| `"none"` | 외부 로깅 없음 — 콘솔 출력만 (본 챕터 기본) |
| `"wandb"` | wandb 서버에 metric 전송 |
| `"trackio"` | trackio 인스턴스에 전송 |
| `"all"` | 설치된 모든 통합 사용 |
| `["wandb", "trackio"]` 리스트 | 여러 트래커에 동시 전송 |

내부 동작은 단순합니다. `Trainer` 가 학습 루프 안에서 `log()` 를 부를 때마다 등록된 모든 트래커에 같은 dict(`{"loss": 0.42, "step": 100, ...}`) 를 보냅니다. 트래커는 그걸 받아 자기 방식대로 시각화·저장합니다.""")

# ----- 3. install -----
code(r"""!pip install -q transformers datasets wandb trackio mlflow""")

# ----- 4. 데이터·모델 준비 (Ch 9 압축) -----
md(r"""## 2. 데이터와 모델 — Ch 9 골격 그대로

본 챕터의 학습 셋업을 압축해 한 셀에 다시 만듭니다. 이번엔 train 1,000 / eval 200으로 더 줄여 두 번 학습해도 시간이 짧게 끝나도록.""")

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
train_ds = ds["train"].shuffle(seed=42).select(range(1000))
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

# ----- 5. wandb 도입 -----
md(r"""## 3. Weights & Biases (wandb) 실습

### 셋업

wandb는 SaaS 서비스라 *계정 + API key* 가 필요합니다.

1. <https://wandb.ai/site> 에서 무료 계정 생성
2. <https://wandb.ai/authorize> 에서 API key 복사
3. 아래 셀에서 `wandb.login()` 입력창에 붙여넣기

**계정 생성을 건너뛰고 싶다면** `WANDB_MODE=offline` 환경변수로 *로컬 디스크에만 저장* 하는 방법도 있습니다 — 실제 dashboard는 못 보지만 통합 자체를 검증하려면 충분합니다.""")

code(r"""import wandb, os

# 옵션 A — 온라인 (계정 + API key 필요): 아래 한 줄을 활성화
# wandb.login()

# 옵션 B — 오프라인 (로컬 디스크에 저장만): 위는 주석으로 두고 아래 환경변수 설정
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "neuqes-101-ch09"

print(f"WANDB_MODE: {os.environ.get('WANDB_MODE')}")""")

# ----- 6. wandb 학습 -----
md(r"""### `report_to="wandb"` 로 학습 한 번

`TrainingArguments` 한 인자만 바꾸면 됩니다. 학습 step마다 wandb가 loss·lr 을 자동 기록하고, 에폭 끝마다 `eval_mse / eval_mae / eval_r2` 도 함께 보냅니다.""")

code(r"""# 모델 새로 (이전 학습 상태가 안 남게)
model_wb = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1, problem_type="regression",
)

args_wb = TrainingArguments(
    output_dir="./wb_out",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=True,
    eval_strategy="epoch",
    logging_steps=10,
    save_strategy="no",
    report_to="wandb",                    # ← 이 한 줄
    run_name="ch09-appendix-wandb",       # dashboard에서 보이는 이름
    seed=42,
)

trainer_wb = Trainer(
    model=model_wb,
    args=args_wb,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer_wb.train()
trainer_wb.evaluate()
wandb.finish()   # 명시적으로 종료해야 다음 학습이 새 run으로 잡힘

print("\n학습 완료. WANDB_MODE=offline 이면 ./wandb/offline-run-*/ 에 결과 저장.")
print("dashboard 보려면 'wandb sync ./wandb/offline-run-*' 명령으로 업로드.")""")

# ----- 7. wandb 결과 안내 -----
md(r"""**무엇을 볼 수 있나** (온라인 모드 기준):

- 학습 곡선 (loss, lr, epoch)
- 평가 metric (mse / mae / r2) 시간순 그래프
- 사용된 하이퍼파라미터 전체 표
- GPU 사용률·메모리·온도 (nvidia-smi 데이터를 자동 수집)
- 시스템 정보 (CUDA 버전, 라이브러리 버전, 코드 git 커밋)

**여러 run 비교** — 같은 프로젝트에 학습 여러 번 실행하면 dashboard에서 모든 run을 한 화면에 겹쳐 봅니다. 학습률·배치 크기를 바꿔가며 grid search 할 때 가장 큰 가치.

**무료 tier 한도**: 100 GB 저장, 무제한 public projects (private 은 유료). 개인 프로젝트엔 충분.""")

# ----- 8. trackio 도입 -----
md(r"""## 4. trackio 실습

trackio는 Hugging Face가 만든 **가벼운 자체 호스팅 트래커** 입니다. 특징:

- 외부 계정 *불필요* — 로컬 SQLite + Gradio dashboard
- HF Spaces에 배포해 *팀 공유* 가능
- API가 wandb와 거의 같음 (`init` / `log` / `finish`)
- `Trainer` 통합도 동일한 `report_to="trackio"` 한 줄

### `report_to="trackio"` 로 학습""")

code(r"""# 모델 새로 (앞 wandb run과 분리)
model_tk = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1, problem_type="regression",
)

args_tk = TrainingArguments(
    output_dir="./tk_out",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=True,
    eval_strategy="epoch",
    logging_steps=10,
    save_strategy="no",
    report_to="trackio",                  # ← wandb 자리에 trackio
    run_name="ch09-appendix-trackio",
    seed=42,
)

trainer_tk = Trainer(
    model=model_tk,
    args=args_tk,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer_tk.train()
trainer_tk.evaluate()""")

# ----- 9. trackio dashboard -----
md(r"""### dashboard 띄우기

trackio는 Gradio 기반 UI를 한 줄로 띄울 수 있습니다.

```python
import trackio
trackio.show()   # Colab에서 inline iframe으로 dashboard 표시
```

서버를 띄우면 학습 곡선·metric·하이퍼파라미터를 wandb와 비슷한 형태로 볼 수 있습니다. SQLite 파일 (`./trackio.db` 또는 사용자 지정 경로)에 모든 run이 누적되므로, **세션이 끝나도 다시 띄우면 과거 run이 보존**됩니다.""")

code(r"""import trackio
trackio.show()""")

# ----- 10. MLflow 도입 -----
md(r"""## 5. MLflow 실습

MLflow는 **회사 내부 인프라에서 가장 흔히 쓰는** 표준입니다. 자체 서버에 띄워 운영하는 게 일반적이고, *experiment tracking* 외에 *model registry*, *deployment* 까지 한 묶음으로 제공합니다. 이 부록에서는 트래킹 부분만 봅니다.

### 셋업 — 로컬 MLflow

MLflow는 로컬 디렉터리(`./mlruns/`)에 SQLite + 파일 기반으로 저장합니다. 외부 계정 불필요.""")

code(r"""import mlflow

# 실험(experiment) 이름 — 비슷한 run을 묶는 단위
mlflow.set_experiment("ch09-appendix-mlflow")

# 저장 위치 (기본 ./mlruns/, 다른 경로로 바꿀 수도 있음)
print(f"tracking URI:  {mlflow.get_tracking_uri()}")
print(f"experiment:    {mlflow.get_experiment_by_name('ch09-appendix-mlflow').name}")""")

# ----- 10b. MLflow 학습 -----
md(r"""### `report_to="mlflow"` 로 학습

다른 트래커와 같은 패턴 — `TrainingArguments(report_to="mlflow")` 한 줄.""")

code(r"""# 모델 새로
model_mf = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1, problem_type="regression",
)

args_mf = TrainingArguments(
    output_dir="./mf_out",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=True,
    eval_strategy="epoch",
    logging_steps=10,
    save_strategy="no",
    report_to="mlflow",                   # ← MLflow 활성
    run_name="ch09-appendix-mlflow",
    seed=42,
)

trainer_mf = Trainer(
    model=model_mf,
    args=args_mf,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer_mf.train()
trainer_mf.evaluate()

print("\n학습 완료. ./mlruns/ 디렉터리에 run이 저장됐습니다.")""")

# ----- 10c. MLflow dashboard -----
md(r"""### dashboard 띄우기

MLflow는 `mlflow ui` 명령으로 웹 dashboard를 띄웁니다. Colab에서는 `pyngrok` 으로 외부 접근 가능한 URL을 잡거나 *디렉터리만 다운로드해 로컬에서 실행* 하는 방식이 흔합니다.

```bash
# 로컬에서:
mlflow ui --backend-store-uri ./mlruns --port 5000
# → http://localhost:5000 에서 dashboard 보기
```

또는 Python으로 직접 run 정보를 조회할 수도 있습니다.""")

code(r"""# 가장 최근 run의 metric을 직접 조회
runs = mlflow.search_runs(experiment_names=["ch09-appendix-mlflow"], order_by=["start_time DESC"])
print(f"run 수: {len(runs)}")
print(f"\n앞 run의 주요 metric:")
if len(runs) > 0:
    latest = runs.iloc[0]
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    for col in metric_cols[:8]:
        print(f"  {col.replace('metrics.', ''):>20}: {latest[col]}")""")

# ----- 11. 세 도구 비교 -----
md(r"""## 6. wandb vs trackio vs MLflow — 비교 표

| 측면 | wandb | trackio | MLflow |
|---|---|---|---|
| 호스팅 | SaaS (cloud) | 자체 호스팅 (로컬 SQLite + Gradio) | 자체 호스팅 (로컬 또는 사내 서버) |
| 가입·로그인 | 필요 (무료 tier 있음) | 불필요 | 불필요 |
| 공유 | URL 한 번에 (private은 유료) | HF Spaces 배포 시 가능 | 사내 서버 띄우면 팀 공유 |
| 시각화 깊이 | 매우 풍부 (system metrics, artifacts, sweeps) | 학습 곡선·metric 위주, 단순함 | 풍부 (artifacts, model registry 포함) |
| 추가 기능 | sweeps (HPO), reports, alerts | 단순 추적만 | model registry, deployment, projects |
| 데이터 보관 | wandb cloud (영구) | 사용자 디스크 | 사용자 디스크 또는 사내 DB |
| `report_to` 값 | `"wandb"` | `"trackio"` | `"mlflow"` |
| 외부 의존 | 인터넷 + wandb 서버 | 없음 (오프라인 가능) | 없음 (자체 서버) |
| 적합한 상황 | 팀 협업, 장기 실험 추적, 하이퍼파라미터 sweep | 개인 프로젝트, 빠른 시작 | 회사 내부 인프라, model registry 필요 시 |

### 여러 트래커 동시에

`report_to` 를 리스트로 넘기면 한 학습을 여러 곳에 동시에 기록할 수 있습니다.

```python
TrainingArguments(..., report_to=["wandb", "mlflow"])
```

### `report_to` 외 다른 통합

`Trainer` 는 다음도 지원합니다.

- `"comet_ml"` — Comet ML
- `"neptune"` — Neptune.ai
- `"clearml"` — ClearML
- `"dvclive"` — DVC
- `"tensorboard"` — TensorBoard (이번 부록에서는 다루지 않음)

대부분 인터페이스가 비슷합니다. 라이브러리 설치, 환경변수 또는 login, `report_to` 값만 바꾸면 끝납니다.""")

# ----- 12. 실무 가이드 -----
md(r"""## 7. 실무 가이드 — 어떤 상황에 무엇을 쓰나

### 빠른 개인 프로젝트
- **trackio**: 외부 계정 없이 즉시 시작, 학습 한두 번 비교용으로 가볍게.

### 본격적인 ML 프로젝트 (학생·연구자·개인 개발자)
- **wandb**: 팀과 결과 공유, 하이퍼파라미터 sweep, artifact 추적, 장기 보관.

### 회사 내부 인프라
- **MLflow**: 자체 서버에 띄워서 운영. model registry·CI/CD·deployment 까지 한 묶음으로 관리.

### Ch 9 본 노트북에서는 왜 `report_to="none"` 인가
이 커리큘럼이 외부 계정·서비스 없이 끝까지 돌아가게 만들기 위해서입니다. 이번 부록에서 한 번 도구를 붙여보고 자기 워크플로에 맞춰 선택하시는 게 좋겠습니다.""")

# ----- 13. 정리 -----
md(r"""## 정리

- `Trainer` + `TrainingArguments(report_to=...)` 한 줄이 어떤 트래커든 연결의 입구입니다.
- **wandb**: SaaS, 풍부한 기능, 팀 협업, sweep 내장.
- **trackio**: 로컬, 가벼움, 외부 의존 없음.
- **MLflow**: 자체 서버, model registry까지 묶음, 회사 인프라 표준.
- 본 챕터(Ch 9)와 이후 학습 챕터는 외부 의존을 피하기 위해 `report_to="none"` 으로 둡니다. 실무로 가져갈 때 위 셋 중 하나(또는 다른 통합)를 골라 붙이면 됩니다.

**Ch 9 본 흐름으로 돌아가기**: [09_bert_regression.ipynb](./09_bert_regression.ipynb)

**관련 부록**: [`appendix_hpo.ipynb`](./appendix_hpo.ipynb) — 하이퍼파라미터 최적화의 어려움과 wandb sweeps · Optuna 활용

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
