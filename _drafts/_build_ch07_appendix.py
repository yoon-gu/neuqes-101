"""Build 07_bert_pipeline/appendix_model_config.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "07_bert_pipeline" / "appendix_model_config.ipynb"

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
md(r"""# Ch 7 부록 — `model.config` 깊이 보기

본 챕터(Ch 7)에서 `model.config` 의 핵심 속성을 짧게 짚었습니다. 이 부록 노트북은 그 자리에서 *더 궁금해진* 사람들을 위한 한 단계 깊은 안내입니다 — 커리큘럼 본 흐름과 분리되어 있으니 시간 될 때 보시면 됩니다.

**무엇을 다루나**

1. `PretrainedConfig` 의 정체 — 모든 모델 config의 공통 부모
2. 가중치 없이 config만 로드하기 (`AutoConfig.from_pretrained`)
3. 다양한 모델 아키텍처의 config를 한 표에 비교 — encoder / decoder / encoder-decoder / vision
4. config 수정 → 모델 재구성 패턴 (분류 헤드 갈아끼우기 등)
5. config 저장·직렬화 (`to_dict`, `save_pretrained`)

**공식 문서**: <https://huggingface.co/docs/transformers/en/main_classes/configuration>

**환경**: Colab CPU도 OK (이번 부록은 추론·학습 없이 config 파일만 다운로드, 각각 수 KB).

**예상 시간**: 약 15분

> Ch 7 본 흐름으로 돌아가기: [07_bert_pipeline.ipynb](./07_bert_pipeline.ipynb)""")

# ----- 2. PretrainedConfig 도입 -----
md(r"""## 1. `PretrainedConfig` — 공통 부모

`transformers` 의 모든 모델 config는 **`PretrainedConfig`** 를 상속합니다.

```
PretrainedConfig (공통 부모)
├── BertConfig
├── DistilBertConfig
├── GPT2Config
├── RobertaConfig
├── T5Config
├── ViTConfig         ← 비전 모델
├── WhisperConfig     ← 음성
└── … 수백 개
```

각 자식 클래스는 모델 아키텍처에 특화된 속성을 *추가* 하고, 공통 속성(예: `hidden_size`, `vocab_size`, `id2label`)은 부모에서 *상속* 받습니다.

**실무에서는 거의 항상 `AutoConfig` 를 씁니다** — 모델 이름만 주면 적합한 클래스를 자동 선택해 인스턴스를 만듭니다.""")

# ----- 3. install -----
code(r"""!pip install -q transformers""")

# ----- 4. AutoConfig 사용 -----
md(r"""## 2. 가중치 없이 config만 로드하기

`AutoConfig.from_pretrained(...)` 는 **모델 가중치 없이 `config.json` 만 다운로드** 합니다 (수 KB). "이 모델이 어떤 구조인지"만 빠르게 보고 싶을 때 편리합니다 — 모델 비교, 호환성 확인, 분류 헤드 미리 설계 등.""")

code(r"""from transformers import AutoConfig

# 가중치 없이 config만 — config.json만 받음
cfg = AutoConfig.from_pretrained("bert-base-uncased")

print(f"클래스:        {type(cfg).__name__}")
print(f"부모 클래스:   {type(cfg).__mro__[1].__name__}")
print(f"model_type:    {cfg.model_type}")
print(f"hidden_size:   {cfg.hidden_size}")
print(f"vocab_size:    {cfg.vocab_size:,}")
print(f"layers:        {cfg.num_hidden_layers}, heads: {cfg.num_attention_heads}")""")

# ----- 5. 다양한 모델 비교 도입 -----
md(r"""## 3. 다양한 모델 아키텍처의 config 한 표 비교

대표적인 모델 5종을 한 번에 가져와 핵심 속성을 표로 봅니다.

| 모델 | 종류 | 사전학습 task |
|---|---|---|
| `bert-base-uncased` | 인코더 전용 | MLM + NSP |
| `distilbert-base-uncased` | 인코더 (경량화) | DistilBERT distillation |
| `gpt2` | 디코더 전용 (autoregressive) | LM (다음 토큰) |
| `t5-small` | 인코더-디코더 (seq2seq) | text-to-text |
| `roberta-base` | 인코더 (BERT 개선) | MLM (NSP 제거) |

**주의**: 같은 개념이라도 **속성 이름이 모델마다 다릅니다** — 예를 들어 hidden state 차원은 BERT/RoBERTa의 `hidden_size`, GPT-2의 `n_embd`, T5의 `d_model` 입니다. 비교할 때 헬퍼로 흡수합니다.""")

# ----- 6. 비교 코드 -----
code(r"""import pandas as pd

MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "gpt2",
    "t5-small",
    "roberta-base",
]

configs = {name: AutoConfig.from_pretrained(name) for name in MODELS}

def attr(c, *names, default="—"):
    # 모델마다 이름 다른 동일 속성을 흡수
    for n in names:
        if hasattr(c, n) and getattr(c, n) is not None:
            return getattr(c, n)
    return default

rows = []
for name, c in configs.items():
    rows.append({
        "model": name.split("/")[-1],
        "model_type": c.model_type,
        "config_class": type(c).__name__,
        "hidden_size": attr(c, "hidden_size", "d_model", "n_embd"),
        "n_layers": attr(c, "num_hidden_layers", "num_layers", "n_layer"),
        "n_heads": attr(c, "num_attention_heads", "num_heads", "n_head"),
        "vocab_size": c.vocab_size,
        "max_pos": attr(c, "max_position_embeddings", "n_positions"),
    })

pd.DataFrame(rows)""")

# ----- 7. 관찰 -----
md(r"""**관찰 포인트**

- **이름이 다른 같은 개념**:
  - hidden state 차원: BERT/RoBERTa `hidden_size` · T5 `d_model` · GPT-2 `n_embd`
  - layer 수: BERT `num_hidden_layers` · T5 `num_layers` · GPT-2 `n_layer`
  - max position: BERT `max_position_embeddings` · GPT-2 `n_positions`
  → 모델 카드 만들 때 *어떤 모델이든* 쓰일 헬퍼를 만들면 편합니다 (위 `attr()` 형태).
- **vocab 크기**: BERT 30K · GPT-2/RoBERTa 50K · T5 32K — 사전학습 데이터·토크나이저 알고리즘에 따라.
- **레이어 수**: distilled는 6, base는 12, large는 24 (BERT 계열).
- **max position**: BERT 계열 512가 가장 흔한 상한. 더 길게 쓰려면 모델을 확장(예: Longformer)하거나 sliding window 등 트릭 필요.""")

# ----- 8. T5 추가 속성 -----
md(r"""### T5 — 인코더-디코더의 추가 속성

T5는 **인코더-디코더** 구조라 인코더 layer와 디코더 layer가 따로 있습니다. 또 FFN 내부 차원(`d_ff`)도 별도로 노출됩니다.""")

code(r"""t5_cfg = configs["t5-small"]

print(f"d_model:             {t5_cfg.d_model}            (hidden state 차원)")
print(f"d_ff:                {t5_cfg.d_ff}             (FFN 내부 차원)")
print(f"num_layers:          {t5_cfg.num_layers}             (인코더 layer 수)")
print(f"num_decoder_layers:  {t5_cfg.num_decoder_layers}  (디코더 layer 수)")
print(f"num_heads:           {t5_cfg.num_heads}")
print(f"is_encoder_decoder:  {t5_cfg.is_encoder_decoder}    (← BERT/GPT-2 모두 False)")""")

# ----- 9. ViT 도입 -----
md(r"""### ViT — 비전 트랜스포머는 vocab 자리에 image_size

같은 `PretrainedConfig` 기반이지만 입력 도메인이 다른 ViT는 vocab 대신 *이미지 차원* 을 갖습니다.""")

code(r"""vit_cfg = AutoConfig.from_pretrained("google/vit-base-patch16-224")

print(f"클래스:           {type(vit_cfg).__name__}")
print(f"model_type:       {vit_cfg.model_type}")
print(f"image_size:       {vit_cfg.image_size}    (입력 이미지 한 변, 픽셀)")
print(f"patch_size:       {vit_cfg.patch_size}     (이미지 → 패치 한 변)")
print(f"num_channels:     {vit_cfg.num_channels}      (RGB)")
print(f"hidden_size:      {vit_cfg.hidden_size}")
print(f"num_hidden_layers: {vit_cfg.num_hidden_layers}")
print()
print(f"hasattr(vocab_size)?  {hasattr(vit_cfg, 'vocab_size')}    (← 텍스트 모델만 vocab을 가짐)")""")

# ----- 10. 도메인별 차이 정리 -----
md(r"""**도메인별 속성 차이 정리**

| 모델 종류 | 입력 표현 속성 | 시퀀스 길이 속성 |
|---|---|---|
| 텍스트 인코더 (BERT/RoBERTa) | `vocab_size`, `hidden_size` | `max_position_embeddings` |
| 텍스트 디코더 (GPT-2) | `vocab_size`, `n_embd` | `n_positions` |
| 인코더-디코더 (T5) | `vocab_size`, `d_model` | (`max_position_embeddings` 명시 X — relative position 사용) |
| 비전 (ViT) | `image_size`, `patch_size`, `num_channels`, `hidden_size` | (이미지는 길이 개념 X) |
| 음성 (Whisper) | `num_mel_bins`, `d_model` | `max_source_positions`, `max_target_positions` |

같은 `PretrainedConfig` 기반이지만 *그 모델이 어떤 입력을 받느냐* 가 속성에 그대로 반영됩니다.""")

# ----- 11. config 수정 도입 -----
md(r"""## 4. config 수정 → 모델 재구성

Fine-tuning에서 가장 흔한 패턴: **사전학습 가중치는 그대로 받되, 분류 헤드를 우리 task에 맞게 갈아끼움.** `from_pretrained` 인자가 config를 자동으로 덮어쓰는 방식입니다.""")

code(r"""from transformers import AutoModelForSequenceClassification

# 5클래스 분류로 갈아끼우기
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,                                          # config.num_labels = 5
    id2label={i: f"class_{i}" for i in range(5)},
    label2id={f"class_{i}": i for i in range(5)},
    problem_type="single_label_classification",
)

print(f"수정된 num_labels:   {model.config.num_labels}")
print(f"수정된 id2label:     {model.config.id2label}")
print(f"수정된 problem_type: {model.config.problem_type}")
print()
print(f"분류 헤드 shape:    {model.classifier.weight.shape}  (5 × 768 — 새로 랜덤 초기화)")""")

# ----- 12. 무슨 일이 일어났나 -----
md(r"""**무엇이 일어났나**

1. `bert-base-uncased` 의 *가중치* 와 *기존 config* 가 모두 다운로드됨.
2. 우리가 준 인자(`num_labels=5`, `id2label`, ...)가 **config에 덮어씀**.
3. 분류 헤드(`Linear(768, 2)`)가 새 차원 `Linear(768, 5)` 으로 *랜덤 초기화* 되어 다시 만들어짐 — 이 부분만 학습으로 채워짐.
4. BERT 본체의 가중치는 사전학습에서 받은 그대로 남음 (transfer learning의 핵심).

`from_pretrained` 가 출력하는 경고를 자세히 보면 "Some weights of BertForSequenceClassification were not initialized from the model checkpoint... You should probably TRAIN this model on a down-stream task" 라는 메시지가 보입니다 — 분류 헤드가 새로 만들어졌다는 알림입니다.

### `Trainer` 가 자동으로 loss를 결정하는 매핑

`problem_type` 이 결정합니다.

| `problem_type` | 자동 적용 loss | 라벨 형식 |
|---|---|---|
| `"regression"` | `MSELoss` | float (1차원) |
| `"single_label_classification"` | `CrossEntropyLoss` | int 인덱스 |
| `"multi_label_classification"` | `BCEWithLogitsLoss` | multi-hot float |

명시 안 하면 `num_labels` 와 라벨 dtype에서 자동 추론. 이 매핑은 Ch 9·11·12에서 직접 사용합니다.""")

# ----- 13. config 저장 / 직렬화 -----
md(r"""## 5. config 저장과 직렬화

학습된 모델을 어디에 저장하든 `config.json` 이 함께 들어갑니다 — 다른 사람이 같은 모델을 재구성할 수 있게.""")

code(r"""# dict 형태로 (모든 설정을 dict 객체로)
cfg_dict = model.config.to_dict()
print(f"dict 키 개수: {len(cfg_dict)}")
print(f"num_labels: {cfg_dict['num_labels']}, problem_type: {cfg_dict['problem_type']}")

# JSON 문자열 (직렬화 가능)
json_str = model.config.to_json_string()
print(f"\nJSON 첫 200자:\n{json_str[:200]}...")

# 모델 + config 모두 디렉터리에 저장
import os
out_dir = "./tmp_model"
model.save_pretrained(out_dir)
print(f"\n저장된 파일들:")
for f in sorted(os.listdir(out_dir)):
    size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
    print(f"  {f}  ({size_kb:.1f} KB)")""")

# ----- 14. 다시 로드 -----
code(r"""# config.json 한 파일로 다시 모델 구성 가능
reloaded_config = AutoConfig.from_pretrained("./tmp_model")
print(f"reload 후 num_labels:    {reloaded_config.num_labels}")
print(f"reload 후 id2label:      {reloaded_config.id2label}")

# 정리
import shutil
shutil.rmtree("./tmp_model")
print("\n임시 디렉터리 정리 완료")""")

# ----- 15. 더 보기 -----
md(r"""## 6. 더 깊이 보고 싶다면

이 부록은 핵심만 짚었습니다. 다음 자료를 추천합니다.

- 📘 **공식 문서**: [Configuration — `transformers`](https://huggingface.co/docs/transformers/en/main_classes/configuration)
  - `PretrainedConfig` API 전체
  - 모델별 Config 클래스 (`BertConfig`, `GPT2Config`, ...) 속성 일람
- 📗 **각 모델 카드** (예: <https://huggingface.co/bert-base-uncased>): 사전학습 데이터, 학습 하이퍼파라미터, 기대 사용 사례
- 📕 **Hugging Face Models Hub** (<https://huggingface.co/models>): 30만+ 모델, 카테고리별 검색

### 정리

- 모든 모델 config는 `PretrainedConfig` 를 상속하고 `AutoConfig.from_pretrained` 로 로드.
- 모델 종류(인코더/디코더/seq2seq/비전/음성)에 따라 추가 속성이 다르지만, *공통 골격* 은 같음.
- Fine-tuning 시 `from_pretrained` 인자가 config를 자동 덮어써 분류 헤드를 갈아끼움 — 사전학습 가중치는 보존, 헤드만 랜덤 초기화.
- `problem_type` 이 `Trainer` 의 자동 loss 선택을 결정.

**Ch 7 본 흐름으로 돌아가기**: [07_bert_pipeline.ipynb](./07_bert_pipeline.ipynb)

**다음 챕터**: [Ch 8 — Tokenizer 깊게 보기 + Datasets 라이브러리](../08_tokenizer_datasets/)""")


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
