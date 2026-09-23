"""Build 32_diffusion_intro/32_diffusion_intro.ipynb — Phase 5 첫 챕터.

작은 mask-diffusion 언어모델 직접 구현 + TinyStories + parallel-denoise generation.
Phase 4 (Ch 24-31) 의 *autoregressive (decoder, next-token, 왼→오 순차 생성)* 패러다임에서
Phase 5 (Ch 32-34) 의 *diffusion (encoder/bidirectional, masked-denoise, 문장 전체 병렬 생성)*
패러다임으로 전환하는 출발점. BERT MLM (Ch 20-23) 의 *고정 15% 마스킹* 을
*0-100% 가변 마스킹 + 반복 denoise* 로 일반화하면 generation 이 된다는 것을 직접 구현해 봅니다.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "32_diffusion_intro"
OUT_NB = OUT_DIR / "32_diffusion_intro.ipynb"
OUT_README = OUT_DIR / "README.md"

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


# ----- 1. 제목 -----
md(r"""# Chapter 32. Diffusion LM — 작은 mask-diffusion 모델 직접 구현

**목표**: Phase 5 의 첫 챕터. Ch 24-31 까지 다룬 **GPT (decoder, autoregressive, 왼→오 순차 생성)** 패러다임에서, 이번엔 **Diffusion LM (encoder/bidirectional, masked-denoise, 문장 전체를 병렬로 생성)** 패러다임으로 전환합니다.

핵심 한 줄: **BERT MLM (Ch 20-23) 의 *고정 15% 마스킹* 을 *0-100% 가변 마스킹* 으로 일반화하고, 한 번에 복원하는 대신 *여러 번 반복 denoise* 하면 그게 generation 입니다.** Ch 1 부터 추적해 온 *마스킹 + 토크나이저* 시각이 여기서 클라이맥스에 도달합니다 — 가려서 맞히던 BERT 가, 가리는 비율을 끝까지 밀어붙이면 *무에서 문장을 만들어내는 생성 모델* 이 됩니다.

작은 BERT-style 모델을 *random init* 으로 from scratch 띄우고, **TinyStories** 로 *가변 마스킹 denoising* 목표로 학습 → reverse process (전부 `[MASK]` 에서 시작해 반복 denoise) 로 텍스트를 *왼→오가 아닌 병렬* 로 생성하는 과정을 직접 눈으로 봅니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 25-30분 (데이터 로드 약 2분 + 토큰화 약 3분 + 학습 전 denoise 약 30초 + 모델 학습 약 15분 + 학습 후 denoise + 궤적 시각화 + AR 비교 약 3분)

---

## 학습 흐름

1. 📊 **변화 추적표 + Phase 전환 도입부** — Autoregressive (GPT) → Diffusion 큰 그림 한 화면
2. 🔄 **변경점** — 생성 방식 (순차 → 병렬 denoise), attention (causal → bidirectional), 마스킹 (고정 15% → 가변 0-100%)
3. 📐 **Loss** — masked-diffusion denoising loss. MLM 의 CE 를 *가변 마스킹 비율 t* 로 일반화 + `1/t` 재가중
4. 💡 **마스킹 thread 클라이맥스** — BERT 의 *고정 15%* vs diffusion 의 *가변 0-100%*. 같은 `-100` 트릭
5. 🔤 **토크나이저 노트** — WordPiece (`bert-base-uncased` 가져옴), `[MASK]` 토큰이 주인공이 되는 순간
6. 🚀 **실습**: TinyStories → 작은 BERT-style 모델 → 가변 마스킹 denoising 학습
7. 🔬 **Reverse process generation** — 전부 `[MASK]` 에서 반복 denoise. 마스크가 *병렬로* 단어로 채워지는 궤적 직접 관찰
8. 🛠️ **변형**: denoise step 수 비교 (1 / 4 / 16 / 32), 조건부 생성 (prompt 고정)
9. ⚖️ **AR vs Diffusion 비교** — Ch 24 (GPT) 와 나란히. Ch 33-34 예고
10. 📦 **등장 라이브러리** / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)

---

> 📒 **사전 학습 자료**: Ch 20-23 (BERT MLM 사전학습 — 고정 15% 마스킹), Ch 24 (GPT from scratch — autoregressive generation). 본 챕터는 *둘을 잇습니다* — BERT 의 *마스킹-복원* 메커니즘을 Ch 24 의 *generation* 목적에 다시 씁니다. 다른 점은 *마스킹 비율을 0-100% 로 일반화* 하고 *복원을 여러 번 반복* 한다는 것뿐.""")

# ----- 2. 변화추적표 + Phase 5 도입 -----
md(r"""## 📊 변화 추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | 생성/학습 방식 | Loss |
|---|---|---|---|---|---|---|
| 20 | 작은 BERT (영어, scratch) | `bert-base-uncased` (가져옴) | Wikitext-103 | MLM head | 고정 15% 마스킹-복원 | `CrossEntropyLoss` (masked 15%) |
| 24 | 작은 GPT2 (직접, scratch) | BPE (직접 학습) | TinyStories | `Linear(H, V)` | autoregressive (왼→오 순차) | `CrossEntropyLoss` (next-token) |
| 31 | Qwen2.5-0.5B-Instruct + GRPO | BBPE (Qwen2.5) | verifiable-reward | `Linear(H, V)` + group adv. | autoregressive + RL | `GRPO loss` |
| **32 ← 여기** | **작은 BERT-style (직접, scratch)** | **WordPiece (`bert-base-uncased` 가져옴)** | **TinyStories** | **`Linear(H, V)`** | **parallel denoise (가변 마스킹 + 반복 복원)** | **masked-diffusion denoising loss (`1/t` 재가중)** |
| 33 (다음) | MDLM (170M) / DiffuGPT (124M) 사전학습 | (각 모델 토크나이저) | 영어 사전학습 추론 시연 | `Linear(H, V)` | parallel denoise (추론만) | — |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.

---

## Phase 전환 — Autoregressive (GPT) → Diffusion LM

Ch 24-31 의 GPT 챕터들이 *decoder + next-token 예측 + 왼→오 순차 생성* 패러다임이라면, Phase 5 (Ch 32-34) 는 *encoder/bidirectional + masked-denoise + 문장 전체 병렬 생성* 패러다임입니다. 본 챕터가 그 출발점.

| 축 | Phase 4 (GPT, Ch 24-31) | **Phase 5 (Diffusion, Ch 32-34)** |
|---|---|---|
| attention | Causal (과거만 봄) | **Bidirectional (양방향 다 봄)** |
| 학습 목표 | next-token 예측 | **가변 마스킹 denoising** |
| 생성 순서 | 왼→오 *한 토큰씩 순차* | **문장 전체를 *동시에* 여러 번 denoise** |
| 생성 step 수 | 토큰 수 = step 수 (길면 느림) | **step 수를 *자유롭게 조절* (4 / 16 / 32 ...)** |
| 출발 상태 | prompt 토큰들 | **전부 `[MASK]` (무에서 시작)** |
| 본체 계보 | GPT (Ch 24) | **BERT (Ch 20)** — MLM 을 일반화 |

> **핵심 직관**: GPT 가 *왼쪽부터 한 글자씩 받아쓰기* 라면, diffusion 은 *흐릿한 전체 그림을 여러 번 선명하게 다듬기* 입니다. 이미지 생성에서 노이즈를 점점 걷어내듯, 텍스트에서는 `[MASK]` 를 점점 진짜 단어로 바꿔 갑니다. 본 챕터는 그 메커니즘을 *작은 모델로 직접 구현* 해 봅니다. Ch 33 (MDLM 170M / DiffuGPT 124M 사전학습) 이 *같은 원리의, 충분한 규모로 학습된 실전 모델* 입니다.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 31)

| 축 | Ch 24-31 (GPT, autoregressive) | Ch 32 (Diffusion LM) |
|---|---|---|
| **생성 방식** | next-token, 왼→오 *순차* | **masked-denoise, 문장 전체 *병렬*** ← *Phase 전환의 핵심* |
| attention | Causal (`GPT2LMHeadModel`) | **Bidirectional (`BertForMaskedLM` 계열)** |
| 학습 목표 | `CrossEntropyLoss` (next-token, 거의 모든 자리) | **masked-diffusion loss (가변 비율 `t` 마스킹 + `1/t` 재가중)** |
| 마스킹 | 없음 (causal mask 가 미래 차단) | **입력 토큰을 `t` 비율로 `[MASK]` 치환** |
| 토크나이저 | BPE (GPT 계열) | **WordPiece (`bert-base-uncased`, `[MASK]` 토큰 보유)** |
| 생성 출발 | prompt 토큰 | **전부 `[MASK]` 인 시퀀스** |
| 생성 step | 토큰 길이 만큼 | **임의 step 수 (속도-품질 trade-off 조절 가능)** |

> **변경점이 한꺼번에 많은 이유** — Phase 가 바뀌는 *전환 챕터* 라 *축 자체* 가 새로 정의됩니다. 하지만 본질은 *Ch 20 의 BERT MLM 을 재활용* 한 것 — *bidirectional + 마스킹-복원* 은 이미 다 배운 메커니즘이고, *마스킹 비율을 가변* 으로 만들고 *복원을 반복* 한 것만 새롭습니다. Ch 33 부터는 다시 *모델 출발점* 만 바뀝니다 (Ch 33: scratch → 사전학습 MDLM/DiffuGPT / Ch 34: 한국어 diffusion).""")

# ----- 4. Loss 노트 -----
md(r"""## 📐 Loss — masked-diffusion denoising loss

BERT MLM 의 CrossEntropyLoss 와 *뼈대는 같습니다* — 가려진 자리의 정답 토큰을 맞히는 CE. 다른 점은 두 가지:

1. 마스킹 비율이 *고정 15%* 가 아니라 *매 샘플마다 $t \sim U(0, 1)$ 로 뽑은 가변 비율*
2. 비율 $t$ 만큼 가렸으니, loss 를 *$1/t$ 로 재가중* 해 *어떤 마스킹 비율이든 공정하게* 평균

### 수식

깨끗한 토큰 시퀀스 $x_0 = (x_1, \dots, x_L)$ 에 대해, 비율 $t$ 를 뽑고 각 토큰을 *독립적으로 확률 $t$* 로 `[MASK]` 치환해 $x_t$ 를 만듭니다. 모델은 $x_t$ 전체를 보고 *가려진 자리* 의 원본 토큰을 예측:

$$L = \mathbb{E}_{t \sim U(0,1)} \left[ \frac{1}{t} \cdot \frac{1}{L} \sum_{i:\, x_t^{(i)} = \texttt{[MASK]}} -\log P_\theta\!\left(x_0^{(i)} \mid x_t\right) \right]$$

- $\sum_{i:\, x_t^{(i)}=\texttt{[MASK]}}$: *가려진 자리에서만* loss 계산 (Ch 20-23 의 `-100` 트릭과 동일)
- $1/t$ 재가중: $t$ 가 작으면 (조금 가림) 가려진 토큰이 적으니 합이 작아지는데, $1/t$ 로 곱해 *마스킹 비율에 무관* 하게 스케일을 맞춤. 이 재가중 덕분에 *학습 목표가 log-likelihood 의 upper bound* 가 됩니다 (LLaDA / MDLM 의 핵심 항)
- $t \sim U(0,1)$: 매 step 마다 *다른 난이도* 의 복원 문제를 풀게 함 — 5% 만 가린 쉬운 문제부터 95% 가린 거의-무에서-생성 문제까지

### 숫자로 감 잡기 (vocab = 30,522)

random init 직후 모델은 가려진 자리를 *균등 추측* → 정답 확률 $1/30522$, 토큰당 $-\log p \approx \ln(30522) = 10.33$.

| 마스킹 비율 $t$ | 가린 토큰 수 (L=128) | 가린 자리 합 ($\approx t L \times 10.33$) | $\times \frac{1}{t}\frac{1}{L}$ 후 | 해석 |
|---|---|---|---|---|
| 0.10 | 약 13 | 약 134 | **10.33** | 조금 가린 쉬운 복원 |
| 0.50 | 약 64 | 약 661 | **10.33** | 절반 가림 |
| 0.90 | 약 115 | 약 1188 | **10.33** | 거의 무에서 생성 |

**관전 포인트**:
- `1/t` 재가중 덕분에 *어떤 t 든 baseline loss 가 똑같이 `ln(vocab) ≈ 10.33`* 으로 정렬됩니다. Ch 20 MLM 의 random baseline `ln(30522) ≈ 10.33` 과 정확히 같은 값 — *마스킹 비율만 일반화* 했지 loss 의 척도는 그대로입니다.
- 학습 첫 step loss 가 약 10.3 부근에서 시작해 빠르게 떨어지면 정상. 목표는 약 4-6 영역 (작은 모델 + TinyStories).
- $t$ 가 1 에 가까울수록 (거의 다 가림) *문맥 정보가 거의 없어* 복원이 어려움 → diffusion 생성이 *여러 step 에 나눠* 조금씩 푸는 이유.""")

# ----- 5. 마스킹 thread 클라이맥스 -----
md(r"""## 💡 마스킹 thread 클라이맥스 — *고정 15%* (BERT) → *가변 0-100%* (diffusion)

Ch 1 부터 *토크나이저와 마스킹* 을 일관되게 추적해 왔습니다. 그 흐름이 여기서 정점에 닿습니다.

| 단계 | 챕터 | 마스킹 비율 | 복원 횟수 | 용도 |
|---|---|---|---|---|
| MLM 사전학습 | Ch 20 (영어), Ch 22 (한국어) | **고정 15%** | 1회 | 표현 학습 (downstream fine-tune 용) |
| GPT CausalLM | Ch 24-31 | 마스킹 없음 (causal mask) | — | autoregressive 생성 |
| **Mask-diffusion** | **Ch 32 (본 챕터)** | **가변 $t \sim U(0,1)$** | **반복 (4-32 step)** | **병렬 생성** |

핵심은 **셋이 모두 같은 `labels = -100` 트릭** 을 쓴다는 점입니다 — *가려진 자리만* loss 계산, 나머지는 `-100` 으로 무시. Ch 20 에서 손에 익힌 그 패턴이 그대로 재등장합니다.

> **"가린다" 의 의미가 바뀝니다.** BERT 에서 마스킹은 *표현을 배우기 위한 수단* (15% 만 살짝 가려 문맥으로 복원). Diffusion 에서 마스킹은 *생성 그 자체* — 100% 가린 `[MASK]` 시퀀스에서 출발해 한 step 씩 단어를 채우면, 그게 *무에서 문장을 만들어내는 것* 입니다. **같은 메커니즘, 다른 목적.** 가리는 비율을 끝까지 밀어붙였더니 *복원이 생성이 되었습니다.*

이 챕터에서 학습 collator 는 매 배치마다 $t$ 를 새로 뽑아 *가변 비율* 로 가립니다. Ch 20 의 `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` 가 *고정 15%* 였다면, 여기서는 *직접 만든 가변 collator* 가 *0-100% 를 매번 다르게* 가립니다.""")

# ----- 6. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트 — WordPiece (`bert-base-uncased` 가져옴)

Diffusion LM 의 주인공 토큰은 **`[MASK]`** 입니다. GPT 의 BPE 토크나이저 (`<|endoftext|>` 하나만 있는 미니멀 컨벤션) 와 달리, BERT 의 WordPiece 는 *`[MASK]` 토큰을 처음부터 내장* 하고 있습니다. 그래서 diffusion 에는 BERT 계열 토크나이저가 자연스럽습니다.

| 토크나이저 | `[MASK]` 토큰 | 등장 챕터 | diffusion 적합성 |
|---|---|---|---|
| WordLevel / WordPiece (직접 학습) | 추가 가능 | Ch 19 | — |
| **WordPiece (`bert-base-uncased`)** | **내장 (`[MASK]`, id 103)** | **Ch 20-23, 본 챕터** | **그대로 사용** |
| BPE (GPT-2) | 없음 (`<\|endoftext\|>` 만) | Ch 24-31 | 별도 추가 필요 |

본 챕터는 Ch 20·22 처럼 *토크나이저를 학습하지 않고* 표준 `bert-base-uncased` WordPiece (vocab 30,522) 를 그대로 가져와 씁니다. `[MASK]` 토큰이 *forward process (가리기)* 와 *reverse process (복원/생성)* 양쪽의 핵심.

> Ch 1 부터 추적한 *토크나이저 시각* 의 클라이맥스 — *같은 문장이 어떻게 토큰화되는가* 를 넘어, 이제 *`[MASK]` 토큰 자체가 생성의 캔버스* 가 됩니다. 전부 `[MASK]` 인 빈 캔버스에서 시작해 토큰을 채우는 게 diffusion 생성입니다.""")

# ----- 7. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업""")

code(r"""%pip install -q -U transformers tokenizers datasets accelerate""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import math
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# device 자동 감지 - Colab T4 / 로컬 MPS / CPU 모두 지원
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device     : cuda  ({device_name})")
    print(f"VRAM total : {vram_gib:.2f} GiB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("device     : mps  (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("device     : cpu  (training will be very slow - Colab T4 recommended)")

print(f"torch      : {torch.__version__}")

# 재현성
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# fp16 은 CUDA 에서만 (MPS 는 미지원, CPU 는 의미 없음)
USE_FP16 = (device.type == "cuda")
print(f"use fp16   : {USE_FP16}")""")

# ----- 8. 데이터 -----
md(r"""## 1. TinyStories 데이터 로드

Ch 24 (GPT) 와 *완전히 같은 데이터* — `roneneldan/TinyStories` (Eldan & Li 2023, arXiv:2305.07759). GPT-3.5 / GPT-4 가 *4세 어린이 어휘* 로 생성한 짧은 영어 동화. 어휘·문법이 단순해 작은 모델로도 의미 있는 생성이 가능합니다.

*데이터를 Ch 24 와 동일* 하게 둔 이유: 나중에 *같은 데이터에서 AR (Ch 24) vs Diffusion (본 챕터) 생성 방식만 다른* 비교를 하기 위함입니다.

학습 split 의 처음 **30,000 stories** 만 사용 (T4 30분 룰 안).""")

code(r"""from datasets import load_dataset

N_TRAIN = 30_000      # 더 길게 돌리려면 키우세요 (full 은 약 2.1M stories)
N_VAL   = 500

raw_train = load_dataset("roneneldan/TinyStories", split=f"train[:{N_TRAIN}]")
raw_val   = load_dataset("roneneldan/TinyStories", split=f"validation[:{N_VAL}]")
print("train:", raw_train)
print("val  :", raw_val)
print("\n=== sample story ===")
print(raw_train[0]["text"][:400])""")

# ----- 9. 토크나이저 -----
md(r"""## 2. WordPiece 토크나이저 가져오기 (`bert-base-uncased`)

Ch 20·22 처럼 표준 BERT 토크나이저를 그대로 로드합니다. 핵심은 `[MASK]` 토큰의 존재.""")

code(r"""from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(f"vocab_size : {tokenizer.vocab_size}")
print(f"[MASK]     : '{tokenizer.mask_token}'  id={tokenizer.mask_token_id}")
print(f"[PAD]      : '{tokenizer.pad_token}'  id={tokenizer.pad_token_id}")

print("\n=== encode/decode demo ===")
sample = "Once upon a time, a little rabbit went to the forest."
enc = tokenizer(sample, add_special_tokens=False)
print(f"input      : {sample}")
print(f"ids        : {enc['input_ids']}")
print(f"tokens     : {tokenizer.convert_ids_to_tokens(enc['input_ids'])}")
print(f"decode     : {tokenizer.decode(enc['input_ids'])}")

# [MASK] 가 섞인 시퀀스의 모습 - diffusion 의 출발 상태 미리보기
masked_demo = enc["input_ids"][:]
for i in [1, 3, 5, 7]:
    if i < len(masked_demo):
        masked_demo[i] = tokenizer.mask_token_id
print(f"\nmasked demo: {tokenizer.convert_ids_to_tokens(masked_demo)}")""")

md(r"""**관전 포인트** — `[MASK]` 가 섞인 시퀀스가 바로 diffusion 의 *중간 상태* $x_t$ 입니다. 학습은 *가려진 자리를 맞히는 것*, 생성은 *전부 `[MASK]` 에서 시작해 반복적으로 채우는 것*. Ch 20 의 MLM 과 토큰 수준에서는 똑같이 생겼습니다 — 차이는 *마스킹 비율* 과 *반복 횟수*.""")

# ----- 10. 토큰화 + group_texts -----
md(r"""## 3. 토큰화 + `group_texts` (고정 길이 블록 스트림)

Ch 20·24 와 같은 전처리 패턴 — 전체 코퍼스를 토큰화해 이어 붙이고 `block_size=128` 단위로 자릅니다. 특수 토큰 (`[CLS]`, `[SEP]`) 은 넣지 않고 *순수 텍스트 스트림* 으로 만듭니다 (diffusion 은 문장 전체를 한 캔버스로 다루므로 경계 토큰이 불필요).""")

code(r"""BLOCK_SIZE = 128

def tokenize_fn(batch):
    # add_special_tokens=False - [CLS]/[SEP] 없이 순수 토큰 스트림
    return tokenizer(batch["text"], add_special_tokens=False)

tok_train = raw_train.map(tokenize_fn, batched=True, remove_columns=raw_train.column_names, desc="tokenize train")
tok_val   = raw_val.map(tokenize_fn,   batched=True, remove_columns=raw_val.column_names,   desc="tokenize val")

# group_texts - 모든 토큰을 이어붙여 BLOCK_SIZE 단위로 자름
def group_texts(batch):
    concatenated = {k: sum(batch[k], []) for k in batch.keys()}
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE
    return {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }

lm_train = tok_train.map(group_texts, batched=True, desc="group train")
lm_val   = tok_val.map(group_texts,   batched=True, desc="group val")

# 학습엔 input_ids 만 필요 (마스킹은 collator 가 매번 새로 함)
lm_train = lm_train.remove_columns([c for c in lm_train.column_names if c != "input_ids"])
lm_val   = lm_val.remove_columns([c for c in lm_val.column_names if c != "input_ids"])

print(f"\ntrain chunks: {len(lm_train):,}  (block_size={BLOCK_SIZE})")
print(f"val   chunks: {len(lm_val):,}")
print(f"approx. train tokens: {len(lm_train) * BLOCK_SIZE / 1e6:.2f} M")
print("\nfirst chunk decode (first 200 chars):")
print(tokenizer.decode(lm_train[0]["input_ids"])[:200])""")

# ----- 11. Diffusion collator -----
md(r"""## 4. Diffusion collator — *가변 비율* 마스킹 직접 구현

여기가 BERT MLM 과 갈리는 지점입니다. Ch 20 은 `DataCollatorForLanguageModeling(mlm_probability=0.15)` 로 *고정 15%* 를 가렸지만, diffusion 은 **매 샘플마다 $t \sim U(\epsilon, 1)$ 을 뽑아 그 비율로** 가립니다.

- 각 토큰을 *독립적으로 확률 $t$* 로 `[MASK]` 치환 (LLaDA 의 forward process 와 동일)
- `labels`: 가려진 자리는 원본 토큰 id, 나머지는 `-100` (Ch 20 의 `-100` 트릭 그대로)
- `t`: $1/t$ 재가중을 위해 샘플별 비율도 함께 반환

`add_special_tokens=False` 로 토큰화했으므로 시퀀스 안에 특수 토큰이 없어 *모든 자리가 마스킹 가능* 합니다.""")

code(r"""class DiffusionCollator:
    '''매 배치마다 t ~ U(eps, 1) 을 뽑아 그 비율로 토큰을 [MASK] 치환.'''

    def __init__(self, tokenizer, eps=1e-3):
        self.mask_id = tokenizer.mask_token_id
        self.eps = eps

    def __call__(self, examples):
        ids = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long)
        B, L = ids.shape

        # 샘플별 마스킹 비율 t ~ U(eps, 1)
        t = torch.rand(B) * (1.0 - self.eps) + self.eps          # (B,)

        # 각 토큰을 독립적으로 확률 t 로 마스킹
        mask = torch.rand(B, L) < t.unsqueeze(1)                  # (B, L) bool
        # 적어도 한 자리는 가리도록 보정 (t 가 아주 작아 전부 안 가려진 경우 방지)
        no_mask_rows = ~mask.any(dim=1)
        if no_mask_rows.any():
            j = torch.randint(0, L, (int(no_mask_rows.sum()),))
            mask[no_mask_rows, j] = True

        input_ids = ids.clone()
        input_ids[mask] = self.mask_id
        labels = ids.clone()
        labels[~mask] = -100                                     # 가린 자리만 학습 신호

        attention_mask = torch.ones(B, L, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "t": t}


diff_collator = DiffusionCollator(tokenizer)

# collator 출력 확인 - 같은 두 chunk 를 여러 번 돌리면 매번 다른 비율로 가려짐
print("=== diffusion collator demo (same 2 chunks, masking ratio varies each call) ===")
for trial in range(3):
    batch = diff_collator([lm_train[0], lm_train[1]])
    labels = batch["labels"]
    t = batch["t"]
    for b in range(labels.shape[0]):
        n_masked = (labels[b] != -100).sum().item()
        frac = 100 * n_masked / labels.shape[1]
        print(f"trial {trial} | sample {b}: t={t[b]:.3f}  ->  masked {n_masked:>3d}/{labels.shape[1]} ({frac:5.1f}%)")
    print()""")

md(r"""**관전 포인트** — Ch 20 MLM collator 가 *항상 약 15%* 를 가렸다면, 이 collator 는 *호출마다 0-100% 사이 아무 값* 으로 가립니다. 같은 chunk 가 어떤 step 엔 5% 만, 다른 step 엔 90% 가려진 채 학습됩니다 → 모델이 *모든 난이도의 복원* 을 골고루 학습 → 생성 시 *어떤 마스킹 비율에서도* denoise 가능.

> **`-100` thread**: 가려진 자리만 `labels`, 나머지는 `-100`. Ch 20 (MLM 15%) → Ch 28 (SFT, prompt 만 `-100`) → 본 챕터 (가변 마스킹) — 같은 트릭의 세 번째 변주.""")

# ----- 12. 모델 -----
md(r"""## 5. 작은 BERT-style 모델 from scratch

diffusion 의 본체는 *bidirectional encoder* — 가려진 자리를 *좌·우 양방향 문맥* 으로 복원해야 하니 BERT 계열이 자연스럽습니다. `BertForMaskedLM` 을 *random init* 으로 작게 띄웁니다 (Ch 20 의 작은 BERT 와 같은 패턴).

- `num_hidden_layers=4, num_attention_heads=4, hidden_size=256` → 약 13M params (대부분 임베딩)
- `max_position_embeddings = BLOCK_SIZE = 128`
- MLM head (`Linear(H, V)`) 가 *가려진 자리의 토큰 분포* 를 출력 — 이게 곧 diffusion 의 denoiser

### GPT (Ch 24) 와 코드로 갈리는 곳

- `GPT2LMHeadModel` 이 아니라 `BertForMaskedLM` — *causal mask 없는 bidirectional attention*
- 같은 `from_pretrained` 없이 `BertForMaskedLM(config)` random init — Ch 20·22 와 동일""")

code(r"""from transformers import BertConfig, BertForMaskedLM

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=BLOCK_SIZE,
    pad_token_id=tokenizer.pad_token_id,
)

model = BertForMaskedLM(config).to(device)
n_params = model.num_parameters()
print(f"#params           : {n_params/1e6:.2f} M")
print(f"vocab_size        : {config.vocab_size}")
print(f"\nmodel: {type(model).__name__}")
print(f"  - body : {type(model.bert).__name__}  (Encoder, bidirectional attention)")
print(f"  - head : MLM head -> Linear(in={config.hidden_size}, out={config.vocab_size})")""")

# ----- 13. denoise generation 함수 (학습 전 시연용) -----
md(r"""## 6. Reverse process — 병렬 denoise 생성 함수

diffusion 생성의 핵심. **전부 `[MASK]` 인 시퀀스에서 시작**해 여러 step 에 걸쳐 점점 진짜 토큰으로 채웁니다 (LLaDA 의 *low-confidence remasking* 방식):

1. 현재 `[MASK]` 자리들을 모델이 *한꺼번에* 예측 (병렬!)
2. 각 예측의 *confidence* (softmax 최대 확률) 계산
3. *확신 높은* 자리부터 확정, *확신 낮은* 자리는 다시 `[MASK]` 로 남김
4. 스케줄에 따라 남기는 `[MASK]` 수를 step 마다 줄여 마지막엔 0개

GPT 의 *왼→오 순차* 와 결정적으로 다른 점: **채우는 순서가 위치가 아니라 confidence 순** — 문장 중간이나 끝 단어가 앞 단어보다 먼저 확정될 수 있습니다.""")

code(r"""@torch.no_grad()
def diffusion_generate(active_model, length=64, steps=16, temperature=1.0, top_k=50,
                       prompt_ids=None, record_trajectory=False):
    '''전부 [MASK] 에서 시작해 steps 번 denoise. prompt_ids 를 주면 앞부분 고정 (조건부 생성).

    기본은 sampling (temperature>0). temperature=0 으로 두면 greedy 인데,
    작은 모델 + 전부-[MASK] 출발에서는 greedy 가 최빈 토큰('.')만 뽑는 *붕괴* 가 잘 일어나
    sampling 을 기본값으로 둡니다 (아래 한계 노트 참고).'''
    active_model.eval()
    dev = active_model.device
    mask_id = tokenizer.mask_token_id

    x = torch.full((1, length), mask_id, dtype=torch.long, device=dev)
    fixed = torch.zeros(length, dtype=torch.bool, device=dev)   # 절대 마스킹 안 할 자리 (prompt)
    if prompt_ids is not None:
        p = torch.tensor(prompt_ids[:length], device=dev)
        x[0, :len(p)] = p
        fixed[:len(p)] = True
    n_gen = int((~fixed).sum().item())                          # 생성해야 할 자리 수

    traj = []
    for step in range(steps):
        logits = active_model(input_ids=x).logits[0]            # (L, V)
        probs = logits.softmax(dim=-1)
        if temperature > 0:
            scaled = logits / temperature
            if top_k > 0:                                       # top-k 로 후보 제한
                kth = scaled.topk(top_k, dim=-1).values[:, -1, None]
                scaled = scaled.masked_fill(scaled < kth, float("-inf"))
            pred = torch.multinomial(scaled.softmax(-1), 1).squeeze(-1)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        else:
            conf, pred = probs.max(dim=-1)                      # greedy (최빈 토큰 붕괴 주의)

        is_mask = (x[0] == mask_id) & (~fixed)                  # 지금 마스킹된 (생성 대상) 자리
        # 일단 마스킹된 자리를 예측으로 채운 잠정 시퀀스
        x_new = torch.where(is_mask, pred, x[0])

        # 이 step 이 끝났을 때 남겨둘 [MASK] 수 (선형 스케줄: n_gen -> 0)
        n_remain = int(round(n_gen * (1.0 - (step + 1) / steps)))
        if n_remain > 0:
            # 마스킹됐던 자리들 중 confidence 가 낮은 n_remain 개를 다시 [MASK] 로
            conf_masked = conf.clone()
            conf_masked[~is_mask] = float("inf")               # 마스킹 안 됐던 자리는 후보에서 제외
            remask_idx = conf_masked.topk(n_remain, largest=False).indices
            x_new[remask_idx] = mask_id

        x[0] = x_new
        if record_trajectory:
            traj.append(x[0].clone())

    text = tokenizer.decode(x[0], skip_special_tokens=True)
    return (text, traj) if record_trajectory else text""")

# ----- 14. 학습 전 denoise -----
md(r"""## 7. 학습 *전* denoise - 비교 기준선 (random init baseline)

학습 전 모델은 가려진 자리를 *균등 추측* 하니, denoise 결과가 *의미 없는 토큰 나열* 이 나옵니다. 학습 후와 나란히 비교하기 위한 기준선 (Ch 20·22 의 *사전학습 전 [MASK] top-5*, Ch 24 의 *학습 전 generation* 과 같은 역할).""")

code(r"""torch.manual_seed(SEED)
print("=" * 70)
print("UNTRAINED model - parallel denoise from all-[MASK]")
print("=" * 70)
for i in range(3):
    text = diffusion_generate(model, length=48, steps=16)
    print(f"\n[sample {i}] {text}")""")

md(r"""**관전 포인트** - 학습 전엔 *영어 문장과 거리가 먼 토큰 나열*. logits 가 random 이라 confidence 순서도 무의미. 학습 후 같은 함수로 다시 생성해 비교하면 *diffusion 학습이 본체에 무엇을 새겼는가* 가 드러납니다.""")

# ----- 15. 학습 -----
md(r"""## 8. `Trainer` 로 diffusion 학습 — `1/t` 재가중 loss

BERT/GPT 챕터들과 같은 `Trainer` 패턴이지만, *loss 를 직접 정의* 합니다. `BertForMaskedLM` 의 기본 loss 는 *가려진 자리 CE 평균* 인데, diffusion 은 거기에 *샘플별 `1/t` 재가중* 을 더해야 합니다 (`compute_loss` 오버라이드).

- `DiffusionCollator` → 매 배치 가변 마스킹 + `t` 반환
- `compute_loss` → 가려진 자리 CE 를 샘플별로 합산해 `1/t` 곱한 뒤 평균
- `max_steps=1500`, `batch_size=32`, `fp16=True` - T4 약 13-15분""")

code(r"""from transformers import Trainer, TrainingArguments, TrainerCallback


class DiffusionTrainer(Trainer):
    '''masked-diffusion loss: 가려진 자리 CE 를 샘플별로 1/t 재가중.'''

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        t = inputs["t"]                                          # (B,)
        labels = inputs["labels"]                               # (B, L)
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        logits = outputs.logits                                 # (B, L, V)
        B, L, V = logits.shape

        per_tok = F.cross_entropy(
            logits.view(-1, V), labels.view(-1),
            ignore_index=-100, reduction="none",
        ).view(B, L)                                            # 가린 자리만 비-0 (나머지 -100 -> 0)

        # 샘플별: (가린 자리 CE 합 / L) * (1/t)
        per_ex = per_tok.sum(dim=1) / L
        loss = (per_ex / t.to(per_ex.dtype)).mean()
        return (loss, outputs) if return_outputs else loss


args = TrainingArguments(
    output_dir="./out_diffusion_intro",
    max_steps=1500,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    fp16=USE_FP16,                       # T4 는 bf16 불가
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=150,
    save_strategy="no",
    report_to="none",
    label_names=["labels"],
    remove_unused_columns=False,         # 'labels','t' 를 collator 가 만들므로 보존
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    seed=SEED,
)


class VRAMCallback(TrainerCallback):
    def __init__(self):
        self.steps, self.peak_MiB = [], []

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            self.steps.append(state.global_step)
            self.peak_MiB.append(torch.cuda.max_memory_allocated() / 1024**2)
            torch.cuda.reset_peak_memory_stats()


vram_cb = VRAMCallback()

trainer = DiffusionTrainer(
    model=model,
    args=args,
    train_dataset=lm_train,
    eval_dataset=lm_val,
    data_collator=diff_collator,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== training summary ===")
print(f"elapsed       : {elapsed/60:.2f} min")
print(f"global_step   : {train_out.global_step}")
print(f"train_loss    : {train_out.training_loss:.4f}")
print(f"random baseline (ln vocab): {math.log(tokenizer.vocab_size):.4f}")
if torch.cuda.is_available():
    print(f"final peak    : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")""")

code(r"""# loss curve + VRAM trace
log = trainer.state.log_history
train_pts = [(r["step"], r["loss"]) for r in log if "loss" in r and "eval_loss" not in r]
eval_pts  = [(r["step"], r["eval_loss"]) for r in log if "eval_loss" in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot([s for s, _ in train_pts], [l for _, l in train_pts], "-",
         color="tab:blue", alpha=0.6, label="train")
if eval_pts:
    ax1.plot([s for s, _ in eval_pts], [l for _, l in eval_pts], "s-",
             color="tab:red", label="eval")
ax1.axhline(math.log(tokenizer.vocab_size), ls=":", color="gray",
            label=f"uniform baseline = ln({tokenizer.vocab_size}) approx. {math.log(tokenizer.vocab_size):.2f}")
ax1.set_xlabel("step"); ax1.set_ylabel("diffusion denoising loss (1/t reweighted)")
ax1.set_title("Small mask-diffusion LM on TinyStories - loss")
ax1.grid(True, alpha=0.3); ax1.legend()

if vram_cb.steps:
    ax2.plot(vram_cb.steps, vram_cb.peak_MiB, "o-", color="tab:green",
             label="peak VRAM (per log window)")
    ax2.set_title(f"VRAM trace  (bs=32, fp16, L={BLOCK_SIZE})")
else:
    ax2.text(0.5, 0.5, "VRAM trace available on CUDA only",
             ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("VRAM trace - CUDA only")
ax2.set_xlabel("step"); ax2.set_ylabel("VRAM (MiB)")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()""")

md(r"""**관전 포인트** - `1/t` 재가중 덕분에 첫 step loss 가 약 10.3 (`ln(30522)`) 부근에서 시작 (Ch 20 MLM 의 random baseline 과 같은 값!). 빠르게 떨어져 1500 step 끝에 *약 4-6* 부근에서 안정화되면 정상. 작은 모델 + TinyStories 라 완벽하진 않지만 *가려진 자리를 문맥으로 복원* 하는 능력이 본체에 새겨집니다.""")

# ----- 16. 학습 후 denoise + 궤적 -----
md(r"""## 9. 학습 *후* denoise + 궤적 시각화

같은 `diffusion_generate` 로 학습 후 생성하고, **denoise 궤적** (각 step 의 시퀀스) 을 출력해 *마스크가 단어로 채워지는 과정* 을 직접 봅니다. 이게 이 챕터의 하이라이트 — GPT 의 왼→오 순차와 달리, *문장 전체가 동시에 흐릿하게 떠오르다 선명해지는* 모습.""")

code(r"""torch.manual_seed(SEED)
print("=" * 70)
print("TRAINED model - parallel denoise from all-[MASK]")
print("=" * 70)
for i in range(3):
    text = diffusion_generate(model, length=48, steps=16)
    print(f"\n[sample {i}] {text}")""")

code(r"""# denoise 궤적 - [MASK] 가 단어로 채워지는 과정을 step 별로
torch.manual_seed(SEED)
text, traj = diffusion_generate(model, length=40, steps=12, record_trajectory=True)

def render(ids):
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    return " ".join("____" if tk == tokenizer.mask_token else tk for tk in toks)

print("=" * 78)
print("DENOISE TRAJECTORY  ('____' = still [MASK])  - filled in parallel, by confidence")
print("=" * 78)
n_steps = len(traj)
for step in [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]:
    n_mask = (traj[step] == tokenizer.mask_token_id).sum().item()
    print(f"\nstep {step:>2d}/{n_steps-1}  ([MASK] remaining: {n_mask:>2d})")
    print("  " + render(traj[step]))

print("\n" + "=" * 78)
print("FINAL:", text)""")

md(r"""**해석 가이드 - 이게 autoregressive 와 결정적으로 다른 점**

- **step 0**: 거의 전부 `____` (`[MASK]`). 모델이 *가장 확신하는* 몇 자리만 먼저 채워짐 — *위치 순서가 아니라 confidence 순서*. 문장 끝/중간 단어가 앞보다 먼저 나타날 수 있음.
- **중간 step**: 단어들이 *여기저기 동시에* 떠오름. GPT 라면 왼쪽부터 한 칸씩 채워졌을 자리가, diffusion 에선 *전 영역이 함께* 선명해짐.
- **마지막 step**: 모든 `[MASK]` 가 채워진 완성 문장.

> Ch 24 의 GPT generation 이 *왼→오 받아쓰기* 였다면, 여기선 *흐릿한 전체 그림을 반복적으로 다듬기*. 같은 TinyStories 데이터, 같은 "다음 단어가 뭘까" 직관이지만 *생성 메커니즘이 근본적으로 다릅니다.*""")

# ----- 16b. 작은 모델의 한계 (솔직 노트) -----
md(r"""## ⚠️ 작은 from-scratch diffusion 의 한계 — 솔직한 이야기

생성 결과가 *그럴듯한 동화* 와는 거리가 멀 겁니다. 이건 숨기지 않고 정확히 짚고 갑니다.

- **unconditional 생성(전부 `[MASK]` 출발)은 가장 어려운 영역**입니다. $t$ 가 1 에 가까운 (거의 다 가린) 상태는 *문맥이 거의 없어* 작은 모델이 학습하기 가장 힘듭니다. 생성은 바로 그 $t{=}1$ 에서 출발하니, 품질이 거칠고 *greedy 로 뽑으면 최빈 토큰(`.`)만 반복하는 붕괴* 가 일어납니다 → 그래서 `diffusion_generate` 의 기본을 **sampling** 으로 둡니다.
- **이건 우리 구현 버그가 아닙니다.** 같은 작은 규모에서 *표준 BERT MLM(고정 15%) 도 복원 정확도가 비슷하게 낮고*, diffusion 의 `1/t` 재가중 유무와도 차이가 없습니다 (셋 다 비슷). 즉 *모델·데이터 규모의 문제*지 알고리즘 문제가 아닙니다. loss 가 `ln(vocab)` 에서 제대로 내려간 것 자체가 *학습은 정상* 이라는 증거입니다.
- **LLaDA 가 8B 인 이유가 이것**입니다. 작은 모델도 *부분 마스킹 복원(infilling)* 처럼 문맥이 충분하면 동작하지만, *무에서 문장 생성* 은 규모를 요구합니다.

> 그래서 본 챕터 from-scratch 모델의 목적은 *메커니즘(병렬 denoise) + loss 동작* 을 **직접 손으로 확인** 하는 것입니다. *제대로 된 diffusion 생성* 은 다음 챕터에서 **사전학습된 작은 모델 (MDLM 170M / DiffuGPT 124M)** 로 확인합니다 — 같은 알고리즘, 충분한 규모.""")

# ----- 17. 조건부 생성 -----
md(r"""## 🛠️ 변형 1 - 조건부 생성 (prompt 고정)

조건부 생성은 *문맥(prompt)이 있어* unconditional 보다 잘 동작합니다. 작은 모델의 강점 영역.

GPT 의 prompt 에 대응하는 diffusion 버전: *앞부분 토큰을 고정* (절대 마스킹 안 함) 하고 *나머지만* denoise. "Once upon a time" 을 주고 뒤를 채우게 합니다.""")

code(r"""prompt = "Once upon a time"
prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

torch.manual_seed(SEED)
print(f"prompt (fixed): {prompt}")
print("=" * 70)
for i in range(3):
    text = diffusion_generate(model, length=48, steps=16, prompt_ids=prompt_ids)
    print(f"\n[sample {i}] {text}")""")

md(r"""**관전 포인트** - 앞 토큰들이 고정된 채 뒤가 채워집니다. 단, diffusion 은 *양방향* 이라 GPT 와 달리 *prompt 앞이나 중간에 빈칸* 을 두고 채우게 할 수도 있습니다 (infilling) — autoregressive 가 구조적으로 못 하는 일.""")

# ----- 18. step 수 변형 -----
md(r"""## 🛠️ 변형 2 - denoise step 수 비교 (속도 - 품질 trade-off)

diffusion 만의 자유도: *생성 step 수* 를 바꿀 수 있습니다. GPT 는 토큰 수 = step 수로 고정이지만, diffusion 은 *적은 step (빠르지만 거침) ↔ 많은 step (느리지만 정교)* 을 조절합니다.""")

code(r"""torch.manual_seed(SEED)
for steps in [1, 4, 16, 32]:
    torch.manual_seed(SEED)
    text = diffusion_generate(model, length=48, steps=steps)
    print(f"[steps={steps:>2d}] {text}\n")""")

md(r"""**관전 포인트**
- `steps=1` - 전부 `[MASK]` 를 *한 번에* 복원. 문맥 정보가 없어 *서로 안 맞는 단어들* 이 섞이기 쉬움 (각 자리가 독립적으로 예측되니 일관성 ↓).
- `steps=16-32` - 확신 높은 자리부터 단계적으로 확정 → 이미 채운 단어가 *다음 자리의 문맥* 이 되어 일관성 ↑.

> diffusion 생성 품질의 핵심 = *step 수*. 적은 step 은 빠르지만 거칠고, 많은 step 은 느리지만 정교 — 실전 모델 (LLaDA 등) 도 이 trade-off 를 조절합니다.""")

# ----- 19. AR vs Diffusion 비교 -----
md(r"""## ⚖️ Autoregressive (Ch 24) vs Diffusion (본 챕터) 비교

같은 TinyStories, 같은 "언어모델" 이지만 생성 메커니즘이 근본적으로 다릅니다.

| 축 | Autoregressive (GPT, Ch 24) | Diffusion (본 챕터) |
|---|---|---|
| attention | causal (과거만) | **bidirectional (양방향)** |
| 생성 순서 | 왼→오 *위치 순* | **confidence 순 (위치 무관)** |
| 생성 step | 토큰 수 = step (고정) | **임의 (1-32+ 조절)** |
| 병렬성 | 생성 시 순차 (느림) | **여러 자리 동시 생성 (잠재적 고속)** |
| infilling (중간 채우기) | 구조적으로 어려움 | **자연스럽게 가능** (양방향) |
| 출발 상태 | prompt | **전부 `[MASK]`** |
| 성숙도 | 표준 (대부분의 LLM) | **신생 (LLaDA, Trida 등 등장 중)** |

> **왜 diffusion 이 주목받는가**: ① *병렬 생성* 으로 잠재적 속도 이점 (autoregressive 는 토큰 수만큼 순차), ② *양방향 문맥* 으로 infilling·편집에 강점, ③ step 수로 *속도-품질* 을 추론 시점에 조절. 아직 autoregressive 만큼 성숙하진 않지만 *대안 패러다임* 으로 빠르게 발전 중입니다. Ch 33 에서 *사전학습된 작은 diffusion LM (MDLM 170M / DiffuGPT 124M)* 으로 제대로 된 생성을, Ch 34 에서 *한국어 diffusion + AR 직접 비교* 를 다룹니다.""")

# ----- 19b. 논문 계보 -----
md(r"""## 📚 이 챕터 알고리즘의 논문 계보

본 챕터에서 *직접 구현* 한 세 요소는 아래 논문들의 방법을 *교육용으로 단순화* 해 옮긴 것입니다. 어느 요소가 어느 논문의 무엇에 대응하는지 정리합니다.

| 구현 요소 (본 챕터) | 대응 논문·수식 | 일치 |
|---|---|---|
| 가변 마스킹 forward (`t ~ U(0,1)`, 토큰별 독립 마스킹) | **LLaDA** Eq. 8 / **D3PM** absorbing-state(=mask) kernel | 동일 |
| `1/t` 재가중 denoising loss (가린 자리 CE 합을 `t·L` 로 정규화) | **LLaDA** Eq. 3 = $-\mathbb{E}[\frac{1}{t}\sum_i \mathbb{1}[x_t^{(i)}{=}\texttt{M}]\log p_\theta]$ / **MDLM** weighted MLM-CE (NELBO) | 동일 |
| low-confidence remasking 생성 (전부 `[MASK]` 시작 → confidence 낮은 자리만 유지) | **LLaDA** sampling (low-confidence remasking) / **MaskGIT** confidence 병렬 디코딩 | 동일 |

> 참고로 LLaDA 논문의 loss 는 본문 수식엔 `1/L` 이 없지만 *구현(Algorithm 1)에서 `t·L` 로 정규화* 합니다. 본 챕터 코드의 `per_tok.sum()/L` 후 `/t` 평균이 정확히 `sum/(t·L)` 으로 *구현 레벨까지 일치* 합니다. 이 loss 는 *negative log-likelihood 의 upper bound* (LLaDA Eq. 4).

### 읽는 순서 추천 (계보)

1. **D3PM** — Austin et al. 2021, [arXiv:2107.03006](https://arxiv.org/abs/2107.03006). 이산 diffusion + *absorbing(=mask) 상태*. 이론 시초.
2. **MaskGIT** — Chang et al. 2022, [arXiv:2202.04200](https://arxiv.org/abs/2202.04200). *confidence 기반 반복 병렬 디코딩* — 본 챕터 생성 절차의 원조 (원래 이미지 분야).
3. **MDLM** — Sahoo et al. 2024, [arXiv:2406.07524](https://arxiv.org/abs/2406.07524). masked diffusion loss = *"고전 MLM loss 들의 가중 혼합"* (NELBO). 본 챕터 `1/t` 재가중의 이론 근거.
4. **LLaDA** — Nie et al. 2025, [arXiv:2502.09992](https://arxiv.org/abs/2502.09992). 위를 *LLM 스케일* 로. **본 챕터가 직접 따른** forward·loss·sampling. 8B 라 Ch 33 의 *대형 맛보기(선택)* 로 다룹니다.

> ⚠️ **혼동 주의** — **Diffusion-LM** (Li et al. 2022, [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)) 은 이름은 비슷하지만 *연속 임베딩 공간* 에서 Gaussian noise 를 더하는 diffusion 이라 본 챕터의 *이산 mask-diffusion* 과 **다른 계열** 입니다. Ch 33 (MDLM/DiffuGPT)·34 는 본 챕터와 같은 이산 mask-diffusion.

> 본 챕터는 *단순화판* 입니다 — 실제 LLaDA 는 semi-autoregressive remasking 등 변형, 대규모 사전학습, 정교한 스케줄을 더합니다. 하지만 *핵심 메커니즘 (가변 마스킹 + `1/t` loss + confidence 병렬 denoise)* 은 동일하므로, 본 챕터를 손으로 구현해 보면 위 논문들의 알고리즘 절을 그대로 읽어낼 수 있습니다.""")

# ----- 20. 등장한 라이브러리 -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·개념

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `BertForMaskedLM(config)` (random init) | bidirectional encoder + MLM head, diffusion 의 denoiser | Ch 33 - MDLM / DiffuGPT (사전학습 diffusion 본체) |
| `DiffusionCollator` (직접 구현) | 매 배치 `t ~ U(0,1)` 가변 마스킹 | Ch 33-34 - 실전 모델은 내부에 동등 로직 |
| `1/t` 재가중 loss (`compute_loss` 오버라이드) | masked-diffusion denoising 목표 (log-likelihood bound) | (개념) LLaDA / MDLM 의 핵심 항 |
| `diffusion_generate` (low-confidence remasking) | 전부 `[MASK]` → 반복 denoise 생성 | Ch 33-34 - 실전 sampler 의 단순화판 |
| `[MASK]` 토큰 (WordPiece 내장) | forward (가리기) + reverse (생성) 의 캔버스 | Ch 33-34 - 모델별 mask 토큰 |
| denoise 궤적 시각화 | 마스크 → 단어 병렬 채움 관찰 | (개념) AR 과의 핵심 대비 |""")

# ----- 21. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. BERT MLM (Ch 20) 의 *고정 15% 마스킹* 과 diffusion 의 *가변 마스킹* 은 collator 코드에서 정확히 무엇이 다른가요? 왜 diffusion 은 비율을 가변으로 둬야 *생성* 이 가능할까요?
2. diffusion loss 의 `1/t` 재가중이 없으면 어떤 일이 생길까요? (힌트: `t=0.05` 인 샘플과 `t=0.95` 인 샘플의 loss 크기 비교)
3. `diffusion_generate` 에서 *왜 confidence 낮은 자리를 다시 `[MASK]` 로 남기는가* 를 설명해 보세요. 한 번에 다 확정하면 (`steps=1`) 왜 품질이 떨어질까요?
4. autoregressive (GPT) 가 구조적으로 못 하는 *infilling (문장 중간 빈칸 채우기)* 을 diffusion 은 왜 자연스럽게 할 수 있나요? (causal vs bidirectional attention 관점)""")

# ----- 22. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) diffusion LM 은 결국 BERT MLM 과 뭐가 다른가요? 같은 거 아닌가요?

**메커니즘은 거의 같고, *목적과 사용법* 이 다릅니다.** BERT MLM 은 *고정 15% 를 한 번 가려 복원* 하며 *표현* 을 배우는 게 목적 (이후 downstream fine-tune). Diffusion LM 은 *가변 0-100% 마스킹 + 반복 denoise* 로 *생성* 그 자체가 목적입니다.

핵심 일반화 두 가지:
- **마스킹 비율 일반화**: 15% (고정) → $t \sim U(0,1)$ (가변). 100% 가린 상태까지 학습했기에 *전부 `[MASK]` 에서 출발하는 생성* 이 가능.
- **반복 적용**: MLM 은 1회 복원, diffusion 은 *여러 step* 에 걸쳐 점진적 복원.

```python
# BERT MLM (Ch 20) - 고정 비율, 1회
DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

# Diffusion (본 챕터) - 가변 비율 + 1/t 재가중, 생성 시 반복 denoise
t = torch.rand(B) * (1 - eps) + eps           # 매번 다른 비율
mask = torch.rand(B, L) < t.unsqueeze(1)
```

즉 *BERT 를 이미 안다면 diffusion LM 의 80% 를 이미 아는 셈* 입니다.

### Q2. (이론) `1/t` 재가중은 왜 필요한가요?

**마스킹 비율에 무관하게 loss 척도를 맞추고, 학습 목표가 *log-likelihood 의 upper bound* 가 되게 하기 위함** 입니다.

재가중이 없으면: `t=0.05` 샘플은 가려진 토큰이 약 6개뿐이라 CE 합이 작고, `t=0.95` 샘플은 약 122개라 CE 합이 큽니다. 그대로 평균하면 *많이 가린 샘플이 loss 를 지배* → 학습이 *어려운 (거의 다 가린) 경우에만* 편향됩니다.

`1/t` 를 곱하면 (수식상 가린 토큰 수가 평균적으로 $tL$ 이므로) *모든 t 의 기여가 비슷해져* 균형이 맞고, 동시에 이 형태가 *연속시간 diffusion 의 변분 하한 (ELBO)* 과 일치합니다 (LLaDA / MDLM 의 유도). 본 챕터에서 첫 step loss 가 *어떤 t 든 `ln(vocab)` 으로 정렬* 되는 게 그 증거.

### Q3. (실무) 생성 결과가 GPT (Ch 24) 보다 거친데 정상인가요?

**정상이고, 작은 from-scratch diffusion 의 *구조적* 한계입니다.** 두 가지를 구분하세요.

1. **greedy 붕괴** — 전부 `[MASK]` 에서 greedy(argmax) 로 뽑으면 문맥 없는 첫 step 에서 최빈 토큰(`.`)이 모든 자리 최고 confidence 라 *마침표만 반복* 됩니다. 그래서 `diffusion_generate` 의 기본은 sampling (`temperature=1.0, top_k=50`). greedy 는 진단·비교용으로만.
2. **규모 한계** — sampling 으로 바꿔도 작은 모델의 unconditional 생성은 거칩니다. 이건 *알고리즘이 아니라 규모* 문제예요: 같은 작은 규모에서 *표준 BERT MLM(고정 15%) 도 복원이 비슷하게 약하고*, `1/t` 재가중 유무도 차이가 없습니다. loss 가 `ln(vocab)` 에서 잘 내려간 것 자체가 학습은 정상이라는 뜻.

품질을 올리려면 규모를 키우거나(아래) — 더 현실적으로는 *사전학습된 작은 모델* 을 쓰면 됩니다 (Ch 33).

```python
# 규모 키우기 (T4 30분 안에서 가능한 선)
args.max_steps = 3000
config.num_hidden_layers = 6; config.hidden_size = 384
diffusion_generate(model, length=64, steps=32)  # 생성 step 도 늘리기
```

*제대로 된 diffusion 생성* 은 Ch 33 에서 — **MDLM (170M) / DiffuGPT (124M)** 같은 사전학습 모델이 *같은 알고리즘, 충분한 규모* 로 얼마나 달라지는지 직접 봅니다.

### Q4. (실무) `steps` 를 늘리면 무조건 좋아지나요?

**어느 지점까지는 좋아지고, 그 뒤로는 포화** 됩니다. 적은 step (`steps=1`) 은 모든 자리를 독립적으로 한 번에 확정해 *서로 안 맞는 단어* 가 섞이기 쉽고, step 을 늘리면 *이미 확정한 단어가 다음 자리의 문맥* 이 되어 일관성이 오릅니다. 하지만 step 수가 시퀀스 길이를 넘어가면 *더 줄일 `[MASK]` 가 없어* 이득이 사라집니다.

trade-off: `steps` ↑ → 품질 ↑, 속도 ↓. 실전에선 *길이의 절반 정도* 가 흔한 출발점 (예: length=64 → steps=32). diffusion 의 매력은 *이 값을 추론 시점에 자유롭게* 정할 수 있다는 것 — autoregressive 는 불가능.

### Q5. (이론) diffusion 이 autoregressive 보다 빠를 수 있다는데 왜 본 챕터는 안 빨라 보이나요?

**잠재적 병렬성** 때문입니다. autoregressive 는 토큰 N 개 생성에 *반드시 N 번 순차* forward (이전 토큰이 있어야 다음을 생성). diffusion 은 *step 수만큼만* forward 하면 되고 (step < N 가능), 각 step 에서 *여러 자리를 동시에* 채웁니다.

본 챕터에서 안 빨라 보이는 이유: 작은 모델 + 짧은 시퀀스라 forward 1회가 워낙 빨라 *오버헤드가 묻힘*. 긴 시퀀스 + 큰 모델 + 최적화된 sampler 에서 이점이 드러납니다. 다만 *현재 실전 성숙도* 는 autoregressive 가 여전히 앞섭니다 (KV-cache 등 최적화 누적). diffusion 은 *발전 중인 대안*.

### Q6. (실무) 왜 GPT 의 BPE 토크나이저 대신 BERT 의 WordPiece 를 썼나요?

**`[MASK]` 토큰이 내장돼 있어서** 입니다. diffusion 의 forward/reverse 모두 `[MASK]` 가 핵심인데, GPT-2 BPE 는 `<|endoftext|>` 하나만 있고 `[MASK]` 가 없습니다. WordPiece (`bert-base-uncased`) 는 `[MASK]` (id 103) 를 처음부터 보유 → 추가 작업 없이 바로 사용. bidirectional encoder (`BertForMaskedLM`) 와도 자연스럽게 짝이 맞습니다 (둘 다 BERT 계열).

GPT BPE 로도 `[MASK]` 를 새 special token 으로 추가하면 가능하지만, 임베딩을 새로 학습해야 해 작은 데이터에선 불리합니다.

### Q7. (이론) 그럼 앞으로 autoregressive 는 사라지나요?

**가까운 미래엔 아닙니다.** autoregressive 는 *성숙도 (KV-cache, 방대한 인프라·최적화), 안정적 품질, 검증된 스케일링* 에서 여전히 표준입니다. diffusion LM 은 *병렬 생성·infilling·step 조절* 이라는 차별점으로 *특정 용도* (빠른 생성, 편집, 제약 만족) 에서 주목받는 *대안* 입니다.

둘은 *대체* 라기보다 *공존·융합* 으로 가는 중 (일부 연구는 둘을 섞음). 본 커리큘럼이 *둘 다 직접 구현* (Ch 24 GPT, Ch 32 diffusion) 해 본 이유 — *생성 패러다임의 지형* 을 손으로 익혀 두면 어느 쪽이 발전하든 따라갈 수 있습니다.""")

# ----- 23. 다음 챕터 -----
md(r"""## 다음 챕터 예고

**Chapter 33. 작은 사전학습 Diffusion LM — MDLM (170M) + DiffuGPT (124M) 추론**

- **MDLM-owt** (`kuleshov-group/mdlm-owt`, 170M, arXiv:2406.07524) — 본 챕터가 직접 따른 *바로 그 masked diffusion 논문* 의 공식 체크포인트. `AutoModelForMaskedLM` (fill-mask) 라 본 챕터 `BertForMaskedLM` 과 *인터페이스가 거의 동일* → 코드가 매끄럽게 이어집니다. T4 여유.
- **DiffuGPT-small** (`diffusionfamily/diffugpt-s`, 124M, arXiv:2410.17891) — *가장 작은* 정식 사전학습 diffusion LM. GPT2 본체라 **Ch 24 (GPT, autoregressive) 와 같은 본체에서 AR vs diffusion 직접 비교** 가능.
- 본 챕터 작은 from-scratch 모델과 *품질 격차* 를 직접 체감 — *같은 알고리즘, 충분한 규모* 면 unconditional 생성이 얼마나 달라지는지.
- (대형 맛보기) LLaDA-8B 는 4bit 양자화로 *선택 실습*.

> **변하는 축**: *모델 출발점* (scratch 약 13M → 사전학습 170M / 124M). 메커니즘 (병렬 denoise) 은 본 챕터에서 이미 손으로 구현해 봤습니다. Ch 34 에서 *한국어 diffusion + autoregressive 직접 비교* 로 Phase 5 를 마무리합니다.""")


# ----- 노트북 저장 -----
NOTEBOOK = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(NOTEBOOK, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT_NB.relative_to(REPO)}  ({len(cells)} cells)")


# ----- README.md 작성 -----
README = r"""# 32_diffusion_intro — 작은 mask-diffusion LM 직접 구현 (Phase 5 첫 챕터)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/32_diffusion_intro/32_diffusion_intro.ipynb)

## 한 줄 목표
Phase 5 의 첫 챕터. Ch 24-31 의 *GPT (decoder, autoregressive, 왼→오 순차 생성)* 패러다임에서, *Diffusion LM (encoder/bidirectional, masked-denoise, 문장 전체 병렬 생성)* 패러다임으로 전환합니다. 핵심: **BERT MLM (Ch 20-23) 의 고정 15% 마스킹을 가변 0-100% 로 일반화하고, 한 번에 복원하는 대신 여러 번 반복 denoise 하면 그게 generation 입니다.** 작은 BERT-style 모델을 from scratch 로 TinyStories 에 diffusion 목표로 학습 → 전부 `[MASK]` 에서 시작해 *병렬 denoise* 로 텍스트를 생성하는 과정을 직접 구현·관찰합니다.

## Phase 5 도입

| 축 | Phase 4 (GPT, Ch 24-31) | **Phase 5 (Diffusion, Ch 32-34)** |
|---|---|---|
| attention | Causal (과거만) | **Bidirectional (양방향)** |
| 학습 목표 | next-token 예측 | **가변 마스킹 denoising** |
| 생성 순서 | 왼→오 한 토큰씩 순차 | **문장 전체를 동시에 반복 denoise** |
| 생성 step | 토큰 수 = step 수 | **자유 조절 (4 / 16 / 32 ...)** |
| 출발 상태 | prompt 토큰 | **전부 `[MASK]`** |
| 본체 계보 | GPT (Ch 24) | **BERT (Ch 20) — MLM 일반화** |

## 다루는 핵심 개념
- **mask-diffusion = MLM 일반화** — 고정 15% (BERT) → 가변 $t \sim U(0,1)$ (diffusion). Ch 1 부터 추적한 마스킹 thread 의 클라이맥스
- **`DiffusionCollator`** (직접 구현) — 매 배치 `t` 를 뽑아 그 비율로 `[MASK]` 치환. Ch 20 의 고정 15% collator 와 정면 대비
- **`1/t` 재가중 denoising loss** (`compute_loss` 오버라이드) — 마스킹 비율 무관하게 척도 정렬, log-likelihood upper bound
- **`BertForMaskedLM(config)` from scratch** — bidirectional encoder 가 diffusion 의 denoiser (Ch 20 과 같은 패턴, 목적만 다름)
- **reverse process generation** (`diffusion_generate`) — 전부 `[MASK]` → low-confidence remasking 으로 반복 denoise. 채우는 순서가 *위치가 아니라 confidence*. 생성은 **sampling 기본** (greedy 는 최빈 토큰 `.` 붕괴)
- **작은 from-scratch 의 한계 (솔직 노트)** — unconditional 생성은 규모를 요구합니다. 같은 작은 규모에서 *표준 MLM 도 복원이 약하고* `1/t` 유무도 차이 없음(알고리즘 아닌 규모 문제). 제대로 된 생성은 Ch 33 사전학습 모델(MDLM/DiffuGPT)에서
- **denoise 궤적 시각화** — 마스크가 *병렬로* 단어로 채워지는 과정 직접 관찰 (AR 의 왼→오와 핵심 대비)
- **조건부 생성 (infilling)** — prompt 고정 + 나머지 denoise. 양방향이라 중간 채우기도 가능 (AR 불가)
- **denoise step 수 trade-off** — 1 (빠르고 거침) ↔ 32 (느리고 정교), 추론 시점 조절
- **AR vs Diffusion 비교** — 같은 TinyStories, 생성 메커니즘만 다름. Ch 24 (GPT) 와 나란히
- **`[MASK]` 토큰** — WordPiece (`bert-base-uncased`) 내장. forward/reverse 양쪽의 캔버스

## Loss
masked-diffusion denoising loss — BERT MLM 의 CrossEntropyLoss 를 *가변 마스킹 비율 $t$* 로 일반화하고 *$1/t$ 재가중*:

$$L = \mathbb{E}_{t \sim U(0,1)} \left[ \frac{1}{t} \cdot \frac{1}{L} \sum_{i:\, x_t^{(i)} = \texttt{[MASK]}} -\log P_\theta\!\left(x_0^{(i)} \mid x_t\right) \right]$$

가려진 자리만 loss 계산 (`-100` 트릭, Ch 20-23 과 동일). `1/t` 재가중 덕분에 random baseline 이 어떤 $t$ 든 `ln(30522) ≈ 10.33` 으로 정렬 — Ch 20 MLM 과 같은 척도.

## 데이터
`roneneldan/TinyStories` (Eldan & Li 2023, arXiv:2305.07759) — Ch 24 (GPT) 와 *완전히 동일*. 학습 split 의 처음 30,000 stories. `block_size=128`, 특수 토큰 없이 순수 스트림으로 chunk 화. 데이터를 Ch 24 와 같게 둔 이유는 *생성 방식만 다른* AR vs Diffusion 비교를 위함.

## 모델
**`BertForMaskedLM`** with `hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024, max_position_embeddings=128`. 약 **13M params** (대부분 임베딩). 완전 random init from scratch — bidirectional encoder 가 diffusion denoiser 역할.

## Hyperparams
- `max_steps=1500`, `per_device_train_batch_size=32`, `learning_rate=3e-4`
- `lr_scheduler_type="cosine"`, `warmup_steps=100`, `weight_decay=0.01`, `max_grad_norm=1.0`
- `fp16=True` (T4 는 bf16 불가), `eval_steps=150`
- `remove_unused_columns=False` (collator 가 만드는 `labels`/`t` 보존), `label_names=["labels"]`
- 생성: `length=48, steps=16` 기본 (변형에서 `steps` 를 1-32 로 비교)

## 환경
Google Colab **T4 GPU 필수**. 약 25-30분 (데이터 로드 약 2분 + 토큰화 약 3분 + 학습 전 denoise 약 30초 + 모델 학습 약 13-15분 + 학습 후 denoise + 궤적 + AR 비교 약 3분).

device 자동 감지 (CUDA / MPS / CPU) - 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | 생성/학습 방식 | Loss |
|---|---|---|---|---|---|
| 24 | 작은 GPT2 (직접, scratch) | BPE (직접 학습) | TinyStories | autoregressive (왼→오) | CE (next-token) |
| 31 | Qwen2.5-0.5B-Instruct + GRPO | BBPE (Qwen2.5) | verifiable-reward | autoregressive + RL | GRPO loss |
| **32** | **작은 BERT-style (직접, scratch)** | **WordPiece (`bert-base-uncased`)** | **TinyStories** | **parallel denoise (가변 마스킹 + 반복)** | **masked-diffusion loss (`1/t` 재가중)** |
| 33 (다음) | MDLM (170M) / DiffuGPT (124M) 사전학습 | (각 모델 토크나이저) | 영어 사전학습 추론 시연 | parallel denoise (추론만) | — |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 알고리즘의 논문 계보
본 챕터에서 직접 구현한 세 요소(가변 마스킹 forward / `1/t` 재가중 loss / low-confidence remasking 생성)는 아래 논문들을 교육용으로 단순화한 것이며, 원문과 일치함을 확인했습니다.

- **D3PM** — Austin et al. 2021, [arXiv:2107.03006](https://arxiv.org/abs/2107.03006). 이산 diffusion + absorbing(=mask) 상태 (이론 시초).
- **MaskGIT** — Chang et al. 2022, [arXiv:2202.04200](https://arxiv.org/abs/2202.04200). confidence 기반 반복 병렬 디코딩 (생성 절차의 원조, 이미지).
- **MDLM** — Sahoo et al. 2024, [arXiv:2406.07524](https://arxiv.org/abs/2406.07524). masked diffusion loss = 가중 MLM-CE (NELBO). `1/t` 재가중의 이론 근거.
- **LLaDA** — Nie et al. 2025, [arXiv:2502.09992](https://arxiv.org/abs/2502.09992). 본 챕터가 직접 따른 forward·loss(Eq.3)·sampling. 구현 정규화(`t·L`)까지 일치. 8B 라 Ch 33 의 대형 맛보기(선택).

> ⚠️ 이름이 비슷한 **Diffusion-LM** (Li et al. 2022, [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)) 은 *연속 임베딩 공간* diffusion 으로 본 챕터의 이산 mask-diffusion 과 다른 계열입니다.

## 다음 챕터
Ch 33 — 사전학습된 작은 diffusion LM 추론: **MDLM-owt (170M, `kuleshov-group/mdlm-owt`)** 메인 (본 챕터가 따른 MDLM 논문의 공식 체크포인트, `AutoModelForMaskedLM` 라 인터페이스 동일) + **DiffuGPT-small (124M, `diffusionfamily/diffugpt-s`)** 보너스 (GPT2 본체, Ch 24 와 AR vs diffusion 비교). LLaDA-8B 는 4bit 대형 맛보기(선택). 본 챕터 작은 from-scratch 모델과 *품질 격차* 를 체감. Ch 34 에서 한국어 diffusion + AR 직접 비교로 Phase 5 마무리.
"""

OUT_README.write_text(README, encoding="utf-8")
print(f"Wrote {OUT_README.relative_to(REPO)}")
