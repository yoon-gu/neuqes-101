# 25_gpt2_continual_pretrain — gpt2 (124M) Continual Pretraining (Phase 4 단계 2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/25_gpt2_continual_pretrain/25_gpt2_continual_pretrain.ipynb)

## 한 줄 목표
Phase 4 의 두 번째 챕터. Ch 24 에서 *random init 작은 GPT (3.7M) 를 TinyStories 로 from scratch 사전학습* 했다면, 이번엔 **OpenAI `gpt2` (124M, WebText 약 40GB 사전학습된 본체)** 를 *같은 TinyStories 데이터* 로 **continual pretraining** (계속 사전학습 / continual learning) 합니다. *같은 CausalLM task, 같은 LM head, 같은 collator, 같은 loss* — 변하는 건 *모델 로드 한 줄 + 학습률* 뿐. 그게 GPT 시대 *학습 단계 2 (continual pretraining)* 의 본질입니다.

## GPT 시대 학습 4단계 — 본 챕터의 위치

| 단계 | 용어 | 본 챕터? | 본 커리큘럼 |
|---|---|---|---|
| 1 | Pretraining | | Ch 24 (영어), Ch 26 (한국어) |
| **2** | **Continual pretraining** (계속 사전학습 / continual learning) | **✅ ← 여기** | **Ch 25** |
| 3 | SFT (Instruction tuning) | | Ch 28 |
| 4 | Alignment (DPO / GRPO) | | Ch 30-31 |

> **Ch 25 ≠ SFT** — *task adaptation 의미의 fine-tune 이 아니라 같은 CausalLM task 를 새 데이터로 더 학습*. head 안 바뀜, loss·trainer 안 바뀜. SFT 는 Ch 28 에서 본격.

## 다루는 핵심 개념
- **`AutoModelForCausalLM.from_pretrained("gpt2")`** — OpenAI WebText 약 40GB 로 사전학습된 124M params 본체. *모델 로드 한 줄* 로 학습 단계 2 진입
- **`AutoTokenizer.from_pretrained("gpt2")`** — gpt2 BPE (vocab 50,257) 그대로. *토크나이저는 본체와 운명공동체*
- **`tokenizer.pad_token = tokenizer.eos_token`** — gpt2 의 pad 컨벤션
- **lr `2e-5`** — continual pretraining 표준 (Ch 24 의 `3e-4` 보다 약 15배 작음). *catastrophic forgetting 방지*
- **`transformers.Trainer` + `DataCollatorForLanguageModeling(mlm=False)`** — *Ch 24 와 정확히 같은 코드*. 학습 단계 2 의 정의
- **`gradient_accumulation_steps`** — T4 16GB + 124M 모델의 메모리 제약 해소 (per_device_batch=4, accumulation=4 → effective batch 16)
- **사전학습된 본체의 시작 loss** — random baseline (`ln(50257) ≈ 10.82`) 이 아니라 *약 3.0-4.0* 에서 시작. *Ch 24 와 본질적 차이*
- **3-way generation 비교** — Ch 24 (3.7M scratch) vs Ch 25 BEFORE (gpt2 그대로) vs Ch 25 AFTER (continual pretrain). *모델 크기와 사전학습 효과는 분리 불가능* 의 정량 표시
- **Catastrophic forgetting** — 긴 학습 / 큰 lr 일 때 사전학습된 일반 도메인 능력이 손실되는 현상. 짧은 학습 + 작은 lr 로 완화
- **Continual pretraining ↔ SFT (Ch 28) 의 정확한 경계** — `labels = -100` 자리가 *pad 만 (단계 2)* vs *prompt 부분 (단계 3)*

## Loss
`CrossEntropyLoss` (next-token, `mlm=False`) — *Ch 24 와 완전히 동일*. `labels = input_ids.clone()`, pad 만 `-100`. 다만 *vocab 차원이 2,048 → 50,257 로 변하고* *시작 weight 가 random 이 아닌 사전학습된 본체* 라는 점이 *loss 곡선의 시작 지점* 을 결정.

수식: $L_{\text{CLM}} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(x_{i+1} \mid x_{\leq i})$  (Ch 24 와 동일)

## 데이터
`roneneldan/TinyStories` — *Ch 24 와 정확히 같은 split* (train 30K + eval 500). *데이터는 통제 변수*.

`block_size=128` 로 `group_texts` 후 train 51,863 chunks (약 6.64M 토큰) / eval 788 chunks. gpt2 vocab 이 커서 같은 텍스트가 Ch 24 보다 적은 토큰·chunk 로 쪼개집니다.

## 모델
**`AutoModelForCausalLM.from_pretrained("gpt2")`** — `n_layer=12, n_head=12, n_embd=768, n_positions=1024`. 약 **124M params** (Ch 24 의 약 33배). WebText 약 40GB 로 사전학습된 본체 그대로 로드 → continual pretraining.

## Hyperparams
- `num_train_epochs=1`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=4` (effective batch 16)
- `learning_rate=2e-5` ← *Ch 24 의 `3e-4` 와 다른 유일한 큰 차이*
- `lr_scheduler_type="cosine"`, `warmup_steps=0.06` (1 미만 값은 전체 step 대비 비율 해석 — 구 `warmup_ratio`)
- AdamW `weight_decay=0.01`, `max_grad_norm=1.0`
- `fp16=True` (T4 는 bf16 불가)
- `eval_strategy="steps"`, `eval_steps=100`

## 환경
Google Colab **T4 GPU 필수**. 약 20-25분 (데이터 로드·gpt2 로드·토큰화 약 2분 + 학습 전 generation 약 30초 + continual pretraining 약 19분 + 학습 후 generation + 3-way 비교 약 1분).

device 자동 감지 (CUDA / MPS / CPU) — 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 23 | 작은 BERT (한국어, scratch) + 분류 head | klue/bert-base | NSMC 이진 | Linear(H, 2) | CE |
| 24 | 작은 GPT2 (3.7M, scratch) | BPE 직접 학습 (vocab 2,048) | TinyStories 30K | Linear(H, V) (LM head, weight tied) | CE (next-token) |
| **25** | **`gpt2` (124M, WebText 사전학습)** | **BPE (gpt2 그대로, vocab 50,257)** | **TinyStories 30K (Ch 24 와 동일)** | **Linear(H, V) (LM head 그대로)** | **CE (next-token) — *continual pretraining*** |
| 26 (다음) | 작은 GPT (한국어, scratch) | BPE 직접 학습 (한국어) | 한국어 TinyStories-Korean | Linear(H, V) (LM head, weight tied) | CE (next-token) |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[26_ko_tiny_gpt](../26_ko_tiny_gpt/) (예정) — *Ch 24 의 한국어판*. 작은 GPT scratch + 한국어 BBPE 직접 학습 + 한국어 TinyStories. *왜 영어 사전학습 모델 (gpt2) 을 한국어에 그대로 적용하기 어려운가* 의 답 — *토크나이저는 본체와 운명공동체* 원칙이 한국어에서 *scratch* 를 강제. 그 다음 Ch 27 (KoGPT2 + 한국어 TinyStories continual pretraining — 본 챕터 Ch 25 의 한국어 짝), SFT (단계 3) 는 Ch 28 (KoGPT2 + KoAlpaca) 에서 본격 등장.
