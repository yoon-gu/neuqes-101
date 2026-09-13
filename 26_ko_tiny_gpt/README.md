# 26_ko_tiny_gpt — 한국어 GPT (TinyStories-Korean) from-scratch 사전학습 (Phase 4 한국어 단계 1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/26_ko_tiny_gpt/26_ko_tiny_gpt.ipynb)

## 한 줄 목표
Ch 24 (영어 GPT scratch + TinyStories) 의 *한국어 대칭본*. *같은 본체 구조* (작은 GPT2, 약 4.2M — Ch 24 는 vocab 차이로 약 3.7M) 로 **한국어 GPT 사전학습** 을 합니다. 변하는 축은 **언어** — 토크나이저는 한국어 코퍼스 위에 직접 학습한 **byte-level BPE (BBPE)**, 데이터는 **`g0ster/TinyStories-Korean`** (영어 TinyStories 의 한국어 번역본). 본체·loss·trainer 는 Ch 24 와 동일 (hyperparams 는 학습률만 `3e-4 → 5e-4`). 같은 한국어 prompt 에 *학습 전 / 학습 후* generation 을 나란히 비교해 *사전학습이 본체에 어떤 next-token 분포를 새겼는가* 를 한국어로 직접 확인합니다.

## Phase 4 의 영어·한국어 대칭

| 학습 단계 | 영어 | **한국어** |
|---|---|---|
| **단계 1: Pretraining** (random init → scratch) | Ch 24 (작은 GPT, 영어 TinyStories) | **Ch 26 ← 여기 (작은 GPT, 한국어 TinyStories)** |
| **단계 2: Continual pretraining** | Ch 25 (`gpt2` 124M) | Ch 27 (KoGPT2 125M) |

> Ch 20(영어 BERT)→Ch 22(한국어 BERT) 의 *언어 축 변화* 가 GPT 에서 그대로 반복. 본 챕터는 한국어 단계 1 — *언어가 달라도 작은 GPT + 30K stories from-scratch 의 학습 동역학은 비슷하다* 가 검증 가설.

## 다루는 핵심 개념
- **언어 한 축 변화** - 토크나이저 학습 코퍼스 + 데이터만 한국어로, 본체 구조·loss·trainer 는 Ch 24 동일 (학습률만 `3e-4 → 5e-4`)
- **BBPE 토크나이저 직접 학습** - `tokenizers.BPE + ByteLevel` 을 한국어 코퍼스 위에 학습 (vocab 약 4,000). byte-level 이 한글을 UNK 없이 처리
- **영어 gpt2 BPE vs 한국어 BBPE 비교** - 같은 한국어 문장의 토큰 수 격차 실측 (Ch 19 §5-4 / Ch 25 Q4 의 cross-language 결론)
- **한국어 TinyStories story 복원** - 줄 단위 데이터를 `<|endoftext|>` 기준으로 이어 붙여 story 단위로 복원 (streaming 로드)
- **`GPT2LMHeadModel(config)` from scratch** - Ch 24 와 같은 패턴, vocab 만 한국어 BBPE 에 맞춤
- **`DataCollatorForLanguageModeling(mlm=False)`** - `labels = input_ids.clone()`, 거의 모든 자리 학습 신호 (한국어 재확인)
- **`group_texts` 패턴** (HF run_clm.py 표준) - 가변 길이 → 고정 `block_size=128` 블록 스트림
- **`-100` thread 환기** - MLM (15%) vs CausalLM (거의 모든 자리) vs SFT (답변만, Ch 28) - 같은 트릭, 정반대 자리
- **학습 전·후 generation 비교** - random init (의미 없는 음절 나열) → 학습 후 (동화 풍 한국어)
- Random baseline loss `ln(4000) ≈ 8.29` 에서 1500 step 뒤 누적 평균 `train_loss` *약 4.5* 까지 도달 (영어 Ch 24 의 약 3.7 보다 다소 높음 — 번역체 데이터 + 학습률 차이)
- **(선택) KoGPT2 reference** - `skt/kogpt2-base-v2` (125M) 의 같은 prompt generation 으로 규모 격차 확인

## Loss
`CrossEntropyLoss` (next-token, `mlm=False`) - Ch 24 와 *완전히 동일*. vocab 차원만 2,048 → 약 4,000 으로 바뀌어 random baseline `ln V` 가 약 7.62 → 약 8.29 로 미세 이동.

수식: $L_{\text{CLM}} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(x_{i+1} \mid x_{\leq i})$

## 데이터
`g0ster/TinyStories-Korean` (Dohoon Kim, 2024, MIT) - 영어 `roneneldan/TinyStories` 의 한국어 번역본. *줄 단위* 저장이라 `<|endoftext|>` 기준으로 story 를 복원해 처음 30,000 stories 사용 (Ch 24 와 같은 규모). `block_size=128` `group_texts` 후 train 45,845 chunks / eval 750 chunks (약 5.87M 토큰).

## 모델
**`GPT2LMHeadModel`** with `n_layer=4, n_head=4, n_embd=256, n_positions=128`. 약 **4.2M params** (weight tying 자동 적용). Ch 24 (약 3.7M) 와 *같은 구조*, vocab 만 한국어 BBPE (약 4,000) 에 맞춰 embedding·LM head 가 그만큼 커짐. *완전 random init* 에서 시작.

## Hyperparams
- `max_steps=1500`, `per_device_train_batch_size=32`, `learning_rate=5e-4` (Ch 24 의 3e-4 에서 한국어 vocab 에 맞춰 조정)
- `lr_scheduler_type="cosine"`, `warmup_steps=100`
- AdamW `betas=(0.9, 0.95)`, `weight_decay=0.1`, `max_grad_norm=1.0`
- `fp16=True` (T4 는 bf16 불가)
- `eval_strategy="steps"`, `eval_steps=150`

## 환경
Google Colab **T4 GPU 필수**. 약 3-5분 (데이터 로드·story 복원 약 20초 + BBPE 토크나이저 학습 약 10초 + 학습 전 generation + 모델 학습 약 1분 + 학습 후 generation + (선택) KoGPT2 reference 로드·생성 약 1분).

device 자동 감지 (CUDA / MPS / CPU) - 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 24 | 작은 GPT2 (약 3.7M, scratch) | BPE (직접 학습, 영어, vocab 2,048) | 영어 TinyStories 30K | Linear(H, V) (LM head, weight tied) | CE (next-token) |
| 25 | gpt2 (124M, WebText 사전학습) | BPE (gpt2 그대로, vocab 50,257) | 영어 TinyStories (Ch 24 와 동일) | Linear(H, V) (LM head 그대로) | CE (next-token) - continual pretraining |
| **26** | **작은 GPT2 (약 4.2M, scratch)** | **BBPE (직접 학습, 한국어, vocab 약 4,000)** | **한국어 TinyStories 30K** | **Linear(H, V) (LM head, weight tied)** | **CE (next-token)** |
| 27 (다음) | KoGPT2 (125M, 대규모 한국어 사전학습) | KoGPT2 BBPE (그대로) | 한국어 TinyStories (Ch 26 과 동일) | Linear(H, V) (LM head 그대로) | CE (next-token) - continual pretraining |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[27_ko_gpt2_continual_pretrain](../27_ko_gpt2_continual_pretrain/) (예정) - 대규모 한국어 사전학습 모델 KoGPT2 (`skt/kogpt2-base-v2`, 125M) 를 *같은 한국어 TinyStories* 로 **continual pretraining**. *데이터를 통제하고 본체 출발점만 다름*. 본 챕터 (약 4.2M, from scratch) vs Ch 27 (125M, continual pretraining) 의 generation 품질·학습 곡선 격차가 *왜 실무는 from-scratch 가 아니라 대규모 사전학습 모델을 활용하는가* 의 한국어 정량 답변. Ch 24→Ch 25 (영어) 의 한국어 짝. *진짜 행동 정렬 (SFT)* 은 Ch 28 에서 본격 등장.
