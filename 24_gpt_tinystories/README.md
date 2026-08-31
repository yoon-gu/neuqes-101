# 24_gpt_tinystories — GPT (TinyStories) from-scratch 사전학습 (Phase 4 첫 챕터)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/24_gpt_tinystories/24_gpt_tinystories.ipynb)

## 한 줄 목표
Phase 4 의 첫 챕터. Ch 7-23 의 *BERT (encoder, MLM, task head 부착 fine-tune)* 패러다임에서, *GPT (decoder-only, causal LM, LM head 그대로)* 패러다임으로 전환합니다. `GPT2LMHeadModel` 을 random init 으로 from scratch 띄우고, **TinyStories 30,000 stories** 로 next-token 예측 사전학습 → 같은 prompt 에 *학습 전 / 학습 후* generation 결과를 나란히 비교 (+ reference `gpt2` 124M 까지 3-way). Ch 20·22 의 *사전·사후 [MASK] top-5 비교* 와 같은 깊이로, *사전학습이 본체에 어떤 next-token 분포를 새겼는가* 를 직접 확인합니다.

## Phase 4 도입

| 축 | Phase 1·2·3 (BERT, Ch 7-23) | **Phase 4 (GPT, Ch 24-30)** |
|---|---|---|
| 본체 | Encoder (양방향 attention) | **Decoder (causal attention)** |
| 사전학습 task | MLM (가려진 토큰 예측) | **CausalLM (next-token 예측)** |
| 학습 신호 위치 | 선택된 약 15% 만 (`-100` 다수) | **거의 모든 토큰** (`-100` pad 만) |
| Output head | task 별 부착 (`Linear(H, K)`) | **LM head (`Linear(H, V)`) 그대로** |
| Downstream 적응 | head 교체 + fine-tune (*task 적응*) | **SFT (*behavior alignment*)** + alignment (DPO / GRPO) |
| "Fine-tune" 의미 | task 별 특화 | **prompt 만 바꿔도 다른 일** |

> 본 챕터는 그 *출발점* - 작은 GPT 를 처음부터 학습해 *next-token 예측이 어떻게 generation 으로 이어지는지* 를 직접 봅니다. Ch 25 (대규모 사전학습 `gpt2` **continual pretraining**) / Ch 28 (SFT) / Ch 30-31 (DPO / GRPO) 가 같은 본체 위에 쌓여 갑니다.

## 다루는 핵심 개념
- **GPT2LMHeadModel(config)** from scratch - `from_pretrained` 없이 random init (Ch 20·22 의 `BertForMaskedLM(config)` 와 같은 패턴, 모델 패밀리만 다름)
- **`GPT2Config` 핵심 필드** - `n_layer / n_head / n_embd / n_positions`, `tie_word_embeddings` (기본 True, 약 0.5M params 절약)
- **causal attention** - encoder (BERT) 와 본질적 차이. 모델 클래스가 내장 처리
- **`DataCollatorForLanguageModeling(mlm=False)`** - labels = input_ids.clone() 자동. *거의 모든 자리* 가 학습 신호 (MLM 의 15% 와 정반대)
- **`group_texts` 패턴** (HF run_clm.py 표준) - 가변 길이 텍스트 → 고정 길이 `block_size=128` 블록 스트림
- **byte-level BPE 토크나이저 직접 학습** - `tokenizers.BPE + ByteLevel`, vocab 2,048. WordPiece 와의 결합 방식 차이
- **GPT-2 special token 컨벤션** - `<|endoftext|>` 하나가 bos / eos / pad 겸용
- **`model.generate(do_sample=True, ...)`** - temperature / top_k / top_p sampling 비교
- **`-100` thread 환기** - MLM (15% 자리) vs CausalLM (거의 모든 자리) vs SFT (response 부분만, Ch 28) - 같은 트릭, 정반대 자리
- **파인튜닝 의미 변화 thread 환기** - BERT 시대 (task head 부착) vs GPT 시대 (head 그대로, 행동 정렬)
- Random baseline loss `ln(2048) ≈ 7.62`, TinyStories 3M 모델은 보통 *약 2.5-3.0* 까지 도달
- **Reference 비교** - `gpt2` (124M, WebText 약 40GB) 의 같은 prompt generation 으로 *모델 크기 + 데이터 격차* 의 generation 품질 차이 직접 확인

## Loss
`CrossEntropyLoss` (next-token, `mlm=False`) - BERT MLM 의 CE 와 수식적으로 동일. 마스킹 위치만 다름 (BERT: 무작위 15% / GPT: 모든 토큰의 다음 위치, 거의 모든 자리). 모델 forward 안에서 `logits` 와 `labels` 가 한 칸 shift 되어 처리.

수식: $L_{\text{CLM}} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(x_{i+1} \mid x_{\leq i})$

## 데이터
`roneneldan/TinyStories` (Eldan & Li 2023, arXiv:2305.07759) - GPT-3.5 / GPT-4 가 *4세 어린이 어휘* 로 생성한 짧은 영어 동화 약 2.1M 편. 본 챕터는 *학습 split 의 처음 30,000 stories* 만 사용 (약 4-6M 토큰).

`block_size=128` 로 `group_texts` 후 train 약 30,000-50,000 chunks / eval 약 500 chunks.

## 모델
**`GPT2LMHeadModel`** with `n_layer=4, n_head=4, n_embd=256, n_positions=128`. 약 **3.7M params** (weight tying 자동 적용). BERT 챕터들 (Ch 20·22 의 작은 BERT 약 10M, Ch 9-18 의 DistilBERT 약 66M) 과 다르게 *완전 random init* 에서 시작.

## Hyperparams
- `max_steps=1500`, `per_device_train_batch_size=32`, `learning_rate=3e-4`
- `lr_scheduler_type="cosine"`, `warmup_steps=100`
- AdamW `betas=(0.9, 0.95)`, `weight_decay=0.1`, `max_grad_norm=1.0`
- `fp16=True` (T4 는 bf16 불가)
- `eval_strategy="steps"`, `eval_steps=150`

## 환경
Google Colab **T4 GPU 필수**. 약 4-6분 (데이터 로드 약 1-2분 + BPE 학습 약 10초 + 학습 전 generation 약 30초 + 모델 학습 약 1분 + 학습 후 generation + reference `gpt2` 비교 약 2분).

device 자동 감지 (CUDA / MPS / CPU) - 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 22-23 | 작은 BERT (한국어, scratch) | klue/bert-base (가져옴) | 한국어 위키 → NSMC | MLM head / Linear(H, 2) | CE (masked / class) |
| **24** | **작은 GPT2 (직접, scratch)** | **BPE (직접 학습, vocab 2,048)** | **TinyStories 30K stories** | **Linear(H, V) (LM head, weight tied)** | **CE (next-token, 거의 모든 자리)** |
| 25 (다음) | gpt2 (124M, OpenAI WebText 사전학습) | BPE (GPT2 그대로, vocab 50,257) | TinyStories (Ch 24 와 동일) | Linear(H, V) (LM head 그대로) | CE (next-token) - **continual pretraining** |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[25_gpt2_continual_pretrain](../25_gpt2_continual_pretrain/) - OpenAI `gpt2` (124M, WebText 약 40GB 사전학습) 을 *같은 TinyStories 30K* 로 **continual pretraining** (계속 사전학습 / continual learning — 같은 CausalLM task, head 그대로). *데이터를 통제하고 본체 출발점만 다름*. 본 챕터 (3.7M, from scratch, 약 4분) vs Ch 25 (124M, continual pretraining, 약 22분) 의 generation 품질·학습 곡선 격차가 *왜 실무는 from-scratch 가 아니라 대규모 사전학습 모델을 활용하는가* 의 정량 답변. *진짜 task adaptation 의미의 fine-tune (instruction tuning)* 은 Ch 28 SFT 에서 본격 등장.
