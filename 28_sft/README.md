# 28_sft — KoGPT2 SFT / Instruction Tuning (Phase 4 학습 단계 3)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/28_sft/28_sft.ipynb)

## 한 줄 목표
Ch 27 (KoGPT2 continual pretraining) 의 *다음 단계*. 같은 **KoGPT2 (`skt/kogpt2-base-v2`, 125M)** 본체를 **KoAlpaca instruction-response 쌍 데이터** 로 **SFT (Supervised Fine-Tuning / Instruction Tuning)** 합니다. 본체도 같고 loss 종류 (next-token `CrossEntropyLoss`) 도 같습니다 — 바뀌는 건 *데이터 형식 (연속 텍스트 → instruction-response 쌍)* + *`labels` 마스킹 (pad 만 → prompt 부분 전체)* + *trainer (`Trainer` → `trl.SFTTrainer`)*. **그 한 줄 `labels[:prompt_len] = -100` 이 모델을 지시를 따르게 만듭니다** — *행동 정렬 (behavior alignment)*.

## 이 챕터가 클라이맥스인 두 thread

### Thread 1 — `labels = -100` 자리의 완성
| 단계 | 챕터 | `labels = -100` 자리 | loss 계산 |
|---|---|---|---|
| MLM | Ch 20·21·22 | 선택 안 된 약 85% | 가린 약 15% 만 |
| CausalLM | Ch 24·25·26·27 | pad 만 | 거의 전 토큰 |
| **SFT (본 챕터)** | **Ch 28** | **prompt 부분 전체** | **답변 토큰만** |

§3 에서 `trl` collator 의 출력 `labels` 를 *토큰별 표* 로 직접 print — prompt 가 `-100`, 답변만 원본 token id 인 것을 *눈으로 확인*. (Ch 21 `[MASK]` 80/10/10 시각화의 SFT 판.)

### Thread 2 — "파인튜닝" 의미 변화의 완성 (behavior alignment)
- BERT 파인튜닝 (Ch 9-23) = task head 부착 (task 적응)
- GPT continual pretraining (Ch 25·27) = head 그대로, 같은 task, 새 데이터
- **GPT SFT (Ch 28) = head 그대로, 같은 next-token CE, 단 *데이터 형식 + prompt 마스킹* → *행동 정렬*** ← *진짜* instruction following

> *같은 모델이 입력 프롬프트만 바꾸면 다른 task* — 그게 *왜 GPT 하나가 모든 task 를 해내는가* 의 답. SFT 가 그 능력을 *깨우는* 단계.

## GPT 시대 학습 4단계 — 본 챕터의 위치

| 단계 | 용어 | 본 챕터? | 본 커리큘럼 |
|---|---|---|---|
| 1 | Pretraining | | Ch 24 (영어), Ch 26 (한국어) |
| 2 | Continual pretraining | | Ch 25 (영어), Ch 27 (한국어) |
| **3** | **SFT (Instruction tuning)** | **✅ ← 여기** | **Ch 28** |
| 4 | Alignment (DPO / GRPO) | | Ch 30·31 |

## 다루는 핵심 개념
- **`trl.SFTTrainer` + `trl.SFTConfig`** — SFT 특화 trainer (첫 등장). `transformers.Trainer` 를 상속, *prompt/completion 전처리 + EOS 부착 + completion 마스킹* 자동
- **`SFTConfig(completion_only_loss=True)`** — *답변 부분만* loss, prompt 는 `-100`. SFT 의 핵심 옵션
- **`prompt` / `completion` 데이터 형식** — instruction-response 쌍 표준. RESPONSE_TEMPLATE `### 응답:\n` 가 답변 시작 경계
- **collator labels 마스킹 시각화** (§3, 클라이맥스) — 토큰별 `position | token | input_id | label | learn?` 표로 prompt=`-100`, 답변만 학습 직접 확인
- **SFT 전·후 instruction following 비교** (§6, 핵심 데모) — 같은 본체가 *지시를 따르는 방향* 으로 행동 정렬
- **`AutoModelForCausalLM` + `PreTrainedTokenizerFast`** — Ch 27 과 같은 KoGPT2 본체·토크나이저 (AutoTokenizer 함정 회피)
- **chat template** 의 의미 — KoGPT2 는 base 모델이라 없어 직접 포맷

## Loss
next-token `CrossEntropyLoss` — *Ch 27 과 같은 종류*. 다른 건 *어느 자리에서 합산하느냐*. SFT 는 *답변 토큰만* 합산 (prompt 는 `-100` 으로 제외).

수식: $L_{\text{SFT}} = -\sum_{i \in \text{response}} \log P(x_{i+1} \mid x_{\leq i})$  (prompt 부분 제외)

## 데이터
`beomi/KoAlpaca-v1.1a` — 한국어 instruction tuning 데이터셋 (`instruction` / `output` 필드). 약 3,000 샘플 subset, `### 명령어:\n{instruction}\n\n### 응답:\n{output}` 로 직접 포맷.

## 모델
**`AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")`** — *Ch 27 과 같은 KoGPT2 본체* (125M params). LM head 그대로, next-token task 그대로.

## Hyperparams
- `num_train_epochs=1`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` (effective batch 16)
- `learning_rate=2e-5` (SFT 표준), `lr_scheduler_type="cosine"`, `warmup_ratio=0.03`
- `max_length=512`, `completion_only_loss=True`, `packing=False`
- `fp16=True` (T4 는 bf16 불가)

## 라이브러리 주의 — `trl` 버전
`trl` 은 버전마다 API 변동이 큽니다 (`DataCollatorForCompletionOnlyLM` 처럼 버전에 따라 사라진 클래스도 있음). 본 노트북은 *`prompt`/`completion` 데이터 + `completion_only_loss=True`* 라는 *최신 trl 의 표준 경로* 를 사용 — 버전 간 가장 안정적. 설치된 `trl` 버전은 셋업 셀 출력에서 확인하세요.

## 환경
Google Colab **T4 GPU 필수**. 약 20-28분 (KoAlpaca 로드·포맷 약 2분 + KoGPT2 로드 약 2분 + collator 시각화 약 1분 + SFT 약 12-18분 + SFT 전·후 비교 약 3분).

device 자동 감지 (CUDA / MPS / CPU) — 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | `labels = -100` 자리 | Loss |
|---|---|---|---|---|---|
| 26 | 작은 GPT2 (한국어, 약 3M, scratch) | BBPE (직접 학습, vocab 약 4,000) | 한국어 TinyStories 30K | pad 만 | CE (next-token) |
| 27 | KoGPT2 (125M) | Character BPE (KoGPT2 그대로, vocab 51,200) | 한국어 TinyStories 30K | pad 만 | CE (next-token) - continual pretraining |
| **28** | **KoGPT2 (125M, 동일)** | **Character BPE (KoGPT2 그대로, 동일)** | **KoAlpaca instruction-response (약 3K)** | **prompt 부분 (답변만 학습)** | **CE (next-token, response-only) — SFT** |
| 29 (다음) | Ch 28 SFT 모델 + 비교 | (동일) | 분야별 벤치마크 | - (평가만) | - (`lm-evaluation-harness`) |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[29_benchmark_eval](../29_benchmark_eval/) (예정) — 본 챕터의 SFT 모델을 *분야별 벤치마크* (KMMLU / HAERAE / MMLU ...) 로 *정량* 평가. §6 의 정성적 BEFORE/AFTER 를 점수로. 그 다음 [30_dpo](../30_dpo/) (DPO alignment) — *preference 정렬*. `labels = -100` thread 가 DPO 에서도 chosen/rejected *response 부분만* 계산으로 이어집니다.
