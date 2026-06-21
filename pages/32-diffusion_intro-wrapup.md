## 이번 챕터에 등장한 라이브러리·개념

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `BertForMaskedLM(config)` (random init) | bidirectional encoder + MLM head, diffusion 의 denoiser | Ch 33 - MDLM / DiffuGPT (사전학습 diffusion 본체) |
| `DiffusionCollator` (직접 구현) | 매 배치 `t ~ U(0,1)` 가변 마스킹 | Ch 33-34 - 실전 모델은 내부에 동등 로직 |
| `1/t` 재가중 loss (`compute_loss` 오버라이드) | masked-diffusion denoising 목표 (log-likelihood bound) | (개념) LLaDA / MDLM 의 핵심 항 |
| `diffusion_generate` (low-confidence remasking) | 전부 `[MASK]` → 반복 denoise 생성 | Ch 33-34 - 실전 sampler 의 단순화판 |
| `[MASK]` 토큰 (BPE 2048 에 special token 으로 추가, id 2) | forward (가리기) + reverse (생성) 의 캔버스 | Ch 33-34 - 모델별 mask 토큰 |
| denoise 궤적 시각화 | 마스크 → 단어 병렬 채움 관찰 | (개념) AR 과의 핵심 대비 |

## 체크포인트 질문

1. BERT MLM (Ch 20) 의 *고정 15% 마스킹* 과 diffusion 의 *가변 마스킹* 은 collator 코드에서 정확히 무엇이 다른가요? 왜 diffusion 은 비율을 가변으로 둬야 *생성* 이 가능할까요?
2. diffusion loss 의 `1/t` 재가중이 없으면 어떤 일이 생길까요? (힌트: `t=0.05` 인 샘플과 `t=0.95` 인 샘플의 loss 크기 비교)
3. `diffusion_generate` 에서 *왜 confidence 낮은 자리를 다시 `[MASK]` 로 남기는가* 를 설명해 보세요. 한 번에 다 확정하면 (`steps=1`) 왜 품질이 떨어질까요?
4. autoregressive (GPT) 가 구조적으로 못 하는 *infilling (문장 중간 빈칸 채우기)* 을 diffusion 은 왜 자연스럽게 할 수 있나요? (causal vs bidirectional attention 관점)

## FAQ

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

### Q6. (실무) BPE 2048 을 직접 학습하면서 `[MASK]` 토큰은 어떻게 마련했나요?

**BPE 를 학습할 때 `special_tokens` 로 함께 등록** 했습니다. diffusion 의 forward/reverse 모두 `[MASK]` 가 핵심인데, 일반 BPE/WordPiece 어휘에는 `[MASK]` 가 없을 수 있습니다. 이 챕터는 작은 from-scratch 모델에 맞춰 *vocab 을 작게* 가져가려고 `bert-base-uncased` 의 WordPiece(30,522) 를 그대로 쓰지 않고, TinyStories 코퍼스에 ByteLevel BPE 를 vocab 2,048 으로 직접 학습합니다. 이때 `BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[MASK]"])` 로 세 특수 토큰을 어휘 맨 앞에 고정 배정해 `[MASK]` 가 id 2 에 자리 잡습니다.

```python
trainer = trainers.BpeTrainer(vocab_size=2048,
                              special_tokens=["[PAD]", "[UNK]", "[MASK]"])
# 학습 후: tokenizer.mask_token_id == 2
```

모델은 `BertForMaskedLM` 을 *random init* 으로 띄우므로 임베딩도 처음부터 함께 학습됩니다 — `[MASK]` 임베딩이 별도 부담이 아니라 본체와 같이 자라납니다. bidirectional encoder (`BertForMaskedLM`) 와 `[MASK]` 기반 denoising 이 자연스럽게 짝을 이룹니다.

### Q7. (이론) 그럼 앞으로 autoregressive 는 사라지나요?

**가까운 미래엔 아닙니다.** autoregressive 는 *성숙도 (KV-cache, 방대한 인프라·최적화), 안정적 품질, 검증된 스케일링* 에서 여전히 표준입니다. diffusion LM 은 *병렬 생성·infilling·step 조절* 이라는 차별점으로 *특정 용도* (빠른 생성, 편집, 제약 만족) 에서 주목받는 *대안* 입니다.

둘은 *대체* 라기보다 *공존·융합* 으로 가는 중 (일부 연구는 둘을 섞음). 본 커리큘럼이 *둘 다 직접 구현* (Ch 24 GPT, Ch 32 diffusion) 해 본 이유 — *생성 패러다임의 지형* 을 손으로 익혀 두면 어느 쪽이 발전하든 따라갈 수 있습니다.

## 다음 챕터 예고

**Chapter 33. 작은 사전학습 Diffusion LM — MDLM (170M) + DiffuGPT (124M) 추론**

- **MDLM-owt** (`kuleshov-group/mdlm-owt`, 170M, arXiv:2406.07524) — 본 챕터가 직접 따른 *바로 그 masked diffusion 논문* 의 공식 체크포인트. `AutoModelForMaskedLM` (fill-mask) 라 본 챕터 `BertForMaskedLM` 과 *인터페이스가 거의 동일* → 코드가 매끄럽게 이어집니다. T4 여유.
- **DiffuGPT-small** (`diffusionfamily/diffugpt-s`, 124M, arXiv:2410.17891) — *가장 작은* 정식 사전학습 diffusion LM. GPT2 본체라 **Ch 24 (GPT, autoregressive) 와 같은 본체에서 AR vs diffusion 직접 비교** 가능.
- 본 챕터 작은 from-scratch 모델과 *품질 격차* 를 직접 체감 — *같은 알고리즘, 충분한 규모* 면 unconditional 생성이 얼마나 달라지는지.
- (대형 맛보기) LLaDA-8B 는 4bit 양자화로 *선택 실습*.

> **변하는 축**: *모델 출발점* (scratch 약 3.79M → 사전학습 170M / 124M). 메커니즘 (병렬 denoise) 은 본 챕터에서 이미 손으로 구현해 봤습니다. Ch 34 에서 *한국어 diffusion + autoregressive 직접 비교* 로 Phase 5 를 마무리합니다.
