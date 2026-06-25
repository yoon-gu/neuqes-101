조건부 생성은 *문맥(prompt)이 있어* unconditional 보다 잘 동작합니다. 작은 모델의 강점 영역.

GPT 의 prompt 에 대응하는 diffusion 버전: *앞부분 토큰을 고정* (절대 마스킹 안 함) 하고 *나머지만* denoise. "Once upon a time" 을 주고 뒤를 채우게 합니다.

```python
prompt = "Once upon a time"
prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

torch.manual_seed(SEED)
print(f"prompt (fixed): {prompt}")
print("=" * 70)
for i in range(3):
    text = diffusion_generate(model, length=48, steps=16, prompt_ids=prompt_ids)
    print(f"\n[sample {i}] {text}")
```

**▶ 실행 결과**

```text
prompt (fixed): Once upon a time
======================================================================

[sample 0]  Once upon a time, there was a little girl named Lily. She loved to play with her with her toys. One day, she went to to play wit …(뒤 74자 생략)
[sample 1]  Once upon a time, there a a little girl named Lily. She loved to play with her toys every toys. One day, Lily went her park to p …(뒤 65자 생략)

[sample 2]  Once upon a time, there was a little girl named Lily. She was very happy and loved to play with her friends. One day, Lily's fri …(뒤 81자 생략)
```

**관전 포인트** - 앞 토큰들이 고정된 채 뒤가 채워집니다. 단, diffusion 은 *양방향* 이라 GPT 와 달리 *prompt 앞이나 중간에 빈칸* 을 두고 채우게 할 수도 있습니다 (infilling) — autoregressive 가 구조적으로 못 하는 일.

## 변형 2 - denoise step 수 비교 (속도 - 품질 trade-off)

diffusion 만의 자유도: *생성 step 수* 를 바꿀 수 있습니다. GPT 는 토큰 수 = step 수로 고정이지만, diffusion 은 *적은 step (빠르지만 거침) ↔ 많은 step (느리지만 정교)* 을 조절합니다.

```python
torch.manual_seed(SEED)
for steps in [1, 4, 16, 32]:
    torch.manual_seed(SEED)
    text = diffusion_generate(model, length=48, steps=steps)
    print(f"[steps={steps:>2d}] {text}\n")
```

**▶ 실행 결과**

```text
[steps= 1] ." very to "
 her She to
. said the with." big day his
 mom
s! and. said a The with " on
 a, his happy very, thes to I the They He in in and it

[steps= 4]  and asked,. They ran to the park. Maybe the other people was gone.


The end and. It was not friendly..
Finally, the, said, the child said, the bird!" The person and Sam
[steps=16]  and go in their house. Sam are scared. They want to play Tim and Mia. They want to climb it.

"See, Sam, Sam, Sam!" Tom says.

"Oh you!" Tom and Sam
[steps=32]  and eat each other. They are very happy. They are happy and happy. Lily and Ben are friends and friends. They like to the park. …(뒤 67자 생략)
```

**관전 포인트**
- `steps=1` - 전부 `[MASK]` 를 *한 번에* 복원. 문맥 정보가 없어 *서로 안 맞는 단어들* 이 섞이기 쉬움 (각 자리가 독립적으로 예측되니 일관성 ↓).
- `steps=16-32` - 확신 높은 자리부터 단계적으로 확정 → 이미 채운 단어가 *다음 자리의 문맥* 이 되어 일관성 ↑.

> diffusion 생성 품질의 핵심 = *step 수*. 적은 step 은 빠르지만 거칠고, 많은 step 은 느리지만 정교 — 실전 모델 (LLaDA 등) 도 이 trade-off 를 조절합니다.

## Autoregressive (Ch 24) vs Diffusion (본 챕터) 비교

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

> **왜 diffusion 이 주목받는가**: ① *병렬 생성* 으로 잠재적 속도 이점 (autoregressive 는 토큰 수만큼 순차), ② *양방향 문맥* 으로 infilling·편집에 강점, ③ step 수로 *속도-품질* 을 추론 시점에 조절. 아직 autoregressive 만큼 성숙하진 않지만 *대안 패러다임* 으로 빠르게 발전 중입니다. Ch 33 에서 *사전학습된 작은 diffusion LM (MDLM 170M / DiffuGPT 124M)* 으로 제대로 된 생성을, Ch 34 에서 *한국어 diffusion + AR 직접 비교* 를 다룹니다.

## 이 챕터 알고리즘의 논문 계보

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

> 본 챕터는 *단순화판* 입니다 — 실제 LLaDA 는 semi-autoregressive remasking 등 변형, 대규모 사전학습, 정교한 스케줄을 더합니다. 하지만 *핵심 메커니즘 (가변 마스킹 + `1/t` loss + confidence 병렬 denoise)* 은 동일하므로, 본 챕터를 손으로 구현해 보면 위 논문들의 알고리즘 절을 그대로 읽어낼 수 있습니다.
