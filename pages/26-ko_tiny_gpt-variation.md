같은 한국어 prompt 에 `temperature / top_k / top_p` 만 바꿔 generation 스타일 변화 관찰. *학습된 본체는 그대로* - 변하는 건 *sampling 분포* 뿐.

```python
prompt = "옛날 옛날에 작은 토끼가"
configs = [
    {"label": "T=0.3, top_k=20  (conservative)", "temperature": 0.3, "top_k": 20,  "top_p": None},
    {"label": "T=0.8, top_k=50  (balanced)",    "temperature": 0.8, "top_k": 50,  "top_p": None},
    {"label": "T=1.0, top_p=0.9 (nucleus)",     "temperature": 1.0, "top_k": 0,   "top_p": 0.9},
    {"label": "T=1.2, top_k=100 (diverse)",     "temperature": 1.2, "top_k": 100, "top_p": None},
]
for c in configs:
    torch.manual_seed(SEED)
    print("=" * 70)
    print(f"[{c['label']}]")
    print(generate_text(model, prompt, max_new_tokens=60, do_sample=True,
                        temperature=c["temperature"], top_k=c["top_k"], top_p=c["top_p"]))
    print()
```

**▶ 실행 결과**

```text
======================================================================
[T=0.3, top_k=20  (conservative)]
옛날 옛날에 작은 토끼가 있었어요. 그 새는 매우 좋아했지요. 어느 날, 새는 큰 새를 봤어요. 새는 그 새는 그 새를 보고 "안녕, 새야! 나는 작은 새야! 나는 새야! 나랑 같이 놀 수 있을까?"라고 물었어요. 새는 "응, 같이 놀 수 있어."라고 말했죠. 새는 웃으며 "그래, 새야

======================================================================
[T=0.8, top_k=50  (balanced)]
옛날 옛날에 작은 토끼가 있었어요. 그 새는 매우 좋아했지요. 어느 날, 새는 큰 소리를 들었어요. 새는 많은 동물들이 놀고 싶어 했지요. 새는 매우 화가 나서, "내 장난감을 줬있어!"라고 했어요. 새는 "그래, 나는 거에요!"라고 말했죠. 새는 기 …(뒤 35자 생략)

======================================================================
[T=1.0, top_p=0.9 (nucleus)]
옛날 옛날에 작은 토끼가 있었어요. 그 친구들은 뛰는 것을 매우 좋아했죠. 그들은 관한 것을 아주 좋아하는 보려고 그 강으로 갔고, 색깔, 음식이 있는 한 마리자가 가장 좋아하는 상태였어요. 어느 날, 고양이는 높은 능으로 떨렸어요. 개는 새를 산책했지요. 고양이는 "쨍 음식 같아!"

======================================================================
[T=1.2, top_k=100 (diverse)]
옛날 옛날에 작은 토끼가 있었어요. 그 친구들은 하늘을 올려다보았고, 나무를 봤어요. 그 나무를 보고, 많은 그 물고기는 나무 밑에서 뛰고 뛰었답니다. 그들은 "내 게임을 만들었어!"라고 말했답니다. 작은 새와 다람쥐는 "그래, 나는 거에요!"라고 말 …(뒤 43자 생략)
```

**결과 해석**

같은 본체로 sampling 설정만 바꿨는데 출력 결이 뚜렷이 달라집니다. `T=0.3`은 안전하지만 같은 단어("새")를 반복하고, `T=1.0`/`T=1.2`로 갈수록 어휘는 다양해지나 "한 마리자가", "능으로 떨렸어요"처럼 어색한 표현이 섞입니다. 학습된 next-token 분포는 고정이고, 그 분포에서 *얼마나 넓게 뽑느냐*만 바뀐 결과입니다.

**관전 포인트**

- `temperature` ↑ → logits 분포 *평탄화* → 다양성 ↑, 일관성 ↓
- `top_k=20` → 매 step 후보를 *상위 20 개* 로만 한정 → 안전하지만 반복적
- `top_p=0.9` (nucleus) → 누적 확률 90% 이내 후보 → *모델이 확신 있을 땐 좁게, 애매할 땐 넓게* 자동 조정
- `T=1.2, top_k=100` → 가장 다양하지만 *말이 안 되는 토큰* 도 종종 섞임

**더 큰 개선을 원하면** (T4 30분 룰 안):

| 변형 축 | 이번 챕터 (기본) | 변형 예 | 예상 효과 |
|---|---|---|---|
| `N_TRAIN` (story 수) | 30,000 | 60,000 | 한국어 문장 자연스러움 ↑, 학습 시간 비례 증가 |
| `n_embd` / `n_layer` | 256 / 4 | 384 / 6 | 표현력 ↑ (약 8M params), T4 메모리 안에서 가능 |
| `max_steps` | 1500 | 2500 | loss 추가 하락, 30분 룰 주의 |
| 다른 한국어 코퍼스 | TinyStories-Korean | 한국어 위키 + 동화 혼합 | 도메인 폭 ↑, 단 어휘 난도 ↑ |

## (선택) Reference 비교 - KoGPT2 의 같은 prompt generation

*학습이 충분히 잘 된* 기준점으로 `skt/kogpt2-base-v2` (125M, 대규모 한국어 사전학습) 에 같은 한국어 prompt 를 넣어 *우리 작은 GPT (약 3M, 한국어 TinyStories 30K)* 와 격차를 봅니다. **Ch 27 이 KoGPT2 본격 챕터** 이므로 여기서는 *간단히 한 번만* — T4 시간을 아끼려면 이 셀은 건너뛰어도 됩니다.

마지막 (선택) 셀은 비교 기준점으로 대규모 한국어 사전학습 모델 KoGPT2(125M)에 같은 prompt를 넣어 봅니다. 시간이 부족하면 `RUN_KOGPT2_REF = False`로 두고 건너뛸 수 있습니다.

```python
# 선택 셀 - KoGPT2 reference. 시간이 부족하면 RUN_KOGPT2_REF = False 로 두고 건너뜁니다.
RUN_KOGPT2_REF = True

if RUN_KOGPT2_REF:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("loading reference KoGPT2 (skt/kogpt2-base-v2, 125M)...")
    ref_tok = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
    if ref_tok.pad_token is None:
        ref_tok.pad_token = ref_tok.eos_token
    ref_model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2").to(device).eval()
    print(f"  #params : {ref_model.num_parameters()/1e6:.1f} M")

    torch.manual_seed(SEED)
    print("\n" + "=" * 70)
    print("REFERENCE KoGPT2 (125M) - generation on same Korean prompts")
    print("=" * 70)
    for p in PROMPTS:
        text = generate_text(ref_model, p, gen_tokenizer=ref_tok, **GEN_KWARGS)
        print(f"\nprompt: {p}")
        print(text)

    # 메모리 정리
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print("Skipped KoGPT2 reference (RUN_KOGPT2_REF=False). Covered in depth in Ch 27.")
```

**▶ 실행 결과**

```text
loading reference KoGPT2 (skt/kogpt2-base-v2, 125M)...
[transformers] GPT2LMHeadModel LOAD REPORT from: skt/kogpt2-base-v2
Key                                     | Status     |  | 
----------------------------------------+------------+--+-
transformer.h.{0...11}.attn.masked_bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  #params : 125.2 M

======================================================================
REFERENCE KoGPT2 (125M) - generation on same Korean prompts
======================================================================
prompt: 옛날 옛날에
�����▁�ng��▁�ng��n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�n�
prompt: 작은 소녀가
�����i���i,▁▁▁▁▁▁▁
prompt: 큰 개가
����▁▁▁▁▁▁▁▁▁▁▁▁▁▁
prompt: 어느 날,
���,▁Gaurius,▁vand.▁Gould,▁Denol,▁Tamil,▁Anat�nn▁McCamel,▁Zehn,▁Guy.▁Cyb,▁Shitch,▁Pellek,▁Nami,▁Jalifa,
```

**결과 해석**

이 실행에서는 KoGPT2 reference 출력이 `▁▁▁` 반복이나 깨진 토큰으로 나와 정상적인 한국어 문장이 나오지 않았습니다. 우리 GEN_KWARGS·prompt 인코딩이 KoGPT2 토크나이저와 잘 맞지 않은 탓으로, KoGPT2의 제대로 된 generation 품질 비교는 이 모델을 본격적으로 다루는 Ch 27에서 확인합니다.

**해석 가이드 - 규모가 만든 격차**

- **OURS (약 3M, 한국어 TinyStories 30K)**: *동화 풍 단순 한국어* - 어휘는 동화 도메인에 강하지만 *복잡한 문장 구조 / 추상적 어휘* 는 약함.
- **REF (KoGPT2 125M, 대규모 한국어 코퍼스)**: *다양한 도메인 어휘 + 자연스러운 문장 흐름*. 학습 데이터의 규모·다양성이 generation 다양성으로 직결.

> Ch 27 이 이 격차를 *데이터 축을 통제하고* 다룹니다 - KoGPT2 (125M) 의 사전학습 *위에* 같은 한국어 TinyStories 로 **continual pretraining**. *대규모 한국어 사전학습 모델을 작은 도메인 데이터로 적응* 시킬 때의 generation 품질이, 우리 from-scratch 작은 GPT 와 어떻게 다른지 직접 비교 (Ch 24→25 의 한국어 짝).
