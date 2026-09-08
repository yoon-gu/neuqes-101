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
옛날 옛날에 작은 토끼가 있었어요. 그 새는 매우 좋아했지요. 어느 날, 새는 큰 나무를 봤어요. 그 새는 그 새는 매우 슬퍼했죠. 새는 매우 기뻐하며 "안녕, 새야! 나는 너를 사랑해."라고 말했죠. 새는 "안녕, 나는 너를 사랑해!"라고 말했죠. …(뒤 26자 생략)

======================================================================
[T=0.8, top_k=50  (balanced)]
옛날 옛날에 작은 토끼가 있었어요. 그 친구들은 매우 좋아했지요. 어느 날, 새는 큰 소리를 들었어요. 새는 큰 그 새는 나무 밑에서 놀고 있었어요. 새는 날지로 가고 싶었습니다. 새는 "저건 뭐지?"라고 물었어요. 새는 무서웠지만, 새는 슬퍼했죠. …(뒤 27자 생략)

======================================================================
[T=1.0, top_p=0.9 (nucleus)]
옛날 옛날에 작은 토끼가 있었어요. 그 친구들은 뛰는 것을 매우 좋아했죠. 그들은 젠프를 얻으라고 생각했어요. 어느 날, 새는 정원에 떨어지고 있는 큰 난장판을 만들었어요. 둘은 질문하고 싶었어요. 토끼는 밖으로 나가며 새 총겼지만, 새로운 게임을 하 …(뒤 32자 생략)

======================================================================
[T=1.2, top_k=100 (diverse)]
옛날 옛날에 작은 토끼가 있었어요. 그 친구들은 길을 찾을 수 있었어요. 그들은 큰 빨간 공으로 노는 걸 보려고 그 차를 보았어요. 보물들은 매우 기뻐했답니다. "아니요, 이 개는 정말 멋진 일을 하며요!"라고 고양이는 "우리, 네 거란다."라고 팀이 …(뒤 23자 생략)
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

*학습이 충분히 잘 된* 기준점으로 `skt/kogpt2-base-v2` (125M, 대규모 한국어 사전학습) 에 같은 한국어 prompt 를 넣어 *우리 작은 GPT (약 4.2M, 한국어 TinyStories 30K)* 와 격차를 봅니다. **Ch 27 이 KoGPT2 본격 챕터** 이므로 여기서는 *간단히 한 번만* — T4 시간을 아끼려면 이 셀은 건너뛰어도 됩니다.

마지막 (선택) 셀은 비교 기준점으로 대규모 한국어 사전학습 모델 KoGPT2(125M)에 같은 prompt를 넣어 봅니다. 토크나이저는 `AutoTokenizer`가 영어 GPT2 토크나이저로 잘못 fallback 하므로(transformers 5.x) `PreTrainedTokenizerFast`로 special token을 직접 지정해 로드합니다 — Ch 27과 같은 방식입니다. 시간이 부족하면 `RUN_KOGPT2_REF = False`로 두고 건너뛸 수 있습니다.

```python
# 선택 셀 - KoGPT2 reference. 시간이 부족하면 RUN_KOGPT2_REF = False 로 두고 건너뜁니다.
RUN_KOGPT2_REF = True

if RUN_KOGPT2_REF:
    from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

    print("loading reference KoGPT2 (skt/kogpt2-base-v2, 125M)...")
    # 주의: KoGPT2 는 AutoTokenizer 가 영어 GPT2 토크나이저로 잘못 fallback 합니다
    # (transformers 5.x — 한국어 prompt 가 깨진 byte 로 인코딩돼 generation 이 �·▁ 나열로 나옴).
    # SKT 공식 방식대로 PreTrainedTokenizerFast 로 special token 을 직접 지정해 로드합니다 (Ch 27 과 동일).
    ref_tok = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>",
    )
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
pytorch_model.bin: downloading bytes:           |  0.00B            
[transformers] GPT2LMHeadModel LOAD REPORT from: skt/kogpt2-base-v2
Key                                     | Status     |  | 
----------------------------------------+------------+--+-
transformer.h.{0...11}.attn.masked_bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
model.safetensors: downloading bytes:           |  0.00B            
  #params : 125.2 M

======================================================================
REFERENCE KoGPT2 (125M) - generation on same Korean prompts
======================================================================
prompt: 옛날 옛날에
옛날 옛날에 네들이 말했잖아. 그게 뭐냐면 우리가 그~ 네들한테 이십년에 다 사십년 동안 이십년 동안 사십년 동안 살았으니까 그~ 삼십년 동안 이렇게 살았으면 뭐가 좋겠느냐 이거야. 그런
prompt: 작은 소녀가
작은 소녀가 된 후 소녀에게 한 번도 성적으로 관심을 갖지 않았다.
그녀가 소녀가 된 후에는 그녀의 사랑을 알게 되었다.
그녀가 고등학교에 진학했을 때, 그들은 소녀를 볼 때마다 미소를 짓고 그녀를 바라봤다.
마치 그처럼 그녀는 소녀처럼 소녀를 보았다.
그때마다 소녀는 소녀를 위해 무엇인가를 해 주고 있었다.
prompt: 큰 개가
큰 개가 되어버린 것입니다.
그래서 모든 것을 초월하여 하나로 통합하고, 하나로 통합하는 것이 바로 그 무엇이라는 것을 강조하고 있는 것입니다.
하나의 힘을 하나로 통합하는 것이 바로 통합인 것입니다.
우리는 하나님의 말씀을 통해서 하나님의 능력을 충분히 깨닫습니다.
그 능력은 하나님의 말씀을 통해서 우리에게 전달되기도 하고, 또 하나
prompt: 어느 날,
어느 날, 김정은은 노동당 대표자회에서 '국지전'을 펼쳤다고 김정은은 밝혔다.
하지만 김정은은 '국지전' 대신 '국지전'이라는 표현을 쓰면서 '국지전'에 대한 언급을 피했다.
북한은 김정은이 국지전 연설을 하며 '국지전
```

**결과 해석**

KoGPT2는 네 prompt 모두 문법이 자연스러운 한국어 문장을 이어 갑니다. 다만 어휘와 도메인은 동화가 아니라 구어 전사·소설·설교·뉴스 등 사전학습 코퍼스 쪽으로 흩어집니다 — 규모와 다양성이 만든 격차이지 동화 도메인 적합성은 아닙니다. 같은 prompt에 동화 풍 한국어를 내는 것은 Ch 27에서 이 모델을 한국어 TinyStories로 continual pretraining 한 뒤에 확인합니다.

**해석 가이드 - 규모가 만든 격차**

- **OURS (약 4.2M, 한국어 TinyStories 30K)**: *동화 풍 단순 한국어* - 어휘는 동화 도메인에 강하지만 *복잡한 문장 구조 / 추상적 어휘* 는 약함.
- **REF (KoGPT2 125M, 대규모 한국어 코퍼스)**: *다양한 도메인 어휘 + 자연스러운 문장 흐름*. 학습 데이터의 규모·다양성이 generation 다양성으로 직결.

> Ch 27 이 이 격차를 *데이터 축을 통제하고* 다룹니다 - KoGPT2 (125M) 의 사전학습 *위에* 같은 한국어 TinyStories 로 **continual pretraining**. *대규모 한국어 사전학습 모델을 작은 도메인 데이터로 적응* 시킬 때의 generation 품질이, 우리 from-scratch 작은 GPT 와 어떻게 다른지 직접 비교 (Ch 24→25 의 한국어 짝).
