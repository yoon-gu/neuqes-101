여기서는 **모델을 다시 학습하지 않습니다.** 방금 30000 step으로 학습한 그 가중치를 그대로 두고, 디코딩(샘플러)만 두 가지로 바꿔 결과가 어떻게 달라지는지 봅니다. 학습이 끝난 모델은 "각 자리에 어떤 토큰이 올 확률이 높은가"를 알고 있을 뿐이고, 그 확률 분포에서 실제 문장을 어떻게 뽑아낼지는 전적으로 샘플러의 몫입니다.

### 비교 A — 반복 억제 없는 단순 샘플러

가장 소박하게, 매 step 모델이 내놓은 로짓을 그대로 받아 확률이 높은 토큰부터 채워 넣습니다. 온도도 1.0, top-p도 없고, 같은 토큰을 다시 써도 막지 않습니다. 이렇게 하면 모델이 한번 "안전한" 토큰(자주 등장하는 단어나 구두점)에 높은 확률을 주기 시작했을 때 그 토큰이 계속 반복되기 쉽습니다.

실제로 이 샘플러의 **4-gram 반복률은 0.177** 입니다. 생성문을 읽어 보면 같은 짧은 구절이 돌림노래처럼 되풀이됩니다(실제 출력은 아래 비교 셀에서 확인합니다). 모델이 틀려서가 아닙니다. 다음에 올 가장 그럴듯한 토큰을 매번 충실히 고르다 보니, 한번 안전한 구절을 고르면 그 구절이 국소적으로 가장 그럴듯한 선택이 되어 버리는 함정에 빠진 것입니다.

### 비교 B — 반복 억제 샘플러 (carry-over semi-AR)

같은 모델, 같은 가중치에 다음 장치들을 더합니다.

- **temperature 0.8** — 로짓을 살짝 날카롭게 해 너무 평평한 분포에서 엉뚱한 토큰이 튀는 걸 줄입니다.
- **top-p 0.92** — 누적 확률 0.92 안쪽의 토큰만 후보로 남기는 nucleus sampling으로 꼬리쪽 저확률 토큰을 잘라냅니다.
- **repetition penalty 1.3** — 이미 써 버린 토큰의 로짓을 깎아 같은 단어가 다시 뽑힐 확률을 낮춥니다.
- **no immediate repeat** — 바로 왼쪽 토큰과 똑같은 예측을 차단해 "the the", "sorry sorry" 같은 인접 중복을 원천 봉쇄합니다.

결과는 극적입니다. **4-gram 반복률이 0.177에서 0.000으로 떨어집니다.** 앞서 돌림노래처럼 반복되던 문장이 사라지고, 인물과 대화가 있는 이야기가 나옵니다(아래 비교 셀의 실제 출력 참고).

### carry-over의 의미 — 확정한 토큰은 건드리지 않는다

이 샘플러의 핵심은 이름에 들어 있는 **carry-over** 입니다. block(32 토큰) 단위로 왼쪽에서 오른쪽으로 진행하면서, 매 step 전체를 다시 예측하되 **이미 확정한(reveal한) 토큰은 다음 step으로 그대로 이월하고 절대 바꾸지 않습니다.** 새로 채울 자리는 mask로 남은 위치 중 모델이 가장 확신하는 곳부터 고신뢰 순으로 확정합니다.

이것이 Ch 32 기본 샘플러가 남긴 반복의 직접적 교정점입니다. Ch 32의 "저신뢰 재마스킹"(기본) 샘플러는 방금 채운 토큰조차 신뢰도가 낮으면 도로 `[MASK]`로 지웠습니다. 디코딩이 단조적이지 않으니 어렵게 만든 문맥이 매 step 허물어졌습니다. carry-over는 한번 확정한 토큰을 불변으로 두어, 오른쪽으로 갈수록 확정된 왼쪽 문맥이 차곡차곡 쌓이게 만듭니다. block 마지막 step에서는 남은 자리를 전부 확정해 빈칸 없이 블록을 마무리합니다.

### "모델이 좋아야 샘플러가 산다"

여기서 한 가지를 분명히 해 둘 필요가 있습니다. 반복 억제 샘플러가 **붕괴한 모델**(삽질 코너에서 vocab·step을 되돌려 망가뜨린 ablation)을 살려내는 건 **아닙니다.** 그렇게 유니그램만 외운 모델에 이 샘플러를 붙였다면 반복은 줄었겠지만 여전히 의미 없는 문장만 나왔을 것입니다. 유니그램 marginal만 학습한 모델에는 애초에 뽑아낼 조건부 구조가 없기 때문입니다.

이번 장에서 반복 억제가 효과를 본 건 **모델이 먼저 제대로 학습됐기 때문**입니다. 고정-t(0.15) top-1 accuracy가 0.262에서 0.717로 오른, 조건부 구조를 실제로 익힌 모델 위에서만 샘플러의 미세 조정이 빛을 봅니다. 샘플러는 좋은 모델의 잠재력을 끌어낼 뿐, 없는 능력을 만들어 내지는 못합니다.

### block 크기와 온도의 trade-off

마지막으로 두 하이퍼파라미터의 균형을 짚어 둡니다.

- **block 크기** — 작게 잡으면(예: 8) 왼→오 진행이 잘게 쪼개져 더 autoregressive에 가까워지고 국소적으로 일관되지만, 한 번에 보는 미래 문맥이 좁아 전역 구성이 약해집니다. 크게 잡으면 양방향 문맥을 넓게 쓰지만 한 블록 안에서 동시에 채워야 할 자리가 많아 거칠어질 수 있습니다. 본 장은 32로 두었습니다.
- **temperature** — 낮추면(0.8) 안전하고 매끄럽지만 단조로워지고, 높이면 다양해지지만 엉뚱한 토큰이 늘어 문장이 깨질 위험이 커집니다.

정답은 한 점이 아니라 "얼마나 안전하게 vs 얼마나 다채롭게"의 저울질입니다. 작은 모델일수록 안전한 쪽으로 살짝 기울여 두는 편이 읽을 만한 결과를 줍니다.

```python
print("=== 샘플러 sweep (같은 학습 모델, 조건부 'Once upon a time') ===")
pid = tokenizer("Once upon a time", add_special_tokens=False)["input_ids"]
configs = [
    ("A) 기존 temp0.7/topk40 (반복억제 없음)", dict(temperature=0.7, top_k=40, top_p=1.0, rep_penalty=1.0, no_immediate_repeat=False)),
    ("B) rep1.3 + 인접금지 + topp0.92",        dict(temperature=0.8, top_p=0.92, rep_penalty=1.3, no_immediate_repeat=True)),
    ("C) rep1.2 + temp0.9 + topp0.95",         dict(temperature=0.9, top_p=0.95, rep_penalty=1.2, no_immediate_repeat=True)),
    ("D) B + block16 (더 촘촘)",                dict(temperature=0.8, top_p=0.92, rep_penalty=1.3, no_immediate_repeat=True, block=16)),
]
torch.manual_seed(SEED)
for name, kw in configs:
    print(f"\n----- {name} -----")
    for i in range(2):
        print(f"[{i}] {generate(model, prompt_ids=pid, **kw)[:360]}")
```

**▶ 실행 결과**

```text
=== 샘플러 sweep (같은 학습 모델, 조건부 'Once upon a time') ===

----- A) 기존 temp0.7/topk40 (반복억제 없음) -----
[0]  Once upon a time, there was a little girl named Lily. She loved to play with her toy friends. One day, Lily's mom came to play with her …(뒤 53자 생략)

Lily's mom asked her to help her mom. She asked her mom if she could play with her toy ball. Lily said, "Yes, you can play with my ball with it."

Lily and her mom said,
[1]  Once upon a time, there was a little girl named Lily. She loved to play with her toys and her friends. One day, she went to the park wi …(뒤 88자 생략)

Suddenly, Lily's friend came outside and saw a big dog playing in the park. It was a big, red red ball, and they were playing in the p

----- B) rep1.3 + 인접금지 + topp0.92 -----
[0]  Once upon a time, there was a little girl named Lily. She loved to play with her friends and explore in her backyard. One day, she went …(뒤 94자 생략)

Lily picked up and tried to grab it inside. But then, she found a rock on the ground and started to run away. 

The boy said, "I
[1]  Once upon a time, there was a little girl named Lily. She loved to play outside and play with her toys. One day, she went to the park a …(뒤 85자 생략)

"Lily's go home me!"
Her mom replied, "I'm sorry, sweet mommy. I don't want it again soon." 

The man smiled and said, "Yes, I love you. 

----- C) rep1.2 + temp0.9 + topp0.95 -----
[0]  Once upon a time, there was a little girl named Lily. She loved to play with her toys and leaves together. One day, she decided to go o …(뒤 25자 생략)

Suddenly, she saw a big red rock on the ground. She picked it up and held it tightly. Lily was sad and cried and said, "It's so cool! I won' …(뒤 57자 생략)
[1]  Once upon a time, there was a little girl named Lily. She loved to play with her toys and make pretty blankets. One day, she went to th …(뒤 55자 생략)

Later that day, Lily's mom heard her said, "I'm sorry, but you can't have some food." Her mom replied, "Don't worry, but I'll touch it." 

Lily felt sad too, but then,

----- D) B + block16 (더 촘촘) -----
[0]  Once upon a time, there was a little girl named Lily. She loved to play with her toys. One day, she was playing in the park and said th …(뒤 36자 생략)

Lily's mom came and replied, "Don't worry, Lily. I'm sorry, but you can change your clothes." Her mom replied, "I don't want to do it again! …(뒤 46자 생략)
[1]  Once upon a time, there was a little girl Lily. She loved to play with her toys and pretty blankets on her shoes. One day, she went to …(뒤 86자 생략)

Lily's mom said, "I don't know what to do it!" Her mom replied, "Of course if I have some new things for you." 

Her mom explained, "You
```

## Autoregressive(Ch 24) vs Diffusion(이 장)

같은 데이터(TinyStories), 같은 토크나이저(ByteLevel BPE vocab 2048), 거의 같은 규모(약 3.7M-3.8M params)로 만든 두 언어 모델을 나란히 놓고 보면, 생성 방식의 차이가 학습과 추론 전반에 어떻게 번지는지가 또렷이 드러납니다.

| 항목 | Autoregressive (Ch 24, GPT) | Diffusion (이 장, mask-diffusion) |
|---|---|---|
| 생성 방향 | 왼→오 단방향, 한 토큰씩 순차 | 전체 mask에서 시작해 양방향 문맥으로 병렬 채움 |
| 학습 감독 | 매 자리마다 다음 토큰 예측 — 모든 자리가 매 step 감독됨 | mask로 가린 자리만 예측 — 가린 자리에서만 학습 신호 |
| 마스킹 | 인과 마스크 고정(왼쪽만 봄) | 매 배치 확률 $t$로 토큰마다 가변 마스킹 |
| 필요 학습량 | 1500 step | 30000 step (약 20배) |
| 생성 비용 | KV-cache로 1패스 | NFE가 대략 생성 길이에 비례, 반복 디코딩 |
| 같은 규모 품질 | 상대적으로 매끄러움 | 같은 규모에서 더 거친 게 정상 |

### 왜 diffusion이 약 20배 더 많은 step을 요구하나

핵심은 **학습 신호의 밀도** 입니다. Autoregressive 모델은 길이 $L$ 문장 한 개에서 $L$개 자리 전부에 대해 "왼쪽 전체 문맥으로 다음 토큰 맞히기" 과제를 풉니다. 한 문장이 곧 $L$개의 감독 신호인 셈이라, 토큰마다 빠짐없이 기울기가 흐릅니다.

Diffusion 모델은 한 배치에서 확률 $t$로 일부 자리만 가립니다. 가리지 않은 자리는 정답이 그대로 입력에 들어 있으니 손실에 기여하지 않고(`labels`를 -100으로 두어 제외), 가린 자리에서만 학습이 일어납니다. 게다가 그 가린 자리도 "어떤 자리가 가려졌는가"가 배치마다 달라지는 양방향 빈칸 채우기여서, 한 토큰을 여러 마스킹 패턴에서 반복적으로 마주쳐야 비로소 안정적으로 익혀집니다. 자리당 감독이 희박하고 과제가 매번 달라지니, 같은 데이터를 더 여러 번 통과시켜야(여기서는 약 20배) 비슷한 수준에 도달합니다.

이건 diffusion의 결함이라기보다 **무엇과 맞바꾸는가** 의 문제입니다. 단방향 순차 예측의 빽빽한 감독을 포기하는 대신, diffusion은 양방향 문맥과 병렬 채움이라는 다른 성질을 얻습니다.

### 생성 비용 — 1패스 vs 길이에 비례한 반복

추론에서도 둘은 갈립니다. Autoregressive는 KV-cache 덕에 이미 생성한 토큰의 계산을 재활용하며 사실상 1패스로 끝까지 흘러갑니다. Diffusion은 전부 `[MASK]`인 상태에서 출발해 매 step 전체를 다시 예측하고 일부만 확정하는 과정을 되풀이하므로, 함수 평가 횟수(NFE)가 대략 생성 길이에 비례합니다. carry-over semi-AR로 block을 잘라 진행해도 step 수가 길이를 따라 늘어나는 구조는 그대로입니다.

### 무엇이 더 거친가 — 그리고 그게 정상인 이유

결과물의 결을 보면, 같은 약 3.7M 규모에서 Ch 24의 GPT는 1500 step만으로도 "there was a girl named Lily" 같은 매끄러운 문장을 냈습니다. 이 장의 diffusion은 30000 step을 들여 인물(Lily, Timmy)·대화·배경이 있는 이야기까지 도달했지만, "big collar tree", "an noise" 같은 자잘한 흠이 남습니다.

이 거칠기는 모델이 잘못 학습됐다는 신호가 아닙니다. **같은 규모에서 diffusion이 autoregressive보다 거친 건 정상** 입니다. 양방향 병렬 채움이라는 더 어려운 과제를, 더 희박한 감독으로, 작은 본체로 풀고 있기 때문입니다. 중요한 건 이 모델이 유니그램 붕괴(". the the.. was" 같은 고빈도 토큰 반복)에 빠지지 않고 조건부 구조를 실제로 학습했다는 점입니다. 고정-t(0.15) top-1 accuracy가 0.717까지 오르고, 생성 토큰 분포가 코퍼스 유니그램과 뚜렷이 다른(KL 0.78) 상태가 그 증거입니다.

### 정리

Autoregressive와 diffusion 중 하나가 일방적으로 우월한 게 아닙니다. AR은 빽빽한 감독과 KV-cache로 작은 규모·짧은 학습에서 유리하고, diffusion은 양방향 문맥과 병렬 생성이라는 다른 길을 택하는 대신 더 많은 학습량과 반복 추론을 치릅니다. 작은 모델·짧은 예산이라는 이 장의 조건에서는 AR이 더 손쉽게 매끄러운 결과를 주지만, diffusion도 레시피만 제대로 잡으면 같은 T4 30분 예산 안에서 충분히 coherent한 이야기를 만들어 낸다는 것을 이 장이 보여 줍니다.
