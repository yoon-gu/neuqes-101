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

여기서 한 가지를 분명히 해 둘 필요가 있습니다. 반복 억제 샘플러가 **collapse한 모델**(삽질 코너에서 vocab·step을 되돌려 망가뜨린 ablation)을 살려내는 건 **아닙니다.** 그렇게 유니그램만 외운 모델에 이 샘플러를 붙였다면 반복은 줄었겠지만 여전히 의미 없는 문장만 나왔을 것입니다. 유니그램 marginal만 학습한 모델에는 애초에 뽑아낼 조건부 구조가 없기 때문입니다.

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

**결과 해석** — 반복 억제 없는 A는 "red red ball", "play with her ball with it"처럼 인접·구절 반복이 보이는 반면, B·C·D는 모두 그 맴돎이 사라지고 대화가 이어지는 이야기가 나옵니다. C(temp 0.9)는 좀 더 다채롭지만 "It's so cool! I won't"처럼 흐름이 거칠고, D(block 16)는 더 촘촘해 국소적으로는 매끄럽되 작은 모델 한계로 "do it!", "have some new things" 같은 어색함은 남습니다.
