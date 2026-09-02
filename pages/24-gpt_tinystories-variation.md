같은 prompt 에 `temperature / top_k / top_p` 만 바꿔 generation 스타일 변화 관찰. *학습된 본체는 그대로* - 변하는 건 *sampling 분포* 뿐.

```python
prompt = "Once upon a time, a little rabbit"
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
Once upon a time, a little rabbit named Timmy. Timmy loved to play with his friends. One day, Timmy's mom asked him to play with his friends …(뒤 141자 생략)

======================================================================
[T=0.8, top_k=50  (balanced)]
Once upon a time, a little rabbit who liked to play in the park. One day, his friends found a big smile on the ground. And they decided to g …(뒤 34자 생략)

One day, the kitchen couldn't find a big boy named Timmy. Timmy was so happy because he saw a big

======================================================================
[T=1.0, top_p=0.9 (nucleus)]
Once upon a time, a little rabbit who liked to read water to play the squirrel. One day, Joe's mommy saw a window, who was gone. He wanted t …(뒤 21자 생략)

"Sure, Mr. That is here, Tom! I have a towel," replied. "Mama," he said

======================================================================
[T=1.2, top_k=100 (diverse)]
Once upon a time, a little rabbit who liked to read water to play the toys. day his friends found a big smile on the ball. And it made it ve …(뒤 156자 생략)
```

**결과 해석**

`T=0.3, top_k=20` 은 "play with his friends" 가 반복될 만큼 안전하지만 단조롭고, `T=0.8` 은 적당히 다양하면서 문장이 이어집니다. `T=1.0, top_p=0.9` 와 `T=1.2, top_k=100` 으로 갈수록 "read water" 처럼 말이 안 되는 조합이 섞이며, 학습된 본체는 그대로인 채 sampling 분포만으로 다양성↔일관성 trade-off 가 조절됨이 드러납니다.

**관전 포인트**

- `temperature` ↑ → logits 분포 *평탄화* → 다양성 ↑, 일관성 ↓
- `top_k=20` → 매 step 후보를 *상위 20 개* 로만 한정 → 안전하지만 반복적
- `top_p=0.9` (nucleus) → 누적 확률 90% 이내 후보 → *모델이 확신 있을 땐 좁게, 애매할 땐 넓게* 자동 조정
- `T=1.2, top_k=100` → 가장 다양하지만 *말이 안 되는 토큰* 도 종종 섞임
