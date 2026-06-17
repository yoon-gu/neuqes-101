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
Once upon a time, a little rabbit named Lily. She loved to play with her toys and play with her toys. One day, Lily's mom saw a big tree wit …(뒤 113자 생략)

Lily was sad

======================================================================
[T=0.8, top_k=50  (balanced)]
Once upon a time, a little rabbit who liked to play in the park. One day, his friends found a big smile on the park. And it was very happy t …(뒤 35자 생략)

One day, Timmy saw a small house, Timmy's mom a shiny tree in a big tree. Timmy was so

======================================================================
[T=1.0, top_p=0.9 (nucleus)]
Once upon a time, a little rabbit who liked to read water to play the squirrel. One day, Joe's mommy saw a window, who was green bunny, but they felt sad. 

She did not know what. She liked to do with some other adventure. She would a lot of fun at her problege. She

======================================================================
[T=1.2, top_k=100 (diverse)]
Once upon a time, a little rabbit who liked to read water to play the toys. One day, Joe's mommy saw fool, who was gone to explore it. They …(뒤 136자 생략)
```

**관전 포인트**

- `temperature` ↑ → logits 분포 *평탄화* → 다양성 ↑, 일관성 ↓
- `top_k=20` → 매 step 후보를 *상위 20 개* 로만 한정 → 안전하지만 반복적
- `top_p=0.9` (nucleus) → 누적 확률 90% 이내 후보 → *모델이 확신 있을 땐 좁게, 애매할 땐 넓게* 자동 조정
- `T=1.2, top_k=100` → 가장 다양하지만 *말이 안 되는 토큰* 도 종종 섞임
