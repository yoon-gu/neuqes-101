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
