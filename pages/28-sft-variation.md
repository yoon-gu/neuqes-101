본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래를 출발점으로:

### 변형 1. 더 많은 데이터 / epoch

subset 크기와 epoch 수를 늘리면 instruction 다양성이 커져 instruction following 능력이 향상됩니다. 다만 T4 + 30분 룰을 넘기지 않도록 학습 시간 증가에 주의해 값을 조절하세요.

```python
# N_SFT = 10000           # subset 확대 (T4 시간 증가 주의)
# sft_config.num_train_epochs = 3   # SFT 는 1-3 epoch 표준
# 더 많은 instruction 다양성 -> instruction following 능력 향상
```

### 변형 2. 다른 response_template

response_template 은 답변 시작 경계를 알리는 표식일 뿐이라, 영어 마커든 chat-style 마커든 자유롭게 바꿀 수 있습니다. 단 collator 가 input_ids 안에서 이 문자열을 찾으므로, 데이터에 일관되게 등장하면서 본문과 충돌하지 않는 특수한 문자열이어야 합니다.

```python
# RESPONSE_TEMPLATE = "### Answer:\n"   # 영어 마커
# RESPONSE_TEMPLATE = "<|assistant|>\n" # chat-style 마커
# response_template 은 '답변 시작 경계' 표시일 뿐 - 데이터에 일관되게만 등장하면 됨.
# 단 본문과 충돌하지 않는 특수 문자열이어야 (collator 가 input_ids 안에서 이걸 찾음).
```

### 변형 3. LoRA / QLoRA — 더 큰 모델 SFT

본체 weight 는 freeze 한 채 작은 low-rank adapter 만 학습하는 LoRA 를 쓰면 메모리를 크게 절감해 7B 급 모델도 SFT 할 수 있습니다. `SFTTrainer` 에 `peft_config` 만 넘기면 적용되며, 마스킹·loss 원리는 full SFT 와 동일하다는 점에 유의하세요.

```python
# from peft import LoraConfig
# peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"],
# lora_dropout=0.05, task_type="CAUSAL_LM")
# trainer = SFTTrainer(model=model, args=sft_config, train_dataset=sft_ds,
# processing_class=tokenizer, peft_config=peft_config)
# 본체 weight 는 freeze, 작은 adapter 만 학습 -> 메모리 대폭 절감.
# 7B 급 모델 SFT 의 실무 표준 (QLoRA 는 4bit 양자화까지 더함). 본 커리큘럼 범위 밖.
```
