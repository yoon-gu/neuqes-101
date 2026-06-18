본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래를 출발점으로:

### 변형 1. 더 많은 데이터 / epoch

```python
# N_SFT = 10000           # subset 확대 (T4 시간 증가 주의)
# sft_config.num_train_epochs = 3   # SFT 는 1-3 epoch 표준
# 더 많은 instruction 다양성 -> instruction following 능력 향상
```

### 변형 2. 다른 response_template

```python
# RESPONSE_TEMPLATE = "### Answer:\n"   # 영어 마커
# RESPONSE_TEMPLATE = "<|assistant|>\n" # chat-style 마커
# response_template 은 '답변 시작 경계' 표시일 뿐 - 데이터에 일관되게만 등장하면 됨.
# 단 본문과 충돌하지 않는 특수 문자열이어야 (collator 가 input_ids 안에서 이걸 찾음).
```

### 변형 3. LoRA / QLoRA — 더 큰 모델 SFT

```python
# from peft import LoraConfig
# peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"],
# lora_dropout=0.05, task_type="CAUSAL_LM")
# trainer = SFTTrainer(model=model, args=sft_config, train_dataset=sft_ds,
# processing_class=tokenizer, peft_config=peft_config)
# 본체 weight 는 freeze, 작은 adapter 만 학습 -> 메모리 대폭 절감.
# 7B 급 모델 SFT 의 실무 표준 (QLoRA 는 4bit 양자화까지 더함). 본 커리큘럼 범위 밖.
```
