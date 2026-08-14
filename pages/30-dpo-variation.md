본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래를 출발점으로:

### 변형 1. β 조정 — reference 제약 강도

`dpo_config.beta` 값을 바꿔 reference 제약 강도를 조절해 봅니다. 너무 키우면 reference 근처에 묶여 정렬이 느려지고, 너무 낮추면 빨리 정렬되지만 reference 에서 멀어져 collapse 위험이 커지는 trade-off 를 직접 관찰하세요.

```python
# dpo_config.beta = 0.5    # 제약 강함 -> reference 근처에 묶여 안전하지만 정렬 느림
# dpo_config.beta = 0.05   # 제약 느슨 -> 빨리 정렬되지만 collapse 위험 (reference 에서 멀어짐)
# 1 이 무난한 출발점. reward accuracy 가 안 오르면 beta 를 약간 낮춰 보세요 (제약 완화).
```

### 변형 2. 더 많은 preference / SFT 모델에서 출발

subset 크기를 키우거나 출발 모델을 base 대신 Ch 28 SFT 체크포인트로 바꾸는 변형입니다. 특히 SFT 모델에서 DPO 를 시작해야 *지시 따름* 위에 *선호만* 얹히므로 정렬 효과가 훨씬 또렷해집니다.

```python
# N_DPO = 5000              # subset 확대 (T4 시간 증가 주의)
# SFT_MODEL = "./out_kogpt2_sft"   # Ch 28 SFT 체크포인트에서 출발 (정석)
# SFT 모델에서 DPO 를 시작해야 '지시 따름' 위에 '선호' 만 정렬됩니다.
```

### 변형 3. DPO 변종 — IPO / KTO / ORPO

`trl` 은 DPO 의 여러 변종을 `loss_type` 으로 지원합니다:

```python
# dpo_config.loss_type = "ipo"   # IPO: sigmoid 대신 squared loss (overfit 완화)
# dpo_config.loss_type = "kto_pair"  # KTO 계열: chosen/rejected 가 쌍이 아니어도 됨
# ORPO: SFT + preference 를 한 번에 (reference 불필요) - trl 의 ORPOTrainer
# 각 변종은 'preference 를 어떻게 loss 로 바꾸나' 의 변주. 핵심 (chosen 선호 ↑) 은 동일.
```

> IPO 는 *DPO 의 overfitting* 을, KTO 는 *쌍이 아닌 개별 좋음/나쁨 라벨* 을, ORPO 는 *reference 없이 SFT 와 동시* 정렬을 노립니다. 모두 *preference 로 정렬* 한다는 점은 같고, *loss 형태·데이터 요구* 만 다릅니다.
