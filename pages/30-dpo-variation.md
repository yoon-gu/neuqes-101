본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래를 출발점으로:

### 변형 1. β 조정 — reference 제약 강도

```python
# dpo_config.beta = 0.5    # 제약 느슨 -> 빨리 정렬되지만 collapse 위험 (reference 에서 멀어짐)
# dpo_config.beta = 0.05   # 제약 강함 -> 안전하지만 정렬 느림
# 0.1 이 무난한 출발점. reward accuracy 가 안 오르면 beta 를 약간 올려 보세요.
```

### 변형 2. 더 많은 preference / SFT 모델에서 출발

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
