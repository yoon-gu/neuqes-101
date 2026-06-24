작은 BERT 의 성능은 *학습량* 에 민감합니다. 두 가지 변형을 시뮬레이션 (실제 실행은 시간 관계상 한 셋업만):

| 변형 축 | 이번 챕터 (기본) | 변형 예 | 예상 효과 |
|---|---|---|---|
| `num_train_epochs` | 2 | 5 | eval loss 하락, perplexity 완만히 하락(8-10 epoch 이후 평탄) |
| `BLOCK_SIZE` | 128 | 64 | 블록 수 약 2배, 한 블록 짧아져 *문맥* 줄음 → loss 약간 상승 |
| `BLOCK_SIZE` | 128 | 256 | 블록 수 절반, 한 블록 길어 *문맥* 풍부 → loss 하락 가능, VRAM 약 4배 (attention $O(n^2)$) |
| `N_TRAIN_TEXT` | 5,000 paragraphs | 30,000 paragraphs | loss 큰 폭 하락, 시간 약 6배 증가 (T4 30분 룰 안에선 1 epoch 만 가능) |
| `mlm_probability` | 0.15 | 0.30 | 더 어려운 task → loss 상승, 학습 신호 증가 (논문 BERT 는 15% 가 sweet spot) |

> **T4 30분 룰 안에서 가능한 가장 큰 개선** — Wikitext-103 paragraphs 를 5K → 20K (약 4배) 로 늘리고 batch 32 유지하면 한 epoch 약 20-25분, 1 epoch 로 마무리. 이번 챕터의 *짧고 빠른* 실험 이후 직접 변형해 보세요.

### 사전학습을 더 돌리면? — 부록 곡선 (이슈 #19)

본편은 **2 epoch** 데모라 perplexity 가 높습니다. [20-4 부록 — 사전학습량과 perplexity 곡선](20-en_bert_pretrain-scaling.md)에서 2 → 16 epoch 으로 길게 학습하며 측정한 곡선을 보면, perplexity 가 **영어 1,173 → 696, 한국어 1,626 → 709 로 ~2배** 내려간 뒤 **epoch 8-10 부터 평탄** 해집니다. 5,000 텍스트로는 모델이 금세 saturate 해 *epoch 만으로는 한계* — 더 낮추려면 **데이터 양** 을 늘려야 합니다 (Ch 21 부록의 '데이터가 가장 큰 lever' 와 같은 결론). 본편의 높은 perplexity 는 버그가 아니라 *짧은 데모 학습의 정직한 반영* 입니다.
