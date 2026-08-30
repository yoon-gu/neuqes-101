# 23_ko_bert_classify — 작은 BERT 분류 (한국어 NSMC 이진, 일반 도메인 사전학습 → 다른 도메인 fine-tune)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/23_ko_bert_classify/23_ko_bert_classify.ipynb)

## 한 줄 목표
Phase 3 의 마지막 챕터. Ch 22 에서 *작은 한국어 BERT 를 일반 도메인 (한국어 Wikipedia) 으로 직접 MLM 사전학습* 했다면, 이번엔 그 위에 **분류 헤드를 얹어 *완전히 다른 도메인 (NSMC 영화 리뷰)* 이진 분류로 fine-tune**. Ch 15 (`klue/bert-base`, 약 110M params, 약 8.4B 토큰 대규모 한국어 사전학습) 와 같은 NSMC 분류 셋업에 *우리가 만든 작은 BERT* (약 10M params, 한국어 Wikipedia 2K paragraphs × 3 epoch MLM) 를 붙여 두 결과를 나란히 비교 — 둘 다 *일반 한국어 사전학습 → NSMC transfer* 라 비교가 *fair*, *사전학습 규모* 차이만 측정됨.

self-contained 노트북: 한국어 Wikipedia MLM 학습을 짧게 재현 → 같은 본체로 NSMC 분류 fine-tune → Ch 15 와 2-way 비교. *random init baseline 비교 + negative transfer 분석* 은 부록 [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb) 으로 분리.

## 다루는 핵심 개념
- **일반 한국어 위키 사전학습 → NSMC 영화 리뷰 분류 transfer** — 원본 BERT 정신의 한국어 대칭본 (Ch 21 의 영어 패턴을 한국어 환경에서 재확인)
- **두 데이터셋이 노트북 안에 공존** — MLM 용 한국어 Wikipedia + 분류용 NSMC. 같은 토크나이저로 처리
- `BertForMaskedLM` -> `BertForSequenceClassification` 헤드 교체 — 본체 (`embeddings + encoder + pooler`) 는 그대로, MLM head 떼고 분류 head (`Linear(256, 2)`) 부착
- in-memory state_dict 전송: `cls_model.bert.load_state_dict(mlm_model.bert.state_dict())` — 디스크 없이 본체 가중치 복사
- 같은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024, 약 10M params) 가 MLM 모델과 분류 모델 양쪽에 적용
- **2-way 비교**: Ch 15 (`klue/bert-base`, 약 110M, 약 8.4B tokens) vs Ch 23 ours (small + ko wiki MLM) — 같은 *위키 → NSMC transfer* 패턴이라 비교가 fair
- 부록 [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb): random init baseline + 한국어 환경 특유의 negative transfer 분석
- `labels = -100` thread 한 줄 환기 (MLM 만 사용, 분류는 사용 안 함) + 파인튜닝 의미 변화 (BERT vs GPT) 예고 — Phase 4 Ch 24 시작

## Loss
`CrossEntropyLoss` — 분류 fine-tune 표준 (K=2, softmax + CE). 라벨은 `int 0/1`, `problem_type="single_label_classification"`. random baseline loss = `ln(2)` 약 0.693.

수식: $L = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i, y_i}$ — Ch 11 / Ch 15 / Ch 21 과 같은 K-class softmax CE.

## 데이터

| 단계 | 데이터셋 | 용도 |
|---|---|---|
| MLM 사전학습 | `wikimedia/wikipedia`, `20231101.ko` 2K paragraphs × 3 epoch (eval 400) | self-supervised MLM, 일반 한국어 위키 본문 |
| 분류 fine-tune | NSMC (e9t/nsmc GitHub raw TSV) 5K train / 1K eval, seed 42 | supervised 이진 분류 (긍정/부정 라벨) |

같은 토크나이저 (`klue/bert-base`) 가 두 도메인의 텍스트를 처리. `block_size=128` `group_texts` 패턴으로 MLM 3 epoch + NSMC 분류 fine-tune 2 epoch.

## 환경
Google Colab T4 GPU (fp16). 약 2-4분 — 대부분이 데이터 다운로드입니다 (T4 실측 전체 약 2분: 한국어 Wikipedia·NSMC 다운로드·전처리 약 1분 30초 + MLM 3 epoch 약 0.2분 + 분류 fine-tune 2 epoch 약 0.2분 + 평가/시각화 수 초). 부록 [`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb) 도 같은 규모로 별도 약 2-4분.

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 15 | `klue/bert-base` 파인튜닝 (약 110M) | WordPiece (한국어, 사전학습) | NSMC (이진) | `Linear(H, 2)` | `CrossEntropyLoss` |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Wikitext-103 (일반 도메인) | MLM head | `CrossEntropyLoss` (masked) |
| 21 | Ch 20 사전학습 BERT + 분류 헤드 | (Ch 20과 동일) | Yelp 이진화 (다른 도메인 transfer) | `Linear(H, 2)` | `CrossEntropyLoss` |
| 22 | 작은 BERT (직접, scratch) — 한국어 | `klue/bert-base` 토크나이저 (가져옴) | 한국어 Wikipedia (일반 도메인) | MLM head | `CrossEntropyLoss` (masked) |
| **23** | **Ch 22 사전학습 BERT + 분류 헤드 (약 10M)** | **(Ch 22와 동일)** | **NSMC 이진 (다른 도메인 transfer)** | **`Linear(H, 2)`** | **`CrossEntropyLoss`** |
| 24 (다음, Phase 4) | GPT-2 (직접, scratch) | BPE 토크나이저 (직접 학습) | TinyStories 영어 동화 | LM head | `CrossEntropyLoss` (causal LM) |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 비교 표 — 2-way (둘 다 일반 한국어 위키 → NSMC transfer 라 fair)

| 차원 | Ch 15 (klue/bert-base) | Ch 23 ours (small + MLM) |
|---|---|---|
| 본체 파라미터 | 약 110M | 약 10M |
| 사전학습 코퍼스 | 한국어 위키 + 모두의 말뭉치 + 뉴스 + 댓글 (약 8.4B 토큰) | 한국어 Wikipedia paragraphs 2K (1,562 블록 ≈ 약 20만 토큰) |
| 사전학습 시간 | TPU 수일 | T4 약 0.2분 (MLM 3 epoch 실측) |
| Fine-tune 도메인 | NSMC 이진 (다른 도메인) | NSMC 이진 (다른 도메인) |
| 분류 fine-tune 셋업 | (둘 다 같음 — 5K/1K, batch 16, lr 2e-5, 2 epoch, fp16) | |
| 기대 accuracy | 약 86% (`executed/15_ko_binary.ipynb`) | 약 55% (T4 실측 — 동전 던지기에 가까운 자리) |

비교가 *공정* 한 이유 — Ch 15 도 Ch 23 ours 도 둘 다 *일반 도메인 한국어 사전학습 → NSMC 분류 transfer* 의 같은 패턴이라 *사전학습 규모* (약 4만 배) 와 *모델 크기* (11배) 만 변수. 격차가 *사전학습 규모의 가치* 를 정량으로 보여줍니다.

## 부록 — random init baseline + negative transfer 분석
[`appendix_random_baseline.ipynb`](./appendix_random_baseline.ipynb) — *MLM 사전학습 없이 random init 으로 바로 분류 fine-tune* 한 결과와의 비교 + *한국어 위키 → NSMC 의 큰 도메인 gap* 에서 발생할 수 있는 **negative transfer** 현상의 메커니즘 분석. 영어 Ch 21 (transfer 양성) 과 한국어 Ch 23 (transfer 음성 가능) 의 비대칭 메커니즘이 핵심. 변형 옵션 (위키 양 늘림 / DAPT / seed 평균 / lr 조정) 4가지로 negative transfer 극복 실험까지.

## 다음 챕터
[24_gpt_tinystories](../24_gpt_tinystories/) — Phase 4 시작. *encoder (BERT) → decoder-only (GPT)*, *MLM → Causal LM*, *task별 head 부착 파인튜닝 → SFT / behavior alignment*. BERT 시대의 *task head 부착* 패러다임은 본 챕터에서 마무리, Phase 4 부터는 *GPT 본체 + LM head 그대로 + 행동 정렬* 흐름. 영어 BERT scratch (Ch 20-21) → 한국어 BERT scratch (Ch 22-23) → 영어 GPT scratch (Ch 24-) 의 대칭 구조.
