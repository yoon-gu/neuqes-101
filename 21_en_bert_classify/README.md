# 21_en_bert_classify — 작은 BERT 분류 (영어 Yelp 이진, 일반 도메인 사전학습 → 다른 도메인 fine-tune)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/21_en_bert_classify.ipynb)

## 한 줄 목표
Phase 3 의 세 번째 챕터. Ch 20 에서 *작은 BERT 를 일반 도메인 (Wikitext-103) 으로 직접 MLM 사전학습* 했다면, 이번엔 그 위에 **분류 헤드를 얹어 *완전히 다른 도메인 (Yelp 영화 리뷰)* 이진 분류로 fine-tune**. Ch 10 (DistilBERT, 약 66M params, 대규모 Wikipedia + BookCorpus 사전학습) 과 같은 Yelp 이진 분류 셋업에 *우리가 만든 작은 BERT* (약 10M params, Wikitext-103 2K paragraphs × 3 epoch MLM — 한국어 Ch 23 와 동일 hyperparams) 를 붙여 두 결과를 나란히 비교 — 둘 다 *일반 도메인 → Yelp transfer* 라 비교가 *fair*, *사전학습 규모* 차이만 측정됨.

self-contained 노트북: Wikitext-103 MLM 학습을 2K × 3 epoch 압축 재현 → 같은 본체로 Yelp 분류 fine-tune → Ch 10 결과와 비교. 본문은 *일반 사전학습 → 다른 도메인 fine-tune* 메인 흐름에 집중. *사전학습 없이 같은 GPU compute 로 분류 fine-tune* 만 했을 때의 fair-compute 비교는 부록 노트북 [`appendix_compute_budget.ipynb`](./appendix_compute_budget.ipynb) 에서 분리해 다룹니다.

## 다루는 핵심 개념
- **일반 도메인 → 다른 도메인 transfer** — 원본 BERT 정신의 핵심. Wikitext-103 일반 위키로 사전학습 → Yelp 영화 리뷰 분류 fine-tune. domain-adaptive pretraining (DAPT) 함정을 피해 *정직한 transfer* 측정
- **두 데이터셋이 노트북 안에 공존** — MLM 용 Wikitext-103 + 분류용 Yelp 이진. 같은 토크나이저로 처리
- `BertForMaskedLM` -> `BertForSequenceClassification` 헤드 교체 — 본체 (`embeddings + encoder + pooler`) 는 그대로, MLM head 떼고 분류 head (`Linear(256, 2)`) 부착
- in-memory state_dict 전송: `cls_model.bert.load_state_dict(mlm_model.bert.state_dict())` — 디스크 없이 본체 가중치 복사
- 같은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024, 약 10M params) 가 MLM 모델과 분류 모델 양쪽에 적용
- 사전학습 효과의 *순 측정* — random init baseline 과 비교
- **Ch 10 (DistilBERT 대규모 일반 위키 사전학습) vs Ch 21 (작은 BERT 자체 일반 위키 사전학습)** 의 정량 비교 — 둘 다 *위키 → Yelp transfer* 라 fair

## Loss
`CrossEntropyLoss` — 분류 fine-tune 표준 (K=2, softmax + CE). 라벨은 `int 0/1`, `problem_type="single_label_classification"`. random baseline loss = `ln(2) ≈ 0.693`.

수식: $L = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i, y_i}$ — Ch 11/15 와 같은 K-class softmax CE.

## 데이터

| 단계 | 데이터셋 | 용도 |
|---|---|---|
| MLM 사전학습 | `Salesforce/wikitext`, `wikitext-103-raw-v1` 2K paragraphs (eval 400) | self-supervised MLM, 일반 위키 본문 |
| 분류 fine-tune | `fancyzhx/yelp_polarity` 5K train / 1K eval, seed 42 | supervised 이진 분류 (긍정/부정 라벨) |

같은 토크나이저 (`bert-base-uncased`) 가 두 도메인의 텍스트를 처리. `block_size=128` `group_texts` 패턴으로 MLM 3 epoch + Yelp 분류 fine-tune 2 epoch.

## 환경
Google Colab T4 GPU (fp16). 약 3-5분 — 대부분이 데이터 다운로드입니다 (실행본 `executed/21_en_bert_classify.ipynb` 기준 전체 2분 13초: 다운로드·전처리 약 1분 40초 + MLM 3 epoch 약 15초 + 분류 fine-tune 2 epoch 약 15초 + 평가·시각화 수 초).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 10 | DistilBERT 파인튜닝 (약 66M) | `bert-base-uncased` WordPiece | Yelp 이진화 | `Linear(H, 1)` | `BCEWithLogitsLoss` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Wikitext-103 paragraphs (일반 도메인) | MLM head | `CrossEntropyLoss` (masked) |
| **21** | **Ch 20 사전학습 BERT + 분류 헤드 (약 10M)** | (Ch 20과 동일) | **Yelp 이진화 (다른 도메인 transfer)** | **`Linear(H, 2)`** | **`CrossEntropyLoss`** |
| 22 (다음) | 작은 BERT (직접, scratch) — 한국어 | `klue/bert-base` 토크나이저 (가져옴) | 한국어 Wikipedia paragraphs (일반 도메인) | MLM head | `CrossEntropyLoss` (masked) |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 비교 표 — Ch 10 vs Ch 21 (둘 다 일반 위키 → Yelp transfer 라 fair)

| 차원 | Ch 10 (DistilBERT) | Ch 21 (small BERT scratch) | 비고 |
|---|---|---|---|
| 본체 파라미터 | 약 66M | 약 10M | Ch 21 은 1/6 작음 |
| 사전학습 코퍼스 | Wikipedia + BookCorpus (약 33억 토큰, 일반 도메인) | Wikitext-103 paragraphs 2K (약 27만 토큰, 일반 도메인) | 약 1.2만배 격차, **둘 다 일반 위키** |
| 사전학습 시간 | TPU 수일 | T4 약 15초 (2K × 3 epoch = 198 step) | |
| Fine-tune 도메인 | Yelp 이진 (다른 도메인) | Yelp 이진 (다른 도메인) | **둘 다 위키 -> Yelp transfer** |
| 분류 fine-tune 셋업 | (같음 — 5K/1K, batch 16, lr 2e-5, 2 epoch, fp16) | | 본체 외 통제 |
| 실측 accuracy | 약 0.90 | random (0.50) 과 Ch 10 의 중간쯤 | 실행마다 흔들려 단일 값으로 적지 않음 — 값은 실행본 `executed/21_en_bert_classify.ipynb` |

비교가 *공정* 한 이유 — 둘 다 *일반 도메인 위키 사전학습 → Yelp 분류 transfer* 의 같은 패턴이라 *사전학습 규모* (약 1.2만배) 와 *모델 크기* (약 6배) 만 변수. 격차가 *사전학습 규모의 가치* 를 정량으로 보여줍니다. *작은 일반 도메인 사전학습도 random init 보다는 분명히 낫다* 는 것, 그리고 *fair-compute (사전학습 compute 를 fine-tune 으로 옮겨도)* 격차가 메워지지 않는다는 것은 부록 [`appendix_compute_budget.ipynb`](./appendix_compute_budget.ipynb) 참조.

## 다음 챕터
[22_ko_bert_pretrain](../22_ko_bert_pretrain/) — Ch 20 의 영어 사전학습 패턴을 한국어로 재현. 같은 작은 BertConfig + `klue/bert-base` 토크나이저 + **한국어 Wikipedia paragraphs MLM** (일반 도메인). Ch 22 → Ch 23 (한국어 NSMC 분류) 가 이번 챕터 (Ch 20 → Ch 21, 영어) 와 *대칭* — 둘 다 *일반 위키 사전학습 → 영화 리뷰 분류 transfer*.
