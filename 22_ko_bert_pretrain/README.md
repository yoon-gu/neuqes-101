# 22_ko_bert_pretrain — 작은 BERT 직접 사전학습 (한국어 MLM scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/22_ko_bert_pretrain/22_ko_bert_pretrain.ipynb)

## 한 줄 목표
Phase 3 의 네 번째 챕터. Ch 20 에서 *영어 작은 BERT* 를 random init 해 일반 도메인 (Wikitext) 으로 MLM 사전학습 했다면, 이번엔 *완전히 같은 본체 구조* 로 **한국어 Wikipedia paragraphs** MLM 사전학습. 변하는 축은 **언어** — 토크나이저 `klue/bert-base` (한국어 WordPiece, vocab 약 32,000), 데이터 `wikimedia/wikipedia` (`20231101.ko`). 본체 hyperparam, loss, training args 는 Ch 20 과 동일. 산출물은 Ch 23 에서 *완전히 다른 도메인* (NSMC 영화 리뷰) 이진 분류 fine-tune.

## 다루는 핵심 개념
- **언어 한 축 변화** — 토크나이저와 데이터만 한국어로, 본체 구조·loss·hyperparams 는 Ch 20 동일
- **일반 도메인 사전학습** — 원본 BERT 의 Wikipedia + BookCorpus 정신을 따라 한국어 Wikipedia 본문 사용. task 도메인 (NSMC) 으로 학습하지 않아 *진정한 transfer* 측정 가능
- `klue/bert-base` 한국어 WordPiece 토크나이저 로드 + `bert-base-uncased` (영어) 와의 *cross-language* 비교 (Ch 19 §5-4 결론의 실측 확인)
- 작은 `BertConfig(hidden=256, layer=4, head=4, intermediate=1024)` + `BertForMaskedLM(config)` random init
- `wikimedia/wikipedia` (`20231101.ko`) HF 정제본 로드 — article 단위 → paragraph 단위로 split 후 5K 사용
- `DataCollatorForLanguageModeling(mlm_probability=0.15)` — 한국어 [MASK] 80/10/10 동작 풀버전 시각화 (자리별 운명 표 + 큰 batch 통계, Ch 20 영어 짝과 동일 깊이)
- `labels = -100` ignore_index — 한국어 MLM 도 동일, Phase 4 SFT (Ch 28) 에서 *같은 트릭, 정반대 자리* 로 재등장
- random baseline `ln(32000) ≈ 10.37` (Ch 20 의 10.33 과 미세 차이)
- 학습 전·후 비교: 일반 위키 도메인 문장 + NSMC 도메인 문장 [MASK] top-5 — 사전학습이 본 분포는 향상이 명확, 다른 도메인은 fine-tune 단계에서 적응
- `model.save_pretrained()` / `tokenizer.save_pretrained()` 로 Ch 23 fine-tune 인계

## Loss
`CrossEntropyLoss` — 가려진 위치들의 *원래 토큰* 을 vocab 약 32,000 차원 softmax 로 예측. Ch 20 과 동일한 MLM CE, vocab 크기만 미세하게 다름.

수식: $L_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid x_{\setminus M})$

## 데이터
한국어 Wikipedia — `wikimedia/wikipedia` config `20231101.ko` (CC-BY-SA, HF Hub 정제본). article 단위로 다운로드 후 paragraph (50-2000자 필터) 단위로 split. train 5,000 / eval 500 paragraphs, seed 42. `block_size=128` `group_texts` 후 약 500-1,500 블록.

## 환경
Google Colab T4 GPU (fp16). 약 2-4분 (토크나이저 로드 + ko 위키 다운로드·paragraph split·토큰화가 대부분을 차지 + MLM 2 epoch 약 0.3분 + 평가/저장 — 전체 실측 약 2분, 네트워크·VM 상태에 따라 늘어날 수 있음). 전체 소요는 데이터 다운로드가 지배합니다.

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Wikitext-103 (일반 도메인) | MLM head | `CrossEntropyLoss` (masked) |
| 21 | Ch 20 사전학습 BERT + 분류 헤드 | (Ch 20과 동일) | Yelp 이진화 | `Linear(H, 2)` | `CrossEntropyLoss` |
| **22** | **작은 BERT (직접, scratch) — 한국어** | **`klue/bert-base` 토크나이저 (가져옴)** | **한국어 Wikipedia (일반 도메인)** | **MLM head** | **`CrossEntropyLoss` (masked)** |
| 23 (다음) | Ch 22 사전학습 BERT + 분류 헤드 | (Ch 22와 동일) | NSMC 이진 (다른 도메인 transfer) | `Linear(H, 2)` | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 산출물
`./ch22_small_bert_mlm_ko/` 폴더에 `config.json + model.safetensors + tokenizer.json + vocab.txt + ...` 저장. Ch 23 에서 `AutoModelForSequenceClassification.from_pretrained("./ch22_small_bert_mlm_ko", num_labels=2)` 한 줄로 *encoder body* 를 가져와 새 분류 헤드를 부착해 fine-tune.

## 다음 챕터
[23_ko_bert_classify](../23_ko_bert_classify/) — 이번 챕터 사전학습 모델을 *완전히 다른 도메인 (NSMC 영화 리뷰)* 이진 분류로 fine-tune. **Ch 15 (`klue/bert-base` 대규모 한국어 사전학습 모델 fine-tune) 과 직접 비교** — 작은 사전학습 BERT (약 10M, 위키 5K paragraphs MLM) vs 표준 한국어 BERT (약 110M, 약 8.4B tokens 대규모 일반 코퍼스) 2-way. 영어 Ch 20 → Ch 21 흐름의 한국어 대칭본 — *일반 사전학습 → 다른 도메인 fine-tune transfer* 메시지가 본 챕터의 클라이맥스. 추가로 [`appendix_random_baseline.ipynb`](../23_ko_bert_classify/appendix_random_baseline.ipynb) 에서 *random init baseline + negative transfer 분석*.
