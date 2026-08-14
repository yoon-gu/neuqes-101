# Hugging Face로 시작하는 언어모델: Colab 무료버전으로 만드는 BERT·GPT·Diffusion LM from scratch

> 영어도 한국어도 — 비싼 GPU도 결제도 없이 세 언어모델을 직접

scikit-learn 한 줄로 텍스트를 숫자로 바꾸는 일에서 출발해, 직접 만든 GPT가 짧은 이야기를 써내려가는 순간까지 한 걸음씩 따라가는 책입니다. 모든 장이 **Google Colab 노트북** 한 개로 되어 있어, 아무것도 설치하지 않고 브라우저에서 열어 무료 T4 GPU로 30분 안에 끝까지 실행해 볼 수 있습니다.

이 책의 약속은 단순합니다. **한 장에서 바뀌는 것은 딱 한 가지뿐입니다.** 모델을 바꾸는 장에서는 데이터와 손실 함수를 그대로 두고, 손실 함수를 바꾸는 장에서는 모델을 건드리지 않습니다. 그래서 결과가 달라질 때마다 무엇이 그 차이를 만들었는지 헷갈리지 않고 짚어낼 수 있습니다.

라이브러리를 호출하는 데서 멈추지 않습니다. sigmoid와 softmax가 같은 분류를 어떻게 다른 방식으로 푸는지, CrossEntropy가 안에서 무슨 계산을 하는지, 토크나이저가 문장을 어떻게 조각내는지를 매 장마다 직접 열어 확인합니다.

## 📘 책 PDF로 받기

웹에서 장별로 읽는 것과 같은 내용을, 인쇄·오프라인용으로 한 권에 묶은 PDF 원고입니다. 표지·목차·각주까지 조판해 그대로 출력하거나 태블릿에서 읽기 좋습니다.

- **최신 PDF 내려받기 (Ch 1-34)** — [neuqes-101-ch01-34-manuscript.pdf](https://github.com/yoon-gu/neuqes-101/releases/latest/download/neuqes-101-ch01-34-manuscript.pdf)
- 버전별 원고는 [Releases](https://github.com/yoon-gu/neuqes-101/releases)에서 받을 수 있습니다.

> 집필이 진행되며 장이 더해질 때마다 새 버전을 올립니다. 최신 PDF는 항상 위 **Releases** 최신 항목에서 확인하실 수 있습니다.

## 누구에게 맞는 책인가요

- 파이썬은 써봤지만 BERT나 GPT를 코드로 다뤄 본 적은 없는 분
- 모델이 왜 그렇게 동작하는지 설명이 아니라 실행 결과로 납득하고 싶은 분
- 값비싼 장비 없이 무료 Colab 한 대로 사전학습부터 정렬(alignment)까지 경험하고 싶은 분

## 이렇게 배웁니다

텍스트 분석을 네 축으로 나눠, 한 번에 한 축씩만 움직입니다.

```
모델:       sklearn ─→ DistilBERT(영어) ─→ KLUE-BERT(한국어) ─→ 작은 BERT(워드레벨)
태스크:     Regression ─→ Binary ─→ Multi-class ─→ Multi-label
손실 함수:  MSELoss ─→ BCEWithLogitsLoss ─→ CrossEntropyLoss ─→ BCEWithLogitsLoss(per-label) ─→ +Auxiliary (Combined)
토크나이저: TF-IDF ─→ WordPiece(영어) ─→ WordPiece(한국어) ─→ 워드레벨(직접) ─→ 형태소기반(직접)
```

> Auxiliary는 새 task가 아니라 기존 loss에 보조 항(예: `λ·MSE`)을 더하는 변화이므로 **손실 함수 축** 끝에 둡니다. Ch 14·18의 메인 task는 직전 장(Multi-label)과 동일합니다.

각 장은 **먼저 돌려보기 → 안에서 무슨 일이 일어나는지 해부하기 → 직접 바꿔보기**의 세 단계로 흐릅니다.

## 챕터별 변화추적표

표의 Colab 버튼을 누르면 그 장의 노트북이 곧바로 열립니다. 한 장에서 어떤 축이 움직이는지(모델·토크나이저·데이터·Output Head·Loss)를 한눈에 비교하도록 정리했습니다.

| Ch | Colab | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss | 라벨 형식 |
|---|---|---|---|---|---|---|---|---|
| 1 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/01_tfidf/01_tfidf.ipynb) | (TF-IDF) | TF-IDF | Yelp 5,000 샘플 | — | — | — | — |
| 2 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/02_sklearn_regression/02_sklearn_regression.ipynb) | LinearReg | TF-IDF | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` | float |
| 3 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/03_sklearn_binary/03_sklearn_binary.ipynb) | LogReg | TF-IDF | Yelp 이진화 (4-5→1, 1-2→0, 3 제외) | (1차원) | sigmoid | `BCEWithLogitsLoss` | int (0/1) |
| 4 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/04_softmax_binary/04_softmax_binary.ipynb) | LogReg (multinomial 자동) | TF-IDF | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` | int (0/1) |
| 5 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/05_sklearn_multiclass/05_sklearn_multiclass.ipynb) | LogReg (multinomial 자동) | TF-IDF | Yelp 5클래스 (별점 0-4) | (5차원) | softmax | `CrossEntropyLoss` | int (0-4) |
| 6 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/06_sklearn_multilabel/06_sklearn_multilabel.ipynb) | OneVsRest LogReg | TF-IDF | Yelp + 항목 키워드 합성 (food/service/price/ambiance/location) | (5차원) | sigmoid (각각) | `BCEWithLogitsLoss` per-label | multi-hot |
| 7 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/07_bert_pipeline/07_bert_pipeline.ipynb) | DistilBERT (추론) | WordPiece | 간단 영어 예시 문장 | 사전학습 헤드 | softmax | — | — |
| 8 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/08_tokenizer_datasets/08_tokenizer_datasets.ipynb) | DistilBERT (추론) | WordPiece | Yelp (datasets 라이브러리 해부) | — | — | — | — |
| 9 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/09_bert_regression.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp (별점 1-5) | `Linear(H,1)` | 없음 | `MSELoss` | float |
| 10 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/10_bert_binary_sigmoid/10_bert_binary_sigmoid.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp 이진화 | `Linear(H,1)` | sigmoid | `BCEWithLogitsLoss` | float (0.0/1.0) |
| 11 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/11_bert_binary_softmax/11_bert_binary_softmax.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp 이진화 | `Linear(H,2)` | softmax | `CrossEntropyLoss` | int (0/1) |
| 12 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/12_bert_multiclass/12_bert_multiclass.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp 5클래스 | `Linear(H,5)` | softmax | `CrossEntropyLoss` | int (0-4) |
| 13 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/13_bert_multilabel/13_bert_multilabel.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp + 항목 (Ch 6과 동일 합성) | `Linear(H,5)` | sigmoid (각각) | `BCEWithLogitsLoss` | multi-hot |
| 14 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/14_auxiliary_loss/14_auxiliary_loss.ipynb) | DistilBERT + 보조 헤드 | WordPiece | Yelp + 항목 + 별점 (한 샘플에 두 라벨) | 메인(5) + 보조(1) | sigmoid + 없음 | `BCEWithLogitsLoss + λ·MSELoss` | multi-hot + float |
| 15 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/15_ko_binary/15_ko_binary.ipynb) | klue/bert-base | WordPiece (한국어) | NSMC (네이버 영화 리뷰) | `Linear(H,2)` | softmax | `CrossEntropyLoss` | int (0/1) |
| 16 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/16_ko_multiclass/16_ko_multiclass.ipynb) | klue/bert-base | WordPiece (한국어) | KLUE-YNAT (뉴스 7분류) | `Linear(H,7)` | softmax | `CrossEntropyLoss` | int (0-6) |
| 17 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/17_ko_multilabel/17_ko_multilabel.ipynb) | klue/bert-base | WordPiece (한국어) | KLUE-YNAT 합성 multi-label (두 문서 결합) | `Linear(H,7)` | sigmoid (각각) | `BCEWithLogitsLoss` | multi-hot |
| 18 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/18_ko_auxiliary/18_ko_auxiliary.ipynb) | klue/bert-base + 보조 | WordPiece (한국어) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | sigmoid + 태스크별 | `BCEWithLogitsLoss + λ·L_aux` | 메인 + 보조 |
| 19 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/19_tokenizer_training/19_tokenizer_training.ipynb) | — (토크나이저 학습 전용) | **WordPiece + WordLevel** (둘 다 직접 학습 비교) | Yelp text + NSMC text subset | — | — | — | — |
| 20 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/20_en_bert_pretrain/20_en_bert_pretrain.ipynb) | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | **Wikitext-103 paragraphs (일반 도메인 — `Salesforce/wikitext`)** | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) | masked token ids |
| 21 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/21_en_bert_classify.ipynb) | Ch 20 사전학습 BERT + 분류 헤드 | (Ch 20과 동일) | **Yelp 이진 (다른 도메인 transfer)** — Ch 10과 직접 비교 (*둘 다 일반 위키 → Yelp transfer 라 fair*) | `Linear(H,2)` | softmax | `CrossEntropyLoss` | int (0/1) |
| 22 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/22_ko_bert_pretrain/22_ko_bert_pretrain.ipynb) | 작은 BERT (직접, scratch) | `klue/bert-base` 토크나이저 (가져옴) | **한국어 Wikipedia paragraphs (일반 도메인 — `wikimedia/wikipedia` 20231101.ko)** | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) | masked token ids |
| 23 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/23_ko_bert_classify/23_ko_bert_classify.ipynb) | Ch 22 사전학습 BERT + 분류 헤드 | (Ch 22와 동일) | **NSMC 이진 (다른 도메인 transfer)** — Ch 15와 직접 비교 (*scratch 작은 BERT* vs *klue/bert-base 대규모 사전학습*) | `Linear(H,2)` | softmax | `CrossEntropyLoss` | int (0/1) |
| 24 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/24_gpt_tinystories/24_gpt_tinystories.ipynb) | small GPT2 (직접) | BPE (GPT2) | TinyStories | `Linear(H,V)` (LM head) | softmax (sampling) | `CrossEntropyLoss` (next-token) | shifted token ids |
| 25 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/25_gpt2_continual_pretrain/25_gpt2_continual_pretrain.ipynb) | gpt2 (사전학습) — **continual pretraining** (계속 사전학습 / continual learning, *같은 CausalLM task 를 새 데이터로 더 학습*. SFT 아님) | BPE (GPT2) | TinyStories (Ch 24와 동일) | `Linear(H,V)` (LM head 그대로) | softmax | `CrossEntropyLoss` (next-token) | shifted token ids |
| 26 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/26_ko_tiny_gpt/26_ko_tiny_gpt.ipynb) | small GPT2 (직접, scratch — Ch 24의 한국어판) | BBPE (직접 학습) | `g0ster/TinyStories-Korean` subset (영어 TinyStories 의 한국어 번역본, MIT) | `Linear(H,V)` | softmax + sampling(top-k/top-p) | `CrossEntropyLoss` (next-token) | shifted token ids |
| 27 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/27_ko_gpt2_continual_pretrain/27_ko_gpt2_continual_pretrain.ipynb) | KoGPT2 (사전학습) — **continual pretraining** (Ch 25의 한국어판, *같은 CausalLM task 를 한국어 데이터로 더 학습*. SFT 아님) | BBPE (KoGPT2 그대로) | `g0ster/TinyStories-Korean` (Ch 26과 동일) | `Linear(H,V)` (LM head 그대로) | softmax | `CrossEntropyLoss` (next-token) | shifted token ids |
| 28 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/28_sft/28_sft.ipynb) | KoGPT2 SFT | BBPE | KoAlpaca (instruction → response) | `Linear(H,V)` | softmax | `CrossEntropyLoss` (`SFTTrainer`, response-only mask 옵션) | chat template token ids |
| 29 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/29_benchmark_eval/29_benchmark_eval.ipynb) | Qwen2.5-0.5B-Instruct (평가 대상 — Ch 28 KoGPT2 SFT 는 §7 대조 서술) | BBPE (Qwen2.5 그대로) | **벤치마크 평가 원리**: KoBEST (HellaSwag·BoolQ subset) MC 직접 구현 + 산술 생성 평가 + `lm-eval` 시연 (분야 지도: MMLU·KMMLU·GSM8K·LogicKor 등) | — (평가만) | — | — (`lm-evaluation-harness`) | task-format별 |
| 30 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/30_dpo/30_dpo.ipynb) | SFT base + frozen ref | BBPE | preference 쌍 (chosen/rejected) | `Linear(H,V)` | log-likelihood ratio | `DPO sigmoid loss` (`DPOTrainer`) | (chosen, rejected) pair |
| 31 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/31_grpo/31_grpo.ipynb) | SFT base | BBPE | verifiable-reward prompts (수학·코드) | `Linear(H,V)` + group advantage | softmax | `GRPO loss` (group relative, `GRPOTrainer`) | rollout group + verifier |
| 32 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/32_diffusion_intro/32_diffusion_intro.ipynb) | 작은 BERT-MLM diffusion (직접, scratch — 패러다임) | BPE 2048 (직접 학습, TinyStories) | TinyStories train[:100000] | MLM head | parallel denoise (기본 confidence remasking) | 흡수형 NELBO (시간가중 `1/t`) | masked token ids |
| 33 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/33_diffusion_train/33_diffusion_train.ipynb) | 작은 BERT-MLM diffusion (Ch 32와 동일 모델) | BPE 2048 (직접 학습, TinyStories) | TinyStories train[:100000] | MLM head | parallel denoise (**carry-over semi-AR + 반복억제**) | 흡수형 NELBO (시간가중 `1/t`) | masked token ids |
| 34 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/34_ko_diffusion/34_ko_diffusion.ipynb) | 작은 BERT-MLM diffusion (한국어, scratch — 80/10/10으로 붕괴 교정) | BPE 4000 (직접 학습, TinyStories-Korean) | TinyStories-Korean | MLM head | parallel denoise (carry-over) | 마스크 자리 평균 CE (80/10/10 마스킹) | masked token ids |

> Ch 4는 Ch 3 이진 분류 데이터를 그대로 가져와 **softmax+CE(2차원)** 로 풀어 sigmoid+BCE와의 동등성을 시연합니다. **Ch 10·11도 BERT에서 같은 두 방식을 따로 학습** 해 비교합니다 — Ch 10이 sigmoid+BCE 방식, Ch 11이 softmax+CE 방식.

## 학습 여정 (Phase 0-5)

큰 흐름은 **sklearn으로 본질 잡기 → DistilBERT로 다시 → 한국어로 재방문 → 바닥부터 직접 → GPT와 정렬 → Diffusion** 입니다. 한 단계가 끝나면 같은 골격을 다음 단계가 한 겹씩 더 깊게 되짚습니다.

### Phase 0 · sklearn으로 태스크와 손실의 본질 (Ch 1-6)

BERT는 아직 등장하지 않습니다. 텍스트를 TF-IDF로 숫자화한 뒤 회귀 → 이진 → 다중 클래스 → 다중 라벨로 태스크를 넓히며, MSE·BCE·CrossEntropy가 각각 무엇을 재는지 sklearn으로 또렷하게 익힙니다. Ch 4에서는 같은 이진 데이터를 sigmoid+BCE와 softmax+CE 두 방식으로 풀어, 둘이 사실은 같은 모델임을 보입니다.

### Phase 1 · DistilBERT로 같은 태스크를 다시 (Ch 7-14)

Phase 0에서 익힌 태스크들을 이번엔 DistilBERT와 `Trainer`로 재정식화합니다. 출력 헤드와 손실 함수만 갈아끼우며 회귀·이진·다중 클래스·다중 라벨을 훑고, Ch 10·11에서 이진 분류의 두 방식(sigmoid / softmax)을 따로 학습해 BERT에서도 같은 결과가 나옴을 확인합니다. 보조 손실(Ch 14)로 손실 축을 마무리합니다.

### Phase 2 · 한국어로 압축 재방문 (Ch 15-18)

`klue/bert-base`로 같은 흐름을 한국어에서 빠르게 되짚습니다. 회귀는 영어에서 이미 다뤘으므로 건너뛰고 이진 분류부터 시작합니다.

### Phase 3 · 토크나이저와 사전학습을 바닥부터 (Ch 19-23)

사전학습된 모델에 기대지 않고 직접 만듭니다. 먼저 토크나이저를 손수 학습해 비교하고(Ch 19), 작은 BERT를 일반 도메인 위키로 MLM 사전학습한 뒤(영어 Ch 20·한국어 Ch 22) 다른 도메인 분류로 파인튜닝합니다(Ch 21·23). 사전학습은 일반 도메인, 파인튜닝은 다른 도메인이라 진짜 transfer가 측정됩니다 — 원본 BERT가 위키·책으로 사전학습하던 정신 그대로입니다. 결과는 기성 모델(영어 Ch 10·한국어 Ch 15)과 직접 견줍니다.

### Phase 4 · GPT로 넘어가 정렬까지 (Ch 24-31)

인코더(BERT)에서 디코더(GPT)로 무대를 옮깁니다. GPT 시대의 학습 네 단계 — **사전학습**(Ch 24·26) → **계속 사전학습**(Ch 25·27) → **SFT 지시학습**(Ch 28) → **정렬**(Ch 30 DPO, Ch 31 GRPO) — 을 영어·한국어 대칭으로 밟고, 그 사이에 분야별 벤치마크 평가(Ch 29)를 끼웁니다. PPO는 T4 한 대에 네 모델을 동시에 올릴 수 없어 제외하고, 그 부담이 없는 DPO로 대신합니다.

### Phase 5 · Diffusion LM (Ch 32-34)

다음 토큰을 하나씩 잇는 autoregressive와 달리, 문장 전체를 한꺼번에 denoise하는 새 패러다임을 봅니다. 한 챕터에 한 축씩만 바꿉니다. 작은 mask-diffusion을 직접 구현해 영어 동화를 생성하고(Ch 32 — **패러다임**, 기본 샘플러라 반복이 남음), 같은 모델에 carry-over 샘플러와 반복 억제를 더해 그 반복을 걷어냅니다(Ch 33 — **샘플러**). Ch 34에서는 같은 레시피를 한국어로 옮기면 무너지는 지점을 진단하고, BERT의 80/10/10 마스킹으로 한국어 diffusion을 되살립니다(Ch 34 — **언어**).

## 학습 환경

- Google Colab T4 GPU (16GB VRAM)
- 챕터당 30분 이내
- bf16 미지원(T4 Compute Capability 7.5) → `fp16=True` 만 사용
- Flash Attention 2 미지원
