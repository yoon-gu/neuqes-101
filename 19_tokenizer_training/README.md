# 19_tokenizer_training — 토크나이저 직접 학습 (WordPiece vs WordLevel, 영어 + 한국어)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/19_tokenizer_training/19_tokenizer_training.ipynb)

## 한 줄 목표
Phase 3 의 첫 챕터. 지금까지 *사전학습된* 토크나이저 (`distilbert-base-uncased`, `klue/bert-base`) 를 받아 쓰기만 했는데, 이번엔 **토크나이저 자체를 직접 학습** 해 비교합니다. 두 알고리즘 (WordPiece, WordLevel) × 두 언어 (영어 Yelp, 한국어 NSMC) = 4 종을 학습해 vocab·토큰 길이·UNK 비율을 나란히 본 뒤, `PreTrainedTokenizerFast` 로 wrap 해 Ch 20+ 의 HF 표준 인터페이스로 변환.

## 다루는 핵심 개념
- **WordPiece** (BERT 표준 subword) vs **WordLevel** (어절 단위) 알고리즘 차이
- `tokenizers.Tokenizer` + `tokenizers.models.{WordPiece, WordLevel}` + `{WordPieceTrainer, WordLevelTrainer}` 패턴
- `pre_tokenizer = BertPreTokenizer()` (공백 + 구두점 분리, BERT 표준) vs `Whitespace()` (공백만)
- `normalizer = NFD + StripAccents + Lowercase` (영어 lowercase, 한국어는 lowercase 무의미)
- `TemplateProcessing` 으로 `[CLS] / [SEP]` 자동 부착
- `tokenizer.save()` / `Tokenizer.from_file()` + `PreTrainedTokenizerFast` 로 HF 표준 인터페이스 변환
- 한국어 WordLevel 의 vocab 비효율 — 교착어 특성상 같은 어근의 다른 활용이 모두 별개 토큰
- vocab 크기 sweep (1K / 4K / 8K / 16K) — 토큰 길이·UNK 비율·임베딩 파라미터의 trade-off

## Loss
이번 챕터는 *분류 task 가 없음* — Loss 도 없습니다. 산출물은 vocab + merge rules.

## 데이터
- 영어: `yelp_polarity` train 5,000 문장 (라벨 무시, text 만)
- 한국어: e9t/nsmc GitHub raw `ratings_train.txt` 에서 5,000 문장 sample (라벨 무시)

## 환경
Google Colab T4 (모델 학습 없음, GPU 거의 안 씀). 약 2분.

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 17 | klue/bert-base | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | `BCEWithLogitsLoss` |
| 18 | klue/bert-base + 보조 | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | `BCEWithLogitsLoss + λ·L_aux` |
| **19** | — (토크나이저 학습 전용) | **WordPiece + WordLevel** (둘 다 *직접 학습*) | **Yelp text + NSMC text** | — | — |
| 20 (다음) | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Yelp text | MLM head | `CrossEntropyLoss` (masked) |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 다음 챕터
[20_en_bert_pretrain](../20_en_bert_pretrain/) — 작은 BERT (`n_layer=4, hidden=256`) 를 *직접 사전학습* (MLM). 토크나이저는 학습 안정성을 위해 표준 `bert-base-uncased` 를 가져옴 — Ch 19 의 *경험* 위에 표준 도구의 신뢰성을 얹는 흐름. Ch 21 에서 이 사전학습 모델을 Yelp 이진 분류에 fine-tune → Ch 10 (DistilBERT 사전학습) 과 직접 비교.
