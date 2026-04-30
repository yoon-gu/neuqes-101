# neuqes-101

Hugging Face 입문 커리큘럼 (19챕터). 모든 실습자료는 **Google Colab 노트북** 으로 제공되며, T4 GPU(16GB) 환경에서 챕터당 30분 이내에 끝까지 돌도록 설계되어 있습니다.

학습 흐름의 두 축:

```
모델 축:    sklearn ─→ DistilBERT(영어) ─→ KLUE-BERT(한국어) ─→ 작은 BERT(워드레벨)
태스크 축:  Regression ─→ Binary ─→ Multi-class ─→ Multi-label
Loss 축:    MSELoss ─→ BCEWithLogitsLoss ─→ CrossEntropyLoss ─→ BCEWithLogitsLoss(per-label) ─→ +Auxiliary (Combined)
토크나이저: TF-IDF ─→ WordPiece(영어) ─→ WordPiece(한국어) ─→ 워드레벨(직접) ─→ 형태소기반(직접)
```

> Auxiliary는 새 task가 아니라 기존 loss에 보조 항(예: `λ·MSE`)을 더하는 변화이므로 **Loss 축** 끝에 둡니다. Ch 13·17의 메인 task는 직전 챕터(Multi-label)와 동일합니다.

각 챕터는 레포 루트의 자체 폴더(예: `01_tfidf/`)에 노트북과 요약 `README.md`가 함께 들어 있습니다.

## 챕터별 변화추적표

각 행의 Colab 버튼을 누르면 해당 챕터 노트북이 Colab에서 바로 열립니다. **진행** 열은 사용자가 Colab에서 직접 실행해 끝까지 정상 동작함을 확인하면 `✅`으로 갱신합니다(미검증: `—`).

| Ch | 진행 | Colab | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss | 라벨 형식 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ✅ | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/01_tfidf/01_tfidf.ipynb) | (TF-IDF) | `TfidfVectorizer` | Yelp 5,000 샘플 | — | — | — | — |
| 2 | ✅ | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/02_sklearn_regression/02_sklearn_regression.ipynb) | `LinearRegression()` | TF-IDF | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` | float |
| 3 | ✅ | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/03_sklearn_binary/03_sklearn_binary.ipynb) | `LogisticRegression()` | TF-IDF | Yelp 이진화 (4-5→1, 1-2→0, 3 제외) | (1차원) | sigmoid | `BCEWithLogitsLoss` | int (0/1) |
| 4 | ✅ | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/04_softmax_binary/04_softmax_binary.ipynb) | `LogisticRegression(multi_class="multinomial")` | TF-IDF | Yelp 이진화 (Ch 3과 동일) | (2차원) | softmax | `CrossEntropyLoss` | int (0/1) |
| 5 | ✅ | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/05_sklearn_multiclass/05_sklearn_multiclass.ipynb) | `LogisticRegression(multi_class="multinomial")` | TF-IDF | Yelp 5클래스 (별점 0-4) | (5차원) | softmax | `CrossEntropyLoss` | int (0-4) |
| 6 | ✅ | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/06_sklearn_multilabel/06_sklearn_multilabel.ipynb) | `OneVsRestClassifier(LogisticRegression())` | TF-IDF | Yelp + 측면 키워드 합성 (food/service/price/ambiance/location) | (5차원) | sigmoid (각각) | `BCEWithLogitsLoss` per-label | multi-hot |
| 7 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/07_bert_pipeline/07_bert_pipeline.ipynb) | DistilBERT (추론) | WordPiece | 간단 영어 예시 문장 | 사전학습 헤드 | softmax | — | — |
| 8 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/08_tokenizer_datasets/08_tokenizer_datasets.ipynb) | DistilBERT (추론) | WordPiece | Yelp (datasets 라이브러리 해부) | — | — | — | — |
| 9 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/09_bert_regression/09_bert_regression.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp (별점 1-5) | `Linear(H,1)` | 없음 | `MSELoss` | float |
| 10a | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/10_bert_binary/10_bert_binary.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp 이진화 | `Linear(H,1)` | sigmoid | `BCEWithLogitsLoss` | float (0.0/1.0) |
| 10b | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/10_bert_binary/10_bert_binary.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp 이진화 | `Linear(H,2)` | softmax | `CrossEntropyLoss` | int (0/1) |
| 11 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/11_bert_multiclass/11_bert_multiclass.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp 5클래스 | `Linear(H,5)` | softmax | `CrossEntropyLoss` | int (0-4) |
| 12 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/12_bert_multilabel/12_bert_multilabel.ipynb) | DistilBERT 파인튜닝 | WordPiece | Yelp + 측면 (Ch 6과 동일 합성) | `Linear(H,5)` | sigmoid (각각) | `BCEWithLogitsLoss` | multi-hot |
| 13 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/13_auxiliary_loss/13_auxiliary_loss.ipynb) | DistilBERT + 보조 헤드 | WordPiece | Yelp + 측면 + 별점 (한 샘플에 두 라벨) | 메인(5) + 보조(1) | sigmoid + 없음 | `BCEWithLogitsLoss + λ·MSELoss` | multi-hot + float |
| 14 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/14_ko_binary/14_ko_binary.ipynb) | klue/bert-base | WordPiece (한국어) | NSMC (네이버 영화 리뷰) | `Linear(H,2)` | softmax | `CrossEntropyLoss` | int (0/1) |
| 15 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/15_ko_multiclass/15_ko_multiclass.ipynb) | klue/bert-base | WordPiece (한국어) | KLUE-YNAT (뉴스 7분류) | `Linear(H,7)` | softmax | `CrossEntropyLoss` | int (0-6) |
| 16 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/16_ko_multilabel/16_ko_multilabel.ipynb) | klue/bert-base | WordPiece (한국어) | KLUE-YNAT 합성 multi-label (두 문서 결합) | `Linear(H,7)` | sigmoid (각각) | `BCEWithLogitsLoss` | multi-hot |
| 17 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/17_ko_auxiliary/17_ko_auxiliary.ipynb) | klue/bert-base + 보조 | WordPiece (한국어) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | sigmoid + 태스크별 | `BCEWithLogitsLoss + λ·L_aux` | 메인 + 보조 |
| 18 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/18_en_word_tokenizer/18_en_word_tokenizer.ipynb) | 작은 BERT (직접) | 워드레벨 (직접 학습) | Yelp 이진화 (Ch 10과 비교) | `Linear(H,2)` | softmax | `CrossEntropyLoss` | int (0/1) |
| 19 | — | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/19_ko_word_tokenizer/19_ko_word_tokenizer.ipynb) | 작은 BERT (직접) | 워드레벨 (형태소 vs 공백) | NSMC | `Linear(H,2)` | softmax | `CrossEntropyLoss` | int (0/1) |

> Ch 4는 Ch 3 이진 분류 데이터를 그대로 가져와 **softmax+CE(2차원)** 로 풀어 sigmoid+BCE와의 동등성을 시연합니다. Ch 10도 BERT에서 같은 두 방식을 비교 (10a/10b는 한 노트북).

## Phase 구분

- **Phase 0 (Ch 1-6)** — sklearn으로 태스크/loss의 본질 학습. BERT 등장하지 않음.
- **Phase 1 (Ch 7-13)** — DistilBERT(영어)로 같은 태스크들을 다시. Auxiliary loss로 마무리.
- **Phase 2 (Ch 14-17)** — 한국어로 압축 재방문 (klue/bert-base). Binary부터 시작 (회귀는 영어에서 다뤘으므로 생략).
- **Phase 3 (Ch 18-19)** — 토크나이저를 직접 학습. 사전학습 의존 없는 경험. Phase 3가 클라이맥스가 되도록 토크나이저 시각을 Ch 1부터 일관되게 추적합니다.

## 학습 환경

- Google Colab T4 GPU (16GB VRAM)
- 챕터당 30분 이내
- bf16 미지원(T4 Compute Capability 7.5) → `fp16=True` 만 사용
- Flash Attention 2 미지원
