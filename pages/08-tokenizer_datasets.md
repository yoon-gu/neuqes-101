**목표**: Ch 7에서 만난 WordPiece 토크나이저와 사전학습 모델로 *Phase 0의 Yelp 데이터를 다시* 만납니다. `padding` / `truncation` / `max_length` 옵션의 의미를 손에 익히고, `datasets` 라이브러리로 65만 건 코퍼스를 메모리 걱정 없이 다룹니다. Ch 9 학습의 *입력 파이프라인* 이 이 챕터에서 완성됩니다.

**환경**: Google Colab — CPU도 OK (이번 챕터도 학습 없음). T4 권장.

**예상 소요 시간**: 약 10분 (모델 가중치 다운로드는 안 함, 토크나이저 + 데이터 로딩만)

## 학습 흐름

1. 🚀 **실습**: `datasets.load_dataset` 으로 Yelp 65만 건 로드 → 5,000건 subsample
2. 🔬 **해부**: 토크나이저 옵션 3종 (`padding`, `truncation`, `max_length`) 직접 실험 + `attention_mask` 가 학습에 어떻게 쓰이는지
3. 🛠️ **변형**: `datasets.map` 으로 5,000건 일괄 토큰화 → `DataLoader` 까지 변환 (Ch 9 학습 입력의 모습)

## 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2-6 | sklearn 모델들 | `TfidfVectorizer()` | Yelp 변형 | 1차원/K차원 | 없음/sigmoid/softmax | MSE/BCE/CE |
| 7 | `pipeline("sentiment-analysis")` | `AutoTokenizer.from_pretrained(...)` | 간단 영어 예시 | 사전학습 헤드 | softmax | — |
| **8 ← 여기** | (모델 없음 — 토크나이저·데이터 파이프라인만) | `AutoTokenizer.from_pretrained(...)` | **Yelp 5,000 (Phase 0과 동일)** | — | — | — |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 7)

| 축 | Ch 7 | Ch 8 |
|---|---|---|
| 모델 | `pipeline` + `AutoModelForSequenceClassification` | **모델 로드 없음** (다음 챕터 학습 준비 단계) |
| 토크나이저 | WordPiece — 한 문장 시연 | **WordPiece + 옵션 학습** (`padding` / `truncation` / `max_length`) |
| 데이터 | 간단 영어 예시 문장 | **`datasets` 로 Yelp 65만 → 5,000 subsample** (Phase 0과 동일 데이터) |
| 데이터 라이브러리 | (없음) | **`datasets`** 첫 등장 — `load_dataset`, `map`, `filter`, `with_format` |
| 학습 단계 | 추론만 | 학습·추론 모두 없음 — 데이터 파이프라인 *연습* |

**왜 이 챕터?** Ch 9 BERT 회귀에서 `Trainer.train()` 한 줄을 부르려면, 그 한 줄에 어떤 입력이 들어가는지 미리 알고 있어야 합니다. `Dataset` 객체, `padding`/`truncation` 결정, `DataLoader` 변환이 그 한 줄을 떠받치는 부품들입니다. 이 챕터는 그 입력 형태를 *학습 없이* 미리 손에 익히는 자리입니다.

**Phase 0와의 다리**: Yelp 5,000건은 Ch 1-6에서 줄곧 쓴 데이터. 같은 텍스트가 TF-IDF에서 sparse vector로 갔던 길이, 이번엔 WordPiece에서 `input_ids` + `attention_mask` 텐서 쌍으로 가는 길을 봅니다.

## 토크나이저 노트 — `padding` / `truncation` / `max_length`

WordPiece가 출력하는 시퀀스 길이는 입력 텍스트마다 다릅니다. 그런데 모델은 **고정된 shape의 텐서 배치** 를 받아야 하므로, 이 둘 사이를 맞추는 세 가지 옵션이 있습니다.

| 옵션 | 의미 | 언제 쓰나 |
|---|---|---|
| `padding=False` | 패딩 없음 (기본값). 시퀀스 길이가 다 다름 | 한 문장씩 처리할 때 |
| `padding=True` | **배치 안 가장 긴 시퀀스 길이까지** padding | 일반 학습 (효율적, dynamic padding) |
| `padding="max_length"` | **항상 `max_length` 까지** padding (짧으면 패딩, 길면 자름) | TPU·고정 shape 필요할 때 |
| `truncation=True` | `max_length` 초과분은 *잘라냄* | 항상 같이 두는 게 안전 (긴 입력 방지) |
| `max_length=N` | 길이 상한 (모델별 사전학습 한도 — BERT 512) | 메모리/속도 trade-off |

**패딩이 들어간 자리는 `attention_mask=0`** 으로 표시됩니다. 모델은 이 mask를 보고 self-attention에서 패딩 토큰을 무시하므로, 아무리 길게 패딩을 붙여도 학습 결과는 달라지지 않습니다. 다만 그만큼 속도와 메모리만 낭비될 뿐입니다.

이번 챕터에서 위 세 옵션을 직접 호출해 input_ids와 attention_mask가 어떻게 변하는지 봅니다.

## 이 장의 구성

[[SubPages]]
