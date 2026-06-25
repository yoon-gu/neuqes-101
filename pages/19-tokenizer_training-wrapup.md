## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `tokenizers.Tokenizer` | 학습·인코딩의 최상위 컨테이너 | Ch 20+ 는 사전학습 토크나이저를 `AutoTokenizer` 로 받음 |
| `tokenizers.models.WordPiece` | BERT 표준 subword 모델 | (동일 알고리즘이 BERT 사전학습에 사용) |
| `tokenizers.models.WordLevel` | 어절 단위 (비교용) | Ch 19 한정 |
| `tokenizers.trainers.WordPieceTrainer` | likelihood 기반 vocab 학습 | 같음 |
| `tokenizers.pre_tokenizers.BertPreTokenizer` | BERT 표준 공백·구두점 분할 | 같음 |
| `tokenizers.processors.TemplateProcessing` | `[CLS] / [SEP]` 자동 부착 | Ch 20+ MLM 입력 포맷 |
| `transformers.PreTrainedTokenizerFast` | HF 표준 인터페이스로 wrap | Ch 20 에서 표준 패턴 |

## 체크포인트 질문

1. WordPiece 의 `##` prefix 가 의미하는 바는 무엇이고, WordLevel 토크나이저에는 왜 `##` 토큰이 0 개인가요?
2. 같은 vocab 크기 (8K) 에서 한국어 WordLevel 의 UNK 비율이 영어 WordLevel 보다 *훨씬* 높은 이유 두 가지를 들어보세요.
3. vocab 크기를 키우면 mean 토큰 수는 줄고 UNK 비율도 줄어드는데, 왜 BERT 의 표준 vocab 이 무한대가 아니라 30K 정도인가요?
4. `unforgettable` 이 학습 코퍼스에 *없는* 단어일 때, WordPiece 와 WordLevel 이 각각 어떻게 처리하나요?

## FAQ

### Q1. (이론) WordPiece 와 BPE 는 무엇이 다른가요?

알고리즘 *흐름* 은 거의 같습니다 — 둘 다 *문자 단위 vocab* 에서 시작해 자주 등장하는 *조각 쌍* 을 병합하며 vocab 을 키웁니다. 차이는 *어떤 쌍을 병합할지* 의 기준:

- **BPE** (GPT 시리즈): 단순히 *코퍼스에서 가장 빈번한* 쌍을 병합 (frequency-based).
- **WordPiece** (BERT 시리즈): *언어 모델 likelihood 가 가장 많이 오르는* 쌍을 병합 (likelihood-based). 식: $\text{score} = \dfrac{\text{count}(AB)}{\text{count}(A) \cdot \text{count}(B)}$.

실무적 차이는 *작아* 두 알고리즘의 토큰화 결과가 비슷한 경우가 많습니다. Ch 24 (GPT prototype) 에서 BPE 를 직접 다룹니다.

### Q2. (실무) `vocab_size` 를 8K 로 잡았는데 BERT 표준 30K 와 무엇이 다른가요?

vocab 이 작으면 *한 단어가 여러 조각* 으로 더 잘게 쪼개져 sequence 길이가 길어집니다. 길어진 sequence 는 attention 비용 ($O(n^2)$) 을 늘려 학습이 *느려짐*. 반대로 vocab 이 크면 *임베딩 테이블* 자체가 커져 모델 파라미터가 늘어남 ($V \times H$, BERT-base 면 $30000 \times 768 \approx 23M$).

실무 trade-off — 모델이 작을수록 (이 챕터 다음 Ch 20 처럼 *scratch* 작은 BERT) 작은 vocab (8K-16K) 이 합리적, 큰 모델은 큰 vocab (30K-50K) 이 안정적.

```python
# 임베딩 테이블 파라미터 수 어림
vocab_size = 8000
hidden = 256        # 작은 BERT 의 hidden
embed_params = vocab_size * hidden  # = 2.05M
```

### Q3. (실무) `[UNK]` 토큰이 학습에 *실제로* 얼마나 나쁜가요?

모델의 `[UNK]` 임베딩은 *단 하나의 벡터* — 어떤 단어가 `[UNK]` 로 변환되든 같은 임베딩으로 들어갑니다. 즉 *모든 모르는 단어가 같은 자리에 모이는* 셈. 분류 task 라면 한 두 개 UNK 는 문맥으로 보완되어 큰 영향 없지만, 생성·번역 task 라면 정보 손실이 직접 출력에 드러납니다.

이 챕터의 결과 (WordPiece 한국어 UNK 거의 0% vs WordLevel 한국어 UNK 5-15%) 는 한국어 생성 모델 (Ch 26 의 한국어 작은 GPT) 가 *왜 byte-level BPE 같은 subword* 를 쓰는지의 직접적인 이유.

### Q4. (실무) 의료·법률 같은 *전문 도메인* 에서 직접 학습한 토크나이저는 얼마나 효과적인가요?

전문 도메인 어휘는 일반 위키·뉴스 코퍼스에 거의 없어서 표준 BERT 토크나이저는 그 단어들을 *너무 잘게* 쪼갭니다. 예: `corticosteroid` → `cor`, `##ti`, `##co`, `##ster`, `##oid` (5 토큰). 도메인 코퍼스로 직접 학습하면 `corticosteroid` 가 1 토큰으로 들어가 (a) 입력이 짧아져 학습 빠름 (b) 모델이 *의미 단위* 로 학습 — BioBERT, LegalBERT 같은 도메인 모델이 모두 이 패턴.

도메인 코퍼스가 1GB+ 정도 모이면 (`vocab_size=20000-30000`) 직접 학습할 가치가 있습니다.

### Q5. (이론) `pre_tokenizer = Whitespace()` 와 `BertPreTokenizer()` 의 차이는?

- **`Whitespace`**: 공백만으로 1차 분할. `"don't"` → `["don't"]` (한 덩어리).
- **`BertPreTokenizer`**: 공백 + *구두점 분리*. `"don't"` → `["don", "'", "t"]`. BERT 표준 동작.

BERT 의 토큰화 결과를 그대로 재현하려면 `BertPreTokenizer`. 단순 비교용·교육용이면 `Whitespace` 가 결과가 직관적. 이 챕터의 WordPiece 는 `BertPreTokenizer`, WordLevel 은 `Whitespace` — 두 알고리즘이 *표준적으로 짝지어지는* 방식을 그대로 따랐습니다.

### Q6. (실무) 학습한 토크나이저를 Ch 20+ 에서 어떻게 사용하나요?

```python
from transformers import PreTrainedTokenizerFast

# Ch 19 에서 저장한 파일 로드 + HF 인터페이스 wrap
hf_tok = PreTrainedTokenizerFast(
    tokenizer_file="./tokenizers_ch19/en_wordpiece.json",
    unk_token="[UNK]", pad_token="[PAD]",
    cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
)

# 이제 AutoModel 과 동일 패턴으로 사용
enc = hf_tok("hello world", return_tensors="pt", padding=True)
```

단, Ch 20+ 의 *실제 노트북* 은 학습 안정성을 위해 `AutoTokenizer.from_pretrained("bert-base-uncased")` 같은 표준 사전학습 토크나이저를 가져옵니다 — 이번 챕터에서 *직접 학습이 가능함을 본 다음에야* 표준 도구로 돌아가는 흐름.

### Q7. (이론) WordPiece 학습이 *어떻게* 작은 단어부터 큰 단어로 쌓아 올라가나요?

대략적인 흐름:

1. **초기 vocab**: 코퍼스의 모든 *문자* 와 `##문자` (subword 형태) — 보통 200-300 개로 출발.
2. **반복**: 가능한 모든 인접 쌍 (`A` + `B`) 의 likelihood score 계산 → 가장 점수가 높은 쌍을 vocab 에 추가, 코퍼스에서 그 쌍을 *하나의 토큰* 으로 병합.
3. **종료**: vocab 크기가 목표 (`vocab_size`) 에 도달하면 중단.

결과적으로 *작은 토큰* 부터 *큰 토큰* 으로 vocab 이 자라며, 최종 vocab 에는 *짧은 조각* (`##s`, `##ing`) 부터 *완성된 단어* (`hello`, `world`) 까지 섞여 있습니다. 인코딩 시에는 *가장 긴 매칭 우선* (longest-match) 으로 단어를 쪼개 사용.

## 다음 챕터 예고

**Chapter 20. 작은 BERT 직접 사전학습 (영어 MLM)**

- 이번 챕터에서 *경험한* 토크나이저 학습 위에, *모델 자체* 를 from-scratch 로 사전학습.
- 모델: 작은 BERT (n_layer=4, hidden=256) — `BertConfig` 로 직접 설계.
- 토크나이저: 표준 `bert-base-uncased` 의 WordPiece 를 가져옴 (학습 안정성 우선).
- 데이터: `yelp_polarity` 의 text (라벨 무시) — *MLM* 사전학습.
- Loss: `CrossEntropyLoss` (마스킹된 위치의 토큰 예측).
- Ch 21 에서 이 사전학습 모델로 Yelp 이진 분류 → Ch 10 (DistilBERT 사전학습) 과 *직접 비교* — *직접 사전학습한 작은 BERT* 가 *대규모 사전학습된 DistilBERT* 와 얼마나 차이 나는지.

> **변하는 축**: Phase 3 안에서 *모델 학습 task 가 등장* — 사전학습 (MLM) 이라는 *새로운 학습 방식* 이 핵심.
