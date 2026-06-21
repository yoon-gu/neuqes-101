**목표**: sklearn 시대를 마치고 `transformers` 라이브러리를 만납니다. **5줄짜리 코드** 로 사전학습된 DistilBERT를 돌려보고, 그 한 줄 뒤에 어떤 일이 일어났는지 단계별로 풀어 헤칩니다.

**환경**: Google Colab — **T4 GPU 권장** (런타임 → 런타임 유형 변경 → T4 GPU). 이번 챕터부터 GPU 메모리(VRAM) 추적이 등장하니 GPU 런타임에서 돌리면 모델 로드 → VRAM 증가가 한눈에 보입니다. CPU 런타임도 추론 자체는 동작하지만 `!nvidia-smi` 셀은 에러납니다.

**예상 소요 시간**: 약 10분 (학습 없음, 추론만)


## 학습 흐름

1. 🚀 **실습**: `pipeline("sentiment-analysis")` 한 줄로 감성 분석 돌리기
2. 🔬 **해부**: `pipeline` 안에서 일어나는 3단계 (tokenizer / model / post-processing)
3. 🛠️ **변형**: `pipeline` 없이 같은 일을 4단계로 직접 재현

## 변화추적표

**Phase 1 시작** — sklearn 시대 끝, `transformers` 등장.

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 이진화 | (2차원) | softmax | `CrossEntropyLoss` |
| 5 | `LogisticRegression()` (multinomial 자동) | `TfidfVectorizer()` | Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| 6 | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 항목 합성 | (5차원) | per-label sigmoid | per-label `BCEWithLogitsLoss` |
| **7 ← 여기** | `pipeline("sentiment-analysis")` | `AutoTokenizer.from_pretrained(...)` | 간단 영어 예시 | **사전학습 헤드** | softmax | — (추론만) |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

## 변경점 (Diff from Ch 6)

| 축 | Ch 6 | Ch 7 |
|---|---|---|
| 라이브러리 | `sklearn` | **`transformers`** (Hugging Face) |
| 모델 | `OneVsRestClassifier(LogisticRegression())` (학습) | **`pipeline("sentiment-analysis")`** (사전학습 + 추론) |
| 토크나이저 | `TfidfVectorizer()` (단어 단위 어휘 학습) | **`AutoTokenizer`** (사전학습된 WordPiece) |
| 학습 단계 | sklearn `fit()` 한 번에 학습 | **학습 없음** — 사전학습 가중치 로드 후 추론만 |
| 데이터 | Yelp 5,000건 | 간단 영어 예시 문장 (분해 시연용) |
| 하드웨어 | CPU | CPU 또는 T4 GPU (이번 챕터는 추론만이라 어느 쪽도 OK) |

**왜 학습 없이 시작하나?** Phase 1 첫 챕터는 `transformers` 의 *추상화 계층* 을 익히는 데 집중합니다. `pipeline` 한 줄 뒤에 토크나이저·모델·후처리 3단계가 어떻게 굴러가는지 손에 잡히면, Ch 8(Tokenizer/Datasets 해부)와 Ch 9(BERT 회귀 첫 학습)에서 `Trainer` 가 등장할 때 코드를 *읽을* 수 있습니다.

## 토크나이저 노트 — 첫 WordPiece 등장

이번 챕터의 토크나이저는 **사전학습된 WordPiece**. Phase 0의 `TfidfVectorizer` 와 *완전히 다른 패러다임* 입니다.

| 비교 | TF-IDF (Phase 0) | WordPiece (Phase 1+) |
|---|---|---|
| 분리 단위 | 단어 (whitespace + 정규식) | **서브워드** (자주 등장하는 문자 시퀀스) |
| 어휘 출처 | 학습 데이터에서 그때그때 학습 | **사전학습된 30,522개 어휘** (BERT 학습 시 정해짐) |
| OOV 처리 | 그냥 무시 | `[UNK]` 또는 더 작은 서브워드로 분해 |
| 특수 토큰 | 없음 | `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]` 등 |
| 출력 | sparse vector (V차원, 거의 0) | 정수 ID 시퀀스 + attention mask |

같은 문장 `"I love using Hugging Face!"` 가 어떻게 토큰화되는지 곧 직접 확인합니다 (Step 2). `##` 접두사가 보이는 단어는 어디고, 왜 그렇게 쪼개졌는지도 같이 봅니다.

> **다음 챕터(Ch 8)**: 같은 WordPiece 토크나이저를 *깊게* — `padding`, `truncation`, `max_length` 옵션과 `datasets` 라이브러리 메모리 효율까지.

## 환경 준비

Colab에는 `transformers`가 보통 설치돼 있지만, 최신 버전을 보장하기 위해 한 번 설치합니다.

### `!nvidia-smi` — GPU 메모리(VRAM) 실시간 추적

이번 챕터부터 학습·추론 코드가 GPU에 모델을 올리기 시작합니다. **`!nvidia-smi`** 는 NVIDIA에서 제공하는 명령행 도구로, 현재 GPU의 VRAM 사용량·온도·전력을 한 번에 보여줍니다. Colab 셀에서 `!` 접두사로 호출 가능.

T4의 총 VRAM은 **약 15.36 GB** (= 15,360 MiB). 모델·옵티마이저·activation을 모두 이 안에 담아야 합니다 — Ch 9 이후 학습 chapter에서는 이 한도와 자주 부딪히게 되어요.

**baseline** — 아직 아무 모델도 안 올린 상태:

**무엇을 봐야 하나** — 출력 가운데 줄 `Memory-Usage` 칸:

```
| ... |  XXX MiB / 15360MiB | ...
        └─ used    └─ total
```

- 처음엔 ~3-200 MiB 정도. CUDA 컨텍스트가 잡혀 있는 만큼만.
- 모델을 GPU에 올릴 때마다 `used` 가 증가합니다.
- `Volatile GPU-Util` 은 *현재* GPU가 일하는 비율 — 학습 중에는 90-100% 가까이.

**Python으로도 확인 가능** (셀 내부에서 변수로 받고 싶을 때):

```python
if torch.cuda.is_available():
    used  = torch.cuda.memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU memory: {used:.0f} / {total:.0f} MiB")
```

> Tip: `!nvidia-smi` 는 *시스템 전체* VRAM을 보여주고, `torch.cuda.memory_allocated()` 는 *현재 PyTorch 프로세스* 의 할당량만 보여줍니다 — 후자는 캐시·예약 메모리는 빼고 실제 텐서가 점유한 양에 가깝습니다.

## 보너스: 토크나이저마다 어휘가 다르다

지금까지는 DistilBERT의 WordPiece 토크나이저 *하나* 만 봤습니다. 그런데 모델이 바뀌면 토크나이저도 바뀌고, **같은 문장이 완전히 다른 토큰 리스트로 쪼개집니다** — 어휘 사전이 사전학습 단계에서 따로 만들어졌기 때문이에요.

세 가지 대표 토크나이저를 나란히 비교합니다.

| 모델 | 알고리즘 | 어휘 크기 | 대소문자 |
|---|---|---|---|
| `distilbert-base-uncased` | **WordPiece** | 30,522 | 모두 소문자로 |
| `bert-base-cased` | **WordPiece** | 28,996 | 대소문자 유지 |
| `gpt2` | **BPE** (Byte Pair Encoding) | 50,257 | 대소문자 유지 |

WordPiece와 BPE는 둘 다 *서브워드 알고리즘* 이지만 학습·표기 방식이 달라서 토큰 모양이 시각적으로도 구분됩니다 — `##` 접두사 vs `Ġ` (공백) 접두사.

### 특수 토큰(special token)이란

`[CLS]`, `[SEP]` 같은 토큰은 *문장 텍스트* 가 아니라 **모델에 신호를 주기 위해 사전학습 단계에서 정해진 약속** 입니다. 어휘 사전에 별도 ID로 들어 있고, 토크나이저가 입력에 자동으로 붙입니다.

| 토큰 | 풀이름 | 위치 | 역할 |
|---|---|---|---|
| `[CLS]` | Classification | 모든 입력 *맨 앞* | 분류 헤드는 *이 위치* 의 hidden state를 사용. attention을 통해 전체 문장 정보가 [CLS]로 모이도록 학습됨. |
| `[SEP]` | Separator | 문장 끝, 두 문장 사이 | 한 문장 입력엔 `[CLS] ... [SEP]`. 두 문장이면 `[CLS] A [SEP] B [SEP]` (NSP·QA·NLI 등). |
| `[PAD]` | Padding | 짧은 문장 끝 | 배치 안 문장 길이를 맞추는 더미 토큰. **`attention_mask=0`** 으로 표시해 모델이 무시. |
| `[UNK]` | Unknown | 어디든 | 어휘 사전에 없는 토큰. WordPiece는 거의 항상 더 작은 서브워드로 쪼개므로 실제 출현은 드뭄. |
| `[MASK]` | Mask | 사전학습 시 입력 일부 | BERT 사전학습의 *Masked LM* — 입력 토큰 15%를 `[MASK]` 로 가리고 모델이 맞추도록. 추론 시엔 거의 안 등장(fill-mask 데모 제외). |

**autoregressive 모델 (GPT-2)** 은 `[CLS]/[SEP]` 가 없습니다 — 다음 토큰을 *순서대로* 예측하는 구조라 문장 시작/끝 마커가 별도로 필요 없고, `<|endoftext|>` 라는 단일 토큰이 BOS/EOS 역할을 겸합니다.

이 약속은 *모델별로 다릅니다*. RoBERTa는 `<s>`, `</s>` 를, T5는 `<pad>`, `<extra_id_0>` 등을 씁니다 — `tokenizer.special_tokens_map` 으로 한 번에 확인 가능.

**관찰 포인트**

- **`##` 접두사 (WordPiece)**: DistilBERT·BERT는 단어 중간 서브워드를 `##xxx` 로 표시. 예: `tokenization → ['token', '##ization']`. 이전 토큰의 *연속* 이라는 신호.
- **`Ġ` 접두사 (BPE)**: GPT-2는 토큰 앞에 공백이 있었는지를 `Ġ` (Latin small letter G with stroke) 로 표시. 예: `Ġlove` 는 "love 앞에 공백이 있었다". 토큰화/디코딩이 정확히 역연산이 되도록 하는 표기.
- **대소문자**: `bert-base-cased` 는 `Hugging Face` 의 `H`, `F` 를 그대로 보존. `distilbert-base-uncased` 는 모두 소문자. 이름·고유명사 처리에서 차이가 큽니다.
- **vocab 크기**: GPT-2가 50K 로 가장 큼. BPE는 영어 외 다양한 토큰(드문 조합·바이트 단위)도 어휘에 포함하기 때문. WordPiece는 영어 중심이라 30K로 충분.
- **특수 토큰**: BERT 계열은 `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` 가 모두 정의되지만 GPT-2는 `[CLS]/[SEP]` 가 없습니다 (autoregressive 모델은 문장 시작/끝 마커를 따로 안 둠 — `<|endoftext|>` 하나가 BOS/EOS 역할). PAD도 없어 추가 설정이 필요한 경우가 흔함.

**왜 같은 문장이 다른 토큰 시퀀스가 되나?** 어휘 사전이 *사전학습 데이터* 에 따라 만들어집니다.

- BERT는 BookCorpus + Wikipedia로 학습됐고, 영어 중심 어휘.
- GPT-2는 더 다양한 웹 텍스트(Reddit 등)로 학습됐고 BPE라 어휘가 더 풍부.
- 한국어 BERT(`klue/bert-base`, Ch 15)는 한국어 코퍼스로 다시 학습돼 한국어 어휘를 보유 — 같은 문장 `"안녕"` 도 영어 BERT면 `[UNK]` 또는 글자 단위로 쪼개지지만 한국어 BERT엔 한 토큰으로 들어갑니다.

**실무 함의**: 모델을 갈아 끼울 때 토크나이저도 *반드시 짝* 으로 바꿔야 합니다. `AutoTokenizer.from_pretrained(model_name)` 의 model_name 이 모델 자체와 일치해야 하는 이유 — 학습 때 본 어휘와 추론 때 입력 어휘가 같아야 모델이 의미를 이해합니다.

## 보너스: `model.config` 안에 뭐가 있나

위에서 `model.config.id2label` 로 라벨 이름을 알아냈습니다. `config` 객체에는 모델의 *설계도* 가 모두 들어있어서, 모델을 받아왔을 때 가장 먼저 들여다보면 좋은 곳입니다.

분류 작업에서 자주 쓰는 속성들을 한 번에 출력합니다.

> 📒 **더 깊이 보고 싶다면 — 부록 노트북**
>
> [`appendix_model_config.ipynb`](./appendix_model_config.ipynb) 에서 다음을 다룹니다:
> - `PretrainedConfig` 의 정체와 클래스 계층 (BertConfig / GPT2Config / T5Config / ViTConfig …)
> - `AutoConfig.from_pretrained` 로 *가중치 없이* config만 로드
> - 5종 모델(bert / distilbert / gpt2 / t5 / roberta) config를 한 표에 비교 + ViT(비전) 사례
> - 분류 헤드 갈아끼우는 `from_pretrained` 인자 패턴 (`num_labels`, `problem_type`)
> - 공식 문서 링크: <https://huggingface.co/docs/transformers/en/main_classes/configuration>
>
> Colab으로 바로: [Open](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/07_bert_pipeline/appendix_model_config.ipynb). 본 챕터 흐름과 별개라 시간 될 때 보시면 됩니다.

**자주 쓰는 속성 한눈에 보기**

| 속성·호출 | 의미 | 자주 쓰는 곳 |
|---|---|---|
| `model.config._name_or_path` | 모델 식별자 (Hugging Face Hub repo 또는 로컬 경로) | 어떤 모델인지 빠르게 확인 |
| `model.config.model_type` | 모델 아키텍처 종류 (`bert`, `distilbert`, `gpt2`, ...) | 분기 처리 |
| `sum(p.numel() for p in model.parameters())` | **파라미터 총 개수** (config 속성은 아니지만 항상 같이 봄) | VRAM 사용량 추정, 모델 비교 |
| `model.config.hidden_size` | hidden state 차원 (예: 768 / 1024) | 분류 헤드를 직접 만들 때 |
| `model.config.vocab_size` | 어휘 크기 (토크나이저와 일치해야 함) | 토크나이저 호환 검증 |
| `model.config.max_position_embeddings` | 입력 토큰 수 상한 | `truncation=True, max_length=...` 결정 |
| `model.config.num_labels` | 분류 헤드 출력 클래스 수 | 모델 로드 시 명시: `num_labels=5` |
| `model.config.id2label` / `label2id` | 클래스 인덱스 ↔ 이름 매핑 | 추론 결과 해석, 학습 후 모델 카드 친절도 |
| `model.config.problem_type` | `"regression"` / `"single_label_classification"` / `"multi_label_classification"` — `Trainer` 가 자동 loss 결정 | Ch 9·11·12에서 명시적으로 사용 |

**실무 패턴**: 새 모델을 받자마자 `print(model.config)` 또는 `cfg.to_dict()` 로 내용을 먼저 본다 → 입력/출력 가정을 확인하고 토크나이저·`Trainer` 설정과 일치시킴.

```python
# 새 모델 받자마자 한 줄 검사
print(model.config)            # 모든 설정 한꺼번에
print(model.config.to_dict())  # dict 형태 (JSON 직렬화 가능)
```

## 이 장의 구성

- [07-1. 실습: 일단 돌려봅시다](07-bert_pipeline-practice.md)
- [07-2. 해부: pipeline 안에서는 뭐가 일어났을까?](07-bert_pipeline-anatomy.md)
- [07-3. 변형: pipeline 없이 직접 해보기](07-bert_pipeline-variation.md)
- [07-4. 정리와 FAQ](07-bert_pipeline-wrapup.md)
