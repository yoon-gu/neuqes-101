"""Build 07_bert_pipeline/07_bert_pipeline.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "07_bert_pipeline" / "07_bert_pipeline.ipynb"

cells = []
_counter = 0


def _cid():
    global _counter
    _counter += 1
    return f"cell{_counter:03d}"


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": text,
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ----- 1. Title -----
md(r"""# Chapter 7. BERT 첫 만남 — `pipeline` 한 줄과 그 안의 4단계

**목표**: sklearn 시대를 마치고 `transformers` 라이브러리를 만납니다. **5줄짜리 코드** 로 사전학습된 DistilBERT를 돌려보고, 그 한 줄 뒤에 어떤 일이 일어났는지 단계별로 풀어 헤칩니다.

**환경**: Google Colab — CPU도 가능, T4 GPU면 더 빠름 (런타임 → 런타임 유형 변경 → T4 GPU)

**예상 소요 시간**: 약 10분 (학습 없음, 추론만)

---

## 학습 흐름

1. 🚀 **실습**: `pipeline("sentiment-analysis")` 한 줄로 감성 분석 돌리기
2. 🔬 **해부**: `pipeline` 안에서 일어나는 3단계 (tokenizer / model / post-processing)
3. 🛠️ **변형**: `pipeline` 없이 같은 일을 4단계로 직접 재현""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

**Phase 1 시작** — sklearn 시대 끝, `transformers` 등장.

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 1 | (TF-IDF) | `TfidfVectorizer()` | Yelp 5,000 | — | — | — |
| 2 | `LinearRegression()` | `TfidfVectorizer()` | Yelp (별점 1-5) | (1차원) | 없음 | `MSELoss` |
| 3 | `LogisticRegression()` | `TfidfVectorizer()` | Yelp 이진화 | (1차원) | sigmoid | `BCEWithLogitsLoss` |
| 4 | `LogisticRegression(multi_class="multinomial")` | `TfidfVectorizer()` | Yelp 이진화 | (2차원) | softmax | `CrossEntropyLoss` |
| 5 | `LogisticRegression(multi_class="multinomial")` | `TfidfVectorizer()` | Yelp 5클래스 | (5차원) | softmax | `CrossEntropyLoss` |
| 6 | `OneVsRestClassifier(LogisticRegression())` | `TfidfVectorizer()` | Yelp + 측면 합성 | (5차원) | per-label sigmoid | per-label `BCEWithLogitsLoss` |
| **7 ← 여기** | `pipeline("sentiment-analysis")` | `AutoTokenizer.from_pretrained(...)` | 간단 영어 예시 | **사전학습 헤드** | softmax | — (추론만) |

전체 19챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 6)

| 축 | Ch 6 | Ch 7 |
|---|---|---|
| 라이브러리 | `sklearn` | **`transformers`** (Hugging Face) |
| 모델 | `OneVsRestClassifier(LogisticRegression())` (학습) | **`pipeline("sentiment-analysis")`** (사전학습 + 추론) |
| 토크나이저 | `TfidfVectorizer()` (단어 단위 어휘 학습) | **`AutoTokenizer`** (사전학습된 WordPiece) |
| 학습 단계 | sklearn `fit()` 한 번에 학습 | **학습 없음** — 사전학습 가중치 로드 후 추론만 |
| 데이터 | Yelp 5,000건 | 간단 영어 예시 문장 (분해 시연용) |
| 하드웨어 | CPU | CPU 또는 T4 GPU (이번 챕터는 추론만이라 어느 쪽도 OK) |

**왜 학습 없이 시작하나?** Phase 1 첫 챕터는 `transformers` 의 *추상화 계층* 을 익히는 데 집중합니다. `pipeline` 한 줄 뒤에 토크나이저·모델·후처리 3단계가 어떻게 굴러가는지 손에 잡히면, Ch 8(Tokenizer/Datasets 해부)와 Ch 9(BERT 회귀 첫 학습)에서 `Trainer` 가 등장할 때 코드를 *읽을* 수 있습니다.""")

# ----- 4. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트 — 첫 WordPiece 등장

이번 챕터의 토크나이저는 **사전학습된 WordPiece**. Phase 0의 `TfidfVectorizer` 와 *완전히 다른 패러다임* 입니다.

| 비교 | TF-IDF (Phase 0) | WordPiece (Phase 1+) |
|---|---|---|
| 분리 단위 | 단어 (whitespace + 정규식) | **서브워드** (자주 등장하는 문자 시퀀스) |
| 어휘 출처 | 학습 데이터에서 그때그때 학습 | **사전학습된 30,522개 어휘** (BERT 학습 시 정해짐) |
| OOV 처리 | 그냥 무시 | `[UNK]` 또는 더 작은 서브워드로 분해 |
| 특수 토큰 | 없음 | `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]` 등 |
| 출력 | sparse vector (V차원, 거의 0) | 정수 ID 시퀀스 + attention mask |

같은 문장 `"I love using Hugging Face!"` 가 어떻게 토큰화되는지 곧 직접 확인합니다 (Step 2). `##` 접두사가 보이는 단어는 어디고, 왜 그렇게 쪼개졌는지도 같이 봅니다.

> **다음 챕터(Ch 8)**: 같은 WordPiece 토크나이저를 *깊게* — `padding`, `truncation`, `max_length` 옵션과 `datasets` 라이브러리 메모리 효율까지.""")

# ----- 5. 환경 준비 -----
md(r"""## 0. 환경 준비

Colab에는 `transformers`가 보통 설치돼 있지만, 최신 버전을 보장하기 위해 한 번 설치합니다.""")

# ----- 6. install -----
code(r"""!pip install -q transformers""")

# ----- 7. GPU check -----
code(r"""import torch
print(f"PyTorch:       {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
else:
    print("CPU로 실행됩니다 (이번 챕터는 추론만이라 OK, 다만 Ch 9부터 학습은 GPU 권장)")""")

# ----- 8. 실습 도입 -----
md(r"""## 1. 🚀 실습: 일단 돌려봅시다

Hugging Face의 `pipeline` 은 **"모델 다운로드 → 토큰화 → 추론 → 결과 후처리"** 를 한 줄로 묶어주는 함수입니다.

감성 분석(sentiment analysis)부터 시작합니다.""")

# ----- 9. pipeline 한 줄 -----
code(r"""from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I love using Hugging Face! It's so simple.")""")

# ----- 10. 첫 실행 안내 -----
md(r"""**결과**: `[{'label': 'POSITIVE', 'score': 0.9998...}]`

처음 실행 시 모델 다운로드(약 250MB)에 30초~1분 정도 걸립니다. 두 번째부터는 캐시되어 즉시 실행.

여러 문장도 한 번에:""")

# ----- 11. 여러 문장 -----
code(r"""results = classifier([
    "This movie was fantastic.",
    "Worst experience ever.",
    "It was okay, nothing special.",
])
for r in results:
    print(r)""")

# ----- 12. 다른 task -----
md(r"""### 다른 task도 같은 패턴

`pipeline` 의 첫 인자만 바꾸면 다른 NLP 작업을 즉시 할 수 있습니다.""")

# ----- 13. text-generation -----
code(r"""# 텍스트 생성 (GPT-2)
generator = pipeline("text-generation", model="gpt2")
generator("Hugging Face is", max_length=30, num_return_sequences=1)""")

# ----- 14. fill-mask -----
code(r"""# 마스크 채우기 (BERT)
unmasker = pipeline("fill-mask", model="bert-base-uncased")
unmasker("Hugging Face is a [MASK] for NLP.")""")

# ----- 15. 해부 도입 -----
md(r"""## 2. 🔬 해부: pipeline 안에서는 뭐가 일어났을까?

`pipeline("sentiment-analysis")` 한 줄이 사실은 **3단계** 로 구성됩니다.

```
입력 텍스트
   ↓ [1] Tokenizer  (텍스트 → 숫자 ID)
input_ids, attention_mask
   ↓ [2] Model      (숫자 → 로짓)
logits
   ↓ [3] Post-processing (로짓 → 라벨)
{'label': 'POSITIVE', 'score': 0.9998}
```

### 등장 인물 정리

| 컴포넌트 | 라이브러리 | 역할 |
|---|---|---|
| `pipeline` | `transformers` | 위 3단계를 묶은 wrapper |
| Tokenizer | `transformers` (내부적으로 `tokenizers`) | 텍스트를 모델이 먹을 수 있는 숫자로 변환 |
| Model | `transformers` + `torch` | 실제 신경망 forward 연산 |

현재 `classifier` 객체가 어떤 모델/토크나이저를 사용하는지 확인합니다.""")

# ----- 16. classifier 내부 확인 -----
code(r"""print(f"모델:               {classifier.model.config._name_or_path}")
print(f"모델 클래스:         {type(classifier.model).__name__}")
print(f"토크나이저 클래스:   {type(classifier.tokenizer).__name__}")
print(f"라벨 매핑:           {classifier.model.config.id2label}")""")

# ----- 17. 모델 정체 설명 -----
md(r"""기본 모델은 **`distilbert-base-uncased-finetuned-sst-2-english`** 입니다.

- **DistilBERT**: BERT를 40% 작게 만든 경량화 모델 (학생 모델, 지식 증류로 만듦).
- **SST-2**: 영화 리뷰 감성 분류 데이터셋 (Stanford Sentiment Treebank).
- 즉, "BERT를 영화 리뷰 데이터로 파인튜닝한 모델"입니다 — 우리가 Ch 9에서 직접 할 작업의 *완성된* 버전.""")

# ----- 18. 변형 도입 -----
md(r"""## 3. 🛠️ 변형: pipeline 없이 직접 해보기

이제 `pipeline` 이 감춰뒀던 단계를 **직접 한 줄씩 실행** 합니다. 이 부분을 이해하면 앞으로 모든 모델을 자유롭게 다룰 수 있습니다.

### Step 1: Tokenizer와 Model 직접 로드""")

# ----- 19. AutoTokenizer + AutoModel -----
code(r"""from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("✅ 로드 완료")
print(f"  tokenizer 클래스: {type(tokenizer).__name__}")
print(f"  model 클래스:     {type(model).__name__}")""")

# ----- 20. Auto 클래스 설명 -----
md(r"""> **잠깐, `Auto`가 뭔가요?**
>
> `AutoTokenizer`, `AutoModel...` 같은 `Auto` 계열 클래스는 모델 이름만 주면 **알아서 적합한 클래스를 골라주는 팩토리** 입니다.
>
> - DistilBERT 모델 → `DistilBertTokenizer`, `DistilBertForSequenceClassification`
> - BERT 모델 → `BertTokenizer`, `BertForSequenceClassification`
> - GPT-2 모델 → `GPT2Tokenizer`, `GPT2LMHeadModel`
>
> 직접 `BertTokenizer.from_pretrained(...)`라고 써도 되지만, `AutoTokenizer` 를 쓰면 모델만 바꿔도 코드가 그대로 동작합니다.""")

# ----- 21. Step 2 도입 -----
md(r"""### Step 2: 텍스트 → 숫자 (Tokenization)""")

# ----- 22. 토큰화 -----
code(r"""text = "I love using Hugging Face!"

# 토큰화 결과 살펴보기
tokens = tokenizer.tokenize(text)
print(f"토큰들: {tokens}")

# 모델 입력용 텐서 만들기
inputs = tokenizer(text, return_tensors="pt")
print("\n모델 입력:")
for key, value in inputs.items():
    print(f"  {key}: {value}")""")

# ----- 23. 토큰화 관찰 -----
md(r"""**관찰 포인트**
- `input_ids`: 각 토큰의 정수 ID. 맨 앞 `101`은 `[CLS]`, 맨 뒤 `102`는 `[SEP]` 특수 토큰.
- `attention_mask`: `1`이면 "이 위치는 진짜 토큰", `0`이면 "패딩이니 무시하라"는 신호 (이번엔 패딩 없음 → 모두 1).
- `tokenize()` 와 `tokenizer()` 의 차이 — 전자는 토큰 문자열만 반환, 후자는 모델 입력용 텐서까지 다 만들어줌.""")

# ----- 24. ID 디코딩 -----
code(r"""# input_ids를 다시 토큰으로 디코딩해서 확인
print(f"{'ID':>5}    토큰")
print("-" * 30)
for token_id in inputs["input_ids"][0]:
    token = tokenizer.decode([token_id])
    print(f"{token_id.item():>5}    {token!r}")""")

# ----- 25. Step 3 도입 -----
md(r"""### Step 3: 숫자 → 로짓 (Model forward)""")

# ----- 26. forward -----
code(r"""# 추론할 때는 gradient 계산을 끄는 것이 메모리/속도에 좋음
with torch.no_grad():
    outputs = model(**inputs)

print(f"출력 객체:    {type(outputs).__name__}")
print(f"로짓 shape:   {outputs.logits.shape}  (배치 1개, 클래스 2개: NEGATIVE, POSITIVE)")
print(f"로짓 값:      {outputs.logits}")""")

# ----- 27. 로짓 설명 -----
md(r"""로짓(logits)은 모델이 뱉은 **정규화되지 않은 점수**. shape `[1, 2]`는 "배치 1개, 클래스 2개"를 의미합니다.

여기서 잠깐 — 익숙하지 않나요? Ch 4의 *softmax + 2차원 head* 구조와 정확히 같습니다. BERT는 사전학습된 *심층* 모델일 뿐, 마지막 분류 헤드는 sklearn에서 본 것과 본질이 같습니다.""")

# ----- 28. Step 4 도입 -----
md(r"""### Step 4: 로짓 → 확률/라벨 (Post-processing)""")

# ----- 29. softmax + argmax -----
code(r"""# softmax로 확률 변환
probs = torch.softmax(outputs.logits, dim=-1)
print(f"확률: {probs}")

# 가장 높은 확률의 클래스 인덱스
predicted_class_id = probs.argmax(dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]
predicted_score = probs[0, predicted_class_id].item()

print(f"\n최종 결과: {{'label': '{predicted_label}', 'score': {predicted_score:.4f}}}")""")

# ----- 30. pipeline 결과와 비교 -----
code(r"""# pipeline이 한 줄로 해주던 일을 4단계로 직접 재현했습니다. 결과를 비교해봅시다.
print(f"pipeline 결과:    {classifier(text)}")
print(f"직접 구현 결과:   [{{'label': '{predicted_label}', 'score': {predicted_score:.4f}}}]")""")

# ----- 31. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.pipeline` | 추론 원스톱 함수 (3단계 묶음) | 학습된 모델을 `pipeline`으로 감싸 사용 (Ch 9 이후) |
| `transformers.AutoTokenizer` | 모델에 맞는 WordPiece 토크나이저 자동 로드 | Ch 8에서 옵션 깊게 보기 |
| `transformers.AutoModelForSequenceClassification` | 분류 헤드가 붙은 모델 자동 로드 | Ch 9부터 직접 파인튜닝 |
| `torch` | PyTorch — 텐서 연산과 학습의 기반 | 계속 사용 (특히 Ch 9 학습부터) |""")

# ----- 32. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. `pipeline("sentiment-analysis")` 을 직접 풀어 쓰면 어떤 단계가 되는지 4단계로 설명할 수 있나요?
2. `input_ids` 와 `attention_mask` 는 각각 무슨 역할인가요?
3. 모델 출력의 `logits` 는 왜 그대로 쓰지 않고 `softmax` 를 거치나요? (Ch 4에서 본 동등성 떠올리기)
4. `AutoTokenizer.from_pretrained(...)` 를 쓸 때와 `BertTokenizer.from_pretrained(...)` 를 직접 쓸 때의 차이는 무엇인가요?""")

# ----- 33. FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) `pipeline` 이 처음 실행될 때 너무 느린데 정상인가요?

네, 정상입니다. 첫 실행 시 다음이 한꺼번에 일어납니다.

1. 모델 가중치 다운로드 (약 250MB for DistilBERT)
2. 토크나이저 사전 다운로드
3. 모델 PyTorch 로드 + (있다면) GPU 이동

두 번째부터는 캐시(`~/.cache/huggingface/`)에서 읽으므로 즉시 실행됩니다. **Colab 세션이 끊기면 캐시도 사라져** 다시 다운로드합니다 — 자주 쓴다면 Google Drive를 마운트해서 `HF_HOME` 환경변수를 Drive로 지정하면 보존됩니다.

```python
# Drive에 캐시 보존 (선택)
import os
from google.colab import drive
drive.mount("/content/drive")
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```

### Q2. (이론) 왜 BERT 토크나이저는 단어를 `##` 조각으로 쪼개나요?

이게 **WordPiece** 의 핵심입니다 — 자주 등장하는 부분 문자열을 하나의 토큰으로 두고, 새로운 단어는 *서브워드* 들의 조합으로 표현합니다.

```python
# 사전학습 어휘에 없는 단어를 어떻게 처리하는지 보기
tokenizer.tokenize("Tokenization")     # ['token', '##ization']
tokenizer.tokenize("antidisestablishmentarianism")
# → ['anti', '##dis', '##est', '##ab', '##lish', '##ment', '##arian', '##ism']
```

`##` 은 "이 토큰은 *이전 토큰의 연속*"이라는 표시입니다.

**왜 이렇게?**
- **OOV 해결**: TF-IDF는 학습 어휘에 없는 단어를 *무시* 했지만, WordPiece는 항상 더 작은 서브워드로 쪼개 표현 가능 (이론적으로 OOV 없음, 최악의 경우 글자 단위까지).
- **어휘 크기 관리**: 영어에는 수백만 단어가 있지만 BERT는 30,522개 토큰만으로 모두 표현. 형태 변형(`-ing`, `-ed`)도 일관되게 처리.
- **희귀 단어 일반화**: `unhappiness` 를 `un + happi + ness` 로 보면, 모델이 `happi` 를 알고 있으면 처음 보는 단어라도 의미 추론 가능.

### Q3. (실무) `pipeline` 결과의 `LABEL_0`, `LABEL_1` 이 무슨 의미인지 어떻게 알아내나요?

`model.config.id2label` 을 확인하면 됩니다.

```python
print(classifier.model.config.id2label)
# {0: 'NEGATIVE', 1: 'POSITIVE'} — 이 모델은 친절히 적어둠
```

**모델마다 다릅니다**. 일부 모델은 `LABEL_0`, `LABEL_1` 처럼 *추상적인* 이름만 붙어 있어 모델 카드(Hugging Face Hub의 모델 페이지) 또는 학습 데이터셋 라벨 정의를 확인해야 합니다. 우리가 Ch 9 이후 직접 학습할 때는 `id2label` 을 명시적으로 설정해 미래의 사용자에게 친절하게 만들 수 있습니다.

### Q4. (이론) `[CLS]` 와 `[SEP]` 토큰은 왜 필요한가요?

BERT의 사전학습 구조에서 비롯된 특수 토큰입니다.

- **`[CLS]`** (Classification): 입력 맨 앞에 항상 붙는 토큰. 사전학습 시 *Next Sentence Prediction* 을 위한 자리였고, 분류 작업에서는 이 위치의 hidden state(전체 문장의 표현)를 분류 헤드에 넣습니다. *모든 토큰의 정보가 attention을 통해 [CLS]로 모이도록* 학습됨.
- **`[SEP]`** (Separator): 문장 끝에 붙거나, 두 문장을 분리. 입력이 한 문장이면 `[CLS] ... [SEP]` 구조, 두 문장(질문-답변 등)이면 `[CLS] ... [SEP] ... [SEP]`.

이번 챕터의 입력 `"I love using Hugging Face!"` 의 token ID 첫 값 `101` 이 `[CLS]`, 마지막 `102` 가 `[SEP]` — 위 셀에서 확인했죠.

`AutoTokenizer` 가 이 특수 토큰을 자동으로 추가해주므로 우리가 신경 쓸 일은 거의 없지만, *왜 길이가 입력 단어 수보다 2 길게 나오는지* 이해하는 데는 이 두 토큰을 알아야 합니다.

### Q5. (실무) GPU 없이도 `pipeline` 이 돌아가나요?

네, CPU에서도 작동합니다. 다만 속도 차이가 큽니다.

| 환경 | DistilBERT 1문장 추론 |
|---|---|
| Colab CPU | ~80-150ms |
| Colab T4 GPU | ~5-15ms |
| 큰 BERT 모델 | CPU는 5-10x 더 느림 |

**추론** 만 한다면 CPU도 실용적입니다 (예: API 서빙은 보통 GPU지만 가벼운 데모는 CPU). **학습** 은 거의 항상 GPU 필수 — Ch 9 이후 학습할 때는 T4 런타임으로 바꿔야 합니다.

### Q6. (실무) `AutoTokenizer` 와 `BertTokenizer` 를 직접 사용하는 것의 차이는?

기능적으론 거의 같지만 **코드 일반성** 이 다릅니다.

```python
# 방식 A: 직접 클래스 (모델 종류를 코드에 박음)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# 만약 GPT-2를 쓰려면 → import 와 클래스 모두 바꿔야 함

# 방식 B: Auto (모델 이름만 주면 알아서)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# GPT-2로 바꾸려면 → 문자열만 "gpt2"로 변경
```

실무에선 **거의 항상 `AutoTokenizer`** 를 씁니다. `Auto` 계열은 모델 카드(`config.json`) 에서 어떤 클래스를 써야 할지 자동 추론하므로, 다른 모델로 갈아끼우는 실험이 매우 쉽습니다. Ch 14의 한국어 BERT(`klue/bert-base`) 도 같은 패턴으로 로드됩니다 — 코드 한 줄도 안 바꾸고요.""")

# ----- 34. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

다음 코드를 돌려보고 에러 메시지를 읽어보세요. 어떤 인자가 빠졌을까요?

```python
# 에러가 나는 코드
inputs_bad = tokenizer("I love HF!")        # return_tensors 빠짐
outputs_bad = model(**inputs_bad)
```

힌트: 모델은 PyTorch 텐서를 기대하는데, 토크나이저가 기본값으로 무엇을 반환할까요? `tokenizer("...")` 의 기본 반환 형식과 `tokenizer("...", return_tensors="pt")` 의 차이를 출력 비교해보세요.""")

# ----- 35. next -----
md(r"""## 다음 챕터 예고

**Chapter 8. Tokenizer 깊게 보기 + Datasets 라이브러리**

- 서브워드 토큰화 옵션 — `padding`, `truncation`, `max_length` 의 의미
- `datasets` 라이브러리: `load_dataset`, `map`, `filter`, Apache Arrow 메모리 효율
- DataLoader 변환 — Ch 9 학습 코드의 입력 준비 단계
- **여전히 학습 없음** (추론 데이터 파이프라인까지만)""")


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT.relative_to(REPO)}  ({len(cells)} cells)")
