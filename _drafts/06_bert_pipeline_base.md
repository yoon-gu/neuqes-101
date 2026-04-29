# Chapter 1. Pipeline으로 시작하는 Hugging Face

**목표**: 5줄짜리 코드로 NLP 모델을 돌려보고, 그 안에서 무슨 일이 일어났는지 한 단계씩 풀어 헤칩니다.

**환경**: Google Colab T4 GPU (런타임 → 런타임 유형 변경 → T4 GPU)

**예상 소요 시간**: 약 10분 (GPU 학습 없음, 추론만)

---

## 학습 흐름

1. 🚀 **실습**: `pipeline` 한 줄로 감성 분석 돌려보기
2. 🔬 **해부**: `pipeline`이 내부에서 뭘 하는지 까보기
3. 🛠️ **변형**: 같은 작업을 직접 단계별로 재현해보기

---

## 0. 환경 준비

Colab에는 `transformers`가 이미 설치되어 있는 경우가 많지만, 최신 버전을 보장하기 위해 한 번 설치합니다.

```python
!pip install -q transformers
```

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 1. 🚀 실습: 일단 돌려봅시다

Hugging Face의 `pipeline`은 **"모델 다운로드 → 토큰화 → 추론 → 결과 후처리"**를 한 줄로 묶어주는 함수입니다.

감성 분석(sentiment analysis)부터 시작해보겠습니다.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I love using Hugging Face! It's so simple.")
```

**결과:**
```
[{'label': 'POSITIVE', 'score': 0.9998...}]
```

처음 실행하면 모델 다운로드(약 250MB)에 30초~1분 정도 걸립니다. 두 번째부터는 캐시되어 즉시 실행됩니다.

여러 문장도 한 번에 처리할 수 있습니다.

```python
results = classifier([
    "This movie was fantastic.",
    "Worst experience ever.",
    "It was okay, nothing special.",
])
for r in results:
    print(r)
```

### 다른 task도 같은 패턴입니다

`pipeline`의 첫 인자만 바꾸면 다른 NLP 작업을 즉시 수행할 수 있습니다.

```python
# 텍스트 생성 (GPT-2)
generator = pipeline("text-generation", model="gpt2")
generator("Hugging Face is", max_length=30, num_return_sequences=1)
```

```python
# 마스크 채우기 (BERT)
unmasker = pipeline("fill-mask", model="bert-base-uncased")
unmasker("Hugging Face is a [MASK] for NLP.")
```

---

## 2. 🔬 해부: pipeline 안에서는 뭐가 일어났을까?

`pipeline("sentiment-analysis")` 한 줄이 사실은 **3단계**로 구성되어 있습니다.

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

현재 `classifier` 객체가 어떤 모델/토크나이저를 사용하는지 확인해봅시다.

```python
print("모델:", classifier.model.config._name_or_path)
print("모델 클래스:", type(classifier.model).__name__)
print("토크나이저 클래스:", type(classifier.tokenizer).__name__)
print("라벨 매핑:", classifier.model.config.id2label)
```

기본 모델은 `distilbert-base-uncased-finetuned-sst-2-english`입니다.
- **DistilBERT**: BERT를 40% 작게 만든 경량화 모델
- **SST-2**: 영화 리뷰 감성 분류 데이터셋 (Stanford Sentiment Treebank)
- 즉, "BERT를 영화 리뷰 데이터로 파인튜닝한 모델"입니다.

---

## 3. 🛠️ 변형: pipeline 없이 직접 해보기

이제 `pipeline`이 감춰뒀던 3단계를 **직접 한 줄씩 실행**해봅니다. 이 부분을 이해하면 앞으로 모든 모델을 자유롭게 다루실 수 있습니다.

### Step 1: Tokenizer와 Model 직접 로드

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("✅ 로드 완료")
```

> **잠깐, `Auto`가 뭔가요?**
>
> `AutoTokenizer`, `AutoModel` 같은 `Auto` 계열 클래스는 모델 이름만 주면 **알아서 적합한 클래스를 골라주는 팩토리**입니다.
> - DistilBERT 모델 → `DistilBertTokenizer`, `DistilBertForSequenceClassification`
> - BERT 모델 → `BertTokenizer`, `BertForSequenceClassification`
> - GPT-2 모델 → `GPT2Tokenizer`, `GPT2LMHeadModel`
>
> 직접 `BertTokenizer.from_pretrained(...)`라고 써도 되지만, `AutoTokenizer`를 쓰면 모델만 바꿔도 코드가 그대로 동작합니다.

### Step 2: 텍스트 → 숫자 (Tokenization)

```python
text = "I love using Hugging Face!"

# 토큰화 결과 살펴보기
tokens = tokenizer.tokenize(text)
print("토큰들:", tokens)

# 모델 입력용 텐서 만들기
inputs = tokenizer(text, return_tensors="pt")
print("\n모델 입력:")
for key, value in inputs.items():
    print(f"  {key}: {value}")
```

**관찰 포인트:**
- `input_ids`: 각 토큰의 정수 ID. 맨 앞 `101`은 `[CLS]`, 맨 뒤 `102`는 `[SEP]` 특수 토큰입니다.
- `attention_mask`: `1`이면 "이 위치는 진짜 토큰", `0`이면 "패딩이니 무시하라"는 신호입니다.

```python
# input_ids를 다시 토큰으로 디코딩해서 확인
for token_id in inputs["input_ids"][0]:
    token = tokenizer.decode([token_id])
    print(f"  {token_id.item():>5}  →  {token!r}")
```

### Step 3: 숫자 → 로짓 (Model forward)

```python
import torch

# 추론할 때는 gradient 계산을 끄는 것이 메모리/속도에 좋습니다
with torch.no_grad():
    outputs = model(**inputs)

print("출력 객체:", type(outputs).__name__)
print("로짓 shape:", outputs.logits.shape)
print("로짓 값:", outputs.logits)
```

로짓(logits)은 모델이 뱉은 **정규화되지 않은 점수**입니다. shape `[1, 2]`는 "배치 1개, 클래스 2개(NEGATIVE, POSITIVE)"를 의미합니다.

### Step 4: 로짓 → 확률/라벨 (Post-processing)

```python
# softmax로 확률 변환
probs = torch.softmax(outputs.logits, dim=-1)
print("확률:", probs)

# 가장 높은 확률의 클래스 인덱스
predicted_class_id = probs.argmax(dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]
predicted_score = probs[0, predicted_class_id].item()

print(f"\n최종 결과: {{'label': '{predicted_label}', 'score': {predicted_score:.4f}}}")
```

🎉 `pipeline`이 한 줄로 해주던 일을 4단계로 직접 재현했습니다. 결과를 비교해봅시다.

```python
print("pipeline 결과:", classifier(text))
print(f"직접 구현 결과: [{{'label': '{predicted_label}', 'score': {predicted_score:.4f}}}]")
```

---

## 📦 이번 챕터에 등장한 라이브러리 정리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.pipeline` | 추론 원스톱 함수 | (3장) 학습된 모델을 `pipeline`으로 감싸 사용 |
| `transformers.AutoTokenizer` | 모델에 맞는 토크나이저 자동 로드 | (2장) 토큰화 옵션 깊게 보기 |
| `transformers.AutoModelForSequenceClassification` | 분류용 헤드가 붙은 모델 | (3장) 직접 파인튜닝 |
| `torch` | PyTorch — 텐서 연산과 학습의 기반 | 계속 사용 |

---

## 🎯 체크포인트 질문

다음 챕터로 넘어가기 전에 스스로 답해보세요.

1. `pipeline("sentiment-analysis")`을 직접 풀어 쓰면 어떤 단계가 되는지 4단계로 설명할 수 있나요?
2. `input_ids`와 `attention_mask`는 각각 무슨 역할인가요?
3. 모델 출력의 `logits`는 왜 그대로 쓰지 않고 `softmax`를 거치나요?
4. `AutoTokenizer`를 쓸 때와 `BertTokenizer`를 직접 쓸 때의 차이는 무엇인가요?

---

## 🚀 삽질 코너 (선택)

다음 코드를 돌려보고 에러 메시지를 읽어보세요. 어떤 인자가 빠졌을까요?

```python
# 에러가 나는 코드
inputs_bad = tokenizer("I love HF!")  # return_tensors 빠짐
outputs_bad = model(**inputs_bad)
```

> 힌트: 모델은 PyTorch 텐서를 기대하는데, 토크나이저가 기본값으로 무엇을 반환할까요?

---

## 다음 챕터 예고

**Chapter 2. Tokenizer 깊게 보기 + Datasets 라이브러리 입문**

- 서브워드 토큰화는 왜 필요한가? (`##ing`, `▁hello` 같은 기호의 정체)
- `padding`, `truncation`, `max_length` 옵션의 의미
- `datasets` 라이브러리로 IMDB 데이터셋 로드하기
- 다음 챕터에서 BERT-Tiny 파인튜닝 준비를 마칩니다 (T4에서 약 5분 학습 분량)
