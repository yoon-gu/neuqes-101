## 이번 챕터에 등장한 라이브러리·함수

### `transformers`

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.pipeline` | 추론 원스톱 함수 (3단계 묶음) | 학습된 모델을 `pipeline`으로 감싸 사용 (Ch 9 이후) |
| `transformers.AutoTokenizer` | 모델에 맞는 토크나이저 자동 로드 | Ch 8에서 옵션 깊게 보기 |
| `transformers.AutoModelForSequenceClassification` | 분류 헤드가 붙은 모델 자동 로드 | Ch 9부터 직접 파인튜닝 |

### `torch` 의 후처리·연산 함수

| 형태 | 호출 예시 | 언제 |
|---|---|---|
| `torch.softmax(x, dim=-1)` | 텐서에 직접 | 추론 후처리 (이번 챕터처럼) |
| `torch.nn.functional.softmax` (보통 `F.softmax`) | `F.softmax(x, dim=-1)` | 함수형, PyTorch 코드에 가장 흔함 |
| `torch.nn.Softmax(dim=-1)` | layer로 모델 안에 박을 때 | 잘 안 씀 (보통 logits 그대로 두고 loss가 처리) |
| `F.log_softmax`, `torch.log_softmax` | softmax + log 한 번에 (수치 안정) | 학습 loss 직접 구현 시 |
| `torch.argmax(x, dim=-1)` 또는 `x.argmax(dim=-1)` | 가장 큰 값의 인덱스 | 분류 예측 인덱스 추출 |
| `torch.no_grad()` (context manager) | 추론 중 gradient 비활성 | 메모리·속도 절약 |

> **요점**: HuggingFace는 softmax·argmax 같은 *수치 함수* 를 따로 제공하지 않습니다. 모두 PyTorch에서 직접 호출합니다 (TensorFlow 백엔드를 쓰면 `tf.nn.softmax` 같은 식). `Trainer` 가 학습 loss 안에서 softmax를 자동 처리하므로, 학습 코드에서는 직접 부를 일이 거의 없고 *추론 후처리* 에서 등장하는 게 보통.

### `model.config` 의 자주 쓰는 속성

| 속성 | 용도 |
|---|---|
| `id2label`, `label2id` | 클래스 인덱스 ↔ 이름 매핑 |
| `num_labels` | 분류 헤드 출력 차원 |
| `hidden_size`, `vocab_size`, `max_position_embeddings` | 모델 구조 파라미터 |
| `model_type`, `_name_or_path` | 모델 정체성 |
| `problem_type` | `Trainer` 자동 loss 선택 (`regression` / `single_label_classification` / `multi_label_classification`) |

### `torch` 자체

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `torch` | PyTorch — 텐서 연산, 자동 미분, GPU 연결 | 계속 사용 (특히 Ch 9 학습부터) |

## 체크포인트 질문

1. `pipeline("sentiment-analysis")` 을 직접 풀어 쓰면 어떤 단계가 되는지 4단계로 설명할 수 있나요?
2. `input_ids` 와 `attention_mask` 는 각각 무슨 역할인가요?
3. 모델 출력의 `logits` 는 왜 그대로 쓰지 않고 `softmax` 를 거치나요? (Ch 4에서 본 동등성 떠올리기)
4. `AutoTokenizer.from_pretrained(...)` 를 쓸 때와 `BertTokenizer.from_pretrained(...)` 를 직접 쓸 때의 차이는 무엇인가요?

## FAQ

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
# ['anti', '##dis', '##est', '##ab', '##lish', '##ment', '##arian', '##ism']
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

실무에선 **거의 항상 `AutoTokenizer`** 를 씁니다. `Auto` 계열은 모델 카드(`config.json`) 에서 어떤 클래스를 써야 할지 자동 추론하므로, 다른 모델로 갈아끼우는 실험이 매우 쉽습니다. Ch 14의 한국어 BERT(`klue/bert-base`) 도 같은 패턴으로 로드됩니다 — 코드 한 줄도 안 바꾸고요.

## 삽질 코너 (선택)

다음 코드를 돌려보고 에러 메시지를 읽어보세요. 어떤 인자가 빠졌을까요?

```python
# 에러가 나는 코드
inputs_bad = tokenizer("I love HF!")        # return_tensors 빠짐
outputs_bad = model(**inputs_bad)
```

힌트: 모델은 PyTorch 텐서를 기대하는데, 토크나이저가 기본값으로 무엇을 반환할까요? `tokenizer("...")` 의 기본 반환 형식과 `tokenizer("...", return_tensors="pt")` 의 차이를 출력 비교해보세요.

## 다음 챕터 예고

**Chapter 8. Tokenizer 깊게 보기 + Datasets 라이브러리**

- 서브워드 토큰화 옵션 — `padding`, `truncation`, `max_length` 의 의미
- `datasets` 라이브러리: `load_dataset`, `map`, `filter`, Apache Arrow 메모리 효율
- DataLoader 변환 — Ch 9 학습 코드의 입력 준비 단계
- **여전히 학습 없음** (추론 데이터 파이프라인까지만)
