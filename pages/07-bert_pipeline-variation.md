이제 `pipeline` 이 감춰뒀던 단계를 **직접 한 줄씩 실행** 합니다. 이 부분을 이해하면 앞으로 모든 모델을 자유롭게 다룰 수 있습니다.

### Step 1: Tokenizer와 Model 직접 로드

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

**위 코드 읽기** — `pipeline` 이 내부에서 알아서 처리하던 토크나이저와 모델을 이제 따로 로드합니다. `AutoTokenizer` 와 `AutoModelForSequenceClassification` 은 모델 이름만 주면 거기에 맞는 클래스(여기선 `BertTokenizer`, `DistilBertForSequenceClassification`)를 자동으로 골라주는 팩토리입니다.

```python
# GPU가 있으면 모델을 VRAM으로 이동 (직접 로드는 default가 CPU라 명시 필요)
if torch.cuda.is_available():
    model = model.to("cuda")

print("Loaded")
print(f"  tokenizer class: {type(tokenizer).__name__}")
print(f"  model class:     {type(model).__name__}")
print(f"  model device:    {next(model.parameters()).device}")
```

**위 코드 읽기** — `pipeline` 과 달리 직접 로드한 모델의 기본 위치는 CPU이므로, `.to("cuda")` 로 명시해야 VRAM에 올라갑니다. `next(model.parameters()).device` 로 실제로 어느 장치에 있는지 확인할 수 있습니다.

**▶ 실행 결과**

```text
Loaded
  tokenizer class: BertTokenizer
  model class:     DistilBertForSequenceClassification
  model device:    cuda:0
```

**`pipeline` 위에 추가로 *같은 DistilBERT* 가 올라간 상태** — VRAM이 또 한 번 늘어났습니다. 같은 가중치라도 별도 객체이면 별도 메모리.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:14:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   49C    P0             26W /   70W |    1671MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           13173      C   /usr/bin/python3                       1668MiB |
+-----------------------------------------------------------------------------------------+
```

> **잠깐, `Auto`가 뭔가요?**
>
> `AutoTokenizer`, `AutoModel...` 같은 `Auto` 계열 클래스는 모델 이름만 주면 **알아서 적합한 클래스를 골라주는 팩토리** 입니다.
>
> - DistilBERT 모델 → `DistilBertTokenizer`, `DistilBertForSequenceClassification`
> - BERT 모델 → `BertTokenizer`, `BertForSequenceClassification`
> - GPT-2 모델 → `GPT2Tokenizer`, `GPT2LMHeadModel`
>
> 직접 `BertTokenizer.from_pretrained(...)`라고 써도 되지만, `AutoTokenizer` 를 쓰면 모델만 바꿔도 코드가 그대로 동작합니다.

### Step 2: 텍스트 → 숫자 (Tokenization)

```python
text = "I love using Hugging Face!"

# 토큰화 결과 살펴보기
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# 모델 입력용 텐서 만들기
inputs = tokenizer(text, return_tensors="pt")
print("\nModel inputs:")
for key, value in inputs.items():
    print(f"  {key}: {value}")
```

**▶ 실행 결과**

```text
Tokens: ['i', 'love', 'using', 'hugging', 'face', '!']

Model inputs:
  input_ids: tensor([[  101,  1045,  2293,  2478, 17662,  2227,   999,   102]])
  token_type_ids: tensor([[0, 0, 0, 0, 0, 0, 0, 0]])
  attention_mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
```

**관찰 포인트**
- `input_ids`: 각 토큰의 정수 ID. 맨 앞 `101`은 `[CLS]`, 맨 뒤 `102`는 `[SEP]` 특수 토큰.
- `attention_mask`: `1`이면 "이 위치는 진짜 토큰", `0`이면 "패딩이니 무시하라"는 신호 (이번엔 패딩 없음 → 모두 1).
- `tokenize()` 와 `tokenizer()` 의 차이 — 전자는 토큰 문자열만 반환, 후자는 모델 입력용 텐서까지 다 만들어줌.

```python
# input_ids를 다시 토큰으로 디코딩해서 확인
print(f"{'ID':>5}    token")
print("-" * 30)
for token_id in inputs["input_ids"][0]:
    token = tokenizer.decode([token_id])
    print(f"{token_id.item():>5}    {token!r}")
```

**▶ 실행 결과**

```text
   ID    token
------------------------------
  101    '[CLS]'
 1045    'i'
 2293    'love'
 2478    'using'
17662    'hugging'
 2227    'face'
  999    '!'
  102    '[SEP]'
```

### Step 3: 숫자 → 로짓 (Model forward)

```python
# 입력 텐서도 모델과 같은 device로 이동시켜야 함 (CPU↔GPU 혼합 forward는 에러)
inputs_on_device = {k: v.to(model.device) for k, v in inputs.items()}

# 추론할 때는 gradient 계산을 끄는 것이 메모리/속도에 좋음
with torch.no_grad():
    outputs = model(**inputs_on_device)

print(f"Output object: {type(outputs).__name__}")
print(f"Logits shape:  {outputs.logits.shape}  (batch=1, classes=2: NEGATIVE, POSITIVE)")
print(f"Logits:        {outputs.logits}")
print(f"Logits device: {outputs.logits.device}")
```

**▶ 실행 결과**

```text
Output object: SequenceClassifierOutput
Logits shape:  torch.Size([1, 2])  (batch=1, classes=2: NEGATIVE, POSITIVE)
Logits:        tensor([[-3.9266,  4.2142]], device='cuda:0')
Logits device: cuda:0
```

로짓(logits)은 모델이 뱉은 **정규화되지 않은 점수**. shape `[1, 2]`는 "배치 1개, 클래스 2개"를 의미합니다.

여기서 잠깐 — 익숙하지 않나요? Ch 4의 *softmax + 2차원 head* 구조와 정확히 같습니다. BERT는 사전학습된 *심층* 모델일 뿐, 마지막 분류 헤드는 sklearn에서 본 것과 본질이 같습니다.

### Step 4: 로짓 → 확률/라벨 (Post-processing)

```python
# softmax로 확률 변환
probs = torch.softmax(outputs.logits, dim=-1)
print(f"Probabilities: {probs}")

# 가장 높은 확률의 클래스 인덱스
predicted_class_id = probs.argmax(dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]
predicted_score = probs[0, predicted_class_id].item()

print(f"\nFinal result: {{'label': '{predicted_label}', 'score': {predicted_score:.4f}}}")
```

**▶ 실행 결과**

```text
Probabilities: tensor([[2.9132e-04, 9.9971e-01]], device='cuda:0')

Final result: {'label': 'POSITIVE', 'score': 0.9997}
```

**잠깐 — `transformers` 안에는 softmax 함수가 없나요?**

없습니다. Hugging Face는 *모델·토크나이저·학습 루프* 를 제공하고, **수치 연산은 PyTorch에 위임**합니다 (혹은 TensorFlow / JAX). 그래서 후처리(softmax, argmax, log 등)는 `torch.*` 를 직접 부릅니다.

PyTorch 안에는 같은 softmax를 표현하는 세 가지 형태가 있습니다 — 결과는 모두 동일하고 *어디서 호출하느냐* 만 다릅니다.

```python
import torch.nn.functional as F

# 형태 1: torch.softmax   ← 이번 챕터에서 쓴 형태 (텐서 메서드 스타일)
p1 = torch.softmax(outputs.logits, dim=-1)

# 형태 2: F.softmax        ← functional 네임스페이스 (가장 흔하게 보이는 PyTorch 패턴)
p2 = F.softmax(outputs.logits, dim=-1)

# 형태 3: nn.Softmax       ← 모듈 형태 (모델 내부에 layer로 박을 때 사용)
softmax_module = torch.nn.Softmax(dim=-1)
p3 = softmax_module(outputs.logits)

print(f"Do all three forms agree?")
print(f"  torch.softmax vs F.softmax:    {torch.allclose(p1, p2)}")
print(f"  torch.softmax vs nn.Softmax:   {torch.allclose(p1, p3)}")
print(f"\n  values: {p1}")
```

**▶ 실행 결과**

```text
Do all three forms agree?
  torch.softmax vs F.softmax:    True
  torch.softmax vs nn.Softmax:   True

  values: tensor([[2.9132e-04, 9.9971e-01]], device='cuda:0')
```

**보너스 — 학습 코드에서 자주 보이는 `log_softmax`**

수치적 안정성 때문에 학습 시에는 `softmax → log` 두 단계 대신 **`log_softmax` 한 번** 으로 묶는 게 표준입니다 (`CrossEntropyLoss` 가 내부에서 이렇게 함).

```python
# softmax 후 log를 따로 (수치 불안정 가능)
log_probs_unstable = torch.log(torch.softmax(logits, dim=-1))

# log_softmax 한 번 (안정적)
log_probs_stable = F.log_softmax(logits, dim=-1)
```

추론 시에는 그냥 `softmax` 가 깔끔합니다 — 확률이 직접 필요하니까요. **학습 시** `CrossEntropyLoss(logits, target)` 는 내부적으로 logit에 `log_softmax` 를 적용하고 NLL을 더하므로, *softmax를 직접 부를 일이 없습니다* (Ch 9 이후 자주 등장).

```python
# pipeline이 한 줄로 해주던 일을 4단계로 직접 재현했습니다. 결과를 비교해봅시다.
print(f"pipeline result:    {classifier(text)}")
print(f"manual 4-step:      [{{'label': '{predicted_label}', 'score': {predicted_score:.4f}}}]")
```

**▶ 실행 결과**

```text
pipeline result:    [{'label': 'POSITIVE', 'score': 0.9997085928916931}]
manual 4-step:      [{'label': 'POSITIVE', 'score': 0.9997}]
```

**결과 해석**

손으로 펼친 4단계(토큰화 → forward → softmax → argmax)의 결과가 `pipeline` 한 줄과 라벨·score까지 일치합니다. `pipeline` 은 마법이 아니라 이 단계들을 묶어둔 wrapper일 뿐임이 수치로 확인됩니다.

## 보너스: 토크나이저마다 어휘가 다르다

지금까지는 DistilBERT의 WordPiece 토크나이저 *하나* 만 봤습니다. 그런데 모델이 바뀌면 토크나이저도 바뀌고, **같은 문장이 완전히 다른 토큰 리스트로 쪼개집니다** — 어휘 사전이 사전학습 단계에서 따로 만들어졌기 때문이에요.

세 가지 대표 토크나이저를 나란히 비교합니다.

| 모델 | 알고리즘 | 어휘 크기 | 대소문자 |
|---|---|---|---|
| `distilbert-base-uncased` | **WordPiece** | 30,522 | 모두 소문자로 |
| `bert-base-cased` | **WordPiece** | 28,996 | 대소문자 유지 |
| `gpt2` | **BPE** (Byte Pair Encoding) | 50,257 | 대소문자 유지 |

WordPiece와 BPE는 둘 다 *서브워드 알고리즘* 이지만 학습·표기 방식이 달라서 토큰 모양이 시각적으로도 구분됩니다 — `##` 접두사 vs `Ġ` (공백) 접두사.

먼저 토크나이저만 비교해 봅니다. 모델 가중치 없이 토크나이저 파일만 받으면 되므로 가볍고, 같은 문장이 모델마다 어떻게 쪼개지는지 한눈에 볼 수 있습니다. DistilBERT/BERT(WordPiece)와 GPT-2(BPE)를 나란히 둡니다.

```python
# 토크나이저 3종 로드 (모델 가중치는 안 받고 토크나이저 파일만 ~수백 KB)
tokenizer_specs = {
    "distilbert-base-uncased": AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    "bert-base-cased":         AutoTokenizer.from_pretrained("bert-base-cased"),
    "gpt2":                    AutoTokenizer.from_pretrained("gpt2"),
}

print(f"{'model':>28}  {'vocab_size':>10}  {'class':>32}")
print("-" * 76)
for name, tok in tokenizer_specs.items():
    print(f"{name:>28}  {tok.vocab_size:>10,}  {type(tok).__name__:>32}")
```

**▶ 실행 결과**

```text
                       model  vocab_size                             class
----------------------------------------------------------------------------
     distilbert-base-uncased      30,522                     BertTokenizer
             bert-base-cased      28,996                     BertTokenizer
                        gpt2      50,257                     GPT2Tokenizer
```

**결과 해석**

어휘 크기가 모델마다 다릅니다 — DistilBERT 30,522, BERT-cased 28,996, GPT-2 50,257. 토크나이저가 다르면 같은 텍스트라도 토큰 ID가 전혀 달라지므로, 모델과 토크나이저는 항상 짝으로 로드해야 합니다.

같은 두 문장을 세 토크나이저로 각각 쪼개 토큰 개수와 조각을 비교합니다.

```python
sample_sentences = [
    "I love using Hugging Face!",
    "Tokenization is fascinating.",
]

for sent in sample_sentences:
    print(f"Input: {sent!r}")
    for name, tok in tokenizer_specs.items():
        tokens = tok.tokenize(sent)
        print(f"  {name:>28}  ({len(tokens)} tokens) {tokens}")
    print()
```

**▶ 실행 결과**

```text
Input: 'I love using Hugging Face!'
       distilbert-base-uncased  (6 tokens) ['i', 'love', 'using', 'hugging', 'face', '!']
               bert-base-cased  (7 tokens) ['I', 'love', 'using', 'Hu', '##gging', 'Face', '!']
                          gpt2  (7 tokens) ['I', 'Ġlove', 'Ġusing', 'ĠHug', 'ging', 'ĠFace', '!']

Input: 'Tokenization is fascinating.'
       distilbert-base-uncased  (5 tokens) ['token', '##ization', 'is', 'fascinating', '.']
               bert-base-cased  (6 tokens) ['To', '##ken', '##ization', 'is', 'fascinating', '.']
                          gpt2  (5 tokens) ['Token', 'ization', 'Ġis', 'Ġfascinating', '.']
```

**결과 해석**

같은 문장이 토크나이저마다 다르게 쪼개집니다. `uncased` 인 DistilBERT는 대소문자를 무시해 `i`/`face` 로, `cased` 인 BERT는 `Hu`+`##gging` 처럼 WordPiece 서브워드로, GPT-2는 `Ġ`(앞 공백) 표시를 붙인 BPE 조각으로 나눕니다. 토큰 개수까지 달라진다는 점에 주목하세요.

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

다음은 각 모델이 `[CLS]`/`[SEP]`/`[PAD]`/`[UNK]` 자리에 어떤 특수 토큰을 두는지 한 표로 모읍니다.

```python
# 특수 토큰: 모델마다 어떤 token을 [CLS]/[SEP]/[PAD]/[UNK] 자리에 두는지
print(f"{'model':>28}  {'BOS/CLS':>16}  {'EOS/SEP':>16}  {'PAD':>10}  {'UNK':>10}")
print("-" * 90)
for name, tok in tokenizer_specs.items():
    cls = tok.cls_token if tok.cls_token else (tok.bos_token or "—")
    sep = tok.sep_token if tok.sep_token else (tok.eos_token or "—")
    pad = tok.pad_token or "—"
    unk = tok.unk_token or "—"
    print(f"{name:>28}  {cls:>16}  {sep:>16}  {pad:>10}  {unk:>10}")

# 모든 special token을 한 번에 보고 싶으면:
print()
for name, tok in tokenizer_specs.items():
    print(f"{name}.special_tokens_map = {tok.special_tokens_map}")
```

**▶ 실행 결과**

```text
                       model           BOS/CLS           EOS/SEP         PAD         UNK
------------------------------------------------------------------------------------------
     distilbert-base-uncased             [CLS]             [SEP]       [PAD]       [UNK]
             bert-base-cased             [CLS]             [SEP]       [PAD]       [UNK]
                        gpt2     <|endoftext|>     <|endoftext|>           —  <|endoftext|>

distilbert-base-uncased.special_tokens_map = {'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}
bert-base-cased.special_tokens_map = {'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}
gpt2.special_tokens_map = {'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}
```

**결과 해석**

BERT 계열은 `[CLS]`/`[SEP]`/`[PAD]`/`[UNK]` 4종을 명확히 구분합니다. 반면 GPT-2는 BOS/EOS/UNK 자리를 모두 `<|endoftext|>` 하나로 쓰고 PAD가 없습니다 — 분류용 인코더(BERT)와 생성용 디코더(GPT-2)의 설계 차이가 특수 토큰에서 드러납니다.

**관찰 포인트**

- **`##` 접두사 (WordPiece)**: DistilBERT·BERT는 단어 중간 서브워드를 `##xxx` 로 표시. 예: `tokenization → ['token', '##ization']`. 이전 토큰의 *연속* 이라는 신호.
- **`Ġ` 접두사 (BPE)**: GPT-2는 토큰 앞에 공백이 있었는지를 `Ġ` (Latin small letter G with stroke) 로 표시. 예: `Ġlove` 는 "love 앞에 공백이 있었다". 토큰화/디코딩이 정확히 역연산이 되도록 하는 표기.
- **대소문자**: `bert-base-cased` 는 `Hugging Face` 의 `H`, `F` 를 그대로 보존. `distilbert-base-uncased` 는 모두 소문자. 이름·고유명사 처리에서 차이가 큽니다.
- **vocab 크기**: GPT-2가 50K 로 가장 큼. BPE는 영어 외 다양한 토큰(드문 조합·바이트 단위)도 어휘에 포함하기 때문. WordPiece는 영어 중심이라 30K로 충분.
- **특수 토큰**: BERT 계열은 `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` 가 모두 정의되지만 GPT-2는 `[CLS]/[SEP]` 가 없습니다 (autoregressive 모델은 문장 시작/끝 마커를 따로 안 둠 — `<|endoftext|>` 하나가 BOS/EOS 역할). PAD도 없어 추가 설정이 필요한 경우가 흔함.

**왜 같은 문장이 다른 토큰 시퀀스가 되나?** 어휘 사전이 *사전학습 데이터* 에 따라 만들어집니다.

- BERT는 BookCorpus + Wikipedia로 학습됐고, 영어 중심 어휘.
- GPT-2는 더 다양한 웹 텍스트(Reddit 등)로 학습됐고 BPE라 어휘가 더 풍부.
- 한국어 BERT(`klue/bert-base`, Ch 14)는 한국어 코퍼스로 다시 학습돼 한국어 어휘를 보유 — 같은 문장 `"안녕"` 도 영어 BERT면 `[UNK]` 또는 글자 단위로 쪼개지지만 한국어 BERT엔 한 토큰으로 들어갑니다.

**실무 함의**: 모델을 갈아 끼울 때 토크나이저도 *반드시 짝* 으로 바꿔야 합니다. `AutoTokenizer.from_pretrained(model_name)` 의 model_name 이 모델 자체와 일치해야 하는 이유 — 학습 때 본 어휘와 추론 때 입력 어휘가 같아야 모델이 의미를 이해합니다.

## 보너스: `model.config` 안에 뭐가 있나

위에서 `model.config.id2label` 로 라벨 이름을 알아냈습니다. `config` 객체에는 모델의 *설계도* 가 모두 들어있어서, 모델을 받아왔을 때 가장 먼저 들여다보면 좋은 곳입니다.

분류 작업에서 자주 쓰는 속성들을 한 번에 출력합니다.

이번엔 분류 모델의 `config` 를 펼쳐, 파라미터 수·은닉 차원·라벨 매핑 같은 모델 정체성을 확인합니다.

```python
cfg = model.config
n_params, size_mb = model_size_summary(model)   # 앞에서 정의한 헬퍼 재사용

print(f"Model name/path:          {cfg._name_or_path}")
print(f"Model type:               {cfg.model_type}")
print(f"Parameters:               {n_params:,}  ({n_params/1e6:.1f} M)")
print(f"fp32 size:                {size_mb:.1f} MB  (= params x 4 bytes)")
print(f"hidden_size:             {cfg.hidden_size}        (BERT-base/DistilBERT: 768)")
print(f"vocab_size:              {cfg.vocab_size:,}     (matches tokenizer vocab)")
print(f"max_position_embeddings: {cfg.max_position_embeddings}  (input length cap)")
print(f"num_labels:              {cfg.num_labels}          (classification head dim)")
print(f"id2label:                {cfg.id2label}")
print(f"label2id:                {cfg.label2id}")
print(f"problem_type:            {cfg.problem_type!r}    (None → auto-inferred from num_labels)")
```

**▶ 실행 결과**

```text
Model name/path:          distilbert-base-uncased-finetuned-sst-2-english
Model type:               distilbert
Parameters:               66,955,010  (67.0 M)
fp32 size:                255.4 MB  (= params x 4 bytes)
hidden_size:             768        (BERT-base/DistilBERT: 768)
vocab_size:              30,522     (matches tokenizer vocab)
max_position_embeddings: 512  (input length cap)
num_labels:              2          (classification head dim)
id2label:                {0: 'NEGATIVE', 1: 'POSITIVE'}
label2id:                {'NEGATIVE': 0, 'POSITIVE': 1}
problem_type:            None    (None → auto-inferred from num_labels)
```

**결과 해석**

이 모델은 SST-2 감성 분류용이라 `num_labels=2`, `id2label={0: 'NEGATIVE', 1: 'POSITIVE'}` 로 분류 헤드가 2차원입니다. `vocab_size`(30,522)가 앞서 본 DistilBERT 토크나이저 어휘와 정확히 일치하는 점, `problem_type` 이 `None` 이라 `num_labels` 로부터 자동 추론된다는 점을 확인하세요.

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
