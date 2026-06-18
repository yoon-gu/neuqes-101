이제 `pipeline` 이 감춰뒀던 단계를 **직접 한 줄씩 실행** 합니다. 이 부분을 이해하면 앞으로 모든 모델을 자유롭게 다룰 수 있습니다.

### Step 1: Tokenizer와 Model 직접 로드

`pipeline` 이 알아서 받아오던 토크나이저와 모델을 이번엔 직접 손에 쥡니다. `AutoTokenizer` 와 `AutoModelForSequenceClassification` 으로 같은 SST-2 DistilBERT를 불러오고, GPU가 있으면 모델을 직접 `cuda` 로 옮깁니다. 직접 로드는 기본이 CPU라 이 이동을 명시해야 한다는 점을 눈여겨보세요.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# GPU가 있으면 모델을 VRAM으로 이동 (직접 로드는 default가 CPU라 명시 필요)
if torch.cuda.is_available():
    model = model.to("cuda")

print("Loaded")
print(f"  tokenizer class: {type(tokenizer).__name__}")
print(f"  model class:     {type(model).__name__}")
print(f"  model device:    {next(model.parameters()).device}")
```

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

문장을 토큰 문자열로 쪼개본 뒤, 모델이 바로 먹을 수 있는 텐서로 변환합니다. `tokenize()` 는 토큰 목록만 돌려주고, `tokenizer(text, return_tensors="pt")` 는 `input_ids`·`attention_mask` 같은 입력 텐서까지 만들어줍니다. 두 호출이 무엇을 다르게 돌려주는지 비교해보세요.

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

방금 얻은 `input_ids` 의 정수들이 실제로 어떤 토큰인지 거꾸로 확인합니다. 각 ID를 `tokenizer.decode()` 로 풀어 ID와 토큰 문자열을 나란히 출력합니다. 맨 앞 `101`·맨 뒤 `102` 가 `[CLS]`·`[SEP]` 특수 토큰으로 되살아나는지 눈여겨보세요.

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

이제 입력 텐서를 모델에 통과시켜 로짓을 얻습니다. 입력을 모델과 같은 device로 옮긴 뒤 `torch.no_grad()` 안에서 forward를 돌려, 추론에 불필요한 gradient 계산을 끕니다. 출력 객체의 타입과 로짓의 shape `[1, 2]` 가 "배치 1개, 클래스 2개"를 뜻한다는 점을 확인하세요.

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

마지막으로 로짓을 사람이 읽을 수 있는 라벨과 확률로 바꿉니다. `softmax` 로 로짓을 확률 분포로 만든 뒤, `argmax` 로 가장 높은 클래스를 고르고 `id2label` 로 이름을 붙입니다. 이렇게 직접 푼 결과가 `pipeline` 한 줄과 같은 라벨·점수로 떨어지는지 확인해보세요.

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

지금까지 손으로 푼 4단계 결과를 `pipeline` 한 줄의 출력과 직접 나란히 찍어 비교합니다. 라벨과 점수가 같게 나오면, `pipeline` 이 내부에서 우리가 한 토큰화→forward→softmax→argmax를 그대로 감싸고 있었다는 뜻입니다.

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

손으로 푼 4단계 결과가 pipeline 한 줄과 라벨·점수까지 똑같습니다. pipeline은 토큰화→forward→softmax→argmax를 감싼 편의 함수일 뿐, 내부에서 우리가 직접 한 일과 같은 연산을 한다는 게 확인됩니다.
