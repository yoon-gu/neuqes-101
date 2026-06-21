> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/07_bert_pipeline/07_bert_pipeline.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q transformers
```

```python
import torch
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU (inference works in this chapter; skip the nvidia-smi cells)")
```

**▶ 실행 결과**

```text
PyTorch:        2.11.0+cu128
CUDA available: True
GPU:            Tesla T4
```

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:13:25 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   43C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

DistilBERT·BERT·GPT-2 세 모델의 토크나이저만 받아 한자리에 모읍니다. 가중치는 건드리지 않고 토크나이저 파일만 받으므로 가볍고 빠릅니다. 각 토크나이저의 어휘 크기와 실제 클래스 이름을 표로 출력해 모델마다 어떻게 다른지 비교할 수 있게 합니다.

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

DistilBERT·BERT는 어휘가 3만 안팎인데 GPT-2는 5만으로 더 큽니다. 어휘가 클수록 단어를 통째로 담을 여지가 커지는 대신 임베딩 테이블도 그만큼 무거워집니다.

이번엔 같은 문장 두 개를 세 토크나이저로 각각 쪼개 토큰 개수와 토큰 목록을 나란히 출력합니다. 어휘 크기만 비교하던 것에서 한 발 더 나아가, 실제로 단어가 어떻게 subword로 분해되는지 눈으로 확인하려는 셀입니다.

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

같은 문장도 토크나이저마다 다르게 쪼갭니다. uncased DistilBERT는 소문자로 낮춰 'hugging'을 통째로 두지만, cased BERT는 대소문자를 살리느라 'Hu'+'##gging'으로 나누고, GPT-2는 단어 앞 공백을 'Ġ'로 표시합니다. 어휘에 없는 단어를 subword로 분해하는 방식이 모델마다 갈린다는 걸 보여줍니다.

이번엔 세 토크나이저가 문장 앞뒤에 끼워 넣는 특수 토큰을 비교합니다. 각 토크나이저의 `[CLS]`·`[SEP]`·`[PAD]`·`[UNK]` 자리에 어떤 토큰이 들어가는지 표로 출력하고, `special_tokens_map` 전체도 함께 찍어봅니다. BERT 계열과 GPT-2가 특수 토큰을 다루는 방식이 어떻게 갈리는지 눈여겨보세요.

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

이번엔 모델의 `config` 를 펼쳐 핵심 설정값을 한눈에 정리합니다. 파라미터 수와 fp32 기준 예상 크기부터 `hidden_size`·`vocab_size`·`num_labels`·`problem_type` 까지 출력해, 이 모델이 어떤 구조이고 어떤 task로 파인튜닝됐는지 읽어냅니다. `num_labels=2` 와 `id2label` 이 감성 분류용 헤드를 가리킨다는 점을 눈여겨보세요.

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

Hugging Face의 `pipeline` 은 **"모델 다운로드 → 토큰화 → 추론 → 결과 후처리"** 를 한 줄로 묶어주는 함수입니다.

감성 분석(sentiment analysis)부터 시작합니다. **GPU가 있으면 `device=0` 으로 명시** — 그래야 모델이 VRAM에 올라가서 nvidia-smi 변화가 보입니다 (기본은 CPU).

```python
from transformers import pipeline

DEVICE = 0 if torch.cuda.is_available() else -1   # 0 = GPU index, -1 = CPU
classifier = pipeline("sentiment-analysis", device=DEVICE)
classifier("I love using Hugging Face! It's so simple.")
```

**▶ 실행 결과**

```text
[transformers] No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f.
Using a pipeline without specifying a model name and revision in production is not recommended.
[{'label': 'POSITIVE', 'score': 0.9998088479042053}]
```

**DistilBERT(SST-2)가 VRAM에 올라간 직후의 nvidia-smi:**

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:13:54 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   45C    P0             26W /   70W |     423MiB /  15360MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           13173      C   /usr/bin/python3                        420MiB |
+-----------------------------------------------------------------------------------------+
```

baseline과 비교하면 `Memory-Usage` 가 늘어났을 겁니다. **얼마나 늘어났을까?** — 모델 파라미터 수에서 거꾸로 추정해봅니다.

```python
def model_size_summary(model, dtype_bytes=4):
    # 파라미터 개수와 dtype 기준 예상 메모리 반환. fp32 = 4 bytes, fp16/bf16 = 2 bytes.
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = n_params * dtype_bytes / 1024**2
    return n_params, size_mb

n, size_mb = model_size_summary(classifier.model)
print(f"DistilBERT (SST-2) parameters:")
print(f"  count:           {n:>13,}  ({n/1e6:.1f} M)")
print(f"  fp32 size:       {size_mb:>10.1f} MB  (= params × 4 bytes)")
```

**▶ 실행 결과**

```text
DistilBERT (SST-2) parameters:
  count:              66,955,010  (67.0 M)
  fp32 size:            255.4 MB  (= params × 4 bytes)
```

**파라미터 수와 VRAM의 관계**

가중치 한 개는 *한 개의 부동소수* — fp32면 4 bytes, fp16/bf16이면 2 bytes를 차지합니다. 그래서:

$$\text{모델 가중치 크기} \approx \text{파라미터 수} \times \text{dtype bytes}$$

| dtype | bytes/param | DistilBERT(67M) 예상 크기 |
|---|---|---|
| **fp32** (기본 학습) | 4 | ~255 MB |
| **fp16 / bf16** (mixed precision) | 2 | ~128 MB |
| **int8** (양자화 추론) | 1 | ~64 MB |

**그런데 nvidia-smi는 파라미터 크기보다 *더 많이* 늘어나는데요?** 차이의 정체:

- **PyTorch CUDA 컨텍스트** (~150-300 MB): 라이브러리·드라이버가 GPU 점유 시 한 번 잡는 고정 비용. 첫 모델 로드에서만 보이고 이후엔 누적되지 않음.
- **CUDA 캐시 할당자** (수십 MB): PyTorch가 자주 쓰일 텐서를 캐싱.
- **추론 중 일시 activation**: forward pass에서 layer 사이 중간 결과. 추론은 작지만 학습은 큼.

**학습이 되면 메모리는 더 커집니다** — Adam 옵티마이저는 모델당 *추가로 2배* (1차·2차 모멘텀)를 더 들고, gradient도 *모델 크기만큼* 한 벌 — 즉 학습 중엔 **fp32 기준 파라미터 × 4배 정도** 의 VRAM이 필요합니다. Ch 9에서 다시 다룹니다.

**참고**: 처음 실행 시 모델 다운로드(약 250MB)에 30초-1분 정도 걸립니다. 두 번째부터는 캐시되어 즉시 실행.

여러 문장도 한 번에:

```python
results = classifier([
    "This movie was fantastic.",
    "Worst experience ever.",
    "It was okay, nothing special.",
])
for r in results:
    print(r)
```

**▶ 실행 결과**

```text
{'label': 'POSITIVE', 'score': 0.9998798370361328}
{'label': 'NEGATIVE', 'score': 0.9997876286506653}
{'label': 'NEGATIVE', 'score': 0.9820851683616638}
```

**결과 해석**

긍정·부정이 뚜렷한 앞 두 문장은 0.999로 확신하지만, "okay, nothing special"은 부정으로 기울되 0.98로 상대적으로 덜 확신합니다. score가 문장의 감정 강도를 그대로 반영합니다.

### 다른 task도 같은 패턴

`pipeline` 의 첫 인자만 바꾸면 다른 NLP 작업을 즉시 할 수 있습니다.

```python
# 텍스트 생성 (GPT-2)
generator = pipeline("text-generation", model="gpt2", device=DEVICE)
generator("Hugging Face is", max_length=30, num_return_sequences=1)
```

**▶ 실행 결과**

```text
[{'generated_text': "Hugging Face is the only new movie ever made about the murder of a girl in India. The film follows an innocent girl, Ja …(593 more chars omitted)
```

이번엔 `fill-mask` task로 문장 속 `[MASK]` 자리에 들어갈 단어를 BERT가 예측하게 합니다. GPT-2가 이어 쓰기를 했다면, BERT는 앞뒤 문맥을 모두 보고 빈칸을 채우는 방식이라는 점이 대비됩니다. 후보 단어와 각 확률(score)이 함께 출력되니 모델이 무엇을 떠올렸는지 살펴보세요.

```python
# 마스크 채우기 (BERT)
unmasker = pipeline("fill-mask", model="bert-base-uncased", device=DEVICE)
unmasker("Hugging Face is a [MASK] for NLP.")
```

**▶ 실행 결과**

```text
[transformers] BertForMaskedLM LOAD REPORT from: bert-base-uncased
Key                         | Status     |  | 
----------------------------+------------+--+-
cls.seq_relationship.weight | UNEXPECTED |  | 
bert.pooler.dense.weight    | UNEXPECTED |  | 
cls.seq_relationship.bias   | UNEXPECTED |  | 
bert.pooler.dense.bias      | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
[{'score': 0.04617200419306755,
  'token': 10675,
  'token_str': 'synonym',
  'sequence': 'hugging face is a synonym for nlp.'},
 {'score': 0.03509846702218056,
  'token': 4431,
  'token_str': 'reference',
  'sequence': 'hugging face is a reference for nlp.'},
 {'score': 0.02785254456102848,
  'token': 2944,
  'token_str': 'model',
  'sequence': 'hugging face is a model for nlp.'},
 {'score': 0.025830449536442757,
  'token': 19240,
  'token_str': 'metaphor',
  'sequence': 'hugging face is a metaphor for nlp.'},
 {'score': 0.02485215663909912,
  'token': 6994,
  'token_str': 'tool',
  'sequence': 'hugging face is a tool for nlp.'}]
```

**3개 pipeline(DistilBERT + GPT-2 + BERT-base)이 모두 VRAM에 쌓인 상태:**

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Wed Jun 17 21:14:16 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   48C    P0             35W /   70W |    1401MiB /  15360MiB |      9%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           13173      C   /usr/bin/python3                       1398MiB |
+-----------------------------------------------------------------------------------------+
```

**파라미터 수 합계로 예측한 모델 가중치 vs 실제 VRAM 증가** — 차이가 PyTorch 오버헤드입니다.

```python
models_info = {
    "DistilBERT (SST-2)":       classifier.model,
    "GPT-2":                    generator.model,
    "BERT-base-uncased":        unmasker.model,
}

print(f"{'model':>30}  {'params':>12}  {'fp32 size':>14}")
print("-" * 65)
total_params, total_size = 0, 0.0
for name, m in models_info.items():
    n, sz = model_size_summary(m)
    total_params += n
    total_size += sz
    print(f"{name:>30}  {n/1e6:>9.1f} M  {sz:>11.1f} MB")
print("-" * 65)
print(f"{'total':>30}  {total_params/1e6:>9.1f} M  {total_size:>11.1f} MB")
print(f"{'in GB':>30}  {' '*12}  {total_size/1024:>11.2f} GB")
```

**▶ 실행 결과**

```text
                         model        params       fp32 size
-----------------------------------------------------------------
            DistilBERT (SST-2)       67.0 M        255.4 MB
                         GPT-2      124.4 M        474.7 MB
             BERT-base-uncased      109.5 M        417.8 MB
-----------------------------------------------------------------
                         total      300.9 M       1147.9 MB
                         in GB                       1.12 GB
```

**실제 nvidia-smi VRAM 사용량과 비교** — 위 합계(~수백 MB)에 **PyTorch CUDA 컨텍스트 + 캐시 할당자 ~ 200-400 MB** 를 더한 값이 nvidia-smi 의 `Memory-Usage` 와 비슷할 겁니다.

> 모델별 파라미터 차이가 흥미롭습니다.
> - **DistilBERT (~67M)**: BERT-base에서 layer를 절반으로 줄여 학습한 경량화 모델. 추론 속도 ~2배.
> - **GPT-2 small (~124M)**: 파라미터는 BERT-base보다 약간 많고, 어휘(50K)도 더 큼.
> - **BERT-base (~110M)**: BERT 표준 사이즈.

**메모리를 비우는 표준 패턴** — 더 이상 안 쓰는 모델은:

```python
import gc, torch
del generator, unmasker         # 파이썬 참조 제거 (refcount=0)
gc.collect()                    # 가비지 컬렉션
if torch.cuda.is_available():
    torch.cuda.empty_cache()    # CUDA 캐시 비우기 (예약된 캐시 반환)
```

T4 메모리(15.36 GB)는 작은 추론 모델 여러 개를 무리 없이 담지만, BERT-large(~340M, fp32 ~1.4 GB)나 학습 시 *옵티마이저 + gradient* 까지 올리면 빠르게 한도에 도달하니 항상 nvidia-smi로 잔여 VRAM 확인하는 습관이 좋습니다.
