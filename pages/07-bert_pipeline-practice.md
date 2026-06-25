> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/07_bert_pipeline/07_bert_pipeline.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

Colab에는 `transformers`가 보통 설치돼 있지만, 최신 버전을 보장하기 위해 한 번 설치합니다.

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

### `!nvidia-smi` — GPU 메모리(VRAM) 실시간 추적

이번 챕터부터 학습·추론 코드가 GPU에 모델을 올리기 시작합니다. **`!nvidia-smi`** 는 NVIDIA에서 제공하는 명령행 도구로, 현재 GPU의 VRAM 사용량·온도·전력을 한 번에 보여줍니다. Colab 셀에서 `!` 접두사로 호출 가능.

T4의 총 VRAM은 **약 15.36 GB** (= 15,360 MiB). 모델·옵티마이저·activation을 모두 이 안에 담아야 합니다 — Ch 9 이후 학습 chapter에서는 이 한도와 자주 부딪히게 되어요.

**baseline** — 아직 아무 모델도 안 올린 상태:

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

**무엇을 봐야 하나** — 출력 가운데 줄 `Memory-Usage` 칸:

```
| ... |  XXX MiB / 15360MiB | ...
        └─ used    └─ total
```

- 처음엔 ~3-200 MiB 정도. CUDA 컨텍스트가 잡혀 있는 만큼만.
- 모델을 GPU에 올릴 때마다 `used` 가 증가합니다.
- `Volatile GPU-Util` 은 *현재* GPU가 일하는 비율 — 학습 중에는 90~100% 가까이.

**Python으로도 확인 가능** (셀 내부에서 변수로 받고 싶을 때):

```python
if torch.cuda.is_available():
    used  = torch.cuda.memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU memory: {used:.0f} / {total:.0f} MiB")
```

> Tip: `!nvidia-smi` 는 *시스템 전체* VRAM을 보여주고, `torch.cuda.memory_allocated()` 는 *현재 PyTorch 프로세스* 의 할당량만 보여줍니다 — 후자는 캐시·예약 메모리는 빼고 실제 텐서가 점유한 양에 가깝습니다.

## 실습: 일단 돌려봅시다

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

**결과 해석**

모델 이름을 한 번도 지정하지 않았는데 `pipeline` 이 SST-2 DistilBERT를 기본값으로 골라 로드했습니다(경고 로그). 결과는 `POSITIVE` 라벨에 score 0.9998 — 토큰화·추론·후처리가 모두 이 한 줄 안에서 끝났습니다.

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

**참고**: 처음 실행 시 모델 다운로드(약 250MB)에 30초~1분 정도 걸립니다. 두 번째부터는 캐시되어 즉시 실행.

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

리스트를 넘기면 문장별 결과가 순서대로 돌아옵니다. 확신이 강한 두 문장은 score 0.999대지만, 모호한 "It was okay, nothing special." 은 `NEGATIVE` 0.982로 상대적으로 확신이 낮습니다 — score가 모델의 확신 정도를 그대로 반영합니다.

### 다른 task도 같은 패턴

`pipeline` 의 첫 인자만 바꾸면 다른 NLP 작업을 즉시 할 수 있습니다.

```python
# 텍스트 생성 (GPT-2)
generator = pipeline("text-generation", model="gpt2", device=DEVICE)
generator("Hugging Face is", max_length=30, num_return_sequences=1)
```

**▶ 실행 결과**

```text
[{'generated_text': "Hugging Face is the only new movie ever made about the murder of a girl in India. The film follows an innocent girl, Ja …(뒤 593자 생략)
```

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

**결과 해석**

`fill-mask` 는 `[MASK]` 자리에 들어갈 후보를 확률 순으로 돌려줍니다. 상위 후보 `synonym`/`reference`/`model`/`tool` 의 score가 모두 5% 미만으로 낮은 건, 빈칸에 들어갈 수 있는 단어가 그만큼 다양하기 때문입니다(감성 분류의 양자택일과 대조적). 첫 두 줄의 LOAD REPORT는 MaskedLM 헤드에 안 쓰이는 가중치(`pooler` 등)를 알리는 정보일 뿐 무시해도 됩니다.

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
