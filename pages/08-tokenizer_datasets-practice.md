> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/08_tokenizer_datasets/08_tokenizer_datasets.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q transformers datasets
```

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
print("\nNo model weights loaded in this chapter; VRAM stays roughly flat.")
```

**▶ 실행 결과**

```text
PyTorch:        2.11.0+cu128
CUDA available: True
GPU:            Tesla T4

No model weights loaded in this chapter; VRAM stays roughly flat.
```

## `datasets` 로 Yelp 로드

`load_dataset("Yelp/yelp_review_full")` 한 줄로 Hugging Face Hub에서 65만 건 학습 데이터를 받아옵니다 (50K test). 처음 받으면 ~150MB 다운로드 + 디스크 캐시.

**주목할 점**: `datasets` 는 Apache Arrow 형식으로 디스크에 저장하고 메모리맵으로 접근합니다. 65만 건이 한꺼번에 RAM에 올라가는 게 아니라, 인덱싱하는 시점에만 디스크에서 필요한 부분을 읽어 옵니다. 그래서 데이터셋이 아무리 커도 RAM 사용량에는 거의 영향이 없습니다.

```python
ds = load_dataset("Yelp/yelp_review_full")
print(ds)
```

**▶ 실행 결과**

```text
DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 650000
    })
    test: Dataset({
        features: ['label', 'text'],
        num_rows: 50000
    })
})
```

로드한 `DatasetDict` 의 train split이 어떤 구조인지 들여다봅니다. 샘플 수, `features` 스키마(라벨이 `ClassLabel`, 텍스트가 `string`), 그리고 첫 샘플의 라벨·텍스트를 직접 출력해 데이터의 모습을 확인합니다.

```python
# train split의 첫 샘플 + features 확인
print(f"train samples: {len(ds['train']):,}")
print(f"test samples:  {len(ds['test']):,}")
print(f"\nfeatures: {ds['train'].features}")
print(f"\nFirst sample:")
print(f"  label: {ds['train'][0]['label']}  (0-4 = stars 1-5)")
print(f"  text:  {ds['train'][0]['text'][:200]}...")
```

**▶ 실행 결과**

```text
train samples: 650,000
test samples:  50,000

features: {'label': ClassLabel(names=['1 star', '2 star', '3 stars', '4 stars', '5 stars']), 'text': Value('string')}

First sample:
  label: 4  (0-4 = stars 1-5)
  text:  dr. goldberg offers everything i look for in a general practitioner.  he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-no...
```

65만 건 전체를 다룰 필요는 없으므로, `shuffle(seed=42)` 로 결정론적으로 섞은 뒤 `select(range(5000))` 로 앞 5,000건만 골라냅니다. Phase 0(Ch 1-6)에서 쓴 것과 동일한 subsample이라, 같은 데이터가 이번엔 토크나이저를 통과하는 모습을 보게 됩니다.

```python
# 5,000건만 subsample (Phase 0와 동일한 처리)
small = ds["train"].shuffle(seed=42).select(range(5000))
print(small)
print(f"\nfirst sample text: {small[0]['text'][:150]}...")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['label', 'text'],
    num_rows: 5000
})

first sample text: I stalk this truck.  I've been to industrial parks where I pretend to be a tech worker standing in line, strip mall parking lots, and of course the fa...
```
