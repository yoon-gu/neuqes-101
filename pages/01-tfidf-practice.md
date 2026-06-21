> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/01_tfidf/01_tfidf.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

```python
!pip install -q datasets scikit-learn pandas matplotlib
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

plt.rcParams["axes.unicode_minus"] = False
```

`yelp_review_full`은 Yelp 식당 리뷰 65만 건에 1-5점 별점이 달린 데이터셋입니다 (라벨은 0-4로 저장됨). 학습 흐름을 가볍게 유지하기 위해 **5,000건만 무작위 샘플링** 합니다.

```python
dataset = load_dataset("Yelp/yelp_review_full")
print(dataset)
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

학습 흐름을 가볍게 유지하려고 train에서 5,000건만 뽑아 pandas로 옮깁니다. `shuffle(seed=42)`로 순서를 섞은 뒤 앞에서부터 잘라 라벨이 한쪽에 몰리지 않게 하고, `seed=42` 덕분에 누가 돌려도 같은 표본이 나옵니다.

```python
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()

print(f"Sample count: {len(df)}")
df.head(3)
```

**▶ 실행 결과**

```text
Sample count: 5000
   label                                               text
0      4  I stalk this truck.  I've been to industrial p...
1      2  who really knows if this is good pho or not, i...
2      4  I LOVE Bloom Salon... all of their stylist are...
```

**결과 해석**

라벨은 0-4 정수로 저장돼 있어 별점 1-5에 그대로 대응하고, `text` 칸에는 리뷰 원문이 그대로 들어 있습니다.

별점별 리뷰 수를 세어 표본이 특정 별점에 치우치지 않았는지 확인합니다. 무작위 샘플이라 다섯 별점이 비슷하게 나뉘는지가 관전 포인트입니다.

```python
counts = df["label"].value_counts().sort_index()
labels = [f"{i+1} star" for i in counts.index]
plt.bar(labels, counts.values)
plt.title("Star rating distribution (sampled 5,000)")
plt.ylabel("Reviews")
plt.show()
print(counts)
```

**▶ 실행 결과**

![output](../assets/01-tfidf-out1.png)

```text
label
0    1017
1    1027
2     960
3    1021
4     975
Name: count, dtype: int64
```

**결과 해석**

다섯 별점이 모두 960-1,027건으로 거의 고르게 분포합니다. 무작위 샘플링이 원본의 균형을 잘 보존했고, 한쪽으로 쏠려 학습이 왜곡될 걱정은 없습니다.

리뷰가 단어 기준으로 얼마나 긴지 분포를 봅니다. 뒤에서 벡터 길이(어휘 크기)와 희소성을 이해할 때 이 문서 길이 감각이 배경이 됩니다.

```python
df["len_words"] = df["text"].str.split().str.len()
df[["len_words"]].describe()
```

**▶ 실행 결과**

```text
         len_words
count  5000.000000
mean    133.811400
std     119.787704
min       1.000000
25%      53.000000
50%     100.000000
75%     177.000000
max     977.000000
```

**결과 해석**

리뷰 길이는 중앙값 100단어, 평균 134단어로 짧은 글이 많고 일부가 길게 늘어진 분포입니다. 최소 1단어에서 최대 977단어까지 편차가 커서, 같은 행렬 안에서도 문서마다 채워지는 칸 수가 크게 달라집니다.
