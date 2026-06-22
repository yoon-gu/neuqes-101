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

65만 건 전체를 그대로 쓰면 무거우니, `shuffle`로 섞은 뒤 앞에서 5,000건만 잘라 표본을 만듭니다. `seed=42`를 고정해 누가 실행해도 같은 표본이 뽑히게 합니다. 이어서 pandas `DataFrame`으로 옮겨 앞 3건을 살펴봅니다.

```python
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()

print(f"Sample count: {len(df)}")
df.head(3)
```

**위 코드 읽기** `shuffle(seed=42).select(range(SAMPLE_SIZE))` 가 전체 65만 건을 섞은 뒤 앞 5,000 건만 잘라 가볍게 다룰 표본을 만듭니다. `seed=42` 를 고정해 누가 돌려도 같은 표본이 나오고, `to_pandas()` 로 옮겨 익숙한 형태로 들여다봅니다.

**▶ 실행 결과**

```text
Sample count: 5000
   label                                               text
0      4  I stalk this truck.  I've been to industrial p...
1      2  who really knows if this is good pho or not, i...
2      4  I LOVE Bloom Salon... all of their stylist are...
```

별점 분포가 한쪽으로 치우치진 않았는지 막대그래프로 확인합니다. 라벨이 0-4로 저장돼 있으니 보기 좋게 `1 star`-`5 star` 표기로 바꿔 그립니다. 표본이 원본의 균형을 잘 물려받았는지 눈여겨봅니다.

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

5개 별점이 약 960-1027건씩 고르게 섞였습니다. `shuffle(seed=42)`로 뽑은 무작위 표본이 원본의 균형을 거의 그대로 물려받아, 뒤 챕터에서 특정 별점에 치우치지 않은 채 회귀·분류를 실험할 수 있습니다.

리뷰가 보통 몇 단어 길이인지 분포를 봅니다. 공백 기준으로 단어 수를 센 뒤 `describe()`로 요약 통계를 뽑습니다. 평균과 중앙값, 최댓값의 차이를 눈여겨보면 길이가 한쪽으로 긴 분포인지 가늠할 수 있습니다.

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

리뷰 길이는 중앙값이 100단어인데 최댓값은 977단어입니다. 소수의 긴 리뷰가 평균(134단어)을 중앙값보다 위로 끌어올린, 오른쪽으로 긴 분포입니다. 뒤 챕터에서 BERT가 `max_length`로 입력을 자를 때 왜 일부 리뷰가 잘리는지를 이 분포가 미리 보여줍니다.
