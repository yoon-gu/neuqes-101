`CountVectorizer`의 한계: `"the"`, `"and"` 같은 단어가 모든 리뷰에 많이 등장하니 횟수만으로는 문서 사이 차이를 잘 드러내지 못합니다.

**TF-IDF**는 두 항을 곱해 이 문제를 다룹니다.

$$\text{tfidf}(t, d) = \underbrace{\text{tf}(t, d)}_{\text{문서 } d \text{에서 } t \text{의 빈도}} \cdot \underbrace{\log\frac{1 + N}{1 + \text{df}(t)}}_{\text{희귀도 가중치 (IDF)}}$$

- `tf`: 한 문서에서 단어가 얼마나 자주 나왔는가
- `idf`: 그 단어가 **얼마나 적은 문서에 등장했는가** (모든 문서에 흔할수록 0에 가까워짐)

직관: "이 단어가 이 문서에서 자주 나오면서 동시에 다른 문서엔 흔하지 않다면, 이 문서를 특징짓는 단어"라는 가중치입니다.

```python
tfidf = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf.fit_transform(df["text"])
print(f"shape: {X_tfidf.shape}")
```

**▶ 실행 결과**

```text
shape: (5000, 10000)
```

한 리뷰(`doc_id=0`)를 골라, 같은 단어의 단순 횟수와 TF-IDF 값을 나란히 비교합니다. TF-IDF 기준으로 상위 10개를 뽑아, 횟수 정렬과 무엇이 달라지는지 한 문서 안에서 직접 확인하는 부분입니다.

```python
doc_id = 0
review = df["text"].iloc[doc_id]
print("Review preview (200 chars):")
print(f"{review[:200]}...\n")
```

**위 코드 읽기.** 비교 대상 문서의 원문 앞 200자를 먼저 출력해, 아래 상위 단어들이 실제로 이 리뷰의 내용과 맞는지 눈으로 대조할 수 있게 합니다.

```python
vocab_tf = tfidf.get_feature_names_out()
cv_row = np.asarray(X_count[doc_id].todense()).flatten()
tfidf_row = np.asarray(X_tfidf[doc_id].todense()).flatten()

top = np.argsort(tfidf_row)[::-1][:10]
```

**위 코드 읽기.** `X_count[doc_id]` 와 `X_tfidf[doc_id]` 는 같은 문서의 행을 각각 횟수·TF-IDF 형태로 꺼낸 것입니다. `.todense()` 로 희소 행을 펼쳐 일반 배열로 바꾼 뒤, `np.argsort(tfidf_row)[::-1][:10]` 으로 TF-IDF 값이 큰 상위 10개 단어 인덱스를 고릅니다.

```python
print(f"{'word':>15}  {'count':>6}  {'tfidf':>8}")
print("-" * 35)
for i in top:
    print(f"{vocab_tf[i]:>15}  {cv_row[i]:>6}  {tfidf_row[i]:>8.4f}")
```

**위 코드 읽기.** 같은 단어에 대해 `count`(단순 횟수)와 `tfidf`(가중치) 값을 한 줄에 나란히 출력합니다. 횟수는 1로 똑같아도 TF-IDF 값은 단어마다 달라지는데, 그 차이가 바로 IDF의 희귀도 가중치에서 나옵니다.

**▶ 실행 결과**

```text
Review preview (200 chars):
I stalk this truck.  I've been to industrial parks where I pretend to be a tech worker standing in line, strip mall parking lots, and of cou …(뒤 63자 생략)

           word   count     tfidf
-----------------------------------
          stalk       1    0.2418
        pretend       1    0.2252
          parks       1    0.2212
     industrial       1    0.2212
         farmer       1    0.2212
         divine       1    0.2092
           tech       1    0.2068
          pride       1    0.2046
          bowls       1    0.1988
          gotta       1    0.1898
```

**관찰**: 단순 횟수 기준 top 10에는 `the`, `and` 같은 흔한 단어가 위로 올라옵니다. TF-IDF 정렬에서는 그 문서를 특징짓는 명사·형용사가 상위로 올라오는 경향을 볼 수 있습니다.

> **떡밥**: 두 방식 모두 단어를 *서로 독립* 으로 취급합니다. `"not bad"`(좋다는 뜻)와 `"bad"`(나쁘다는 뜻)을 구분하지 못합니다. BERT가 등장하는 Phase 1에서 이 한계가 어떻게 깨지는지 확인하게 됩니다.
