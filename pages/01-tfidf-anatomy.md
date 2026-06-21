`CountVectorizer`는 가장 단순한 변환입니다.

> "이 문서에 단어 X가 몇 번 나왔는가?"

각 문서가 길이 V짜리 벡터로 변환됩니다 (V는 어휘 크기). 대부분의 칸은 0이라 **희소(sparse)** 행렬로 저장합니다.

```python
cv = CountVectorizer(max_features=10000)
X_count = cv.fit_transform(df["text"])
```

**위 코드 읽기.** `CountVectorizer(max_features=10000)` 은 빈도 상위 1만 개 단어만 어휘로 남기겠다는 설정입니다. `fit_transform(df["text"])` 한 번이 어휘 학습(fit)과 행렬 변환(transform)을 동시에 수행해, 5,000개 리뷰를 `(문서 수, 어휘 크기)` 모양의 희소 행렬 `X_count`로 만듭니다.

```python
print(f"shape: {X_count.shape}  (n_docs, vocab_size)")
print(f"non-zero entries: {X_count.nnz:,}")
print(f"total cells: {X_count.shape[0] * X_count.shape[1]:,}")
sparsity = 1 - X_count.nnz / (X_count.shape[0] * X_count.shape[1])
print(f"sparsity: {sparsity:.2%}  (fraction of empty cells)")
```

**위 코드 읽기.** `X_count.nnz` 는 0이 아닌 칸의 수(non-zero)를 셉니다. 전체 칸 수는 `shape[0] * shape[1]` 이고, sparsity는 `1 - nnz / 전체칸` 으로 빈 칸의 비율을 구합니다. 문서 하나에는 어휘 1만 개 중 극히 일부만 등장하니 대부분의 칸이 0임을 수치로 확인하는 부분입니다.

**▶ 실행 결과**

```text
shape: (5000, 10000)  (n_docs, vocab_size)
non-zero entries: 405,803
total cells: 50,000,000
sparsity: 99.19%  (fraction of empty cells)
```

**결과 해석**

5,000만 칸 중 채워진 칸은 약 40만 개뿐이라 sparsity가 99.19%에 이릅니다. 횟수 벡터가 이렇게 비어 있기 때문에 dense가 아닌 희소 행렬로 저장하는 것이 메모리 면에서 필수입니다.

벡터라이저가 문장을 어떤 토큰으로 쪼개는지 직접 확인합니다. `build_analyzer()` 는 `fit` 때 적용된 것과 같은 전처리·토큰화 규칙을 함수로 꺼내 주므로, 한 문장을 넣어 결과 토큰 리스트를 눈으로 볼 수 있습니다.

```python
sample = "I love using Hugging Face!"
analyzer = cv.build_analyzer()
print(f"Input sentence: {sample!r}")
print(f"Tokenized: {analyzer(sample)}")
```

**▶ 실행 결과**

```text
Input sentence: 'I love using Hugging Face!'
Tokenized: ['love', 'using', 'hugging', 'face']
```

**관찰 포인트**

- 모두 **소문자** 로 변환됩니다 (기본 `lowercase=True`).
- 구두점 `!`은 사라집니다 (정규식 패턴이 영숫자만 매칭).
- `"I"` 같은 **단일 문자도 사라집니다** (기본 `token_pattern`은 2자 이상만 인식).
- 학습 어휘에 없는 단어는 OOV로 **무시**됩니다 — BERT처럼 `[UNK]`로 보존하지 않습니다.

학습된 어휘가 어떻게 구성되는지, 그리고 가장 자주 등장한 단어가 무엇인지 확인합니다. `X_count.sum(axis=0)` 으로 단어별 전체 등장 횟수를 더한 뒤 상위 10개를 뽑아, 다음 단계인 TF-IDF가 왜 필요한지에 대한 동기를 만드는 부분입니다.

```python
vocab = cv.get_feature_names_out()
print(f"Vocab size: {len(vocab):,}")
print(f"First 20: {list(vocab[:20])}")
```

**위 코드 읽기.** `get_feature_names_out()` 은 정수 인덱스에 대응하는 실제 단어 목록을 돌려줍니다. 어휘는 알파벳·숫자 순으로 정렬돼 있어, `First 20` 에는 `'00'`, `'000'` 처럼 숫자로 시작하는 토큰이 먼저 나옵니다.

```python
word_counts = np.asarray(X_count.sum(axis=0)).flatten()
top = np.argsort(word_counts)[::-1][:10]
print("\nTop 10 most frequent words")
for i in top:
    print(f"  {vocab[i]:>15}  {word_counts[i]:>6,}")
```

**위 코드 읽기.** `X_count.sum(axis=0)` 은 열(단어) 방향으로 합산해 단어별 총 등장 횟수를 구합니다. `np.argsort(...)[::-1][:10]` 은 정렬 결과를 뒤집어 가장 많이 등장한 상위 10개 단어의 인덱스를 골라냅니다.

**▶ 실행 결과**

```text
Vocab size: 10,000
First 20: ['00', '000', '00am', '00pm', '05', '05nparfrm9annokwdi3bbq', '08', '09', '10', '100', '1000', '100th', '101', '10am', '10min', '1 …(뒤 33자 생략)

Top 10 most frequent words
              the  33,748
              and  21,311
               to  16,702
              was  12,295
               it  10,682
               of  10,226
              for   7,839
               is   7,760
               in   7,593
             that   6,756
```

**결과 해석**

상위 10개가 전부 `the`, `and`, `to` 같은 기능어로 채워졌습니다. 정작 리뷰의 별점이나 주제를 가르는 단어가 아니어서, 단순 횟수만으로는 문서 사이 차이를 드러내기 어렵다는 것이 TF-IDF로 넘어가는 동기입니다.
