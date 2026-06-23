`CountVectorizer`는 가장 단순한 변환입니다.

> "이 문서에 단어 X가 몇 번 나왔는가?"

각 문서가 길이 V짜리 벡터로 변환됩니다 (V는 어휘 크기). 대부분의 칸은 0이라 **희소(sparse)** 행렬로 저장합니다.

```python
cv = CountVectorizer(max_features=10000)
X_count = cv.fit_transform(df["text"])
```

**위 코드 읽기** `fit_transform` 한 번이 두 일을 동시에 합니다 — `fit` 으로 5,000건 리뷰에서 어휘를 학습하고(`max_features=10000` 으로 빈도 상위 1만 단어만), `transform` 으로 각 리뷰를 그 어휘 길이의 횟수 벡터로 바꿉니다. 결과 `X_count` 는 희소 행렬입니다.

```python
print(f"shape: {X_count.shape}  (n_docs, vocab_size)")
print(f"non-zero entries: {X_count.nnz:,}")
print(f"total cells: {X_count.shape[0] * X_count.shape[1]:,}")
sparsity = 1 - X_count.nnz / (X_count.shape[0] * X_count.shape[1])
print(f"sparsity: {sparsity:.2%}  (fraction of empty cells)")
```

**위 코드 읽기** `X_count.nnz` 는 0이 아닌 칸 수(실제로 등장한 단어)이고, 이를 전체 칸 수(문서 수 × 어휘 크기)로 나눠 빈 칸의 비율 `sparsity` 를 구합니다.

**▶ 실행 결과**

```text
shape: (5000, 10000)  (n_docs, vocab_size)
non-zero entries: 405,803
total cells: 50,000,000
sparsity: 99.19%  (fraction of empty cells)
```

**결과 해석**

5,000 × 10,000 = 5천만 칸 중 실제로 채워진 건 40만 칸뿐(문서당 평균 약 81개 단어)이라 99.19%가 0입니다. 이렇게 0이 압도적이라 dense 배열로 두면 메모리 낭비가 커서, sklearn 은 0이 아닌 칸만 저장하는 희소 행렬을 씁니다.

학습한 어휘로 임의의 새 문장이 실제로 어떻게 토큰화되는지 확인합니다. `build_analyzer()` 로 벡터라이저 내부의 토큰화 함수를 그대로 꺼내 예시 문장 하나에 적용하고, 결과 토큰 리스트를 출력합니다.

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

```python
vocab = cv.get_feature_names_out()
print(f"Vocab size: {len(vocab):,}")
print(f"First 20: {list(vocab[:20])}")
```

**위 코드 읽기** `get_feature_names_out()` 은 정수 인덱스 ↔ 단어를 잇는 어휘 배열입니다. `vocab[i]` 로 i번째 칼럼이 어떤 단어인지 되짚을 수 있어, 아래에서 빈도 상위 단어의 이름을 찾는 데 씁니다.

```python
word_counts = np.asarray(X_count.sum(axis=0)).flatten()
top = np.argsort(word_counts)[::-1][:10]
print("\nTop 10 most frequent words")
for i in top:
    print(f"  {vocab[i]:>15}  {word_counts[i]:>6,}")
```

**위 코드 읽기** `X_count.sum(axis=0)` 으로 열(단어)별 총 등장 횟수를 구하고, `argsort(...)[::-1][:10]` 으로 그 횟수를 내림차순 정렬해 가장 자주 나온 단어 10개의 인덱스를 뽑습니다.

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

가장 자주 등장하는 단어 10개가 전부 `the`·`and`·`to` 같은 기능어입니다. 모든 리뷰에 흔하게 깔려 있어 정작 그 리뷰가 무엇에 대한 내용인지는 거의 알려주지 않습니다 — 단순 횟수만으로는 문서를 구분하기 어렵다는 뜻이고, 바로 다음 절의 TF-IDF 가 이 흔한 단어들의 비중을 깎는 이유입니다.
