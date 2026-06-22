`CountVectorizer`는 가장 단순한 변환입니다.

> "이 문서에 단어 X가 몇 번 나왔는가?"

각 문서가 길이 V짜리 벡터로 변환됩니다 (V는 어휘 크기). 대부분의 칸은 0이라 **희소(sparse)** 행렬로 저장합니다.

```python
cv = CountVectorizer(max_features=10000)
X_count = cv.fit_transform(df["text"])

print(f"shape: {X_count.shape}  (n_docs, vocab_size)")
print(f"non-zero entries: {X_count.nnz:,}")
print(f"total cells: {X_count.shape[0] * X_count.shape[1]:,}")
sparsity = 1 - X_count.nnz / (X_count.shape[0] * X_count.shape[1])
print(f"sparsity: {sparsity:.2%}  (fraction of empty cells)")
```

**▶ 실행 결과**

```text
shape: (5000, 10000)  (n_docs, vocab_size)
non-zero entries: 405,803
total cells: 50,000,000
sparsity: 99.19%  (fraction of empty cells)
```

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

word_counts = np.asarray(X_count.sum(axis=0)).flatten()
top = np.argsort(word_counts)[::-1][:10]
print("\nTop 10 most frequent words")
for i in top:
    print(f"  {vocab[i]:>15}  {word_counts[i]:>6,}")
```

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
