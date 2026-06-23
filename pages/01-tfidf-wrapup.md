## 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `datasets` | Hugging Face의 데이터셋 로딩 라이브러리 (Apache Arrow 기반) | Ch 8에서 깊게 본다 |
| `sklearn.feature_extraction.text.CountVectorizer` | 횟수 벡터화 | 이후 챕터의 비교 기준 |
| `sklearn.feature_extraction.text.TfidfVectorizer` | TF-IDF 벡터화 | Ch 2-6에서 입력으로 계속 사용 |

## 체크포인트 질문

1. `CountVectorizer.fit_transform(...)`이 만든 행렬의 shape가 `(N, V)`일 때, N과 V는 각각 무엇을 의미하나요?
2. sparsity가 99%를 넘는 이유는 무엇인가요?
3. TF-IDF가 단순 횟수보다 "문서를 특징짓는 단어"를 더 잘 뽑아내는 이유는 IDF의 어느 부분에서 오나요?
4. `CountVectorizer`가 학습 어휘에 없는 단어를 만났을 때 어떻게 처리하나요? BERT의 처리 방식과는 무엇이 다른가요?

## FAQ

### Q1. (실무) Yelp 데이터가 너무 커서 메모리에 안 올라가는데 어떻게 하나요?

`datasets`는 Apache Arrow 메모리맵을 사용해서 65만 건 전체를 RAM에 올리지 않습니다. 인덱싱 시점에만 디스크에서 읽어 옵니다. 그래서 위 셀에서 `dataset["train"]`을 가져와도 메모리는 거의 늘어나지 않습니다.

그래도 다운스트림(예: pandas 변환, sklearn fit) 단계에서 메모리가 부담되면 두 가지를 씁니다.

```python
# (a) 일부만 잘라서 사용 — 위 셀에서 쓴 패턴
ds = dataset["train"].shuffle(seed=42).select(range(5000))

# (b) streaming 모드 — 전체를 다운로드하지 않고 한 줄씩 받음
stream = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
for i, ex in enumerate(stream):
    if i >= 5000: break
    ...
```

### Q2. (실무) `CountVectorizer`와 `TfidfVectorizer` 중 뭘 써야 하나요?

분류·검색 같은 일반 NLP 작업은 **TF-IDF가 기본 선택**입니다. 흔한 단어의 영향력을 자동으로 깎아주기 때문에 `LogisticRegression`, `LinearSVC` 같은 선형 모델과 잘 맞습니다.

`CountVectorizer`는 두 경우에 유리합니다.

- `MultinomialNB` (Naive Bayes): 빈도(count)를 가정한 모델이라 TF-IDF보다 Count가 더 잘 맞습니다.
- "단어가 등장한 횟수 자체"가 의미 있는 분석(예: 단순 키워드 카운팅, 토픽 모델링 입력).

### Q3. (이론) TF-IDF의 IDF는 정확히 무슨 역할을 하나요?

IDF(Inverse Document Frequency)는 **"이 단어가 얼마나 희귀한가"** 를 점수로 매깁니다. 식은 (sklearn 기본 옵션 기준):

$$\text{idf}(t) = \log\frac{1 + N}{1 + \text{df}(t)} + 1$$

- $N$: 전체 문서 수, $\text{df}(t)$: 단어 $t$가 등장한 문서 수.
- 모든 문서에 등장하는 단어($\text{df}=N$) → $\log 1 = 0$, 거기에 `+1`이 더해져 IDF는 **1**.
- 한 문서에만 등장 → $\log\frac{1+N}{2}$ 큰 값.

즉 흔한 단어는 IDF가 작아 TF-IDF 전체 값이 줄고, 드문 단어는 IDF가 커서 TF가 같아도 점수가 커집니다. "이 문서를 특징짓는 단어"를 골라내는 가중치가 IDF에서 옵니다.

### Q4. (이론) 어휘 크기는 무엇을 기준으로 정해야 하나요? `max_features=10000`은 어떻게 정한 값인가요?

세 가지 트레이드오프가 있습니다.

1. **너무 작으면**: 정보 손실. 핵심 단어가 어휘에서 잘려나감.
2. **너무 크면**: sparsity↑, 노이즈(오타·해시태그·고유명사 1회 등장 단어)도 같이 학습.
3. **모델 학습 시간/메모리**: V가 커지면 선형 모델 가중치도 V만큼 커짐.

실무에선 보통 두 가지를 조합합니다.

```python
TfidfVectorizer(
    max_features=10000,   # 빈도 상위 K개만 남김
    min_df=5,             # 5개 미만 문서에 나오는 단어는 버림
    max_df=0.9,           # 90% 이상 문서에 나오는 단어도 버림
)
```

5,000건짜리 영어 문서 데이터에선 5K-30K가 흔한 출발점입니다. `10000`은 학습 흐름을 빠르게 가져가기 위한 보수적 설정이고, 실험으로 조정하면 됩니다.

### Q5. (실무) sklearn에서 한국어도 처리되나요? 다음 챕터에서도 그대로 동작할까요?

**기본 정규식 토크나이저로도 동작은 합니다** (한국어도 영숫자 패턴으로 잡힘). 다만 한국어는 조사 때문에 같은 어근이 다른 토큰으로 쪼개져 어휘가 폭증합니다 — 예: `학교`, `학교는`, `학교가`, `학교를`이 전부 다른 토큰.

실무에선 형태소 분석기를 토크나이저로 끼워 넣습니다.

```python
# Colab에서: !pip install konlpy
from konlpy.tag import Mecab  # 또는 Okt, Komoran
tokenizer = Mecab().morphs  # 함수: str -> list[str]

TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)
```

이 커리큘럼에서는 Phase 2(Ch 15-18, 한국어)에서 `klue/bert-base`의 한국어 WordPiece 토크나이저를, Phase 3(Ch 19부터)에서 형태소 기반 워드레벨 토크나이저를 직접 다룹니다.

### Q6. (이론) sparse 행렬이 dense 행렬보다 메모리에 유리한 이유는 무엇인가요? `.toarray()`로 바꾸면 왜 메모리가 폭발할 수 있나요?

shape `(5000, 10000)` 행렬을 dense(`float64`)로 만들면 `5000 × 10000 × 8 byte ≈ 400MB`입니다. 칸 대부분이 0인데 그 0까지 다 저장합니다.

sparse(CSR) 행렬은 0이 아닌 칸만 `(값, 열 인덱스)`로 저장합니다. nnz가 50만이면 대략 `500000 × (8 + 4) ≈ 6MB` — 70배 가까이 절약됩니다.

```python
print(f"sparse nbytes ≈ {(X_count.data.nbytes + X_count.indices.nbytes + X_count.indptr.nbytes) / 1e6:.1f} MB")
print(f"dense nbytes  = {(X_count.shape[0] * X_count.shape[1] * 8) / 1e6:.1f} MB")
```

`X_count.toarray()`를 호출하면 dense로 풀어 그 400MB짜리 배열을 메모리에 올립니다. Yelp 5,000건 정도는 버티지만, 50,000건 + max_features 50,000으로 키우면 `50000 × 50000 × 8 = 20GB`가 되어 Colab이 그 자리에서 죽습니다. **sklearn 모델 대부분(LinearRegression, LogisticRegression, LinearSVC 등)은 sparse 입력을 그대로 받기 때문에 굳이 `.toarray()`로 바꿀 일이 없습니다.**

## 삽질 코너 (선택)

다음 코드를 돌려보고 결과를 비교해보세요. 어디가 달라졌나요?

```python
cv2 = CountVectorizer(max_features=10000, lowercase=False, token_pattern=r"\b\w+\b")
X2 = cv2.fit_transform(df["text"])
print(len(cv2.get_feature_names_out()))
```

힌트: `lowercase=False`로 두면 `"good"`과 `"Good"`이 같은 토큰일까요, 다른 토큰일까요? 어휘 크기가 변하는 방향을 예측해보세요.

## 다음 챕터 예고

**Chapter 2. sklearn Regression — 시작점**

- `LinearRegression`으로 별점(1-5)을 회귀합니다.
- 활성화 함수도 없이 출력값을 그대로 사용 — 음수도, 5보다 큰 값도 나올 수 있습니다.
- 다음 챕터의 첫 Loss 등장: `MSELoss` (sklearn: squared error).
