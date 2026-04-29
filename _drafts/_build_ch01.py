"""Build 01_tfidf/01_tfidf.ipynb from inline cell content.

Run:  python3 _drafts/_build_ch01.py
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "01_tfidf" / "01_tfidf.ipynb"

cells = []
_counter = 0


def _cid():
    global _counter
    _counter += 1
    return f"cell{_counter:03d}"


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": text,
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ----- 1. Title -----
md(r"""# Chapter 1. 텍스트를 숫자로 — TF-IDF로 만나는 첫 벡터

**목표**: 자연어 모델을 만나기 전에, 텍스트를 숫자 벡터로 바꾸는 가장 단순한 방법을 손에 익힙니다.

**환경**: Google Colab (GPU 불필요 — sklearn만 사용)

**예상 소요 시간**: 약 5분 (학습 없음, 변환만)

---

## 학습 흐름

1. 🚀 **실습**: Yelp 리뷰 데이터를 로드하고 살펴보기
2. 🔬 **해부**: `CountVectorizer`로 텍스트 → 횟수 벡터
3. 🛠️ **변형**: `TfidfVectorizer`로 흔한 단어의 영향력 줄이기""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

지금은 출발점이라 표가 한 줄뿐입니다. 챕터가 진행될수록 행이 누적됩니다.

| Ch | 모델 | 토크나이저 | Output Head | Activation | Loss |
|---|---|---|---|---|---|
| **1 ← 여기** | (TF-IDF) | `TfidfVectorizer` (학습) | — | — | — |

전체 18챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 시작점 -----
md(r"""## 🔄 시작점에서

이 챕터는 출발점입니다. 모델도 없고, 학습도 없고, loss도 없습니다. **텍스트를 숫자로 바꾸는 변환** 만 다룹니다.

왜 여기서 시작할까요? 자연어 모델은 결국 숫자 벡터를 다루는 함수입니다. BERT든 GPT든 입력 단계에서는 "텍스트 → 숫자" 변환이 반드시 필요합니다. 가장 단순한 변환부터 손에 익혀두면, 이후 챕터에서 BERT의 토크나이저와 임베딩이 어떻게 다른지 비교할 기준이 생깁니다.""")

# ----- 4. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

이번 챕터의 토크나이저는 **`CountVectorizer` / `TfidfVectorizer` 자체** 입니다. 따로 토크나이저 객체가 분리돼 있지 않고, 벡터라이저 한 클래스가 아래 세 가지를 동시에 합니다.

1. **분리(tokenize)**: 기본은 공백·구두점 기준 단어 단위 분리. 정규식 패턴 `(?u)\b\w\w+\b`(영숫자 2자 이상)이 기본값.
2. **어휘 학습(vocabulary)**: 학습 데이터에 등장한 토큰을 모두 모아 정수 인덱스로 매핑.
3. **OOV 처리**: 학습 어휘에 없는 단어는 그냥 **무시**합니다 (BERT의 `[UNK]`처럼 별도 토큰으로 보존하지 않음).

조금 뒤 같은 문장 `"I love using Hugging Face!"` 가 어떻게 토큰화되는지 직접 확인합니다.

> **다음 챕터(Ch 2)**: 같은 `TfidfVectorizer`를 그대로 사용. 토크나이저는 변하지 않습니다.""")

# ----- 5. 환경 셋업 -----
code(r"""!pip install -q datasets scikit-learn pandas matplotlib""")

# ----- 6. import -----
code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

plt.rcParams["axes.unicode_minus"] = False""")

# ----- 7. 실습 도입 -----
md(r"""## 🚀 실습: Yelp 리뷰 데이터 살펴보기

`yelp_review_full`은 Yelp 식당 리뷰 65만 건에 1~5점 별점이 달린 데이터셋입니다 (라벨은 0~4로 저장됨). 학습 흐름을 가볍게 유지하기 위해 **5,000건만 무작위 샘플링** 합니다.""")

# ----- 8. 데이터 로드 -----
code(r"""dataset = load_dataset("yelp_review_full")
print(dataset)""")

# ----- 9. 샘플링 -----
code(r"""SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()

print(f"샘플 수: {len(df)}")
df.head(3)""")

# ----- 10. 별점 분포 -----
code(r"""counts = df["label"].value_counts().sort_index()
labels = [f"{i+1} star" for i in counts.index]
plt.bar(labels, counts.values)
plt.title("Star rating distribution (sampled 5,000)")
plt.ylabel("Reviews")
plt.show()
print(counts)""")

# ----- 11. 텍스트 길이 -----
code(r"""df["len_words"] = df["text"].str.split().str.len()
df[["len_words"]].describe()""")

# ----- 12. 해부 도입 -----
md(r"""## 🔬 해부: CountVectorizer — 텍스트를 "단어 횟수"로

`CountVectorizer`는 가장 단순한 변환입니다.

> "이 문서에 단어 X가 몇 번 나왔는가?"

각 문서가 길이 V짜리 벡터로 변환됩니다 (V는 어휘 크기). 대부분의 칸은 0이라 **희소(sparse)** 행렬로 저장합니다.""")

# ----- 13. CountVectorizer fit -----
code(r"""cv = CountVectorizer(max_features=10000)
X_count = cv.fit_transform(df["text"])

print(f"shape: {X_count.shape}  (문서 수, 어휘 크기)")
print(f"non-zero entries: {X_count.nnz:,}")
print(f"전체 칸 수: {X_count.shape[0] * X_count.shape[1]:,}")
sparsity = 1 - X_count.nnz / (X_count.shape[0] * X_count.shape[1])
print(f"sparsity: {sparsity:.2%}  (비어있는 칸의 비율)")""")

# ----- 14. 같은 문장 토큰화 -----
code(r"""sample = "I love using Hugging Face!"
analyzer = cv.build_analyzer()
print(f"입력 문장: {sample!r}")
print(f"토큰화 결과: {analyzer(sample)}")""")

# ----- 15. 토큰화 관찰 -----
md(r"""**관찰 포인트**

- 모두 **소문자** 로 변환됩니다 (기본 `lowercase=True`).
- 구두점 `!`은 사라집니다 (정규식 패턴이 영숫자만 매칭).
- `"I"` 같은 **단일 문자도 사라집니다** (기본 `token_pattern`은 2자 이상만 인식).
- 학습 어휘에 없는 단어는 OOV로 **무시**됩니다 — BERT처럼 `[UNK]`로 보존하지 않습니다.""")

# ----- 16. 어휘 살펴보기 -----
code(r"""vocab = cv.get_feature_names_out()
print(f"어휘 크기: {len(vocab):,}")
print(f"처음 20개: {list(vocab[:20])}")

word_counts = np.asarray(X_count.sum(axis=0)).flatten()
top = np.argsort(word_counts)[::-1][:10]
print("\n가장 자주 등장한 단어 top 10")
for i in top:
    print(f"  {vocab[i]:>15}  {word_counts[i]:>6,}")""")

# ----- 17. 변형 도입 -----
md(r"""## 🛠️ 변형: TfidfVectorizer — 흔한 단어의 영향력 줄이기

`CountVectorizer`의 한계: `"the"`, `"and"` 같은 단어가 모든 리뷰에 많이 등장하니 횟수만으로는 문서 사이 차이를 잘 드러내지 못합니다.

**TF-IDF**는 두 항을 곱해 이 문제를 다룹니다.

$$\text{tfidf}(t, d) = \underbrace{\text{tf}(t, d)}_{\text{문서 } d \text{에서 } t \text{의 빈도}} \cdot \underbrace{\log\frac{1 + N}{1 + \text{df}(t)}}_{\text{희귀도 가중치 (IDF)}}$$

- `tf`: 한 문서에서 단어가 얼마나 자주 나왔는가
- `idf`: 그 단어가 **얼마나 적은 문서에 등장했는가** (모든 문서에 흔할수록 0에 가까워짐)

직관: "이 단어가 이 문서에서 자주 나오면서 동시에 다른 문서엔 흔하지 않다면, 이 문서를 특징짓는 단어"라는 가중치입니다.""")

# ----- 18. TfidfVectorizer fit -----
code(r"""tfidf = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf.fit_transform(df["text"])
print(f"shape: {X_tfidf.shape}")""")

# ----- 19. 같은 문서 비교 -----
code(r"""doc_id = 0
review = df["text"].iloc[doc_id]
print(f"리뷰 (앞 200자): {review[:200]}...\n")

vocab_tf = tfidf.get_feature_names_out()
cv_row = np.asarray(X_count[doc_id].todense()).flatten()
tfidf_row = np.asarray(X_tfidf[doc_id].todense()).flatten()

top = np.argsort(tfidf_row)[::-1][:10]

print(f"{'단어':>15}  {'count':>6}  {'tfidf':>8}")
print("-" * 35)
for i in top:
    print(f"{vocab_tf[i]:>15}  {cv_row[i]:>6}  {tfidf_row[i]:>8.4f}")""")

# ----- 20. 정리 + 떡밥 -----
md(r"""**관찰**: 단순 횟수 기준 top 10에는 `the`, `and` 같은 흔한 단어가 위로 올라옵니다. TF-IDF 정렬에서는 그 문서를 특징짓는 명사·형용사가 상위로 올라오는 경향을 볼 수 있습니다.

> **떡밥**: 두 방식 모두 단어를 *서로 독립* 으로 취급합니다. `"not bad"`(좋다는 뜻)와 `"bad"`(나쁘다는 뜻)을 구분하지 못합니다. BERT가 등장하는 Phase 1에서 이 한계가 어떻게 깨지는지 확인하게 됩니다.""")

# ----- 21. 라이브러리 정리 -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `datasets` | Hugging Face의 데이터셋 로딩 라이브러리 (Apache Arrow 기반) | Ch 7에서 깊게 본다 |
| `sklearn.feature_extraction.text.CountVectorizer` | 횟수 벡터화 | 이후 챕터의 비교 기준 |
| `sklearn.feature_extraction.text.TfidfVectorizer` | TF-IDF 벡터화 | Ch 2~5에서 입력으로 계속 사용 |""")

# ----- 22. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. `CountVectorizer.fit_transform(...)`이 만든 행렬의 shape가 `(N, V)`일 때, N과 V는 각각 무엇을 의미하나요?
2. sparsity가 99%를 넘는 이유는 무엇인가요?
3. TF-IDF가 단순 횟수보다 "문서를 특징짓는 단어"를 더 잘 뽑아내는 이유는 IDF의 어느 부분에서 오나요?
4. `CountVectorizer`가 학습 어휘에 없는 단어를 만났을 때 어떻게 처리하나요? BERT의 처리 방식과는 무엇이 다른가요?""")

# ----- 23. FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) Yelp 데이터가 너무 커서 메모리에 안 올라가는데 어떻게 하나요?

`datasets`는 Apache Arrow 메모리맵을 사용해서 65만 건 전체를 RAM에 올리지 않습니다. 인덱싱 시점에만 디스크에서 읽어 옵니다. 그래서 위 셀에서 `dataset["train"]`을 가져와도 메모리는 거의 늘어나지 않습니다.

그래도 다운스트림(예: pandas 변환, sklearn fit) 단계에서 메모리가 부담되면 두 가지를 씁니다.

```python
# (a) 일부만 잘라서 사용 — 위 셀에서 쓴 패턴
ds = dataset["train"].shuffle(seed=42).select(range(5000))

# (b) streaming 모드 — 전체를 다운로드하지 않고 한 줄씩 받음
stream = load_dataset("yelp_review_full", split="train", streaming=True)
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

5,000건짜리 영어 문서 데이터에선 5K~30K가 흔한 출발점입니다. `10000`은 학습 흐름을 빠르게 가져가기 위한 보수적 설정이고, 실험으로 조정하면 됩니다.

### Q5. (실무) sklearn에서 한국어도 처리되나요? 다음 챕터에서도 그대로 동작할까요?

**기본 정규식 토크나이저로도 동작은 합니다** (한국어도 영숫자 패턴으로 잡힘). 다만 한국어는 조사 때문에 같은 어근이 다른 토큰으로 쪼개져 어휘가 폭증합니다 — 예: `학교`, `학교는`, `학교가`, `학교를`이 전부 다른 토큰.

실무에선 형태소 분석기를 토크나이저로 끼워 넣습니다.

```python
# Colab에서: !pip install konlpy
from konlpy.tag import Mecab  # 또는 Okt, Komoran
tokenizer = Mecab().morphs  # 함수: str -> list[str]

TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)
```

이 커리큘럼에서는 Phase 2(Ch 13~16, 한국어)에서 `klue/bert-base`의 한국어 WordPiece 토크나이저를, Phase 3(Ch 18)에서 형태소 기반 워드레벨 토크나이저를 직접 다룹니다.

### Q6. (이론) sparse 행렬이 dense 행렬보다 메모리에 유리한 이유는 무엇인가요? `.toarray()`로 바꾸면 왜 메모리가 폭발할 수 있나요?

shape `(5000, 10000)` 행렬을 dense(`float64`)로 만들면 `5000 × 10000 × 8 byte ≈ 400MB`입니다. 칸 대부분이 0인데 그 0까지 다 저장합니다.

sparse(CSR) 행렬은 0이 아닌 칸만 `(값, 열 인덱스)`로 저장합니다. nnz가 50만이면 대략 `500000 × (8 + 4) ≈ 6MB` — 70배 가까이 절약됩니다.

```python
print(f"sparse nbytes ≈ {(X_count.data.nbytes + X_count.indices.nbytes + X_count.indptr.nbytes) / 1e6:.1f} MB")
print(f"dense nbytes  = {(X_count.shape[0] * X_count.shape[1] * 8) / 1e6:.1f} MB")
```

`X_count.toarray()`를 호출하면 dense로 풀어 그 400MB짜리 배열을 메모리에 올립니다. Yelp 5,000건 정도는 버티지만, 50,000건 + max_features 50,000으로 키우면 `50000 × 50000 × 8 = 20GB`가 되어 Colab이 그 자리에서 죽습니다. **sklearn 모델 대부분(LinearRegression, LogisticRegression, LinearSVC 등)은 sparse 입력을 그대로 받기 때문에 굳이 `.toarray()`로 바꿀 일이 없습니다.**""")

# ----- 24. 삽질 코너 -----
md(r"""## 🚀 삽질 코너 (선택)

다음 코드를 돌려보고 결과를 비교해보세요. 어디가 달라졌나요?

```python
cv2 = CountVectorizer(max_features=10000, lowercase=False, token_pattern=r"\b\w+\b")
X2 = cv2.fit_transform(df["text"])
print(len(cv2.get_feature_names_out()))
```

힌트: `lowercase=False`로 두면 `"good"`과 `"Good"`이 같은 토큰일까요, 다른 토큰일까요? 어휘 크기가 변하는 방향을 예측해보세요.""")

# ----- 25. 다음 챕터 예고 -----
md(r"""## 다음 챕터 예고

**Chapter 2. sklearn Regression — 시작점**

- `LinearRegression`으로 별점(1~5)을 회귀합니다.
- 활성화 함수도 없이 출력값을 그대로 사용 — 음수도, 5보다 큰 값도 나올 수 있습니다.
- 다음 챕터의 첫 Loss 등장: `MSELoss` (sklearn: squared error).""")


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT.relative_to(REPO)}  ({len(cells)} cells)")
