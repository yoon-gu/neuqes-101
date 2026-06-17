## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModelForSequenceClassification(num_labels=2, problem_type="single_label_classification")` | BERT 표준 분류 셋업 | Ch 12 (multi-class)에서 `num_labels=5` 만 바꿔 그대로 |
| `id2label` / `label2id` | config에 사람이 읽는 라벨 이름 등록 (Ch 7 appendix에서 본 것) | 분류 챕터마다 권장 |
| `numpy.exp / sum` 으로 직접 softmax | 안정 softmax 수동 구현 (`max` 빼고 정규화) | Ch 12·13에서 multi-class 확률 추출에 동일 패턴 |
| `seaborn.scatterplot(hue=, alpha=)` | 두 모델의 sample-level prediction 비교에 적합 | Ch 14 auxiliary loss 비교에서 다시 사용 |
| `numpy.corrcoef`, `numpy.abs(a-b).mean()` | 두 분포의 동등성을 수치로 정량화 | 모델 비교가 필요한 챕터마다 |

## 체크포인트 질문

1. 방식 B의 출력 logit shape이 `(B, 2)` 인데, 방식 A와 비교할 1차원 logit을 어떻게 만드나요? 왜 그게 정당한가요?
2. `problem_type="single_label_classification"` 과 `"multi_label_classification"` 의 차이를 한 줄로 설명한다면?
3. 방식 A와 방식 B의 *학습된 가중치* 가 똑같지 않은데도 *최종 확률* 이 거의 같은 이유는?
4. scatter plot에서 점들이 $y=x$ 직선에서 *체계적으로* 위 또는 아래로 치우친다면 무엇을 의심해야 하나요?

## FAQ

### Q1. (실무) BERT 이진 분류는 그냥 방식 B 쓰면 되는 거죠?

네, 99%의 경우 방식 B가 답입니다.

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    problem_type="single_label_classification",  # 사실 num_labels>=2면 default
)
```

방식 A는 *이진 라벨이 multi-label 형식으로 들어와야 하는 특수 시나리오* (예: K개 라벨 중 하나만 binary로 처리하고 나머지는 다른 헤드에 붙이는 구조)에서만 의식적으로 씁니다. 일반 binary classification은 그냥 B.

### Q2. (이론) Ch 4의 sklearn 동등성 시연(`predict_proba` 일치)과 Ch 11의 BERT 동등성 시연(scatter plot) 의 차이는?

| | Ch 4 (sklearn) | Ch 11 (BERT) |
|---|---|---|
| 두 모델이 *같은 가중치* 를 학습? | **사실상 그렇다** — 작은 모델이라 두 방식이 같은 최적해로 수렴 | 아니다 — 67M 파라미터 중 일부가 random init 차이로 다른 곳에 정착 |
| `predict_proba` 가 정확히 같은가? | **거의 일치** (소수점 4-5자리까지) | *근사적으로* 일치 (Pearson 0.99+, MAE 0.01-0.05) |
| 차이의 출처 | sklearn 옵티마이저 수렴 정밀도 | random init + dropout + GPU 비결정성 |

수학적 동등성($\sigma(z) = \mathrm{softmax}(z_0,z_1)[1]$ where $z=z_1-z_0$)은 두 경우 모두 성립합니다. 구현 차원의 노이즈 양이 다를 뿐.

### Q3. (실무) `compute_metrics` 안의 softmax를 직접 구현하지 말고 `torch.softmax` 쓰면 안 되나요?

써도 됩니다. 단, `compute_metrics` 가 받는 `logits` 는 *numpy 배열* 이라 `torch.from_numpy` 로 한 번 감싸야 합니다.

```python
import torch
probs_full = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
```

이 챕터에선 *softmax가 어떻게 동작하는지* 노출하기 위해 `np.exp / sum` 으로 직접 짰습니다 — `max` 빼서 안정화하는 부분이 곧 PyTorch 내부 구현과 동일한 트릭입니다.

### Q4. (이론) 방식 A·B 가 sklearn 안에선 사실상 같은 모델이라고 했는데, 그러면 BERT 에서도 두 학습 결과가 같아야 하지 않나요?

sklearn 의 binary `LogisticRegression()` 은 modern 기본 동작이 *softmax+CE* 든 *sigmoid+BCE* 든 동일한 형태로 학습 (Ch 4 FAQ Q4 — `coef_.shape=(1, V)` 로 collapse). 즉 sklearn 에선 두 방식이 거의 *수치적 등가*. BERT 에선 그렇지 않은 이유는 다음입니다.

sklearn에서는 두 *방식* 의 차이가 BERT보다 훨씬 작습니다. 이유는:

1. **모델 크기**: sklearn LogReg는 ~10K 파라미터. BERT는 67M. 작은 모델이 큰 모델보다 *유일한 최적해* 로 수렴하기 쉽습니다.
2. **무작위성 원천**: sklearn LogReg는 `liblinear`/`lbfgs` 같은 결정론적 옵티마이저 기본. BERT는 SGD 기반 + GPU + dropout 으로 *원천적으로* 비결정적.
3. **데이터 표현**: sklearn TF-IDF는 sparse fixed feature. BERT는 매 backprop마다 hidden state도 업데이트.

그래서 Ch 4에서는 두 sklearn 모델의 `predict_proba` 가 *소수점 4-5 자리* 까지 같았고, Ch 11에서는 *소수점 1-2 자리* 수준에서 같습니다.

### Q5. (실무) 학습된 모델을 저장·로드할 때 두 방식 모델은 호환되나요?

**아니요, 분류 헤드 shape이 달라서 직접 호환되지 않습니다.**

- 방식 A: `classifier.weight.shape = (1, 768)`, `classifier.bias.shape = (1,)`
- 방식 B: `classifier.weight.shape = (2, 768)`, `classifier.bias.shape = (2,)`

`from_pretrained` 로 다른 방식의 체크포인트를 로드하면 분류 헤드를 *새로 초기화* 한다는 경고가 뜹니다 (DistilBERT body는 그대로 로드됨). 따라서 두 방식 사이의 변환은 *처음부터 다시 학습* 이 정확합니다.

만약 **같은 표현력을 유지하면서** 변환하고 싶다면, 방식 A의 $w_A, b_A$ 로부터 방식 B의 헤드를 다음과 같이 만들 수 있습니다 — 한쪽 클래스 logit을 0으로 고정하면 됩니다.

```python
# 방식 A → 방식 B (수학적 동등 변환)
W_A = model_A.classifier.weight    # shape (1, 768)
b_A = model_A.classifier.bias      # shape (1,)

W_B = torch.cat([torch.zeros_like(W_A), W_A], dim=0)  # (2, 768) — 0번 클래스는 0 logit
b_B = torch.cat([torch.zeros_like(b_A), b_A], dim=0)  # (2,)
```

이러면 $z_1 - z_0 = (W_A x + b_A) - 0 = z_A$ 가 정확히 일치합니다. 실무에선 거의 안 쓰지만 *수학적 동등성* 의 구체적 모습을 보여주는 좋은 예시.

### Q6. (실무) `id2label` / `label2id` 는 꼭 등록해야 하나요?

학습·평가 자체에는 영향 없지만 **추론 시 편리** 합니다.

```python
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
pipe("Great food!")
# [{'label': 'POSITIVE', 'score': 0.99}]   ← id2label 등록했으면 사람이 읽는 라벨
# [{'label': 'LABEL_1', 'score': 0.99}]    ← 등록 안 했으면 LABEL_0/1 같은 기본
```

또 `model.config.id2label` 이 huggingface hub에 모델을 올릴 때 widget 라벨 표시용으로 쓰입니다. 한 줄 추가로 큰 이득이라 항상 등록하는 게 좋습니다.

## 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# 라벨을 길이 1 multi-hot 벡터로 두고 num_labels=2 모델에 학습 시도 (Ch 10 라벨 형식 ↔ Ch 11 모델)
def tokenize_wrong(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [[float(b)] for b in batch["binary"]]   # ← 잘못: Ch 10 형식
    return out
```

힌트: `CrossEntropyLoss` 가 받는 라벨은 *int 인덱스 텐서* (shape `(B,)`)인데 위 코드는 *float 2차원 텐서* (shape `(B, 1)`)을 넘깁니다. 텐서 dtype과 shape을 동시에 틀린 케이스라 메시지가 다소 길게 나올 수 있습니다.

## 다음 챕터 예고

**Chapter 12. BERT Multi-class — Yelp 5클래스**

- Yelp 별점 1-5 를 그대로 5클래스 분류로 (Ch 5의 sklearn 버전을 BERT로)
- `num_labels=5`, `problem_type="single_label_classification"` (Ch 11과 같은 표준 셋업, K만 2 → 5)
- Activation은 그대로 softmax, Loss는 그대로 `CrossEntropyLoss`
- **변하는 축은 *task* 하나** — Loss·activation·문제 셋업이 동일. K=2가 K=5로 자연스럽게 일반화되는 모습 확인

> **Phase 1 흐름 정리 (Ch 7 - Ch 14)**: BERT (영어) 위에서 Regression(9) → Binary 방식 A(10) → Binary 방식 B(11) → Multi-class(12) → Multi-label(13) → Auxiliary(14). 한 챕터에 한 축씩만 변합니다.
