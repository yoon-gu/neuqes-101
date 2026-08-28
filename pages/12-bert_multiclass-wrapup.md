## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoModelForSequenceClassification(num_labels=5, problem_type="single_label_classification")` | Ch 11 셋업에서 K만 5로 | Ch 13에서 `multi_label_classification` 으로 K=5 multi-label 매핑 |
| `sklearn.metrics.confusion_matrix` | 혼동 행렬 raw 카운트 | 분류 챕터마다 사용 |
| `seaborn.heatmap(annot=..., fmt="d")` | 혼동 행렬 시각화 (색은 비율, 숫자는 카운트) | 분류 챕터마다 동일 패턴 |
| `roc_auc_score(..., multi_class="ovr")` | multi-class AUC를 One-vs-Rest로 계산 | Ch 13·15-18에서 multi-label / multi-class 평가 |
| `precision_recall_fscore_support(..., average="macro")` | 클래스 불균형에서 모든 클래스에 같은 가중치 | 분류 챕터마다 등장 |

## 체크포인트 질문

1. K=5에서 학습 시작 시 train loss가 약 1.6 정도라면 모델이 무엇을 학습한 상태인가요?
2. *macro* F1과 *micro* (= accuracy) 의 차이가 크다면 무엇을 의심해야 하나요?
3. 혼동 행렬에서 ±1 이웃 클래스 혼동이 많다는 것은 무엇을 시사합니까?
4. BERT의 67M 파라미터가 sklearn의 ~100K 파라미터(TF-IDF 5클래스)보다 정확도가 *몇 %p* 높다면 그 비용을 감수할 가치가 있다고 판단할 수 있나요?

## FAQ

### Q1. (실무) BERT multi-class에서 클래스 불균형이 심하면 어떻게 하나요?

가장 흔한 처치 두 가지:

1. **`class_weight` 적용** — `Trainer.compute_loss` 를 오버라이드해서 `CrossEntropyLoss(weight=...)` 를 명시적으로 사용. 가중치는 보통 `1 / class_count` 의 정규화 형태.

```python
import torch
from torch import nn

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # 가중치를 logits와 같은 device로 — .to("cuda") 하드코딩보다 안전
        w = self.class_weights.to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=w)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# 클래스별 빈도의 역수를 정규화해 가중치로
counts = torch.tensor([1017., 1027., 960., 1021., 975.])   # §1에서 센 클래스 분포
weights = (1.0 / counts) / (1.0 / counts).sum() * len(counts)

trainer_w = WeightedTrainer(
    model=model, args=training_args,
    train_dataset=train_tok, eval_dataset=eval_tok,
    processing_class=tokenizer, compute_metrics=compute_metrics,
    class_weights=weights,
)
```

2. **데이터 단계 oversampling/undersampling** — `imbalanced-learn` 의 `RandomOverSampler` 같은 도구로 학습 데이터를 균형있게. 단순하고 효과적.

### Q2. (이론) softmax는 왜 *상대 logit* 만 보나요? 그러면 절대 logit 값에 의미가 없나요?

수식으로:

$$\mathrm{softmax}(z + c \cdot \mathbf 1)_k = \dfrac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \dfrac{e^{z_k} e^c}{e^c \sum_j e^{z_j}} = \mathrm{softmax}(z)_k$$

모든 logit에 같은 상수 $c$ 를 더해도 softmax 출력이 동일합니다. 즉 K-차원 logit 벡터의 K-1 자유도만 학습됨 (K=5에선 4자유도).

**절대 logit 값은 *학습 신호 크기* 를 결정**합니다. logit 값이 크면 (예: $|z_k| \gg 1$) softmax가 한쪽에 압착되어 gradient가 작아지고, 작으면 (예: $|z_k| \approx 0$) gradient가 균등해 모든 클래스가 학습됩니다. 학습 초기엔 logit이 작아 모든 클래스에 신호가 가는 게 좋고, 후반엔 logit이 커져 confident한 결정을 내리게 됩니다.

### Q3. (실무) eval 데이터에 어떤 클래스가 *하나도* 안 나타나면 AUC 계산이 실패하는데 어떻게 하나요?

`roc_auc_score(multi_class="ovr")` 는 *각 클래스에 대해* positive/negative를 나누어 AUC를 계산하는데, positive 샘플이 0개인 클래스가 있으면 AUC를 정의할 수 없어서 `ValueError` 가 납니다.

처치는 두 가지:

1. **try/except로 NaN 반환** — 이번 챕터의 `compute_metrics` 가 사용하는 패턴. 학습 자체엔 영향 없음.
2. **eval 데이터를 더 모아서 모든 클래스가 등장하도록** — 운영 환경에선 이게 정석. 평가의 통계적 신뢰도 자체에도 도움.

평가 셋이 1,000건 정도면 5클래스가 모두 등장할 가능성이 높지만, 각 클래스 표본이 ~200건 안팎이라 통계 잡음이 큽니다. 가능하면 평가 셋을 늘리세요.

### Q4. (이론) 별점 1-5는 *순서형 라벨* 인데 왜 회귀(Ch 9) 대신 분류(Ch 12)로 푸나요?

둘 다 가능하고 각각 장단점이 있습니다.

| | 회귀 (Ch 9 방식) | 분류 (Ch 12 방식) |
|---|---|---|
| 라벨 의미 | 별점이 *연속* 값 (1.0-5.0) | 별점이 *5개의 명목 클래스* |
| 손실 | MSE — 4★ vs 5★ 오차 $(1)^2 = 1$, 1★ vs 5★ 오차 $(4)^2 = 16$ | CE — 4★ vs 5★ 도 1★ vs 5★ 도 *틀린 건 똑같이* 취급 |
| 출력 | scalar 1.0-5.0 | (5,) 확률 벡터 |
| ±1 인접 오류 페널티 | **작음** (거리에 비례) | 먼 오류와 **동일** (거리 개념 없음) |
| 모델이 *순서* 를 인지? | **자동으로 인지** | 명시적으론 인지 안 함 (학습 데이터 통계로 우회 학습) |

**언제 어느 방식을?** 별점이 *진짜 순서형이고 distance가 의미있다* (1★→5★ 오답이 1★→2★ 오답보다 훨씬 나쁘다) 면 회귀가 자연스럽습니다. 별점이 *카테고리에 가까워서 distance 의미가 약하다* (예: 영화 장르 분류) 면 분류가 자연스럽습니다.

**ordinal regression** 이라는 별도의 분야가 있어 *순서형* 의 특수 구조를 살리는 손실(예: cumulative link)을 씁니다. 입문 수준에선 다루지 않습니다.

### Q5. (실무) BERT가 sklearn 대비 *조금만* 좋다면 BERT를 안 쓰는 게 나을까요?

5-10%p 정도 정확도 향상이라면 trade-off:

- **inference 비용**: BERT는 GPU 권장 (CPU에선 ~50배 느림). sklearn은 CPU로 충분. 운영 비용 차이 5-10배.
- **메모리**: BERT 모델 ~250MB 디스크, ~500MB 메모리. sklearn은 ~10MB.
- **학습 시간**: BERT는 GPU 수십 초-수 분. sklearn은 CPU 수 초-수십 초. 실험 cycle 차이가 큼.
- **유연성**: BERT는 fine-tune이 가능 (도메인 특화 추가 학습). sklearn은 *처음부터 다시* 학습.

단순 *정확도-vs-비용* 만 보면 **5%p 이내 차이면 sklearn**, **10%p 이상이면 BERT 고려** 같은 룰이 무난합니다. 별점 task처럼 *단어 빈도가 강한 신호* 인 도메인은 sklearn 유리, *부정·반어·다단계 추론* 이 중요한 NLI/감성분석 도메인은 BERT 유리.

**그런데 정확도 너머의 가치가 있습니다.** 다음 시나리오에서는 정확도 차이가 *2-3%p* 만 나도 BERT가 압도적으로 유리합니다 — sklearn으로는 *애초에 불가능* 하거나 별도 파이프라인을 통째로 다시 만들어야 하기 때문.

**(1) 새 도메인으로의 빠른 적응 — *transfer learning***

Yelp로 파인튜닝한 이 노트북의 `model` 을 *의료 환자 리뷰* 100-500건만으로 추가 학습해 즉시 도메인 모델을 얻을 수 있습니다. sklearn은 도메인이 바뀌면 *어휘 통계가 처음부터 다시* — 의료 용어 빈도가 일반 리뷰와 달라 TF-IDF가 새 분포에서 학습돼야 하고, 100건은 통계적으로 너무 적습니다.

```python
# §3에서 Yelp로 학습한 model 을 그대로 이어받아 의료 도메인으로 추가 학습
med_args = TrainingArguments(
    output_dir="./ch12_medical", num_train_epochs=3,
    per_device_train_batch_size=16, learning_rate=2e-5,
    fp16=True, save_strategy="no", report_to="none", seed=42,
)
med_trainer = Trainer(
    model=model,                 # ← 사전학습 + Yelp 지식이 이미 들어 있는 상태에서 출발
    args=med_args,
    train_dataset=medical_tok,   # 의료 리뷰 500건을 같은 tokenize_fn 으로 토크나이즈한 것
    processing_class=tokenizer,
)
med_trainer.train()
```

**(2) 다국어 / cross-lingual — 같은 코드로 한국어·일본어·영어**

`xlm-roberta-base` 같은 다국어 BERT는 *동일 모델 + 동일 코드* 로 100+ 언어를 처리합니다. 영어로 파인튜닝한 모델이 한국어 리뷰에도 상당 부분 전이됩니다 (zero-shot cross-lingual transfer). sklearn은 언어마다 토크나이저(형태소 분석기), 불용어 사전, TF-IDF vocabulary를 *각각* 만들어야 합니다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

xlm_tok   = AutoTokenizer.from_pretrained("xlm-roberta-base")   # 100+ 언어 공통
xlm_model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base", num_labels=5,
)

batch = xlm_tok("이 식당 음식이 정말 별로였어요", return_tensors="pt")
with torch.no_grad():
    print(xlm_model(**batch).logits.softmax(-1))   # (1, 5)
# ※ 이 xlm_model 은 아직 파인튜닝 전이라 출력 자체는 무의미합니다.
#    요점은 "한국어 문장이 같은 API·같은 코드로 그대로 들어간다" 는 것.
```

**(3) 분류 너머의 task로 확장 — 같은 백본, 다른 헤드**

같은 `distilbert-base-uncased` 본체로:

| Task | 모델 클래스 | 예시 |
|---|---|---|
| 분류 (이번 챕터) | `AutoModelForSequenceClassification` | 별점 1-5 분류 |
| 토큰 분류 (NER) | `AutoModelForTokenClassification` | "*Bob* 가 *Apple* 에서 *iPhone* 을 샀다" → 사람/회사/제품 추출 |
| 질의응답 | `AutoModelForQuestionAnswering` | "이 리뷰의 만족도는?" 답: "음식은 좋지만 서비스가" |
| 텍스트 생성 | `AutoModelForCausalLM` | 자동 답변 생성 |
| 임베딩 | `AutoModel` (CLS hidden state) | 문장 의미 벡터 |

sklearn LogReg는 *분류 한 가지* 만. 다른 task는 모델 자체가 달라져야 합니다.

**(4) 임베딩 기반 의미 검색·중복 제거**

BERT [CLS] 임베딩은 *문장 의미를 768-dim 벡터* 로 만듭니다. 단어가 달라도 *의미가 비슷하면 가까운 벡터*. TF-IDF는 *같은 단어* 가 있어야 매칭되므로 공통 단어가 없으면 유사도가 0에 가깝습니다.

```python
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel

# 분류 헤드가 아닌 *백본* 을 씁니다 — 헤드는 5클래스 logit만 내놓기 때문
backbone = AutoModel.from_pretrained("distilbert-base-uncased")

def cls_embed(text):
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        return backbone(**batch).last_hidden_state[:, 0]      # (1, 768) — [CLS]

emb1 = cls_embed("The food was great and the staff were kind.")
emb2 = cls_embed("Excellent dishes, and the employees were nice.")
print(cosine_similarity(emb1, emb2))   # 공통 단어가 거의 없어도 높게 나옵니다
```

리뷰 수만 건 중 *의미 중복* 을 제거하거나, 새 리뷰가 들어왔을 때 *비슷한 과거 리뷰* 를 검색하는 시스템 — sklearn으로는 거의 불가능. (`distilbert-base-uncased` 는 영어 모델이라 예시도 영어입니다. 한국어라면 `klue/bert-base` 같은 한국어 백본을 쓰세요 — Ch 15에서 등장합니다.)

**(5) Zero-shot 분류 — 학습 데이터 없이도 동작**

NLI(natural language inference)로 fine-tune된 모델 (`facebook/bart-large-mnli` 등)은 *임의의 라벨* 에 대해 학습 *없이* 분류합니다.

```python
from transformers import pipeline

zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(zsc("The food was terrible.", candidate_labels=["positive", "negative", "neutral"]))
# → {'sequence': ..., 'labels': [...], 'scores': [...]} 형태로 나옵니다.
# ※ bart-large-mnli 는 영어 MNLI 모델이라 한국어 입력 결과는 신뢰할 수 없습니다.
#    한국어가 필요하면 다국어 NLI 모델(예: joeddav/xlm-roberta-large-xnli)을 쓰세요.
```

새 라벨 카테고리가 생길 때마다 학습 데이터 라벨링 비용이 없습니다. 빠른 프로토타이핑·콜드스타트에 강력. sklearn은 *학습 데이터 필수*.

**정리** — sklearn은 *오늘의 task* 를 가장 빠르고 싸게 푸는 도구. BERT는 *task 자체가 진화하거나, 도메인이 늘어나거나, 분류 외 응용으로 확장될 때* 의 자산. 정확도 비교만으로 BERT 가치를 판단하기 부족한 이유입니다.

### Q6. (실무) 한 모델로 *binary와 5-class를 동시에* 풀 수 있나요?

**multi-task learning** 이라는 패턴으로 가능합니다.

```python
import torch
from torch import nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name="distilbert-base-uncased"):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(backbone_name)
        self.head_binary = nn.Linear(768, 2)    # binary head
        self.head_5class = nn.Linear(768, 5)    # 5-class head

    def forward(self, **inputs):
        h = self.backbone(**inputs).last_hidden_state[:, 0]   # CLS
        return {"binary": self.head_binary(h), "5class": self.head_5class(h)}

mt_model = MultiTaskModel()
out = mt_model(**tokenizer("The food was great!", return_tensors="pt"))
print(out["binary"].shape, out["5class"].shape)   # torch.Size([1, 2]) torch.Size([1, 5])

# loss는 두 head의 CE를 *합* (또는 가중합)
# ce = nn.CrossEntropyLoss()
# loss = ce(out["binary"], y_binary) + ce(out["5class"], y_5class)
```

DistilBERT body가 *공유* 되어 두 task가 서로의 학습에 도움을 줍니다. Ch 14 (Auxiliary Loss)에서 비슷한 패턴을 본격적으로 다룹니다.

## 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 에러가 날까요?

```python
# 라벨을 *원-핫* 으로 두고 single_label_classification 모델에 학습 시도
def tokenize_wrong(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    onehot = []
    for l in batch["label"]:
        v = [0.0] * 5
        v[l] = 1.0
        onehot.append(v)
    out["labels"] = onehot   # ← 잘못: shape (B, 5) float
    return out
```

힌트: `single_label_classification` + `CrossEntropyLoss` 가 받는 라벨은 *int 인덱스 1차원 텐서* (shape `(B,)`)인데 위 코드는 *(B, 5)* 형태를 넘깁니다. multi-label 형식의 라벨을 single-label 모델에 넣으려는 흔한 실수.

## 다음 챕터 예고

**Chapter 13. BERT Multi-label — Yelp 항목 키워드**

- 같은 BERT, 같은 데이터에 *항목(food/service/price/ambiance/location) 키워드 자동 라벨링* 추가
- `num_labels=5` 그대로 (Ch 12와 같음), 단 `problem_type="multi_label_classification"` 으로 전환
- Activation은 (per-label) sigmoid, Loss는 (per-label) `BCEWithLogitsLoss`
- 한 리뷰에 *여러 항목이 동시에 등장* 할 수 있음 — single-label과 본질적으로 다른 task
- Ch 6의 sklearn `OneVsRestClassifier(LogisticRegression)` 의 BERT 버전

> **변하는 축**: Ch 12 → Ch 13 은 *Loss 축* (CE → BCE per-label)이 변합니다 — task가 *single-label* 에서 *multi-label* 로 바뀌는 것이 본질이고 그에 맞춰 loss/activation/라벨 형식이 동시에 따라옵니다.
