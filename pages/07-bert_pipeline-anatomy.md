`pipeline("sentiment-analysis")` 한 줄이 사실은 **3단계** 로 구성됩니다.

```
입력 텍스트
   ↓ [1] Tokenizer  (텍스트 → 숫자 ID)
input_ids, attention_mask
   ↓ [2] Model      (숫자 → 로짓)
logits
   ↓ [3] Post-processing (로짓 → 라벨)
{'label': 'POSITIVE', 'score': 0.9998}
```

### 등장 인물 정리

| 컴포넌트 | 라이브러리 | 역할 |
|---|---|---|
| `pipeline` | `transformers` | 위 3단계를 묶은 wrapper |
| Tokenizer | `transformers` (내부적으로 `tokenizers`) | 텍스트를 모델이 먹을 수 있는 숫자로 변환 |
| Model | `transformers` + `torch` | 실제 신경망 forward 연산 |

현재 `classifier` 객체가 어떤 모델/토크나이저를 사용하는지 확인합니다.

```python
print(f"Model:               {classifier.model.config._name_or_path}")
print(f"Model class:         {type(classifier.model).__name__}")
print(f"Tokenizer class:     {type(classifier.tokenizer).__name__}")
print(f"Label mapping:       {classifier.model.config.id2label}")
```

**▶ 실행 결과**

```text
Model:               distilbert/distilbert-base-uncased-finetuned-sst-2-english
Model class:         DistilBertForSequenceClassification
Tokenizer class:     BertTokenizer
Label mapping:       {0: 'NEGATIVE', 1: 'POSITIVE'}
```

기본 모델은 **`distilbert-base-uncased-finetuned-sst-2-english`** 입니다.

- **DistilBERT**: BERT를 40% 작게 만든 경량화 모델 (학생 모델, 지식 증류로 만듦).
- **SST-2**: 영화 리뷰 감성 분류 데이터셋 (Stanford Sentiment Treebank).
- 즉, "BERT를 영화 리뷰 데이터로 파인튜닝한 모델"입니다 — 우리가 Ch 9에서 직접 할 작업의 *완성된* 버전.
