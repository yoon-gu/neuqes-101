## 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `AutoTokenizer.from_pretrained("klue/bert-base")` | 한국어 BERT-base 토크나이저 (vocab 32K) | Ch 16-18 모두 같은 토크나이저 |
| `AutoModelForSequenceClassification.from_pretrained("klue/bert-base", ...)` | 한국어 BERT-base + 분류 헤드 | Ch 16-18 모델 본체 |
| `pandas.read_csv(URL, sep="\t")` | GitHub raw TSV 직접 다운로드 | NSMC 외에도 자주 쓰는 패턴 |
| `Dataset.from_pandas(df)` | pandas DataFrame → datasets.Dataset 변환 | 외부 데이터를 transformers 파이프라인에 연결 |

## 체크포인트 질문

1. 같은 한국어 문장을 영어 토크나이저 (`distilbert-base-uncased`) 와 한국어 토크나이저 (`klue/bert-base`) 로 토큰화한 결과가 왜 *그렇게* 다른가요? vocab 의 어떤 차이가 핵심인가요?
2. NSMC 가 영어 Yelp 와 비교해 *왜 더 어려운* 데이터인지 두 가지만 들어보세요.
3. `klue/bert-base` 의 파라미터 수가 110M, DistilBERT 가 67M 인데 정확도 차이가 *극적이지 않은* 이유는?
4. Ch 11 과 Ch 15 사이에 *셋업 코드 자체* 가 거의 똑같은데도 챕터를 분리한 이유는?

## FAQ

### Q1. (실무) 한국어 BERT 모델로 `klue/bert-base` 외에 어떤 선택지가 있나요?

**기본 (입문)**:
- `klue/bert-base` (110M) — 이번 챕터. KLUE 벤치마크와 함께 공개, 한국어 표준.
- `klue/roberta-base` (110M) — RoBERTa 변형, NSP 없이 학습. 보통 KLUE 벤치마크에서 BERT 보다 약간 좋음.

**경량화 (속도/메모리)**:
- `monologg/kobert` (90M) — 초기 한국어 BERT, 입력 vocab 이 작음.
- `monologg/distilkobert` (~28M) — DistilBERT 한국어 버전. 로컬 inference 빠름.

**대형 (정확도)**:
- `klue/roberta-large` (340M) — 메모리·시간 충분하면.
- `kykim/bert-kor-base` (110M) — 다른 사전학습 코퍼스 (대화체에 강함).

**선택 기준**:
- *입문/실험*: klue/bert-base
- *프로덕션 + 속도*: distilkobert
- *최고 정확도*: klue/roberta-large 또는 deberta-v3-large 한국어 변형

### Q2. (이론) 왜 한국어 BERT 가 영어 BERT 와 *같은 아키텍처* 로도 잘 동작하나요?

BERT 의 트랜스포머 인코더는 *언어 가정* 을 거의 안 합니다 — 단지 "토큰 시퀀스의 self-attention" 만. 언어별 차이는 *토크나이저 + 사전학습 데이터* 에서 발생합니다.

```
영어 BERT 와 한국어 BERT 의 차이:
  - 토크나이저 vocab → 32K (한국어) / 30K (영어)
  - 사전학습 corpus → 한국어 위키+뉴스 / 영어 위키+책
  - 모델 weight → 두 corpus 에 맞춰 업데이트 (구조는 동일)
```

같은 아키텍처가 *어떤 텍스트로 학습됐는가* 만 다릅니다. 그래서 다국어 BERT (`xlm-roberta-base`) 는 *하나의 모델* 로 100+ 언어를 처리 — 토크나이저 vocab 만 다국어로 통합.

### Q3. (실무) NSMC 외에 한국어 binary 분류로 흔히 쓰이는 데이터셋은?

| 데이터셋 | 도메인 | 라벨 | 크기 |
|---|---|---|---|
| **NSMC** (이번 챕터) | 영화 리뷰 | 긍정/부정 | 200K (15만 train, 5만 test) |
| **KOSAC** | 다양 (뉴스/리뷰/SNS) | 긍정/부정 + sentiment intensity | 7K |
| **steam-korean-review** | 게임 리뷰 | 긍정/부정 | 100K+ |
| **AI Hub 감성대화** | 대화 | 7가지 감성 (binary 변환 가능) | 70K |

**주의**: 위 데이터셋들은 라이선스가 다양 (CC-BY, MIT, AI Hub 가입 필요 등). 상업 이용 전엔 라이선스 확인 필수.

### Q4. (실무) `klue/bert-base` 가 NSMC 학습에 *왜 잘 들어맞나*?

KLUE 사전학습 코퍼스(약 62GB)에 뉴스·모두의 말뭉치 같은 격식 문어뿐 아니라 *나무위키·국민청원·구어 대화·웹 크롤* 같은 **비격식 문체** 가 섞여 있어, NSMC 의 *짧고 구어체* 인 문장도 BERT 에게 낯설지 않습니다. 형태소 기반 32K vocab 이 `##었`·`##어요` 같은 구어 어미를 온전한 토큰으로 담는 것도 한몫합니다. (영화 리뷰 자체는 코퍼스에 없고, 벤치마크 데이터와 겹치는 문장은 오히려 제거했습니다.) 위키만으로 사전학습된 한국어 모델은 NSMC 같은 *비-격식* 텍스트에 약함.

도메인이 *격식체 문서* (뉴스·법률·특허) 면 그 도메인 코퍼스로 사전학습한 모델이 더 나을 수 있음 — 예: 한국언론진흥재단 **KPF-BERT** (뉴스 기사 특화, GitHub `KPFBERT/kpfbert` 공개).

### Q5. (이론) NSMC 의 *짧은 한 줄 리뷰* 가 학습에 어떤 영향을 주나요?

긍정/부정 신호가 보통 *한두 단어에 집중* 됩니다 (`"명작"`, `"시간 낭비"`, `"감동"`). BERT 입장에서 *문맥 이해* 의 의의가 줄어들고 *키워드 매칭* 에 가까워짐 — sklearn TF-IDF + LogReg 도 NSMC 에서 80% 정확도 가능.

BERT 의 진짜 강점 (긴 문맥에서 *반어*, *조건절*, *비유* 추론) 은 Ch 9-14 의 Yelp 리뷰처럼 *본문이 긴* 데이터에서 더 잘 드러남. Ch 16 의 KLUE-YNAT 은 뉴스 헤드라인이라 NSMC 처럼 짧지만, 감성 키워드가 아니라 *주제어* 를 봐야 하는 task 라 성격이 다름.

### Q6. (실무) 한국어 sentiment task 에서 prediction confidence 가 0.5 근처에 몰리는 샘플들은 보통 어떤 케이스?

세 가지 패턴:

1. **반어/풍자** — `"이걸 영화라고 만든 거야 ㅎㅎ"` 처럼 *문자적 의미와 반대* 의 sentiment. 모델이 표면 단어 (`영화`, `만든`) 와 부정 신호 (`ㅎㅎ` 비웃음) 사이에서 갈팡질팡.
2. **모호한 평가** — `"그냥 그랬음"`, `"볼만함"` 같은 *중립에 가까운* 표현. NSMC 라벨 자체가 binary 라 양쪽 다 가능.
3. **너무 짧음** — `"음..."`, `"글쎄"` 같은 1-2 글자 리뷰. 모델이 판단할 정보 부족.

운영 환경에선 prob ∈ [0.4, 0.6] 샘플들을 *human review* 로 보내는 패턴이 흔함 (active learning).

## 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 결과가 나올까요?

```python
# 한국어 데이터를 *영어* 토크나이저로 학습 시도
tokenizer_wrong = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_wrong = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2,
)
# ... 같은 학습 코드 ...
```

힌트: *코드는 에러 없이 돌아가지만* accuracy 가 50% (random baseline) 근처에 머뭅니다. 영어 vocab 이 한국어를 *이해할 수 없는 부스러기* 로 토큰화해 모델이 학습할 신호가 거의 없음. *Phase 2 의 핵심 교훈* — 한국어엔 *반드시* 한국어 토크나이저 + 한국어 사전학습 모델.

## 다음 챕터 예고

**Chapter 16. 한국어 BERT Multi-class — KLUE-YNAT (뉴스 7분류)**

- 같은 `klue/bert-base`, 같은 토크나이저, 같은 학습 hyperparams
- 변하는 축: *task 차원* (binary K=2 → multi-class K=7)
- 데이터: KLUE-YNAT (뉴스 헤드라인 7분류 — 정치/경제/사회/문화/세계/IT/스포츠)
- Ch 12 의 한국어 버전. 같은 셋업이 K=2·5·7 어디서나 똑같이 동작하는 *일관성* 확인

> **Phase 2 흐름**: Binary (Ch 15) → Multi-class (Ch 16) → Multi-label (Ch 17) → Auxiliary (Ch 18). 토크나이저·모델·hyperparams 가 *Phase 2 안에서는 고정* 이고, *task 만* 바뀝니다.
