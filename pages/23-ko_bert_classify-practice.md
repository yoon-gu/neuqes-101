> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/23_ko_bert_classify/23_ko_bert_classify.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

```python
%pip install -q -U transformers datasets accelerate
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 123.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 47.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 37.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━ 17.9/48.9 MB 260.1 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 158.1 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 158.1 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 158.1 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 16.4 MB/s eta 0:00:00
```

```python
import warnings
warnings.filterwarnings("ignore")

import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score, confusion_matrix,
)

plt.rcParams["axes.unicode_minus"] = False

# matplotlib 한글 폰트 (Colab — NanumGothic). plot 의 한국어가 □ 로 깨지지 않게.
import matplotlib.font_manager as fm, subprocess, os
_fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(_fp):
    subprocess.run("apt-get -qq -y install fonts-nanum", shell=True)
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "NanumGothic"

# device 자동감지 — Colab(T4) 은 CUDA, 로컬 Mac 은 MPS, 그 외 CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device:         {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
elif DEVICE == "cpu":
    print("Warning: CPU runtime — both MLM and classification will be very slow. Switch to T4 recommended.")
```

**▶ 실행 결과**

```text
PyTorch:        2.11.0+cu128
CUDA available: True
Device:         cuda
GPU:             Tesla T4
```

**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 12:21:55 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   34C    P8             10W /   70W |       3MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## NSMC 이진 분류 데이터 로드 — Ch 15 와 같은 split

NSMC = Naver Sentiment Movie Corpus. 한국어 *binary* 감성 분류의 표준 벤치마크. 한 줄짜리 짧은 리뷰 + 긍정(1) / 부정(0) 라벨. **5,000 train / 1,000 eval, seed 42** — Ch 15 와 *완전히 같은* 셋업.

**원본**: `e9t/nsmc` GitHub 의 `ratings_train.txt` / `ratings_test.txt` TSV. Hugging Face datasets hub 의 nsmc 레포는 *로더 스크립트* 기반이라 최신 datasets 라이브러리에서 deprecated — 그래서 GitHub raw URL 에서 직접 받습니다 (Ch 15 와 동일 패턴).

분류 fine-tune 에 쓸 NSMC 를 먼저 받습니다. `e9t/nsmc` GitHub 의 raw TSV 를 `pandas` 로 직접 읽어 `document` 가 비어 있는 행만 제거합니다. Hugging Face hub 의 nsmc 로더 스크립트가 최신 `datasets` 에서 deprecated 됐기 때문에, Ch 15 와 같은 raw URL 직접 다운로드 패턴을 씁니다.

```python
SEED = 42
N_TRAIN = 5000
N_EVAL = 1000

TRAIN_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
TEST_URL  = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

print("downloading NSMC train/test from GitHub...")
df_train_full = pd.read_csv(TRAIN_URL, sep="\t").dropna(subset=["document"])
df_test_full  = pd.read_csv(TEST_URL,  sep="\t").dropna(subset=["document"])
print(f"  train: {len(df_train_full):,} rows")
print(f"  test:  {len(df_test_full):,} rows")
print(f"  label distribution (train): {df_train_full['label'].value_counts().to_dict()}")
```

**위 코드 읽기** — 전체 원본은 약 15만 train / 5만 test 규모이고, `label` 분포가 0/1 거의 5:5 로 균형 잡혀 있습니다. `random_state` 를 `SEED=42` 로 고정해 두는 게 Ch 15 와 *같은 5K/1K subsample* 을 재현하는 열쇠입니다.

```python
# 5K/1K subsample (Ch 15 와 같은 seed·크기)
df_train = df_train_full.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)
df_eval  = df_test_full.sample(n=N_EVAL,  random_state=SEED).reset_index(drop=True)

print(f"\nsampled train: {len(df_train):,}")
print(f"  positive rate: {df_train['label'].mean():.1%}  (label 1)")
print(f"sampled eval:  {len(df_eval):,}")
print(f"  positive rate: {df_eval['label'].mean():.1%}  (label 1)")

print(f"\nfirst 3 train samples:")
for _, row in df_train.head(3).iterrows():
    label_name = "positive" if row["label"] == 1 else "negative"
    print(f"  label={row['label']} ({label_name})  text={row['document'][:80]}")
```

**위 코드 읽기** — `sample(n=..., random_state=SEED)` 로 5K/1K 만 추출하면 positive rate 가 49% 안팎으로 원본의 균형을 그대로 물려받습니다. 첫 3개 샘플을 찍어 *짧은 한 줄 리뷰* (`원본이 최고` 등) 라는 NSMC 의 특성을 눈으로 확인합니다.

```python
# datasets.Dataset 형태로 변환
ds_train_full = Dataset.from_pandas(df_train[["document", "label"]]).rename_column("document", "text")
ds_eval_full  = Dataset.from_pandas(df_eval[["document", "label"]]).rename_column("document", "text")
print()
print(ds_train_full)
```

**위 코드 읽기** — `pandas.DataFrame` 을 `datasets.Dataset` 으로 옮기고 `document` 컬럼을 `text` 로 이름만 바꿔, 뒤의 `map` 토큰화 파이프라인이 곧바로 받을 수 있는 형태로 만듭니다.

**▶ 실행 결과**

```text
downloading NSMC train/test from GitHub...
  train: 149,995 rows
  test:  49,997 rows
  label distribution (train): {0: 75170, 1: 74825}

sampled train: 5,000
  positive rate: 49.2%  (label 1)
sampled eval:  1,000
  positive rate: 49.9%  (label 1)

first 3 train samples:
  label=1 (positive)  text=원본이 최고
  label=1 (positive)  text=스릴감과 훈훈함이 있는 영화.
  label=1 (positive)  text=굉장히 저평가되는 영화중 하나라고 생각함

Dataset({
    features: ['text', 'label'],
    num_rows: 5000
})
```

**결과 해석**

train positive rate 49.2%, eval positive rate 49.9% 로 두 split 모두 거의 균형이 잡혀 있어, accuracy 가 곧바로 해석 가능한 지표가 됩니다 (불균형 보정 불필요). 첫 샘플들이 `원본이 최고`, `스릴감과 훈훈함이 있는 영화.` 처럼 10~30자 안팎의 짧은 구어체 리뷰임이 확인됩니다.

## 토크나이저 — `klue/bert-base` (Ch 22 와 동일)

vocab 약 32,000 의 한국어 WordPiece. MLM 사전학습과 분류 fine-tune 전 구간에서 *같은 토크나이저* 를 써야 본체가 학습한 임베딩의 의미가 유지됩니다.

MLM 사전학습과 분류 fine-tune 전 구간에서 *같은* 토크나이저를 써야 본체가 학습한 임베딩의 의미가 유지됩니다. 그래서 Ch 22 와 동일하게 `klue/bert-base` 를 불러오고, NSMC 도메인 문장 하나로 토큰화 결과를 미리 확인합니다.

```python
TOKENIZER_NAME = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"tokenizer:        {TOKENIZER_NAME}")
print(f"vocab_size:       {tokenizer.vocab_size:,}")
print(f"model_max_length: {tokenizer.model_max_length}")

# 분류 입력 예시 (NSMC 도메인)
SAMPLE = "이 영화 정말 재미있었고 배우들 연기도 훌륭했어요."
enc = tokenizer(SAMPLE, return_tensors="pt", truncation=True, max_length=128)
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(f"\nNSMC-domain sample: {SAMPLE!r}")
print(f"tokens ({len(tokens)}): {tokens}")
```

**▶ 실행 결과**

```text
tokenizer:        klue/bert-base
vocab_size:       32,000
model_max_length: 512

NSMC-domain sample: '이 영화 정말 재미있었고 배우들 연기도 훌륭했어요.'
tokens (16): ['[CLS]', '이', '영화', '정말', '재미있', '##었', '##고', '배우', '##들', '연기', '##도', '훌륭', '##했', '##어요', '.', '[SEP]']
```

**결과 해석**

vocab 32,000 의 한국어 WordPiece 가 `재미있/##었/##고` 처럼 어간과 어미를 subword 로 분해해, NSMC 의 비격식 구어체도 미등록 토큰 없이 처리합니다. MLM 때와 달리 분류 입력에는 `[CLS]` / `[SEP]` 가 붙는데, 이 `[CLS]` 의 최종 hidden state 가 분류 헤드의 입력이 됩니다.

## MLM 사전학습 — Ch 22 패턴 압축 재현 (한국어 Wikipedia, 3 epoch)

이 노트북을 *self-contained* 로 만들기 위해 Ch 22 의 MLM 사전학습을 여기서 짧게 재현합니다. Ch 22 (5K paragraphs) 보다 *데이터를 줄이고 (2K) 3 epoch* 로 돌려 시간을 단축하면서도, *random init 보다는 낫다* 는 차이를 만들기에는 충분합니다.

**MLM 사전학습 데이터는 *분류용 NSMC 와 별도*** — `wikimedia/wikipedia` config `20231101.ko` paragraphs 5K 를 *새로 로드*. 본 챕터의 *진짜 transfer 메시지* — *일반 한국어 위키 사전학습 → NSMC 영화 리뷰 분류 transfer* 가 노트북 한 구조에 자연스럽게 들어맞도록 *두 데이터셋이 공존*. 같은 토크나이저 (`klue/bert-base`) 가 두 도메인을 모두 처리.

같은 작은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024) → `BertForMaskedLM(config)` random init → 한국어 Wikipedia paragraphs 2K MLM **3 epoch** (1 epoch 만으로는 본체 정렬이 약해 *random init 보다 못한* 결과가 나올 수 있음 — Ch 21 부록 패턴 따라 3 epoch).

이제 Ch 22 와 *완전히 같은* 작은 BERT 본체를 정의합니다 (hidden 256, layer 4, head 4, intermediate 1024). 먼저 `BertForMaskedLM` 으로 random init 한 뒤 한국어 위키로 MLM 사전학습할 것이라, `klue/bert-base` 약 110M 과의 규모 격차를 파라미터 수로 미리 확인해 둡니다.

```python
# Ch 22 와 같은 작은 BERT 설정
HIDDEN_SIZE         = 256
NUM_HIDDEN_LAYERS   = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE   = 1024
MAX_POS_EMBED       = 128
BLOCK_SIZE          = 128

mlm_config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
)

mlm_model = BertForMaskedLM(mlm_config)  # random init
total = sum(p.numel() for p in mlm_model.parameters())
print(f"Small BERT config: hidden={HIDDEN_SIZE}, layer={NUM_HIDDEN_LAYERS}, head={NUM_ATTENTION_HEADS}")
print(f"Total parameters:  {total:,}  ({total/1e6:.2f} M)")
```

**▶ 실행 결과**

```text
Small BERT config: hidden=256, layer=4, head=4
Total parameters:  11,483,136  (11.48 M)
```

**결과 해석**

총 11.48M 파라미터로, `klue/bert-base` (약 110M) 의 약 1/10 규모입니다. 이 작은 본체에 어느 정도의 일반 한국어 표상을 담을 수 있는지가 뒤의 2-way 비교의 핵심 변수입니다.

분류용 NSMC 와 *별도* 로, MLM 사전학습에 쓸 일반 도메인 코퍼스인 한국어 Wikipedia 를 새로 받습니다. *일반 위키 사전학습 → NSMC 영화 리뷰 분류* 라는 진짜 transfer 메시지를 위해, 두 데이터셋이 한 노트북에 공존합니다.

```python
# MLM 사전학습용 일반 도메인 코퍼스: 한국어 Wikipedia (분류용 NSMC 와 별도)
N_MLM_TRAIN = 2000
N_MLM_EVAL  = 400

print("downloading Korean Wikipedia (wikimedia/wikipedia, 20231101.ko)...")
raw_wiki = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
print(f"  total articles: {len(raw_wiki):,}")
```

**위 코드 읽기** — `wikimedia/wikipedia` 의 `20231101.ko` config 를 받습니다. Ch 22 (5K) 보다 작은 2K paragraphs 만 쓰되, 시간 단축을 위해 3 epoch 로 돌릴 예정입니다.

```python
# article 본문을 paragraph 단위로 잘라 N_MLM_TRAIN + N_MLM_EVAL 채우기 (Ch 22 와 같은 collect_paragraphs)
def collect_paragraphs(ds, target, min_len=50, max_len=2000):
    out = []
    for ex in ds:
        for para in ex["text"].split("\n\n"):
            para = para.strip()
            if min_len <= len(para) <= max_len:
                out.append(para)
                if len(out) >= target:
                    return out
    return out

shuffled = raw_wiki.shuffle(seed=SEED)
TARGET = N_MLM_TRAIN + N_MLM_EVAL
all_paragraphs = collect_paragraphs(shuffled, target=TARGET)

mlm_train_raw = Dataset.from_dict({"text": all_paragraphs[:N_MLM_TRAIN]})
mlm_eval_raw  = Dataset.from_dict({"text": all_paragraphs[N_MLM_TRAIN:N_MLM_TRAIN + N_MLM_EVAL]})

print(f"\nMLM train paragraphs: {len(mlm_train_raw):,}  (Korean Wikipedia)")
print(f"MLM eval paragraphs:  {len(mlm_eval_raw):,}")
print(f"first MLM sample: {mlm_train_raw[0]['text'][:120]}")
```

**위 코드 읽기** — `collect_paragraphs` 가 article 을 `\n\n` 으로 잘라 길이 50~2000자 paragraph 만 모읍니다 (Ch 22 와 같은 함수). `shuffle(seed=SEED)` 로 재현성을 확보한 뒤, 모은 paragraph 를 2000/400 으로 train/eval 분리합니다.

```python
def mlm_tokenize(examples):
    return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

mlm_tokenized_train = mlm_train_raw.map(mlm_tokenize, batched=True, remove_columns=["text"])
mlm_tokenized_eval  = mlm_eval_raw.map(mlm_tokenize,  batched=True, remove_columns=["text"])

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result

lm_train = mlm_tokenized_train.map(group_texts, batched=True, batch_size=1000)
lm_eval  = mlm_tokenized_eval.map(group_texts,  batched=True, batch_size=1000)

print(f"\nMLM train blocks: {len(lm_train):,}  (block_size={BLOCK_SIZE})")
print(f"MLM eval blocks:  {len(lm_eval):,}")
```

**위 코드 읽기** — MLM 단계에서는 `add_special_tokens=False` 로 토큰화한 뒤 `group_texts` 가 전체를 하나의 스트림으로 이어 붙여 `BLOCK_SIZE=128` 단위로 다시 자릅니다. 분류 때와 달리 `[CLS]`/`[SEP]` 없이 *연속 토큰 블록* 을 만드는 게 MLM 학습 포맷입니다.

**▶ 실행 결과**

```text
downloading Korean Wikipedia (wikimedia/wikipedia, 20231101.ko)...
  total articles: 647,897
MLM train paragraphs: 2,000  (Korean Wikipedia)
MLM eval paragraphs:  400
first MLM sample: 원(元)은 시호에 쓰이는 글자다. 《일주서》 시법해에는 능사변중(能思辨衆), 행의열민(行義說民), 시건국도(始建國都), 주의덕행(主義行德)을 일컫는다 한다.
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (610 > 512). Running this s …(뒤 56자 생략)
MLM train blocks: 1,562  (block_size=128)
MLM eval blocks:  293
```

**결과 해석**

2,000 paragraph 가 이어 붙여진 뒤 128 토큰 블록 1,562개로 재구성됐습니다. `610 > 512` 경고는 토큰화 단계에서 일부 paragraph 가 길다는 안내일 뿐, `group_texts` 로 어차피 128 단위로 다시 자르므로 학습에 영향이 없습니다.

MLM 학습용 collator 를 만듭니다. `mlm_probability=0.15` 로 매 배치에서 토큰의 약 15% 를 무작위로 가리고, 가려지지 않은 자리는 `labels = -100` 으로 채워 해당 위치의 CE loss 를 무시합니다. 분류 fine-tune 에서는 이 `-100` 트릭을 전혀 쓰지 않습니다 — 모든 sample 에 정답 라벨이 있기 때문입니다.

```python
mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)
```

**`labels = -100` 한 줄 환기** — `DataCollatorForLanguageModeling` 이 가려지지 않은 자리에 `labels = -100` 을 채워 *해당 위치의 CE loss 를 무시*. 분류 fine-tune (다음 섹션) 에서는 -100 을 *전혀 사용하지 않습니다* — 모든 sample 에 *정답 라벨* (0/1) 이 명시되어 있기 때문. 같은 `-100` 트릭이 Phase 4 의 SFT (Ch 27) 에서 *prompt 자리* 를 가리는 정반대 자리로 다시 등장합니다 — 풀버전 표는 Ch 21 §3 *labels = -100 thread* 참조.

MLM 사전학습용 `TrainingArguments` 와 `Trainer` 를 구성합니다. scratch MLM 이라 학습률을 5e-4 로 높게 두고, T4 에서는 `fp16=True` 로 돌립니다 (bf16 은 T4 미지원). 3 epoch 인 이유는 1 epoch 만으로는 본체 정렬이 약해 random init 보다 못한 경우가 생길 수 있어서입니다.

```python
USE_FP16 = (DEVICE == "cuda")
MLM_EPOCHS = 3   # 1 epoch 은 본체 정렬이 약해 random init 보다 못한 경우 발생 — Ch 21 부록 패턴 따라 3 epoch 로

mlm_args = TrainingArguments(
    output_dir="./ch23_mlm_output",
    num_train_epochs=MLM_EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.06,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

mlm_trainer = Trainer(
    model=mlm_model,
    args=mlm_args,
    train_dataset=lm_train,
    eval_dataset=lm_eval,
    data_collator=mlm_collator,
    processing_class=tokenizer,
)

print(f"MLM epochs:        {MLM_EPOCHS}")
print(f"MLM batch size:    {mlm_args.per_device_train_batch_size}")
print(f"MLM learning rate: {mlm_args.learning_rate}")
print(f"MLM fp16:          {USE_FP16}")
print(f"MLM steps:         {len(lm_train) // mlm_args.per_device_train_batch_size * MLM_EPOCHS}")
```

**▶ 실행 결과**

```text
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
MLM epochs:        3
MLM batch size:    32
MLM learning rate: 0.0005
MLM fp16:          True
MLM steps:         144
```

이제 MLM 사전학습을 실행합니다. 평균 train loss 가 random baseline (`ln vocab` ≈ 10.37) 에서 얼마나 내려갔는지로 본체가 일반 한국어 구조를 얼마나 학습했는지 가늠합니다.

```python
t0 = time.time()
mlm_result = mlm_trainer.train()
mlm_elapsed = time.time() - t0
print(f"\nMLM pretraining done in {mlm_elapsed/60:.1f} min")
print(f"mean train loss: {mlm_result.training_loss:.4f}")
print(f"random baseline (ln vocab): {math.log(tokenizer.vocab_size):.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
MLM pretraining done in 0.2 min
mean train loss: 7.9198
random baseline (ln vocab): 10.3735
```

**결과 해석**

학습이 0.2분(약 12초) 만에 끝났고 평균 train loss 가 random baseline 10.37 에서 7.92 로 내려갔습니다. 개요의 예상(5~7 부근)보다 높은 값으로, 데이터·시간이 매우 짧아 본체가 *일반 한국어 구조의 일부* 만 얕게 잡은 상태입니다 — 이 얕은 사전학습이 뒤의 낮은 분류 정확도로 이어집니다.

이번엔 학습에 쓰지 않은 eval 블록으로 사전학습 수준을 객관적으로 잽니다. `eval_loss` 를 `math.exp` 로 변환한 perplexity 는 "가려진 자리에서 모델이 사실상 몇 개 후보로 좁혔는가" 를 뜻해, 토큰 단위 학습 정도를 직관적인 숫자 하나로 보여 줍니다. random baseline PPL 은 vocab 전체를 균등 추측하는 값(약 32,000)이라, 이와의 거리로 본체가 얼마나 학습됐는지 가늠합니다.

```python
mlm_eval_metrics = mlm_trainer.evaluate()
mlm_eval_loss = mlm_eval_metrics["eval_loss"]
print(f"MLM eval loss:        {mlm_eval_loss:.4f}")
print(f"MLM eval perplexity:  {math.exp(mlm_eval_loss):.2f}")
print(f"(random baseline PPL: {tokenizer.vocab_size:,})")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
MLM eval loss:        7.8092
MLM eval perplexity:  2463.13
(random baseline PPL: 32,000)
```

**결과 해석**

eval perplexity 2,463 으로 random baseline 32,000 대비 약 13배 좁혀졌습니다 — 가려진 자리에서 vocab 전체가 아니라 수천 개 후보로 압축한 정도로, 학습이 일어나긴 했으나 매우 얕은 수준입니다.

**관전 포인트** — 한국어 Wikipedia paragraphs 에서 MLM loss 가 *random baseline 10.37* 에서 시작해 5-7 부근까지 떨어졌다면 본체가 *일반 한국어 어휘·문맥 구조의 일부* 를 학습한 상태. perplexity 로 환산하면 vocab 약 32,000 중 *수백-수천 개 후보* 로 좁혀진 정도. Ch 22 의 2 epoch 보다는 약간 얕지만, *NSMC 분류 fine-tune 출발점* 으로는 충분합니다 — 본체가 *일반 한국어 구조* 를 가지면 *영화 리뷰 비격식 도메인* 도 fine-tune 으로 빠르게 적응.

> **체크포인트 저장은 생략** — 노트북 안에서 바로 본체 가중치를 분류 모델로 옮기기 때문. Ch 22 처럼 디스크에 저장하려면 `mlm_model.save_pretrained("./ch23_mlm_ckpt")` 한 줄.

## 헤드 교체 — MLM → 분류 + Fine-tune

이제 *방금 학습된 작은 한국어 BERT 본체* 를 분류 모델로 옮깁니다. 두 가지 흐름:

1. `BertForMaskedLM.bert` (embedding + encoder) 를 그대로 가져옴
2. 새 `BertForSequenceClassification(config)` 을 만들고, 1 의 본체를 *복사*. 분류 헤드는 새로 random init

이렇게 만든 모델을 같은 NSMC 데이터의 *라벨* 까지 사용해 분류 fine-tune. Ch 15 의 hyperparams 와 *완전히 같이* (`lr=2e-5, batch=16, epoch=2, fp16=True`) 둬서 *본체 출발점* 외 모든 조건을 통제.

방금 학습한 MLM 본체를 분류 모델로 옮깁니다. 같은 구조의 config 에 `num_labels=2`, `problem_type="single_label_classification"` 만 더해 `BertForSequenceClassification` 을 만들고, `bert` 본체 가중치만 `load_state_dict` 로 통째로 복사합니다. MLM head 는 버려지고 분류 head (`Linear(256, 2)`) 는 새로 random init 됩니다.

```python
# 분류용 config: 같은 본체 구조 + num_labels=2 + problem_type
cls_config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
    num_labels=2,
    problem_type="single_label_classification",
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)

cls_model = BertForSequenceClassification(cls_config)

# MLM 본체 (embeddings + encoder) 를 분류 모델로 *복사* — pooler 까지 같이
missing, unexpected = cls_model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)
print(f"본체 가중치 복사 완료")
print(f"  missing keys (분류 측에만 있는 부분): {len(missing)}  e.g. {missing[:3] if missing else []}")
print(f"  unexpected keys (MLM 측 잉여):       {len(unexpected)}  e.g. {unexpected[:3] if unexpected else []}")
```

**위 코드 읽기** — `problem_type="single_label_classification"` 이 `CrossEntropyLoss` 를 자동 선택하게 합니다. `load_state_dict(..., strict=False)` 의 반환값 `missing` / `unexpected` 가 *어떤 가중치가 새로 init 되고 어떤 게 버려졌는지* 를 그대로 보여 줍니다.

```python
# 파라미터 수 비교
total_cls = sum(p.numel() for p in cls_model.parameters())
total_body = sum(p.numel() for n, p in cls_model.named_parameters() if "classifier" not in n)
total_head = sum(p.numel() for n, p in cls_model.named_parameters() if "classifier" in n)
print(f"\nClassification model parameters:")
print(f"  body (embeddings + encoder + pooler): {total_body:>10,}  ({total_body/total_cls:.1%})")
print(f"  classifier head Linear(256, 2):       {total_head:>10,}  ({total_head/total_cls:.1%})")
print(f"  total:                                 {total_cls:>10,}  ({total_cls/1e6:.2f} M)")
```

**위 코드 읽기** — 파라미터를 본체와 `classifier` 로 나눠 세어, 분류 head 가 전체에서 차지하는 비중이 사실상 0% (`Linear(256, 2)` = 514개) 임을 확인합니다. *재사용되는 일반 표상은 본체, 새로 배우는 건 작은 head* 라는 사전학습-fine-tune 패러다임의 비율을 눈으로 보여 줍니다.

**▶ 실행 결과**

```text
본체 가중치 복사 완료
  missing keys (분류 측에만 있는 부분): 2  e.g. ['pooler.dense.weight', 'pooler.dense.bias']
  unexpected keys (MLM 측 잉여):       0  e.g. []

Classification model parameters:
  body (embeddings + encoder + pooler): 11,450,624  (100.0%)
  classifier head Linear(256, 2):              514  (0.0%)
  total:                                 11,451,138  (11.45 M)
```

**결과 해석**

missing keys 가 `pooler.dense.weight/bias` 2개뿐이고 unexpected 가 0개 — MLM 본체의 embedding/encoder 가 그대로 옮겨졌고 분류 측에만 있는 pooler 와 classifier 만 새로 init 됐다는 뜻입니다. 분류 head 는 514개로 전체의 0.0%, 사실상 모든 표상을 본체에서 물려받습니다.

**`bert.load_state_dict` 가 한 일** — `BertForMaskedLM` 과 `BertForSequenceClassification` 둘 다 *내부에 같은 `BertModel`* (이름 `self.bert`) 을 갖습니다. 그 본체만 통째로 옮긴 셈. MLM head (`cls.predictions`) 와 분류 head (`classifier`) 는 *모델 객체의 다른 자리* 라 자동으로 분리됩니다.

> Ch 7-18 의 `AutoModelForSequenceClassification.from_pretrained(...)` 가 디스크에서 같은 일을 합니다. 우리는 *방금 학습한 본체* 를 디스크 없이 in-memory 로 옮긴 셈. 디스크 경유는 부록이 아닌 FAQ Q3 에서 짧게 다룹니다.

분류 데이터를 토큰화합니다. MLM 때의 `group_texts` 와 달리 *문장 단위* 로 `[CLS]`/`[SEP]` 를 붙이고 `max_length=128` 로 자르며, `label` 을 정수 `labels` 로 옮깁니다. NSMC 가 짧은 한 줄 리뷰라 토큰 길이가 어느 정도인지도 같이 확인합니다.

```python
# 분류용 토큰화 — 문장 단위, [CLS]/[SEP] 부착, max_length=128
def cls_tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(l) for l in batch["label"]]
    return out

cls_train = ds_train_full.map(cls_tokenize, batched=True).remove_columns(
    [c for c in ds_train_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)
cls_eval  = ds_eval_full.map(cls_tokenize,  batched=True).remove_columns(
    [c for c in ds_eval_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)

print(cls_train)
print(f"\nFirst sample label: {cls_train[0]['labels']}  (int 0 or 1)")
# 토큰화된 첫 샘플의 길이 — NSMC 는 짧은 한 줄 리뷰
lens = [len(s) for s in cls_train["input_ids"]]
print(f"Token length stats — mean: {np.mean(lens):.1f}, median: {np.median(lens):.0f}, max: {max(lens)}")
```

**▶ 실행 결과**

```text
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 5000
})

First sample label: 1  (int 0 or 1)
Token length stats — mean: 21.9, median: 17, max: 117
```

**결과 해석**

평균 21.9 토큰, 중앙값 17 토큰으로 NSMC 리뷰가 매우 짧음이 확인됩니다 (`max_length=128` 안에 거의 모두 들어감). 짧은 문장은 분류 신호가 한두 단어에 집중되는 경향이 있어, 얕게 사전학습된 작은 본체에는 더 까다로운 조건입니다.

평가용 `compute_metrics` 를 정의합니다. logits 에 안정 softmax 를 적용해 클래스 1 의 확률을 뽑고, Ch 15 / Ch 21 과 같은 5종 지표(accuracy / precision / recall / f1 / auc)를 반환합니다. AUC 에는 argmax 가 아니라 *확률* `probs_pos` 를 넣는 점에 주의합니다.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 안정 softmax (K=2)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_full = exp / exp.sum(axis=1, keepdims=True)
    preds = probs_full.argmax(axis=1)
    probs_pos = probs_full[:, 1]   # 클래스 1 의 확률 = AUC 입력

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(labels, probs_pos)),
    }
```

이제 분류 fine-tune 을 돌립니다. Ch 15 와 *완전히 같은* hyperparams (lr 2e-5, batch 16, 2 epoch, fp16) 를 써서 *본체 출발점* 외 모든 조건을 통제합니다. 학습률이 MLM 의 5e-4 보다 훨씬 작은 fine-tune 표준값임에 주목하세요.

```python
# Ch 15 와 같은 hyperparams — 변하는 건 *본체 출발점* 뿐
cls_args = TrainingArguments(
    output_dir="./ch23_cls_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

cls_trainer = Trainer(
    model=cls_model,
    args=cls_args,
    train_dataset=cls_train,
    eval_dataset=cls_eval,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

t0 = time.time()
cls_result = cls_trainer.train()
cls_elapsed = time.time() - t0
print(f"\nClassification fine-tune done in {cls_elapsed/60:.1f} min")
print(f"mean train loss: {cls_result.training_loss:.4f}")
print(f"random baseline (ln 2): {math.log(2):.4f}")
```

**▶ 실행 결과**

```text
<IPython.core.display.HTML object>
Classification fine-tune done in 0.2 min
mean train loss: 0.6911
random baseline (ln 2): 0.6931
```

**결과 해석**

평균 train loss 0.6911 이 random baseline `ln 2` ≈ 0.6931 과 거의 같습니다 — 모델이 사실상 균등 추측(50:50) 단계를 거의 벗어나지 못했다는 신호로, 본체 사전학습이 너무 얕았던 탓입니다.

```python
!nvidia-smi
```

**▶ 실행 결과**

```text
Mon Jun 22 12:22:57 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   46C    P0             40W /   70W |     821MiB /  15360MiB |     23%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4332      C   /usr/bin/python3                        818MiB |
+-----------------------------------------------------------------------------------------+
```
