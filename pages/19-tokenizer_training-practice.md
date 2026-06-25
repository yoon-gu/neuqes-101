> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/19_tokenizer_training/19_tokenizer_training.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 셋업

```python
%pip install -q -U tokenizers transformers datasets
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 85.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 48.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/48.9 MB ? eta -:--:--
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━ 38.3/48.9 MB 171.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 128.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 17.0 MB/s eta 0:00:00
```

```python
import warnings
warnings.filterwarnings("ignore")

import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece, WordLevel
from tokenizers.trainers import WordPieceTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSequence
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import WordPiece as WordPieceDecoder
from transformers import PreTrainedTokenizerFast

# matplotlib 한글 폰트 (Colab — NanumGothic). plot 의 한국어가 □ 로 깨지지 않게.
import matplotlib.pyplot as plt, matplotlib.font_manager as fm, subprocess, os
_fp = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(_fp):
    subprocess.run("apt-get -qq -y install fonts-nanum", shell=True)
fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
else:
    print("Note: this chapter does not train a model, so CPU is fine.")
```

**▶ 실행 결과**

```text
PyTorch:        2.11.0+cu128
CUDA available: True
GPU:             Tesla T4
```

## 영어 코퍼스 — Yelp text 5,000건

`yelp_polarity` 의 train split 에서 5,000 문장만 sample. *라벨은 무시* — 이번 챕터는 *문장 자체* 만 필요.

먼저 영어 코퍼스를 준비합니다. Yelp 리뷰 5,000문장을 받아 토크나이저 학습용 텍스트 리스트로 만들고, 문장당 문자 길이 분포를 확인합니다.

```python
SEED = 42
N_EN = 5000

ds_yelp = load_dataset("fancyzhx/yelp_polarity", split=f"train[:{N_EN}]")
texts_en = list(ds_yelp["text"])
print(f"english corpus: {len(texts_en):,} sentences")
print(f"first sample (truncated):\n  {texts_en[0][:200]}...")
print(f"\nchar length stats:")
char_lens_en = [len(t) for t in texts_en]
print(f"  mean: {np.mean(char_lens_en):.0f}, median: {np.median(char_lens_en):.0f}, max: {max(char_lens_en)}")
```

**▶ 실행 결과**

```text
english corpus: 5,000 sentences
first sample (truncated):
  Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- …(뒤 65자 생략)

char length stats:
  mean: 735, median: 548, max: 5038
```

## 한국어 코퍼스 — NSMC text 5,000건

Ch 15 와 같은 패턴으로 e9t/nsmc GitHub raw 에서 직접 다운로드. 라벨 무시, `document` 컬럼만 사용.

같은 방식으로 한국어 코퍼스도 준비합니다. NSMC(네이버 영화 리뷰) 학습셋을 내려받아 5,000문장을 무작위 추출합니다. 영어와 한국어를 나란히 학습해 언어별 토큰화 차이를 비교하기 위함입니다.

```python
TRAIN_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"

print("downloading NSMC train from GitHub...")
df_nsmc = pd.read_csv(TRAIN_URL, sep="\t").dropna(subset=["document"])
print(f"  total rows: {len(df_nsmc):,}")

N_KO = 5000
texts_ko = df_nsmc["document"].sample(n=N_KO, random_state=SEED).tolist()
print(f"\nkorean corpus: {len(texts_ko):,} sentences")
print(f"first sample:\n  {texts_ko[0]}")
print(f"\nchar length stats:")
char_lens_ko = [len(t) for t in texts_ko]
print(f"  mean: {np.mean(char_lens_ko):.0f}, median: {np.median(char_lens_ko):.0f}, max: {max(char_lens_ko)}")
```

**▶ 실행 결과**

```text
downloading NSMC train from GitHub...
  total rows: 149,995

korean corpus: 5,000 sentences
first sample:
  원본이 최고

char length stats:
  mean: 34, median: 26, max: 143
```

## 토크나이저 4종 학습

같은 두 코퍼스 (영어 / 한국어) × 두 알고리즘 (WordPiece / WordLevel) = 4 개.

**공통 hyperparams**:
- `vocab_size = 8000` — 입문 비교용 작은 크기 (BERT 표준은 30K 안팎)
- 특수 토큰: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` (BERT 컨벤션)
- pre-tokenizer: `Whitespace` (공백·구두점 단위로 1차 분할)
- WordPiece 만 normalizer 적용 (NFD + StripAccents + Lowercase)

이 챕터의 핵심입니다. 사전학습된 토크나이저를 받아오는 대신, `tokenizers` 라이브러리로 토크나이저를 *코퍼스에서 직접 학습* 합니다. `build_wordpiece()` 는 BERT 표준인 WordPiece(서브워드) 토크나이저를, `build_wordlevel()` 은 비교용 WordLevel(어절 단위) 토크나이저를 만듭니다. 두 함수 모두 모델·normalizer·pre-tokenizer 를 조립한 뒤 `train_from_iterator()` 로 vocab 을 학습합니다.

```python
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
VOCAB_SIZE = 8000


def build_wordpiece(corpus_iter, vocab_size=VOCAB_SIZE, lowercase=True):
    '''같은 corpus 에 대해 WordPiece 토크나이저를 학습해 반환.'''
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    norms = [NFD(), StripAccents()]
    if lowercase:
        norms.append(Lowercase())
    tok.normalizer = NormSequence(norms)
    tok.pre_tokenizer = BertPreTokenizer()
    tok.decoder = WordPieceDecoder(prefix="##")

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
        show_progress=False,
    )
    tok.train_from_iterator(corpus_iter, trainer=trainer)

    cls_id = tok.token_to_id("[CLS]")
    sep_id = tok.token_to_id("[SEP]")
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    return tok


def build_wordlevel(corpus_iter, vocab_size=VOCAB_SIZE):
    '''같은 corpus 에 대해 WordLevel (어절 단위) 토크나이저를 학습해 반환.'''
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tok.train_from_iterator(corpus_iter, trainer=trainer)
    return tok


print("helper builders ready: build_wordpiece(), build_wordlevel()")
```

**▶ 실행 결과**

```text
helper builders ready: build_wordpiece(), build_wordlevel()
```

이제 (영어/한국어) × (WordPiece/WordLevel) 조합으로 토크나이저 4개를 모두 같은 `vocab_size=8000` 으로 학습합니다. 각 학습에 걸린 시간을 함께 출력해, 모델 학습과 달리 토크나이저 학습이 GPU 없이 수 초 만에 끝남을 확인합니다.

```python
# 4개 토크나이저 학습 (vocab_size=8000)
t0 = time.time()
tok_en_wp = build_wordpiece(texts_en, lowercase=True)
t_en_wp = time.time() - t0
print(f"[1/4] en WordPiece  trained in {t_en_wp:.2f}s  vocab={tok_en_wp.get_vocab_size()}")

t0 = time.time()
tok_ko_wp = build_wordpiece(texts_ko, lowercase=False)  # 한국어는 lowercase 의미 없음
t_ko_wp = time.time() - t0
print(f"[2/4] ko WordPiece  trained in {t_ko_wp:.2f}s  vocab={tok_ko_wp.get_vocab_size()}")

t0 = time.time()
tok_en_wl = build_wordlevel(texts_en)
t_en_wl = time.time() - t0
print(f"[3/4] en WordLevel  trained in {t_en_wl:.2f}s  vocab={tok_en_wl.get_vocab_size()}")

t0 = time.time()
tok_ko_wl = build_wordlevel(texts_ko)
t_ko_wl = time.time() - t0
print(f"[4/4] ko WordLevel  trained in {t_ko_wl:.2f}s  vocab={tok_ko_wl.get_vocab_size()}")

print(f"\ntotal time: {t_en_wp + t_ko_wp + t_en_wl + t_ko_wl:.2f}s")
```

**▶ 실행 결과**

```text
[1/4] en WordPiece  trained in 1.44s  vocab=8000
[2/4] ko WordPiece  trained in 0.51s  vocab=8000
[3/4] en WordLevel  trained in 0.51s  vocab=8000
[4/4] ko WordLevel  trained in 0.08s  vocab=8000

total time: 2.53s
```

**결과 해석**

토크나이저 4개 모두 목표 `vocab=8000` 에 정확히 도달했고, 총 학습 시간이 2.53초에 불과합니다. 사전학습 모델 없이 코퍼스만으로 vocab 을 처음부터 쌓는 작업이 GPU 없이도 순식간에 끝남을 보여줍니다.

### 3-1. 학습된 vocab 안을 들여다보기

각 vocab 에서 *어떤 토큰이 등장* 했는지 일부 확인. WordPiece 는 `##` prefix 토큰이 보여야 정상.

학습된 vocab 안을 직접 들여다봅니다. id 순서로 앞쪽 특수 토큰과 그 뒤 토큰을 출력하고, `##` prefix 가 붙은 서브워드 토큰이 전체 vocab 에서 차지하는 비율을 셉니다.

```python
def vocab_peek(tok, name, n=15):
    vocab = tok.get_vocab()
    items = sorted(vocab.items(), key=lambda x: x[1])  # id 순서
    print(f"=== {name}  (size={len(vocab)}) ===")
    print(f"  first 5 ids (specials): {[t for t, _ in items[:5]]}")
    print(f"  ids 5-20             : {[t for t, _ in items[5:20]]}")
    # ## prefix 토큰 (subword) 개수
    sub = sum(1 for t in vocab if t.startswith('##'))
    print(f"  subword (##) tokens  : {sub}  ({sub/len(vocab):.1%} of vocab)")
    print()

vocab_peek(tok_en_wp, "en WordPiece")
vocab_peek(tok_ko_wp, "ko WordPiece")
vocab_peek(tok_en_wl, "en WordLevel")
vocab_peek(tok_ko_wl, "ko WordLevel")
```

**▶ 실행 결과**

```text
=== en WordPiece  (size=8000) ===
  first 5 ids (specials): ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
  ids 5-20             : ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/']
  subword (##) tokens  : 1735  (21.7% of vocab)

=== ko WordPiece  (size=8000) ===
  first 5 ids (specials): ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
  ids 5-20             : ['!', '"', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1']
  subword (##) tokens  : 3349  (41.9% of vocab)

=== en WordLevel  (size=8000) ===
  first 5 ids (specials): ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
  ids 5-20             : ['.', 'the', ',', 'and', 'I', 'a', 'to', "'", 'was', 'of', '\\', 'it', 'in', 'is', 'for']
  subword (##) tokens  : 0  (0.0% of vocab)

=== ko WordLevel  (size=8000) ===
  first 5 ids (specials): ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
  ids 5-20             : ['.', '..', '...', '영화', ',', '?', '정말', '!', '너무', '진짜', '~', '이', '....', '더', '왜']
  subword (##) tokens  : 0  (0.0% of vocab)
```

**결과 해석**

WordPiece 한국어는 `##` 서브워드가 vocab 의 41.9% 로, 영어(21.7%)의 거의 두 배입니다. 교착어인 한국어가 어근+조사·어미를 서브워드로 더 많이 쪼개기 때문입니다. WordLevel 은 어절 단위라 두 언어 모두 `##` 토큰이 0개이며, 한국어 vocab 상위에는 `영화`·`정말`·`너무` 같은 통째 어절이 자리합니다.

**관찰**

- **WordPiece** 는 `##ing`, `##ed`, `##ly` 같은 *접미사* 조각이 vocab 의 큰 비중을 차지 — 영어에서 보통 30-50%.
- **WordLevel** 은 `##` 토큰이 0 개 — 어절 단위라 *조각* 개념 자체가 없음.
- 한국어 WordPiece 는 한 글자 조각 (`##다`, `##요`, `##고`) 비중이 높음 — 조사·어미 분리에 효율적.
