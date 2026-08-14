> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/29_benchmark_eval/29_benchmark_eval.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

모델 없이 문자열만으로 *왜 exact match 가 부족한지* 를 즉시 시연합니다. 같은 정답 `24` 를 형식만 다르게 표현한 두 답변을, 완전 일치 채점과 숫자 추출 채점으로 각각 채점해 차이를 봅니다. 생성 평가가 task 마다 정교한 채점기를 요구하는 이유를 손에 잡히게 보여 주는 도입 셀입니다.

```python
# 같은 수학 문제의 "정답" 과, 형식만 다른 두 모델 답변
gold = "24"                       # 채점 기준 정답 (최종 숫자)
answer_a = "정답은 24입니다."        # 내용 O, 형식 다름 (설명이 붙음)
answer_b = "이십사"                # 내용 O, 형식 다름 (한글 표기)

# 방식 1) exact match - 문자열이 완전히 같아야 정답
def exact_match(pred, gold):
    return pred.strip() == gold.strip()

# 방식 2) 숫자 추출 후 비교 - 생성 평가가 실제로 쓰는 방식 (§3)
import re
def extract_int_match(pred, gold):
    nums = re.findall(r"-?\d+", pred)
    return bool(nums) and nums[0] == gold

print("question        : 6 곱하기 4는?")
print(f"gold answer     : {gold!r}\n")
for name, ans in [("answer_a", answer_a), ("answer_b", answer_b)]:
    em = exact_match(ans, gold)
    nm = extract_int_match(ans, gold)
    print(f"{name} = {ans!r}")
    print(f"   exact match        : {em}   (둘 다 내용은 맞는데 exact 는 False)")
    print(f"   extract-int match  : {nm}\n")

print("=> exact match 는 형식이 다르면 내용이 맞아도 '틀림'.")
print("   생성 평가는 이래서 task 마다 정교한 채점(숫자 추출/n-gram/LLM judge)이 필요합니다.")
```

**▶ 실행 결과**

```text
question        : 6 곱하기 4는?
gold answer     : '24'

answer_a = '정답은 24입니다.'
   exact match        : False   (둘 다 내용은 맞는데 exact 는 False)
   extract-int match  : True

answer_b = '이십사'
   exact match        : False   (둘 다 내용은 맞는데 exact 는 False)
   extract-int match  : False

=> exact match 는 형식이 다르면 내용이 맞아도 '틀림'.
   생성 평가는 이래서 task 마다 정교한 채점(숫자 추출/n-gram/LLM judge)이 필요합니다.
```

**결과 해석**

두 답변 모두 내용은 정답 `24` 인데도 exact match 는 둘 다 `False` 입니다. 숫자 추출 방식은 `answer_a` (`정답은 24입니다.`) 는 `True` 로 잡지만, 한글로 쓴 `answer_b` (`이십사`) 는 추출 자체가 실패해 `False` — 정답을 *인식하는 규칙* 이 task 마다 달라야 함을 보여 줍니다.

> 위에서 `answer_a` 는 *숫자 추출* 로는 맞다고 잡히지만, `answer_b` (`"이십사"`) 는 *숫자 추출조차* 실패합니다 — 내용은 정답인데도요. **이것이 생성형 정량 평가의 본질적 어려움** 입니다. 정답을 *인식하는 것 자체* 가 task 마다 다른 규칙을 요구합니다. §2 의 MC 평가가 *생성을 피해* log-likelihood 만 보는 것도, 이 형식 변동 문제를 우회하려는 설계입니다.

## 환경 셋업

평가 챕터라 학습 라이브러리는 가볍게, 대신 표준 평가 도구 `lm-eval` (lm-evaluation-harness) 를 설치합니다. §2-§4 의 직접 구현은 `transformers` + `datasets` 만으로 동작하고, `lm-eval` 은 §5 (표준 도구 소개) 에서만 씁니다.

> `lm-eval` 은 의존성이 많아 설치에 1-2분 걸립니다. §5 를 건너뛰려면 `lm-eval` 설치 라인을 빼도 §1-§4 는 그대로 동작합니다.

```python
%pip install -q -U datasets transformers accelerate
# lm-eval 은 §5 (표준 도구 소개) 에서만 사용. 설치가 무거우면 이 줄만 주석 처리하세요.
%pip install -q lm-eval
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 559.1/559.1 kB 22.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.7/11.7 MB 120.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/50.1 MB ? eta -:--:--
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━ 41.9/50.1 MB 254.7 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 276.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 50.1/50.1 MB 276.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.1/50.1 MB 17.2 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.9/58.9 kB 3.3 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
  Preparing metadata (setup.py) ... done
  Preparing metadata (setup.py) ... done
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.9/8.9 MB 62.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 9.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 11.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 91.1/91.1 kB 10.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.9/65.9 kB 7.2 MB/s eta 0:00:00
  Building wheel for rouge-score (setup.py) ... done
  Building wheel for sqlitedict (setup.py) ... done
  Building wheel for word2number (setup.py) ... done
```

평가 디바이스를 자동 감지하고 (CUDA / MPS / CPU), CUDA 일 때만 추론에 fp16 을 켭니다. 마지막으로 `torch`·`numpy`·`random` 시드를 모두 고정해 *같은 코드가 같은 점수* 를 내도록 재현성을 확보합니다 — 평가에서 결정성은 특히 중요합니다.

```python
import re
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# device 자동 감지 - Colab T4 / 로컬 MPS / CPU 모두 지원
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"device : cuda  ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("device : mps  (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("device : cpu  (evaluation will be slow - Colab T4 recommended)")

# 추론에서도 fp16 은 CUDA 에서만 (MPS 는 미지원, CPU 는 의미 없음)
USE_FP16 = (device.type == "cuda")
DTYPE = torch.float16 if USE_FP16 else torch.float32

# 재현성
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"torch  : {torch.__version__}")
print(f"fp16   : {USE_FP16}")
```

**▶ 실행 결과**

```text
device : cuda  (Tesla T4)
torch  : 2.11.0+cu128
fp16   : True
```

## 평가 대상 모델 로드

**`Qwen/Qwen2.5-0.5B-Instruct`** — 약 0.5B (494M) 파라미터의 *작은 instruct 모델* 입니다. T4 에서 가볍게 돌고, 한국어·영어를 모두 지원해 한국어 벤치마크에서도 *random 보다 나은* 점수를 냅니다. 작은 모델이라 점수 자체는 낮지만, *평가 파이프라인을 끝까지 돌려보기* 에 적합합니다.

> 비교용으로 Ch 28 에서 만든 KoGPT2 SFT 모델을 함께 평가할 수도 있습니다. KoGPT2 (125M) 는 너무 작아 대부분의 벤치마크에서 *거의 random* 입니다 — **그 자체가 §7 의 교훈**: 작은 모델의 벤치마크 한계. 본 노트북은 Qwen 을 메인으로 진행합니다.

평가 대상인 `Qwen2.5-0.5B-Instruct` 를 로드합니다. *학습이 아니라 추론* 만 할 것이므로 `model.eval()` 로 평가 모드를 켜고 (dropout 비활성화 등), 파라미터 수·vocab 크기·EOS 토큰을 출력해 어떤 모델을 다루는지 확인합니다.

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"model     : {MODEL_NAME}")
print(f"params    : {n_params/1e6:.1f}M")
print(f"vocab     : {tokenizer.vocab_size}")
print(f"eos token : {tokenizer.eos_token!r}")
```

**▶ 실행 결과**

```text
model.safetensors: downloading bytes:           |  0.00B            
model     : Qwen/Qwen2.5-0.5B-Instruct
params    : 494.0M
vocab     : 151643
eos token : '<|im_end|>'
```

## Multiple-choice 평가 직접 구현 (핵심)

객관식 벤치마크 (MMLU, KMMLU, HellaSwag, KoBEST ...) 는 **모델에게 답을 생성하게 하지 않습니다**. 대신 각 선택지를 *문맥에 이어붙였을 때 모델이 얼마나 그럴듯하게 보는가* — 즉 **log-likelihood** 를 계산해 가장 높은 선택지를 정답으로 예측합니다.

### 왜 생성이 아니라 log-likelihood 인가

- 생성하면 *형식 변동* (모델이 "정답은 3번" vs "세 번째" vs 그냥 본문을 이어 씀) 때문에 채점이 불안정합니다.
- log-likelihood 는 *각 선택지에 대한 모델의 확신* 을 직접 수치로 비교 — 형식에 흔들리지 않고 *재현 가능* 합니다.

### 계산 원리

선택지 $c$ 의 토큰을 $(t_1, ..., t_k)$ 라 하면, 문맥 (prompt) 뒤에 이어질 확률의 로그는

$$\log P(c \mid \text{prompt}) = \sum_{i=1}^{k} \log P(t_i \mid \text{prompt}, t_{<i})$$

모델의 `logits` 에 `log_softmax` 를 씌워 *각 선택지 토큰 위치의 정답 토큰 log-prob* 를 뽑아 더하면 됩니다. **`lm-eval-harness` 도 내부에서 이 계산을 합니다** — 우리는 그 원리를 그대로 구현합니다.

### log-prob 합 (sum) vs 평균 (mean) — 길이 정규화

선택지마다 토큰 수가 다르면 *단순 합* 은 *짧은 선택지* 에 유리합니다 (log-prob 는 음수라, 토큰이 적을수록 합이 덜 깎임). 그래서 **토큰 수로 나눈 평균 log-prob** (length-normalized) 를 쓰기도 합니다.

| 방식 | 식 | 성질 |
|---|---|---|
| sum (`acc`) | $\sum_i \log P(t_i)$ | 길이가 비슷한 선택지에 적합 |
| mean (`acc_norm`) | $\frac{1}{k}\sum_i \log P(t_i)$ | 길이 편향 완화 (HellaSwag 처럼 선택지 길이가 다를 때) |

`lm-eval-harness` 가 `acc` 와 `acc_norm` 두 점수를 함께 내는 이유가 이것입니다. 아래에서 둘 다 계산해 비교합니다.

이 셀이 MC 평가의 심장입니다. 선택지를 *생성하지 않고*, prompt 뒤에 각 선택지를 이어붙였을 때 그 선택지 토큰들의 log-likelihood 를 forward 한 번으로 계산합니다. 길이가 길어 핵심 단계별로 나눠 읽겠습니다.

```python
@torch.no_grad()
def continuation_logprob(prompt: str, continuation: str):
    '''prompt 뒤에 continuation 이 이어질 때, continuation 토큰들의 log-prob 를
    (sum, mean) 으로 반환. teacher forcing - 생성하지 않고 한 번의 forward 로 계산.'''
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids
    # continuation 이 차지하는 토큰 수 (경계는 tokenizer 가 정함)
    cont_len = max(1, full_ids.shape[1] - prompt_ids.shape[1])
```

**위 코드 읽기** — `prompt` 만 토큰화한 길이와 `prompt + continuation` 을 합쳐 토큰화한 길이의 차이로 `cont_len` (선택지가 차지하는 토큰 수) 을 구합니다. 경계를 글자가 아니라 *토크나이저가 정한 토큰 수* 로 잡는 것이 핵심이며, `@torch.no_grad()` 로 그래디언트 없이 추론만 합니다.

```python
    full_ids = full_ids.to(device)
    logits = model(full_ids).logits[0]                 # (T, V)
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # (T, V)
```

**위 코드 읽기** — 합친 시퀀스를 모델에 한 번 통과시켜 각 위치의 `logits` (T×V) 를 얻고, `log_softmax` 로 각 위치의 *전체 vocab 에 대한 로그 확률* 로 바꿉니다. 생성 루프 없이 forward 한 번이면 모든 토큰 위치의 확률이 동시에 나오는 것이 log-likelihood 평가가 빠른 이유입니다.

```python
    # 위치 i 의 토큰은 위치 i-1 의 logits 가 예측 -> 한 칸 시프트
    target = full_ids[0, 1:]                            # (T-1,)
    pred_lp = log_probs[:-1]                            # (T-1, V)
    token_lp = pred_lp[torch.arange(target.shape[0]), target]  # (T-1,)

    cont_lp = token_lp[-cont_len:]                     # 마지막 cont_len 개 = continuation
    return cont_lp.sum().item(), cont_lp.mean().item()
```

**위 코드 읽기** — 자기회귀 모델은 위치 `i-1` 의 logits 가 위치 `i` 의 토큰을 예측하므로 target 과 예측을 한 칸 어긋나게 맞춘 뒤, 실제 정답 토큰 자리의 log-prob 만 골라 뽑습니다 (`token_lp`). 그중 *마지막 `cont_len` 개* 가 선택지에 해당하므로, 그 합 (sum) 과 평균 (mean) 을 함께 반환합니다 — sum 은 `acc`, mean 은 길이 정규화된 `acc_norm` 의 재료입니다.

```python
def mc_predict(prompt: str, choices: list[str]):
    '''각 선택지의 (sum, mean) log-prob 를 구해 argmax. 두 방식의 예측을 모두 반환.'''
    sums, means = [], []
    for c in choices:
        s, m = continuation_logprob(prompt, c)
        sums.append(s)
        means.append(m)
    return int(np.argmax(sums)), int(np.argmax(means)), sums, means
```

**위 코드 읽기** — `mc_predict` 는 같은 prompt 에 대해 모든 선택지의 (sum, mean) log-prob 를 모은 뒤 각각 argmax 해 *가장 그럴듯한 선택지* 를 고릅니다. 생성 없이 "고르기만" 하는 MC 평가의 본질이 이 argmax 두 줄에 담겨 있습니다.

```python
# 동작 확인 - 정답이 명확한 간단한 한 문항 (4지선다)
demo_prompt = "1 더하기 1은 "
demo_choices = ["2입니다.", "3입니다.", "5입니다.", "10입니다."]
pred_sum, pred_mean, sums, means = mc_predict(demo_prompt, demo_choices)
demo_df = pd.DataFrame({
    "choice": demo_choices,
    "logprob_sum": [round(x, 2) for x in sums],
    "logprob_mean": [round(x, 3) for x in means],
})
print(demo_df.to_string(index=False))
print(f"\npredicted (sum)  : {demo_choices[pred_sum]}")
print(f"predicted (mean) : {demo_choices[pred_mean]}")
```

**위 코드 읽기** — 정답이 자명한 `1 더하기 1은` 문항으로 함수가 제대로 도는지 확인합니다. 네 선택지의 log-prob 를 표로 출력하고, sum·mean 두 방식이 모두 정답 `2입니다.` 를 고르는지 봅니다.

**▶ 실행 결과**

```text
choice  logprob_sum  logprob_mean
 2입니다.        -5.01        -1.670
 3입니다.        -5.42        -1.808
 5입니다.        -6.45        -2.151
10입니다.        -7.59        -1.898

predicted (sum)  : 2입니다.
predicted (mean) : 2입니다.
```

**결과 해석**

정답 `2입니다.` 의 log-prob 가 sum (-5.01), mean (-1.670) 모두에서 가장 높아 두 방식 다 정답을 골랐습니다. 합(sum) 기준으로는 오답일수록 log-prob 합이 낮습니다 (`5입니다.` -6.45 > `10입니다.` -7.59) — 다만 mean(길이 정규화) 기준에서는 길이가 다른 오답의 순서가 뒤집힐 수 있어 (`10입니다.` -1.898 > `5입니다.` -2.151), 다음 절에서 정규화를 다룹니다.

### KoBEST HellaSwag — 4지선다 상식추론

**`skt/kobest_v1`** 의 `hellaswag` task 는 *문맥 (context)* 뒤에 가장 자연스러운 *다음 문장* 을 4개 후보 (`ending_1..4`) 중에서 고르는 한국어 상식추론 벤치마크입니다. 선택지 길이가 제각각이라 *길이 정규화 (mean)* 효과가 잘 드러납니다. T4 + 시간 제약상 **test split 의 앞 50문항** 만 평가합니다.

KoBEST HellaSwag (한국어 4지선다 상식추론) 의 test split 에서 앞 50문항만 불러옵니다. T4 30분 제약 때문에 subset 만 쓰며, 한 문항의 context·4개 ending·정답 label 구조를 출력해 데이터 형태를 확인합니다.

```python
from datasets import load_dataset

N_HELLASWAG = 50  # T4 30분 룰 - subset 만. 전체 500문항은 너무 오래 걸림
hellaswag = load_dataset("skt/kobest_v1", "hellaswag", split="test").select(range(N_HELLASWAG))
print(f"HellaSwag subset : {len(hellaswag)} 문항 (4지선다)")
print(f"columns          : {hellaswag.column_names}")

# 한 문항 예시
ex = hellaswag[0]
print("\n--- example ---")
print("context :", ex["context"][:60], "...")
for i in range(1, 5):
    print(f"ending_{i} :", ex[f'ending_{i}'][:40])
print("label   :", ex["label"])
```

**▶ 실행 결과**

```text
HellaSwag subset : 50 문항 (4지선다)
columns          : ['context', 'ending_1', 'ending_2', 'ending_3', 'ending_4', 'label']

--- example ---
context : 여자는 새벽까지 잔업을 마치고 기지개를 켠다. 몇 시인지 확인하려고 핸드폰을 켜자 여자와 헤어진 남자친구의  ...
ending_1 : 여자는 울리는 벨소리를 무시한다.
ending_2 : 벨소리가 그치자 여자는 헤어진 남자친구의 번호를 차단한다.
ending_3 : 여자는 핸드폰을 내려놓고 잠자리에 든다.
ending_4 : 여자가 답장하지 않자 전 남자친구에게서 전화가 걸려온다.
label   : 3
```

각 문항마다 context 를 prompt 로 삼아 4개 ending 의 log-prob 를 비교, argmax 한 선택지가 정답 label 과 같은지 셉니다. sum 방식 (`acc`) 과 mean 방식 (`acc_norm`) 정확도를 함께 내어 *길이 정규화 효과* 를 random baseline (0.25) 과 비교합니다.

```python
def eval_hellaswag(dataset):
    '''각 문항에서 context 뒤 4개 ending 의 log-prob 를 비교해 argmax.
    sum 방식 (acc) 과 mean 방식 (acc_norm) 정확도를 함께 반환.'''
    correct_sum = correct_mean = 0
    for ex in dataset:
        prompt = ex["context"] + " "
        choices = [ex[f"ending_{i}"] for i in range(1, 5)]
        pred_sum, pred_mean, _, _ = mc_predict(prompt, choices)
        correct_sum += int(pred_sum == ex["label"])
        correct_mean += int(pred_mean == ex["label"])
    n = len(dataset)
    return correct_sum / n, correct_mean / n


acc_sum, acc_mean = eval_hellaswag(hellaswag)
print(f"KoBEST HellaSwag  (n={len(hellaswag)})")
print(f"  acc      (sum  / log-prob)     : {acc_sum:.3f}")
print(f"  acc_norm (mean / length-norm)  : {acc_mean:.3f}")
print(f"  random baseline (1/4)          : 0.250")
```

**▶ 실행 결과**

```text
KoBEST HellaSwag  (n=50)
  acc      (sum  / log-prob)     : 0.320
  acc_norm (mean / length-norm)  : 0.400
  random baseline (1/4)          : 0.250
```

**결과 해석**

`acc` 0.320, `acc_norm` 0.400 으로 둘 다 random (0.250) 을 웃돕니다. 선택지 길이가 제각각인 HellaSwag 답게 *길이 정규화한 mean 방식 (`acc_norm`)* 이 sum 방식보다 8%p 높아, 길이 편향 완화의 효과가 그대로 드러납니다.

### KoBEST BoolQ — 2지선다 (예 / 아니오)

`boolq` task 는 *본문 (paragraph)* 과 *질문 (question)* 을 주고 **예 / 아니오** 를 묻습니다. 선택지가 2개라 *random baseline 은 0.5* — MC 평가를 *가장 단순한 형태* 로 보여줍니다. 여기서는 질문 뒤에 "예" / "아니오" 를 이어붙여 log-prob 를 비교합니다 (label 0 = 아니오, 1 = 예).

KoBEST BoolQ 는 본문·질문을 주고 *예 / 아니오* 를 묻는 2지선다입니다. 같은 MC 코드로, 질문 뒤에 "아니오" / "예" 를 이어붙여 log-prob (mean) 가 높은 쪽을 고릅니다. 선택지 인덱스가 곧 label (0=아니오, 1=예) 이라 random baseline 은 0.5 입니다.

```python
N_BOOLQ = 50
boolq = load_dataset("skt/kobest_v1", "boolq", split="test").select(range(N_BOOLQ))
BOOLQ_CHOICES = ["아니오", "예"]  # 인덱스가 곧 label (0=아니오, 1=예)


def eval_boolq(dataset):
    correct = 0
    for ex in dataset:
        prompt = f"본문: {ex['paragraph']}\n질문: {ex['question']}\n답변: "
        _, pred_mean, _, _ = mc_predict(prompt, BOOLQ_CHOICES)
        correct += int(pred_mean == ex["label"])
    return correct / len(dataset)


acc_boolq = eval_boolq(boolq)
print(f"KoBEST BoolQ  (n={len(boolq)})")
print(f"  acc             : {acc_boolq:.3f}")
print(f"  random baseline : 0.500  (2지선다)")
```

**▶ 실행 결과**

```text
KoBEST BoolQ  (n=50)
  acc             : 0.460
  random baseline : 0.500  (2지선다)
```

**결과 해석**

BoolQ 정확도 0.460 으로 random (0.500) 을 *오히려 살짝 밑돕니다*. 0.5B 라는 작은 모델이 한국어 본문 이해·예/아니오 판단에서 신호를 거의 못 내는 경우로, §7 의 "작은 모델은 벤치마크에서 random 근처" 라는 교훈을 그대로 보여 줍니다.

> **여기까지가 MC 평가의 전부입니다.** 생성은 한 번도 하지 않았습니다 — 오직 *각 선택지의 log-likelihood* 를 forward 한 번씩으로 계산해 argmax 했을 뿐입니다. 점수가 random 근처라면, 그건 *0.5B 라는 작은 모델의 한계* 입니다 (§7). 같은 코드를 더 큰 모델에 그대로 적용하면 점수가 오릅니다.

## Generation + 정답 추출 평가 (생성 기반)

두 번째 format 은 **모델이 실제로 답을 생성** 한 뒤, 그 텍스트에서 *정답을 파싱* 해 채점합니다. GSM8K (초등 수학), HumanEval (코드 실행) 가 대표적입니다. 여기서는 가벼운 **산술 문제 subset** 으로 *생성 → 정규식으로 숫자 추출 → 정답 비교* 흐름을 시연합니다.

### few-shot prompt

모델에게 *예시 몇 개* (few-shot) 를 먼저 보여줘 *답변 형식* 을 유도합니다. 예시 없이 (zero-shot) 던지면 모델이 형식을 못 맞춰 추출이 실패하기 쉽습니다 — §4 에서 그 차이를 정량으로 봅니다.

두 번째 format 인 *생성 + 정답 추출* 을 시연합니다. GSM8K 대신 정답이 명확한 가벼운 산술 subset 으로, 모델이 답을 *실제로 생성* 한 뒤 정규식으로 숫자를 뽑아 채점합니다. 길어서 단계별로 나눠 읽겠습니다.

```python
# 가벼운 산술 subset (GSM8K 대신 - 빠르고 정답이 명확해 추출 평가에 적합)
ARITHMETIC = [
    ("6 곱하기 4는 얼마인가요?", 24),
    ("15 더하기 9는 얼마인가요?", 24),
    ("20 빼기 8은 얼마인가요?", 12),
    ("7 곱하기 7은 얼마인가요?", 49),
    ("100 빼기 37은 얼마인가요?", 63),
    ("13 더하기 28은 얼마인가요?", 41),
]

# few-shot 예시 (답변 형식을 보여줌)
FEWSHOT = (
    "Q: 사과 3개와 5개를 더하면 몇 개인가요?\nA: 정답은 8입니다.\n\n"
    "Q: 12에서 7을 빼면 얼마인가요?\nA: 정답은 5입니다.\n\n"
)
```

**위 코드 읽기** — 채점 대상 산술 문제 6개와 그 정답을 정의하고, `FEWSHOT` 에 *답변 형식* (`정답은 N입니다.`) 을 보여 주는 예시 2개를 미리 문자열로 조립합니다. 이 few-shot 프롬프트가 §4 의 zero-shot 과 대비될 핵심 변수입니다.

```python
def extract_first_int(text: str):
    '''생성 텍스트에서 첫 정수를 추출 (정답 파싱). 없으면 None.'''
    m = re.findall(r"-?\d+", text)
    return int(m[0]) if m else None


@torch.no_grad()
def generate_answer(prompt: str, max_new_tokens: int = 24):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,                       # greedy - 평가는 재현성 위해 deterministic
        pad_token_id=tokenizer.eos_token_id,
    )
    # 새로 생성된 토큰만 디코드
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
```

**위 코드 읽기** — `extract_first_int` 는 생성 텍스트에서 *첫 정수* 를 정규식으로 뽑는 채점기로, MC 와 달리 생성 결과를 파싱해야 함을 보여 줍니다. `generate_answer` 는 `do_sample=False` (greedy) 로 결정적으로 생성하고 (평가 재현성), 입력 길이 이후의 *새로 생성된 토큰만* 잘라 디코드합니다.

```python
def eval_generation(problems, shots: str):
    correct = 0
    rows = []
    for q, ans in problems:
        prompt = shots + f"Q: {q}\nA:"
        gen = generate_answer(prompt).strip()
        pred = extract_first_int(gen)
        ok = (pred == ans)
        correct += int(ok)
        rows.append({"question": q, "generated": gen[:30], "pred": pred, "answer": ans, "ok": ok})
    return correct / len(problems), pd.DataFrame(rows)


acc_gen, gen_df = eval_generation(ARITHMETIC, FEWSHOT)
print(gen_df.to_string(index=False))
print(f"\nfew-shot 산술 정확도 : {acc_gen:.3f}  (n={len(ARITHMETIC)})")
```

**위 코드 읽기** — `eval_generation` 은 `shots + "Q: ...\nA:"` 로 프롬프트를 만들어 생성 → 추출 → 정답 비교를 반복하고 정확도를 냅니다. 여기서는 `FEWSHOT` 을 넣어 호출하므로 *2-shot* 정확도가 나오며, 문항별 생성 결과·추출값을 표로 함께 보여 줍니다.

**▶ 실행 결과**

```text
         question                        generated  pred  answer   ok
  6 곱하기 4는 얼마인가요? 정답은 24입니다. \n\n이런 문제들은 어떤 숫자를 곱하    24      24 True
 15 더하기 9는 얼마인가요? 정답은 24입니다. \n\n이런 문제들은 어떤 숫자를 더하    24      24 True
  20 빼기 8은 얼마인가요? 정답은 12입니다. \n\n이런 문제들은 대부분의 경우,     12      12 True
  7 곱하기 7은 얼마인가요? 정답은 49입니다. \n\n이런 문제들은 어떤 숫자를 더하    49      49 True
100 빼기 37은 얼마인가요? 정답은 63입니다. \n\n이런 문제들은 대부분 숫자의 차    63      63 True
13 더하기 28은 얼마인가요? 정답은 41입니다. \n\n이런 문제들은 어떤 숫자를 더하    41      41 True

few-shot 산술 정확도 : 1.000  (n=6)
```

**결과 해석**

6문항 모두 `정답은 N입니다.` 형식으로 생성돼 숫자 추출이 깔끔히 성공, 정확도 1.000 입니다. 예시 2개가 *답변 형식* 을 정렬해 준 덕분으로, 생성 뒤 군더더기 (`\n\n이런 문제들은...`) 가 붙어도 *첫 정수만* 뽑는 추출기가 정답을 안정적으로 잡아냅니다.

> **생성 평가의 어려움**: 모델이 정답 숫자를 *맞게 말해도* 형식이 다르면 (`24` vs `이십사` vs 본문에 다른 숫자가 먼저 등장) 정규식 추출이 어긋날 수 있습니다. GSM8K·HumanEval 의 공식 평가가 *정교한 파싱 / 코드 실행* 을 쓰는 이유입니다. MC (§2) 가 *형식에 안 흔들리는* 것과 대조됩니다.

## zero-shot vs few-shot — in-context learning

같은 산술 task 를 **예시 없이 (zero-shot)** 과 **예시 2개 (2-shot)** 으로 평가해 점수 차이를 봅니다. 모델 가중치는 *전혀 바뀌지 않는데* (학습이 아닙니다), 프롬프트에 예시를 넣는 것만으로 성능이 달라지는 현상이 **in-context learning** 입니다.

- **zero-shot**: 예시 없이 문제만. 모델이 *답변 형식* 을 스스로 정해야 함 → 추출 실패 잦음
- **few-shot**: 예시가 *형식 (정답은 N입니다)* 을 보여줘 모델이 그대로 따라 함 → 추출 성공률·정확도 상승

같은 산술 task 를 *예시 없이* (zero-shot) 다시 평가해, §3 의 few-shot (2-shot) 정확도와 나란히 비교합니다. 모델 가중치는 전혀 바뀌지 않았는데 프롬프트 속 예시만으로 점수가 달라지는 *in-context learning* 효과를 정량으로 봅니다.

```python
# zero-shot (예시 없음) - 형식 유도가 없어 더 어려움
acc_zero, zero_df = eval_generation(ARITHMETIC, shots="")
# few-shot (위에서 정의한 FEWSHOT 2개)
acc_few = acc_gen  # §3 에서 이미 계산

compare = pd.DataFrame({
    "setting": ["zero-shot (0 examples)", "few-shot (2 examples)"],
    "accuracy": [round(acc_zero, 3), round(acc_few, 3)],
})
print(compare.to_string(index=False))
print(f"\nin-context learning 효과 : {acc_few - acc_zero:+.3f}  (few - zero)")
print("(작은 모델·작은 subset 이라 변동 큼 - 경향만 참고. 큰 모델일수록 효과 뚜렷)")
```

**▶ 실행 결과**

```text
               setting  accuracy
zero-shot (0 examples)     0.333
 few-shot (2 examples)     1.000

in-context learning 효과 : +0.667  (few - zero)
(작은 모델·작은 subset 이라 변동 큼 - 경향만 참고. 큰 모델일수록 효과 뚜렷)
```

**결과 해석**

zero-shot 0.333 → few-shot 1.000 으로 +0.667 의 큰 폭 상승입니다. 예시 없이는 모델이 답변 형식을 못 맞춰 추출이 자주 실패한 반면, 예시 2개가 형식을 정렬하자 정확도가 급등 — *지식이 아니라 형식 정렬* 이 점수를 올린다는 in-context learning 의 핵심이 드러납니다 (단, n=6 으로 작아 경향으로만 해석).

> few-shot 이 점수를 올리는 핵심은 *지식을 새로 가르치는 게 아니라* **답변 형식을 정렬** 시키는 데 있습니다. MMLU 같은 벤치마크가 보통 *5-shot* 으로 보고되는 이유 — 모델이 *객관식 답 형식* 에 적응하도록.

## `lm-evaluation-harness` 소개 — 표준 도구

§2-§4 에서 직접 구현한 것과 **정확히 같은 원리** 를, EleutherAI 의 **`lm-evaluation-harness`** (`lm-eval` 패키지) 가 *표준화된 방식* 으로 수행합니다. 수백 개 벤치마크 (MMLU, HellaSwag, GSM8K, KoBEST, KMMLU ...) 가 *task 정의로 내장* 되어, *프롬프트 포맷 · few-shot 예시 선택 · log-likelihood 계산 · 정규식 추출* 이 모두 통일됩니다. 논문·리더보드의 점수가 이 도구로 측정됩니다.

### 직접 구현 (§2) 과의 관계

| | 직접 구현 (§2) | `lm-eval-harness` |
|---|---|---|
| MC log-likelihood | `continuation_logprob` 손으로 | 내부에서 동일 계산 (`loglikelihood`) |
| 프롬프트 포맷 | 우리가 문자열 조립 | task yaml 에 정의 |
| few-shot 선택 | 우리가 고정 | seed 기반 자동 샘플링 |
| 결과 | `acc` 하나 | `acc` + `acc_norm` + stderr |

> *원리는 같고, 표준화·재현성이 다릅니다.* 직접 구현으로 *무슨 일이 일어나는지* 를 이해한 뒤, 실제 보고용 점수는 `lm-eval` 로 내는 것이 일반적인 흐름입니다.

### `lm_eval.simple_evaluate` API (실행은 선택)

`lm-eval` 의 파이썬 API 핵심은 `simple_evaluate` 한 함수입니다. 아래는 *KoBEST BoolQ* 한 task 를 우리 Qwen 모델로 평가하는 코드입니다. **`lm-eval` 은 버전마다 task 이름·인자가 달라질 수 있어**, 무거우면 실행을 건너뛰고 *사용법* 만 익혀도 됩니다 (직접 구현 §2 가 메인). 셀 상단에서 설치 여부를 확인하고, 설치돼 있을 때만 실행합니다.

표준 도구 `lm-eval` 을 쓰기 전에, 설치 여부부터 `try / except ImportError` 로 확인합니다. 결과를 `HAS_LM_EVAL` 플래그에 담아, 미설치 환경에서도 §2-§4 직접 구현은 그대로 돌아가도록 다음 셀이 이 플래그로 분기합니다.

```python
# lm-eval 실행은 선택 - 미설치거나 무거우면 건너뜀 (직접 구현 §2 가 메인)
try:
    import lm_eval
    print(f"lm-eval version : {lm_eval.__version__}")
    HAS_LM_EVAL = True
except ImportError:
    print("lm-eval 미설치 - 이 셀은 건너뜁니다 (셋업 셀의 pip install lm-eval 참고).")
    print("§2-§4 의 직접 구현은 lm-eval 없이도 모두 동작합니다.")
    HAS_LM_EVAL = False
```

**▶ 실행 결과**

```text
lm-eval version : 0.4.12
```

표준 도구 `lm-eval-harness` 로 *같은 KoBEST BoolQ* 를 평가해, §2 의 직접 구현과 같은 원리 위에 있음을 확인합니다. 이미 로드한 `model`/`tokenizer` 를 `HFLM` 으로 래핑해 중복 로드를 피하고, `simple_evaluate` 한 함수로 프롬프트 포맷·log-likelihood·집계를 자동 처리합니다 (subset `limit=50`).

```python
# lm-eval 표준 도구로 한 task 평가 (설치돼 있을 때만)
# API 는 버전 변동이 큽니다 - 설치된 버전 기준으로 task 이름이 다르면 lm_eval.tasks 로 확인하세요.
if HAS_LM_EVAL:
    from lm_eval.models.huggingface import HFLM

    # 이미 로드한 model/tokenizer 를 그대로 lm-eval 에 래핑 (중복 로드 방지)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["kobest_boolq"],   # 버전에 따라 "kobest_boolq" / "kobest" 등 - 미존재 시 except 로
        num_fewshot=0,
        limit=50,                 # subset - T4 30분 룰
    )
    # 결과 표 출력
    for task, metrics in results["results"].items():
        print(f"[{task}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:14s}: {v:.3f}")
else:
    print("lm-eval 미설치 - §2 의 직접 구현 결과를 표준 점수로 참고하세요.")
```

**▶ 실행 결과**

```text
WARNING:lm_eval.models.huggingface:`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not l …(뒤 85자 생략)
WARNING:lm_eval.models.huggingface:Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or c …(뒤 29자 생략)
Downloading builder script: 0.00B [00:00, ?B/s]
WARNING:lm_eval.evaluator:Overwriting default num_fewshot of kobest_boolq from None to 0
[kobest_boolq]
  acc,none      : 0.560
  acc_stderr,none: 0.071
  f1,none       : 0.359
```

**결과 해석**

`lm-eval` 의 BoolQ `acc` 0.560 은 §2 직접 구현 (0.460) 과 정확히 같지는 않은데, 이는 *프롬프트 포맷·정답 표현이 task yaml 정의를 따르기* 때문입니다 — 점수의 절대값이 아니라 *둘이 같은 log-likelihood 원리* 위에 있다는 점이 핵심입니다. `acc_stderr` (0.071) 와 `f1` 까지 함께 내주는 것이 표준 도구의 표준화·재현성 이점입니다.

> 위 셀이 버전 문제로 실패하면 (`task 'kobest_boolq' not found` 등), `lm_eval.tasks.TaskManager().all_tasks` 로 *설치된 버전의 task 이름* 을 확인해 바꾸면 됩니다. **핵심은 점수 자체가 아니라, §2 의 직접 구현과 `lm-eval` 이 *같은 log-likelihood 원리* 위에 있다는 것** 입니다.

## 분야별 벤치마크 지도 — 무엇을 측정하나

벤치마크는 *측정하는 능력* 에 따라 분류됩니다. 하나의 점수가 아니라 *여러 능력* 을 *각기 다른 벤치마크* 로 재는 것이 현대 LLM 평가입니다. 아래는 영어·한국어 대표 벤치마크를 능력별로 정리한 지도입니다.

| 측정 능력 | 영어 벤치마크 | 한국어 벤치마크 | format |
|---|---|---|---|
| **지식** (전문 분야) | MMLU | KMMLU, HAERAE-Bench | ① MC |
| **상식추론** | HellaSwag, ARC | KoBEST (HellaSwag/COPA/BoolQ) | ① MC |
| **수학** | GSM8K, MATH | — (GSM8K 번역본) | ② 생성+추출 |
| **코드** | HumanEval, MBPP | — | ② 생성+실행 |
| **진실성** | TruthfulQA | — | ① MC (+ 생성) |
| **종합 대화/지시** | MT-Bench | LogicKor | ③ LLM-judge |

### format 별 다시 보기

- **① MC** (지식·상식·진실성): §2 에서 직접 구현한 *log-likelihood argmax*. 가장 흔하고 재현성 높음
- **② 생성+추출** (수학·코드): §3 의 *생성 후 파싱/실행*. 채점이 까다롭지만 *실제 풀이 능력* 측정
- **③ LLM-judge** (대화·지시): 다른 강한 LLM (예: GPT-4) 이 답변을 1-10 점으로 채점. *주관적 품질* 을 측정 — §7 의 한계 참고

> 한 모델을 *제대로 평가* 하려면 이 표 전체를 가로질러야 합니다 — *지식은 높은데 수학은 약한* 모델이 흔하기 때문입니다. **한 벤치마크 점수만으로 모델을 판단하면 안 되는** 이유입니다.
