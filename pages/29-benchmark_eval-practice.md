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

```python
%pip install -q -U datasets transformers accelerate
# lm-eval 은 §5 (표준 도구 소개) 에서만 사용. 설치가 무거우면 이 줄만 주석 처리하세요.
%pip install -q lm-eval
```

**▶ 실행 결과**

```text
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 555.1/555.1 kB 19.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.2/11.2 MB 117.9 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 389.2/389.2 kB 37.2 MB/s eta 0:00:00
   ━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.2/48.9 MB 216.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━ 46.2/48.9 MB 180.1 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 170.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 48.9/48.9 MB 170.0 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.9/48.9 MB 17.1 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.9/58.9 kB 3.0 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
  Preparing metadata (setup.py) ... done
  Preparing metadata (setup.py) ... done
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 8.9/8.9 MB 132.9 MB/s eta 0:00:01
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.9/8.9 MB 86.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.1/84.1 kB 9.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.8/100.8 kB 11.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 91.1/91.1 kB 10.0 MB/s eta 0:00:00
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
model     : Qwen/Qwen2.5-0.5B-Instruct
params    : 494.0M
vocab     : 151643
eos token : '<|im_end|>'
```

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

정답 `2입니다.` 의 log-prob 가 sum (-5.01), mean (-1.670) 모두에서 가장 높아 두 방식 다 정답을 골랐습니다. 오답일수록 (`5입니다.`, `10입니다.`) log-prob 가 더 낮아져, 함수가 *모델의 확신* 을 제대로 수치화함을 확인할 수 있습니다.

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
