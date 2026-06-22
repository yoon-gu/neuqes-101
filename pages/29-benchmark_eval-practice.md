> ▶ **[Google Colab에서 이 장 실습 열기](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/29_benchmark_eval/29_benchmark_eval.ipynb)** — 브라우저에서 바로 실행해 볼 수 있습니다.

## 환경 준비

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

```python
@torch.no_grad()
def continuation_logprob(prompt: str, continuation: str):
    '''prompt 뒤에 continuation 이 이어질 때, continuation 토큰들의 log-prob 를
    (sum, mean) 으로 반환. teacher forcing - 생성하지 않고 한 번의 forward 로 계산.'''
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids
    # continuation 이 차지하는 토큰 수 (경계는 tokenizer 가 정함)
    cont_len = max(1, full_ids.shape[1] - prompt_ids.shape[1])

    full_ids = full_ids.to(device)
    logits = model(full_ids).logits[0]                 # (T, V)
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # (T, V)

    # 위치 i 의 토큰은 위치 i-1 의 logits 가 예측 -> 한 칸 시프트
    target = full_ids[0, 1:]                            # (T-1,)
    pred_lp = log_probs[:-1]                            # (T-1, V)
    token_lp = pred_lp[torch.arange(target.shape[0]), target]  # (T-1,)

    cont_lp = token_lp[-cont_len:]                     # 마지막 cont_len 개 = continuation
    return cont_lp.sum().item(), cont_lp.mean().item()


def mc_predict(prompt: str, choices: list[str]):
    '''각 선택지의 (sum, mean) log-prob 를 구해 argmax. 두 방식의 예측을 모두 반환.'''
    sums, means = [], []
    for c in choices:
        s, m = continuation_logprob(prompt, c)
        sums.append(s)
        means.append(m)
    return int(np.argmax(sums)), int(np.argmax(means)), sums, means


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
