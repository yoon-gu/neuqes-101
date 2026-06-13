#!/usr/bin/env python3
"""executed/run_on_colab.ipynb 생성기.

Colab T4에서 여는 '실행 결과 러너' 노트북을 만든다. 러너는 각 챕터 clean 노트북을
nbclient로 끝까지 실행해 출력이 포함된 executed/<폴더>.ipynb 를 만들고, 포크 master 로
직접 커밋·푸시한다(고민 8, 결정 #8=A). 멱등: clean 노트북 해시가 그대로면 건너뛴다.

사용:
  python3 .claude/skills/notebook-to-wikidocs/scripts/make_colab_runner.py
  # → executed/run_on_colab.ipynb 재생성
"""
from __future__ import annotations

import json
from pathlib import Path

# 레포 루트 = 이 파일 기준 ../../../../ (.claude/skills/notebook-to-wikidocs/scripts)
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "executed" / "run_on_colab.ipynb"


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


MD_INTRO = """\
# 📥 Colab 실행 결과 러너 (`executed/` 생산)

이 노트북을 **Colab T4** 에서 열어, 각 챕터 노트북을 끝까지 실행하고
**출력이 포함된 `executed/<폴더>.ipynb`** 를 **본인 fork** `master` 로 커밋·푸시합니다.
(원본을 그대로 열어도 됩니다 — 아래 *설정* 셀의 `REPO` 에 **본인 fork** 만 지정하면 됩니다.)

- 변환기(`build_wikidocs.py`)가 `executed/<폴더>.ipynb` 가 있으면 **자동으로** 실제 출력 원천
  (`▶ 실행 결과`)으로 씁니다. 없으면 합성(`▶ 출력 형태`)으로 폴백합니다.
- **멱등·재개**: clean 노트북이 안 바뀌었으면(해시 동일) 건너뜁니다. 없거나 바뀐 챕터만 실행.
  Colab 세션 한계(아이들 끊김·최대 ~12h) 때문에 **여러 번 나눠 돌려도 이어집니다.**
- tex 의 결괏값은 신뢰하지 않습니다 — **executed/ 만 canonical** 입니다(고민 2·8).

**사용 순서**: ① 아래 *설정* 셀에서 대상 챕터·토큰 지정 → ② 위에서부터 전부 실행(`런타임 > 모두 실행`).
런타임 유형이 **T4 GPU** 인지 먼저 확인하세요(`런타임 > 런타임 유형 변경`).
"""

MD_TOKEN = """\
## 🔑 GitHub 토큰(PAT) 발급 방법

아래 **설정 셀**을 실행하면 `GitHub PAT ...:` 입력창이 뜹니다. 결과(`executed/`)를 포크에
push 하려면 토큰이 필요해요. **레포 하나에만 권한을 주는 Fine-grained PAT** 를 권장합니다.

> getpass 입력창이라 **붙여넣어도 화면에 안 보이는 게 정상**입니다. 토큰은 저장·출력되지 않습니다.

**발급 순서 (Fine-grained, 권장)**

1. 바로가기 → **https://github.com/settings/personal-access-tokens/new**
   (메뉴: GitHub 우상단 프로필 → *Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token*)
2. **Token name**: 아무거나 (예: `neuqes-executed-runner`)
3. **Expiration**: 7일 또는 30일 (짧게)
4. **Resource owner**: 본인 계정 (위 `REPO` 의 fork 를 소유한 계정)
5. **Repository access** → **Only select repositories** → **본인 fork(`neuqes-101`)** 선택
6. **Permissions** → **Repository permissions** → **Contents** 를 **Read and write** 로
   (이걸 켜면 `Metadata: Read-only` 가 자동 포함됩니다 — 그대로 두세요)
7. 맨 아래 **Generate token** → 나오는 `github_pat_...` 문자열 **복사**
8. 설정 셀의 입력창에 **붙여넣고 Enter**

> 토큰은 페이지를 떠나면 다시 못 보니 그 자리에서 복사하세요. 만료/분실 시 새로 발급하면 됩니다.

**대안 (Classic PAT)** — 더 간단하지만 계정 전체 레포에 권한이 생깁니다:
**https://github.com/settings/tokens** → *Generate new token (classic)* → scope **`repo`** 체크 → 생성.
"""

CODE_SETUP = """\
# 1) 의존성 설치 + 본인 fork 클론
import os

# ▶ 본인 GitHub fork 를 지정하세요. executed/ 결과를 여기 master 로 push 합니다.
#   원본(upstream)에는 push 권한이 없으니 **반드시 본인이 fork 한 레포**여야 합니다.
#   fork 가 없다면: 원본 레포 페이지 우상단 'Fork' → 본인 계정에 복제한 뒤 그 이름을 넣으세요.
REPO   = ""               # 예: "your-username/neuqes-101"
BRANCH = "master"

assert REPO and "/" in REPO, \\
    "REPO 를 본인 fork 로 설정하세요 — 예: your-username/neuqes-101 (원본이 아니라 본인 fork)"

WORK   = "/content/" + REPO.split("/")[-1]

get_ipython().system('pip -q install nbclient nbformat')

if not os.path.isdir(WORK):
    get_ipython().system(f'git clone -q https://github.com/{REPO}.git {WORK}')
get_ipython().run_line_magic('cd', WORK)
get_ipython().system(f'git checkout -q {BRANCH} && git pull -q')

print("GPU 확인:")
get_ipython().system('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️ GPU 없음 → 런타임 유형을 T4 로 변경하세요"')
"""

CODE_CONFIG = """\
# 2) 설정 — 대상 챕터 + GitHub 토큰
#   TARGET: "stale"  → 없거나 바뀐 챕터만 (기본, 권장)
#           "all"    → 전 챕터
#           "gpu"    → 07 이상(GPU) 전부
#           리스트    → 예: [1, 7, "24_gpt_tinystories"]  (번호 또는 폴더명)
TARGET = "stale"
FORCE  = False             # True 면 해시가 같아도 재실행
PER_CELL_TIMEOUT = 60 * 60 # 셀당 최대 실행 시간(초). GPU 학습 챕터 여유 있게.

# 커밋 작성자 — 원하면 본인 것으로 바꾸세요(아무 값이나 가능, 토큰이 push 권한을 줍니다).
GIT_NAME  = "colab-runner"
GIT_EMAIL = "colab-runner@users.noreply.github.com"

from getpass import getpass
# contents:write 권한의 fine-grained PAT 권장. 입력값은 저장/출력되지 않습니다.
GH_TOKEN = getpass(f"GitHub PAT ({REPO}, contents:write): ").strip()
"""

CODE_HELPERS = """\
# 3) 챕터 탐색 + 해시(멱등 판단)
import hashlib, datetime
from pathlib import Path
import nbformat

ROOT = Path(WORK)
EXEC = ROOT / "executed"; EXEC.mkdir(exist_ok=True)

def chapters():
    out = []
    for d in sorted(ROOT.glob("[0-9][0-9]_*")):
        nb = d / (d.name + ".ipynb")
        if nb.exists():
            out.append((d.name, nb))
    return out

def source_hash(nb_path):
    \"\"\"clean 노트북의 셀 소스(출력 제외) 해시 — 내용이 바뀌면 달라진다.\"\"\"
    nb = nbformat.read(nb_path, as_version=4)
    h = hashlib.sha256()
    for c in nb.cells:
        h.update(c.cell_type.encode()); h.update(b"\\0")
        h.update((c.source or "").encode()); h.update(b"\\0")
    return h.hexdigest()

def executed_hash(folder):
    p = EXEC / (folder + ".ipynb")
    if not p.exists():
        return None
    try:
        nb = nbformat.read(p, as_version=4)
        return nb.metadata.get("executed_from", {}).get("source_sha256")
    except Exception:
        return None

def is_stale(folder, nb_path):
    return source_hash(nb_path) != executed_hash(folder)

ALL = chapters()
print("발견한 챕터:", len(ALL))
"""

CODE_SELECT = """\
# 4) 대상 결정 + 현황표
def base_set(t):
    if t == "all":
        return ALL
    if t == "gpu":
        return [(f, p) for f, p in ALL if int(f[:2]) >= 7]
    if t == "stale":
        return ALL                      # staleness 필터는 아래에서
    if isinstance(t, list):
        keys = {str(x).zfill(2) if str(x).isdigit() else str(x) for x in t}
        return [(f, p) for f, p in ALL if f in keys or f[:2] in keys]
    return []

base = base_set(TARGET)
sel  = base if FORCE else [(f, p) for f, p in base if is_stale(f, p)]
sel_keys = {f for f, _ in sel}

print(f"{'':2}{'챕터':<28}{'executed':<10}상태")
for f, p in ALL:
    has = (EXEC / (f + '.ipynb')).exists()
    state = '최신' if (has and not is_stale(f, p)) else ('낡음' if has else '없음')
    mark = '▶' if f in sel_keys else ' '
    print(f"{mark} {f:<28}{'있음' if has else '-':<10}{state}")
print(f"\\n실행 대상: {len(sel)}개  (TARGET={TARGET!r}, FORCE={FORCE})")
"""

CODE_EXECUTE = """\
# 5) 실행 → executed/<폴더>.ipynb 저장 (실패해도 다음 챕터 계속)
import time
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

def fmt_dur(sec):
    \"\"\"초 → '3분 5초' / '42초' 형태.\"\"\"
    m, s = divmod(int(round(sec)), 60)
    return f"{m}분 {s}초" if m else f"{s}초"

manifest = []                 # (폴더, 상태, 소요초)
total_t0 = time.time()
for f, p in sel:
    print(f"\\n=== 실행: {f} ===", flush=True)
    nb = nbformat.read(p, as_version=4)
    client = NotebookClient(
        nb, timeout=PER_CELL_TIMEOUT, kernel_name="python3",
        resources={"metadata": {"path": str(p.parent)}},  # 챕터 폴더에서 실행
        allow_errors=False,
    )
    status = "ok"
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as e:
        status = "error: " + str(e).splitlines()[-1][:120]
        print("  ⚠️", status)
    except Exception as e:  # 커널 타임아웃 등
        status = "fail: " + str(e)[:120]
        print("  ⚠️", status)
    elapsed = time.time() - t0
    nb.metadata["executed_from"] = {
        "source_sha256": source_hash(p),
        "executed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "runtime": "colab-t4",
        "status": status,
        "elapsed_sec": round(elapsed, 1),
    }
    out = EXEC / (f + ".ipynb")
    nbformat.write(nb, out)
    manifest.append((f, status, elapsed))
    print(f"  → 저장 executed/{f}.ipynb  [{status}]  ⏱ {fmt_dur(elapsed)}")

total_elapsed = time.time() - total_t0
print("\\n=== 요약 (챕터별 소요 시간) ===")
for f, s, dt in manifest:
    print(f"  {f:<28}{fmt_dur(dt):>10}   {s}")
print(f"  {'-'*28}{'-'*10}")
print(f"  {'합계 (' + str(len(manifest)) + '개 챕터)':<28}{fmt_dur(total_elapsed):>10}")
"""

CODE_PUSH = """\
# 6) executed/ 만 커밋·푸시 (성공한 것만 올리고 싶으면 manifest 보고 위에서 거른 뒤 재실행)
import subprocess

subprocess.run(["git", "config", "user.name",  GIT_NAME], cwd=WORK, check=True)
subprocess.run(["git", "config", "user.email", GIT_EMAIL], cwd=WORK, check=True)
subprocess.run(["git", "add", "executed/"], cwd=WORK, check=True)

staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                        cwd=WORK, capture_output=True, text=True).stdout.strip()
if not staged:
    print("커밋할 executed/ 변경이 없습니다.")
else:
    print("커밋 대상:\\n" + staged)
    names = ", ".join(m[0] for m in manifest) or "executed"
    msg = f"executed: Colab 실행본 갱신 ({names})"
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=WORK, check=True)
    push_url = f"https://{GH_TOKEN}@github.com/{REPO}.git"
    r = subprocess.run(["git", "push", "-q", push_url, f"HEAD:{BRANCH}"],
                       cwd=WORK, capture_output=True, text=True)
    print("push:", "✅ 성공" if r.returncode == 0 else "❌ 실패\\n" + r.stderr)
"""


def build() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            _md(MD_INTRO),
            _code(CODE_SETUP),
            _md(MD_TOKEN),
            _code(CODE_CONFIG),
            _code(CODE_HELPERS),
            _code(CODE_SELECT),
            _code(CODE_EXECUTE),
            _code(CODE_PUSH),
        ],
    }


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
