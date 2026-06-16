# executed/ — 실행본 노트북 보관소

WikiDocs 변환 시 **코드의 실제 실행 결과**(표·로그·그림)를 싣기 위한 출력 원천입니다.

## 왜 필요한가

챕터 폴더의 `NN_slug/NN_slug.ipynb`는 **출력이 없는 clean 상태**입니다(Colab에서 학습자가
직접 실행하라고 비워둠). 그래서 그대로 마크다운으로 바꾸면 결과가 비어 보입니다.
특히 GPU 챕터(7–31)는 로컬에서 돌릴 수 없어, **실제 결과의 출처가 이 폴더뿐**입니다.

## 규약

- 파일명: `executed/<폴더명>.ipynb` (예: `executed/24_gpt_tinystories.ipynb`).
- 내용: 해당 챕터를 **Colab T4에서 끝까지 실행**한 뒤 출력이 포함된 노트북.
- 챕터 폴더에는 **clean 노트북만** 둡니다(README의 Colab 버튼 대상). 실행본은 여기로 분리.

## 만드는 법

> **권장 경로는 C(colab-cli)** — macOS/Linux 면 터미널 한 줄로 자동화되고 PAT 도 필요 없습니다.
> 아래 **A(브라우저 러너)** 는 **폴백**입니다: Windows, 무료 CLI 세션 **~11분 캡 초과 챕터**(이 커리큘럼 25·27),
> CLI 인증을 못 쓰는 경우에 씁니다. (B 는 한 챕터만 손으로 받을 때.)

### A. 러너 노트북으로 일괄 (폴백 — Windows·캡초과·인증불가용) — `run_on_colab.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/executed/run_on_colab.ipynb)

> 🔧 **아직 upstream PR 머지 전이라면** 위 배지(정본: `yoon-gu/master`)는 파일이 없어 안 열릴 수 있습니다.
> 그동안은 **fork 브랜치 사본**으로 여세요 → **[Colab에서 열기](https://colab.research.google.com/github/fluentmin/neuqes-101/blob/feat/notebook-to-wikidocs/executed/run_on_colab.ipynb)**
> 안 열리면 Colab → *파일 > 노트북 열기 > GitHub* 에서 `fluentmin/neuqes-101` → 브랜치 `feat/notebook-to-wikidocs` 선택.
> _(머지되면 배지가 살아나므로 이 임시 안내는 제거합니다.)_

위 배지로 **`run_on_colab.ipynb` 를 Colab T4 에서 열어** 위에서부터 실행하면, 선택한 챕터를
끝까지 돌려 `executed/<폴더>.ipynb` 를 만들고 **본인 fork** `master` 로 **직접 커밋·푸시**합니다.
(배지는 정본 러너를 엽니다 — 설정 셀의 `REPO` 에 **본인 fork** 만 적으면 누구나 그대로 사용.)

- 설정 셀의 `TARGET`: `"stale"`(없거나 바뀐 것만, 기본) · `"all"` · `"gpu"`(07+) · 리스트(`[1, 7, 24]`).
- **멱등·재개**: clean 노트북 해시를 executed 메타데이터에 심어, 안 바뀐 챕터는 건너뜁니다.
  Colab 세션 한계(아이들·~12h) 때문에 **여러 번 나눠 돌려도 이어집니다.**
- 푸시에는 `contents:write` 권한의 GitHub PAT 가 필요(getpass 입력, 저장 안 됨).

### B. 수동 1챕터 (가끔 한 챕터만)

1. README.md의 Colab 버튼으로 해당 챕터를 열어 **끝까지 실행**(T4).
2. Colab 메뉴 **파일 > .ipynb 다운로드** (출력이 함께 저장됩니다).
3. 받은 파일을 `executed/<폴더명>.ipynb`로 저장하고 커밋.

### C. CLI 자동 — 전체/일부 (로컬, macOS/Linux) — `run_via_cli.sh`

브라우저에서 "모두 실행"을 손으로 누르는 A 대신, **로컬 터미널 한 줄**로 T4 VM 할당 →
챕터 실행 → 결과 회수까지 자동화합니다. [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli)
를 씁니다. 결과를 로컬 `executed/` 로 내려받으므로 **GitHub PAT 가 필요 없습니다**(평소 git 으로 커밋).

```bash
# 사전 1회: 설치 + 인증
uv tool install "git+https://github.com/googlecolab/google-colab-cli"   # issue #14(keep-alive) 수정본 필요
colab --auth=oauth2 whoami                 # 브라우저 동의에서 "모두 선택" 체크 후 계속

# 실행 — 인자 없으면 전 챕터, 인자 주면 그 챕터만 (REPO 는 git origin 에서 자동 인식)
./executed/run_via_cli.sh             # 전 챕터
./executed/run_via_cli.sh 7 24        # 일부 (번호 또는 폴더명)
./executed/run_via_cli.sh 07_bert_pipeline
FORCE=1 ./executed/run_via_cli.sh 7   # 로컬에 ok 여도 강제 재실행
# REPO=other/repo ./executed/run_via_cli.sh   # origin 외 다른 저장소를 쓸 때만
```

- **인자 없음 → 전 챕터, `7 24` 같은 인자 → 해당 챕터만.** 어느 쪽이든 **챕터마다 새 VM**(격리·resume 단순화)으로
  순차 실행하고, `resume`(이미 ok 면 skip, `FORCE=1` 로 무시)·일시 드롭 1회 재시도를 한다. **전 32챕터 CLI 처리 가능**
  (무거운 25·24분, 27·19분 포함 — 실측 완주).
- **⚠️ colab-cli 버전 주의**: 위 git 설치는 **issue #14(keep-alive) 수정본**을 받기 위함이다. PyPI 최신 **v0.5.11 이하**는
  keep-alive RPC 가 일반 계정에 403 → **VM 이 ~11분에 idle-prune** 돼 무거운 챕터(25·27 등)가 실패한다. 그 경우
  업데이트하거나, 부득이하면 `OVER_CAP="25_gpt2_continual_pretrain 27_ko_gpt2_continual_pretrain"` 로 스킵하고 그 둘만 브라우저 러너(A)로 충당한다.
- **무료 Colab 계정도 T4 가 잡힙니다**(실측). 단 무료 GPU 는 가용성·일일 한도가 있어 혼잡 시
  거부될 수 있습니다 — 그 경우 A(브라우저 러너)로 폴백하세요.
- A 와 **같은 멱등·재개**(clean 노트북 해시 비교) 로직을 공유합니다. 실제 실행은 VM 위
  `colab_cli_exec.py`(러너 노트북과 동일 로직)가 하고, 래퍼가 `colab download` 로 회수합니다.
- VM 은 끝나면 **자동 종료**(`trap`)되어 컴퓨트 유닛/시간 낭비를 막습니다.
- **인증 주의**: 이 CLI 는 토큰 갱신 때 6개 스코프 전체를 다시 요구하므로 Colab 하나만 허가하면
  다음 실행에서 `invalid_scope` 로 깨집니다 → 동의 화면에서 **"모두 선택"**. 깨진 토큰이 남으면
  `rm ~/.config/colab-cli/token.json` 후 재인증. 권한이 부담되면 검증 뒤
  [myaccount.google.com/permissions](https://myaccount.google.com/permissions) 에서 회수할 수 있습니다.

> A vs C: **A(노트북)** 는 설치·인증·OS 제약이 없어 누구나 쓰는 기본 경로이고, **C(CLI)** 는
> macOS/Linux 에서 반복 자동화(여러 챕터·재실행)에 편리한 선택지입니다. 둘은 같은 실행 로직을 씁니다.

> **왜 `colab exec -f run_on_colab.ipynb` 로 바로 안 돌리나** (이슈 #17 검토 결과):
> ① `colab exec -f <노트북>` 은 노트북을 **실행만 하고 stdout 을 흘릴 뿐, 출력이 임베드된 결과
> 노트북(예: `*_output.ipynb`)을 만들지 않습니다**(이미지는 `--output-image` 로만 저장). 그래서
> "노트북을 exec 로 돌리면 executed 노트북이 나온다"는 성립하지 않습니다.
> ② `run_on_colab.ipynb` 자체는 `getpass`(PAT)·설정 셀이 있는 **대화형**이라 비대화형 exec 에선 멈춥니다.
> → 그래서 C 는 VM 위에서 **`nbclient` 로 챕터를 실행해 출력을 임베드**한 뒤 `colab download` 로 회수합니다
> (`colab_cli_exec.py`). "노트북을 그대로 exec" 가 아니라 **".py 실행기로 챕터를 돌려 결과를 받아오는"** 방식입니다.

## 이 실행본의 쓰임

여기 모인 실행본(`executed/<폴더>.ipynb`)은 이후 **WikiDocs 변환 도구**(별도 PR)가
출력 원천으로 자동으로 집어 씁니다 — 변환된 마크다운에 실제 표·로그·그림을 싣기 위해서입니다.

> 이 폴더에는 **러너 도구만** 들어 있습니다 — 브라우저 러너 `run_on_colab.ipynb`,
> CLI 러너 `run_via_cli.sh` + VM 실행기 `colab_cli_exec.py`, 그리고 이 문서.
> 실제 실행본(`executed/<폴더>.ipynb`)은 위 도구로 사람이 생성합니다.
