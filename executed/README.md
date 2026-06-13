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

### A. 러너 노트북으로 일괄 (GPU 챕터 권장) — `run_on_colab.ipynb`

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
- 러너는 생성기로 다시 만들 수 있습니다:
  `python3 .claude/skills/notebook-to-wikidocs/scripts/make_colab_runner.py`

### B. 수동 1챕터 (가끔 한 챕터만)

1. README.md의 Colab 버튼으로 해당 챕터를 열어 **끝까지 실행**(T4).
2. Colab 메뉴 **파일 > .ipynb 다운로드** (출력이 함께 저장됩니다).
3. 받은 파일을 `executed/<폴더명>.ipynb`로 저장하고 커밋.

## 이 실행본의 쓰임

여기 모인 실행본(`executed/<폴더>.ipynb`)은 이후 **WikiDocs 변환 도구**(별도 PR)가
출력 원천으로 자동으로 집어 씁니다 — 변환된 마크다운에 실제 표·로그·그림을 싣기 위해서입니다.

> 이 PR에는 **러너 도구만** 들어 있습니다(`run_on_colab.ipynb` + 생성기 + 이 문서).
> 실행본(`executed/<폴더>.ipynb`)과 변환 도구는 후속 PR로 추가됩니다.
