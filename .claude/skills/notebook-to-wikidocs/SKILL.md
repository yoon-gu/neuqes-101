---
name: notebook-to-wikidocs
description: 챕터 노트북(.ipynb)을 WikiDocs용 장→절 마크다운으로 변환한다. 코드와 함께 실제 실행 결과(표·로그·그림)를 싣고, 웹·PDF·EPUB(전자책) 어디서도 깨지지 않게 만든다.
argument-hint: "[챕터] 예: 7 · 07_bert_pipeline · 7 24 · --all"
disable-model-invocation: true
---

# notebook → WikiDocs 변환

챕터 노트북을 WikiDocs 연동용 `pages/NN-slug*.md`(장 1 + 절 여러 개)로 바꾼다.

**호출**: 사용자가 **변환할 챕터를 인자로** 주며 직접 호출한다(모델이 자동 호출하지 않음).
```
/notebook-to-wikidocs 7                # 7장
/notebook-to-wikidocs 7 24             # 여러 장
/notebook-to-wikidocs 07_bert_pipeline # 폴더명으로도 가능
/notebook-to-wikidocs --all            # 전체 (사용자 확인 후)
```

**핵심 원칙**
- 코드를 실으면 그 코드의 **실제 실행 결과**도 함께 싣는다 — 가짜 출력을 지어내지 않는다.
- 같은 `.md`가 **웹(WikiDocs)·PDF·EPUB 세 타깃** 어디서도 깨지지 않게 한다(서점 판매엔 EPUB 필수).

설계 결정·실측 기록은 같은 폴더의 `DESIGN_NOTES.md` 참조.

## 파이프라인

`① 실행 결과 확보(executed/) → ② 변환(build_wikidocs.py) → ③ 검증(check_wikidocs_md.py) → ④ 결과 해석 덧붙이기`

### ① 실행 결과 확보 — `executed/<폴더>.ipynb`

챕터의 진짜 출력은 **Colab에서 실행해야** 나온다. 러너 **`executed/run_on_colab.ipynb` 를 Colab T4에서 열어**
대상 챕터를 끝까지 실행 → 출력 포함 `executed/<폴더>.ipynb` 를 **본인 fork**로 커밋·푸시한다.
**이 러너 실행은 사람이 한다 — Claude(스킬)가 대신 돌리지 않는다.**

- **멱등·재개**: clean 노트북 해시를 실행본에 도장 → 안 바뀐 챕터는 skip, 세션이 끊겨도 이어 채운다. 챕터별·총 소요시간 출력.
- **repo-agnostic**: 설정 셀 `REPO`에 본인 fork만 지정(원본 push 권한 불필요). PAT은 `getpass`(미저장).

**변환 전 반드시 확인**: 변환할 챕터의 `executed/<폴더>.ipynb` 가 있는지 먼저 본다.
**없으면 합성으로 조용히 넘어가지 말고, 사용자에게 알리고 멈춘다** — `executed/README.md` 를 참고해
**먼저 Colab에서 러너로 실행해 `executed/<폴더>.ipynb` 를 `executed/` 아래에 만들어 두라**고 안내한다.

### ② 변환 — `scripts/build_wikidocs.py`

챕터 인자: 번호(`7`/`07`)·폴더명(`07_bert_pipeline`), 여러 개 가능. 전체는 `--all`
(**대량 실행이라 사용자 확인 후**). 제목·장→절 분할은 자동.

```bash
python3 .claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py 7 24    # executed/<폴더>.ipynb 자동 사용
python3 .claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py --all   # 전체 (사용자 확인 후)
```

**출력 원천 우선순위**(셀마다 자동): `--executed-notebook` > `executed/<폴더>.ipynb` > `--execute` > 합성.
실제 결과는 `▶ 실행 결과`, 합성 골격(값 `...`)은 `▶ 출력 형태`로 라벨을 구분한다. 합성 로직은 tex와 단일 출처를 공유한다(재구현 금지 — import해서 사용).

**전자책 안전 출력**(자동) — [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723) 기준:
- 출력 기본 스타일 `code` — HTML `<pre>` 박스는 PDF에서 전멸하고 EPUB XML이 깨져 회피.
- 규칙 자동 방어: 헤딩 위아래 빈 줄 삽입, 수평선 제거, 본문 H1→H2 강등, 각주 이름 유니크화, 윈도우 경로(`C:\`)→인라인 코드.
- 노이즈 필터(HF Hub 인증·`tqdm`·`generation_config` 보일러플레이트) + EPUB 긴 산문 줄 트렁케이트.
  단 **출력이 곧 학습 내용인 토큰화 챕터(08·15·22)는 opt-out**(원본 보존, 표는 항상 보존).

산출물: `pages/NN-slug.md`(개요) + `-{practice,anatomy,variation,wrapup}.md`(절), 그림 `assets/NN-slug-outK.png`,
`TOC.md`(해당 장 블록만 교체, 나머지 번호 순서 보존). 챕터별 실패는 격리되어 배치를 멈추지 않는다.

### ③ 검증 — `scripts/check_wikidocs_md.py`

변환 후 **린터로 전자책 규칙을 전수 검사**한다(위반 0 확인). 변환기가 자동 방어하지만, 변환기와 별개로
**회귀·수기 편집**을 잡는 독립 점검이다(코드펜스 안은 제외).

```bash
python3 .claude/skills/notebook-to-wikidocs/scripts/check_wikidocs_md.py   # pages/*.md 전체. 위반 시 종료코드 1
```

이어서 사람이 확인: 코드 셀에 `▶ 실행 결과`가 붙었는가(`<!-- 실행 결과 없음 -->`이면 출력 없는 셀이 맞는지),
`assets/` PNG 생성·상대경로(`../assets/...`), 첫 H1 제거(페이지 제목은 `TOC.md` 담당) 여부.

### ④ 결과 해석 덧붙이기 (스킬이 직접 작성 — 스크립트 아님)

②변환·③린터를 통과한 뒤, 생성된 `pages/*.md` 의 **의미 있는 실행 결과**(`▶ 실행 결과`) 뒤에 짧은
**결과 해석**을 덧붙인다. 스크립트가 아니라 **스킬 사용 시 직접** 쓴다.

- 근거: 해당 챕터 **노트북의 마크다운 셀 설명 + 실제 출력**. 거기 없는 새 사실을 지어내지 않는다.
- **기존 내용은 삭제·수정하지 않는다** — 출력 블록 뒤에 머릿말 `**결과 해석**` 을 붙여 **추가만** 한다.
- **형식 고정**: `**결과 해석**` 을 **단독 줄**로 두고, 위·아래 빈 줄 + 그다음 줄부터 본문.
  같은 줄에 본문을 붙이지 않는다(`**결과 해석** 본문 …` 금지 — 전자책 렌더·일관성 때문). 출력 블록의 닫는 ``` 바로 다음에 빈 줄 → 헤더 순서.
- 대상: 해석이 도움 되는 출력(학습 지표·분류 결과·표·생성 샘플 등). import 로그 등 사소한 출력은 건너뛴다.
  **개요(`NN-slug.md`)·FAQ뿐인 wrapup 절은 보통 `▶ 실행 결과`가 없어 대상이 아니다** — 실제 출력이 실린 practice·anatomy·variation 절이 주 대상.
- 분량·톤: **1~2문장**으로 간략히 — 그 출력이 무엇을 보여주는지/왜 그런지를, **챕터 노트북 문체 + 전자책(존댓말·간결)** 에 맞춘다.
- **노트북 산문의 추정치(예상 소요시간·예상 loss)와 실행본 실제값이 다르면 실제값을 기준으로 해석**하고(가짜 출력 금지 원칙), 그 불일치는 **사용자에게 보고**한다 — 실행본이 의도와 다른 설정으로 돌았을 수 있다.
- 챕터를 병렬 서브에이전트로 나눠 작성할 때도 **위 형식 규칙을 각 에이전트에 명시**한다(헤더 인라인 등 형식이 갈리지 않게).

## 주의

- `book/chapters/*.tex`(인쇄책용)는 건드리지 않는다(회귀 방지) — 출력 렌더 로직만 tex→md로 포팅.
- 한 번에 한 챕터씩, 검증 안 된 챕터를 두고 다음으로 넘어가지 않는다(CLAUDE.md 워크플로).
- 챕터 단위로 의미 있게 커밋. WikiDocs 출판 워크플로는 메모리 `wikidocs-publishing` 참조.
