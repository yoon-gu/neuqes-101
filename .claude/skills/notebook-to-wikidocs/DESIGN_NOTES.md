# notebook → WikiDocs 변환 스킬 — 설계 노트 / 고민 로그

> 사용자가 "고민 생기면 여기 적어두면 틈틈이 보겠다"고 하셔서, 결정·미결 사항을
> 이 파일에 누적합니다. 의사결정이 필요한 항목은 **[결정 필요]** 로 표시합니다.

## 0. 한 줄 요약

기존 노트북(`NN_slug/NN_slug.ipynb`)을 WikiDocs용 장→절 `.md`로 바꾸되,
**코드뿐 아니라 그 코드의 실행 결과(표·로그·그림)까지** 담는 스킬을 만든다.
"파싱만" 하면 안 된다는 것이 사용자의 핵심 요구.

## 1. 레포 현황 (조사 결과)

- 노트북은 **출력이 없는 clean 상태** (Colab에서 학습자가 직접 실행하라고 비워둠).
  → 단순 파싱하면 결과가 안 보임. 이게 메워야 할 간극.
- `book/tools/notebook_to_tex.py` (인쇄책용)는 이미 이 문제를 풀어둠:
  - `--execute` 플래그 → `nbclient.NotebookClient`로 **메모리 실행** 후 출력 캡처.
  - 비실행 시 `synthetic_output_text()`로 **코드만 보고 그럴듯한 가짜 출력** 생성.
  - GPU/학습 챕터는 figure를 `book/tools/generate_book_figures.py`가 만든
    **대표 그림**으로 치환(실제 matplotlib 출력 대신 큐레이션 그림).
  - 출력 렌더링이 견고함: HTML 표→텍스트표, ANSI 제거, pip 노이즈 필터,
    길이 truncation, 에러 트레이스백 처리. (`output_text`, `PandasTableParser`)
- `book/tools/notebook_to_md.py` (WikiDocs용)는 장→절 분할은 잘 하지만,
  출력 없는 노트북을 받으므로 **결과가 비어 있음**. `_render_outputs`는 있으나
  입력 노트북에 출력이 없어 무용지물.
- `pages/`에 ch01만 5개 페이지로 분할되어 있음(개요+실습+해부+변형+정리).
- 루트 `TOC.md`가 WikiDocs 목차의 단일 출처.

## 2. 핵심 결정: "실제 실행 결과"를 어디서 가져오나

세 가지 출처가 가능. 사용자 요구("파싱 금지, 실제 결과")상 ③은 폴백/최후수단.

| 출처 | 장점 | 단점 |
|---|---|---|
| ① 실행된 노트북(outputs 포함)을 입력으로 | 모든 챕터(GPU 포함) 커버, 진짜 결과 | 사용자가 Colab 실행본을 저장·커밋해야 함 |
| ② 스킬이 로컬에서 `--execute` | 자동 | CPU 챕터(1–6,일부 8/19)만. GPU 불가 |
| ③ synthetic(코드 기반 추정) | 의존성 0 | **사용자가 거부한 "파싱만"에 해당** → 기본 비활성 |

> **확정 사실 (조사로 검증).** tex 변환은 **GPU를 쓰지 않는다.** CPU 챕터만
> `--execute`(nbclient)로 실제 실행하고(`출력.` 라벨, 예: Yelp `num_rows 650000`),
> GPU 챕터는 `synthetic_output_text()`가 만든 **수기 스켈레톤**을 쓴다
> (`출력 형태.` 라벨, 실제 숫자 자리에 `...`). 그림도 `generate_book_figures.py`의
> 대표 그림(하드코딩 수치)이다. → **레포에 재활용할 진짜 GPU 출력이 없다.**
> 그러므로 GPU 챕터의 실제 결과는 Colab 실행본 `.ipynb`에서만 얻을 수 있다.

**[결정됨 #1]** GPU 챕터의 실제 출력은 **Colab T4 실행본 .ipynb(출력 포함)** 에서만 얻는다.
레포엔 재활용할 진짜 GPU 출력이 없고(tex의 GPU는 `...` 골격), tex도 출처가 못 되므로 확정.

**[결정됨 #2 — 실행본 보관 위치]** 전용 폴더 **`executed/<폴더>.ipynb`** 에 모아 커밋한다.
(사용자 선택, 2026-06-09) 챕터 폴더에는 clean 노트북만 남겨 Colab 버튼 대상과 분리.
- Colab 검증 시 "파일 > .ipynb 다운로드"(출력 포함) → `executed/<폴더>.ipynb` 로 커밋.
- 스킬 출처 우선순위: `--executed-notebook` → `executed/<폴더>.ipynb` → `--execute` → clean.
- CPU 챕터는 `--execute --save-executed` 로 실행과 동시에 `executed/`를 채울 수 있음.

### tex → md 재사용을 검토했으나 기각 (2026-06-09)
"tex가 이미 출력을 다루니 그걸 md로 쓰자"는 제안 검토. **기각**. 이유:
1. tex의 GPU 출력은 `synthetic_output_text()`의 `...` 골격 → "파싱/합성 금지" 요구 위반.
2. tex는 인쇄 전용 구조(`\inlinecode`, `lstlisting`, `equation`, `faqBox`, `\index`...)라
   md로 되돌리는 **역파싱이 손실적**. 노트북 마크다운 셀은 이미 GFM이라 ipynb→md가 더 깨끗.
3. CLAUDE.md·tex 헤더 모두 "노트북이 단일 출처". tex·md는 서로가 아니라 각자 ipynb에서 파생.

→ **공유해야 할 층은 tex 포맷이 아니라 "실행 결과(executed/)"** 다. md가 먼저 소비하고,
나중에 tex의 synthetic 폴백도 `executed/`를 읽게 바꾸면 tex의 GPU `...`도 진짜 값으로 승급.
실행은 챕터당 1회(Colab), 결과를 tex·md가 공유 → 중복 제거 + 양쪽 모두 진짜 결과. (후속 작업)

## 3. tex 도구와의 의도적 차이

- **그림**: tex는 큐레이션 대표 그림으로 치환. WikiDocs md는 **노트북의 실제
  matplotlib 출력 PNG**를 `assets/`에 저장해 참조(정직·재현가능). 큐레이션 그림을
  쓰고 싶으면 옵션으로 전환 가능하게 둘 수 있음. **[결정 필요 #2]**
- **출력 주석**: tex는 "코드 읽기/출력 읽기" 자동 해설을 붙임. md는 우선 군더더기 없이
  코드 펜스 + `**실행 결과**` 라벨 + 출력 블록만. (원하면 해설도 포팅 가능)
- tex 파일은 **건드리지 않음**(회귀 방지). 출력 렌더링 로직은 tex에서 md로 **포팅**.

## 4. 산출물 형태: 스킬

- `.claude/skills/notebook-to-wikidocs/` 스킬로 제공(슬래시로 호출 가능, 레포에 체크인).
  - `SKILL.md`: 호출 시 따르는 절차(출처 확인→실행/병합→분할→assets→TOC→품질검토).
  - `scripts/build_wikidocs.py`: 실제 변환기(기존 notebook_to_md.py 로직 + 견고한 출력 병합).
- subagent/plugin이 아니라 skill인 이유: 재현 가능한 in-repo 도구 + 호출형 절차가 가장 맞음.

## 5. 품질 기준(스킬이 자체 점검)

- 모든 코드 셀 뒤에 출력 블록이 있거나, "출력 없음"이 의도된 셀인지 판단.
- 이미지가 assets/에 실제로 저장되고 상대경로가 맞는지.
- 장→절 분할이 표준 구조(개요/실습/해부/변형/정리)와 맞는지.
- TOC.md 갱신이 기존 항목과 충돌하지 않는지.

## 6. 미결 질문 모음 (사용자 확인 대기)

- ~~#1 GPU 출력 출처~~ → **결정됨**: Colab 실행본.
- ~~#1b 실행본 보관~~ → **결정됨**: `executed/` 전용 폴더.
- ~~tex→md 재사용~~ → **기각** (§3 사유).
- **#2** 그림: 실제 실행 PNG vs 큐레이션 대표 그림 중 무엇을 기본으로? (현재: 실제 실행 PNG)
- **#3** 첫 PR 범위: 스킬 + 변환기 + ch01 실제 실행 증명. 32챕터 일괄 변환은
  `executed/` 채워지는 대로 후속. (권장)
- **#4** 출력 "해설"(tex의 codeRead류)을 md에도 넣을지, 코드+결과만 둘지. (현재: 코드+결과만)
- **#5 (후속)** tex의 `synthetic_output_text()` 폴백을 `executed/` 소비로 교체 → tex GPU 출력 승급.
- **#6 (신규)** 합성 출력(`▶ 출력 형태`, 값 `...`)을 유료 EPUB에도 허용할지, 판매 챕터는 실제 결과를 강제할지. (→ 고민 7-ⓐ)
- **#7 (신규)** 출력 구분(`<pre>` 색깔박스)·이미지·수식이 PDF/EPUB에서도 유지/렌더되는지 미검증. (→ 고민 7)

### 출력을 코드와 시각적으로 구분 (2026-06-09, PR #1 머지 후) — 2차 시도로 확정
WikiDocs 렌더에서 코드 펜스와 출력 펜스가 같은 회색 박스로 보여 헷갈린다는 피드백.
- **1차(블록인용 + 코드펜스): 실패.** WikiDocs는 블록인용 안 코드펜스를 렌더 못 해
  ` ``` `가 그대로 노출됨(사용자 스크린샷 확인). 폐기.
- **확정(HTML `<pre style>` 색깔 박스).** WikiDocs는 HTML+`style` 속성을 지원하고
  (`<p style>`,`<center>` 등) 코드블록은 highlight.js로 색칠하므로, 출력은 색칠 안 되는
  `<pre>`에 **왼쪽 파란 바(#5B8DEF) + 옅은 배경(#eef3fb)** 을 줘 코드와 구분. (인쇄책의
  bookoutput 색깔 바와 같은 의도 — 사용자가 image #3로 요청.)
  - `<pre>`는 공백·줄바꿈 보존 → 표 정렬 OK, ` ``` ` 문제 없음. 내용은 HTML escape.
  - 표 출력도 text/plain(콘솔 표현)을 `<pre>`에. HTML 표는 text/plain 없을 때만 정렬 텍스트 폴백.
  - 이미지는 `<pre>`에 못 넣으므로 마크다운 `![](../assets/..)`로.
  - 라벨 `**▶ 실행 결과**`는 **셀당 1회만 맨 위**(이미지+텍스트 셀에서 중복 방지). 각 블록엔 라벨 없음.
  - 색상은 OUTPUT_PRE_STYLE 상수에서 조정 가능.

### 방침 전환: 합성 출력 허용 (2026-06-09, 후반)
사용자 결정: **"실제 결과만" 요구를 철회**. 노트북을 매번 실행/저장하는 비용이 더 크다고 판단.
- 새 기본값: ipynb→md 파싱 + 출력은 **기존 tex의 `synthetic_output_text()`를 import해 재사용**.
  (재구현하지 않음 — tex·md가 합성 로직을 단일 출처로 공유.)
- tex와 동일 게이트: `print(`가 있는 셀에만 합성. 값은 `...` 골격. 라벨 `▶ 출력 형태`.
- 실제 결과는 선택: `executed/<폴더>.ipynb`나 `--execute`가 있을 때만 승급(`▶ 실행 결과`).
- 따라서 [[결정됨 #1/#2]]의 executed/ 경로는 "필수"에서 "선택"으로 격하. ch01은 executed/가
  있어 실제 결과 유지, 나머지는 합성으로 충분.
- §3 "tex→md 기각"은 여전히 유효(역파싱 손실 때문). 다만 합성 "데이터"는 공유함 — 포맷이 아니라.

## 7. 고민 7 — 전자책(PDF/EPUB) 다운스트림 제약 (2026-06-11 신규)

> 계기: WikiDocs 전자책 포맷 조사(공식 FAQ, wikidocs.net/198724·49478). **서점(교보·예스24)
> 판매하려면 EPUB이 사실상 필수**(서점엔 EPUB만 등록, PDF만으론 불가). 즉 같은 `.md`가
> 이제 **웹 / PDF / EPUB 세 타깃**으로 렌더된다.
>
> **핵심 프레이밍:** 지금까지의 변환기·결정은 전부 **"WikiDocs 웹 렌더링" 단일 타깃 전제**였다
> (고민 3의 `<pre>` 색깔박스, 이미지 `../assets/` 상대링크, 수식 통과 등). 전자책 두 타깃은
> 미검증이며, 세 타깃의 요구가 충돌하는 지점이 이 고민의 본체다. 라이브 변환기는
> `.claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py`.

### 7-1. 전자책이 새로 강제하는 세부 제약 (모두 "웹은 통과, 전자책 미검증")

1. **출력 박스 `<pre style="…overflow-x:auto">` ↔ PDF/EPUB** (`build_wikidocs.py:266` `OUTPUT_PRE_STYLE`).
   `overflow-x:auto`는 웹 브라우저에서만 가로 스크롤 → PDF·e-ink엔 스크롤이 없어 넓은
   콘솔 표/배열이 **잘리거나 페이지 폭을 넘침**. inline `style`(파란 바 `#5B8DEF`, 배경 `#eef3fb`)이
   PDF 변환기·EPUB 리더에서 **살아남는지 미검증** — 사라지면 코드/출력 구분(고민 3 결정)이 무력화.
2. **코드 펜스 긴 줄 reflow.** highlight.js는 웹 전용. **EPUB은 `<pre><code>` 긴 줄을 wrap 안 함**
   → `max_length=128`·긴 시그니처가 e-reader에서 오른쪽이 잘림. 변환기는 줄 길이를 제어하지 않음(원본 그대로).
3. **이미지 `../assets/` 상대 링크** (`build_wikidocs.py:376` `![output](../assets/{val})`). git 레포 안에서만
   resolve. **FAQ가 직접 경고: "외부 이미지는 PDF에서 누락 → WikiDocs에 직접 업로드하라."** 출판 단계에서
   `assets/` PNG를 WikiDocs 페이지에 실제 업로드(임베드)하지 않으면 PDF/EPUB에서 그림이 통째로 빠짐.
   → "git용 md"와 "WikiDocs 출판용 md"의 이미지 참조 방식이 다를 수 있는 미해결 간극. (장→절 5분할이라
   각 페이지에서 `../assets/`가 맞는지도 별도 확인.)
4. **셀 사이 빈 줄** (FAQ: 빈 줄 없으면 PDF 변환 오류). `<pre>`/이미지/헤더 **앞뒤 blank-line 불변식**을
   변환기가 강제해야 마크다운이 블록 HTML로 인식. 웹은 관대하지만 PDF에서 깨질 수 있음.
5. **수식 LaTeX `$…$`.** 변환기는 마크다운 셀을 **그대로 통과**(수식 처리 코드 없음). 웹은 MathJax OK지만
   **EPUB 리더의 MathML/MathJax 지원이 들쭉날쭉 → `$L=-\sum y\log p$`가 원문 노출 위험.** 전자책 타깃엔
   **수식 이미지 사전 렌더링(폴백)** 고려.
6. **페이지 간 상대 `.md` 링크** ("이 장의 구성"이 형제 `(stem-practice.md)` 링크). 웹은 페이지 이동이지만
   **전자책은 페이지들이 하나로 합쳐짐 → 파일 링크가 깨지거나 무의미.** 전자책에선 장-절 앵커 구조 필요.

### 7-2. 기존 결정의 전자책 관점 재검토 (재결정 트리거)

- **ⓐ 합성 출력(`▶ 출력 형태`, 값 `...`) ↔ 유료 판매책.** 고민 2의 "합성을 기본값으로" 격하는
  **무료 WikiDocs 공개** 전제였다. **돈 받고 파는 EPUB**에 `...` 골격이 실리면 품질·신뢰 문제.
  → "무료 웹은 합성 허용 / 판매 EPUB은 실제 결과(`executed/`) 필수"로 **타깃별 정책 분리** 검토.
- **ⓑ 출력 구분 방식(고민 3) 자체가 웹 전제.** `<pre>` 색깔박스는 "WikiDocs는 HTML+style 지원"이라는
  웹 관찰 기반. **PDF·EPUB에서도 구분되는지 검증 전.** 세 타깃 공통의 **safe subset** 표현 재탐색 필요.

### 7-3. 스킬에 담을 형태 (설계 방향)

- **타깃 프로파일.** `--target web|pdf|epub`. 단, **EPUB이 가장 제약이 크므로 EPUB 기준 safe-subset으로
  단일화하면 웹·PDF는 자동으로 안전** → 분기보다 safe-subset 단일화가 단순.
- **변환 후 자동 점검 확장** (SKILL.md 4번 체크리스트에 전자책 항목): 긴 코드 줄 / `<pre>` 앞뒤 빈 줄 /
  상대 이미지 링크 / 수식 노출.
- **검증 루프 = Playwright 캡처 비교.** ch01을 실제 WikiDocs에 올려 **PDF·EPUB로 내보낸 뒤 캡처해 웹과 비교**.
  "T4 30분"에 준하는 객관 기준을 전자책에도 둔다("EPUB에서 코드·수식·그림이 안 깨지는가").

### 7-4. 다음 액션 (권장)
가장 임팩트 큰 **③(이미지 업로드)·⑤(수식)**은 **ch01을 실제 EPUB으로 내보내 한 번 깨뜨려보는** 검증이 결정적.
이 실험으로 7-1의 어느 항목이 진짜 깨지는지 확정한 뒤 safe-subset을 설계한다.

### 7-5. EPUB/PDF 실측 확인 (2026-06-11) — 추정 → 측정으로 확정

> **변환 엔진 확정:** WikiDocs "전자책 버전 관리" 페이지의 오류 예시에 실제 명령이 노출됨 →
> `pandoc X.md -o X.pdf --template eisvogel.tex --listings --pdf-engine xelatex
> --top-level-division=chapter --filter pandoc-latex-environment`. 즉 **PDF = pandoc→xelatex,
> EPUB = pandoc epub writer**. 로컬 `pandoc 3.9`로 ch01 5개 페이지를 변환해 실측.
>
> **참고:** EPUB은 셀프서비스 미리보기가 아님. [전자책] 탭 판매 신청 → 위키독스 심사
> (**최소 100페이지** + 품질) → "전자책 생성" 권한 → "전자책 개정 신청"(서버에서 ~10분 비동기 생성).
> 즉 우리가 직접 즉석 EPUB을 못 뽑으므로 **로컬 pandoc 재현이 사실상 유일한 미리보기 수단**이다.

실측 결과 (4건, 모두 재현됨):

1. **🔴 PDF에서 출력 박스 전멸 (치명적).** 출력 박스가 raw HTML `<pre style>`이라 **LaTeX writer가
   통째로 드롭**. 출력에만 있는 값(`405,789`/`00am`/`'hugging','face'`)이 `-t latex`에서 **모두 0**,
   `-t html`에서는 생존. → WikiDocs **기본 제공 PDF**에서 모든 `▶ 실행 결과` 라벨 아래가 **빈칸**이 된다.
2. **🟠 EPUB XML well-formedness 깨짐.** 출력 없는 셀의 플레이스홀더 주석
   `<!-- 실행 결과 없음: --execute 또는 --executed-notebook ... -->`가 **XML 금지 `--`(이중 하이픈)** 포함
   → `xmllint`: `Comment must not contain '--'`. 엄격한 EPUB 리더는 이 챕터를 에러로 표시.
3. **🟠 추적표 가로 잘림.** 390px 렌더에서 표 `scrollW 471 > 390`, `overflow` 처리 없음 → 마지막 컬럼
   (Activation/Loss)이 잘림(스크린샷 `epub-table-overflow.png`).
4. **🟡 `<pre>` 코드/출력 reflow.** `overflow-x:auto`는 웹만 스크롤, e-ink/페이지 매체는 잘림. 긴 줄 wrap 안 됨.
   - 🟢 **잘 됨:** 이미지는 EPUB에 실제 임베드(`media/file0.png`), 수식은 `--mathml`로 `<math>` 변환(리더 편차 있음),
     `<pre style>` 색깔 바는 EPUB에선 유지(PDF에서만 소실).

**해결의 핵심 — `--filter pandoc-latex-environment`.** WikiDocs pandoc 명령에 이미 이 필터가 있다.
이 필터는 **fenced div(`::: {.output}`)를 지정한 LaTeX 환경(tcolorbox 등)으로 매핑**한다. 따라서 출력 박스를
raw HTML `<pre style>`이 아니라 **fenced div + 내부 code block**으로 내보내면:
- **PDF:** 필터가 `.output` → 색깔 환경. (필터 없이도 내부 code block은 verbatim으로 **내용 생존** → 발견 1 구조적 해결)
- **EPUB/HTML:** `.output` → `<div class="output">`, CSS로 색깔 바. raw `<pre style>` 제거로 발견 2의 `--` 문제와도 분리.
→ **출력 박스 표현을 `<pre style>` → fenced div로 교체**가 발견 1·2를 동시에 푸는 정답. (시제품 검증: §7-6)

### 7-6. 시제품 검증 — fenced div 출력 박스 (2026-06-11, 실측 통과)

ch01 anatomy의 출력 박스 3개를 `<pre style>` → **`::: {.output}` + 내부 ```` ```text ```` 코드블록**으로 바꾼
시제품(`/tmp/proto/anatomy-proto.md`)을 양 타깃으로 변환해 검증. CSS는 `div.output{파란 바+옅은 배경}`,
`div.output pre{white-space:pre-wrap}`(reflow). LaTeX 환경 매핑은 `pandoc-latex-environment: {tcolorbox:[output]}`.

| 항목 | 기존 `<pre style>` | 시제품 fenced div |
|---|---|---|
| PDF(LaTeX) 출력 내용 생존 | ❌ 전멸(405,789=0) | ✅ 생존(405,789=1) |
| PDF 색깔 박스 | ❌ | ✅ `\begin{tcolorbox}` ×3 (필터 적용) |
| EPUB XML 적법성 | ❌ `--` 주석 에러 | ✅ xmllint clean |
| EPUB 출력 박스 | ✅(색 유지) | ✅ `<div class="output">` ×3 + 색 유지 |
| EPUB 390px reflow | ❌ overflow 잘림 | ✅ pre-wrap, 넘침 0 (스크린샷 `proto-output-box-390w.png`) |

→ **fenced div가 PDF·EPUB 양쪽에서 출력 박스를 살린다(발견 1·2·4 동시 해결).** 단:
- **필터 호출 주의:** `pandoc --filter <bin>` 직접 호출은 tcolorbox 매핑이 안 먹는 케이스가 있었고,
  `pandoc -t json | <filter> latex | pandoc -f json -t latex` 파이프라인에선 정상(×3). WikiDocs는 서버에서
  자체 config로 `--filter pandoc-latex-environment`를 돌리므로 우리 쪽 책임은 **표현(fenced div)을 내보내는 것**까지.
- **코드 블록 긴 줄은 여전히 잘림**(출력 박스만 pre-wrap 처리). 코드 wrap은 가독성 해쳐 비권장 → **노트북 단계에서 줄 길이 관리**가 정답(7-1②).
- **⚠️ 미해결: WikiDocs '웹' 렌더러가 `:::` fenced div를 렌더하는지 미검증.** 웹은 pandoc이 아니라 자체 JS 마크다운.
  `:::`가 웹에서 리터럴로 노출되면 곤란. → 후속 검증 필요(§7-7의 PDF_EXCLUDE/INCLUDE가 대안).

### 7-7. WikiDocs 공식 [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723) 반영

공식 문서가 7-1~7-6의 발견을 **직접 확인**해줌. 변환기/노트북이 지켜야 할 규칙:

- **🔴 "HTML 코드는 전자책 변환 시 정상 표시되지 않습니다. HTML 대신 마크다운을 쓰세요.
  책 설정에서 HTML 사용 '아니오' 권장."** → 우리 `<pre style>` 출력 박스가 정확히 이 "HTML 코드". **fenced div 전환의 공식 근거.**
- **빈 줄 불변식(공식):** 헤딩·이미지 **위아래 빈 줄 필수**, 없으면 PDF 변환 오류. (7-1④ 확정)
- **이미지(공식):** 외부 링크 이미지는 PDF에 안 나옴 → **다운로드 후 위키독스에 직접 업로드**. GIF 애니메이션 금지(JPG/PNG). (7-1③ 확정)
- **역슬래시 경로:** `C:\dir` 류는 코드블록으로 감싸거나 `/`로. (코드/경로의 `\` 주의)
- **각주 이름 유니크:** 전자책은 **전 페이지를 한 문서로 통합** → 동일 각주명 충돌. (페이지 통합 사실 = 7-1⑥ 링크 문제와 같은 뿌리)
- **`-----` 줄 구분선 금지:** PDF에서 표로 오인되어 오류. (변환기가 `---`/`***` 구분선을 쓰는지 점검 필요)
- **엔터키 줄바꿈 모드:** 전자책엔 미적용 → 전자책 목적이면 해제하고 본문 재조정.
- **🟢 PDF_EXCLUDE / PDF_INCLUDE 태그 = 타깃 분기 네이티브 지원.** 웹 전용/PDF 전용 콘텐츠를 WikiDocs가 직접 지원.
  추가로 `\newpage`(강제 페이지 넘김), 이미지 `{ width=8cm }`(PDF 이미지 크기). → 7-3의 `--target` 분기 대신
  **WikiDocs 네이티브 PDF_EXCLUDE/INCLUDE로 web↔ebook 차이를 흡수**하는 설계가 가능(예: 웹은 `<pre style>` 박스를
  PDF_EXCLUDE로 감싸고, PDF용 fenced div를 PDF_INCLUDE로 — 7-6의 웹 렌더 미검증 리스크를 우회).

### 7-8. WikiDocs 웹 실측 → 기본값 `code` 확정 (2026-06-11)

§7-6의 미해결 리스크("웹이 `:::`를 렌더하는지")를 **실제 업로드 후 확인**(wikidocs.net/365451, Playwright).
**결과: 웹은 fenced div 를 렌더하지 못함.** `::: {.output}` 와 `:::` 가 **글자 그대로 노출**되고
`div.output` 은 0개(스크린샷 `playwright_screenshot/wikidocs-web-fenced-div-broken.png`). 단 **안쪽
```` ```text ```` 코드펜스는 웹에서 정상 렌더**(출력 내용 박스 자체는 나옴) → `:::` 래퍼만 문제.

세 타깃 정리 (실측 종합):

| 표현 | WikiDocs 웹 | PDF(pandoc) | EPUB(pandoc) |
|---|---|---|---|
| `html-box` (`<pre style>`) | ✅ 색 박스 | ❌ 내용 드롭 | ❌ XML 깨짐 |
| `fenced-div` (`:::`) | ❌ `:::` 노출 | ✅ tcolorbox | ✅ div+CSS |
| **`code` (```` ```text ````)** | ✅ 코드박스 | ✅ 코드블록 | ✅ 코드블록 |

→ **세 타깃을 모두 만족하는 건 `code` 뿐. 변환기 기본값을 `code` 로 변경**(2026-06-11).
색깔 박스는 포기하되, `▶ 실행 결과` 라벨 + (코드=python 하이라이트 / 출력=plain)로 구분.
`fenced-div`(전자책 전용 빌드 시 색 박스)·`html-box`(웹 전용)는 옵션으로 남김.

> 미련: 웹+전자책 동시에 색을 지키려면 [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723)의 `PDF_EXCLUDE/INCLUDE` 이중 출력이 유일 후보지만,
> EPUB 이 그 태그를 존중하는지 미검증(서버 업로드 필요)이라 보류. 필요해지면 그때 검증.

### 7-9. [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723) 규칙 변환기 방어 + 사후 린터 (2026-06-11)

공식 [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723)을 ch01 md·변환기에 대해 전수 감사. 결과:
- **변환기가 만든 위반 1건**: 로드맵 `## 이 장의 구성` 헤딩 아래 빈 줄 누락 → 수정(전 챕터 영향).
- **노트북 유래 1건**: overview 의 수평선 `---` → 변환기 방어로 제거.

**변환기 방어 추가(`_sanitize_md_cell`)** — 코드펜스 밖에서:
- [11] 수평선(`---`/`***`/`___`) 제거(앞이 빈 줄일 때만 — setext 헤딩 오인 방지).
- [1] 2번째+ H1(`#`) → H2 강등(본문 H1 금지; 첫 제목 H1 은 기존대로 제거).
- [10] 각주 이름에 챕터 stem 접두 → 전자책 전 페이지 통합 시 충돌 방지.
- [6]/[3] raw HTML·외부 이미지는 자동수정 불가 → 변환 로그에 **경고**만(사용자가 원본 수정).
변환 시 `방어(전자책 규칙): 수평선 N 제거 …` 로 요약 출력.

**사후 린터 `scripts/check_wikidocs_md.py`** — 변환 후 한 번 더 점검(회귀·수기 편집 방지):
- E1 본문 H1 / E2 헤딩 빈줄 / E3 이미지 빈줄 / E4 외부이미지 / E5 GIF / E6 raw HTML /
  E7 수평선 / E8 코드펜스 짝 / E9 각주 전페이지 유니크 / W1 윈도우 경로. 코드펜스 안은 제외.
- 위반 시 종료코드 1(CI/스킬 게이트용). 음성 테스트로 탐지 확인.
- 기본 대상 `pages/*.md`. **실측: ch01 ✅ 통과 / ch02·ch24 는 구버전(`<pre style>`)이라 위반 다수**
  → 그 챕터들도 새 변환기로 재생성 필요(후속).

## 8. 고민 8 — Colab 실행 결과 관리 (2026-06-11 신규)

> 계기: 출력 정책은 "합성 기본 + executed/ 승급"으로 정리됐지만(고민 2), **executed/를
> 실제로 채우는 절차가 전부 수동**(챕터마다 Colab 열기→끝까지 실행→`.ipynb` 다운로드→
> executed/에 저장→커밋)이다. GPU 챕터(07~32, 약 26개)를 손으로 돌리는 건 비현실적.
> 판매 EPUB은 실제 결과 권장(고민 7-2ⓐ)이라 **Colab 실행을 체계적으로 돌리고 결과를
> 관리하는 워크플로**가 필요하다. → 출력 원천 소비(build_wikidocs.py)는 이미 있고, **생산 측(실행·수집)이 비어 있음.**

### 8-1. 러너 노트북 `executed/run_on_colab.ipynb` (구현 완료, 실측 동작)
Colab T4에서 여는 단일 러너 노트북 (레포에 체크인, GitHub→Colab 링크로 1클릭):
1. 포크를 `git clone`(또는 pull).
2. 챕터 선택: `TARGET = "stale"(기본) | "all" | "gpu"(07+) | 리스트([1,7,24])`.
3. 각 챕터 clean ipynb 를 `nbclient`로 **끝까지 실행, 출력 캡처**(셀당 `PER_CELL_TIMEOUT` 기본 1h).
4. `executed/<폴더>.ipynb` 로 저장 + `executed_from{source_sha256,executed_at,runtime,status}` 도장. (변환기가 자동 소비 — 우선순위 ②)
5. `executed/` 만 staging → 커밋 → **포크 master 로 직접 push**(§8-3 A).

> **setup 셀 교훈(버그→수정):** `'...{REPO}.git {WORK}'.format(REPO=REPO)` 가 문자열 안 `{WORK}`까지
> 치환하려다 `KeyError: 'WORK'`. → IPython `system()` 명령은 **f-string**으로 작성(부분 `.format` 금지). 생성기 반영 완료.

### 8-2. 멱등·재개 (필수)
- Colab 무료 세션은 GPU 시간/연속 사용 한계(아이들 끊김, 최대 ~12h) → **한 세션에 26개 일괄은 위험**.
- 러너는 **재개 가능**해야: 이미 `executed/<폴더>.ipynb`가 있고 clean 노트북 해시가 그대로면 **건너뜀**.
- clean 노트북이 실행 후 바뀌면 executed 가 **stale** → 러너가 소스 해시를 executed 메타데이터에
  심어두고 비교, 갱신 대상만 재실행. 어느 챕터가 최신/낡음인지 **manifest(표)** 로 출력.
- yoongu 결정(스레드): **실행 환경은 Colab 기준으로 고정**(M시리즈 맥북 가능하나 "검증한 환경" 우선).
  T4 30분 가정은 **완화 가능**(GPU 챕터는 더 걸려도 됨) — 단 세션 한계 때문에 배치/재개가 필수.

### 8-3. [결정됨 #8] 실행본 반환 = Colab 직접 push (2026-06-11, 사용자 선택 A)
- **A. Colab에서 직접 git push (채택)** — 러너가 PAT(getpass 입력, 저장 안 함)로 `executed/`만 커밋·푸시.
  본인 포크라 blast radius 작음. 전 챕터 자동화에 최적. push URL `https://<token>@github.com/...`.
- (기각) B. zip 다운로드 → 로컬 커밋 — 토큰 불필요하나 세션마다 수동 한 단계.
- 구현: 러너 노트북 `executed/run_on_colab.ipynb`.
  소스 해시를 executed 메타데이터(`executed_from.source_sha256`)에 심어 멱등/재개. 대상 `all|gpu|stale|리스트`.

### 8-4. 결정론 / 신뢰성 메모
- tex 결괏값은 "가짜가 많다"(yoongu)→ **신뢰 금지, executed/ 만 canonical**(고민 2와 같은 뿌리).
- 가능한 곳은 seed 고정. 실행 로그(시간/실패 챕터)를 manifest 에 남겨 재현·디버그.

### 8-5. 실행 현황 + 전수 점검 (2026-06-11, 러너 첫 가동)
사용자가 러너를 Colab T4에서 돌려 **01~15 챕터를 실행·push**(GPU 챕터 07~15 포함). 로컬 pull 후 전수 점검:
- **15개 전부 `status=ok`, 에러 트레이스백 0건, 모든 코드 셀 실행 완료**(조기 중단 없음).
- 출력 없는 셀은 챕터당 1~2개(import/정의 전용) — 정상. `stderr`(warn)은 학습 진행바·HF 경고로 무해.
- GPU 챕터(07~15) stderr 스캔: CUDA/fp16 언급 있고 **OOM·CPU 폴백·RuntimeError 적신호 0** → T4에서 정상 학습 확인.
- ch01 도 러너로 재실행되어 `executed_from` 도장이 박힘 → 이제 멱등 관리에 통일(이전엔 로컬 `--save-executed` 산).
- 점검 스니펫: `executed_from.status` + output_type=='error' 카운트 + 셀 실행카운트 == 코드셀 수.
- **남은 실행분: 16~32**(한국어/사전학습/SFT/DPO/GRPO/diffusion). `TARGET="stale"` 로 이어서 채우면 됨.
- 다음: executed/ 가 채워진 **01~15 페이지를 실제 결과(`▶ 실행 결과`)로 재생성** + 린터 통과 + TOC 갱신(고민 7-9 후속).

### 8-6. CLI 러너 `executed/run_via_cli.sh` 추가 (2026-06-15, upstream 제안 반영)
계기: upstream(yoongu) 디스코드 공유 — [`googlecolab/google-colab-cli`](https://github.com/googlecolab/google-colab-cli)
("구글 코랩 매번 손으로 실행하는걸 없앨 수 있을 것 같아요"). §8-1 노트북 러너는 브라우저에서
"모두 실행"을 손으로 눌러야 하는데, CLI 로 그 단계를 없앨 수 있는지 검토 → 채택(폴백 유지).

- **무료 T4 실측(2026-06-15)**: 무료 Colab 계정에서 `colab new --gpu T4` → READY, `torch.cuda.is_available()==True`(Tesla T4).
  결제·compute-unit 요구 없음. CLI 문서 `docs/demos.md` 는 "GPU 프로비저닝은 계정 quota 게이팅, 대부분 안 됨"이라
  보수적으로 적었으나 실계정에선 됨. 단 **무료 GPU 가용성·일일 한도**가 있어 *항상* 보장은 아님 → A(브라우저)를 폴백으로 유지.
- **구조**: VM 위 `colab_cli_exec.py`(§8-1 노트북과 동일 실행 로직 — 해시 멱등·`executed_from` 도장·챕터별 소요시간) +
  로컬 래퍼 `run_via_cli.sh`. `colab run --keep` 로 VM 할당→실행, `colab download` 로 결과 회수, `trap` 으로 VM 자동 종료.
- **§8-3 B 의 부활**: 결정 #8 에서 기각했던 "다운로드→로컬 커밋"(B)을 CLI 가 자동화하니 **PAT 불필요**가 장점이 됨.
  A(노트북, PAT push)와 C(CLI, download→로컬 커밋)는 산출물·실행 로직 동일, 회수 방식만 다름.
- **인증 함정(실측)**: 이 CLI 는 토큰 refresh 때 6개 스코프 전체(openid/profile/email/cloud-platform/colaboratory/drive.file)를
  재요구 → Colab 하나만 허가하면 첫 실행은 되나 다음에 `invalid_scope`. 동의 화면 **"모두 선택"** 필요.
  `OAUTHLIB_RELAX_TOKEN_SCOPE=1` 은 최초 파싱만 풀 뿐 refresh 는 못 고침. 깨진 토큰은 `rm ~/.config/colab-cli/token.json`.
- **제약**: CLI 는 macOS/Linux 전용(Windows 미지원) — 단 실행은 유지보수자 작업이라 학습자 영향 없음.

### 8-7. 전 챕터 CLI 실행 실측 — 한계와 버그 3건 (2026-06-16)
01~32 전체를 CLI(FORCE)로 돌려보며 발견·수정한 것들. **결론: CLI 경로는 가벼운~중간 챕터엔 잘 되나, 무료 티어의 VM 수명 한계로 무거운 챕터는 못 끝낸다.**

- **사실 ⓪: `colab exec -f <노트북>` 은 출력 노트북을 만들지 않는다.** (upstream 이슈 #17 의 핵심 가정 정정)
  exec 은 실행만 하고 stdout 을 스트리밍할 뿐 — 출력이 임베드된 `*_output.ipynb` 를 생성하지 않는다(이미지는 `--output-image` 로만).
  게다가 `run_on_colab.ipynb` 는 getpass·설정 셀이 있는 대화형이라 비대화형 exec 에선 멈춘다. → "노트북을 exec 로 돌려 executed 를 얻는다"는 불가.
  그래서 **VM 위 `nbclient` 로 챕터를 실행해 출력을 임베드 + `colab download` 회수**(`colab_cli_exec.py`)가 본질적으로 필요했다.
- **버그 ①: `colab download` 가 `.ipynb` 를 깨뜨림.** colab-cli 의 download(contents.py)는 Jupyter Contents API 를 쓰는데,
  `.ipynb` 는 API 가 **파싱된 dict**(`format:json`)로 돌려준다. CLI 는 base64 가 아니면 `str(content)` 로 저장 → **Python repr**(작은따옴표)
  이 되어 JSON 파싱 불가. → **회피: VM 에서 노트북을 base64 사본(`*.ipynb.b64`)으로 떨궈 받아 로컬에서 무손실 디코드.**
  (평문·base64 만 무손실. manifest 같은 `.txt` 는 멀쩡.)
- **한계 ②: 무료 CLI 세션 ~11분 수명 캡.** 한 VM 을 `colab new` 후 `colab exec` 로 여러 챕터를 순차 실행하면
  **두 번 모두 정확히 ~11분 만에** `Connection lost`로 세션이 끊겼다(일일 한도 아님 — 분 단위, 재로그인해도 동일).
  → **한 VM 재사용 포기, 챕터마다 새 VM**(`colab run` per chapter, 각자 신선한 11분 예산). 클론은 `--depth 1`.
- **버그 ③: 죽은 VM 에서 10시간 hang.** 실행기가 nbclient **자식 커널**로 챕터를 돌려 부모(run 커널)는 무출력 →
  `colab run --timeout 36000` 이 "죽은 세션"과 "조용히 일하는 중"을 구분 못 해, VM 이 11분에 죽어도 **10h 비활성 타임아웃까지 대기**(ch25 에서 40분+ 멈춤).
  → **수정: 실행기에 하트비트(20초마다 1줄) + `--timeout` 을 120초로.** 정상 챕터는 하트비트로 안 끊기고, VM 이 죽으면 출력이 멈춰 2분 내 빠르게 실패.
- **최종 실측(2026-06-16)**: **30/32 CLI 성공** — 25·27 제외 전부. GPU 학습 챕터 다수 포함(07 BERT, 20 en_bert_pretrain,
  22 ko_bert_pretrain, 24 gpt_tinystories, 26 ko_tiny_gpt, 28 SFT, 30 DPO, 31 GRPO, 32 diffusion). 전수 검증: 유효 JSON·status=ok·err 0.
- **CLI 로 못 끝낸 챕터는 단 2개** — 사용자 제공 코랩 실측 소요시간상 단일 실행이 11분 캡을 넘는 것:
  **25 gpt2_continual_pretrain(21분 6초), 27 ko_gpt2_continual_pretrain(18분 54초).** → **`git show origin/master:executed/<…> > …` 로 충당**(동일 포맷, runtime=colab-t4). 변환 테스트엔 지장 없음.
  (참고 소요시간: 가장 무거운 합격 챕터도 28 SFT 3:20·30 DPO 3:30·24 GPT 3:29 수준 — 25·27만 두 자릿수 분.)
- **`Connection was lost` 일시 드롭**: 17·20·26·29 가 첫 시도엔 이 에러로 미완 → **재시도 1회로 전부 성공**. 무료 Colab 웹소켓 플레이크라 캡 초과와 구분해 **재시도로 처리**(hang 방지책 덕에 빠르게 실패하고 넘어가므로 재시도 비용 작음).
- **함의(중간 결론, §8-8에서 정정됨)**: 위 시점엔 11분 캡을 무료 T4의 본질적 한계로 봤으나, **사실은 colab-cli 버그였다 → §8-8 참조.**

### 8-8. ★정정★ "11분 캡"의 정체 = colab-cli keep-alive 버그(issue #14) (2026-06-16)
계기: 25·27을 정말 CLI 로 못 돌리는지 커밋 전 재확인. **단일 새 VM 에서 ch25 를 직접 끝까지 돌려본 적이 없었음**(여태 OVER_CAP 으로 스킵만).

- **생존 테스트**: ch25 를 돌리고 stop 하지 않은 채 VM 상태를 폴링 → **VM 이 ~10분33초에 DEAD**(connection 만 끊긴 게 아니라 실제 회수). keep-alive 데몬(60s ping)도 못 막음.
- **문서에서 원인 발견**: `colab-cli/docs/01_session_management.md`(표준 idle timeout ~90분) + `AGENTS.md`/issue #14 — 옛 keep-alive RPC(`colab.pa.googleapis.com/.../KeepAliveAssignment`)는 **일반 계정에 403 `USER_PROJECT_DENIED`** → 세션이 **몇 분 내 idle-prune**. **2026-06-15 TFE 터널 핑(`GET /tun/m/<endpoint>/keep-alive/`, `X-Colab-Tunnel`)으로 수정**. 그런데 설치본 **v0.5.11**(PyPI/최신 태그)은 옛 RPC 그대로 — 데몬이 연속 4xx 로 루프 중단 → VM 이 ~11분에 prune. **우리가 본 캡과 정확히 일치.**
- **수정 후 실증**: `uv tool install git+.../google-colab-cli`(main, v0.5.12.dev2, TFE 핑) 로 교체 후 ch25 재실행 → **VM 이 24분 생존, 완주**(`status=ok`, elapsed 1437s, 마지막 하트비트 1420s). ch27(19분)도 동일(※ 1회는 로컬 DNS 블립으로 즉시 실패 — 캡 무관 — 재시도로 완주).
- **결론**: **11분 캡은 무료 T4 한계가 아니라 v0.5.11 의 keep-alive 버그.** issue #14 수정본부터 **전 32챕터 CLI 완주 가능** → `OVER_CAP` 기본값을 **빈 값**으로(옛 버전용 안전장치로만 유지), 설치 안내를 **git main(수정본)** 으로 변경(README/SKILL/run_via_cli.sh). 무료 표준 idle timeout 은 ~90분.
- **클린 재현(2026-06-16)**: 전 챕터 드라이버를 레포 파일로 정리(하드코드 제거·`REPO` env·챕터 폴더 스캔·`OVER_CAP` env·**일시 드롭 1회 자동 재시도**)하고 **초기화 후 단일 패스** 실행 → **30/32 전부 status=ok, 실패 0건**(retry-once 로 이전의 17·20·26·29 수동 재시도 불필요해짐), 25·27 만 캡 스킵→master 충당. 소요 **1시간 8분**(per-chapter 새 VM).
- **도구 통합(2026-06-16)**: `run_all_via_cli.sh` 와 `run_via_cli.sh` 를 **`run_via_cli.sh` 하나로 병합** — 인자 없으면 전 챕터, `7 24` 처럼 주면 그 챕터만(per-chapter 새 VM·재시도·resume·OVER_CAP 전체모드 한정). `executed/` colab-cli 도구 **2종 확정**: `run_via_cli.sh`(오케스트레이터)·`colab_cli_exec.py`(VM 실행기).

## 9. 진행 로그

- 2026-06-16 ★정정★ "11분 캡"=colab-cli keep-alive 버그(issue #14) 규명(§8-8) — v0.5.11 옛 RPC가 일반계정 403→VM ~11분 prune. main(v0.5.12.dev2, TFE 핑)으로 교체 후 ch25 24분 완주·ch27도 완주 → **전 32챕터 CLI 가능**. OVER_CAP 기본 빈 값, 설치 안내 git main 으로.
- 2026-06-16 전 챕터 드라이버 일반화 + 초기화 후 **단일 패스 30/32 ok·0실패**로 검증(§8-7, retry-once 로 일시 드롭 자동 복구). 이후 `run_all_via_cli.sh`+`run_via_cli.sh` 를 **`run_via_cli.sh` 하나로 병합**(인자 없으면 전체, `7 24` 면 해당 챕터) → 도구 2종(`run_via_cli.sh`·`colab_cli_exec.py`). README/SKILL 갱신.
- 2026-06-16 upstream 이슈 #17(colab-cli 로 executed 자동화) 검토 — 작업이 이미 충족. 이슈의 가정 `exec -f → *_output.ipynb` 는 성립 안 함(exec 은 출력 노트북 미생성)을 README/§8-7 에 명시. 그래서 nbclient+download 방식이 필연.
- 2026-06-16 전 챕터 CLI 실행 실측(§8-7) — **30/32 CLI 성공**(GPU 학습 SFT·DPO·GRPO·diffusion 포함), 캡 초과 25·27만 master 실행본으로 충당 → **executed/ 32개 전수 유효(status=ok·err0)**. 버그 3건 수정: `colab download` 의 `.ipynb`→repr 깨짐(base64 사본 회피), 무료 세션 ~11분 캡(챕터별 새 VM 전환), 죽은 VM 10h hang(하트비트+`--timeout 120`). 일시 드롭(17·20·26·29)은 재시도 1회로 해결.
- 2026-06-15 CLI 러너 `executed/run_via_cli.sh` + VM 실행기 `executed/colab_cli_exec.py` 추가(§8-6) — upstream 의 `google-colab-cli` 제안 반영. 무료 계정 T4 실측 확인 후 채택, 브라우저 러너는 폴백 유지. README/SKILL ① 에 CLI 경로(A/C) 병기.
- 2026-06-14 executed 16~32 실행·push 완료(사용자, Colab 러너) → **01~32 전 챕터 실행본 확보**(§8-5 후속).
- 2026-06-14 executed 01~32 전수 변환·린트로 잡은 보강을 변환기/린터에 반영: **헤딩 위아래 빈 줄 자동 삽입**(E2, ch29 8건 해소) + **트렁케이트 opt-out 에 22 추가**(한국어 토큰화 보존) + 노이즈 필터에 `generation_config` 보일러플레이트 추가 + **윈도우 경로 방어**(변환기는 진짜 `C:\` 경로만 인라인코드, 린터 W1은 LaTeX 수식 제외해 `i:\,` 오탐 제거). 재검 전 32챕터 **위반 0·경고 0**.
- 2026-06-14 upstream(yoon-gu) PR 준비 — upstream/master 기준 깨끗한 브랜치 `feat/notebook-to-wikidocs` 에 항목별 커밋(러너 / 변환기·린터 / 보강 / 스킬). 러너 **repo-agnostic 화**(`REPO`=본인 fork)·챕터별 소요시간·Colab 배지 추가. `198723` 언급을 [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723) 링크로 통일.
- 2026-06-14 SKILL.md 전면 갱신 — 파이프라인 3단계(① executed/ 실행결과 확보 ② 변환 ③ 린터 검증)로 재정리. 변환할 챕터의 `executed/<폴더>.ipynb` 가 없으면 합성으로 넘어가지 말고 사용자에게 Colab 러너 실행을 안내·중단.
- 2026-06-11 러너 첫 가동 — Colab 에서 01~15 실행·push, 로컬 전수 점검 전부 ok/에러0/GPU정상(§8-5). setup 셀 KeyError(.format→f-string) 수정.
- 2026-06-11 고민 8(Colab 실행 결과 관리) 추가 — 러너 노트북 `executed/run_on_colab.ipynb` + 생성기 구현, 결정 #8=A(Colab 직접 push).
- 2026-06-11 [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723) 전수 감사 → 변환기 방어(_sanitize_md_cell) + 사후 린터 check_wikidocs_md.py 추가(§7-9). ch01 통과.
- 2026-06-11 WikiDocs 웹 실측: fenced div(:::) 웹에서 깨짐 → 기본 출력 스타일 code 로 확정(§7-8). ch01 재생성.
- 2026-06-11 pandoc으로 ch01 EPUB/PDF 실측(§7-5) → fenced div 출력박스 시제품 검증(§7-6) → 공식 [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723) 반영(§7-7).
- 2026-06-11 WikiDocs 전자책 포맷 조사(웹 403 → Playwright). EPUB/PDF 다운스트림 제약을 고민 7로 누적.
- 2026-06-09 브랜치 `feat/notebook-to-wikidocs-skill` 생성. 스킬 스캐폴딩 + 변환기 작성.
- 2026-06-09 ch01 `--execute` 실제 실행 → pages/01-tfidf-* 에 진짜 출력(표·그림) 검증. 커밋 f903aa9.
- 2026-06-09 GPU 챕터 출력이 tex에서 synthetic임을 확인 → executed/ 방식 확정.
- 2026-06-09 변환기를 배치/자동발견 + executed/ 출처 우선순위로 개편 중.
