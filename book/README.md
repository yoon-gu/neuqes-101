# 출판용 LaTeX 원고

이 폴더는 1-34장 Colab 노트북을 원천으로 만든 출판용 LaTeX 프로젝트입니다.

## 빌드

레포 루트에서 실행합니다.

```bash
python3 book/tools/notebook_to_tex.py --use-executed
latexmk -xelatex book/main.tex
```

원고 변환에는 **pandoc 2.19.2** 가 필요합니다. 3.x 로 돌리면 표 구조와 `\label`
처리가 달라져 장마다 수백 줄씩 무관한 차이가 생깁니다.

`--use-executed` 는 `executed/` 실행본의 저장된 출력과 그림을 싣습니다. 노트북을
직접 돌려 출력을 새로 만들려면 대신 `--execute` 를 씁니다.

`book/chapters/` 는 **압축본** 이고 이것이 기본값입니다. 코드 셀은 개념상 핵심만
남고 나머지는 Colab/QR 안내로 대체되며 FAQ도 추려집니다. 압축하지 않은 긴
판본이 필요하면 `--full` 을 주되, 정본을 덮어쓰지 않도록 `--output-dir` 도 함께
지정합니다. `--output-dir` 은 `book/` 기준 상대 경로입니다.

PDF는 `book/build/neuqes-101-ch01-34-manuscript.pdf`에 생성됩니다.

EPUB 리플로우 미리보기 샘플은 다음 명령으로 생성합니다.

```bash
./book/build-epub.sh
```

현재 EPUB 샘플은 1-34장 변환 검증용이며, 산출물은 `book/build/epub/neuqes-101-ch01-34-preview.epub`에 생성됩니다. PDF용 원고와 EPUB용 진입점을 분리해 두었기 때문에, 리디북스 같은 EPUB 뷰어에서 본문, 코드, 출력, 표, 그림의 읽힘을 먼저 확인한 뒤 범위를 조정할 수 있습니다.

EPUB 표지 후보는 `book/epub/covers/` 아래 여러 종으로 생성됩니다. 현재 기본 표지는 `cover-illustration-token-core.png`이며, 후보를 바꾸려면 `book/build-epub.sh`의 `--epub-cover-image` 값을 원하는 PNG로 바꿔 다시 빌드합니다.

## 구조

- `main.tex`: 책 전체 진입점
- `designsystem.sty`: 박스, 코드 블록, 표/그림, 표지, Phase 속지 명령
- `themes/`: 색상 팔레트 테마. 현재 기본값은 `slate.sty` (Cool Slate)
- `preamble/`: 폰트, 페이지 레이아웃, 색인 스타일
- `frontmatter/`: 표지, 속표지, 서문
- `chapters/`: 1-34장 출판용 원고
- `appendices/`: 검증 전 부록 노트북의 자리 표시 원고
- `backmatter/`: 마무리와 색인
- `tools/notebook_to_tex.py`: 노트북 원천에서 장 원고를 재생성하는 스크립트
- `ebook-main.tex`: EPUB 리플로우 미리보기용 진입점
- `epub/`: EPUB 변환용 shim과 CSS
- `build-epub.sh`: EPUB 미리보기 빌드 스크립트

본문 폰트는 `NanumGothic`, 코드 폰트는 `NanumGothicCoding` 파일을 직접 지정합니다.

테마는 `main.tex`의 `\booktheme` 값으로 선택합니다. 지금은 `slate` 한 벌만 유지하며, 본문 원고는 그대로 두고 `themes/<name>.sty`만 추가하면 다른 색상 테마를 실험할 수 있게 분리해 두었습니다.

## 조판 규칙

- FAQ의 각 질문과 답변은 `faqBox`로 묶어 하나의 시각 단위로 표시합니다.
- 다음 장 힌트와 장 끝 예고는 `previewBox`로 묶어 “미리보기”처럼 표시합니다.
- 박스류는 본문 폭 기준으로 고정합니다. 제목 앞 기호는 미리보기 박스에만 사용합니다.
- 인라인 코드는 `inlinecode` 매크로로 회색 음영 처리합니다.
- `texttt`로 들어오는 verbatim 계열 항목도 회색 음영 처리합니다.
- 코드 셀과 마크다운 코드 펜스는 모두 `lstlisting` 블록으로 표시합니다.
- 실습·해부·토크나이저 노트 코드 뒤에는 박스가 아닌 본문 산문으로 행 번호, 주요 코드 조각, 설명을 함께 제공합니다.
- 노트북을 `--execute`로 변환하면 코드 읽기 뒤에 핵심 출력과 출력 해석을 함께 붙입니다.
- 넓은 표는 `adjustbox`로 페이지 폭 안에 맞춥니다.
- 표시 수식은 번호가 있는 `equation` 환경으로 변환하고, 설명 문장에서 `eqref`로 참조합니다.
- 수식 글꼴은 본문과 맞도록 sans-serif 계열로 통일합니다.
- 유니코드 수학 기호는 가능한 한 LaTeX 명령으로 씁니다. 예: `λ` 대신 `$\lambda$`, `Δ` 대신 `$\Delta$`, `≈` 대신 `$\approx$`. 코드와 출력 문자열에서는 `lambda`, `delta`, `<=`, `->` 같은 ASCII 표기를 우선합니다.
- 이미지와 TikZ 다이어그램은 각각 `bookfigure`, `bookdiagram` 환경을 사용하면 LaTeX `figure` 번호와 캡션으로 관리됩니다.
- `plt.show()`가 있는 실습 코드는 원칙적으로 대응하는 그림을 `book/assets/figures/`에 생성하거나 추출해 `bookfigurelabel`로 본문에 추가합니다.
