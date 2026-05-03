# 출판용 LaTeX 원고

이 폴더는 1-14장 Colab 노트북을 원천으로 만든 출판용 LaTeX 프로젝트입니다.

## 빌드

레포 루트에서 실행합니다.

```bash
python3 book/tools/notebook_to_tex.py --execute
latexmk -xelatex book/main.tex
```

PDF는 `book/build/neuqes-101-ch01-14-manuscript.pdf`에 생성됩니다.

## 구조

- `main.tex`: 책 전체 진입점
- `preamble/`: 폰트, 코드, 박스, 색인 스타일
- `frontmatter/`: 표지, 속표지, 서문
- `chapters/`: 1-14장 출판용 원고
- `appendices/`: 검증 전 부록 노트북의 자리 표시 원고
- `backmatter/`: 마무리와 색인
- `tools/notebook_to_tex.py`: 노트북 원천에서 장 원고를 재생성하는 스크립트

본문 폰트는 `NanumGothic`, 코드 폰트는 `NanumGothicCoding` 파일을 직접 지정합니다.

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
- 이미지와 TikZ 다이어그램은 각각 `bookfigure`, `bookdiagram` 환경을 사용하면 LaTeX `figure` 번호와 캡션으로 관리됩니다.
- `plt.show()`가 있는 실습 코드는 원칙적으로 대응하는 그림을 `book/assets/figures/`에 생성하거나 추출해 `bookfigurelabel`로 본문에 추가합니다.
