# hfcover.sty — A1 표지 LaTeX 템플릿

`covers.jsx` 의 **A1** 시안을 LaTeX 로 옮긴 표지 스타일. 책 본 빌드(`book/main.tex`)에 통합되거나, 단독 (`standalone-example.tex`) 으로도 시연 가능합니다.

```
book/cover/
├── hfcover.sty               ← 표지 스타일 (TikZ 로 그려진 A1 레이아웃)
├── standalone-example.tex    ← 단독 빌드 시연 (B5 페이지 + Nanum Gothic 본문)
└── README.md
```

## 빌드

### 책 본 빌드 (Phase 0-1 manuscript 의 표지로 들어감)

`book/main.tex` 의 preamble 에서 `\usepackage{cover/hfcover}` 가, `book/frontmatter/cover.tex` 가 `\drawcover` 를 호출합니다. 책 빌드는 일반적으로:

```bash
latexmk -xelatex book/main.tex
```

산출물: `book/build/neuqes-101-ch01-34-manuscript.pdf` 의 첫 페이지가 이 표지.

### 단독 빌드 (표지만 시연)

```bash
cd book/cover
xelatex standalone-example.tex
```

`\usepackage[standalone]{hfcover}` 가 자체 geometry (B5 176×250 mm) + Nanum Gothic 본문 폰트 셋업까지 함께 처리.

엔진은 **XeLaTeX 필수** (한글 + fontspec, 단독 모드에선 xeCJK 도).

## 필요한 폰트

시스템에 설치되어 있어야 합니다:

| 용도 | 패밀리 |
|---|---|
| 본문 / 디스플레이 | Nanum Gothic, Nanum Gothic ExtraBold |
| 코드 / 모노 | Nanum Gothic Coding |
| 손글씨 액센트 (배너 카피) | Nanum Pen Script |

### macOS

```bash
brew install --cask font-nanum-gothic font-nanum-gothic-coding font-nanum-pen-script
```

설치 후 패밀리 이름이 *공백 없는* 형태 (`NanumGothic`, `NanumGothicCoding`) 로 노출됩니다 — `hfcover.sty` 가 `\IfFontExistsTF` 로 자동 감지.

### Ubuntu / Debian

```bash
sudo apt install fonts-nanum fonts-nanum-coding
# Nanum Pen Script 은 패키지에 없음. Google Fonts 에서 직접:
mkdir -p ~/.fonts && cd ~/.fonts
wget https://github.com/google/fonts/raw/main/ofl/nanumpenscript/NanumPenScript-Regular.ttf
fc-cache -f -v
```

설치 후 패밀리 이름이 *공백 있는* 형태 (`Nanum Gothic`, `Nanum Gothic Coding`) 로 노출됩니다 — `hfcover.sty` 가 자동으로 이쪽 분기 사용.

### CI 환경 (GitHub Actions Ubuntu runner 등)

위 Ubuntu 절차를 워크플로 step 으로:

```yaml
- name: Install Korean fonts
  run: |
    sudo apt-get update
    sudo apt-get install -y fonts-nanum fonts-nanum-coding texlive-xetex texlive-fonts-extra latexmk
    mkdir -p ~/.fonts
    curl -sL -o ~/.fonts/NanumPenScript-Regular.ttf \
      https://github.com/google/fonts/raw/main/ofl/nanumpenscript/NanumPenScript-Regular.ttf
    fc-cache -f
```

폰트가 없으면 `fontspec` 가 build 시점에 *명확한 오류 메시지* 를 띄우고 종료. 조용히 깨지지 않음.

## 커스터마이즈

`\drawcover` 호출 전에 다음 매크로를 `\renewcommand` 로 덮어쓰면 됩니다:

| 매크로 | 기본값 |
|---|---|
| `\hfTitleLineOne` | 텍스트 분석을 위한 |
| `\hfTitleLineTwo` | Hugging Face 입문 |
| `\hfSubtitle` | 손으로 따라가며 익히는 텍스트 분류 · 26장 원고 |
| `\hfRunMark` | HF · 101 |
| `\hfVersionMark` | v1.0 · 2026 |
| `\hfBandKicker` | Hugging Face 입문 커리큘럼 · 26장 원고 (손글씨 톤) |
| `\hfBlurb` | 4줄 본문 카피 |
| `\hfAuthor` | 황윤구 |
| `\hfAxisOneK` … `\hfAxisFourV` | 4축 칩 키/값 |

또는 한 번에:

```latex
\makehfcover{텍스트 분석을 위한}{Hugging Face 입문}%
  {손으로 따라가며 익히는 텍스트 분류 · 20챕터}
```

## 색 팔레트 (Hugging Face)

| 토큰 | HEX |
|---|---|
| `hfPaper` | `#FFFDF6` |
| `hfInk` | `#1F1F1F` |
| `hfAccent` (HF 오렌지) | `#FF9D00` |
| `hfAccentTwo` (HF 옐로우) | `#FFD21E` |
| `hfAccentSoft` | `#FFF1C2` |

## 패키지 옵션

`\usepackage{hfcover}` (기본, *book 모드*):
- 호출 측 (book/main.tex) 의 geometry / 본문 폰트 / kotex 같은 한글 처리는 *건드리지 않음*.
- cover-only 매크로폰트 (`\hfDisplay` / `\hfMono` / `\hfPen`) 만 새로 정의.
- `xeCJK` *비로드* — kotex 와 충돌 회피.

`\usepackage[standalone]{hfcover}` (*standalone 모드*):
- 자체 geometry (B5 176×250 mm) + `\setmainfont{NanumGothic}` 셋업.
- `xeCJK` 로드 (단독 빌드라 다른 한글 패키지 없음).

## 한계 / 메모

- 🤗 이모지는 LaTeX 에서 직접 렌더링이 까다로워 `HF` 텍스트 마크로 대체했습니다. 이모지가 필요하면 `\node` 의 텍스트를 `{\fontspec{Apple Color Emoji}🤗}` 같은 식으로 OS 의 컬러 이모지 폰트를 지정해 사용.
- 점 패턴은 4mm 그리드 1점 — TikZ `\foreach` 로 그렸습니다. 인쇄 시 해상도 문제 없음.
- **카드 그림자**: 처음에는 TikZ `shadows.blur` 라이브러리의 `blur shadow` 를 썼으나 book 빌드 환경 (`kotex` + `sansmath` + `fontspec` 조합) 에서 `\pgf@selectfontorig` 무한 재귀를 일으켰습니다. 단순 *오프셋 사각형 두 장* 으로 부드러운 단계적 농도 차이를 만드는 방식으로 대체. 시각적으로는 거의 동일.
