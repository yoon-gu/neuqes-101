#!/usr/bin/env python3
"""노트북(.ipynb)을 WikiDocs 연동용 "장→절" 다중 페이지 마크다운으로 변환합니다.

`book/tools/notebook_to_md.py`의 장→절 분할 규칙을 이어받되, 핵심 차이는
**코드 실행 결과(표·로그·그림)를 함께 싣는다**는 점입니다. 노트북은 보통 출력이
비어 있으므로(Colab용 clean 상태), 다음 우선순위로 "실제 결과"를 확보합니다.

출력 원천 우선순위 (챕터별 자동):
  1) --executed-notebook PATH    : (단일 챕터) 미리 실행해 outputs를 담은 노트북
  2) executed/<폴더>.ipynb 존재   : 자동으로 출력 원천으로 사용
                                    (Colab/GPU에서 끝까지 돌린 뒤 저장·커밋한 실행본)
  3) --execute                   : 이 자리에서 nbclient로 직접 실행(주로 CPU 챕터).
                                    --save-executed 면 결과를 executed/<폴더>.ipynb 로 저장.
  4) (없음)                      : 노트북에 든 outputs만 사용. 없으면 코드만 출력하고
                                    "<!-- 실행 결과 없음 -->" 주석을 남겨 누락을 드러냄
                                    (가짜 출력을 지어내지 않음 — "파싱만" 금지 요구의 핵심).

실행본 보관 규약: GPU 챕터의 진짜 결과는 Colab T4에서 끝까지 돌린 뒤
"파일 > .ipynb 다운로드"(출력 포함)한 노트북을 `executed/<폴더>.ipynb` 로 커밋해 둔다.
챕터 폴더에는 clean 노트북만 남긴다(Colab 버튼 대상). 자세한 건 executed/README.md.

챕터 지정 (동적):
  - 위치 인자로 챕터를 받음: 폴더명(`07_bert_pipeline`), 번호(`7`/`07`) 모두 허용. 여러 개 가능.
  - 아무 챕터도 안 주고 `--all`도 없으면 에러 — 호출자가 의도(전체/일부)를 명시하게 함.
  - `--all` 이면 레포 루트의 `NN_slug/NN_slug.ipynb` 를 전부 자동 발견해 변환.
  - 챕터 메타(제목): book/tools/notebook_to_tex.py 의 CHAPTERS 레지스트리 → 노트북 첫 H1
    ("Chapter N." 접두 제거) → 슬러그 순으로 해석. 레지스트리에 없는 새 챕터도 동작.

사용:
  # 전체 (호출자가 사용자 확인 후)
  python3 build_wikidocs.py --all --execute
  # 일부
  python3 build_wikidocs.py 1 7 15 --execute
  python3 build_wikidocs.py 07_bert_pipeline --executed-notebook 07_bert_pipeline/07_bert_pipeline.executed.ipynb
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import traceback
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

# 레포 루트: 이 스크립트는 .claude/skills/notebook-to-wikidocs/scripts/ 아래 있음 → parents[4]
ROOT = Path(__file__).resolve().parents[4]


def _github_repo() -> str:
    """git origin 에서 owner/repo 를 자동 인식(실패 시 빈 문자열). Colab 버튼 URL 용."""
    try:
        url = subprocess.check_output(
            ["git", "-C", str(ROOT), "remote", "get-url", "origin"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""
    m = re.search(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?/?$", url)
    return m.group(1) if m else ""


GITHUB_REPO = _github_repo()


def _colab_button(folder: str) -> str:
    """실습 페이지 맨 위에 넣을 Colab '바로 열기' 링크.
    배지 이미지(외부 SVG)는 EPUB/PDF에서 깨지므로 전자책 안전한 텍스트 링크로 둔다(린터 E4 회피).
    repo 를 못 찾으면 빈 문자열 → 링크 생략."""
    if not GITHUB_REPO:
        return ""
    url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/master/{folder}/{folder}.ipynb"
    return f"> ▶ **[Google Colab에서 이 장 실습 열기]({url})** — 브라우저에서 바로 실행해 볼 수 있습니다."


COLAB_BADGE_RE = re.compile(r"^\s*\[!\[.*?Colab.*?\]\(.*?\)\]\(.*?\)\s*$", re.IGNORECASE)
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
EMOJI_RE = re.compile(r"^[\s←-⇿⌀-➿⬀-⯿️\U0001F000-\U0001FAFF]+")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
CHAPTER_FOLDER_RE = re.compile(r"^(\d{2})_(.+)$")
H1_CHAPTER_PREFIX_RE = re.compile(r"^\s*Chapter\s+\d+\s*[.．]\s*")

SKIP_PATTERNS = (
    "TqdmWarning:",
    "IProgress not found",
    "Requirement already satisfied:",
    "WARNING: Running pip",
    "[notice] A new release of pip",
    "notice] A new release of pip",
    "To update, run:",
    # Hugging Face Hub 인증/다운로드 경고 (Colab 환경 노이즈 — 책 내용 아님)
    "huggingface_hub/utils/_auth.py",
    "secret value from your vault",
    "not authenticated with the Hugging Face Hub",
    "If the error persists, please let us know",
    "warnings.warn(",
    "unauthenticated requests to the HF Hub",
    # transformers 생성(generation) 보일러플레이트 경고 — 교육적 의미 없음.
    # (주의: "Some weights … newly initialized … should TRAIN" 류는 의도적 교육 포인트일 수
    #  있어 일부러 제외함 — 노이즈로 싸잡아 지우지 않는다.)
    "Setting `pad_token_id`",
    "`max_new_tokens`",
    "clean_up_tokenization_spaces",
    "Passing `generation_config`",
    "aligned accordingly, being updated with the tokenizer",  # genconfig 정렬 보일러플레이트(노이즈)
)

# tqdm 진행바(다운로드·맵핑 등): `README.md:  0%|...| [00:00<?, ?B/s]` 류 — 책 내용 아님.
TQDM_BAR_RE = re.compile(r"\d+%\s*\|")

# 긴 '산문' 출력 줄(리뷰 샘플·생성문 등)은 트렁케이트 — EPUB 은 <pre> 안을 안 접어 잘림.
# 표(정렬 컬럼)는 자르면 정렬이 깨지므로 제외한다. 한 블록에 표와 산문이 섞여 있어도
# 줄별로 판별: 표 행은 토큰이 적고(숫자 몇 개), 산문은 공백 분리 단어가 많다.
MAX_OUTPUT_LINE_CHARS = 160      # 이보다 길면 트렁케이트 대상
TRUNC_KEEP_CHARS = 140           # 자른 뒤 남기는 앞부분 길이
PROSE_MIN_TOKENS = 12            # 공백 분리 토큰 이만큼+ 면 산문으로 보고 자름(표는 토큰 적음)

# 트렁케이트 opt-out 챕터 — '출력 자체가 학습 내용'인 토큰화 메커니즘 챕터.
# (근거: 출력 자체가 학습 내용인 챕터 — ch08 의 1602>512 indexing 경고·토큰 ID 리스트,
#  ch15 의 한국어 토큰 분절 비교가 잘리면 챕터 교훈이 깨짐. 향후 ch19·22 등 추가.)
NO_TRUNCATE_CHAPTERS = {"08_tokenizer_datasets", "15_ko_binary", "22_ko_bert_pretrain"}


def _truncate_long_lines(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        if (len(ln) > MAX_OUTPUT_LINE_CHARS and "|" not in ln
                and len(ln.split()) >= PROSE_MIN_TOKENS):  # 산문 줄만 (표 행은 보존)
            omitted = len(ln) - TRUNC_KEEP_CHARS
            ln = ln[:TRUNC_KEEP_CHARS].rstrip() + f" …(뒤 {omitted}자 생략)"
        out.append(ln)
    return out

MAX_OUTPUT_LINES = 40
MAX_OUTPUT_CHARS = 2000

SECTION_RULES: list[tuple[str, str]] = [
    ("삽질", "wrapup"),
    ("라이브러리", "wrapup"),
    ("체크포인트", "wrapup"),
    ("FAQ", "wrapup"),
    ("다음 챕터", "wrapup"),
    ("다음 장", "wrapup"),
    ("예고", "wrapup"),
    ("실습", "practice"),
    ("해부", "anatomy"),
    ("변형", "variation"),
]

SUBPAGES = [
    ("practice", "practice", "실습"),
    ("anatomy", "anatomy", "해부"),
    ("variation", "variation", "변형"),
    ("wrapup", "wrapup", "정리와 FAQ"),
]

DEFAULT_BOOK_TITLE = "neuqes-101 — Hugging Face 입문 커리큘럼"


# --------------------------------------------------------------------------- #
# 텍스트 유틸
# --------------------------------------------------------------------------- #
def _cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _strip_emoji(text: str) -> str:
    return EMOJI_RE.sub("", text).strip()


def _clean_heading_text(text: str) -> str:
    """헤더 텍스트 정리: 선두 'N.'/'N)' 순번 제거 → 선두 이모지 제거.
    절 제목 중복("07-1. 1. 🚀 실습")과 이모지 잔존을 막는다. (예: '1. 🚀 실습: …' → '실습: …')
    """
    text = re.sub(r"^\s*\d+[.)]\s*", "", text.strip())
    return EMOJI_RE.sub("", text).strip()


SUBTITLE_LABELS = {"practice": "실습", "anatomy": "해부", "variation": "변형"}


def _normalize_subtitle(group: str, text: str) -> str:
    """절 제목을 'label: 소개' 형식으로 통일한다.
    노트북 헤더가 '실습 2:', '변형 —', '… 실습'(말미)처럼 제각각이어도
    키워드+순번+구분자를 떼어내 'label: 나머지'로 맞춘다. 나머지가 없으면 label만.
    예: '변형 — λ 스윕' → '변형: λ 스윕', 'Collator 추가 실습' → '실습: Collator 추가'.
    """
    label = SUBTITLE_LABELS.get(group)
    if not label:
        return text
    rest = re.sub(rf"^{label}\s*\d*\s*[:\-–—]?\s*", "", text)  # 선두 '실습 2:' 류 제거
    rest = re.sub(rf"\s*{label}\s*$", "", rest)                # 말미 '… 실습' 제거
    rest = rest.strip(" :-–—")
    return f"{label}: {rest}" if rest else label


def _first_header(md: str) -> tuple[int, str] | None:
    for line in md.splitlines():
        m = HEADER_RE.match(line)
        if m:
            return len(m.group(1)), m.group(2).strip()
    return None


def _classify(header_text: str) -> str:
    for kw, group in SECTION_RULES:
        if kw in header_text:
            return group
    return "overview"


# --- 선형(위→아래) 절 분류 상태 기계 -------------------------------------- #
# 노트북을 위에서 아래로 읽으며 개요(장) → 본문 절들 → 정리(wrapup) 세 구간으로
# 나눈다. 본문 절은 실습→해부→변형 순서로만 진행(단조)하며, 마커 없는 절은
# 직전 절을 그대로 물려받는다. Ch 7까지의 '실습/해부/변형' 키워드와, Ch 9부터의
# '이모지(🚀/🔬/🛠️) + 내용 이름' 헤더를 모두 같은 규칙으로 처리한다.
_BODY_ORDER = {"practice": 0, "anatomy": 1, "variation": 2}
_WRAPUP_START_KW = ("라이브러리", "체크포인트", "FAQ", "삽질",
                    "다음 챕터", "다음 장", "예고", "회고", "마무리")
_NUM_HEAD_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s")


def _is_wrapup_start(h: str) -> bool:
    return any(k in h for k in _WRAPUP_START_KW)


def _is_body_start(h: str) -> bool:
    """본문(첫 실습/번호 절)의 시작 헤더인가."""
    return (bool(_NUM_HEAD_RE.match(h)) or "환경 셋업" in h or "환경 준비" in h
            or ("🚀" in h and "실습" in h))


def _body_group(h: str) -> str | None:
    """본문 헤더를 실습/해부/변형으로 분류. 못 정하면 None(직전 절 유지)."""
    if "환경 셋업" in h or "환경 준비" in h:
        return "practice"
    if "🔬" in h or "해부" in h or "해석" in h:
        return "anatomy"
    if "변형" in h or "클라이맥스" in h or "🛠️" in h:
        return "variation"
    if "🚀" in h or "실습" in h:
        return "practice"
    return None


def _strip_colab_badge(md: str) -> str:
    return "\n".join(ln for ln in md.splitlines() if not COLAB_BADGE_RE.match(ln)).strip("\n")


def _demote_first_header(md: str) -> str:
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if HEADER_RE.match(line):
            del lines[i]
            break
    return "\n".join(lines).strip("\n")


def _strip_header_emoji(md: str) -> str:
    out = []
    for line in md.splitlines():
        m = HEADER_RE.match(line)
        if m:
            out.append(f"{m.group(1)} {_clean_heading_text(m.group(2))}")
        else:
            out.append(line)
    return "\n".join(out)


# 전자책 작성 규칙([wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723)) 방어용 패턴
HR_LINE_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
H1_LINE_RE = re.compile(r"^#\s+(.*)$")
HEADING_RE = re.compile(r"^#{1,6}\s")  # [2] 모든 헤딩 (위아래 빈 줄 보장용)
WIN_PATH_RE = re.compile(r"[A-Za-z]:\\[\w.\\-]+")   # [W1] 진짜 윈도우 경로 (LaTeX \, 등 제외)
CODE_MATH_RE = re.compile(r"`[^`]*`|\$[^$]+\$")     # 인라인 코드/수식 (경로 방어 시 마스킹)
RAW_HTML_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
EXT_IMG_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
FOOTNOTE_RE = re.compile(r"\[\^([^\]]+)\]")


def _wrap_win_paths(ln: str, stats: dict) -> str:
    """코드/수식 밖의 진짜 윈도우 경로(C:\\...)를 인라인 코드로 감싼다 (W1).
    LaTeX 수식 속 `i:\\,` 류는 코드/수식 마스킹으로 건너뛴다."""
    spans: list[str] = []

    def _stash(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    masked = CODE_MATH_RE.sub(_stash, ln)
    cnt = [0]

    def _wrap(m):
        cnt[0] += 1
        return f"`{m.group(0)}`"

    masked = WIN_PATH_RE.sub(_wrap, masked)
    if not cnt[0]:
        return ln
    stats["win_paths"] += cnt[0]
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], masked)


def _sanitize_md_cell(md: str, stem: str, stats: dict) -> str:
    """마크다운 셀을 전자책 규칙([wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723))에 맞게 방어 정리한다. 코드펜스 안은 손대지 않음.

    - [11] 코드펜스 밖 수평선(---/***/___) 제거 (앞이 빈 줄일 때만 — setext 헤딩 오인 방지).
    - [1]  2번째 이후 H1(#) → H2(##) 강등 (본문 H1 금지; 첫 제목 H1은 convert 가 따로 제거).
    - [10] 각주 이름에 챕터 stem 접두 → 전자책이 전 페이지를 한 문서로 통합할 때 충돌 방지.
    - [6]/[3] raw HTML·외부 이미지는 자동 수정이 어려워 stats 에 경고만 모은다(인라인 코드는 제외).
    """
    out: list[str] = []
    fence = False
    pending_blank = False  # 직전이 헤딩 → 다음 비공백 줄 앞 빈 줄 보장 [2]
    for ln in md.split("\n"):
        if pending_blank:  # [2] 헤딩 아래 빈 줄 (이미 빈 줄이면 중복 삽입 안 함)
            if ln.strip() != "":
                out.append("")
                stats["heading_blanks"] += 1
            pending_blank = False
        if ln.lstrip().startswith("```"):
            fence = not fence
            out.append(ln)
            continue
        if fence:
            out.append(ln)
            continue
        if HR_LINE_RE.match(ln) and (not out or out[-1].strip() == ""):
            stats["hr_removed"] += 1
            continue
        m = H1_LINE_RE.match(ln)
        if m:
            stats["h1_demoted"] += 1
            ln = "## " + m.group(1)
        if HEADING_RE.match(ln):  # [2] 헤딩 위 빈 줄 + 아래 빈 줄 예약
            if out and out[-1].strip() != "":
                out.append("")
                stats["heading_blanks"] += 1
            pending_blank = True
        if ":\\" in ln:  # [W1] 코드/수식 밖 윈도우 경로 → 인라인 코드
            ln = _wrap_win_paths(ln, stats)
        if "[^" in ln:
            new = FOOTNOTE_RE.sub(lambda x: f"[^{stem}-{x.group(1)}]", ln)
            if new != ln:
                stats["footnotes"] += 1
                ln = new
        scan = INLINE_CODE_RE.sub("", ln)
        stats["html_warn"].extend(RAW_HTML_RE.findall(scan))
        stats["extimg_warn"].extend(EXT_IMG_RE.findall(scan))
        out.append(ln)
    return "\n".join(out)


def _clean_text_output(text: str, truncate: bool = True) -> str:
    text = ANSI_RE.sub("", text)
    lines = [seg.split("\r")[-1] for seg in text.split("\n")]
    lines = [
        ln for ln in lines
        if not any(p in ln for p in SKIP_PATTERNS)
        and not ln.strip().startswith("from .autonotebook import tqdm")
        and not TQDM_BAR_RE.search(ln)
    ]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if truncate:
        lines = _truncate_long_lines(lines)
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[: MAX_OUTPUT_LINES - 1] + [
            f"... (출력 {len(lines) - MAX_OUTPUT_LINES + 1}줄 생략) ..."
        ]
    text = "\n".join(lines)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[: MAX_OUTPUT_CHARS - 4].rstrip() + "\n..."
    return text


def latex_title_to_plain(title: str) -> str:
    """레지스트리 제목의 LaTeX 이스케이프 해제: '\\&' → '&', '\\_' → '_' 등."""
    return (
        title.replace("\\&", "&").replace("\\_", "_")
        .replace("\\%", "%").replace("\\#", "#").replace("\\$", "$")
    )


# --------------------------------------------------------------------------- #
# HTML 표 → 마크다운 표
# --------------------------------------------------------------------------- #
class _PandasTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict[str, list]] = []
        self.in_table = self.in_row = self.in_cell = False
        self.cell_is_header = False
        self.current_cell: list[str] = []
        self.current_row: list[tuple[bool, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.tables.append({"headers": [], "rows": []})
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_table and self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_is_header = tag == "th"
            self.current_cell = []
        elif self.in_cell and tag == "br":
            self.current_cell.append(" ")

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.in_cell:
            text = unescape("".join(self.current_cell))
            text = re.sub(r"\s+", " ", text).strip()
            self.current_row.append((self.cell_is_header, text))
            self.in_cell = False
            self.current_cell = []
        elif tag == "tr" and self.in_row:
            if self.current_row and self.tables:
                values = [v for _, v in self.current_row]
                header_count = sum(1 for is_h, _ in self.current_row if is_h)
                data_count = len(self.current_row) - header_count
                table = self.tables[-1]
                if header_count >= data_count:
                    table["headers"] = values
                else:
                    table["rows"].append(values)
            self.in_row = False
            self.current_row = []
        elif tag == "table":
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def _html_tables_to_text(html: str) -> list[str]:
    """HTML 표 → 공백 정렬된 모노스페이스 텍스트(코드펜스에 넣어 블록인용 안전).
    text/plain이 없을 때의 폴백."""
    parser = _PandasTableParser()
    parser.feed(html)
    out: list[str] = []
    for table in parser.tables:
        headers, rows = table["headers"], table["rows"]
        if not rows:
            continue
        width = max([len(headers)] + [len(r) for r in rows])
        headers = (headers + [""] * width)[:width] if headers else [""] * width
        shown = [(r + [""] * width)[:width] for r in rows[:30]]
        grid = [headers] + shown
        colw = [max(len(str(row[c])) for row in grid) for c in range(width)]
        def fmt(row): return "  ".join(str(row[c]).ljust(colw[c]) for c in range(width)).rstrip()
        lines = [fmt(headers)] + [fmt(r) for r in shown]
        if len(rows) > 30:
            lines.append("...")
        out.append("\n".join(lines))
    return out


# --------------------------------------------------------------------------- #
# 셀 출력 렌더링
# --------------------------------------------------------------------------- #
# 실행 결과 박스 스타일 — 회색 코드블록과 구분되도록 왼쪽 색깔 바 + 옅은 배경.
# WikiDocs는 HTML+style 을 지원하고 highlight.js로 코드만 색칠하므로, 출력은 <pre>로 둔다.
# (블록인용 안 코드펜스는 WikiDocs에서 ``` 가 노출되어 사용 불가.)
OUTPUT_PRE_STYLE = (
    "background:#eef3fb;border-left:4px solid #5B8DEF;"
    "padding:0.7em 1em;border-radius:4px;overflow-x:auto;"
    "font-size:0.92em;line-height:1.45;"
)
OUTPUT_LABEL = "▶ 실행 결과"
SYNTH_LABEL = "▶ 출력 형태"

# 출력 박스 표현 방식 (변환 타깃별). 실측(DESIGN_NOTES §7-5~7-8) 결과 세 타깃(웹/PDF/EPUB)을
# 모두 만족하는 건 code 뿐이라 기본값으로 둔다:
#   code       : 평범한 코드펜스(```text) + "▶ 실행 결과" 라벨. 웹·PDF·EPUB 모두 정상 동작.
#                색깔 박스는 없지만 라벨 + (코드는 python 하이라이트 / 출력은 plain)로 구분. → 기본값.
#   fenced-div : "::: {.output}" + 코드펜스. EPUB/PDF(pandoc)에선 색깔 박스로 살아나나,
#                WikiDocs '웹' 마크다운은 fenced div 를 몰라 ::: 가 글자로 노출됨(2026-06-11 실측 확인).
#   html-box   : "<pre style>" 색깔 박스. WikiDocs '웹'에선 잘 보이나 전자책(pandoc)에선 드롭/깨짐
#                (공식 문서 [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723): "HTML 코드는 전자책 변환 시 정상 표시되지 않습니다").
OUTPUT_STYLES = ("code", "fenced-div", "html-box")
DEFAULT_OUTPUT_STYLE = "code"


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _output_box(text: str, style: str) -> str:
    """출력 텍스트 한 덩어리를 선택된 스타일의 박스로 감싼다.

    fenced-div / code 는 내부 코드펜스가 출력의 공백·`<`·`>` 를 이스케이프 없이 보존한다
    (표 정렬 OK). 출력에 ``` 가 들어 있으면 더 긴 펜스로 회피한다.
    """
    if style == "html-box":
        return f'<pre style="{OUTPUT_PRE_STYLE}">{_html_escape(text)}</pre>'
    fence = "```"
    while fence in text:
        fence += "`"
    if style == "code":
        return f"{fence}text\n{text}\n{fence}"
    # fenced-div (기본)
    return f"::: {{.output}}\n{fence}text\n{text}\n{fence}\n:::"


_SYNTH_FN = "uninit"  # notebook_to_tex.synthetic_output_text (지연 import, 캐시)


def _synthetic_output_text(source: str) -> str:
    """기존 ipynb→tex 합성 로직 재사용. tex와 동일하게 `print(`가 있는 셀에만 적용.

    실제 실행 결과가 없을 때 '출력은 이런 모양' 골격(값은 ...)을 보여준다.
    로직을 복제하지 않고 book/tools/notebook_to_tex.py 의 함수를 그대로 가져온다.
    """
    global _SYNTH_FN
    if _SYNTH_FN == "uninit":
        try:
            sys.path.insert(0, str(ROOT / "book" / "tools"))
            import notebook_to_tex as t  # noqa: E402
            _SYNTH_FN = t.synthetic_output_text
        except Exception:
            _SYNTH_FN = None
    if _SYNTH_FN is None or "print(" not in source:
        return ""
    try:
        return _SYNTH_FN(source) or ""
    except Exception:
        return ""


def _synthetic_block(source: str, style: str = DEFAULT_OUTPUT_STYLE) -> str:
    text = _clean_text_output(_synthetic_output_text(source))
    if not text.strip():
        return ""
    return f"**{SYNTH_LABEL}**\n\n" + _output_box(text, style)


def _next_image_version(assets_dir: Path | None, base: str, new_bytes: bytes) -> tuple[str, list[Path], bool]:
    """`base`(예: '01-tfidf-out1')의 버전 파일명과 지울 옛 버전 목록, 그리고 새로 쓸
    필요가 없는지(unchanged) 를 반환.

    WikiDocs는 파일명이 같으면 내용이 달라도 캐시된 이미지를 그대로 보여주므로,
    이미지가 실제로 바뀌었을 때만 `-1` → `-2` → `-3` 으로 버전 postfix를 올려
    매번 새 파일명을 만든다(예: '01-tfidf-out1-2.png'). 페이지가 더는 참조하지
    않는 옛 버전 파일은 함께 지운다.

    최신 버전 파일이 이미 있고 그 바이트가 새로 렌더링된 이미지와 **완전히 같으면**
    (재실행이 시드 고정 등으로 픽셀까지 동일한 그림을 다시 낸 경우) 버전을 올리지
    않고 기존 파일명을 그대로 재사용한다 — 그림 내용이 안 바뀌었는데도 재변환마다
    버전이 올라가 자산 diff 만 불필요하게 늘어나는 것을 막는다.

    assets_dir 가 없으면(드라이런) 디스크를 못 보므로 버전 1 이름만 돌려준다.
    `{base}-` 뒤에 숫자가 오는 파일만 같은 출력의 버전으로 인식하므로
    'out1' 과 'out11' 처럼 번호가 다른 출력끼리는 섞이지 않는다.
    """
    if assets_dir is None or not assets_dir.exists():
        return f"{base}-1.png", [], False
    pat = re.compile(rf"^{re.escape(base)}-(\d+)\.png$")
    olds: list[Path] = []
    max_v = 0
    latest: Path | None = None
    for p in assets_dir.glob(f"{base}-*.png"):
        m = pat.match(p.name)
        if not m:
            continue
        olds.append(p)
        v = int(m.group(1))
        if v > max_v:
            max_v, latest = v, p
    legacy = assets_dir / f"{base}.png"  # 버전 도입 전 무버전 이름도 함께 정리
    if legacy.exists():
        olds.append(legacy)
    if latest is not None and latest.read_bytes() == new_bytes:
        return latest.name, [], True
    return f"{base}-{max_v + 1}.png", olds, False


def _render_outputs(cell: dict, assets_dir: Path | None, stem: str, counter: list[int],
                    style: str = DEFAULT_OUTPUT_STYLE, truncate: bool = True) -> str:
    """실행 결과를 코드와 구분되는 **색깔 박스**(HTML <pre>)로 렌더링.

    WikiDocs에서 출력 펜스가 코드 펜스와 같은 회색 박스로 보여 헷갈리던 문제를 해결한다.
    <pre>는 공백·줄바꿈을 그대로 보존(표 정렬 OK)하고 highlight.js 색칠 대상이 아니라
    코드블록과 확실히 구분된다. 이미지는 <pre>에 못 넣으므로 라벨 + 마크다운 이미지로.
    """
    items: list[tuple[str, str]] = []  # ("text", str) | ("image", name)
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            text = _clean_text_output("".join(out.get("text", [])), truncate)
            if text.strip():
                items.append(("text", text))
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "image/png" in data:
                counter[0] += 1
                base = f"{stem}-out{counter[0]}"
                raw = data["image/png"]
                raw = raw if isinstance(raw, str) else "".join(raw)
                raw_bytes = base64.b64decode(raw)
                if assets_dir is not None:
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    img_name, old_versions, unchanged = _next_image_version(assets_dir, base, raw_bytes)
                    if not unchanged:
                        (assets_dir / img_name).write_bytes(raw_bytes)
                        for old in old_versions:  # 페이지가 더는 안 가리키는 옛 버전 제거
                            old.unlink()
                else:
                    img_name = f"{base}-1.png"
                items.append(("image", img_name))
                continue
            raw_plain = data.get("text/plain")
            raw_plain = "".join(raw_plain) if isinstance(raw_plain, list) else (raw_plain or "")
            html = data.get("text/html")
            if isinstance(html, list):
                html = "".join(html)
            # Trainer 진행률 위젯 등은 text/plain 이 "<IPython.core.display.HTML object>" 같은
            # 무의미한 repr 뿐이고 진짜 표는 text/html 에 있다 — 이때만 html 표 파싱으로 폴스루.
            is_widget_placeholder = bool(re.match(r"^<IPython\.core\.display\.\w+ object>$", raw_plain.strip()))
            if is_widget_placeholder:
                if isinstance(html, str) and "<table" in html:
                    for t in _html_tables_to_text(html):
                        items.append(("text", t))
                # else: 진행률 바(progress) 처럼 표가 없는 중간 렌더 — 정보 없어 조용히 버림
                continue
            if raw_plain:
                text = _clean_text_output(raw_plain, truncate)
                if text.strip():
                    items.append(("text", text))
                continue
            if isinstance(html, str) and "<table" in html:
                for t in _html_tables_to_text(html):
                    items.append(("text", t))
        elif otype == "error":
            tb = out.get("traceback", [])
            if tb:
                text = _clean_text_output("\n".join(str(l) for l in tb[-8:]), truncate)
            else:
                text = f"{out.get('ename', 'Error')}: {out.get('evalue', '')}"
            if text.strip():
                items.append(("text", text))

    if not items:
        return ""

    # 라벨은 셀당 1번만 맨 위에. 연속 text는 하나의 <pre>로 합치고 image는 그대로(순서 보존).
    blocks: list[str] = [f"**{OUTPUT_LABEL}**"]
    buf: list[str] = []

    def flush_text():
        if buf:
            blocks.append(_output_box("\n".join(buf), style))
            buf.clear()

    for kind, val in items:
        if kind == "text":
            buf.append(val)
        else:
            flush_text()
            blocks.append(f"![output](../assets/{val})")
    flush_text()
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 노트북 실행 (선택)
# --------------------------------------------------------------------------- #
def execute_notebook(path: Path, timeout: int = 1800) -> dict:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    return nb


def _has_any_outputs(nb: dict) -> bool:
    return any(c.get("cell_type") == "code" and c.get("outputs") for c in nb.get("cells", []))


def chapter_h1_title(nb: dict) -> str:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        m = _first_header(_cell_text(cell))
        if m and m[0] == 1:
            return H1_CHAPTER_PREFIX_RE.sub("", m[1]).strip()
    return ""


# --------------------------------------------------------------------------- #
# 변환
# --------------------------------------------------------------------------- #
def convert(nb: dict, num: int, slug: str, title: str,
            pages_dir: Path, assets_dir: Path | None,
            style: str = DEFAULT_OUTPUT_STYLE) -> tuple[list[tuple[str, str]], dict]:
    stem = f"{num:02d}-{slug}"
    truncate = f"{num:02d}_{slug}" not in NO_TRUNCATE_CHAPTERS  # 토큰화 챕터는 원본 길이 유지
    img_counter = [0]
    stats = {"code_cells": 0, "code_with_output": 0, "synthetic": 0, "images": 0,
             "hr_removed": 0, "h1_demoted": 0, "footnotes": 0, "heading_blanks": 0, "win_paths": 0,
             "html_warn": [], "extimg_warn": []}

    groups: dict[str, list[str]] = {
        "overview": [], "practice": [], "anatomy": [], "variation": [], "wrapup": []
    }
    sub_titles: dict[str, str] = {}
    overview_intro: list[str] = []
    setup_code: list[str] = []

    current = "overview"
    seen_h1 = False
    in_body = False    # 첫 본문 절을 지났는가
    in_wrap = False    # 정리(wrapup) 구간에 진입했는가
    setup_mode = False  # 첫 본문 헤더 전, 첫 코드셀 이후의 '환경 준비' 영역

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            md = _strip_colab_badge(_cell_text(cell))
            if not md.strip():
                continue
            hdr = _first_header(md)
            if hdr and hdr[0] == 1 and not seen_h1:
                seen_h1 = True
                body = "\n".join(md.splitlines()[1:]).strip("\n")
                if body.strip():
                    overview_intro.append(_sanitize_md_cell(body, stem, stats))
                continue
            cell_md = _strip_header_emoji(_sanitize_md_cell(md, stem, stats))
            if hdr and hdr[0] == 2:
                setup_mode = False   # 본문/정리 헤더가 나오면 환경 준비 영역 종료
                htext = hdr[1]
                if not in_wrap and _is_wrapup_start(htext):
                    in_wrap, current = True, "wrapup"
                elif in_wrap:
                    current = "wrapup"
                elif not in_body and _is_body_start(htext):
                    in_body = True
                    current = _body_group(htext) or "practice"
                elif in_body:
                    g = _body_group(htext)
                    # 단조 가드: 실습→해부→변형 앞으로만, 뒤로는 안 감.
                    if g and _BODY_ORDER[g] >= _BODY_ORDER.get(current, 0):
                        current = g
                else:
                    current = "overview"
                if current in ("practice", "anatomy", "variation") \
                        and current not in sub_titles:
                    sub_titles[current] = _normalize_subtitle(
                        current, _clean_heading_text(htext))
                groups[current].append(cell_md)
            elif setup_mode and current == "overview":
                # 환경 준비 영역의 헤더 없는 산문(예: 'baseline VRAM —')은
                # 뒤따르는 셋업 코드와 떨어지지 않게 setup_code 와 같은 흐름에 둔다.
                setup_code.append(cell_md)
            else:
                groups[current].append(cell_md)
        elif ctype == "code":
            code = _cell_text(cell).rstrip("\n")
            if not code.strip():
                continue
            stats["code_cells"] += 1
            block = "```python\n" + code + "\n```"
            outs = _render_outputs(cell, assets_dir, stem, img_counter, style, truncate)
            if outs:
                stats["code_with_output"] += 1  # 실제 실행 결과 (executed/ 또는 --execute)
            else:
                # 실제 출력이 없으면 기존 tex 합성 로직 재사용(print 셀에 한해 ... 골격).
                syn = _synthetic_block(code, style)
                if syn:
                    outs = syn
                    stats["synthetic"] += 1
                # 그 외(함수 정의·import 등)는 코드만 남김.
            piece = block + ("\n\n" + outs if outs else "")
            if current == "overview":
                setup_mode = True   # 본문 전 코드 → 환경 준비 영역 시작
                setup_code.append(piece)
            else:
                groups[current].append(piece)

    stats["images"] = img_counter[0]
    pages_dir.mkdir(parents=True, exist_ok=True)
    toc_entries: list[tuple[str, str]] = []

    ov: list[str] = []
    ov.extend(overview_intro)
    ov.extend(groups["overview"])
    present_subs = [(g, sl, sub_titles.get(g, dt)) for g, sl, dt in SUBPAGES
                    if groups[g] or (g == "practice" and setup_code)]
    # 헤딩 아래 빈 줄 — 전자책 PDF 변환 오류 방지([wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723)).
    # [[SubPages]]는 WikiDocs가 하위 페이지 목록을 자동 생성하는 매크로(명시적 링크 목록 대신 사용).
    roadmap = ["## 이 장의 구성", "", "[[SubPages]]"]
    ov.append("\n".join(roadmap))
    (pages_dir / f"{stem}.md").write_text("\n\n".join(ov).strip() + "\n", encoding="utf-8")
    toc_entries.append((f"{num:02d}. {title}", f"pages/{stem}.md"))

    for idx, (g, sl, dt) in enumerate(present_subs, 1):
        parts: list[str] = []
        body_blocks = list(groups[g])
        if g == "practice":
            btn = _colab_button(f"{num:02d}_{slug}")
            if btn:
                parts.append(btn)
        if g == "practice" and setup_code:
            parts.append("## 환경 준비\n\n" + "\n\n".join(setup_code))
        if g in ("practice", "anatomy", "variation") and body_blocks:
            # 첫 헤더가 절 라벨(실습/해부/변형)을 그대로 반복하면 제거(중복 회피).
            # 내용 헤더(데이터 준비·평가 등)면 그대로 살려 절 구조를 보존한다.
            label = SUBTITLE_LABELS.get(g, "")
            fh = _first_header(body_blocks[0])
            if fh and label and fh[1].strip().startswith(label):
                body_blocks[0] = _demote_first_header(body_blocks[0])
        parts.extend(body_blocks)
        (pages_dir / f"{stem}-{sl}.md").write_text(
            "\n\n".join(p for p in parts if p).strip() + "\n", encoding="utf-8")
        t = sub_titles.get(g, dt)
        toc_entries.append((f"{num:02d}-{idx}. {t}", f"pages/{stem}-{sl}.md"))

    # 이번 빌드에서 생성되지 않은 표준 절 페이지(구조 변경으로 남은 고아) 삭제 —
    # 안 그러면 옛 절 내용이 새 절과 중복으로 남는다(예: 해부→변형 재배치 시 옛 anatomy).
    generated = {sl for _, sl, _ in present_subs}
    for suf in ("practice", "anatomy", "variation", "wrapup"):
        if suf not in generated:
            stale = pages_dir / f"{stem}-{suf}.md"
            if stale.exists():
                stale.unlink()

    return toc_entries, stats


# --------------------------------------------------------------------------- #
# TOC
# --------------------------------------------------------------------------- #
def upsert_toc(toc_path: Path, book_title: str, num: int, stem: str,
               entries: list[tuple[str, str]]) -> None:
    """TOC.md에서 이 장(NN. / NN-N.) 블록만 교체하거나 추가. 다른 장은 보존.

    이 장 블록 안에는 이번 실행이 새로 찍어낸 표준 5페이지(개요 + practice/anatomy/
    variation/wrapup) 항목 외에, `-data_scaling`·`-lambda_sweep` 같은 **부록** 항목이
    섞여 있을 수 있다. 부록은 별도 노트북에서 만든 페이지라 이번 `entries` 에는 전혀
    안 잡히는데, 번호 접두사만으로 블록을 통째로 교체하면 페이지 파일은 멀쩡히 있는데도
    TOC 링크만 조용히 사라진다(고아 페이지). 표준 경로가 아닌 기존 항목은 그대로
    보존해 이 문제를 막는다.
    """
    nn = f"{num:02d}"
    standard_paths = {f"pages/{stem}.md"} | {
        f"pages/{stem}-{suf}.md" for suf in ("practice", "anatomy", "variation", "wrapup")}
    new_lines = []
    for title, path in entries:
        indent = "" if re.match(r"^\d+\.\s", title) else "  "
        new_lines.append(f"{indent}* [{title}]({path})")

    if not toc_path.exists():
        toc_path.write_text(f"# {book_title}\n\n" + "\n".join(new_lines) + "\n", encoding="utf-8")
        return

    lines = toc_path.read_text(encoding="utf-8").splitlines()
    chapter_re = re.compile(rf"^\s*\*\s*\[{nn}[.\-]")
    path_re = re.compile(r"\]\((pages/[^)]+)\)")
    start = end = None
    extra: list[str] = []   # 이 장 블록 안에 있던 비표준(부록 등) 항목 — 그대로 보존
    for i, ln in enumerate(lines):
        if chapter_re.match(ln):
            if start is None:
                start = i
            end = i
            m = path_re.search(ln)
            if m and m.group(1) not in standard_paths:
                extra.append(ln)
    if start is None:
        # 번호 오름차순 유지: 다음으로 큰 장 앞에 삽입, 없으면 끝에 추가
        insert_at = len(lines)
        any_chapter = re.compile(r"^\s*\*\s*\[(\d{2})[.\-]")
        for i, ln in enumerate(lines):
            m = any_chapter.match(ln)
            if m and int(m.group(1)) > num:
                insert_at = i
                break
        out = lines[:insert_at] + new_lines + lines[insert_at:]
    else:
        out = lines[:start] + new_lines + extra + lines[end + 1:]
    toc_path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 챕터 발견 / 선택 / 메타
# --------------------------------------------------------------------------- #
def discover_chapters() -> dict[int, tuple[str, str, Path]]:
    """{num: (folder, slug, nb_path)} — 레포 루트의 NN_slug/NN_slug.ipynb 자동 발견."""
    found: dict[int, tuple[str, str, Path]] = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = CHAPTER_FOLDER_RE.match(d.name)
        if not m:
            continue
        nb = d / f"{d.name}.ipynb"
        if nb.exists():
            found[int(m.group(1))] = (d.name, m.group(2), nb)
    return found


def load_registry_titles() -> dict[int, str]:
    """book/tools/notebook_to_tex.py 의 CHAPTERS 에서 {num: plain_title}."""
    try:
        sys.path.insert(0, str(ROOT / "book" / "tools"))
        import notebook_to_tex as t  # noqa: E402
        return {c.number: latex_title_to_plain(c.title) for c in t.CHAPTERS}
    except Exception:
        return {}


def resolve_title(num: int, slug: str, nb: dict, registry: dict[int, str]) -> str:
    if num in registry and registry[num].strip():
        return registry[num]
    h1 = chapter_h1_title(nb)
    if h1:
        return h1
    return slug.replace("_", " ")


def parse_chapter_args(tokens: list[str], available: dict[int, tuple]) -> list[int]:
    """'7' / '07' / '07_bert_pipeline' → 정렬된 챕터 번호 리스트."""
    nums: list[int] = []
    for tok in tokens:
        m = CHAPTER_FOLDER_RE.match(tok)
        if m:
            n = int(m.group(1))
        elif tok.isdigit():
            n = int(tok)
        else:
            raise SystemExit(f"챕터 인자를 해석할 수 없습니다: {tok!r} (예: 7, 07, 07_bert_pipeline)")
        if n not in available:
            raise SystemExit(f"챕터 {n:02d} 를 찾을 수 없습니다 (NN_slug/NN_slug.ipynb 없음)")
        if n not in nums:
            nums.append(n)
    return sorted(nums)


def pick_source_notebook(folder: str, slug: str, nb_path: Path,
                         executed_dir: Path, args) -> tuple[dict, str]:
    """출력 원천 우선순위에 따라 (노트북 dict, 원천설명) 반환.

    --execute 로 새로 실행했고 --save-executed 면 executed/<폴더>.ipynb 로 저장한다.
    """
    if args.executed_notebook:
        p = Path(args.executed_notebook)
        p = p if p.is_absolute() else ROOT / p
        return json.loads(p.read_text(encoding="utf-8")), f"executed-notebook({p.name})"
    archived = executed_dir / f"{folder}.ipynb"
    if archived.exists():
        return json.loads(archived.read_text(encoding="utf-8")), f"executed/{archived.name}"
    if args.execute:
        nb = execute_notebook(nb_path, timeout=args.timeout)
        if args.save_executed:
            import nbformat
            executed_dir.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, str(executed_dir / f"{folder}.ipynb"))
        return nb, "live --execute" + (" (executed/ 저장됨)" if args.save_executed else "")
    return json.loads(nb_path.read_text(encoding="utf-8")), "clean(출력없음 가능)"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chapters", nargs="*",
                    help="변환할 챕터(폴더명/번호). 비우고 --all 로 전체 지정.")
    ap.add_argument("--all", action="store_true", help="발견된 모든 챕터를 변환")
    ap.add_argument("--pages-dir", default="pages")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--toc", default="TOC.md")
    ap.add_argument("--book-title", default=DEFAULT_BOOK_TITLE)
    ap.add_argument("--output-style", choices=OUTPUT_STYLES, default=DEFAULT_OUTPUT_STYLE,
                    help="실행 결과 박스 표현: code(기본, 웹·PDF·EPUB 모두 안전) | "
                         "fenced-div(전자책 색 박스, 웹에선 ::: 노출) | html-box(웹 전용 색 박스, 전자책 깨짐)")
    ap.add_argument("--execute", action="store_true",
                    help="nbclient로 실행해 실제 출력을 채움 (CPU 챕터용; GPU 챕터엔 비권장)")
    ap.add_argument("--executed-notebook", default=None,
                    help="(단일 챕터) 출력이 담긴 실행본 .ipynb 경로")
    ap.add_argument("--executed-dir", default="executed",
                    help="실행본 보관 폴더 (executed/<폴더>.ipynb 를 출력 원천으로 자동 사용)")
    ap.add_argument("--save-executed", action="store_true",
                    help="--execute 결과를 executed/<폴더>.ipynb 로 저장")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    available = discover_chapters()
    if not available:
        raise SystemExit("변환할 챕터를 찾지 못했습니다 (NN_slug/NN_slug.ipynb 없음)")

    if args.chapters:
        selected = parse_chapter_args(args.chapters, available)
    elif args.all:
        selected = sorted(available)
    else:
        raise SystemExit(
            "변환할 챕터를 지정하거나 --all 을 주세요.\n"
            f"  발견된 챕터: {', '.join(f'{n:02d}' for n in sorted(available))}"
        )

    if args.executed_notebook and len(selected) != 1:
        raise SystemExit("--executed-notebook 은 챕터 1개만 지정했을 때 씁니다.")

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else ROOT / pp

    pages_dir = _abs(args.pages_dir)
    assets_dir = _abs(args.assets) if args.assets else None
    toc_path = _abs(args.toc)
    executed_dir = _abs(args.executed_dir)
    registry = load_registry_titles()

    print(f"변환 대상 {len(selected)}개 챕터: {', '.join(f'{n:02d}' for n in selected)}\n")
    ok, failed = [], []
    for num in selected:
        folder, slug, nb_path = available[num]
        try:
            nb, source = pick_source_notebook(folder, slug, nb_path, executed_dir, args)
            title = resolve_title(num, slug, nb, registry)
            entries, stats = convert(nb, num, slug, title, pages_dir, assets_dir,
                                     args.output_style)
            upsert_toc(toc_path, args.book_title, num, f"{num:02d}-{slug}", entries)
            print(f"[{num:02d}] {title}")
            print(f"     원천={source}  코드셀 {stats['code_cells']}개 "
                  f"(실제출력 {stats['code_with_output']} / 합성 {stats['synthetic']}) "
                  f"이미지 {stats['images']}")
            fixes = []
            if stats["hr_removed"]:
                fixes.append(f"수평선 {stats['hr_removed']} 제거")
            if stats["h1_demoted"]:
                fixes.append(f"H1→H2 {stats['h1_demoted']}")
            if stats["footnotes"]:
                fixes.append(f"각주 {stats['footnotes']} 유니크화")
            if stats.get("heading_blanks"):
                fixes.append(f"헤딩 빈 줄 {stats['heading_blanks']}")
            if stats.get("win_paths"):
                fixes.append(f"윈도우 경로 {stats['win_paths']} 코드화")
            if fixes:
                print("     방어(전자책 규칙):", " / ".join(fixes))
            if stats["html_warn"]:
                print(f"     ⚠ 마크다운 셀 raw HTML {len(stats['html_warn'])}건(전자책에서 깨질 수 있음): "
                      f"{stats['html_warn'][:4]}")
            if stats["extimg_warn"]:
                print(f"     ⚠ 외부 이미지 {len(stats['extimg_warn'])}건(PDF 누락 위험, 위키독스 업로드 필요): "
                      f"{stats['extimg_warn'][:3]}")
            ok.append(num)
        except Exception as e:  # 챕터별 실패 격리
            failed.append((num, e))
            print(f"[{num:02d}] 실패: {e}")
            traceback.print_exc(limit=2)

    print(f"\n완료: 성공 {len(ok)} / 실패 {len(failed)}")
    if failed:
        print("실패 챕터: " + ", ".join(f"{n:02d}" for n, _ in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
