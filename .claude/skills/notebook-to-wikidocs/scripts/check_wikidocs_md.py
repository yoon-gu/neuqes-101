#!/usr/bin/env python3
"""변환된 WikiDocs md 가 전자책 작성 규칙([wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723))을 지키는지 검사한다.

`build_wikidocs.py` 로 변환한 뒤 이 스크립트로 한 번 더 점검한다. 변환기가 자동 방어하는
규칙이라도, 사람이 손댄 페이지나 회귀를 잡기 위한 독립 린터다. 코드펜스(``` ... ```) 안은
검사에서 제외한다(출력·코드의 내용은 자유).

검사 규칙 (괄호 = [wikidocs 전자책 작성시 주의할 점](https://wikidocs.net/198723) 항목):
  E1  본문 H1(#) 금지              [헤딩 레벨 제한]   — 페이지 제목은 TOC 가 담당
  E2  헤딩(##~) 위아래 빈 줄        [헤딩 포맷팅]      — 없으면 PDF 변환 오류
  E3  이미지 위아래 빈 줄          [이미지 포맷팅]
  E4  외부 이미지(http) 금지        [이미지 소스]      — PDF 에서 누락 → 업로드 필요
  E5  GIF 금지                      [이미지 형식]
  E6  raw HTML 금지                 [HTML 코드]        — 전자책 변환 시 깨짐
  E7  수평선(---/***/___) 금지       [줄 구분선]        — PDF 에서 표로 오인
  E8  코드펜스 짝 맞음              [구조]
  E9  각주 이름 전 페이지 유니크     [각주]            — 전자책은 전 페이지를 한 문서로 통합
  W1  윈도우 경로(C:\\) 코드밖        [경로명]          — 경고(코드블록/슬래시 권장)

사용:
    python3 check_wikidocs_md.py                  # 기본: 레포 루트 pages/*.md
    python3 check_wikidocs_md.py pages/01-*.md     # 일부만
종료코드: 위반(E*) 이 하나라도 있으면 1, 없으면 0(경고 W* 는 0).
"""
from __future__ import annotations

import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

H1_RE = re.compile(r"^#\s+\S")
HEADING_RE = re.compile(r"^#{2,6}\s+\S")
IMG_ONLY_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
EXT_IMG_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)")
GIF_RE = re.compile(r"\.gif\b", re.IGNORECASE)
RAW_HTML_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
HR_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
WIN_PATH_RE = re.compile(r"[A-Za-z]:\\")
MATH_RE = re.compile(r"\$[^$]+\$")  # LaTeX 인라인 수식 — W1 오탐(수식 속 i:\ 등) 방지용 제외
INLINE_CODE_RE = re.compile(r"`[^`]*`")
FOOTNOTE_RE = re.compile(r"\[\^([^\]]+)\]")


def classify_lines(text: str):
    """각 줄에 대해 (lineno, raw, in_code) 를 만든다. in_code 는 코드펜스 내부 여부."""
    rows = []
    fence = False
    for i, ln in enumerate(text.split("\n"), 1):
        if ln.lstrip().startswith("```"):
            fence = not fence
            rows.append((i, ln, True))   # 펜스 줄 자체는 코드로 간주
            continue
        rows.append((i, ln, fence))
    return rows, fence


def check_file(path: Path, footnotes: dict):
    """파일 1개 검사 → (errors, warnings). footnotes 에 {name: [locations]} 누적."""
    errors: list[str] = []
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8")
    rows, fence_open = classify_lines(text)
    rel = path.name

    if fence_open:
        errors.append(f"{rel}: 코드펜스(```)가 닫히지 않음 [E8]")

    n = len(rows)
    for idx, (lineno, ln, in_code) in enumerate(rows):
        if in_code:
            continue
        prev_blank = (idx == 0) or (rows[idx - 1][1].strip() == "")
        next_blank = (idx == n - 1) or (rows[idx + 1][1].strip() == "")
        scan = INLINE_CODE_RE.sub("", ln)  # 인라인 코드 제거 후 검사(오탐 감소)

        if H1_RE.match(ln):
            errors.append(f"{rel}:{lineno}: 본문 H1(#) 사용 [E1] → ## 이하로")
        elif HEADING_RE.match(ln):
            if not prev_blank:
                errors.append(f"{rel}:{lineno}: 헤딩 위 빈 줄 없음 [E2]")
            if not next_blank:
                errors.append(f"{rel}:{lineno}: 헤딩 아래 빈 줄 없음 [E2]")

        if IMG_ONLY_RE.match(ln):
            if not prev_blank:
                errors.append(f"{rel}:{lineno}: 이미지 위 빈 줄 없음 [E3]")
            if not next_blank:
                errors.append(f"{rel}:{lineno}: 이미지 아래 빈 줄 없음 [E3]")
        for url in EXT_IMG_RE.findall(ln):
            errors.append(f"{rel}:{lineno}: 외부 이미지 [E4] → 업로드 필요: {url}")
        if GIF_RE.search(scan):
            errors.append(f"{rel}:{lineno}: GIF 참조 [E5] → JPG/PNG 로")

        for tag in RAW_HTML_RE.findall(scan):
            errors.append(f"{rel}:{lineno}: raw HTML [E6] → 마크다운으로: {tag}")
        if HR_RE.match(ln):
            errors.append(f"{rel}:{lineno}: 수평선 [E7] → 제거 권장")
        if WIN_PATH_RE.search(MATH_RE.sub("", scan)):  # 수식 제외 후 검사 (i:\ 류 오탐 방지)
            warnings.append(f"{rel}:{lineno}: 윈도우 경로 [W1] → 코드블록/슬래시 권장")

        for name in FOOTNOTE_RE.findall(ln):
            footnotes[name].append(f"{rel}:{lineno}")

    return errors, warnings


def main() -> None:
    args = sys.argv[1:]
    if args:
        targets = []
        for a in args:
            p = Path(a)
            targets += [Path(x) for x in glob.glob(str(p if p.is_absolute() else ROOT / a))]
    else:
        targets = sorted((ROOT / "pages").glob("*.md"))
    targets = sorted(set(targets))
    if not targets:
        raise SystemExit("검사할 md 파일이 없습니다 (기본: pages/*.md)")

    all_errors: list[str] = []
    all_warnings: list[str] = []
    footnotes: dict[str, list[str]] = defaultdict(list)

    for path in targets:
        errs, warns = check_file(path, footnotes)
        all_errors += errs
        all_warnings += warns

    # E9: 각주 이름이 여러 파일(=전자책 통합 시 한 문서)에서 중복되면 충돌
    for name, locs in sorted(footnotes.items()):
        files = {loc.split(":")[0] for loc in locs}
        if len(files) > 1:
            all_errors.append(
                f"각주 이름 '{name}' 가 여러 페이지에 중복 [E9] → 페이지별로 유니크하게: {sorted(files)}"
            )

    print(f"검사 대상 {len(targets)}개 파일\n")
    if all_errors:
        print(f"❌ 위반 {len(all_errors)}건:")
        for e in all_errors:
            print("  -", e)
    else:
        print("✅ 위반 없음 (전자책 규칙 준수)")
    if all_warnings:
        print(f"\n⚠️  경고 {len(all_warnings)}건:")
        for w in all_warnings:
            print("  -", w)

    sys.exit(1 if all_errors else 0)


if __name__ == "__main__":
    main()
