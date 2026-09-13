#!/usr/bin/env python3
"""Check generated chapter LaTeX for broken Korean particles after "장".

"챕터" 를 "장" 으로 바꾸는 과정에서 조사 이형태가 어긋나거나 (장로/장를/장와),
조사가 띄어 쓰인 채 남거나 (20장 에서), 산문용 치환이 코드 리스팅까지 들어가는
문제를 빌드 후에 한 번 훑는다. 문제가 있으면 종료 코드 1.

    python3 book/tools/lint_chapter_josa.py
    python3 book/tools/lint_chapter_josa.py book/chapters/ch20_en_bert_pretrain.tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAPTER_DIR = ROOT / "book" / "chapters"

VERBATIM_BEGIN = ("\\begin{lstlisting}", "\\begin{verbatim}", "\\begin{bookoutputbox}")
VERBATIM_END = ("\\end{lstlisting}", "\\end{verbatim}", "\\end{bookoutputbox}")

CHECKS = (
    ("조사 이형태", re.compile(r"장(를|와|로|가|는|라)(?![가-힣])"), False),
    (
        "조사 띄어쓰기",
        re.compile(
            r"장 (은|는|이|가|을|를|과|와|의|에|에서|으로|부터|까지|도|만|보다|처럼)"
            r"(?![가-힣])"
        ),
        False,
    ),
    ("남은 '챕터'", re.compile(r"(?<!\\#)챕터"), False),
)

# 장 번호는 책 전체에서 "N장" 으로 통일한다. 코드와 그 실행 출력이 어긋나지
# 않으려면 한쪽만 남아서는 안 되므로, 산문·코드·출력 어디든 걸리면 알린다.
LEFTOVER_CH = ("남은 'Ch N' 표기", re.compile(r"\bCh(?:apter)?\s?[0-9]{1,2}\b"))


def scan(path: Path) -> list[str]:
    problems: list[str] = []
    in_verbatim = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(VERBATIM_BEGIN):
            in_verbatim = True
            continue
        if stripped.startswith(VERBATIM_END):
            in_verbatim = False
            continue
        for label, pattern, code_only in CHECKS:
            if code_only != in_verbatim:
                continue
            match = pattern.search(line)
            if match:
                excerpt = line.strip()[:96]
                problems.append(f"{path.name}:{number}  [{label}] {excerpt}")
        label, pattern = LEFTOVER_CH
        # \index{...} 는 큐레이션한 색인 항목이라 표기를 그대로 둔다.
        if not stripped.startswith("\\index{") and pattern.search(line):
            problems.append(f"{path.name}:{number}  [{label}] {line.strip()[:96]}")
    return problems


def main() -> int:
    targets = [Path(arg) for arg in sys.argv[1:]] or sorted(CHAPTER_DIR.glob("*.tex"))
    problems = [problem for target in targets for problem in scan(target)]
    for problem in problems:
        print(problem)
    print(f"\n{len(targets)}개 파일 검사, 문제 {len(problems)}건")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
