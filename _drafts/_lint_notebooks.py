"""Lint all curriculum notebooks for static Python issues.

Two passes per notebook:

1. **AST parse per cell** — catches syntax errors (literal "\\n" mid-statement,
   unclosed strings, indentation, etc).
2. **pyflakes on concatenated cells** — catches undefined names, unused imports,
   typos in attribute/function references that survive parse.

Shell/cell magics (`!cmd`, `%magic`) are replaced with `pass` lines so the
analyzer ignores them.

Usage:
    python3 _drafts/_lint_notebooks.py             # lint all notebooks
    python3 _drafts/_lint_notebooks.py 09_*        # lint matching folder(s)

Exit code 1 when any issue is found. Use as a manual gate before pushing.
"""

from __future__ import annotations

import ast
import io
import json
import sys
from pathlib import Path

try:
    from pyflakes import api as pyflakes_api
    HAS_PYFLAKES = True
except ImportError:
    HAS_PYFLAKES = False


REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"_drafts", "book", ".git", "__pycache__"}


def neutralize_magics(src: str) -> str:
    """Replace lines starting with `!` or `%` with `pass`-comments so AST/pyflakes
    don't choke on Jupyter magics."""
    out = []
    for line in src.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}pass  # {stripped}")
        else:
            out.append(line)
    return "\n".join(out)


def collect_code_cells(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        cells.append((i, neutralize_magics(src)))
    return cells


def lint_notebook(nb_path: Path) -> list[str]:
    findings: list[str] = []
    cells = collect_code_cells(nb_path)

    # Pass 1 — AST parse per cell
    for i, src in cells:
        try:
            ast.parse(src)
        except SyntaxError as e:
            findings.append(f"  [AST] cell {i}: {e.msg} (line {e.lineno})")

    # Pass 2 — pyflakes over concatenated source.
    # Goal: catch *execution-breaking* issues (undefined names, attribute typos
    # that pyflakes can detect statically). Style warnings are filtered out.
    if HAS_PYFLAKES and cells:
        joined = "\n\n".join(
            f"# === cell {i} ===\n{src}" for i, src in cells
        )
        out, err = io.StringIO(), io.StringIO()
        pyflakes_api.check(joined, str(nb_path.name), reporter=_PyflakesReporter(out, err))
        for line in out.getvalue().strip().split("\n"):
            if not line:
                continue
            # Filter style-only warnings (code still runs fine):
            if any(skip in line for skip in (
                "f-string is missing placeholders",
                "imported but unused",
                "redefinition of unused",
                "assigned to but never used",
                "list comprehension redefines",
                "may be undefined, or defined from star imports",
            )):
                continue
            findings.append(f"  [pyflakes] {line.replace(str(nb_path.name) + ':', 'L')}")

        # Stderr is for syntax errors pyflakes itself caught (different from AST)
        for line in err.getvalue().strip().split("\n"):
            if line:
                findings.append(f"  [pyflakes-syntax] {line.replace(str(nb_path.name) + ':', 'L')}")

    return findings


class _PyflakesReporter:
    """pyflakes Reporter API: unexpected_error / syntax_error / flake."""

    def __init__(self, out, err):
        self.out, self.err = out, err

    def unexpectedError(self, filename, msg):
        self.err.write(f"{filename}: unexpected error: {msg}\n")

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.err.write(f"{filename}:{lineno}:{offset or 0}: {msg}\n")

    def flake(self, message):
        self.out.write(str(message) + "\n")


def main(argv: list[str]) -> int:
    patterns = argv[1:] if len(argv) > 1 else None
    notebooks = sorted(REPO.glob("*/[!_]*.ipynb"))
    notebooks = [
        nb for nb in notebooks
        if nb.parent.name not in SKIP_DIRS
        and not nb.parent.name.startswith(".")
    ]
    if patterns:
        notebooks = [
            nb for nb in notebooks
            if any(p in str(nb.relative_to(REPO)) for p in patterns)
        ]

    if not HAS_PYFLAKES:
        print("⚠️  pyflakes not installed — only AST parse pass will run")
        print("    install: python3 -m pip install --user --break-system-packages pyflakes")
        print()

    total_issues = 0
    for nb in notebooks:
        findings = lint_notebook(nb)
        rel = nb.relative_to(REPO)
        if findings:
            total_issues += len(findings)
            print(f"{rel}  ({len(findings)} issue{'s' if len(findings) > 1 else ''})")
            for f in findings:
                print(f)
        else:
            print(f"{rel}  ✓")

    print()
    if total_issues:
        print(f"❌ {total_issues} issue(s) across {len(notebooks)} notebook(s)")
        return 1
    print(f"✓ all {len(notebooks)} notebooks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
