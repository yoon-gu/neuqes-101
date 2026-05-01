#!/usr/bin/env python3
"""Convert the current notebook chapters into book-ready LaTeX chapters.

The notebooks stay the source of truth. This script extracts markdown and code
cells, removes notebook-only emoji from headings, and writes LaTeX chapter files
under book/chapters/.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "book"
CHAPTER_DIR = BOOK / "chapters"
GITHUB_RAW = "https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master"


@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    title: str
    short_title: str
    focus: str
    indexes: tuple[str, ...]

    @property
    def notebook(self) -> Path:
        return ROOT / f"{self.number:02d}_{self.slug}" / f"{self.number:02d}_{self.slug}.ipynb"

    @property
    def tex_name(self) -> str:
        return f"ch{self.number:02d}_{self.slug}.tex"

    @property
    def colab_url(self) -> str:
        rel = f"{self.number:02d}_{self.slug}/{self.number:02d}_{self.slug}.ipynb"
        return f"{GITHUB_RAW}/{rel}"


CHAPTERS = [
    Chapter(
        1,
        "tfidf",
        "TF-IDF로 만나는 첫 벡터",
        "TF-IDF",
        "텍스트를 숫자 벡터로 바꾸는 첫 관문",
        ("TF-IDF", "CountVectorizer", "TfidfVectorizer", "sparse matrix", "vocabulary"),
    ),
    Chapter(
        2,
        "sklearn_regression",
        "회귀와 MSELoss",
        "Regression",
        "별점 예측을 통해 첫 Loss인 MSE를 관찰",
        ("Regression", "MSELoss", "LinearRegression", "mean_squared_error", "Output Head"),
    ),
    Chapter(
        3,
        "sklearn_binary",
        "이진 분류와 BCEWithLogitsLoss",
        "Binary",
        "logit, sigmoid, BCE가 만나는 방식",
        ("Binary classification", "BCEWithLogitsLoss", "sigmoid", "LogisticRegression", "predict_proba"),
    ),
    Chapter(
        4,
        "softmax_binary",
        "sigmoid와 softmax의 동등성",
        "Softmax Binary",
        "2차원 softmax 이진 분류와 1차원 sigmoid의 관계",
        ("softmax", "CrossEntropyLoss", "sigmoid", "multinomial", "reparameterization"),
    ),
    Chapter(
        5,
        "sklearn_multiclass",
        "다중 클래스와 CrossEntropyLoss",
        "Multi-class",
        "K=5 출력 헤드와 softmax 일반화",
        ("Multi-class classification", "CrossEntropyLoss", "confusion_matrix", "classification_report"),
    ),
    Chapter(
        6,
        "sklearn_multilabel",
        "다중 라벨과 per-label BCE",
        "Multi-label",
        "softmax의 합=1 제약을 풀고 라벨별 sigmoid로 확장",
        ("Multi-label classification", "OneVsRestClassifier", "BCEWithLogitsLoss", "hamming_loss", "micro F1", "macro F1"),
    ),
]


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


def strip_heading_emoji(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            line = EMOJI_PATTERN.sub("", line)
            line = re.sub(r"(#+)\s+", r"\1 ", line)
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def sanitize_symbols(text: str) -> str:
    return (
        text.replace("❌", "X")
        .replace("✅", "OK")
        .replace("⚠️", "주의")
        .replace("⚠", "주의")
        .replace("\ufe0f", "")
    )


def escape_table_math_pipes(markdown: str) -> str:
    """Prevent pipe-table parsing from splitting absolute-value math cells."""
    converted_lines = []
    for line in markdown.splitlines():
        if "|" not in line or "$" not in line:
            converted_lines.append(line)
            continue

        def repl(match: re.Match[str]) -> str:
            body = match.group(1).replace("|", r"\vert{}")
            return f"${body}$"

        converted_lines.append(re.sub(r"\$([^$\n]+)\$", repl, line))
    return "\n".join(converted_lines) + "\n"


def promote_headings(text: str) -> str:
    """Turn notebook h2 sections into book h1 sections inside a chapter."""
    promoted = []
    for line in text.splitlines():
        match = re.match(r"^(#{2,6})(\s+.*)$", line)
        if match:
            promoted.append(match.group(1)[1:] + match.group(2))
        else:
            promoted.append(line)
    return "\n".join(promoted) + "\n"


def strip_pandoc_targets(latex: str) -> str:
    """Remove Pandoc's repeated hypertarget/label wrappers.

    Notebook chapters reuse headings such as FAQ and 토크나이저 노트. Let LaTeX
    number the sections instead of carrying duplicate PDF anchors into the book.
    """
    cleaned = []
    for line in latex.splitlines():
        if line.startswith("\\hypertarget{"):
            continue
        line = re.sub(r"\\label\{[^{}]*\}\}$", "", line)
        cleaned.append(line)
    return "\n".join(cleaned)


def normalize_code_blocks(latex: str) -> str:
    latex = latex.replace("\\begin{verbatim}", "\\begin{lstlisting}")
    latex = latex.replace("\\end{verbatim}", "\\end{lstlisting}")
    return latex


def faq_subsections_to_questions(latex: str) -> str:
    # Pandoc turns notebook FAQ "### Q..." headings into \subsection. In book
    # form these should read as question blocks, not structural headings.
    latex = re.sub(
        r"\\subsection\{(Q\d+\..*?)\}",
        r"\\faqquestion{\1}",
        latex,
        flags=re.DOTALL,
    )
    latex = re.sub(
        r"\\subsection\{\\texorpdfstring\{(Q\d+\..*?)\}\{.*?\}\}",
        r"\\faqquestion{\1}",
        latex,
        flags=re.DOTALL,
    )
    return latex


def table_spec_to_xltabular(match: re.Match[str]) -> str:
    spec = match.group(1)
    return f"\\begin{{adjustbox}}{{max width=\\textwidth}}\n\\begin{{tabular}}{{@{{}}{spec}@{{}}}}"


def normalize_tables(latex: str) -> str:
    latex = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}", table_spec_to_xltabular, latex)
    latex = latex.replace("\\endhead", "")
    latex = latex.replace("\\end{longtable}", "\\end{tabular}\n\\end{adjustbox}")
    latex = latex.replace("\\toprule()", "\\toprule")
    latex = latex.replace("\\midrule()", "\\midrule")
    latex = latex.replace("\\bottomrule()", "\\bottomrule")
    latex = re.sub(
        r"(\\textbar\{\}.*?\\textbar\{\})(?:\n(\\textbar[-\\/\{\}A-Za-z0-9\s]+\\textbar\{\}))",
        lambda match: match.group(0),
        latex,
        flags=re.DOTALL,
    )
    return latex


def unescape_texttt_content(text: str) -> str:
    return (
        text.replace(r"\_", "_")
        .replace(r"\#", "#")
        .replace(r"\%", "%")
        .replace(r"\$", "$")
        .replace(r"\&", "&")
        .replace(r"\{", "{")
        .replace(r"\}", "}")
        .replace(r"\textasciitilde{}", "~")
        .replace(r"\textasciicircum{}", "^")
        .replace(r"\textbackslash{}", "\\")
        .replace(r"\ ", " ")
    )


def normalize_inline_code(latex: str) -> str:
    def repl(match: re.Match[str]) -> str:
        content = match.group(1)
        if "}" in content:
            return match.group(0)
        return "\\inlinecode{" + content + "}"

    return re.sub(r"\\texttt\{([^{}]*)\}", repl, latex)


def polish_book_prose(latex: str) -> str:
    """Normalize notebook-style shorthand and informal wording for book prose."""

    # Chapter references.
    latex = re.sub(r"\bChapter\s+([0-9]+)", r"\1장", latex)
    latex = re.sub(r"\bCh\s*([0-9]+)\s*-\s*([0-9]+)", r"\1-\2장", latex)
    latex = re.sub(r"\bCh\s*([0-9]+)\s*·\s*([0-9]+)", r"\1·\2장", latex)
    latex = re.sub(r"\bCh\s*([0-9]+)", r"\1장", latex)

    replacements = {
        "이번 챕터": "이 장",
        "다음 챕터": "다음 장",
        "이전 챕터": "이전 장",
        "챕터가": "장이",
        "챕터는": "장은",
        "챕터의": "장의",
        "챕터에": "장에",
        "챕터에서": "장에서",
        "챕터마다": "장마다",
        "챕터": "장",
        "삽질 코너": "오류 실험",
        "떡밥": "후속 논점",
        "그냥 \"틀림\"": "동일한 오분류",
        "그냥 숫자": "비활성 스칼라 출력",
        "그냥": "단순히",
        "죽습니다": "중단될 수 있습니다",
        "죽고": "실패하고",
        "뱉는": "출력하는",
        "뱉으면": "출력하면",
        "뱉습니다": "출력합니다",
        "뱉은": "출력한",
        "뱉을": "출력할",
        "듣지 않습니다": "포함하지 않습니다",
        "깔끔하고": "명확하고",
        "헷갈리는": "혼동하는",
        "헷갈림": "혼동",
        "깔끔하게": "일관되게",
        "깔끔합니다": "명확합니다",
        "비추": "권장하지 않음",
        "망가지면": "불안정해지면",
        "낯설지 않습니다": "익숙하게 이해할 수 있습니다",
        "딱 무엇이": "정확히 무엇이",
        "딱 하나": "정확히 하나",
        "손에 잡힙니다": "구체적으로 이해할 수 있습니다",
        "손에 익힙니다": "실습합니다",
        "펴 봅니다": "확인합니다",
        "펼쳐 봅니다": "확인합니다",
        "보여줬습니다": "표시했습니다",
        "가벼운": "간단한",
        "화려한 형태": "복합적인 형태",
        "거의 그대로": "대부분 동일하게",
        "살아 있습니다": "유지됩니다",
        "sklearn 시대": "scikit-learn 단계",
        "BERT 시대": "BERT 단계",
        "원본 형태": "기본 형태",
        "출력 직전": "출력층 직전",
        "fit 한 줄": "fit 호출 한 줄",
        "fit이 첫 줄에서": "fit 호출이 즉시",
        "Binary Cross Entropy": "Binary Cross-Entropy",
        "비활성 스칼라 출력다": "비활성 스칼라 출력입니다",
        "다음 장를": "다음 장을",
        "1·2장와": "1·2장과",
        "2장와": "2장과",
    }
    for before, after in replacements.items():
        latex = latex.replace(before, after)

    # More formal section/table labels after chapter-reference normalization.
    latex = re.sub(r"변경점 \(Diff from ([0-9]+장)\)", r"변경점: \1 대비", latex)
    latex = latex.replace("전체 18장 표", "전체 18개 장의 표")
    latex = latex.replace("전체 19장 표", "전체 19개 장의 표")
    return latex


def polish_code_comments(source: str) -> str:
    source = re.sub(r"\bChapter\s+([0-9]+)", r"\1장", source)
    source = re.sub(r"\bCh\s*([0-9]+)\s*-\s*([0-9]+)", r"\1-\2장", source)
    source = re.sub(r"\bCh\s*([0-9]+)", r"\1장", source)
    source = source.replace("그냥", "직접")
    source = source.replace("뱉는", "출력하는")
    source = source.replace("뱉을", "출력할")
    return source


def wrap_faq_blocks(latex: str) -> str:
    lines = latex.splitlines()
    wrapped: list[str] = []
    faq_open = False

    def close_faq() -> None:
        nonlocal faq_open
        if faq_open:
            wrapped.append("\\end{faqBox}")
            wrapped.append("")
            faq_open = False

    for line in lines:
        if line.startswith("\\faqquestion{"):
            close_faq()
            title = line[len("\\faqquestion{") : -1]
            wrapped.append(f"\\begin{{faqBox}}{{{title}}}")
            faq_open = True
            continue

        if faq_open and (line.startswith("\\section{") or line.startswith("\\chapter{")):
            close_faq()

        wrapped.append(line)

    close_faq()
    return "\n".join(wrapped)


def wrap_preview_blocks(latex: str) -> str:
    lines = latex.splitlines()
    wrapped: list[str] = []
    preview_open = False

    def close_preview() -> None:
        nonlocal preview_open
        if preview_open:
            wrapped.append("\\end{previewBox}")
            wrapped.append("")
            preview_open = False

    for line in lines:
        if line.startswith("\\section{") and ("다음 장 예고" in line or "다음 챕터 예고" in line):
            close_preview()
            wrapped.append("\\begin{previewBox}{미리보기: 다음 장}")
            preview_open = True
            continue

        if preview_open and (line.startswith("\\section{") or line.startswith("\\chapter{")):
            close_preview()

        if not preview_open and line.startswith("\\textbf{다음 장"):
            wrapped.append("\\begin{previewBox}{미리보기}")
            wrapped.append(line)
            wrapped.append("\\end{previewBox}")
            continue

        wrapped.append(line)

    close_preview()
    latex = "\n".join(wrapped)
    latex = latex.replace("\\begin{quote}\n\\begin{previewBox}", "\\begin{previewBox}")
    latex = latex.replace("\\end{previewBox}\n\\end{quote}", "\\end{previewBox}")
    return latex


def display_math_to_numbered_equations(latex: str, chapter_number: int) -> str:
    counter = 0

    def equation_note(body: str, label: str) -> str:
        compact = re.sub(r"\s+", " ", body)
        ref = f"식~\\eqref{{{label}}}"
        if "Hamming loss" in compact:
            return f"{ref}은 multi-label 평가에서 사용하는 Hamming loss의 정의입니다."
        if "BCE" in compact or "y_{ik}\\log" in compact or "y_i \\log \\hat p_i" in compact:
            return f"{ref}은 정답 라벨과 예측 확률 사이의 Binary Cross-Entropy를 정의합니다."
        if "\\text{CE}" in compact or "\\sum_{k=0}^{1}" in compact:
            return f"{ref}은 Cross-Entropy가 K=2에서 BCE와 같은 형태로 정리됨을 보여줍니다."
        if "softmax" in compact and "\\sigma" in compact:
            return f"{ref}은 2차원 softmax가 logit 차이에 대한 sigmoid로 표현됨을 보여줍니다."
        if "\\text{softmax}" in compact:
            return f"{ref}은 logit 벡터를 확률 분포로 바꾸는 softmax의 정의입니다."
        if "\\log K" in compact or "\\log(1/K)" in compact:
            return f"{ref}은 균등 추측 baseline이 \\(\\log K\\)가 되는 이유를 설명합니다."
        if "(y_i - \\hat y_i)^2" in compact:
            return f"{ref}은 회귀에서 사용하는 Mean Squared Error의 정의입니다."
        if "w^\\top x + b" in compact:
            return f"{ref}은 선형 모델의 출력이 특성 벡터와 가중치의 선형 결합임을 나타냅니다."
        if "(X^\\top X)^{-1}" in compact:
            return f"{ref}은 선형회귀의 정규방정식 해를 나타냅니다."
        if "\\text{tfidf}" in compact:
            return f"{ref}은 단어 빈도와 희귀도 가중치를 결합한 TF-IDF의 정의입니다."
        if "\\text{idf}" in compact:
            return f"{ref}은 문서 빈도로부터 IDF 값을 계산하는 방식입니다."
        return f"{ref}은 이 절에서 사용하는 핵심 관계를 정리한 것입니다."

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        label = f"eq:ch{chapter_number:02d}-{counter:02d}"
        body = match.group(1).strip()
        return (
            "\\begin{equation}\n"
            f"\\label{{{label}}}\n"
            f"{body}\n"
            "\\end{equation}\n\n"
            f"{equation_note(body, label)}"
        )

    return re.sub(r"\\\[(.*?)\\\]", repl, latex, flags=re.DOTALL)


def markdown_to_latex(markdown: str, chapter_number: int) -> str:
    markdown = sanitize_symbols(promote_headings(strip_heading_emoji(markdown)))
    markdown = escape_table_math_pipes(markdown)
    proc = subprocess.run(
        [
            "pandoc",
            "-f",
            "gfm+tex_math_dollars+pipe_tables",
            "-t",
            "latex",
            "--wrap=preserve",
            "--no-highlight",
        ],
        input=markdown,
        text=True,
        check=True,
        capture_output=True,
    )
    latex = proc.stdout
    latex = strip_pandoc_targets(latex)
    latex = normalize_code_blocks(latex)
    latex = faq_subsections_to_questions(latex)
    latex = normalize_tables(latex)
    latex = normalize_inline_code(latex)
    latex = wrap_faq_blocks(latex)
    latex = polish_book_prose(latex)
    latex = wrap_preview_blocks(latex)
    latex = latex.replace("\\begin{Shaded}", "\\begin{noteBox}[코드]")
    latex = latex.replace("\\end{Shaded}", "\\end{noteBox}")
    return latex.strip()


def code_walkthrough(source: str) -> str:
    statements: list[tuple[int, int, list[str]]] = []
    start = 0
    current: list[str] = []
    paren_balance = 0

    def flush(end_line: int) -> None:
        nonlocal current, start, paren_balance
        content = [line for line in current if line.strip() and not line.strip().startswith("#")]
        if content:
            statements.append((start, end_line, content))
        current = []
        start = 0
        paren_balance = 0

    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            if current and paren_balance <= 0:
                flush(lineno - 1)
            continue
        if not current:
            start = lineno
        current.append(line)
        paren_balance += line.count("(") + line.count("[") + line.count("{")
        paren_balance -= line.count(")") + line.count("]") + line.count("}")
        if paren_balance <= 0 and not stripped.endswith((",", "\\", ".")):
            flush(lineno)
    if current:
        flush(len(source.splitlines()))

    def latex_escape_text(text: str) -> str:
        return (
            text.replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("$", r"\$")
            .replace("#", r"\#")
            .replace("_", r"\_")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}")
        )

    def summarize_code(content: list[str]) -> str:
        code_lines = [line.strip() for line in content if line.strip() and not line.strip().startswith("#")]
        joined = " ".join(code_lines)
        if len(joined) > 58:
            joined = joined[:55].rstrip() + "..."
        return f"\\inlinecode{{{latex_escape_text(joined)}}}"

    def imported_modules() -> list[str]:
        known = {
            "numpy": "numpy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "datasets": "datasets",
            "sklearn": "scikit-learn",
            "transformers": "transformers",
            "torch": "PyTorch",
            "tokenizers": "tokenizers",
        }
        found: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            module = ""
            if stripped.startswith("import "):
                module = stripped.split()[1].split(".")[0]
            elif stripped.startswith("from "):
                module = stripped.split()[1].split(".")[0]
            label = known.get(module)
            if label and label not in found:
                found.append(label)
        return found

    notes: list[str] = []
    import_note_added = False
    major_imports = imported_modules()
    for start_line, end_line, content in statements[:8]:
        text = " ".join(line.strip() for line in content)
        if text.startswith(("print(", "display(")) or " print(" in text:
            continue
        if text.startswith("!pip "):
            message = "Colab 실행에 필요한 패키지를 설치합니다."
        elif text.startswith(("import ", "from ")):
            if import_note_added or not major_imports:
                continue
            import_note_added = True
            snippet = "\\inlinecode{" + ", ".join(major_imports[:5]) + "}"
            notes.append(f"{snippet} 같은 주요 패키지를 불러옵니다.")
            continue
        elif ".fit(" in text:
            message = "학습 데이터로 모델 또는 변환기를 적합합니다."
        elif ".transform(" in text or ".fit_transform(" in text:
            message = "텍스트나 라벨을 모델이 처리할 수 있는 수치 표현으로 변환합니다."
        elif ".predict_proba(" in text:
            message = "클래스별 예측 확률을 계산합니다."
        elif ".predict(" in text:
            message = "학습된 모델로 최종 예측값을 만듭니다."
        elif "train_test_split" in text:
            message = "학습용 데이터와 평가용 데이터를 분리합니다."
        elif "LogisticRegression" in text or "LinearRegression" in text or "OneVsRestClassifier" in text:
            message = "이번 실습에서 관찰할 모델 객체를 정의합니다."
        elif "=" in text:
            message = "이후 단계에서 재사용할 중간 값을 계산해 변수에 저장합니다."
        else:
            message = "앞 단계에서 만든 값을 바탕으로 다음 계산을 수행합니다."
        snippet = summarize_code(content)
        notes.append(f"{snippet}에서는 {message}")

    if not notes:
        return ""

    return (
        "\\vspace{0.45em}\n"
        "\\noindent\\textbf{위 코드 읽기.}\\quad "
        + " ".join(notes)
        + "\n\\par\\vspace{0.9em}"
    )


def code_to_latex(source: str, include_notes: bool = False) -> str:
    source = sanitize_symbols(source)
    source = polish_code_comments(source)
    source = source.rstrip()
    if not source:
        return ""
    listing = "\\begin{lstlisting}\n" + source + "\n\\end{lstlisting}"
    if include_notes:
        notes = code_walkthrough(source)
        if notes:
            return listing + "\n\n" + notes
    return listing


def chapter_tex(chapter: Chapter) -> str:
    nb = json.loads(chapter.notebook.read_text(encoding="utf-8"))
    chunks: list[str] = [
        "% Generated by book/tools/notebook_to_tex.py. Do not edit by hand.",
        f"\\chapter[{chapter.short_title}]{{{chapter.title}}}",
        "\\markboth{" + chapter.short_title + "}{" + chapter.short_title + "}",
        "\\chaptermeta{"
        + f"{chapter.number:02d}_{chapter.slug}/{chapter.number:02d}_{chapter.slug}.ipynb"
        + "}{"
        + chapter.colab_url
        + "}{"
        + chapter.focus
        + "}",
        "",
    ]

    for term in chapter.indexes:
        safe = term.replace("_", "\\_")
        chunks.append(f"\\index{{{safe}}}")
    chunks.append("")
    explain_code = False

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        if cell.get("cell_type") == "markdown":
            first = source.strip().splitlines()[0]
            if first.startswith("# Chapter"):
                continue
            if first.lstrip().startswith("##"):
                explain_code = any(section in first for section in ("토크나이저", "실습", "해부"))
            chunks.append(markdown_to_latex(source, chapter.number))
        elif cell.get("cell_type") == "code":
            chunks.append(code_to_latex(source, include_notes=explain_code))

        chunks.append("")

    chapter_latex = "\n\n".join(chunks).rstrip() + "\n"
    chapter_latex = display_math_to_numbered_equations(chapter_latex, chapter.number)
    return chapter_latex


def main() -> None:
    CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
    for chapter in CHAPTERS:
        if not chapter.notebook.exists():
            raise FileNotFoundError(chapter.notebook)
        out = CHAPTER_DIR / chapter.tex_name
        out.write_text(chapter_tex(chapter), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
