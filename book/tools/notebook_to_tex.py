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
import argparse
import io
import textwrap
import tokenize
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "book"
CHAPTER_DIR = BOOK / "chapters"
GITHUB_RAW = "https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master"
RENDER_DATAFRAME_TABLES = False


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
        "텍스트 벡터화 (TF-IDF)",
        "텍스트 벡터화 (TF-IDF)",
        "텍스트를 숫자 벡터로 바꾸는 첫 관문",
        (
            "TF-IDF",
            "CountVectorizer",
            "TfidfVectorizer",
            "sparse matrix",
            "vocabulary",
            "텍스트 벡터화",
            "단어 빈도",
            "희귀도 가중치",
            "어휘",
            "희소 행렬",
        ),
    ),
    Chapter(
        2,
        "sklearn_regression",
        "회귀 분석 (Regression \\& MSE)",
        "회귀 분석 (Regression \\& MSE)",
        "별점 예측을 통해 첫 손실인 평균제곱오차를 관찰",
        (
            "Regression",
            "MSELoss",
            "LinearRegression",
            "mean_squared_error",
            "Output Head",
            "회귀",
            "평균제곱오차",
            "별점 예측",
            "비활성 출력",
            "선형회귀",
            "정규방정식",
            "평균절대오차",
            "결정계수",
        ),
    ),
    Chapter(
        3,
        "sklearn_binary",
        "이진 분류 (Binary Classification \\& BCE)",
        "이진 분류 (Binary Classification \\& BCE)",
        "logit, sigmoid, BCE가 만나는 방식",
        (
            "Binary classification",
            "BCEWithLogitsLoss",
            "sigmoid",
            "LogisticRegression",
            "predict_proba",
            "이진 분류",
            "로짓",
            "시그모이드",
            "예측 확률",
            "임계값",
            "정밀도",
            "재현율",
            "혼동 행렬",
        ),
    ),
    Chapter(
        4,
        "softmax_binary",
        "sigmoid와 softmax의 동등성 (Binary Classification)",
        "sigmoid와 softmax의 동등성 (Binary Classification)",
        "2차원 softmax 이진 분류와 1차원 sigmoid의 관계",
        (
            "softmax",
            "CrossEntropyLoss",
            "sigmoid",
            "multinomial",
            "reparameterization",
            "소프트맥스",
            "교차 엔트로피",
            "원-핫",
            "리파라미터화",
            "이진 분류 동등성",
        ),
    ),
    Chapter(
        5,
        "sklearn_multiclass",
        "다중 클래스 분류 (Multi-class Classification \\& CE)",
        "다중 클래스 분류 (Multi-class Classification \\& CE)",
        "K=5 출력 헤드와 softmax 일반화",
        (
            "Multi-class classification",
            "CrossEntropyLoss",
            "confusion_matrix",
            "classification_report",
            "다중 클래스 분류",
            "균등 추측",
            "클래스 불균형",
            "혼동 행렬",
            "매크로 F1",
        ),
    ),
    Chapter(
        6,
        "sklearn_multilabel",
        "다중 라벨 분류 (Multi-label Classification \\& Per-label BCE)",
        "다중 라벨 분류 (Multi-label Classification \\& Per-label BCE)",
        "softmax의 합=1 제약을 풀고 라벨별 sigmoid로 확장",
        (
            "Multi-label classification",
            "OneVsRestClassifier",
            "BCEWithLogitsLoss",
            "hamming_loss",
            "micro F1",
            "macro F1",
            "다중 라벨 분류",
            "멀티핫",
            "라벨별 BCE",
            "해밍 손실",
            "마이크로 F1",
            "매크로 F1",
            "임계값 조정",
            "측면 라벨",
        ),
    ),
    Chapter(
        7,
        "bert_pipeline",
        "BERT 파이프라인 (Pipeline)",
        "BERT 파이프라인 (Pipeline)",
        "pipeline 한 줄 뒤의 tokenizer, model, post-processing 분해",
        (
            "BERT",
            "DistilBERT",
            "pipeline",
            "AutoTokenizer",
            "AutoModelForSequenceClassification",
            "WordPiece",
            "pretrained model",
            "special token",
            "post-processing",
            "BERT 첫 만남",
            "사전학습 모델",
            "파이프라인",
            "워드피스",
            "특수 토큰",
            "토크나이저",
            "추론",
        ),
    ),
    Chapter(
        8,
        "tokenizer_datasets",
        "토크나이저와 데이터셋 (Tokenizer \\& Datasets)",
        "토크나이저와 데이터셋 (Tokenizer \\& Datasets)",
        "padding, truncation, max_length와 datasets 입력 파이프라인",
        (
            "Tokenizer",
            "datasets",
            "load_dataset",
            "Dataset.map",
            "Dataset.filter",
            "padding",
            "truncation",
            "max_length",
            "attention_mask",
            "DataCollatorWithPadding",
            "DataLoader",
            "Apache Arrow",
            "토크나이저 옵션",
            "패딩",
            "잘림",
            "최대 길이",
            "어텐션 마스크",
            "데이터셋",
            "입력 파이프라인",
        ),
    ),
    Chapter(
        9,
        "bert_regression",
        "BERT 회귀 분석 (Regression \\& Trainer)",
        "BERT 회귀 분석 (Regression \\& Trainer)",
        "DistilBERT 파인튜닝과 Trainer의 첫 사용",
        (
            "BERT regression",
            "DistilBERT",
            "Trainer",
            "TrainingArguments",
            "problem_type",
            "regression",
            "MSELoss",
            "fp16",
            "Adam",
            "compute_metrics",
            "fine-tuning",
            "파인튜닝",
            "트레이너",
            "학습 인자",
            "회귀 헤드",
            "평균제곱오차",
            "혼합 정밀도",
            "GPU 메모리",
        ),
    ),
    Chapter(
        10,
        "bert_binary_sigmoid",
        "BERT 이진 분류: Sigmoid (BCE)",
        "BERT 이진 분류: Sigmoid (BCE)",
        "num_labels=1, sigmoid, BCEWithLogitsLoss 방식의 BERT 이진 분류",
        (
            "BERT binary classification",
            "sigmoid",
            "BCEWithLogitsLoss",
            "num_labels=1",
            "multi_label_classification",
            "binary threshold",
            "ROC AUC",
            "precision_recall_fscore_support",
            "prediction cache",
            "이진 분류",
            "시그모이드",
            "이진 교차 엔트로피",
            "확률 임계값",
            "예측 저장",
            "AUC",
        ),
    ),
    Chapter(
        11,
        "bert_binary_softmax",
        "BERT 이진 분류: Softmax (CE)",
        "BERT 이진 분류: Softmax (CE)",
        "num_labels=2, softmax, CrossEntropyLoss 표준 BERT 분류 방식",
        (
            "BERT binary softmax",
            "softmax",
            "CrossEntropyLoss",
            "num_labels=2",
            "single_label_classification",
            "id2label",
            "label2id",
            "logit difference",
            "prediction agreement",
            "소프트맥스",
            "교차 엔트로피",
            "라벨 매핑",
            "로짓 차이",
            "예측 일치율",
            "이진 분류 동등성",
        ),
    ),
    Chapter(
        12,
        "bert_multiclass",
        "BERT 다중 클래스 분류 (Multi-class \\& CE)",
        "BERT 다중 클래스 분류 (Multi-class \\& CE)",
        "Yelp 5클래스 분류로 확장한 DistilBERT softmax 분류",
        (
            "BERT multi-class classification",
            "multi-class classification",
            "CrossEntropyLoss",
            "num_labels=5",
            "confusion_matrix",
            "classification_report",
            "roc_auc_score",
            "macro F1",
            "calibration",
            "random baseline",
            "다중 클래스 분류",
            "혼동 행렬",
            "분류 리포트",
            "매크로 F1",
            "캘리브레이션",
            "랜덤 기준선",
        ),
    ),
    Chapter(
        13,
        "bert_multilabel",
        "BERT 다중 라벨 분류 (Multi-label \\& Per-label BCE)",
        "BERT 다중 라벨 분류 (Multi-label \\& Per-label BCE)",
        "Yelp 측면 합성 라벨을 BERT의 라벨별 sigmoid와 BCE로 학습",
        (
            "BERT multi-label classification",
            "multi-label classification",
            "BCEWithLogitsLoss",
            "num_labels=5",
            "multi_label_classification",
            "multi-hot label",
            "per-label sigmoid",
            "hamming_loss",
            "micro F1",
            "macro F1",
            "OneVsRestClassifier",
            "label co-occurrence",
            "다중 라벨 분류",
            "멀티핫",
            "라벨별 시그모이드",
            "라벨별 BCE",
            "공동 활성",
            "측면 라벨",
            "해밍 손실",
        ),
    ),
    Chapter(
        14,
        "auxiliary_loss",
        "보조 손실과 멀티태스크 학습 (Auxiliary Loss)",
        "보조 손실과 멀티태스크 학습 (Auxiliary Loss)",
        "다중 라벨 측면 분류에 별점 회귀 보조 헤드를 더해 결합 손실을 학습",
        (
            "Auxiliary loss",
            "multi-task learning",
            "combined loss",
            "BCEWithLogitsLoss",
            "MSELoss",
            "lambda",
            "auxiliary head",
            "compute_loss",
            "DataCollatorWithPadding",
            "remove_unused_columns",
            "custom Trainer",
            "보조 손실",
            "멀티태스크 학습",
            "결합 손실",
            "보조 헤드",
            "람다",
            "커스텀 Trainer",
            "별점 보조 회귀",
        ),
    ),
]


EXTRA_INDEXES = {
    1: (
        "Bag of Words",
        "BoW",
        "n-gram",
        "token_pattern",
        "fit_transform",
        "get_feature_names_out",
        "vocabulary_",
        "max_features",
        "min_df",
        "max_df",
        "OOV",
        "out-of-vocabulary",
        "CSR matrix",
        "dense matrix",
        "load_dataset",
        "Yelp review full",
        "pandas DataFrame",
        "단어 가방",
        "엔그램",
        "토큰 패턴",
        "어휘 사전",
        "어휘 수",
        "어휘 밖 단어",
        "밀집 행렬",
        "데이터 샘플링",
    ),
    2: (
        "train_test_split",
        "mean_absolute_error",
        "r2_score",
        "MAE",
        "R2 score",
        "residual",
        "prediction clipping",
        "target normalization",
        "np.clip",
        "regression head",
        "continuous target",
        "잔차",
        "타깃 정규화",
        "예측값 클리핑",
        "연속 타깃",
        "회귀 헤드",
        "평가 지표",
        "과대 예측",
        "과소 예측",
    ),
    3: (
        "binary cross entropy",
        "log loss",
        "probability threshold",
        "threshold tuning",
        "precision_score",
        "recall_score",
        "f1_score",
        "accuracy_score",
        "class_weight",
        "decision boundary",
        "positive class",
        "negative class",
        "이진 교차 엔트로피",
        "로그 손실",
        "확률 임계값",
        "임계값 튜닝",
        "양성 클래스",
        "음성 클래스",
        "결정 경계",
        "정확도",
        "F1 점수",
    ),
    4: (
        "logit difference",
        "2-logit softmax",
        "1-logit sigmoid",
        "softmax CE",
        "log-sum-exp",
        "coef_",
        "intercept_",
        "multi_class",
        "predict_proba",
        "binary equivalence",
        "one-hot label",
        "두 로짓 차이",
        "2차원 출력",
        "1차원 출력",
        "소프트맥스 CE",
        "원-핫 라벨",
        "동등성 증명",
    ),
    5: (
        "multinomial logistic regression",
        "OvR",
        "One-vs-Rest",
        "argmax",
        "weighted F1",
        "macro average",
        "weighted average",
        "baseline",
        "log K",
        "precision",
        "recall",
        "F1",
        "class_weight",
        "multi_class",
        "다항 로지스틱 회귀",
        "일대나머지",
        "상호배타 클래스",
        "클래스 경쟁",
        "가중 F1",
        "매크로 평균",
        "가중 평균",
        "기준선",
        "분류 리포트",
    ),
    6: (
        "per-label sigmoid",
        "subset accuracy",
        "label cardinality",
        "label density",
        'average="micro"',
        'average="macro"',
        "multi-hot vector",
        "aspect label",
        "label-wise threshold",
        "independent labels",
        "binary relevance",
        "label imbalance",
        "라벨별 시그모이드",
        "서브셋 정확도",
        "라벨 카디널리티",
        "라벨 밀도",
        "라벨별 임계값",
        "독립 라벨",
        "측면 키워드",
        "라벨 불균형",
        "이진 관련성",
    ),
    7: (
        "transformers",
        "pipeline(\"sentiment-analysis\")",
        "from_pretrained",
        "model.forward",
        "logits",
        "softmax",
        "argmax",
        "[CLS]",
        "[SEP]",
        "WordPiece prefix",
        "WordPiece 접두사",
        "SST-2",
        "sentiment analysis",
        "감성 분석",
        "모델 다운로드",
        "캐시",
        "토큰 ID",
        "후처리",
    ),
    8: (
        "memory mapping",
        "Dataset.select",
        "Dataset.shuffle",
        "with_format",
        "torch format",
        "token length distribution",
        "95th percentile",
        "padding=True",
        "padding=\"max_length\"",
        "truncation=True",
        "input_ids",
        "토큰 길이 분포",
        "메모리 매핑",
        "배치 패딩",
        "고정 길이 패딩",
        "동적 패딩",
    ),
    9: (
        "AutoModelForSequenceClassification",
        "num_labels=1",
        "problem_type=\"regression\"",
        "TrainingArguments",
        "evaluation_strategy",
        "save_strategy",
        "learning_rate",
        "per_device_train_batch_size",
        "weight_decay",
        "VRAM",
        "nvidia-smi",
        "정규방정식",
        "경사하강법",
        "학습률",
        "배치 크기",
        "가중치 감쇠",
    ),
    10: (
        "problem_type=\"multi_label_classification\"",
        "sigmoid probability",
        "threshold=0.5",
        "roc_auc_score",
        "seaborn.kdeplot",
        "probability distribution",
        "positive class",
        "negative class",
        "확률 분포",
        "양성 클래스",
        "음성 클래스",
        "결과 캐시",
    ),
    11: (
        "problem_type=\"single_label_classification\"",
        "stable softmax",
        "exp(x - max)",
        "scatter plot",
        "correlation",
        "four-quadrant analysis",
        "표준 분류 셋업",
        "안정 소프트맥스",
        "상관계수",
        "4분면 분석",
    ),
    12: (
        "seaborn.heatmap",
        "row-normalized confusion matrix",
        "multi-class AUC",
        "top-1 probability",
        "log K baseline",
        "ordinal label",
        "행 정규화 혼동 행렬",
        "다중 클래스 AUC",
        "최상위 확률",
        "순서형 라벨",
    ),
    13: (
        "AutoModelForSequenceClassification",
        "problem_type=\"multi_label_classification\"",
        "roc_auc_score",
        "seaborn.FacetGrid",
        "co-occurrence matrix",
        "conditional probability",
        "binary relevance",
        "측면 키워드",
        "조건부 확률",
        "라벨 공기",
        "이진 관련성",
    ),
    14: (
        "nn.Linear",
        "torch.nn.functional",
        "Trainer.compute_loss",
        "aux_labels",
        "lambda_aux",
        "uncertainty weighting",
        "custom data collator",
        "hidden_states",
        "return_outputs",
        "보조 라벨",
        "불확실성 가중치",
        "커스텀 데이터 콜레이터",
        "은닉 상태",
    ),
}


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
HANGUL_PATTERN = re.compile(r"[가-힣]")


def index_sort_prefix(term: str) -> str:
    return "0" if HANGUL_PATTERN.search(term) else "1"


def index_sort_key(term: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z가-힣]+", " ", term).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return f"{index_sort_prefix(term)}{normalized}"


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


def latex_escape_prose(text: str) -> str:
    """Escape prose that is inserted directly into LaTeX macro arguments."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


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


def format_embedded_listings(latex: str) -> str:
    """Apply book code wrapping to fenced code blocks embedded in markdown."""

    def repl(match: re.Match[str]) -> str:
        options = match.group(1) or ""
        source = match.group(2).strip("\n")
        if not source.strip():
            return match.group(0)
        formatted = format_code_for_book(source)
        line_count = len(formatted.splitlines())
        needspace = max(6, min(line_count + 4, 24))
        return (
            f"\\Needspace{{{needspace}\\baselineskip}}\n"
            f"\\begin{{lstlisting}}{options}\n"
            f"{formatted}\n"
            "\\end{lstlisting}"
        )

    return re.sub(
        r"\\begin\{lstlisting\}(\[[^\]]*\])?\n(.*?)\n\\end\{lstlisting\}",
        repl,
        latex,
        flags=re.DOTALL,
    )


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


def clean_table_caption_title(section_title: str) -> str:
    section_title = re.sub(
        r"\\texorpdfstring\{.*?\}\{(.*?)\}",
        r"\1",
        section_title,
        flags=re.DOTALL,
    )
    section_title = re.sub(r"\\inlinecode\{([^{}]+)\}", r"\1", section_title)
    section_title = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", section_title)
    section_title = section_title.replace("---", "-")
    return re.sub(r"\s+", " ", section_title).strip()


def caption_for_table(chapter_number: int, section_title: str, table_index: int) -> str:
    title = clean_table_caption_title(section_title)
    if "변화추적표" in title:
        return f"{chapter_number}장 변화추적표"
    if "변경점" in title:
        return f"{chapter_number}장 변경점 요약"
    if "등장한 라이브러리" in title:
        return f"{chapter_number}장 새로 등장한 라이브러리"
    if "Loss" in title or "수치 예시" in title:
        return f"{chapter_number}장 손실 수치 예시"
    if title:
        return f"{chapter_number}장 {title} 표"
    return f"{chapter_number}장 표 {table_index}"


def wrap_tabular_tables(latex: str, chapter_number: int) -> str:
    lines = latex.splitlines()
    wrapped: list[str] = []
    section_title = ""
    table_index = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        section_match = re.match(r"\\(?:section|subsection)\{(.+)\}$", line)
        if section_match:
            section_title = section_match.group(1)

        if line.startswith(r"\begin{adjustbox}"):
            block = [line]
            depth = 1
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].startswith(r"\begin{adjustbox}"):
                    depth += 1
                if lines[i].startswith(r"\end{adjustbox}"):
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            block_text = "\n".join(block)
            if r"\begin{tabular}" in block_text or r"\begin{tabularx}" in block_text:
                table_index += 1
                caption = caption_for_table(chapter_number, section_title, table_index)
                label = f"tab:ch{chapter_number:02d}-{table_index:02d}"
                wrapped.append(f"\\begin{{booktable}}{{{caption}}}{{{label}}}")
                wrapped.extend(block)
                wrapped.append(r"\end{booktable}")
            else:
                wrapped.extend(block)
        else:
            wrapped.append(line)
        i += 1
    return "\n".join(wrapped)


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


def normalize_prose_quotes(latex: str) -> str:
    """Use directional quotes in prose while preserving code-like fragments."""

    protected_pattern = re.compile(r"\\(?:inlinecode|texttt)\{[^{}]*\}")

    def normalize_segment(segment: str) -> str:
        protected: list[str] = []

        def protect(match: re.Match[str]) -> str:
            protected.append(match.group(0))
            return f"PROTECTEDQUOTE{len(protected) - 1}END"

        segment = protected_pattern.sub(protect, segment)
        segment = re.sub(r'"([^"\n]+)"', r"“\1”", segment)
        for idx, original in enumerate(protected):
            segment = segment.replace(f"PROTECTEDQUOTE{idx}END", original)
        return segment

    normalized: list[str] = []
    in_listing = False
    for line in latex.splitlines():
        if line.startswith(r"\begin{lstlisting}"):
            in_listing = True
            normalized.append(line)
            continue
        if line.startswith(r"\end{lstlisting}"):
            in_listing = False
            normalized.append(line)
            continue
        normalized.append(line if in_listing else normalize_segment(line))
    return "\n".join(normalized)


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
        "비활성 스칼라 출력를": "비활성 스칼라 출력을",
        "어휘 크기": "어휘 수",
        "전체 칸 수": "전체 원소 수",
        "비어있는 칸": "0인 원소",
        "처음 20개": "어휘 앞 20개",
        "가장 자주 등장한 단어 top 10": "등장 빈도 상위 10개 단어",
        "앞 3개": "첫 3개",
        "앞 5개": "첫 5개",
        "성공? coef_ shape": "학습 성공: coef_ shape",
        "OvR fit 성공!": "OvR 학습 성공",
        "실제 별점": "정답 별점",
        "다음 장를": "다음 장을",
        "1·2장와": "1·2장과",
        "2장와": "2장과",
        "2장. sklearn Regression --- 시작점": "2장. 회귀 분석 (Regression \\& MSE) --- 첫 모델과 손실",
        "3장. sklearn Binary --- 출력에 sigmoid가 붙다": "3장. 이진 분류 (Binary Classification \\& BCE) --- 출력에 sigmoid가 붙다",
        "4장. sklearn Multi-class --- sigmoid가 softmax로": "4장. sigmoid와 softmax의 동등성 (Binary Classification) --- 같은 문제, 다른 표현",
        "5장. sklearn Multi-class --- K=5로 진짜 일반화": "5장. 다중 클래스 분류 (Multi-class Classification \\& CE) --- K=5로 일반화",
        "6장. sklearn Multi-label --- softmax 합=1 제약을 푼다": "6장. 다중 라벨 분류 (Multi-label Classification \\& Per-label BCE) --- softmax 합=1 제약을 푼다",
        "2장. 회귀 분석과 MSELoss --- 첫 모델과 Loss": "2장. 회귀 분석 (Regression \\& MSE) --- 첫 모델과 손실",
        "3장. 이진 분류와 BCEWithLogitsLoss --- 출력에 sigmoid가 붙다": "3장. 이진 분류 (Binary Classification \\& BCE) --- 출력에 sigmoid가 붙다",
        "4장. 이진 분류의 sigmoid-softmax 동등성 --- 같은 문제, 다른 표현": "4장. sigmoid와 softmax의 동등성 (Binary Classification) --- 같은 문제, 다른 표현",
        "5장. 다중 클래스 분류와 CrossEntropyLoss --- K=5로 일반화": "5장. 다중 클래스 분류 (Multi-class Classification \\& CE) --- K=5로 일반화",
        "6장. 다중 라벨 분류와 per-label BCE --- softmax 합=1 제약을 푼다": "6장. 다중 라벨 분류 (Multi-label Classification \\& Per-label BCE) --- softmax 합=1 제약을 푼다",
        "2장. 회귀 분석과 평균제곱오차 --- 첫 모델과 손실": "2장. 회귀 분석 (Regression \\& MSE) --- 첫 모델과 손실",
        "3장. 이진 분류와 이진 교차 엔트로피 --- 출력에 sigmoid가 붙다": "3장. 이진 분류 (Binary Classification \\& BCE) --- 출력에 sigmoid가 붙다",
        "4장. 이진 분류: sigmoid와 softmax는 어떻게 같은가 --- 같은 문제, 다른 표현": "4장. sigmoid와 softmax의 동등성 (Binary Classification) --- 같은 문제, 다른 표현",
        "5장. 다중 클래스 분류와 교차 엔트로피 --- K=5로 일반화": "5장. 다중 클래스 분류 (Multi-class Classification \\& CE) --- K=5로 일반화",
        "6장. 다중 라벨 분류와 라벨별 이진 교차 엔트로피 --- softmax 합=1 제약을 푼다": "6장. 다중 라벨 분류 (Multi-label Classification \\& Per-label BCE) --- softmax 합=1 제약을 푼다",
        "Loss 함수의 변화 --- \\inlinecode{MSELoss} 등장": "손실 함수의 변화 --- 평균제곱오차 등장",
        "Loss 함수의 변화 --- \\inlinecode{BCEWithLogitsLoss} 등장": "손실 함수의 변화 --- 이진 교차 엔트로피 등장",
        "Loss 함수의 변화 --- \\inlinecode{CrossEntropyLoss} 등장": "손실 함수의 변화 --- 교차 엔트로피 등장",
        "Loss 함수의 변화 --- \\inlinecode{BCEWithLogitsLoss} per-label": "손실 함수의 변화 --- 라벨별 이진 교차 엔트로피",
        "Loss 함수의 변화 --- MSELoss 등장": "손실 함수의 변화 --- 평균제곱오차 등장",
        "Loss 함수의 변화 --- BCEWithLogitsLoss 등장": "손실 함수의 변화 --- 이진 교차 엔트로피 등장",
        "Loss 함수의 변화 --- CrossEntropyLoss 등장": "손실 함수의 변화 --- 교차 엔트로피 등장",
        "Loss 함수의 변화 --- BCEWithLogitsLoss per-label": "손실 함수의 변화 --- 라벨별 이진 교차 엔트로피",
        "Loss 노트 --- 같은 CE, K=5 수치 예시": "손실 노트 --- 같은 교차 엔트로피, K=5 수치 예시",
        "Loss 한 단계 더: 학습된 모델의 실제 예측으로 BCE 분해": "손실 한 단계 더: 학습된 모델의 실제 예측으로 BCE 분해",
        "7장. BERT 첫 만남 --- \\inlinecode{pipeline} 한 줄과 그 안의 4단계": "7장. BERT 첫 만남 (Pipeline) --- 한 줄 뒤의 4단계",
        "8장. Tokenizer 깊게 보기 + Datasets 라이브러리": "8장. 토크나이저 옵션과 데이터셋 (Tokenizer \\& Datasets)",
        "8장. Tokenizer 옵션 깊이 + \\inlinecode{datasets} 라이브러리": "8장. 토크나이저 옵션과 데이터셋 (Tokenizer \\& Datasets)",
        "9장. BERT 회귀 --- 첫 파인튜닝, 첫 \\inlinecode{Trainer}": "9장. BERT 회귀 분석 (Regression \\& Trainer)",
        "10장. BERT Binary 방식 A --- sigmoid+BCE": "10장. BERT 이진 분류 A (Sigmoid \\& BCE)",
        "10장. BERT Binary 방식 A --- sigmoid + BCEWithLogitsLoss": "10장. BERT 이진 분류 A (Sigmoid \\& BCE)",
        "11장. BERT Binary 방식 B --- softmax+CE": "11장. BERT 이진 분류 B (Softmax \\& CE)",
        "11장. BERT Binary 방식 B --- softmax + CrossEntropyLoss": "11장. BERT 이진 분류 B (Softmax \\& CE)",
        "12장. BERT Multi-class --- Yelp 5클래스": "12장. BERT 다중 클래스 분류 (Multi-class \\& CE)",
        "13장. BERT Multi-label --- Yelp 측면 키워드": "13장. BERT 다중 라벨 분류 (Multi-label \\& Per-label BCE)",
        "14장. BERT Auxiliary Loss --- 측면 분류 + 별점 보조 회귀 (Phase 1 클라이맥스)": "14장. 보조 손실과 멀티태스크 학습 (Auxiliary Loss)",
        "Loss 노트": "손실 노트",
    }
    for before, after in replacements.items():
        latex = latex.replace(before, after)

    # More formal section/table labels after chapter-reference normalization.
    latex = re.sub(r"변경점 \(Diff from ([0-9]+장)\)", r"변경점: \1 대비", latex)
    latex = latex.replace("전체 18장 표", "전체 18개 장의 표")
    latex = latex.replace("전체 19장 표", "전체 19개 장의 표")
    latex = latex.replace(r"\#장별-변화추적표", r"\#챕터별-변화추적표")
    latex = normalize_heading_titles(latex)
    return latex


def split_latex_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
    return None


def clean_heading_title(title: str) -> str:
    replacements = {
        "0. 환경 준비": "환경 준비",
        "1. 실습: 일단 돌려봅시다": "실습: 파이프라인 실행",
        "실습: 일단 돌려봅시다": "실습: 파이프라인 실행",
        "2. 해부: pipeline 안에서는 뭐가 일어났을까?": "해부: 파이프라인 내부",
        "해부: pipeline 안에서는 뭐가 일어났을까?": "해부: 파이프라인 내부",
        "3. 변형: pipeline 없이 직접 해보기": "변형: 직접 추론",
        "변형: pipeline 없이 직접 해보기": "변형: 직접 추론",
        "보너스: 토크나이저마다 어휘가 다르다": "토크나이저 어휘 비교",
        "보너스: \\inlinecode{model.config} 안에 뭐가 있나": "모델 설정 확인",
        "보너스: model.config 안에 뭐가 있나": "모델 설정 확인",
        "다른 task도 같은 패턴": "다른 태스크의 파이프라인",
        "\\inlinecode{!nvidia-smi} --- GPU 메모리(VRAM) 실시간 추적": "GPU 메모리 확인",
        "!nvidia-smi --- GPU 메모리(VRAM) 실시간 추적": "GPU 메모리 확인",
        "등장 인물 정리": "구성 요소",
        "Tokenizer와 Model 직접 로드": "토크나이저와 모델 로드",
        "텍스트 → 숫자 (Tokenization)": "토큰화",
        "숫자 → 로짓 (Model forward)": "모델 추론",
        "로짓 → 확률/라벨 (Post-processing)": "후처리",
        "특수 토큰(special token)이란": "특수 토큰",
        "\\inlinecode{model.config} 의 자주 쓰는 속성": "모델 설정 속성",
        "model.config 의 자주 쓰는 속성": "모델 설정 속성",
        "\\inlinecode{torch} 의 후처리·연산 함수": "후처리 연산",
        "torch 의 후처리·연산 함수": "후처리 연산",
        "\\inlinecode{torch} 자체": "PyTorch",
        "torch 자체": "PyTorch",
        "토크나이저 노트 --- \\inlinecode{padding} / \\inlinecode{truncation} / \\inlinecode{max\\_length}": "토크나이저 노트",
        "토크나이저 노트 --- padding / truncation / max\\_length": "토크나이저 노트",
        "\\inlinecode{datasets} 로 Yelp 로드": "데이터셋 로드",
        "datasets 로 Yelp 로드": "데이터셋 로드",
        "토크나이저 옵션 직접 실험": "토크나이저 옵션",
        "옵션 없이 --- 한 문장 토큰화 (기본 동작)": "기본 토큰화",
        "두 문장 배치 + \\inlinecode{padding=True} --- \\emph{동적 패딩}": "동적 패딩",
        "두 문장 배치 + padding=True --- 동적 패딩": "동적 패딩",
        "\\inlinecode{padding=\"max\\_length\"}, \\inlinecode{max\\_length=128} --- \\emph{고정 길이}": "고정 길이 패딩",
        "padding=“max\\_length”, max\\_length=128 --- 고정 길이": "고정 길이 패딩",
        "\\inlinecode{truncation=True} --- 긴 입력 자르기": "긴 입력 자르기",
        "truncation=True --- 긴 입력 자르기": "긴 입력 자르기",
        "attention\\_mask가 self-attention에서 하는 일": "attention mask",
        "\\inlinecode{max\\_length} 결정 --- 데이터 길이 분포 보고 정하기": "max\\_length 결정",
        "max\\_length 결정 --- 데이터 길이 분포 보고 정하기": "max\\_length 결정",
        "\\inlinecode{datasets.map} --- 5,000건 일괄 토큰화": "일괄 토큰화",
        "datasets.map --- 5,000건 일괄 토큰화": "일괄 토큰화",
        "\\inlinecode{dataset.filter} --- 조건에 맞는 샘플만 선별": "샘플 필터링",
        "dataset.filter --- 조건에 맞는 샘플만 선별": "샘플 필터링",
        "\\inlinecode{with\\_format(\"torch\")} --- 텐서 형식으로": "텐서 형식 변환",
        "with\\_format(“torch”) --- 텐서 형식으로": "텐서 형식 변환",
        "\\inlinecode{DataLoader} 변환 --- 9장 학습 입력 미리보기": "DataLoader 변환",
        "DataLoader 변환 --- 9장 학습 입력 미리보기": "DataLoader 변환",
        "\\inlinecode{DataCollator} --- 동적 padding을 배치 시점에": "DataCollator",
        "DataCollator --- 동적 padding을 배치 시점에": "DataCollator",
        "향후 학습 코드 관점 --- 9-13장에서 실제로 어떻게 쓰이나": "학습 코드와의 연결",
        "Collator 추가 실습": "Collator 실습",
        "실험 1 --- 정적 vs 동적 padding 효율을 숫자로": "정적 패딩과 동적 패딩",
        "실험 2 --- \\inlinecode{DataCollatorForLanguageModeling} 으로 MLM masking 직접 보기": "MLM 마스킹",
        "실험 2 --- DataCollatorForLanguageModeling 으로 MLM masking 직접 보기": "MLM 마스킹",
        "실험 2b --- GPT-style CLM 도 같은 collator로": "CLM 입력 구성",
        "실험 3 --- 커스텀 \\inlinecode{collate\\_fn} 직접 작성": "커스텀 collate\\_fn",
        "실험 3 --- 커스텀 collate\\_fn 직접 작성": "커스텀 collate\\_fn",
        "손실 노트 --- \\inlinecode{MSELoss} 그대로, 최소화 방식만 바뀜": "손실 노트",
        "손실 노트 --- MSELoss 그대로, 최소화 방식만 바뀜": "손실 노트",
        "데이터 준비": "데이터 준비",
        "\\inlinecode{num\\_labels=1}, \\inlinecode{problem\\_type=\"regression\"}": "모델 로드",
        "모델 로드 --- num\\_labels=1, problem\\_type=“regression”": "모델 로드",
        "\\inlinecode{TrainingArguments} + \\inlinecode{Trainer}": "Trainer 설정",
        "TrainingArguments + Trainer": "Trainer 설정",
        "평가 --- sklearn(2장)과 직접 비교": "평가",
        "시각 1 --- 예측 분포 per actual class": "예측 분포",
        "시각 2 --- 잔차(Residual = Predicted − Actual) 분포 per actual class": "잔차 분포",
        "변형 --- 학습이 어디서 망가지는지 (개념만)": "변형: 학습 실패 요인",
        "학습되는 파라미터 vs 동결된 파라미터": "학습 파라미터",
        "시연: BERT 본체 동결 패턴": "BERT 본체 동결",
        "\\inlinecode{transformers} 학습 도구": "transformers 학습 도구",
        "Trainer가 자동으로 해주는 일": "Trainer의 역할",
        "\\inlinecode{compute\\_metrics} 함수 시그니처": "compute\\_metrics",
        "num\\_labels=1 + problem\\_type=“multi\\_label\\_classification” 의 트릭": "Sigmoid 방식의 설정",
        "손실 노트 --- \\inlinecode{BCEWithLogitsLoss} (3장 그대로, BERT 맥락에서 다시)": "손실 노트",
        "손실 노트 --- BCEWithLogitsLoss (3장 그대로, BERT 맥락에서 다시)": "손실 노트",
        "데이터 --- Yelp 이진화 (3·4장와 동일)": "데이터 준비",
        "모델 로드 --- 방식 A 셋업": "모델 로드",
        "학습 --- 9장 골격 그대로": "학습",
        "평가 --- sigmoid 확률 분포 직접 확인": "평가",
        "메인 그림 --- \\emph{확률 공간} 에서 라벨별 분포 (\\inlinecode{seaborn.kdeplot})": "확률 분포",
        "메인 그림 --- 확률 공간 에서 라벨별 분포 (seaborn.kdeplot)": "확률 분포",
        "보조 그림 --- \\emph{logit 공간} 에서 같은 분포 (\\inlinecode{BCE가\\ 실제로\\ 동작하는\\ 자리})": "로짓 분포",
        "보조 그림 --- logit 공간 에서 같은 분포 (BCE가 실제로 동작하는 자리)": "로짓 분포",
        "결과 저장 --- 11장에서 비교용": "결과 저장",
        "왜 두 방식이 거의 같은 결과를 내야 하는가 (수식 한 줄)": "두 방식의 동등성",
        "손실 노트 --- \\inlinecode{CrossEntropyLoss} (4장 그대로, BERT 맥락)": "손실 노트",
        "손실 노트 --- CrossEntropyLoss (4장 그대로, BERT 맥락)": "손실 노트",
        "데이터 --- Yelp 이진화 (10장과 정확히 동일)": "데이터 준비",
        "모델 로드 --- 방식 B 셋업": "모델 로드",
        "학습 --- 10장과 동일한 hyperparams": "학습",
        "평가 --- softmax 확률 분포": "평가",
        "메인 그림 --- \\emph{확률 공간} 분포 (10장과 같은 KDE)": "확률 분포",
        "메인 그림 --- 확률 공간 분포 (10장과 같은 KDE)": "확률 분포",
        "보조 그림 --- \\(z = z_1 - z_0\\) 의 logit 공간 분포": "로짓 차이 분포",
        "클라이맥스 --- 방식 A 를 \\emph{이 노트북 안에서} 다시 학습해 비교": "방식 A/B 비교",
        "클라이맥스 --- 방식 A 를 이 노트북 안에서 다시 학습해 비교": "방식 A/B 비교",
        "두 방식의 metric 표 비교": "평가지표 비교",
        "샘플 단위 확률 비교 --- scatter plot": "확률 산점도",
        "예측 일치율 (threshold 0.5)": "예측 일치율",
        "손실 노트 --- \\inlinecode{CrossEntropyLoss} 가 K=5 에서 어떻게 보이나": "손실 노트",
        "손실 노트 --- CrossEntropyLoss 가 K=5 에서 어떻게 보이나": "손실 노트",
        "데이터 --- Yelp 별점 1-5 (5장와 동일)": "데이터 준비",
        "모델 로드 --- \\inlinecode{num\\_labels=5} 만 바뀜": "모델 로드",
        "모델 로드 --- num\\_labels=5 만 바뀜": "모델 로드",
        "학습 --- 11장과 동일한 hyperparams": "학습",
        "평가 --- softmax 확률 분포와 혼동 패턴": "평가",
        "메인 그림 --- 혼동 행렬 (\\inlinecode{seaborn.heatmap})": "혼동 행렬",
        "메인 그림 --- 혼동 행렬 (seaborn.heatmap)": "혼동 행렬",
        "보조 그림 --- top-1 확률의 분포 (정답/오답 갈림)": "최상위 확률 분포",
        "클라이맥스 --- sklearn TF-IDF + LogReg 와의 비교 (5장의 BERT 검증)": "BERT와 TF-IDF 비교",
        "두 모델의 metric 표 비교": "평가지표 비교",
        "두 모델의 혼동 행렬 비교": "혼동 행렬 비교",
        "Loss 노트 --- \\inlinecode{BCEWithLogitsLoss} per-label (6장 그대로, BERT 맥락)": "손실 노트",
        "손실 노트 --- \\inlinecode{BCEWithLogitsLoss} per-label (6장 그대로, BERT 맥락)": "손실 노트",
        "Loss 노트 --- BCEWithLogitsLoss per-label (6장 그대로, BERT 맥락)": "손실 노트",
        "손실 노트 --- BCEWithLogitsLoss per-label (6장 그대로, BERT 맥락)": "손실 노트",
        "데이터 --- Yelp + 측면(aspect) 합성 라벨 (6장과 동일)": "데이터 준비",
        "모델 로드 --- \\inlinecode{num\\_labels=5} + \\inlinecode{multi\\_label\\_classification}": "모델 로드",
        "모델 로드 --- num\\_labels=5 + multi\\_label\\_classification": "모델 로드",
        "학습 --- 12장과 동일한 hyperparams": "학습",
        "평가 --- 라벨별 sigmoid 확률 + 활성 패턴": "평가",
        "메인 그림 --- 라벨별 sigmoid 확률 KDE (5 패널)": "라벨별 확률 분포",
        "보조 그림 --- 라벨 간 공동 활성 패턴": "라벨 공동 활성",
        "클라이맥스 --- 6장 sklearn \\inlinecode{OneVsRestClassifier(LogisticRegression)} 와 비교": "BERT와 sklearn 비교",
        "클라이맥스 --- 6장 sklearn OneVsRestClassifier(LogisticRegression) 와 비교": "BERT와 sklearn 비교",
        "두 모델의 metric 비교": "평가지표 비교",
        "라벨별 F1 비교 --- 어디서 BERT가 이기나": "라벨별 F1 비교",
        "왜 Auxiliary Loss 인가 --- 다섯 가지 동기": "왜 보조 손실인가",
        "Loss 노트 --- Combined loss \\inlinecode{L = L\\_main + λ · L\\_aux}": "손실 노트",
        "손실 노트 --- Combined loss \\inlinecode{L\\ =\\ L\\_main\\ +\\ λ\\ ·\\ L\\_aux}": "손실 노트",
        "Loss 노트 --- Combined loss L = L\\_main + λ · L\\_aux": "손실 노트",
        "손실 노트 --- Combined loss L = L\\_main + λ · L\\_aux": "손실 노트",
        "데이터 --- Yelp + 측면 (13장) + 별점 보조 라벨": "데이터 준비",
        "토큰화 --- 메인 multi-hot + 보조 float 같이 부착": "토큰화",
        "커스텀 Data Collator --- \\inlinecode{aux\\_labels} 도 batch에 같이 담기": "커스텀 Data Collator",
        "커스텀 Data Collator --- aux\\_labels 도 batch에 같이 담기": "커스텀 Data Collator",
        "모델 셋업 --- 13장 모델 + 보조 헤드 한 줄 추가": "모델 셋업",
        "커스텀 Trainer --- \\inlinecode{compute\\_loss} 오버라이드": "커스텀 Trainer",
        "커스텀 Trainer --- compute\\_loss 오버라이드": "커스텀 Trainer",
        "학습 --- λ=1 (보조 ON)": "학습",
        "평가 --- 메인 task + 보조 task": "평가",
        "클라이맥스 --- \\emph{λ=0 baseline} 학습 (= 13장 재현)": "λ=0 기준선 비교",
        "클라이맥스 --- λ=0 baseline 학습 (= 13장 재현)": "λ=0 기준선 비교",
        "메인 metric 비교 --- λ=0 baseline vs λ=1 aux": "메인 지표 비교",
        "라벨별 F1 비교 --- 어느 측면이 보조 loss로 가장 도움받았나": "라벨별 F1 비교",
        "보조 task 자체는 얼마나 잘 학습됐나": "보조 태스크 평가",
    }
    title = re.sub(r"^\s*(?:[0-9]+(?:-[0-9]+)?|[0-9]+[A-Za-z]?)\.\s*", "", title)
    title = re.sub(r"^\s*Step\s+[0-9]+:\s*", "", title)
    title = replacements.get(title, title)
    generic_prefixes = {
        "토크나이저 노트 ---": "토크나이저 노트",
        "데이터 ---": "데이터 준비",
        "모델 로드 ---": "모델 로드",
        "학습 ---": "학습",
        "평가 ---": "평가",
        "메인 그림 ---": "",
        "보조 그림 ---": "",
        "클라이맥스 ---": "비교 실험",
        "변형 ---": "변형",
    }
    for prefix, replacement in generic_prefixes.items():
        if title.startswith(prefix):
            tail = title[len(prefix) :].strip()
            if replacement in {"데이터 준비", "모델 로드", "학습", "평가", "토크나이저 노트"}:
                title = replacement
            elif replacement:
                title = replacement if not tail else f"{replacement}: {tail}"
            else:
                title = tail
            break
    return title.strip()


def normalize_heading_content(content: str) -> str:
    if content.startswith(r"\texorpdfstring"):
        pos = len(r"\texorpdfstring")
        first = split_latex_group(content, pos)
        if first is None:
            return clean_heading_title(content)
        first_text, first_end = first
        second = split_latex_group(content, first_end)
        if second is None:
            return clean_heading_title(content)
        second_text, second_end = second
        return (
            r"\texorpdfstring"
            + "{"
            + clean_heading_title(first_text)
            + "}{"
            + clean_heading_title(second_text)
            + "}"
            + content[second_end:]
        )
    return clean_heading_title(content)


def normalize_heading_titles(latex: str) -> str:
    normalized: list[str] = []
    for line in latex.splitlines():
        match = re.match(r"^(\\(?:section|subsection|subsubsection)\*?)(.*)$", line)
        if not match:
            normalized.append(line)
            continue
        command, rest = match.groups()
        group = split_latex_group(rest, 0)
        if group is None:
            normalized.append(line)
            continue
        content, end = group
        normalized.append(command + "{" + normalize_heading_content(content) + "}" + rest[end:])
    return "\n".join(normalized)


def polish_code_comments(source: str) -> str:
    source = re.sub(r"\bChapter\s+([0-9]+)", r"\1장", source)
    source = re.sub(r"\bCh\s*([0-9]+)\s*-\s*([0-9]+)", r"\1-\2장", source)
    source = re.sub(r"\bCh\s*([0-9]+)", r"\1장", source)
    source = source.replace('multi_class="multinomial", ', "")
    source = source.replace(", multi_class=\"multinomial\"", "")
    source = source.replace("그냥", "직접")
    source = source.replace("뱉는", "출력하는")
    source = source.replace("뱉을", "출력할")
    source = source.replace("뱉습니다", "출력합니다")
    source = source.replace("뱉은", "출력한")
    source = source.replace("어휘 크기", "어휘 수")
    source = source.replace("전체 칸 수", "전체 원소 수")
    source = source.replace("비어있는 칸", "0인 원소")
    source = source.replace("처음 20개", "어휘 앞 20개")
    source = source.replace("가장 자주 등장한 단어 top 10", "등장 빈도 상위 10개 단어")
    source = source.replace("앞 3개", "첫 3개")
    source = source.replace("앞 5개", "첫 5개")
    source = source.replace("성공? coef_ shape", "학습 성공: coef_ shape")
    source = source.replace("OvR fit 성공!", "OvR 학습 성공")
    source = source.replace("실제 별점", "정답 별점")
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
    raw_blocks: list[str] = []

    def protect_raw_latex(match: re.Match[str]) -> str:
        raw_blocks.append(match.group(0))
        return f"\nRAWLATEXBLOCK{len(raw_blocks) - 1}END\n"

    markdown = re.sub(
        r"\\begin\{bookfigure(?:label)?\}.*?\\end\{bookfigure(?:label)?\}",
        protect_raw_latex,
        markdown,
        flags=re.DOTALL,
    )
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
    for idx, raw_block in enumerate(raw_blocks):
        latex = latex.replace(f"RAWLATEXBLOCK{idx}END", raw_block)
    latex = strip_pandoc_targets(latex)
    latex = normalize_code_blocks(latex)
    latex = format_embedded_listings(latex)
    latex = faq_subsections_to_questions(latex)
    latex = normalize_tables(latex)
    latex = normalize_inline_code(latex)
    latex = re.sub(
        r"\\textbackslash ref\\\{([^{}]+)\\\}",
        r"\\ref{\1}",
        latex,
    )
    latex = latex.replace(r"\textasciitilde\{\}\ref", r"~\ref")
    latex = latex.replace(r"\textasciitilde\ref", r"~\ref")
    latex = wrap_faq_blocks(latex)
    latex = polish_book_prose(latex)
    latex = normalize_prose_quotes(latex)
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
        joined = " ".join(code_lines).split("  # ", 1)[0].strip()
        name = variable_name(joined)
        if name:
            joined = f"{name} = ..."
        elif joined.startswith("display("):
            joined = "display(...)"
        elif "." in joined and "(" in joined:
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\()", joined)
            if match:
                joined = match.group(1) + "...)"
        if len(joined) > 28:
            joined = joined[:25].rstrip() + "..."
        return f"\\inlinecode{{{latex_escape_text(joined)}}}"

    def variable_name(text: str) -> str:
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", text)
        return match.group(1) if match else ""

    def assignment_message(text: str) -> str:
        name = variable_name(text)
        if "get_feature_names_out" in text or name == "vocab":
            return "벡터라이저가 학습한 어휘 목록을 가져옵니다."
        if "CountVectorizer" in text:
            return "단어 횟수 기반 벡터라이저를 만듭니다."
        if "TfidfVectorizer" in text:
            return "TF-IDF 기반 벡터라이저를 만듭니다."
        if name == "sample":
            return "토큰화 예제로 사용할 문장을 정합니다."
        if name == "sparsity":
            return "행렬에서 0인 원소의 비율을 계산합니다."
        if "value_counts" in text:
            return "라벨별 샘플 개수를 집계합니다."
        if "to_pandas()" in text or name == "df":
            return "데이터셋을 표 형태로 바꿔 이후 셀에서 다루기 쉽게 합니다."
        if "shuffle(" in text and "select(" in text:
            return "전체 데이터에서 실습에 사용할 샘플만 추립니다."
        if "sum(axis=0)" in text or name in {"word_counts", "raw_sums"}:
            return "열 방향으로 값을 더해 특성별 합계를 계산합니다."
        if "argsort" in text or name == "top":
            return "값이 큰 항목부터 볼 수 있도록 인덱스를 정렬합니다."
        if ".build_analyzer()" in text or name == "analyzer":
            return "벡터라이저 내부의 토큰화 규칙을 직접 호출할 함수로 꺼냅니다."
        if "np.array" in text:
            return "비교 실험에 사용할 작은 배열을 만듭니다."
        if ".values" in text or ".to_numpy" in text:
            return "계산에 바로 쓸 수 있도록 배열 형태로 변환합니다."
        if "np.abs" in text or name in {"diff", "manual_bce", "manual_mse", "sklearn_mse"}:
            return "두 계산 결과가 얼마나 다른지 확인할 값을 만듭니다."
        if "threshold" in text or name.endswith("thr"):
            return "예측 확률을 0/1 라벨로 바꿀 기준값을 정합니다."
        if "DataFrame" in text:
            return "결과를 표로 보기 좋게 정리합니다."
        if "np.clip" in text:
            return "예측값을 허용 범위 안으로 잘라 후처리합니다."
        return "이후 분석에서 사용할 값을 준비합니다."

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
    for start_line, end_line, content in statements[:6]:
        text = " ".join(line.strip() for line in content)
        if text.startswith(("print(", "display(")) or " print(" in text:
            continue
        if text.startswith(("warnings.", "plt.")):
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
            message = assignment_message(text)
        else:
            message = "앞 단계에서 만든 값을 바탕으로 다음 계산을 수행합니다."
        snippet = summarize_code(content)
        if start_line == end_line:
            line_label = f"{start_line}행"
        else:
            line_label = f"{start_line}--{end_line}행"
        notes.append(f"\\textbf{{{line_label}}}의 {snippet}에서는 {message}")

    if not notes:
        return ""

    return (
        "\\begin{codeRead}\n"
        + " ".join(notes)
        + "\n\\end{codeRead}"
    )


def output_text(outputs: list[dict]) -> str:
    chunks: list[str] = []
    for output in outputs:
        output_type = output.get("output_type")
        if output_type == "stream":
            text = output.get("text", "")
            chunks.append("".join(text) if isinstance(text, list) else str(text))
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            html = data.get("text/html")
            if isinstance(html, list):
                html = "".join(html)
            if isinstance(html, str) and "<table" in html:
                table_text = "\n\n".join(html_tables_to_plain_text(html))
                if table_text:
                    chunks.append(table_text)
                    continue
            text = data.get("text/plain")
            if text:
                chunks.append("".join(text) if isinstance(text, list) else str(text))
        elif output_type == "error":
            traceback = output.get("traceback", [])
            if traceback:
                chunks.append("\n".join(str(line) for line in traceback[-8:]))
            else:
                chunks.append(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}")
    text = "\n".join(chunk.rstrip() for chunk in chunks if chunk and chunk.strip()).strip()
    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)
    if not text:
        return ""
    skip_patterns = (
        "TqdmWarning:",
        "IProgress not found",
        "Requirement already satisfied:",
        "WARNING: Running pip",
        "[notice] A new release of pip",
        "notice] A new release of pip",
        "To update, run:",
    )
    lines = [
        line
        for line in text.splitlines()
        if not any(pattern in line for pattern in skip_patterns)
        and not line.strip().startswith("from .autonotebook import tqdm")
    ]
    text = sanitize_symbols("\n".join(lines).strip())
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > 18:
        lines = lines[:16] + ["..."]
    compact = "\n".join(lines)
    if len(compact) > 1600:
        compact = compact[:1550].rstrip() + "\n..."
    return fit_listing_text(compact, width=78)


class PandasTableParser(HTMLParser):
    """Extract simple pandas DataFrame HTML tables from notebook output."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict[str, list[list[str]]]] = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_is_header = False
        self.current_cell: list[str] = []
        self.current_row: list[tuple[bool, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.in_cell:
            text = unescape("".join(self.current_cell))
            text = re.sub(r"\s+", " ", text).strip()
            self.current_row.append((self.cell_is_header, text))
            self.in_cell = False
            self.current_cell = []
        elif tag == "tr" and self.in_row:
            if self.current_row and self.tables:
                values = [value for _, value in self.current_row]
                header_count = sum(1 for is_header, _ in self.current_row if is_header)
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

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def latex_escape_cell(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_\allowbreak{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def html_tables_to_latex(html: str) -> list[str]:
    parser = PandasTableParser()
    parser.feed(html)
    tables: list[str] = []
    for table in parser.tables:
        headers = table["headers"]
        rows = table["rows"]
        if not headers or not rows:
            continue
        width = max(len(headers), *(len(row) for row in rows))
        headers = (headers + [""] * width)[:width]
        rows = [(row + [""] * width)[:width] for row in rows]
        spec = "Y" * width
        body = [
            "\\par\\noindent\\textbf{출력 표.}\\par\\smallskip",
            "\\begingroup\\scriptsize",
            "\\begin{adjustbox}{max width=.98\\linewidth}",
            f"\\begin{{tabularx}}{{.98\\linewidth}}{{@{{}}{spec}@{{}}}}",
            "\\toprule",
            " & ".join(latex_escape_cell(cell) for cell in headers) + r" \\",
            "\\midrule",
        ]
        for row in rows[:18]:
            body.append(" & ".join(latex_escape_cell(cell) for cell in row) + r" \\")
        if len(rows) > 18:
            body.append(r"\multicolumn{" + str(width) + r"}{@{}l@{}}{\ldots} \\")
        body.extend(["\\bottomrule", "\\end{tabularx}", "\\end{adjustbox}", "\\endgroup", "\\par\\vspace{0.9em}"])
        tables.append("\n".join(body))
    return tables


def output_tables(outputs: list[dict]) -> list[str]:
    tables: list[str] = []
    for output in outputs:
        if output.get("output_type") not in {"execute_result", "display_data"}:
            continue
        data = output.get("data", {})
        html = data.get("text/html")
        if isinstance(html, list):
            html = "".join(html)
        if isinstance(html, str) and "<table" in html:
            tables.extend(html_tables_to_latex(html))
    return tables


def truncate_display(text: str, width: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if display_width(text) <= width:
        return text
    ellipsis = "..."
    result = ""
    for char in text:
        if display_width(result + char + ellipsis) > width:
            break
        result += char
    return result.rstrip() + ellipsis


def html_tables_to_plain_text(html: str, width: int = 78) -> list[str]:
    parser = PandasTableParser()
    parser.feed(html)
    rendered: list[str] = []
    for table in parser.tables:
        headers = table["headers"]
        rows = table["rows"]
        if not headers or not rows:
            continue
        col_count = max(len(headers), *(len(row) for row in rows))
        headers = (headers + [""] * col_count)[:col_count]
        rows = [(row + [""] * col_count)[:col_count] for row in rows[:12]]
        max_cell = max(8, min(30, (width - max(1, col_count - 1) * 2) // col_count))
        table_rows = [headers] + rows
        truncated = [[truncate_display(cell, max_cell) for cell in row] for row in table_rows]
        col_widths = [
            min(max(display_width(row[idx]) for row in truncated), max_cell)
            for idx in range(col_count)
        ]

        def pad(cell: str, size: int) -> str:
            return cell + " " * max(0, size - display_width(cell))

        lines = []
        for row_idx, row in enumerate(truncated):
            lines.append("  ".join(pad(cell, col_widths[i]) for i, cell in enumerate(row)).rstrip())
            if row_idx == 0:
                lines.append("  ".join("-" * width for width in col_widths).rstrip())
        if len(table["rows"]) > len(rows):
            lines.append("...")
        rendered.append("\n".join(lines))
    return rendered


def display_width(text: str) -> int:
    return sum(1 if ord(char) < 128 else 2 for char in text)


def fit_listing_text(text: str, width: int = 78) -> str:
    fitted: list[str] = []
    for line in text.splitlines():
        fitted.append(truncate_display(line, width) if display_width(line) > width else line)
    return "\n".join(fitted)


def wrap_listing_text(text: str, width: int = 58) -> str:
    wrapped: list[str] = []
    for line in text.splitlines():
        if display_width(line) <= width:
            wrapped.append(line)
            continue
        indent = re.match(r"^\s*", line).group(0)
        wrapped.extend(
            textwrap.wrap(
                line,
                width=width,
                initial_indent="",
                subsequent_indent=indent + "  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [line]
        )
    return "\n".join(wrapped)


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ""
    escaped = False
    for char in text:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def split_string_content(content: str, width: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for word in re.split(r"(\s+)", content):
        if not word:
            continue
        candidate = current + word
        if current and display_width(candidate) > width:
            chunks.append(current)
            current = word.lstrip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_long_print(line: str, max_width: int = 58) -> list[str] | None:
    if display_width(line) <= max_width:
        return None
    indent = re.match(r"^\s*", line).group(0)
    stripped = line.strip()
    if not (stripped.startswith("print(") and stripped.endswith(")")):
        return None
    arg = stripped[len("print(") : -1]
    child = indent + "    "
    if arg.startswith('f"') and arg.endswith('"') and "{" in arg and "}" in arg:
        last_open = arg.rfind("{")
        if last_open > 2:
            literal = arg[2:last_open]
            expr = arg[last_open:-1]
            return [
                indent + "print(",
                child + f'f"{literal}"',
                child + f'f"{expr}"',
                indent + ")",
            ]
    string_match = re.fullmatch(r"([fFrRbBuU]*)\"(.*)\"", arg)
    if string_match and "{" not in arg and "}" not in arg:
        prefix, content = string_match.groups()
        chunks = split_string_content(content, max_width - display_width(child) - display_width(prefix) - 2)
        if len(chunks) > 1:
            return [indent + "print("] + [child + f'{prefix}"{chunk}"' for chunk in chunks] + [indent + ")"]
    return [indent + "print(", child + arg, indent + ")"]


def split_long_call(line: str, max_width: int = 58) -> list[str] | None:
    if display_width(line) <= max_width:
        return None
    indent = re.match(r"^\s*", line).group(0)
    stripped = line.strip()
    if stripped.startswith(("print(", "#")):
        return None
    if "(" not in stripped or "," not in stripped or stripped.endswith("\\"):
        return None
    open_idx = stripped.find("(")
    if not stripped.endswith(")"):
        return None
    head = stripped[: open_idx + 1]
    args = stripped[open_idx + 1 : -1]
    parts = split_top_level_commas(args)
    if len(parts) < 2:
        return None
    return [indent + head] + [indent + "    " + part + "," for part in parts] + [indent + ")"]


def split_trailing_comment(line: str, max_width: int = 58) -> list[str] | None:
    if display_width(line) <= max_width or "  # " not in line:
        return None
    code, comment = line.split("  # ", 1)
    indent = re.match(r"^\s*", line).group(0)
    return [code.rstrip(), indent + "# " + comment.strip()]


def split_long_comment(line: str, max_width: int = 58) -> list[str] | None:
    stripped = line.lstrip()
    if display_width(line) <= max_width or not stripped.startswith("#"):
        return None
    indent = line[: len(line) - len(stripped)]
    content = stripped[1:].strip()
    chunks = split_string_content(content, max_width - display_width(indent) - 2)
    if len(chunks) <= 1:
        return None
    return [indent + "# " + chunk for chunk in chunks]


def contains_hangul(text: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in text)


def strip_hangul_comments(source: str) -> str:
    """Hide Korean comments in book code listings while preserving notebook code."""
    lines = source.splitlines()
    if not lines:
        return source

    edited = lines[:]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT or not contains_hangul(tok.string):
                continue
            row, col = tok.start
            end_row, end_col = tok.end
            if row != end_row:
                continue
            line = edited[row - 1]
            before = line[:col].rstrip()
            after = line[end_col:]
            edited[row - 1] = (before + after).rstrip()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        stripped_lines: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#") and contains_hangul(stripped):
                stripped_lines.append("")
            elif "  # " in line:
                code, comment = line.split("  # ", 1)
                stripped_lines.append(code.rstrip() if contains_hangul(comment) else line)
            else:
                stripped_lines.append(line)
        edited = stripped_lines

    cleaned: list[str] = []
    blank_pending = False
    for line in edited:
        if line.strip():
            if blank_pending and cleaned:
                cleaned.append("")
            cleaned.append(line.rstrip())
            blank_pending = False
        else:
            blank_pending = True
    return "\n".join(cleaned)


def format_code_for_book(source: str) -> str:
    source = strip_hangul_comments(source)
    formatted: list[str] = []
    for line in source.splitlines():
        split = (
            split_long_print(line)
            or split_long_call(line)
            or split_trailing_comment(line)
            or split_long_comment(line)
        )
        if split:
            formatted.extend(split)
        else:
            formatted.append(line)
    return "\n".join(formatted)


def output_interpretation(source: str, output: str) -> str:
    source_lower = source.lower()
    output_lower = output.lower()
    if "warning" in output_lower or "traceback" in output_lower:
        return "이 출력은 실행 환경이나 입력 형식과 관련된 경고·오류를 보여주므로, 본문에서 의도한 확인 지점인지 구분해 읽어야 합니다."
    if "shape" in source_lower or "shape" in output_lower:
        return "출력된 shape는 데이터가 코드에서 기대한 차원으로 변환되었는지 확인하는 점검 지점입니다."
    if "accuracy" in source_lower or "accuracy" in output_lower:
        return "accuracy 값은 현재 설정에서 모델이 평가 데이터의 라벨을 어느 정도 맞히는지 보여줍니다."
    if "mse" in source_lower or "mae" in source_lower or "r²" in source_lower or "r2" in source_lower:
        return "회귀 지표 출력은 예측 오차의 크기와 모델 설명력을 함께 확인하기 위한 요약입니다."
    if "predict_proba" in source_lower or "proba" in source_lower or "확률" in output:
        return "확률 출력은 각 클래스 또는 라벨에 대해 모델이 어느 정도 자신감을 갖는지 보여줍니다."
    if "classification_report" in source_lower:
        return "classification report는 precision, recall, F1을 클래스별로 나누어 보여주므로 정확도 하나로 가려지는 오류 패턴을 확인할 수 있습니다."
    if "confusion_matrix" in source_lower or "confusion matrix" in output_lower:
        return "혼동 행렬은 어떤 정답 클래스가 어떤 예측 클래스로 잘못 이동했는지 보여주는 오류 지도입니다."
    if "value_counts" in source_lower or "분포" in output:
        return "분포 출력은 학습 데이터가 특정 라벨에 치우쳐 있는지 확인하기 위한 기본 점검입니다."
    if "token" in source_lower or "vocab" in source_lower or "어휘" in output:
        return "토큰과 어휘 출력은 텍스트가 모델 입력 단위로 어떻게 분해되는지 확인하게 해줍니다."
    return "이 출력은 앞 코드가 만든 중간 결과를 확인해 다음 단계의 입력이 올바르게 준비되었는지 점검합니다."


def output_to_latex(source: str, outputs: list[dict]) -> str:
    tables = output_tables(outputs) if RENDER_DATAFRAME_TABLES else []
    if tables:
        return "\n\n".join(tables)
    text = output_text(outputs)
    if not text:
        return ""
    return (
        "\\noindent\\textbf{출력.}\n"
        "\\begin{lstlisting}[style=bookoutput]\n"
        + text
        + "\n\\end{lstlisting}\n"
        "\\par\\vspace{0.9em}"
    )


def code_to_latex(source: str, include_notes: bool = False, outputs: list[dict] | None = None) -> str:
    source = sanitize_symbols(source)
    source = polish_code_comments(source)
    source = source.rstrip()
    if not source:
        return ""
    display_source = format_code_for_book(source)
    display_line_count = len(display_source.splitlines())
    needspace = max(8, min(display_line_count + 4, 26))
    listing = (
        f"\\Needspace{{{needspace}\\baselineskip}}\n"
        "\\begin{lstlisting}\n"
        + display_source
        + "\n\\end{lstlisting}"
    )
    parts = [listing]
    if include_notes:
        notes = code_walkthrough(display_source)
        if notes:
            parts.append(notes)
    if outputs:
        output_latex = output_to_latex(source, outputs)
        if output_latex:
            parts.append(output_latex)
    return "\n\n".join(parts)


def execute_notebook(path: Path) -> dict:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["source"] = polish_code_comments(cell.get("source", ""))
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    return nb


def chapter_tex(chapter: Chapter, execute: bool = False) -> str:
    if execute:
        nb = execute_notebook(chapter.notebook)
    else:
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
        + latex_escape_prose(chapter.focus)
        + "}",
        "",
    ]

    chapter_index_terms = tuple(dict.fromkeys(chapter.indexes + EXTRA_INDEXES.get(chapter.number, ())))
    for term in chapter_index_terms:
        safe = latex_escape_prose(term)
        chunks.append(f"\\index{{{index_sort_key(term)}@{safe}}}")
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
            chunks.append(code_to_latex(source, include_notes=explain_code, outputs=cell.get("outputs", [])))

        chunks.append("")

    chapter_latex = "\n\n".join(chunks).rstrip() + "\n"
    chapter_latex = wrap_tabular_tables(chapter_latex, chapter.number)
    chapter_latex = display_math_to_numbered_equations(chapter_latex, chapter.number)
    return chapter_latex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute notebooks in memory and include saved outputs in the generated LaTeX",
    )
    parser.add_argument(
        "--chapters",
        nargs="*",
        type=int,
        help="chapter numbers to regenerate; defaults to every configured chapter",
    )
    args = parser.parse_args()

    selected = set(args.chapters or [chapter.number for chapter in CHAPTERS])
    CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
    for chapter in CHAPTERS:
        if chapter.number not in selected:
            continue
        if not chapter.notebook.exists():
            raise FileNotFoundError(chapter.notebook)
        out = CHAPTER_DIR / chapter.tex_name
        out.write_text(chapter_tex(chapter, execute=args.execute), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
