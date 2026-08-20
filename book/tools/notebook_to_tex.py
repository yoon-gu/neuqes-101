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
import base64
import io
import textwrap
import tokenize
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "book"
CHAPTER_DIR = BOOK / "chapters"
FIGURE_DIR = BOOK / "assets" / "figures"
GITHUB_RAW = "https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master"
RENDER_DATAFRAME_TABLES = False
COMPACT_CODE_MODE = False

EXECUTED_EXTRA_NOTEBOOKS = {
    (31, "appendix_qwen_grpo_hpo.ipynb"): "31_grpo_appendix.ipynb",
}


@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    title: str
    short_title: str
    focus: str
    indexes: tuple[str, ...]
    extra_notebooks: tuple[str, ...] = ()

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

    def extra_colab_url(self, filename: str) -> str:
        rel = f"{self.number:02d}_{self.slug}/{filename}"
        return f"{GITHUB_RAW}/{rel}"


@dataclass(frozen=True)
class FigureSpec:
    filename: str
    caption: str
    label: str
    width: str = r"0.86\linewidth"


@dataclass(frozen=True)
class CompactAppendixSpec:
    title: str
    summary: str
    figure: FigureSpec | None = None


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
            "항목 라벨",
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
            "data scaling",
            "sample size",
            "learning curve",
            "sklearn baseline",
            "데이터 스케일링",
            "학습 데이터 규모",
            "학습 곡선",
            "스케일링 곡선",
        ),
        ("12_bert_multiclass_data_scaling.ipynb",),
    ),
    Chapter(
        13,
        "bert_multilabel",
        "BERT 다중 라벨 분류 (Multi-label \\& Per-label BCE)",
        "BERT 다중 라벨 분류 (Multi-label \\& Per-label BCE)",
        "Yelp 항목 합성 라벨을 BERT의 라벨별 sigmoid와 BCE로 학습",
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
            "항목 라벨",
            "해밍 손실",
        ),
    ),
    Chapter(
        14,
        "auxiliary_loss",
        "보조 손실과 멀티태스크 학습 (Auxiliary Loss)",
        "보조 손실과 멀티태스크 학습 (Auxiliary Loss)",
        "다중 라벨 항목 분류에 별점 회귀 보조 헤드를 더해 결합 손실을 학습",
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
            "lambda sweep",
            "sweet spot",
            "lambda=0.05",
            "람다 스윕",
        ),
        ("14_auxiliary_loss_lambda_sweep.ipynb",),
    ),
    Chapter(
        15,
        "ko_binary",
        "한국어 BERT 이진 분류 (Korean Binary Classification)",
        "한국어 BERT 이진 분류 (Korean Binary Classification)",
        "Ch 11의 softmax 이진 분류 셋업을 NSMC와 klue/bert-base로 재현",
        (
            "Korean BERT",
            "KLUE-BERT",
            "klue/bert-base",
            "NSMC",
            "Korean WordPiece",
            "CrossEntropyLoss",
            "single_label_classification",
            "binary classification",
            "한국어 BERT",
            "한국어 WordPiece",
            "네이버 영화 리뷰",
            "한국어 이진 분류",
            "감성 분류",
            "샘플 단위 해석",
        ),
    ),
    Chapter(
        16,
        "ko_multiclass",
        "한국어 BERT 다중 클래스 분류 (Korean Multi-class Classification)",
        "한국어 BERT 다중 클래스 분류 (Korean Multi-class Classification)",
        "KLUE-YNAT 7분류로 한국어 BERT softmax 헤드를 K=7까지 확장",
        (
            "KLUE-YNAT",
            "Korean BERT",
            "KLUE-BERT",
            "klue/bert-base",
            "multi-class classification",
            "CrossEntropyLoss",
            "num_labels=7",
            "confusion_matrix",
            "top-1 probability",
            "macro F1",
            "한국어 BERT",
            "한국어 다중 클래스 분류",
            "뉴스 분류",
            "혼동 행렬",
            "최상위 확률",
            "매크로 F1",
        ),
    ),
    Chapter(
        17,
        "ko_multilabel",
        "한국어 BERT 다중 라벨 분류 (Korean Multi-label Classification)",
        "한국어 BERT 다중 라벨 분류 (Korean Multi-label Classification)",
        "KLUE-YNAT 결합 샘플로 한국어 BERT의 라벨별 sigmoid와 BCE를 학습",
        (
            "Korean BERT",
            "KLUE-BERT",
            "klue/bert-base",
            "KLUE-YNAT",
            "multi-label classification",
            "BCEWithLogitsLoss",
            "multi-hot label",
            "per-label sigmoid",
            "hamming_loss",
            "micro F1",
            "macro F1",
            "threshold sweep",
            "co-occurrence matrix",
            "한국어 BERT",
            "한국어 다중 라벨 분류",
            "멀티핫",
            "라벨별 시그모이드",
            "라벨별 BCE",
            "임계값 탐색",
            "공동 활성",
            "뉴스 다중 라벨",
        ),
    ),
    Chapter(
        18,
        "ko_auxiliary",
        "한국어 BERT 보조 손실 (Korean Auxiliary Loss)",
        "한국어 BERT 보조 손실 (Korean Auxiliary Loss)",
        "한국어 다중 라벨 분류에 활성 라벨 개수 회귀 보조 헤드를 더해 결합 손실을 학습",
        (
            "Korean BERT",
            "KLUE-BERT",
            "klue/bert-base",
            "Auxiliary loss",
            "multi-task learning",
            "combined loss",
            "BCEWithLogitsLoss",
            "MSELoss",
            "lambda",
            "AutoModel",
            "custom Trainer",
            "compute_loss",
            "custom data collator",
            "auxiliary head",
            "active label count",
            "한국어 BERT",
            "보조 손실",
            "멀티태스크 학습",
            "결합 손실",
            "보조 헤드",
            "활성 라벨 수",
            "커스텀 Trainer",
            "커스텀 데이터 콜레이터",
            "lambda sweep",
            "sweet spot",
            "lambda=0.05",
            "람다 스윕",
            "약한 보조 신호",
        ),
        ("18_ko_auxiliary_lambda_sweep.ipynb",),
    ),
    Chapter(
        19,
        "tokenizer_training",
        "토크나이저 직접 학습 (Tokenizer Training)",
        "토크나이저 직접 학습 (Tokenizer Training)",
        "WordPiece와 WordLevel을 영어·한국어 코퍼스에서 직접 학습해 비교",
        (
            "Tokenizer training",
            "WordPiece",
            "WordLevel",
            "tokenizers",
            "PreTrainedTokenizerFast",
            "vocab_size",
            "UNK token",
            "BertPreTokenizer",
            "Whitespace",
            "TemplateProcessing",
            "토크나이저 직접 학습",
            "워드피스",
            "워드레벨",
            "어휘 수",
            "미등록 토큰",
            "교차 언어 적용",
            "토큰 길이 분포",
        ),
    ),
    Chapter(
        20,
        "en_bert_pretrain",
        "작은 BERT 사전학습 (English MLM Pretraining)",
        "작은 BERT 사전학습 (English MLM)",
        "영어 일반 도메인 위키 코퍼스로 작은 BERT를 MLM 방식으로 직접 사전학습",
        (
            "BERT pretraining",
            "Masked Language Modeling",
            "MLM",
            "BertConfig",
            "BertForMaskedLM",
            "DataCollatorForLanguageModeling",
            "group_texts",
            "perplexity",
            "random initialization",
            "Wikitext-103",
            "bert-base-uncased",
            "작은 BERT",
            "사전학습",
            "마스크드 언어 모델링",
            "마스크 토큰",
            "퍼플렉서티",
            "일반 도메인",
            "사전학습량 곡선",
            "perplexity curve",
            "scaling curve",
            "epoch saturation",
            "데이터 병목",
        ),
        (
            "20_en_bert_pretrain_scaling.ipynb",
        ),
    ),
    Chapter(
        21,
        "en_bert_classify",
        "작은 BERT 이진 분류 (English Yelp Fine-tuning)",
        "작은 BERT 이진 분류 (Yelp)",
        "20장에서 사전학습한 작은 BERT 본체를 Yelp 이진 분류로 파인튜닝",
        (
            "BERT fine-tuning",
            "BertForSequenceClassification",
            "Yelp polarity",
            "binary classification",
            "CrossEntropyLoss",
            "classification head",
            "transfer learning",
            "DistilBERT comparison",
            "confusion_matrix",
            "classification_report",
            "roc_auc_score",
            "이진 분류",
            "파인튜닝",
            "전이 학습",
            "분류 헤드",
            "혼동 행렬",
            "전이 성능",
        ),
    ),
    Chapter(
        22,
        "ko_bert_pretrain",
        "작은 BERT 사전학습 (Korean MLM Pretraining)",
        "작은 BERT 사전학습 (Korean MLM)",
        "한국어 일반 도메인 위키 코퍼스로 작은 BERT를 MLM 방식으로 직접 사전학습",
        (
            "Korean BERT pretraining",
            "Korean Masked Language Modeling",
            "MLM",
            "BertConfig",
            "BertForMaskedLM",
            "DataCollatorForLanguageModeling",
            "klue/bert-base",
            "wikimedia/wikipedia",
            "Korean Wikipedia",
            "perplexity",
            "random initialization",
            "한국어 작은 BERT",
            "한국어 사전학습",
            "마스크드 언어 모델링",
            "한국어 위키백과",
            "한국어 WordPiece",
            "퍼플렉서티",
        ),
    ),
    Chapter(
        23,
        "ko_bert_classify",
        "작은 BERT 이진 분류 (Korean NSMC Fine-tuning)",
        "작은 BERT 이진 분류 (NSMC)",
        "22장에서 사전학습한 한국어 작은 BERT 본체를 NSMC 이진 분류로 파인튜닝",
        (
            "Korean BERT fine-tuning",
            "BertForSequenceClassification",
            "NSMC",
            "binary classification",
            "CrossEntropyLoss",
            "classification head",
            "transfer learning",
            "KLUE-BERT comparison",
            "confusion_matrix",
            "classification_report",
            "roc_auc_score",
            "한국어 이진 분류",
            "NSMC",
            "파인튜닝",
            "전이 학습",
            "분류 헤드",
            "혼동 행렬",
        ),
    ),
    Chapter(
        24,
        "gpt_tinystories",
        "작은 GPT 사전학습 (TinyStories Causal LM)",
        "작은 GPT 사전학습 (TinyStories CLM)",
        "작은 GPT2를 random init으로 만들고 직접 학습한 BPE 토크나이저와 TinyStories로 next-token 사전학습",
        (
            "GPT",
            "decoder-only",
            "causal language modeling",
            "CausalLM",
            "next-token prediction",
            "CrossEntropyLoss",
            "GPT2Config",
            "GPT2LMHeadModel",
            "BPE",
            "ByteLevel",
            "TinyStories",
            "DataCollatorForLanguageModeling",
            "labels=-100",
            "generate",
            "temperature",
            "top_k",
            "top_p",
            "작은 GPT",
            "디코더",
            "인과 언어 모델링",
            "다음 토큰 예측",
            "BPE 토크나이저",
            "직접 사전학습",
            "생성",
            "샘플링",
        ),
    ),
    Chapter(
        25,
        "gpt2_continual_pretrain",
        "GPT-2 계속 사전학습 (Continual Pretraining)",
        "GPT-2 계속 사전학습 (Continual Pretraining)",
        "OpenAI gpt2를 같은 TinyStories 데이터로 continual pretraining하며 scratch pretraining과 비교",
        (
            "GPT-2",
            "continual pretraining",
            "continual learning",
            "AutoModelForCausalLM",
            "AutoTokenizer",
            "gpt2",
            "WebText",
            "catastrophic forgetting",
            "learning rate",
            "gradient accumulation",
            "CausalLM",
            "DataCollatorForLanguageModeling",
            "labels=-100",
            "TinyStories",
            "generation comparison",
            "계속 사전학습",
            "연속 학습",
            "파국적 망각",
            "학습률",
            "그래디언트 누적",
            "생성 비교",
            "사전학습 본체",
        ),
    ),
    Chapter(
        26,
        "ko_tiny_gpt",
        "한국어 작은 GPT 사전학습 (TinyStories-Korean Causal LM)",
        "한국어 작은 GPT 사전학습 (TinyStories-Korean CLM)",
        "작은 GPT2를 한국어 BBPE 토크나이저와 TinyStories-Korean으로 from-scratch next-token 사전학습",
        (
            "Korean GPT",
            "TinyStories-Korean",
            "g0ster/TinyStories-Korean",
            "byte-level BPE",
            "BBPE",
            "Korean tokenizer",
            "CausalLM",
            "CrossEntropyLoss",
            "next-token prediction",
            "GPT2Config",
            "GPT2LMHeadModel",
            "DataCollatorForLanguageModeling",
            "labels=-100",
            "group_texts",
            "KoGPT2",
            "skt/kogpt2-base-v2",
            "한국어 GPT",
            "한국어 토크나이저",
            "한국어 BBPE",
            "한국어 TinyStories",
            "직접 사전학습",
            "한글 토큰화",
            "학습 전 생성",
            "학습 후 생성",
            "한국어 생성",
        ),
    ),
    Chapter(
        27,
        "ko_gpt2_continual_pretrain",
        "KoGPT2 계속 사전학습 (Korean Continual Pretraining)",
        "KoGPT2 계속 사전학습 (Korean CPT)",
        "KoGPT2를 같은 TinyStories-Korean 데이터로 continual pretraining하며 한국어 scratch GPT와 비교",
        (
            "KoGPT2",
            "skt/kogpt2-base-v2",
            "continual pretraining",
            "continual learning",
            "Korean continual pretraining",
            "AutoModelForCausalLM",
            "PreTrainedTokenizerFast",
            "DataCollatorForLanguageModeling",
            "CausalLM",
            "CrossEntropyLoss",
            "labels=-100",
            "learning_rate=2e-5",
            "gradient_accumulation_steps",
            "catastrophic forgetting",
            "TinyStories-Korean",
            "g0ster/TinyStories-Korean",
            "KoGPT2 tokenizer",
            "encode-decode round trip",
            "한국어 continual pretraining",
            "계속 사전학습",
            "한국어 사전학습 본체",
            "KoGPT2 토크나이저",
            "토크나이저 왕복 검증",
            "파국적 망각",
            "학습률",
            "그래디언트 누적",
            "한국어 생성 비교",
        ),
    ),
    Chapter(
        28,
        "sft",
        "KoGPT2 SFT (Instruction Tuning)",
        "KoGPT2 SFT (Instruction Tuning)",
        "KoGPT2를 KoAlpaca instruction-response 데이터로 SFT해 지시를 따르는 행동 정렬을 확인",
        (
            "KoGPT2",
            "SFT",
            "Instruction Tuning",
            "Supervised Fine-Tuning",
            "behavior alignment",
            "KoAlpaca",
            "beomi/KoAlpaca-v1.1a",
            "trl",
            "SFTTrainer",
            "SFTConfig",
            "completion_only_loss=True",
            "completion_mask",
            "labels=-100",
            "response-only loss",
            "prompt masking",
            "PreTrainedTokenizerFast",
            "AutoModelForCausalLM",
            "CrossEntropyLoss",
            "지시 튜닝",
            "지도 미세조정",
            "행동 정렬",
            "응답 구간 손실",
            "프롬프트 마스킹",
            "KoAlpaca",
            "답변 토큰",
            "instruction following",
        ),
    ),
    Chapter(
        29,
        "benchmark_eval",
        "분야별 벤치마크 평가 (Benchmark Evaluation)",
        "벤치마크 평가 (Benchmark Evaluation)",
        "생성형 LLM 평가가 분류 평가와 다른 이유를 task format, log-likelihood, few-shot, LLM-as-judge 관점에서 정리",
        (
            "Benchmark Evaluation",
            "LLM evaluation",
            "KoBEST",
            "Qwen2.5-0.5B-Instruct",
            "multiple choice",
            "log-likelihood",
            "acc_norm",
            "exact match",
            "generation evaluation",
            "answer extraction",
            "few-shot",
            "zero-shot",
            "in-context learning",
            "lm-evaluation-harness",
            "lm-eval",
            "LLM-as-judge",
            "human evaluation",
            "Goodhart's law",
            "contamination",
            "leaderboard",
            "벤치마크 평가",
            "생성형 평가",
            "객관식 평가",
            "로그우도",
            "길이 정규화",
            "정답 추출",
            "퓨샷",
            "제로샷",
            "문맥 내 학습",
            "LLM 심판",
            "사람 평가",
            "벤치마크 오염",
            "리더보드",
        ),
        ("appendix_eval_landscape.ipynb",),
    ),
    Chapter(
        30,
        "dpo",
        "DPO: 사람 선호로 정렬 (Preference Alignment)",
        "DPO 정렬 (Preference Alignment)",
        "reward model 없이 preference 쌍으로 policy를 직접 정렬하는 DPO 학습 단계를 정리",
        (
            "DPO",
            "Direct Preference Optimization",
            "preference alignment",
            "alignment",
            "chosen",
            "rejected",
            "reward margin",
            "implicit reward",
            "reference model",
            "frozen reference",
            "KL constraint",
            "DPOTrainer",
            "DPOConfig",
            "beta",
            "RLAIF",
            "UltraFeedback",
            "maywell/ko_Ultrafeedback_binarized",
            "policy model",
            "PPO",
            "reward model",
            "labels=-100",
            "response-only",
            "사람 선호",
            "선호 정렬",
            "정렬 학습",
            "선택 응답",
            "거절 응답",
            "보상 마진",
            "참조 모델",
            "고정 참조",
            "KL 제약",
            "응답 구간 손실",
        ),
    ),
    Chapter(
        31,
        "grpo",
        "GRPO: 검증 가능한 보상으로 정렬 (Verifiable Reward)",
        "GRPO 정렬 (Verifiable Reward)",
        "preference 쌍 대신 verifier가 자동 채점하는 reward로 policy를 정렬하는 GRPO 학습 단계를 정리",
        (
            "GRPO",
            "Group Relative Policy Optimization",
            "verifiable reward",
            "verifier",
            "group relative advantage",
            "rollout group",
            "reward std",
            "critic-free RL",
            "PPO",
            "DPO",
            "DeepSeek-R1",
            "GRPOTrainer",
            "GRPOConfig",
            "num_generations",
            "format reward",
            "correctness reward",
            "Qwen2.5-0.5B-Instruct",
            "fp16 AMP",
            "fp32 load",
            "dtype",
            "SFT",
            "alignment",
            "검증 가능한 보상",
            "검증기",
            "그룹 상대 어드밴티지",
            "롤아웃 그룹",
            "보상 표준편차",
            "critic 없는 강화학습",
            "형식 보상",
            "정답 보상",
            "능력 증폭",
        ),
        ("appendix_qwen_grpo_hpo.ipynb",),
    ),
    Chapter(
        32,
        "diffusion_intro",
        "작은 Diffusion LM 입문: 병렬 Denoise 생성",
        "작은 Diffusion LM 입문",
        "작은 BERT-style masked LM을 from scratch로 학습하고 전부 [MASK]인 캔버스에서 병렬 denoise로 영어 동화를 생성",
        (
            "diffusion LM",
            "masked diffusion",
            "absorbing mask diffusion",
            "parallel denoising",
            "BertForMaskedLM",
            "ByteLevel BPE",
            "DiffusionCollator",
            "confidence remasking",
            "time weighted CE",
            "병렬 denoise",
            "마스크 diffusion",
            "가변 마스킹",
            "빈 캔버스 생성",
            "시간 가중 손실",
        ),
    ),
    Chapter(
        33,
        "diffusion_train",
        "Diffusion LM 샘플러 교정: 반복 억제 생성",
        "Diffusion LM 샘플러 교정",
        "같은 작은 diffusion LM에서 생성 샘플러만 바꾸어 반복을 줄이는 디코딩 실험",
        (
            "diffusion sampler",
            "carry-over sampler",
            "semi-autoregressive",
            "repeat penalty",
            "n-gram repeat",
            "confidence threshold",
            "샘플러 교정",
            "반복 억제",
            "부분 자기회귀",
            "신뢰도 임계값",
            "n-gram 반복률",
        ),
    ),
    Chapter(
        34,
        "ko_diffusion",
        "한국어 Diffusion LM: 80/10/10 마스킹",
        "한국어 Diffusion LM",
        "한국어 TinyStories에서 diffusion LM을 학습하며 naive 마스킹 붕괴와 BERT식 80/10/10 마스킹을 비교",
        (
            "Korean Diffusion LM",
            "BERT 80/10/10 masking",
            "ByteLevel BPE",
            "TinyStories-Korean",
            "unigram collapse",
            "plain CE",
            "fixed-t accuracy",
            "한국어 diffusion",
            "한국어 BBPE",
            "BERT식 마스킹",
            "유니그램 붕괴",
            "복원 정확도",
        ),
        ("34_ko_diffusion_appendix.ipynb",),
    ),
]


FIGURE_OUTPUTS: dict[tuple[int, int], FigureSpec] = {
    (1, 1): FigureSpec("ch01_star_distribution.png", "Yelp 5,000건 샘플의 별점 분포", "fig:ch01-star-distribution", r"0.72\linewidth"),
    (2, 1): FigureSpec("ch02_prediction_distribution.png", "회귀 예측값과 실제 별점 분포", "fig:ch02-prediction-distribution", r"0.76\linewidth"),
    (9, 1): FigureSpec("ch09_predicted_violin.png", "정답 별점별 BERT와 sklearn 회귀 예측 분포", "fig:ch09-predicted-violin", r"0.92\linewidth"),
    (9, 2): FigureSpec("ch09_residual_violin.png", "정답 별점별 잔차 분포 비교", "fig:ch09-residual-violin", r"0.92\linewidth"),
    (10, 1): FigureSpec("ch10_probability_kde.png", "방식 A의 sigmoid 확률 분포", "fig:ch10-probability-kde", r"0.88\linewidth"),
    (10, 2): FigureSpec("ch10_logit_kde.png", "방식 A의 sigmoid 이전 logit 분포", "fig:ch10-logit-kde", r"0.88\linewidth"),
    (11, 1): FigureSpec("ch11_probability_kde.png", "방식 B의 softmax 확률 분포", "fig:ch11-probability-kde", r"0.88\linewidth"),
    (11, 2): FigureSpec("ch11_logit_kde.png", "방식 B를 1차원 logit 차이로 환산한 분포", "fig:ch11-logit-kde", r"0.88\linewidth"),
    (11, 3): FigureSpec("ch11_probability_scatter.png", "방식 A와 방식 B의 샘플별 확률 일치도", "fig:ch11-probability-scatter", r"0.72\linewidth"),
    (12, 1): FigureSpec("ch12_confusion_matrix.png", "5클래스 Yelp 분류의 혼동 행렬", "fig:ch12-confusion-matrix", r"0.76\linewidth"),
    (12, 2): FigureSpec("ch12_top1_probability.png", "정답 여부에 따른 최상위 예측 확률 분포", "fig:ch12-top1-probability", r"0.88\linewidth"),
    (12, 3): FigureSpec("ch12_confusion_compare.png", "sklearn TF-IDF와 BERT의 혼동 행렬 비교", "fig:ch12-confusion-compare", r"0.96\linewidth"),
    (12, 4): FigureSpec("ch12_data_scaling.png", "학습 데이터 규모에 따른 sklearn과 BERT 정확도", "fig:ch12-data-scaling", r"0.86\linewidth"),
    (13, 1): FigureSpec("ch13_label_probability_facets.png", "라벨별 sigmoid 확률 분포", "fig:ch13-label-probability-facets", r"0.94\linewidth"),
    (13, 2): FigureSpec("ch13_cooccurrence.png", "정답 라벨과 예측 라벨의 공동 활성 패턴", "fig:ch13-cooccurrence", r"0.96\linewidth"),
    (13, 3): FigureSpec("ch13_f1_compare.png", "라벨별 F1: sklearn OvR와 BERT 비교", "fig:ch13-f1-compare", r"0.88\linewidth"),
    (14, 1): FigureSpec("ch14_aux_f1_compare.png", "라벨별 F1: 보조 손실 적용 전후 비교", "fig:ch14-aux-f1-compare", r"0.88\linewidth"),
    (14, 2): FigureSpec("ch14_aux_star_violin.png", "보조 별점 회귀 헤드의 예측 분포", "fig:ch14-aux-star-violin", r"0.82\linewidth"),
    (14, 3): FigureSpec("ch14_lambda_sweep.png", "lambda 스윕에서 확인한 보조 손실 sweet spot", "fig:ch14-lambda-sweep", r"0.86\linewidth"),
    (15, 1): FigureSpec("ch15_probability_kde.png", "NSMC 이진 분류의 positive 확률 분포", "fig:ch15-probability-kde", r"0.88\linewidth"),
    (15, 2): FigureSpec("ch15_logit_kde.png", "NSMC 이진 분류의 logit 차이 분포", "fig:ch15-logit-kde", r"0.88\linewidth"),
    (16, 1): FigureSpec("ch16_confusion_matrix.png", "KLUE-YNAT 7분류 혼동 행렬", "fig:ch16-confusion-matrix", r"0.86\linewidth"),
    (16, 2): FigureSpec("ch16_top1_probability.png", "정답 여부에 따른 KLUE-YNAT 최상위 확률 분포", "fig:ch16-top1-probability", r"0.88\linewidth"),
    (17, 1): FigureSpec("ch17_label_probability_facets.png", "KLUE-YNAT 다중 라벨 카테고리별 sigmoid 확률 분포", "fig:ch17-label-probability-facets", r"0.90\linewidth"),
    (17, 2): FigureSpec("ch17_cooccurrence.png", "KLUE-YNAT 합성 다중 라벨의 공동 활성 행렬", "fig:ch17-cooccurrence", r"0.92\linewidth"),
    (17, 3): FigureSpec("ch17_threshold_sweep.png", "KLUE-YNAT 다중 라벨 임계값 탐색", "fig:ch17-threshold-sweep", r"0.86\linewidth"),
    (18, 1): FigureSpec("ch18_per_label_f1_compare.png", "한국어 보조 손실 적용 전후의 카테고리별 F1 비교", "fig:ch18-per-label-f1-compare", r"0.88\linewidth"),
    (18, 2): FigureSpec("ch18_aux_count_violin.png", "활성 라벨 개수 보조 회귀의 예측 분포", "fig:ch18-aux-count-violin", r"0.76\linewidth"),
    (18, 3): FigureSpec("ch18_lambda_sweep.png", "한국어 보조 손실 lambda 스윕에서 확인한 sweet spot", "fig:ch18-lambda-sweep", r"0.86\linewidth"),
    (19, 1): FigureSpec("ch19_token_length_distribution.png", "WordPiece와 WordLevel의 문장당 토큰 수 분포", "fig:ch19-token-length-distribution", r"0.92\linewidth"),
    (19, 2): FigureSpec("ch19_unk_rate_bar.png", "토크나이저별 미등록 토큰 비율 비교", "fig:ch19-unk-rate-bar", r"0.72\linewidth"),
    (19, 3): FigureSpec("ch19_cross_language_heatmap.png", "학습 언어와 입력 언어가 다를 때의 미등록 토큰 비율", "fig:ch19-cross-language-heatmap", r"0.78\linewidth"),
    (19, 4): FigureSpec("ch19_vocab_sweep.png", "WordPiece 어휘 수에 따른 토큰 길이와 미등록 토큰 비율", "fig:ch19-vocab-sweep", r"0.78\linewidth"),
    (20, 1): FigureSpec("ch20_mlm_training_loss.png", "작은 BERT의 MLM 사전학습 loss 곡선", "fig:ch20-mlm-training-loss", r"0.82\linewidth"),
    (20, 2): FigureSpec("ch20_eval_loss_ppl.png", "사전학습 전후 MLM eval loss와 perplexity 비교", "fig:ch20-eval-loss-ppl", r"0.88\linewidth"),
    (20, 3): FigureSpec("ch20_pretrain_scaling_curve.png", "사전학습 epoch에 따른 영어와 한국어 MLM perplexity 곡선", "fig:ch20-pretrain-scaling-curve", r"0.86\linewidth"),
    (21, 1): FigureSpec("ch21_finetune_loss.png", "작은 BERT의 Yelp 이진 분류 fine-tune loss 곡선", "fig:ch21-finetune-loss", r"0.82\linewidth"),
    (21, 2): FigureSpec("ch21_confusion_matrix.png", "작은 BERT의 Yelp 이진 분류 혼동 행렬", "fig:ch21-confusion-matrix", r"0.68\linewidth"),
    (21, 3): FigureSpec("ch21_ch10_compare.png", "Yelp 이진 분류에서 DistilBERT와 작은 BERT 비교", "fig:ch21-ch10-compare", r"0.82\linewidth"),
    (22, 1): FigureSpec("ch22_mlm_training_loss.png", "한국어 작은 BERT의 MLM 사전학습 loss 곡선", "fig:ch22-mlm-training-loss", r"0.82\linewidth"),
    (22, 2): FigureSpec("ch22_eval_loss_ppl.png", "한국어 MLM 평가 손실과 perplexity 변화", "fig:ch22-eval-loss-ppl", r"0.82\linewidth"),
    (23, 1): FigureSpec("ch23_finetune_loss.png", "한국어 작은 BERT의 NSMC fine-tune loss 곡선", "fig:ch23-finetune-loss", r"0.82\linewidth"),
    (23, 2): FigureSpec("ch23_confusion_matrix.png", "한국어 작은 BERT의 NSMC 혼동 행렬", "fig:ch23-confusion-matrix", r"0.68\linewidth"),
    (23, 3): FigureSpec("ch23_ch15_compare.png", "KLUE-BERT와 한국어 작은 BERT의 NSMC 지표 비교", "fig:ch23-ch15-compare", r"0.82\linewidth"),
    (24, 1): FigureSpec("ch24_loss_vram_trace.png", "작은 GPT 사전학습 손실과 VRAM 추적", "fig:ch24-loss-vram-trace", r"0.88\linewidth"),
    (25, 1): FigureSpec("ch25_loss_vram_trace.png", "GPT-2 continual pretraining 손실과 VRAM 추적", "fig:ch25-loss-vram-trace", r"0.88\linewidth"),
    (26, 1): FigureSpec("ch26_loss_vram_trace.png", "한국어 작은 GPT의 TinyStories-Korean 학습 손실과 VRAM 추적", "fig:ch26-loss-vram-trace", r"0.88\linewidth"),
    (27, 1): FigureSpec("ch27_loss_vram_trace.png", "KoGPT2 continual pretraining 손실과 VRAM 추적", "fig:ch27-loss-vram-trace", r"0.88\linewidth"),
    (28, 1): FigureSpec("ch28_sft_masking_bar.png", "SFT labels 마스킹: prompt는 제외하고 답변만 학습", "fig:ch28-sft-masking", r"0.86\linewidth"),
    (28, 2): FigureSpec("ch28_sft_loss_vram_trace.png", "KoGPT2 SFT 학습 곡선과 VRAM 추적", "fig:ch28-sft-loss-vram", r"0.88\linewidth"),
    (30, 1): FigureSpec("ch30_dpo_loss_margin.png", "DPO loss와 preference margin의 관계", "fig:ch30-dpo-loss-margin", r"0.86\linewidth"),
    (30, 2): FigureSpec("ch30_dpo_margin_shift.png", "DPO 전후 reward margin 분포 비교", "fig:ch30-dpo-margin-shift", r"0.86\linewidth"),
    (30, 3): FigureSpec("ch30_dpo_training_curves.png", "KoGPT2 DPO 학습 곡선과 reward 지표", "fig:ch30-dpo-training-curves", r"0.90\linewidth"),
    (31, 1): FigureSpec("ch31_grpo_kogpt2_accuracy.png", "KoGPT2 GRPO 전후 산술 verifier 통과율", "fig:ch31-grpo-kogpt2-accuracy", r"0.72\linewidth"),
    (31, 2): FigureSpec("ch31_grpo_training_curves.png", "KoGPT2 GRPO 학습 중 reward와 VRAM 흐름", "fig:ch31-grpo-kogpt2-curves", r"0.88\linewidth"),
    (31, 3): FigureSpec("ch31_qwen_grpo_reward_curves.png", "Qwen GRPO 부록의 reward 상승과 HPO 요약", "fig:ch31-qwen-grpo-reward-curves", r"0.86\linewidth"),
    (32, 1): FigureSpec("ch32_training_trace.png", "32장 diffusion LM 학습 loss와 VRAM 추적", "fig:ch32-training-trace", r"0.86\linewidth"),
    (34, 1): FigureSpec("ch34_masking_ablation.png", "한국어 diffusion 마스킹 방식 비교", "fig:ch34-masking-ablation", r"0.86\linewidth"),
}


SUPPLEMENTAL_FIGURES: dict[int, tuple[FigureSpec, ...]] = {
    25: (
        FigureSpec("ch25_ch24_loss_compare.png", "TinyStories CLM의 scratch 학습과 continual pretraining 손실 비교", "fig:ch25-ch24-loss-compare", r"0.72\linewidth"),
    ),
    26: (
        FigureSpec("ch26_ch24_loss_compare.png", "Scratch Causal LM의 영어/한국어 학습 손실 비교", "fig:ch26-ch24-loss-compare", r"0.72\linewidth"),
    ),
    27: (
        FigureSpec("ch27_ch26_loss_compare.png", "한국어 TinyStories에서 scratch와 continual pretraining 손실 비교", "fig:ch27-ch26-loss-compare", r"0.72\linewidth"),
        FigureSpec("ch27_appendix_fertility_bar.png", "GPT 계열 14개 토크나이저의 한국어 fertility 비교", "fig:ch27-appendix-fertility", r"0.86\linewidth"),
        FigureSpec("ch27_appendix_vocab_share.png", "토크나이저 vocabulary 안의 한국어 점유율 비교", "fig:ch27-appendix-vocab-share", r"0.86\linewidth"),
        FigureSpec("ch27_appendix_vocab_fertility_scatter.png", "한국어 vocab 점유율과 fertility의 관계", "fig:ch27-appendix-vocab-fertility", r"0.72\linewidth"),
    ),
}


# The compact publisher edition keeps the experiment's conclusion and one
# representative figure in print. The full appendix remains executable through
# its own Colab link, so reducing page count does not remove reproducibility.
COMPACT_APPENDICES: dict[int, CompactAppendixSpec] = {
    12: CompactAppendixSpec(
        "데이터 스케일링: BERT와 sklearn의 교차점",
        "작은 데이터에서는 TF-IDF와 선형 모델이 앞서지만 약 1,000개 샘플 부근에서 BERT가 따라잡습니다. "
        "30,000개에서도 격차가 약 0.04로 남는다는 결과는 5클래스 난이도와 데이터 규모를 함께 해석해야 함을 보여줍니다.",
        FigureSpec(
            "ch12_data_scaling.png",
            "학습 데이터 규모에 따른 sklearn과 BERT 정확도",
            "fig:ch12-data-scaling",
            r"0.72\linewidth",
        ),
    ),
    14: CompactAppendixSpec(
        "보조 손실 가중치 스윕",
        r"$\lambda=0.05$에서 micro-F1이 0.8399에서 0.8469로, macro-F1이 0.8023에서 0.8109로 올랐습니다. "
        r"$\lambda$를 더 키우면 보조 손실이 메인 태스크를 누르므로 한 점의 성능보다 전체 곡선으로 sweet spot을 확인해야 합니다.",
        FigureSpec(
            "ch14_lambda_sweep.png",
            "보조 손실 가중치 스윕에서 확인한 sweet spot",
            "fig:ch14-lambda-sweep",
            r"0.72\linewidth",
        ),
    ),
    18: CompactAppendixSpec(
        "한국어 보조 손실 가중치 스윕",
        r"$\lambda=0.05$에서 micro-F1은 0.8491에서 0.8523으로, macro-F1은 0.8451에서 0.8493으로 올랐습니다. "
        "별점처럼 강한 보조 신호보다 향상 폭은 작지만, 약한 활성 라벨 개수 신호도 적절한 가중치에서는 메인 태스크를 돕습니다.",
        FigureSpec(
            "ch18_lambda_sweep.png",
            "한국어 보조 손실 가중치 스윕에서 확인한 sweet spot",
            "fig:ch18-lambda-sweep",
            r"0.72\linewidth",
        ),
    ),
    20: CompactAppendixSpec(
        "사전학습량과 perplexity",
        "영어 perplexity는 1,173에서 696으로, 한국어는 1,626에서 709로 내려가지만 8--10 epoch 이후에는 개선이 평탄해집니다. "
        "같은 5,000개 텍스트를 반복하는 것보다 데이터와 compute를 늘리는 편이 다음 성능 레버라는 결론입니다.",
        FigureSpec(
            "ch20_pretrain_scaling_curve.png",
            "사전학습 epoch에 따른 영어와 한국어 MLM perplexity",
            "fig:ch20-pretrain-scaling-curve",
            r"0.72\linewidth",
        ),
    ),
    29: CompactAppendixSpec(
        "생성형 LLM 평가 항해 가이드",
        "벤치마크 생태계, 리더보드, 평가 도구, LLM-as-judge와 사람 평가의 편향을 한 흐름으로 정리합니다. "
        "공개 점수는 참고 지표로 쓰고 실제 배포 판단은 use-case 맞춤 평가셋으로 내린다는 원칙이 부록의 결론입니다.",
    ),
    31: CompactAppendixSpec(
        "Qwen GRPO와 HPO 요약",
        "Qwen2.5-0.5B-Instruct와 형식 보상을 사용하면 reward가 실제로 오릅니다. "
        "부록은 sweep 전체를 다시 싣는 대신 최종 채택값, fp32 로드와 fp16 AMP의 dtype 함정, reward 곡선 읽기에 집중합니다.",
        FigureSpec(
            "ch31_qwen_grpo_reward_curves.png",
            "Qwen GRPO의 reward 상승과 HPO 최종 채택값",
            "fig:ch31-qwen-grpo-reward-curves",
            r"0.72\linewidth",
        ),
    ),
    34: CompactAppendixSpec(
        "한국어 diffusion collapse 복구 실험",
        r"100\% 마스크 방식에서 나타나는 유니그램 collapse를 재현하고 BERT식 80/10/10 마스킹으로 복원합니다. "
        "본문에는 두 방식의 핵심 대조만 남기고, 두 번의 경량 학습과 진단 코드는 온라인 부록에서 실행합니다.",
        FigureSpec(
            "ch34_masking_ablation.png",
            "한국어 diffusion 마스킹 방식 비교",
            "fig:ch34-masking-ablation",
            r"0.72\linewidth",
        ),
    ),
}


# Three questions per chapter preserve a theory/practice/application balance.
# Later language variants deliberately keep delta-specific questions instead of
# repeating the same FAQ from their English counterparts.
COMPACT_FAQ_SELECTIONS: dict[int, tuple[int, ...]] = {
    1: (2, 3, 6),
    2: (1, 5, 6),
    3: (3, 4, 5),
    4: (1, 2, 4),
    5: (1, 3, 5),
    6: (1, 2, 6),
    7: (1, 3, 6),
    8: (1, 2, 3),
    9: (1, 2, 3),
    10: (1, 3, 4),
    11: (1, 2, 4),
    12: (1, 4, 5),
    13: (1, 3, 4),
    14: (1, 4, 5),
    15: (1, 4, 6),
    16: (1, 2, 6),
    17: (3, 6, 7),
    18: (1, 3, 6),
    19: (1, 2, 4),
    20: (1, 3, 5),
    21: (2, 3, 4),
    22: (1, 5, 6),
    23: (1, 4, 5),
    24: (1, 3, 5),
    25: (1, 2, 5),
    26: (1, 5, 7),
    27: (2, 5, 6),
    28: (1, 2, 5),
    29: (1, 2, 4),
    30: (1, 3, 4),
    31: (1, 2, 4),
    32: (1, 2, 5),
    33: (1, 3, 5),
    34: (1, 2, 5),
}


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
        "항목 키워드",
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
        "항목 키워드",
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
    15: (
        "load_dataset",
        "NSMC",
        "klue/bert-base",
        "AutoTokenizer",
        "AutoModelForSequenceClassification",
        "num_labels=2",
        "CrossEntropyLoss",
        "softmax",
        "classification_report",
        "roc_auc_score",
        "한국어 WordPiece",
        "감성 분석",
        "네이버 영화 리뷰",
        "확률 KDE",
        "로짓 분포",
    ),
    16: (
        "KLUE-YNAT",
        "load_dataset(\"klue\", \"ynat\")",
        "num_labels=7",
        "problem_type=\"single_label_classification\"",
        "CrossEntropyLoss",
        "roc_auc_score",
        "multi_class=\"ovr\"",
        "confusion_matrix",
        "seaborn.heatmap",
        "top-1 probability",
        "뉴스 분류",
        "7분류",
        "행 정규화 혼동 행렬",
        "캘리브레이션",
    ),
    17: (
        "problem_type=\"multi_label_classification\"",
        "BCEWithLogitsLoss",
        "sigmoid probability",
        "multi-hot vector",
        "precision_recall_fscore_support",
        "classification_report",
        "roc_auc_score",
        "seaborn.FacetGrid",
        "threshold=0.5",
        "pos_weight",
        "synthetic multi-label data",
        "active label count",
        "conditional probability",
        "라벨별 확률",
        "활성 라벨 수",
        "합성 다중 라벨 데이터",
        "카테고리별 임계값",
        "조건부 확률",
        "불균형 가중치",
    ),
    18: (
        "AutoModel.from_pretrained",
        "nn.Module",
        "nn.Linear",
        "SequenceClassifierOutput",
        "Trainer.compute_loss",
        "remove_unused_columns=False",
        "lambda_aux",
        "count_head",
        "MSELoss",
        "Pearson correlation",
        "R2 score",
        "RMSE",
        "lambda sweep",
        "layer-wise learning rate",
        "보조 라벨",
        "활성 개수 회귀",
        "람다 스윕",
        "계층별 학습률",
        "보조 지표",
        "표현 공유",
    ),
    19: (
        "Tokenizer",
        "WordPieceTrainer",
        "WordLevelTrainer",
        "SPECIAL_TOKENS",
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "NFD",
        "Lowercase",
        "StripAccents",
        "NormSequence",
        "WordPieceDecoder",
        "Tokenizer.from_file",
        "tokenizer.save",
        "cross-language tokenization",
        "unknown token rate",
        "vocab sweep",
        "subword token",
        "special token",
        "직접 학습 토크나이저",
        "특수 토큰",
        "서브워드",
        "어절 단위",
        "어휘 스윕",
        "언어별 코퍼스",
        "한국어 코퍼스",
        "영어 코퍼스",
    ),
    20: (
        "AutoTokenizer",
        "AutoModelForMaskedLM",
        "BertForMaskedLM",
        "BertConfig",
        "Trainer",
        "TrainingArguments",
        "mlm_probability",
        "ignore_index",
        "labels=-100",
        "mask token prediction",
        "top-k prediction",
        "save_pretrained",
        "Salesforce/wikitext",
        "Wikitext-103",
        "random baseline",
        "language modeling head",
        "언어 모델링 헤드",
        "마스킹 비율",
        "80/10/10 규칙",
        "무작위 초기화",
        "체크포인트 저장",
        "위키텍스트",
    ),
    21: (
        "BertForSequenceClassification",
        "load_state_dict",
        "classifier head",
        "single_label_classification",
        "softmax",
        "fine-tune loss",
        "Yelp binary classification",
        "GLUE",
        "domain transfer",
        "DAPT",
        "random baseline",
        "head replacement",
        "body weight transfer",
        "본체 가중치",
        "헤드 교체",
        "도메인 전이",
        "일반 도메인 사전학습",
        "태스크 도메인",
        "정확도",
        "AUC",
    ),
    22: (
        "AutoTokenizer",
        "AutoModelForMaskedLM",
        "BertForMaskedLM",
        "BertConfig",
        "Trainer",
        "TrainingArguments",
        "mlm_probability",
        "ignore_index",
        "labels=-100",
        "mask token prediction",
        "top-k prediction",
        "save_pretrained",
        "wikimedia/wikipedia",
        "Korean Wikipedia",
        "klue/bert-base",
        "random baseline",
        "language modeling head",
        "한국어 위키백과",
        "한국어 사전학습",
        "마스킹 비율",
        "80/10/10 규칙",
        "무작위 초기화",
        "체크포인트 저장",
    ),
    23: (
        "BertForSequenceClassification",
        "load_state_dict",
        "classifier head",
        "single_label_classification",
        "softmax",
        "fine-tune loss",
        "NSMC binary classification",
        "domain transfer",
        "random baseline",
        "head replacement",
        "body weight transfer",
        "negative transfer",
        "본체 가중치",
        "헤드 교체",
        "도메인 전이",
        "한국어 분류",
        "태스크 도메인",
        "정확도",
        "AUC",
    ),
    24: (
        "PreTrainedTokenizerFast",
        "BpeTrainer",
        "ByteLevelDecoder",
        "GPT2Config",
        "GPT2LMHeadModel",
        "GPT2LMHeadModel(config)",
        "tie_word_embeddings",
        "causal attention",
        "language modeling head",
        "random baseline",
        "ln(vocab)",
        "perplexity",
        "VRAM trace",
        "model.generate",
        "sampling hyperparameter",
        "SFT",
        "alignment",
        "pretraining",
        "response-only loss",
        "TinyStories 사전학습",
        "바이트 레벨 BPE",
        "언어 모델링 헤드",
        "가중치 공유",
        "무작위 초기화",
        "랜덤 기준선",
        "학습 전 생성",
        "학습 후 생성",
        "응답 구간 손실",
    ),
    25: (
        "AutoModelForCausalLM.from_pretrained",
        "AutoTokenizer.from_pretrained",
        "tokenizer.pad_token",
        "eos_token",
        "WebText pretraining",
        "learning_rate=2e-5",
        "gradient_accumulation_steps",
        "effective batch",
        "domain adaptation",
        "continual pretraining loss",
        "Ch 24 vs Ch 25",
        "scale effect",
        "pretraining effect",
        "SFT boundary",
        "prompt masking",
        "도메인 적응",
        "효과적 배치",
        "본체 출발점",
        "스케일 효과",
        "사전학습 효과",
        "SFT 경계",
        "프롬프트 마스킹",
    ),
    26: (
        "BpeTrainer",
        "ByteLevel",
        "ByteLevelDecoder",
        "PreTrainedTokenizerFast",
        "special_tokens",
        "endoftext token",
        "story restoration",
        "streaming dataset",
        "Korean BBPE vs GPT-2 BPE",
        "token length comparison",
        "random baseline",
        "ln(vocab)",
        "perplexity",
        "VRAM trace",
        "model.generate",
        "sampling hyperparameter",
        "continual pretraining preview",
        "story 복원",
        "줄 단위 데이터",
        "토큰 길이 비교",
        "무작위 기준선",
        "한국어 사전학습",
        "한국어 생성 비교",
        "KoGPT2 reference",
        "한국어 continual pretraining",
    ),
    27: (
        "AutoTokenizer fallback",
        "English GPT2 fallback",
        "special token explicit loading",
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "mask_token",
        "KoGPT2 BBPE",
        "vocab 51200",
        "random baseline",
        "ln(vocab)",
        "catastrophic forgetting",
        "effective batch",
        "VRAM trace",
        "Ch 25 vs Ch 27",
        "Ch 26 vs Ch 27",
        "SFT boundary",
        "prompt masking",
        "tokenizer fertility",
        "Korean vocab share",
        "byte decomposition",
        "jamo decomposition",
        "한국어 토크나이저 품질",
        "한국어 vocab 점유율",
        "자모 분해",
        "byte 분해",
        "fertility",
        "토큰화 품질",
        "SFT 경계",
        "프롬프트 마스킹",
    ),
    28: (
        "DataCollatorForCompletionOnlyLM removed",
        "TRL 1.5.1",
        "SFTConfig(completion_only_loss=True)",
        "prompt column",
        "completion column",
        "completion mask",
        "prompt tokens masked",
        "answer tokens learned",
        "labels=-100 thread",
        "MLM vs CausalLM vs SFT",
        "instruction following",
        "behavior alignment",
        "fine-tuning meaning shift",
        "task head vs behavior alignment",
        "response template",
        "chat template",
        "KoGPT2 SFT",
        "KoAlpaca SFT",
        "응답 마스킹",
        "지시 따르기",
        "행동 정렬",
        "파인튜닝 의미 변화",
        "프롬프트 제외",
        "답변만 학습",
        "completion_only_loss",
    ),
    29: (
        "KoBEST HellaSwag",
        "KoBEST BoolQ",
        "HFLM",
        "simple_evaluate",
        "TaskManager",
        "loglikelihood",
        "acc",
        "stderr",
        "MMLU",
        "KMMLU",
        "GSM8K",
        "HumanEval",
        "MT-Bench",
        "LogicKor",
        "Chatbot Arena",
        "position bias",
        "verbosity bias",
        "judge ceiling",
        "gold standard",
        "Goodhart's law",
        "EXAONE",
        "Gemma",
        "Qwen",
        "GLM",
        "DeepSeek",
        "tech report",
        "평가 생태계",
        "평가 도구",
        "평가 함정",
        "정확도",
        "정규식 추출",
        "판정 편향",
        "전문가 평가",
        "대중 평가",
        "평가 항해 전략",
    ),
    30: (
        "DPO loss",
        "margin=0",
        "loss=0.6931",
        "sigmoid loss",
        "policy",
        "frozen ref",
        "ref_model=None",
        "completion-only preference",
        "instruction-following",
        "truthfulness",
        "honesty",
        "helpfulness",
        "binarized preference",
        "GPT-4 judge",
        "PPO 4 models",
        "T4 training",
        "gradient_accumulation_steps",
        "fp16",
        "reward accuracy",
        "DPO beta",
        "선호 쌍",
        "마진 직관",
        "기준 정책",
        "정책 최적화",
        "품질 정렬",
        "정직성",
        "유용성",
        "사실성",
        "지시 준수",
        "이진화 선호",
    ),
    31: (
        "reward=0",
        "std=0",
        "advantage=0",
        "baseline",
        "group mean",
        "response generation",
        "verifier pass rate",
        "arithmetic reward",
        "base capability",
        "reward hacking",
        "RL before SFT",
        "DeepSeek-R1 style",
        "Qwen appendix",
        "HPO",
        "hyperparameter",
        "temperature",
        "learning_rate",
        "beta=0.04",
        "max_steps",
        "format compliance",
        "Diffusion LM",
        "Phase 5",
        "보상 0",
        "기준선",
        "그룹 평균",
        "자동 채점",
        "산술 보상",
        "기반 능력",
        "보상 해킹",
        "하이퍼파라미터",
        "형식 준수",
        "Diffusion 언어모델",
    ),
}


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F100-\U0001F1FF"
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
    text = (
        text.replace("❌", "X")
        .replace("✅", "OK")
        .replace("✓", "OK")
        .replace("✔", "OK")
        .replace("✗", "X")
        .replace("✘", "X")
        .replace("📚", "")
        .replace("⚠️", "주의")
        .replace("⚠", "주의")
        .replace("Ġ", "<sp>")
        .replace("Ċ", "<nl>")
        .replace("\uFFFD", "?")
        .replace("①", "1")
        .replace("②", "2")
        .replace("③", "3")
        .replace("④", "4")
        .replace("⑤", "5")
        .replace("⑥", "6")
        .replace("⑦", "7")
        .replace("⑧", "8")
        .replace("⑨", "9")
        .replace("⑩", "10")
        .replace("\ufe0f", "")
    )
    return EMOJI_PATTERN.sub("", text)


def normalize_markdown_math_symbols(text: str) -> str:
    return (
        text.replace("λ", r"$\lambda$")
        .replace("β", r"$\beta$")
        .replace("α", r"$\alpha$")
        .replace("θ", r"$\theta$")
        .replace("Δ", r"$\Delta$")
        .replace("≈", r"$\approx$")
        .replace("≤", r"$\le$")
        .replace("≥", r"$\ge$")
        .replace("×", r"$\times$")
        .replace("→", r"$\to$")
        .replace("←", r"$\leftarrow$")
        .replace("↓", r"$\downarrow$")
        .replace("↔", r"$\leftrightarrow$")
        .replace("≠", r"$\ne$")
        .replace("−", "-")
    )


def sanitize_markdown_unicode(text: str) -> str:
    """Normalize notebook markdown before pandoc sees it."""
    text = unicodedata.normalize("NFC", text)
    cleaned: list[str] = []
    for char in text:
        code = ord(char)
        if 0x1100 <= code <= 0x11FF:
            cleaned.append(f"U+{code:04X}")
        elif unicodedata.category(char)[0] == "C" and char not in "\n\t":
            continue
        else:
            cleaned.append(char)
    return "".join(cleaned)


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


def sanitize_latex_text_unicode(text: str) -> str:
    replacements = {
        "Ġ": "<sp>",
        "Ċ": "<nl>",
        "\uFFFD": "?",
        "✓": "OK",
        "✔": "OK",
        "✗": "X",
        "✘": "X",
        "β": r"$\beta$",
        "α": r"$\alpha$",
        "θ": r"$\theta$",
    }
    text = EMOJI_PATTERN.sub("", text)
    cleaned: list[str] = []
    for char in unicodedata.normalize("NFC", text):
        if char in replacements:
            cleaned.append(replacements[char])
            continue
        code = ord(char)
        if 0x1100 <= code <= 0x11FF:
            cleaned.append(f"U+{code:04X}")
            continue
        if unicodedata.category(char)[0] == "C" and char not in "\n\t":
            continue
        cleaned.append(char)
    return "".join(cleaned)


def sanitize_latex_unicode(latex: str) -> str:
    """Remove remaining unsupported glyphs while preserving verbatim blocks."""
    lines: list[str] = []
    in_verbatim = False
    for line in latex.splitlines():
        stripped = line.strip()
        if stripped.startswith(("\\begin{lstlisting}", "\\begin{bookoutputbox}", "\\begin{verbatim}")):
            in_verbatim = True
            lines.append(line)
            continue
        if stripped.startswith(("\\end{lstlisting}", "\\end{bookoutputbox}", "\\end{verbatim}")):
            in_verbatim = False
            lines.append(line)
            continue
        lines.append(sanitize_listing_unicode(line) if in_verbatim else sanitize_latex_text_unicode(line))
    return "\n".join(lines) + ("\n" if latex.endswith("\n") else "")


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


MAX_CODE_LINES_PER_BLOCK = 16


def semantic_code_chunks(source: str, max_lines: int = MAX_CODE_LINES_PER_BLOCK) -> list[list[str]]:
    """Split long listings into smaller reading units without changing code."""
    lines = source.splitlines()
    chunks: list[list[str]] = []
    start = 0
    while start < len(lines):
        remaining = len(lines) - start
        if remaining <= max_lines:
            chunks.append(lines[start:])
            break

        lower = start + 8
        upper = min(start + max_lines, len(lines))
        split_at: int | None = None

        for idx in range(upper, lower, -1):
            if not lines[idx - 1].strip():
                split_at = idx
                break
        if split_at is None:
            for idx in range(upper, lower, -1):
                stripped = lines[idx].lstrip() if idx < len(lines) else ""
                if stripped.startswith((
                    "def ",
                    "class ",
                    "for ",
                    "if ",
                    "with ",
                    "trainer.",
                    "model.",
                    "plt.",
                    "fig,",
                    "ax.",
                    "g.",
                    "sns.",
                    "records",
                    "df_",
                )):
                    split_at = idx
                    break
        if split_at is None:
            split_at = upper

        chunks.append(lines[start:split_at])
        start = split_at
        while start < len(lines) and not lines[start].strip():
            start += 1
    return chunks


def code_chunk_summary(lines: list[str]) -> str:
    joined = "\n".join(lines).lower()
    if re.search(r"^\s*(import|from)\s+", joined, flags=re.MULTILINE):
        return "필요한 라이브러리와 기본 설정을 준비하는 단계"
    if "load_dataset" in joined or "read_csv" in joined or "to_pandas" in joined or "train_test_split" in joined:
        return "데이터를 불러오고 학습에 맞는 형태로 정리하는 단계"
    if "tokenizer" in joined or "tokenize" in joined or "data_collator" in joined:
        return "텍스트를 모델 입력 텐서로 바꾸는 단계"
    if "trainingarguments" in joined or "trainer" in joined or ".train(" in joined:
        return "학습 설정을 만들고 학습 루프를 실행하는 단계"
    if "metric" in joined or "classification_report" in joined or "confusion_matrix" in joined or "precision_recall" in joined:
        return "예측 결과를 지표와 표로 요약하는 단계"
    if "plt." in joined or "sns." in joined or "figure" in joined:
        return "숫자 결과를 그림으로 확인하는 단계"
    if "predict" in joined or "logits" in joined or "proba" in joined:
        return "모델 출력을 확률이나 예측값으로 바꾸는 단계"
    return "앞에서 만든 중간 값을 다음 계산으로 넘기는 단계"


def code_transition(previous: list[str], next_chunk: list[str]) -> str:
    before = code_chunk_summary(previous)
    after = code_chunk_summary(next_chunk)
    if before == after:
        return (
            "\\noindent\\emph{앞뒤 블록은 모두 "
            + before
            + "입니다. 길이를 나누어 같은 흐름을 단계별로 확인합니다.}"
        )
    return (
        "\\noindent\\emph{앞 블록은 "
        + before
        + "입니다. 이어지는 블록에서는 "
        + after
        + "로 넘어갑니다.}"
    )


def listing_needspace(line_count: int) -> int:
    return max(5, min(line_count + 2, 11))


def listing_block(source: str, options: str = "", firstnumber: int | None = None) -> str:
    lines = source.splitlines()
    option_text = options or ""
    if firstnumber is not None and firstnumber > 1 and "style=bookoutput" not in option_text:
        if option_text:
            option_text = option_text[:-1] + f",firstnumber={firstnumber}]"
        else:
            option_text = f"[firstnumber={firstnumber}]"
    return (
        f"\\Needspace{{{listing_needspace(len(lines))}\\baselineskip}}\n"
        f"\\begin{{lstlisting}}{option_text}\n"
        + source
        + "\n\\end{lstlisting}"
    )


def split_listing_for_book(source: str, options: str = "") -> str:
    base_first_line = 1
    firstnumber_match = re.search(r"firstnumber\s*=\s*(\d+)", options)
    if firstnumber_match:
        base_first_line = int(firstnumber_match.group(1))
        cleaned = re.sub(r",?\s*firstnumber\s*=\s*\d+\s*", "", options[1:-1]).strip()
        options = f"[{cleaned}]" if cleaned else ""

    line_count = len(source.splitlines())
    if "style=bookoutput" in options or line_count <= MAX_CODE_LINES_PER_BLOCK:
        return listing_block(source, options, base_first_line)

    chunks = semantic_code_chunks(source)
    blocks: list[str] = []
    first_line = base_first_line
    for idx, chunk in enumerate(chunks):
        blocks.append(listing_block("\n".join(chunk), options, first_line))
        first_line += len(chunk)
        if idx < len(chunks) - 1:
            blocks.append(code_transition(chunk, chunks[idx + 1]))
    return "\n\n".join(blocks)


def format_embedded_listings(latex: str) -> str:
    """Apply book code wrapping to fenced code blocks embedded in markdown."""

    def repl(match: re.Match[str]) -> str:
        options = match.group(1) or ""
        source = match.group(2).strip("\n")
        if not source.strip():
            return match.group(0)
        formatted = format_code_for_book(source)
        return split_listing_for_book(formatted, options)

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


# "챕터"(받침 없음) 를 "장"(받침 있음) 으로 바꾸면 뒤따르는 조사의 이형태가 달라진다.
# 개별 문구를 치환 사전에 계속 추가하는 대신, 조사 대응표로 한 번에 처리한다.
JOSA_AFTER_JANG = {
    "가": "이",
    "는": "은",
    "를": "을",
    "와": "과",
    "로": "으로",
    "라": "이라",
    "란": "이란",
    "랑": "이랑",
    "나": "이나",
    "며": "이며",
    "와의": "과의",
    "와는": "과는",
    "와도": "과도",
    "로의": "으로의",
    "로만": "으로만",
    "로는": "으로는",
    "로도": "으로도",
    "로서": "으로서",
    "로써": "으로써",
    "로부터": "으로부터",
    "라는": "이라는",
    "라도": "이라도",
    "라서": "이라서",
    "라면": "이라면",
    "라도록": "이라도록",
}

# 받침 유무와 무관하게 형태가 같아 그대로 붙는 조사·어미.
JOSA_INVARIANT_AFTER_JANG = (
    "은", "이", "을", "과", "으로", "의", "에", "에는", "에도", "에서",
    "에서는", "에서도", "에선", "엔", "부터", "부터는", "까지", "까지는", "도", "만",
    "보다", "처럼", "마다", "조차", "밖에", "이라", "이란", "이라는",
    "이라도", "이라서", "이라면", "이며", "이나", "이랑", "이면", "이고",
    "이지만", "이므로",
    # 조사가 겹친 형태. 뒤에 한글이 이어지므로 낱개 목록만으로는 안 잡힌다.
    "과의", "과는", "과도", "의의", "에서의", "에의", "부터의", "까지의",
    "으로의", "으로만", "만의", "도의",
)

_JANG_JOSA_ALTS = sorted(
    set(JOSA_AFTER_JANG) | set(JOSA_INVARIANT_AFTER_JANG), key=len, reverse=True
)

# "챕터" 바로 뒤에 붙은 조사. 긴 형태부터 시도해야 "와의" 가 "와" 로 잘리지 않는다.
CHAPTER_JOSA_PATTERN = re.compile(
    "챕터(" + "|".join(sorted(JOSA_AFTER_JANG, key=len, reverse=True)) + ")(?![가-힣])"
)

# "이번 챕터" 는 "이 장" 으로 다듬는다. 조사까지 한 패턴에 넣어야 순서 문제가 없고,
# 원고가 처음부터 "이번 장" 이라고 쓴 곳은 건드리지 않는다.
THIS_CHAPTER_PATTERN = re.compile(
    "이번 챕터(" + "|".join(_JANG_JOSA_ALTS) + ")?(?![가-힣])"
)

# "\ref{ch:04}장와" (이형태 오류) 와 "20장 에서" (조사 띄어쓰기) 를 함께 잡는다.
# 앞이 숫자나 닫는 중괄호일 때만 적용해, 장(場)·장(章) 이 아닌 일반 명사는 건드리지 않는다.
JANG_JOSA_PATTERN = re.compile(
    r"(?<=[0-9}])장 ?(" + "|".join(_JANG_JOSA_ALTS) + r")(?![가-힣])"
)


def fix_chapter_josa(text: str) -> str:
    """'챕터'+조사를 '장'+맞는 이형태로 바꾼다 (챕터를 -> 장을)."""
    text = THIS_CHAPTER_PATTERN.sub(
        lambda m: "이 장" + JOSA_AFTER_JANG.get(m.group(1), m.group(1) or ""), text
    )
    return CHAPTER_JOSA_PATTERN.sub(lambda m: "장" + JOSA_AFTER_JANG[m.group(1)], text)


def fix_jang_josa(text: str) -> str:
    """장 번호 뒤 조사의 띄어쓰기와 이형태를 바로잡는다 (20장 에서 -> 20장에서)."""
    return JANG_JOSA_PATTERN.sub(
        lambda m: "장" + JOSA_AFTER_JANG.get(m.group(1), m.group(1)), text
    )


def polish_prose_line(latex: str) -> str:
    """Normalize notebook-style shorthand and informal wording for one prose line."""

    # Chapter references.
    latex = re.sub(r"\bChapter\s+([0-9]+)", r"\1장", latex)
    latex = re.sub(r"\bCh\s*([0-9]+)\s*-\s*([0-9]+)", r"\1-\2장", latex)
    latex = re.sub(r"\bCh\s*([0-9]+)\s*·\s*([0-9]+)", r"\1·\2장", latex)
    latex = re.sub(r"\bCh\s*([0-9]+)", r"\1장", latex)

    # 조사는 사전이 아니라 규칙으로 처리한다. 이형태가 없는 조사(의/에/에서/마다 …)
    # 는 아래 "챕터" -> "장" 이 그대로 넘겨주므로 여기서 다룰 필요가 없다.
    latex = fix_chapter_josa(latex)
    latex = fix_jang_josa(latex)

    replacements = {
        # 조사가 붙은 "챕터" 는 위 fix_chapter_josa 가 이미 처리했다.
        # 여기 남는 것은 조사 없이 쓰인 "챕터" 와 "챕터별/챕터들" 같은 파생형이다.
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
        "13장. BERT Multi-label --- Yelp 항목 키워드": "13장. BERT 다중 라벨 분류 (Multi-label \\& Per-label BCE)",
        "14장. BERT Auxiliary Loss --- 항목 분류 + 별점 보조 회귀 (Phase 1 클라이맥스)": "14장. 보조 손실과 멀티태스크 학습 (Auxiliary Loss)",
        "15장. 한국어 BERT Binary --- NSMC": "15장. 한국어 BERT 이진 분류 (Korean Binary Classification)",
        "16장. 한국어 BERT Multi-class --- KLUE-YNAT (뉴스 7분류)": "16장. 한국어 BERT 다중 클래스 분류 (Korean Multi-class Classification)",
        "17장. 한국어 BERT Multi-label --- KLUE-YNAT 합성 multi-label": "17장. 한국어 BERT 다중 라벨 분류 (Korean Multi-label Classification)",
        "18장. 한국어 BERT Auxiliary Loss --- KLUE-YNAT 합성 multi-label + 활성 라벨 개수 보조 회귀 (Phase 2 클라이맥스)": "18장. 한국어 BERT 보조 손실 (Korean Auxiliary Loss)",
        "Loss 노트": "손실 노트",
    }
    for before, after in replacements.items():
        latex = latex.replace(before, after)

    # More formal section/table labels after chapter-reference normalization.
    latex = re.sub(r"변경점 \(Diff from ([0-9]+장)\)", r"변경점: \1 대비", latex)
    latex = latex.replace("전체 18장 표", "전체 18개 장의 표")
    latex = latex.replace("전체 19장 표", "전체 19개 장의 표")
    latex = latex.replace(r"\#장별-변화추적표", r"\#챕터별-변화추적표")
    return latex


# 코드 리스팅과 실행 출력은 원문 그대로 두어야 한다. 산문용 치환이 안까지 들어가면
# 코드에는 "18장", 그 출력에는 "Ch 18" 이 찍히는 식으로 둘이 어긋난다.
VERBATIM_BEGINS = (
    r"\begin{lstlisting}",
    r"\begin{verbatim}",
    r"\begin{bookoutputbox}",
)
VERBATIM_ENDS = (
    r"\end{lstlisting}",
    r"\end{verbatim}",
    r"\end{bookoutputbox}",
)


def polish_book_prose(latex: str) -> str:
    """Apply prose polishing to every line outside code and output blocks."""

    polished: list[str] = []
    in_verbatim = False
    for line in latex.splitlines():
        stripped = line.strip()
        if stripped.startswith(VERBATIM_BEGINS):
            in_verbatim = True
            polished.append(line)
            continue
        if stripped.startswith(VERBATIM_ENDS):
            in_verbatim = False
            polished.append(line)
            continue
        # 코드 블록 안에서는 장 번호 표기만 통일하고 표현은 손대지 않는다.
        polished.append(polish_chapter_refs(line) if in_verbatim else polish_prose_line(line))
    joined = "\n".join(polished) + ("\n" if latex.endswith("\n") else "")
    return normalize_heading_titles(joined)


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
        "데이터 --- Yelp + 항목(aspect) 합성 라벨 (6장과 동일)": "데이터 준비",
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
        "데이터 --- Yelp + 항목 (13장) + 별점 보조 라벨": "데이터 준비",
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
        "라벨별 F1 비교 --- 어느 항목이 보조 loss로 가장 도움받았나": "라벨별 F1 비교",
        "보조 task 자체는 얼마나 잘 학습됐나": "보조 태스크 평가",
        "Loss 노트 --- Ch 11 그대로": "손실 노트",
        "토크나이저 노트 --- Phase 2 의 핵심": "토크나이저 노트",
        "토크나이저 비교 --- 같은 한국어 문장, 두 토크나이저": "토크나이저 비교",
        "데이터 --- NSMC (네이버 영화 리뷰)": "데이터 준비",
        "토큰화 --- Ch 11 패턴 그대로, 토크나이저만 한국어로": "토큰화",
        "모델 로드 --- \\inlinecode{klue/bert-base} + binary 분류 헤드": "모델 로드",
        "모델 로드 --- klue/bert-base + binary 분류 헤드": "모델 로드",
        "학습 --- Ch 11 과 동일한 hyperparams": "학습",
        "평가 --- softmax 확률 분포": "평가",
        "메인 그림 --- 확률 공간 KDE (Ch 11 와 동일 패턴)": "확률 분포",
        "보조 그림 --- logit 공간 KDE (z = z\\_1 - z\\_0)": "로짓 분포",
        "보조 그림 --- logit 공간 KDE (z = z_1 - z_0)": "로짓 분포",
        "샘플 단위 해석 --- 실제 한국어 리뷰가 어떻게 분류되나": "샘플 단위 해석",
        "Loss 노트 --- \\inlinecode{CrossEntropyLoss} 가 K=7 에서 보이는 모습": "손실 노트",
        "Loss 노트 --- CrossEntropyLoss 가 K=7 에서 보이는 모습": "손실 노트",
        "데이터 --- KLUE-YNAT (뉴스 헤드라인 7분류)": "데이터 준비",
        "토큰화 --- Ch 15 패턴 그대로": "토큰화",
        "모델 로드 --- \\inlinecode{num\\_labels=7} 만 바뀜": "모델 로드",
        "모델 로드 --- num\\_labels=7 만 바뀜": "모델 로드",
        "학습 --- Ch 15 와 동일한 hyperparams": "학습",
        "평가 --- softmax 확률 분포 + 혼동 패턴": "평가",
        "혼동 행렬 --- 어디서 헷갈리는가": "혼동 행렬",
        "Top-1 확률 분포 --- 모델 자신감 진단": "최상위 확률 분포",
        "샘플 단위 해석 --- 실제 헤드라인이 어떻게 분류되나": "샘플 단위 해석",
        "Loss 함수의 변화 --- \\inlinecode{CrossEntropyLoss} \\(\n\\to\n\\) \\inlinecode{BCEWithLogitsLoss} per-label": "손실 함수의 변화 --- 라벨별 이진 교차 엔트로피",
        "Loss 함수의 변화 --- CrossEntropyLoss → BCEWithLogitsLoss per-label": "손실 함수의 변화 --- 라벨별 이진 교차 엔트로피",
        "Loss 함수의 변화 --- CrossEntropyLoss \\texorpdfstring{\\(\n\\to\n\\)}{→} BCEWithLogitsLoss per-label": "손실 함수의 변화 --- 라벨별 이진 교차 엔트로피",
        "데이터 --- KLUE-YNAT 결합으로 multi-label 합성": "데이터 준비",
        "두 헤드라인을 결합해 multi-label 샘플 합성": "다중 라벨 샘플 합성",
        "토큰화 --- Ch 16 패턴, 라벨 형식만 multi-hot": "토큰화",
        "모델 로드 --- \\inlinecode{num\\_labels=7} 그대로, \\inlinecode{problem\\_type} 만 전환": "모델 로드",
        "모델 로드 --- num\\_labels=7 그대로, problem\\_type 만 전환": "모델 로드",
        "학습 --- Ch 16 과 동일한 hyperparams": "학습",
        "평가 --- 카테고리별 sigmoid 확률 + 공동 활성 패턴": "평가",
        "카테고리별 sigmoid 확률 KDE (7 패널)": "카테고리별 확률 분포",
        "카테고리 간 공동 활성 패턴": "공동 활성 패턴",
        "변형 --- 합성 샘플 직접 읽기 + threshold 옮겨보기": "변형: 임계값 탐색",
        "Loss 노트 --- Combined loss \\inlinecode{L = L\\_main + λ · L\\_aux}": "손실 노트 --- 결합 손실",
        "Loss 노트 --- Combined loss L = L\\_main + λ · L\\_aux": "손실 노트 --- 결합 손실",
        "데이터 --- KLUE-YNAT 합성 multi-label + 활성 개수 보조 라벨": "데이터 준비",
        "합성 함수 --- Ch 17 의 \\inlinecode{make\\_multilabel} 재사용": "합성 함수",
        "합성 함수 --- Ch 17 의 make\\_multilabel 재사용": "합성 함수",
        "토큰화 --- 메인 multi-hot + 보조 \\inlinecode{n\\_active} 같이 부착": "토큰화",
        "토큰화 --- 메인 multi-hot + 보조 n\\_active 같이 부착": "토큰화",
        "커스텀 Data Collator --- \\inlinecode{n\\_active} 도 batch 에 같이 담기": "커스텀 Data Collator",
        "커스텀 Data Collator --- n\\_active 도 batch 에 같이 담기": "커스텀 Data Collator",
        "모델 --- \\inlinecode{AutoModel} 본체 + 메인 헤드 + 보조 헤드 직접 부착": "모델 정의",
        "모델 --- AutoModel 본체 + 메인 헤드 + 보조 헤드 직접 부착": "모델 정의",
        "커스텀 Trainer --- \\inlinecode{compute\\_loss} 오버라이드": "커스텀 Trainer",
        "커스텀 Trainer --- compute\\_loss 오버라이드": "커스텀 Trainer",
        "학습 --- λ=0.1 (보조 ON)": "학습",
        "학습 --- lambda=0.1 (보조 ON)": "학습",
        "평가 --- 메인 task + 보조 task": "평가",
        "클라이맥스 --- \\emph{λ=0 baseline} 학습 (= Ch 17 재현)": "λ=0 기준선 비교",
        "클라이맥스 --- λ=0 baseline 학습 (= Ch 17 재현)": "λ=0 기준선 비교",
        "메인 metric 비교 --- λ=0 baseline vs λ=0.1 aux": "메인 지표 비교",
        "카테고리별 F1 비교 --- 어느 카테고리가 보조 loss 로 가장 도움받았나": "카테고리별 F1 비교",
        "보조 task 자체는 얼마나 잘 학습됐나": "보조 태스크 평가",
        "변형 --- λ 스윕 효과 비교 (선택)": "변형: λ 스윕",
        "결과 해석 --- 보조 loss 가 \\emph{항상 좋게 나오지는 않습니다}": "결과 해석",
        "19장. 토크나이저 직접 학습 --- WordPiece vs WordLevel (영어 + 한국어)": "19장. 토크나이저 직접 학습 (Tokenizer Training)",
        "변경점 (Diff from Ch 18)": "변경점: 18장 대비",
        "왜 토크나이저를 \\emph{직접} 학습해야 하나": "왜 직접 학습하는가",
        "토크나이저 알고리즘 노트 --- WordPiece vs WordLevel": "알고리즘 노트",
        "WordLevel --- 단순 어절 (whole-word)": "WordLevel",
        "WordPiece --- subword (BERT 표준)": "WordPiece",
        "수치 예시 (같은 문장이 두 알고리즘에서 몇 토큰?)": "수치 예시",
        "BERT 가 WordPiece 를 쓰는 이유": "BERT와 WordPiece",
        "토크나이저 노트 --- 이 장의 \\emph{주제} 자체": "토크나이저 노트",
        "환경 셋업": "환경 준비",
        "영어 코퍼스 --- Yelp text 5,000건": "영어 코퍼스",
        "한국어 코퍼스 --- NSMC text 5,000건": "한국어 코퍼스",
        "토크나이저 4종 학습": "토크나이저 학습",
        "학습된 vocab 안을 들여다보기": "어휘 확인",
        "해부 --- 같은 문장을 4 토크나이저로 비교": "해부: 토큰화 비교",
        "비교 시각화": "비교 시각화",
        "토큰 길이 분포 --- 같은 텍스트를 4 토크나이저로": "토큰 길이 분포",
        "Unknown 토큰 비율 --- vocab 한계가 드러나는 곳": "미등록 토큰 비율",
        "2×2 비교 표 --- 한눈에 정리": "2x2 비교 표",
        "교차 적용 --- 영어 토크나이저로 한국어를, 그 반대도": "교차 언어 적용",
        "저장·로드 --- \\inlinecode{tokenizer.save()} / \\inlinecode{PreTrainedTokenizerFast} 로 wrap": "저장과 로드",
        "저장·로드 --- tokenizer.save() / PreTrainedTokenizerFast 로 wrap": "저장과 로드",
        "변형 --- vocab 크기 sweep": "변형: 어휘 수 스윕",
        "이번 장에 등장한 라이브러리·함수": "이 장에 등장한 라이브러리·함수",
        "20장. 작은 BERT 직접 사전학습 --- 영어 MLM (scratch)": "20장. 작은 BERT 사전학습 (English MLM Pretraining)",
        "21장. 작은 BERT 분류 --- 영어 Yelp 이진 (일반 도메인 사전학습 \\(\n\\to\n\\) 다른 도메인 fine-tune)": "21장. 작은 BERT 이진 분류 (English Yelp Fine-tuning)",
        "21장. 작은 BERT 분류 --- 영어 Yelp 이진 (일반 도메인 사전학습 → 다른 도메인 fine-tune)": "21장. 작은 BERT 이진 분류 (English Yelp Fine-tuning)",
        "22장. 작은 BERT 직접 사전학습 --- 한국어 MLM (scratch)": "22장. 작은 BERT 사전학습 (Korean MLM Pretraining)",
        "23장. 작은 BERT 분류 --- 한국어 NSMC 이진 (일반 도메인 사전학습 \\(\n\\to\n\\) 다른 도메인 fine-tune)": "23장. 작은 BERT 이진 분류 (Korean NSMC Fine-tuning)",
        "23장. 작은 BERT 분류 --- 한국어 NSMC 이진 (일반 도메인 사전학습 → 다른 도메인 fine-tune)": "23장. 작은 BERT 이진 분류 (Korean NSMC Fine-tuning)",
        "왜 토크나이저는 가져오고 모델만 직접 학습하나": "왜 모델만 직접 학습하는가",
        "왜 task corpus (Yelp) 가 아니라 일반 위키인가 --- 원본 BERT 의 정신": "왜 일반 위키로 사전학습하는가",
        "Loss 함수의 변화 --- Masked Language Modeling (MLM)": "손실 함수의 변화: MLM",
        "숫자로 감 잡기 (vocab=30,522)": "수치 예시",
        "Perplexity (PPL)": "Perplexity",
        "같은 문장의 토큰화 --- Ch 19 직접 학습 vs Ch 20 가져옴": "토큰화 비교",
        "\"토크나이저는 모델과 운명공동체\"": "토크나이저와 모델의 결합",
        "토크나이저 --- \\inlinecode{bert-base-uncased} 그대로 로드": "토크나이저 로드",
        "토크나이저 --- bert-base-uncased 그대로 로드": "토크나이저 로드",
        "데이터 --- Wikitext-103 paragraphs (일반 도메인 사전학습 코퍼스)": "데이터 준비",
        "토큰화 + \\inlinecode{group\\_texts} --- HF \\inlinecode{run\\_mlm.py} 표준 패턴": "토큰화와 토큰 그룹화",
        "토큰화 + group_texts --- HF run_mlm.py 표준 패턴": "토큰화와 토큰 그룹화",
        "작은 \\inlinecode{BertConfig} + \\inlinecode{BertForMaskedLM} --- random init": "작은 BERT 모델 정의",
        "작은 BertConfig + BertForMaskedLM --- random init": "작은 BERT 모델 정의",
        "\\inlinecode{DataCollatorForLanguageModeling} + Trainer 학습": "MLM 학습",
        "DataCollatorForLanguageModeling + Trainer 학습": "MLM 학습",
        "\\texttt{{[}MASK{]}} 가 들어가는 원리 --- 한 눈에 보는 80/10/10": "마스킹 규칙",
        "[MASK] 가 들어가는 원리 --- 한 눈에 보는 80/10/10": "마스킹 규칙",
        "학습 직전 baseline --- 사전학습 전·후 비교 준비": "사전학습 전 기준선",
        "평가 --- MLM loss 곡선 + perplexity + masked token 예측": "평가",
        "사전학습 전·후 비교 --- random init 본체 vs 2 epoch 학습 후": "사전학습 전후 비교",
        "eval_loss / perplexity --- 수치 비교": "평가 손실과 perplexity",
        "학습이 \\emph{충분히 잘 된 경우} 의 기준점 --- 표준 \\inlinecode{bert-base-uncased} 비교": "표준 BERT 기준점",
        "학습이 충분히 잘 된 경우 의 기준점 --- 표준 bert-base-uncased 비교": "표준 BERT 기준점",
        "\\texttt{{[}MASK{]}} top-5 --- 3-way 비교 (before / ours / reference BERT)": "MASK top-5 비교",
        "[MASK] top-5 --- 3-way 비교 (before / ours / reference BERT)": "MASK top-5 비교",
        "모델 저장 --- Ch 21 에서 재사용": "모델 저장",
        "변형 --- 학습 step 더 늘리거나 block_size 변경": "변형: 학습량과 블록 크기",
        "변경점 (Diff from Ch 20)": "변경점: 20장 대비",
        "두 데이터셋이 노트북 안에 공존": "두 데이터셋의 역할",
        "Ch 10 (DistilBERT) 과의 비교가 본 장의 메인 메시지 --- 이제 \\emph{fair}": "DistilBERT 비교의 의미",
        "Ch 10 (DistilBERT) 과의 비교가 본 챕터의 메인 메시지 --- 이제 fair": "DistilBERT 비교의 의미",
        "Loss 함수의 변화 --- MLM CE (vocab=30,522) \\(\n\\to\n\\) 분류 CE (K=2)": "손실 함수의 변화: MLM에서 분류 CE로",
        "Loss 함수의 변화 --- MLM CE (vocab=30,522) → 분류 CE (K=2)": "손실 함수의 변화: MLM에서 분류 CE로",
        "두 CE 비교 (random baseline)": "두 CE 기준선 비교",
        "사전학습 효과가 \\emph{loss 곡선} 에 어떻게 드러나나": "사전학습 효과와 loss 곡선",
        "두 도메인의 어휘 --- 위키 vs Yelp": "위키와 Yelp 어휘",
        "분류 task 에서 [CLS] 토큰의 의미": "분류에서 CLS 토큰",
        "헤드 교체 시 어떤 파라미터가 어떻게 이어지나": "헤드 교체와 파라미터",
        "Yelp 이진 분류 데이터 로드 --- Ch 10 과 같은 split": "Yelp 데이터 준비",
        "토크나이저 --- \\inlinecode{bert-base-uncased} (Ch 20 과 동일)": "토크나이저 로드",
        "토크나이저 --- bert-base-uncased (Ch 20 과 동일)": "토크나이저 로드",
        "MLM 사전학습 --- Ch 20 패턴 압축 재현 (Wikitext-103, 2K × 3 epoch)": "MLM 사전학습",
        "MLM 사전학습 --- Ch 20 패턴 압축 재현 (Wikitext-103, 2K \\(\n\\times\n\\) 3 epoch)": "MLM 사전학습",
        "같은 단어 \"파인튜닝\", BERT 시대와 GPT 시대의 의미가 살짝 다릅니다": "파인튜닝이라는 말의 범위",
        "\\inlinecode{labels = -100} ignore_index 는 BERT-만의 트릭이 아닙니다 --- Phase 4 (GPT) 의 핵심으로 다시": "labels=-100과 ignore index",
        "labels = -100 ignore_index 는 BERT-만의 트릭이 아닙니다 --- Phase 4 (GPT) 의 핵심으로 다시": "labels=-100과 ignore index",
        "헤드 교체 --- MLM → 분류 + Fine-tune": "헤드 교체와 파인튜닝",
        "헤드 교체 --- MLM \\(\n\\to\n\\) 분류 + Fine-tune": "헤드 교체와 파인튜닝",
        "평가 --- Ch 10 과 같은 5종 metric + 학습 곡선": "평가",
        "학습 곡선 --- MLM 사전학습 효과가 보이는 자리": "학습 곡선",
        "Confusion matrix": "혼동 행렬",
        "Ch 10 (DistilBERT) vs Ch 21 (작은 BERT scratch) --- 본 장의 핵심 결과": "DistilBERT와 작은 BERT 비교",
        "부록 --- fair-compute 비교 (사전학습 없이 같은 GPU compute 로 분류만)": "참고: fair-compute 비교",
        "변경점 (Diff from Ch 20)": "변경점: 20장 대비",
        "Loss 함수의 변화 --- *없음*. Ch 20 과 같은 MLM CE": "손실 함수의 변화: MLM 유지",
        "토크나이저 노트 --- 본 챕터의 핵심 한 자리": "토크나이저 노트",
        "한국어 Wikipedia 데이터 로드 --- 일반 도메인 사전학습 코퍼스": "한국어 Wikipedia 데이터 준비",
        "토크나이저 --- \\inlinecode{klue/bert-base} 로드 + 영어 토크나이저와 한국어 비교": "토크나이저 로드와 비교",
        "토크나이저 --- klue/bert-base 로드 + 영어 토크나이저와 한국어 비교": "토크나이저 로드와 비교",
        "같은 한국어 문장을 두 토크나이저로 --- Ch 19 §5-4 cross-language 검증": "한국어 문장의 토큰화 비교",
        "토큰화 + \\inlinecode{group\\_texts} --- Ch 20 패턴 그대로": "토큰화와 토큰 그룹화",
        "토큰화 + group_texts --- Ch 20 패턴 그대로": "토큰화와 토큰 그룹화",
        "토큰화 + \\inlinecode{group\\_texts} --- \\ref{ch:20}장 패턴 그대로": "토큰화와 토큰 그룹화",
        "토큰화 + group_texts --- \\ref{ch:20}장 패턴 그대로": "토큰화와 토큰 그룹화",
        "작은 \\inlinecode{BertConfig} + \\inlinecode{BertForMaskedLM} --- random init (Ch 20 과 동일)": "작은 BERT 모델 정의",
        "작은 BertConfig + BertForMaskedLM --- random init (Ch 20 과 동일)": "작은 BERT 모델 정의",
        "\\inlinecode{DataCollatorForLanguageModeling} + Trainer 학습": "MLM 학습",
        "DataCollatorForLanguageModeling + Trainer 학습": "MLM 학습",
        "\\texttt{{[}MASK{]}} 가 들어가는 원리 --- 한 눈에 보는 80/10/10 (한국어 풀버전)": "마스킹 규칙",
        "[MASK] 가 들어가는 원리 --- 한 눈에 보는 80/10/10 (한국어 풀버전)": "마스킹 규칙",
        "학습 결과 --- Loss / Perplexity 곡선": "학습 결과",
        "사전학습 전·후 비교 --- random init 본체 vs 2 epoch 학습 후": "사전학습 전후 비교",
        "eval_loss / perplexity --- 수치 비교": "평가 손실과 perplexity",
        "학습이 \\emph{충분히 잘 된 경우} 의 기준점 --- 표준 \\inlinecode{klue/bert-base} 비교": "표준 KLUE-BERT 기준점",
        "학습이 충분히 잘 된 경우 의 기준점 --- 표준 klue/bert-base 비교": "표준 KLUE-BERT 기준점",
        "\\texttt{{[}MASK{]}} top-5 --- 3-way 비교 (before / ours / reference klue/bert-base)": "MASK top-5 비교",
        "[MASK] top-5 --- 3-way 비교 (before / ours / reference klue/bert-base)": "MASK top-5 비교",
        "모델 저장 --- Ch 23 에서 재사용": "모델 저장",
        "변형 --- 데이터 / 학습량 / 다른 한국어 코퍼스": "변형: 데이터와 학습량",
        "이번 챕터에 등장한 라이브러리·함수 (Ch 20 과의 차이만)": "이 장에 등장한 라이브러리·함수",
        "변경점 (Diff from Ch 22)": "변경점: 22장 대비",
        "Loss 함수의 변화 --- MLM CE (vocab 약 32,000) → 분류 CE (K=2)": "손실 함수의 변화: MLM에서 분류 CE로",
        "Loss 함수의 변화 --- MLM CE (vocab 약 32,000) \\(\\to\\) 분류 CE (K=2)": "손실 함수의 변화: MLM에서 분류 CE로",
        "NSMC 이진 분류 데이터 로드 --- Ch 15 와 같은 split": "NSMC 데이터 준비",
        "토크나이저 --- \\inlinecode{klue/bert-base} (Ch 22 와 동일)": "토크나이저 로드",
        "토크나이저 --- klue/bert-base (Ch 22 와 동일)": "토크나이저 로드",
        "MLM 사전학습 --- Ch 22 패턴 압축 재현 (한국어 Wikipedia, 1 epoch)": "MLM 사전학습",
        "헤드 교체 --- MLM → 분류 + Fine-tune": "헤드 교체와 파인튜닝",
        "헤드 교체 --- MLM \\(\n\\to\n\\) 분류 + Fine-tune": "헤드 교체와 파인튜닝",
        "헤드 교체 --- MLM \\(\\to\\) 분류 + Fine-tune": "헤드 교체와 파인튜닝",
        "평가 --- Ch 15 / Ch 21 과 같은 5종 metric + 학습 곡선": "평가",
        "학습 곡선 --- MLM 사전학습 효과가 보이는 자리": "학습 곡선",
        "2-way 비교 --- Ch 15 (klue/bert-base) vs Ch 23 ours (small BERT + ko wiki MLM)": "KLUE-BERT와 작은 BERT 비교",
        "부록 --- random init baseline + negative transfer 분석": "참고: random init baseline",
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


CODE_COMMENT_REPLACEMENTS = {
    "그냥": "직접",
    "뱉는": "출력하는",
    "뱉을": "출력할",
    "뱉습니다": "출력합니다",
    "뱉은": "출력한",
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
}


def polish_chapter_refs(text: str) -> str:
    """Ch 12 -> 12장. 장 번호 표기는 책 전체에서 한국어로 통일한다.

    코드 문자열과 그 실행 출력 양쪽에 똑같이 적용해야 한다. 한쪽만 바꾸면
    코드에는 "18장", 바로 아래 출력에는 "Ch 18" 이 찍혀 둘이 어긋난다.
    """
    text = re.sub(r"\bChapter\s+([0-9]+)", r"\1장", text)
    text = re.sub(r"\bCh\s*([0-9]+)\s*-\s*([0-9]+)", r"\1-\2장", text)
    text = re.sub(r"\bCh\s*([0-9]+)", r"\1장", text)
    # "Ch 4와" -> "4장와" 처럼 어긋난 조사와 "Ch 18 의" 식 띄어쓰기를 바로잡는다.
    return fix_jang_josa(text)


def polish_comment_text(comment: str) -> str:
    """Rewrite one Python comment into book wording.

    표현 다듬기는 주석에만 적용한다. 문자열 리터럴에 걸면 그 문구가 그대로
    실행 출력·그림 라벨이 되는데, 저장된 출력은 함께 바뀌지 않아 어긋난다.
    주석 안의 "챕터" 는 예전부터 그대로 두었으므로 여기서도 건드리지 않는다.
    """
    for before, after in CODE_COMMENT_REPLACEMENTS.items():
        comment = comment.replace(before, after)
    return comment


def apply_to_python_comments(source: str, transform: Callable[[str], str]) -> str:
    """Run a text transform over Python comments only.

    문자열 리터럴은 건드리지 않는다. 리터럴을 고치면 그대로 실행 출력이 되는데,
    저장된 출력은 함께 바뀌지 않아 코드에는 "18장", 출력에는 "Ch 18" 이 남는다.
    """
    lines = source.splitlines()
    if not lines:
        return source

    edited = lines[:]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT:
                continue
            row, col = tok.start
            line = edited[row - 1]
            edited[row - 1] = line[:col] + transform(line[col:])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # 토큰화가 안 되는 셀(매직 명령 등)은 줄 단위 어림짐작으로 처리한다.
        edited = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                indent = len(line) - len(stripped)
                edited.append(line[:indent] + transform(stripped))
            elif "  # " in line:
                code, comment = line.split("  # ", 1)
                edited.append(code + "  # " + transform(comment))
            else:
                edited.append(line)
    return "\n".join(edited)


def polish_code_comments(source: str) -> str:
    """Polish Korean wording in code comments, leaving executable code untouched."""
    # 장 번호는 코드 전체(문자열 리터럴 포함)에 적용한다. 같은 변환을
    # output_to_latex 가 실행 출력에도 걸어 주므로 둘이 어긋나지 않는다.
    source = polish_chapter_refs(source)
    source = apply_to_python_comments(source, polish_comment_text)
    # 아래 두 가지는 표현 다듬기가 아니라 코드 자체의 정리라 주석 밖에도 적용한다.
    source = source.replace('multi_class="multinomial", ', "")
    source = source.replace(', multi_class="multinomial"', "")
    source = source.replace("−", "-")
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


def compact_faq_section(latex: str, chapter_number: int) -> str:
    """Keep three chapter-specific FAQs and link the complete set online."""
    selected = set(COMPACT_FAQ_SELECTIONS.get(chapter_number, (1, 2, 3)))
    lines = latex.splitlines()
    question_numbers: list[int] = []
    in_faq = False
    for line in lines:
        if line == "\\section{FAQ}":
            in_faq = True
            continue
        if in_faq and (
            (line.startswith("\\section{") and line != "\\section{FAQ}")
            or line.startswith("\\chapter{")
            or line.startswith("\\begin{previewBox}")
        ):
            in_faq = False
        if not in_faq:
            continue
        match = re.match(r"^\\begin\{faqBox\}\{Q(\d+)\.", line)
        if match is None:
            match = re.match(r"^\\textbf\{Q(\d+)\.", line)
        if match:
            question_numbers.append(int(match.group(1)))

    omitted = max(0, len(question_numbers) - len([number for number in question_numbers if number in selected]))
    if omitted == 0:
        return latex

    compacted: list[str] = []
    in_faq = False
    skip_box = False
    skip_plain = False
    plain_box_open = False
    note_added = False

    def add_more_note() -> None:
        nonlocal note_added
        if not note_added:
            compacted.append(f"\\compactFAQMore{{{omitted}}}")
            compacted.append("")
            note_added = True

    for line in lines:
        if line == "\\section{FAQ}":
            in_faq = True
            skip_plain = False
            compacted.append(line)
            continue

        if in_faq and (
            (line.startswith("\\section{") and line != "\\section{FAQ}")
            or line.startswith("\\chapter{")
            or line.startswith("\\begin{previewBox}")
        ):
            if skip_box:
                skip_box = False
            if plain_box_open:
                compacted.append("\\end{faqBox}")
                compacted.append("")
                plain_box_open = False
            add_more_note()
            in_faq = False
            skip_plain = False

        if not in_faq:
            compacted.append(line)
            continue

        if skip_box:
            if line == "\\end{faqBox}":
                skip_box = False
            continue

        box_match = re.match(r"^\\begin\{faqBox\}\{Q(\d+)\.", line)
        if box_match:
            number = int(box_match.group(1))
            skip_plain = False
            if number not in selected:
                skip_box = True
                continue
            compacted.append(line)
            continue

        plain_match = re.match(r"^\\textbf\{Q(\d+)\.", line)
        if plain_match:
            if plain_box_open:
                compacted.append("\\end{faqBox}")
                compacted.append("")
                plain_box_open = False
            number = int(plain_match.group(1))
            skip_plain = number not in selected
            if skip_plain:
                continue
            title = line[len("\\textbf{") : -1] if line.endswith("}") else line
            compacted.append(f"\\begin{{faqBox}}{{{title}}}")
            plain_box_open = True
            continue

        if skip_plain:
            continue
        compacted.append(line)

    if plain_box_open:
        compacted.append("\\end{faqBox}")
        compacted.append("")
    if in_faq:
        add_more_note()
    return "\n".join(compacted)


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


def link_chapter_references(latex: str) -> str:
    """Turn prose references such as 3장 and 9-13장 into hyperlinked refs."""
    single = r"(?:[1-9]|[12][0-9]|3[01])"
    range_pat = re.compile(
        rf"(?<!ch:)(?<!ref\{{ch:)(?<!tab:ch)(?<!eq:ch)\b({single})\s*[-–]\s*({single})장"
    )
    dot_pat = re.compile(
        rf"(?<!ch:)(?<!ref\{{ch:)(?<!tab:ch)(?<!eq:ch)\b({single})·({single})장"
    )
    single_pat = re.compile(
        rf"(?<!ch:)(?<!ref\{{ch:)(?<!tab:ch)(?<!eq:ch)\b({single})장"
    )

    def ch_ref(number: str) -> str:
        return rf"\ref{{ch:{int(number):02d}}}"

    def convert(line: str) -> str:
        # 표 캡션과 장 제목 줄에는 \ref 를 넣지 않는다 (목차·표목차가 깨진다).
        # 다만 조사 정리는 이 줄들에도 적용해야 한다 — 예전에는 통째로 건너뛰어서
        # "20장 모델 저장 - 21장 에서 재사용" 같은 캡션이 그대로 남았다.
        linkable = not line.startswith(
            ("\\begin{booktable}{", "\\chapter", "\\chaptermeta")
        )
        if linkable:
            line = range_pat.sub(
                lambda m: f"{ch_ref(m.group(1))}--{ch_ref(m.group(2))}장", line
            )
            line = dot_pat.sub(
                lambda m: f"{ch_ref(m.group(1))}·{ch_ref(m.group(2))}장", line
            )
            line = single_pat.sub(lambda m: f"{ch_ref(m.group(1))}장", line)
            line = re.sub(
                r"\\href\{[^{}]+\}\{(\\ref\{ch:[0-9]{2}\}장[^{}]*)\}", r"\1", line
            )
        return fix_jang_josa(line)

    linked: list[str] = []
    in_listing = False
    for line in latex.splitlines():
        stripped = line.strip()
        # 출력 블록도 verbatim 이라 \ref 가 들어가면 명령이 그대로 인쇄된다.
        if stripped.startswith(VERBATIM_BEGINS):
            in_listing = True
            linked.append(line)
            continue
        if stripped.startswith(VERBATIM_ENDS):
            in_listing = False
            linked.append(line)
            continue
        linked.append(line if in_listing else convert(line))
    return "\n".join(linked) + ("\n" if latex.endswith("\n") else "")


def markdown_to_latex(markdown: str, chapter_number: int) -> str:
    markdown = sanitize_markdown_unicode(markdown)
    markdown = sanitize_symbols(promote_headings(strip_heading_emoji(markdown)))
    markdown = normalize_markdown_math_symbols(markdown)
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
    latex = sanitize_latex_unicode(latex)
    return latex.strip()


def code_walkthrough(source: str, compact: bool = False) -> str:
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
    note_limit = 3 if compact else 6
    # 본편은 예전처럼 "앞 note_limit 개 문장" 만 훑는다. print 문도 자리를 차지해서,
    # 셀 앞머리가 print 로 채워져 있으면 설명이 짧게 끝난다 (원고가 기대하는 모양).
    # 압축판은 1454ac0 에서 정한 대로 "노트 note_limit 개" 를 채울 때까지 계속 본다.
    candidates = statements if compact else statements[:note_limit]
    for start_line, end_line, content in candidates:
        if len(notes) >= note_limit:
            break
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
        "[transformers] No model was supplied",
        "[transformers] Passing `generation_config`",
        "[transformers] Setting `pad_token_id`",
        "[transformers] Both `max_new_tokens`",
        "[transformers] Ignoring clean_up_tokenization_spaces",
        "Using a pipeline without specifying a model name",
        "You are not authenticated with the Hugging Face Hub",
        "Error while fetching `HF_TOKEN`",
        "Warning: You are sending unauthenticated requests",
        "WARNING:huggingface_hub",
        "huggingface_hub/utils/_auth.py",
        "warnings.warn(",
        "LOAD REPORT from:",
        "Key                         | Status",
        "UNEXPECTED",
        "Notes:",
        "- UNEXPECTED:",
        "<IPython.core.display.HTML object>",
    )
    lines = [
        line
        for line in text.splitlines()
        if not any(pattern in line for pattern in skip_patterns)
        and not line.strip().startswith("from .autonotebook import tqdm")
        and not re.fullmatch(r"<Figure size [0-9.]+x[0-9.]+ with \d+ Axes>", line.strip())
        and not re.search(r":\s+\d+%\|", line)
        and "[00:00<" not in line
        and "[00:00?" not in line
        and not any(char in line for char in "━╺╸")
    ]
    text = sanitize_listing_unicode(sanitize_symbols("\n".join(lines).strip()))
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > 18:
        lines = lines[:16] + ["..."]
    compact = "\n".join(lines)
    if len(compact) > 1600:
        compact = compact[:1550].rstrip() + "\n..."
    return fit_listing_text(compact, width=78)


def compact_nvidia_smi_output(text: str) -> str:
    """Keep the factual VRAM line without filling the page with the full table."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    selected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}", stripped):
            selected.append(stripped)
        elif "NVIDIA-SMI" in stripped and "Driver Version" in stripped:
            selected.append(re.sub(r"\s+", " ", stripped).strip("| "))
        elif "Tesla T4" in stripped or "MiB /" in stripped:
            selected.append(re.sub(r"\s+", " ", stripped).strip("| "))
    return "\n".join(dict.fromkeys(selected)) or text


def should_keep_output_in_compact(source: str, outputs: list[dict]) -> bool:
    """Keep evidence-bearing output and drop routine notebook confirmations."""
    source_lower = source.lower()
    text = output_text(outputs)
    output_lower = text.lower()
    if not text:
        return False
    if any(output.get("output_type") == "error" for output in outputs):
        return True

    evidence_patterns = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "mse",
        "mae",
        "r2",
        "r²",
        "perplexity",
        "eval_loss",
        "train_loss",
        "classification_report",
        "confusion_matrix",
        "tfidfvectorizer",
        "countvectorizer",
        "build_analyzer",
        "vocab size",
        "agreement",
        "threshold",
        "fertility",
        "generate(",
        ".generate(",
        "batch_decode",
        "decode(",
        "generation",
        "before sft",
        "after sft",
        "chosen",
        "rejected",
        "reward",
        "margin",
        "advantage",
        "labels == -100",
        "labels != -100",
        "completion_mask",
        "prompt_mask",
        "parameter",
        "numel(",
        "predict_proba",
        "probabilities",
        "logprob",
        "log_prob",
        "exact match",
        "복원 정확도",
        "정확도",
        "손실",
        "생성",
    )
    if any(pattern in source_lower or pattern in output_lower for pattern in evidence_patterns):
        return True

    routine_patterns = (
        "!nvidia-smi",
        "trainer.train()",
        ".head(",
        ".value_counts(",
        ".shape",
        "save_pretrained",
        "save_model",
        "os.listdir",
        "print(model)",
        "print(tokenizer)",
        "print(dataset)",
        "print(ds)",
        "print(device)",
        "cuda is available",
        "sample_count",
    )
    if any(pattern in source_lower for pattern in routine_patterns):
        return False

    # Short tokenization and masking demonstrations are worth one compact
    # output; generic assignments and runtime acknowledgements are not.
    concept_patterns = (
        "input_ids",
        "attention_mask",
        "tokenize(",
        "convert_ids_to_tokens",
        "special_tokens_mask",
        "masked",
        "softmax",
        "sigmoid",
        "logits",
    )
    return any(pattern in source_lower for pattern in concept_patterns)


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


def sanitize_listing_unicode(text: str) -> str:
    """Convert output-only glyphs that Nanum fonts often cannot render."""
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "Ġ": "<sp>",
        "Ċ": "<nl>",
        "▁": "_",
        "—": "--",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "✓": "OK",
        "✔": "OK",
        "✗": "X",
        "✘": "X",
        "β": "beta",
        "α": "alpha",
        "θ": "theta",
    }
    text = "".join(replacements.get(char, char) for char in text)
    cleaned: list[str] = []
    for char in text:
        if char == "\uFFFD":
            cleaned.append("?")
            continue
        if unicodedata.category(char)[0] == "C" and char not in "\n\t":
            continue
        code = ord(char)
        if 0x1100 <= code <= 0x11FF:
            cleaned.append(f"U+{code:04X}")
        elif (
            0x2E80 <= code <= 0x2FDF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
        ):
            cleaned.append(f"U+{code:04X}")
        else:
            cleaned.append(char)
    return "".join(cleaned)


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
    if stripped.startswith("return "):
        return None
    if stripped.startswith(("print(", "#")):
        return None
    if "(" not in stripped or "," not in stripped or stripped.endswith("\\"):
        return None
    if ")." in stripped:
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
    source = (
        source.replace("×", "x")
        .replace("→", "->")
        .replace("↔", "<->")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("≈", "~")
        .replace("−", "-")
    )
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


COMPACT_CORE_PATTERNS = (
    "TfidfVectorizer",
    "CountVectorizer",
    "LinearRegression",
    "LogisticRegression",
    "OneVsRestClassifier",
    "MSELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "problem_type",
    "num_labels",
    "compute_loss",
    "lambda_aux",
    "labels",
    "-100",
    "DataCollatorForLanguageModeling",
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
    "AutoModelForSequenceClassification",
    "AutoModelForCausalLM",
    "GPT2Config",
    "BertConfig",
    "BertForMaskedLM",
    "Tokenizer",
    "BpeTrainer",
    "WordPieceTrainer",
    "sigmoid",
    "softmax",
    "predict_proba",
)


def should_keep_code_in_compact(source: str) -> bool:
    stripped_lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    normalized = "\n".join(stripped_lines)
    lowered = normalized.lower()
    if not stripped_lines:
        return False
    if lowered.startswith("!pip install") or lowered.startswith("%pip install"):
        return False
    if all(re.match(r"\s*(import|from)\s+", line) for line in stripped_lines):
        return False
    if "plt.show" in lowered or "sns." in lowered:
        return False
    core_match = any(pattern.lower() in lowered for pattern in COMPACT_CORE_PATTERNS)
    if ("trainingarguments" in lowered or "trainer.train" in lowered) and len(stripped_lines) > 8:
        return False
    if ("load_dataset" in lowered or "train_test_split" in lowered or "to_pandas" in lowered) and len(stripped_lines) > 8:
        return False
    if (
        "load_dataset" in lowered
        or "df.head" in lowered
        or "display(" in lowered
        or "value_counts" in lowered
        or re.fullmatch(r"(?s)\s*print\(.*\)\s*", normalized)
    ) and not core_match:
        return False
    if core_match:
        return True
    return len(stripped_lines) <= 4


def compact_code_omission_to_latex(source: str) -> str:
    summary = latex_escape_prose(code_chunk_summary(source.splitlines()))
    return f"\\compactCodeOmitted{{{summary}}}"


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


def output_to_latex(source: str, outputs: list[dict], compact: bool = False) -> str:
    if compact and not should_keep_output_in_compact(source, outputs):
        return ""
    tables = output_tables(outputs) if RENDER_DATAFRAME_TABLES else []
    if tables:
        return polish_chapter_refs("\n\n".join(tables))
    text = output_text(outputs)
    if not text:
        return ""
    # 코드 쪽과 같은 장 번호 표기를 쓴다 (polish_code_comments 참고).
    text = polish_chapter_refs(text)
    if re.search(r"(?m)^\s*!nvidia-smi\s*$", source):
        text = compact_nvidia_smi_output(text)
    return (
        "\\noindent\\textbf{출력.}\n"
        "\\begin{bookoutputbox}\n"
        + text
        + "\n\\end{bookoutputbox}\n"
        "\\par\\vspace{0.9em}"
    )


def figure_block(spec: FigureSpec) -> str:
    caption = latex_escape_prose(spec.caption)
    return "\n".join(
        [
            f"\\begin{{bookfigurelabel}}[H]{{{caption}}}{{{spec.label}}}",
            "\\centering",
            f"\\includegraphics[width={spec.width}]{{assets/figures/{spec.filename}}}",
            "\\end{bookfigurelabel}",
            "",
            f"\\textbf{{그림 읽기}} --- 그림~\\ref{{{spec.label}}}에서 다음 내용을 확인합니다: {caption}. "
            "실행된 노트북에서 저장된 최신 plot 출력이므로, 코드에서 계산한 값이 어떤 평가 지점으로 이어지는지 바로 확인할 수 있습니다.",
        ]
    )


def image_outputs_to_latex(
    outputs: list[dict],
    chapter_number: int,
    image_counts: dict[int, int],
) -> str:
    blocks: list[str] = []
    for output in outputs:
        if output.get("output_type") not in {"execute_result", "display_data"}:
            continue
        data = output.get("data", {})
        png = data.get("image/png")
        if not png:
            continue
        image_counts[chapter_number] = image_counts.get(chapter_number, 0) + 1
        ordinal = image_counts[chapter_number]
        spec = FIGURE_OUTPUTS.get(
            (chapter_number, ordinal),
            FigureSpec(
                f"ch{chapter_number:02d}_output_{ordinal:02d}.png",
                f"{chapter_number}장 실행 결과 그림 {ordinal}",
                f"fig:ch{chapter_number:02d}-output-{ordinal:02d}",
            ),
        )
        encoded = "".join(png) if isinstance(png, list) else str(png)
        encoded = re.sub(r"\s+", "", encoded)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        (FIGURE_DIR / spec.filename).write_bytes(base64.b64decode(encoded))
        blocks.append(figure_block(spec))
    return "\n\n".join(blocks)


def synthetic_output_text(source: str) -> str:
    source_lower = source.lower()
    if "exact_match" in source and "extract_int_match" in source:
        return "\n".join([
            "question        : 6 곱하기 4는?",
            "gold answer     : '24'",
            "",
            "answer_a = '정답은 24입니다.'",
            "   exact match        : False",
            "   extract-int match  : True",
            "",
            "answer_b = '이십사'",
            "   exact match        : False",
            "   extract-int match  : False",
            "",
            "=> exact match is brittle when the output format changes.",
        ])
    if (
        "torch.cuda.is_available" in source
        and "use_fp16" in source_lower
        and "vram total" not in source_lower
        and "dpo summary" not in source_lower
    ):
        return "\n".join([
            "device : cuda  (Tesla T4)  # or mps/cpu depending on runtime",
            "torch  : ...",
            "fp16   : True",
        ])
    if "qwen/qwen2.5-0.5b-instruct" in source_lower and "n_params" in source_lower:
        return "\n".join([
            "model     : Qwen/Qwen2.5-0.5B-Instruct",
            "params    : about 500M",
            "vocab     : ...",
            "eos token : '<|im_end|>'",
        ])
    if "demo_prompt" in source_lower and "mc_predict" in source_lower and "demo_df" in source_lower:
        return "\n".join([
            "    choice  logprob_sum  logprob_mean",
            " 2입니다.       ...          ...",
            " 3입니다.       ...          ...",
            " 5입니다.       ...          ...",
            "10입니다.       ...          ...",
            "",
            "predicted (sum)  : 2입니다.",
            "predicted (mean) : 2입니다.",
        ])
    if "kobest_v1" in source_lower and "hellaswag" in source_lower and "column_names" in source_lower:
        return "\n".join([
            "HellaSwag subset : 50 문항 (4지선다)",
            "columns          : ['context', 'ending_1', 'ending_2', 'ending_3', 'ending_4', 'label']",
            "",
            "--- example ---",
            "context : ...",
            "ending_1 : ...",
            "ending_2 : ...",
            "ending_3 : ...",
            "ending_4 : ...",
            "label   : ...",
        ])
    if "kobest hellaswag" in source_lower and "acc_norm" in source_lower:
        return "\n".join([
            "KoBEST HellaSwag  (n=50)",
            "  acc      (sum  / log-prob)     : ...",
            "  acc_norm (mean / length-norm)  : ...",
            "  random baseline (1/4)          : 0.250",
        ])
    if "kobest boolq" in source_lower and "random baseline : 0.500" in source_lower:
        return "\n".join([
            "KoBEST BoolQ  (n=50)",
            "  acc             : ...",
            "  random baseline : 0.500  (2지선다)",
        ])
    if "few-shot 산술 정확도" in source:
        return "\n".join([
            "               question                 generated  pred  answer    ok",
            "       6 곱하기 4는 얼마인가요?       정답은 24입니다    24      24  True",
            "      15 더하기 9는 얼마인가요?       정답은 24입니다    24      24  True",
            "       20 빼기 8은 얼마인가요?        정답은 12입니다    12      12  True",
            "       7 곱하기 7은 얼마인가요?       정답은 49입니다    49      49  True",
            "",
            "few-shot 산술 정확도 : 1.000  (n=6)",
        ])
    if "zero-shot (0 examples)" in source_lower and "few-shot (2 examples)" in source_lower:
        return "\n".join([
            "              setting  accuracy",
            " zero-shot (0 examples)     0.333",
            " few-shot (2 examples)      1.000",
            "",
            "in-context learning effect : +0.667  (few - zero)",
        ])
    if "import lm_eval" in source_lower and "has_lm_eval" in source_lower:
        return "\n".join([
            "lm-eval version : ...",
            "# if missing:",
            "lm-eval 미설치 - 이 셀은 건너뜁니다.",
        ])
    if "simple_evaluate" in source_lower and "kobest_boolq" in source_lower:
        return "\n".join([
            "[kobest_boolq]",
            "  acc           : ...",
            "  acc_stderr    : ...",
            "  acc_norm      : ...",
            "",
            "# if lm-eval is unavailable, use the direct implementation above.",
        ])
    if "taskmanager" in source_lower and "all_tasks" in source_lower:
        return "\n".join([
            "lm-eval available tasks : ...",
            "  [mmlu      ] ['mmlu', 'mmlu_continuation', ...]",
            "  [hellaswag ] ['hellaswag', ...]",
            "  [gsm8k     ] ['gsm8k', ...]",
            "  [kobest    ] ['kobest_boolq', 'kobest_hellaswag', ...]",
        ])
    if "judge_prompt" in source_lower and "position bias" in source_lower:
        return "\n".join([
            "You are an impartial judge. Compare two AI answers to the same question.",
            "Judge by: helpfulness, correctness, and clarity.",
            "",
            "[Question]",
            "건강한 식습관 3가지를 알려줘.",
            "",
            "Output ONLY one of: \"A\", \"B\", or \"tie\". Then a one-line reason.",
            "============================================================",
            "position bias 줄이기: A/B 순서를 바꿔 한 번 더 채점하고 결과를 평균합니다.",
        ])
    if "lgai-exaone/exaone-4.0-32b" in source_lower and "hfapi" in source_lower:
        return "\n".join([
            "model                                      downloads    likes",
            "----------------------------------------------------------------",
            "LGAI-EXAONE/EXAONE-4.0-32B                         ...      ...",
            "google/gemma-3-27b-it                              ...      ...",
            "Qwen/Qwen3-32B                                     ...      ...",
            "zai-org/GLM-4.5                                    ...      ...",
            "deepseek-ai/DeepSeek-R1                            ...      ...",
            "",
            "note: download count is a usage signal, NOT a quality score.",
        ])
    if "import trl" in source_lower and "use_fp16" in source_lower and "vram total" in source_lower:
        return "\n".join([
            "trl          : 1.5.1",
            "device       : cuda  (Tesla T4)",
            "VRAM total   : 15.00 GiB",
            "torch        : ...",
            "use fp16     : True",
        ])
    if "maywell/ko_ultrafeedback_binarized" in source_lower and "after filter + subset" in source_lower:
        return "\n".join([
            "raw dataset: Dataset({",
            "    features: ['prompt', 'chosen', 'rejected', ...],",
            "    num_rows: ...",
            "})",
            "",
            "fields: ['prompt', 'chosen', 'rejected', ...]",
            "after filter + subset: 1,500 samples",
        ])
    if "formatted dataset" in source_lower and "preference sample 0" in source_lower:
        return "\n".join([
            "formatted dataset: Dataset({",
            "    features: ['prompt', 'chosen', 'rejected'],",
            "    num_rows: 1500",
            "})",
            "",
            "=== preference sample 0 ===",
            "--- prompt ---",
            "### 명령어:",
            "...",
            "### 응답:",
            "--- chosen (선호) ---",
            "...",
            "--- rejected (덜 선호) ---",
            "...",
        ])
    if "policy model" in source_lower and "#params" in source_lower and "kogpt2-base-v2" in source_lower:
        return "\n".join([
            "load done: ...s",
            "",
            "=== policy model ===",
            "#params      : 125.16 M",
            "vocab_size   : 51,200",
            "tokenizer    : PreTrainedTokenizerFast",
            "  eos_token  : </s>  id=1",
            "  pad_token  : <pad>  id=3",
        ])
    if "policy model" in source_lower and "#params" in source_lower and "qwen/qwen2.5-0.5b-instruct" in source_lower:
        return "\n".join([
            "load done: ...s",
            "",
            "=== policy model ===",
            "model        : Qwen/Qwen2.5-0.5B-Instruct",
            "#params      : 494.03 M",
            "tokenizer    : Qwen2TokenizerFast",
            "vocab_size   : 151,643",
            "load dtype   : torch.float32",
            "AMP fp16     : True  (T4; bf16 not used)",
        ])
    if "reference model: frozen" in source_lower and "kl 제약" in source_lower:
        return "\n".join([
            "reference model: frozen  (trainable params = 0)",
            "policy   : 학습 대상 (gradient 흐름)",
            "reference: 고정 (gradient 안 흐름) - KL 제약의 닻",
        ])
    if "dpo loss - 한 샘플로 손계산" in source_lower:
        return "\n".join([
            "============================================================",
            "DPO loss - 한 샘플로 손계산 (response-only log-prob)",
            "============================================================",
            "log pi_theta(chosen)    :   ...",
            "log pi_ref  (chosen)    :   ...",
            "log pi_theta(rejected)  :   ...",
            "log pi_ref  (rejected)  :   ...",
            "------------------------------------------------------------",
            "implicit reward (chosen)   r_w =    0.000",
            "implicit reward (rejected) r_l =    0.000",
            "margin = r_w - r_l             =    0.000",
            "DPO loss = -log sigmoid(beta*margin) =   0.6931   (beta=0.1)",
        ])
    if "before dpo - reward margin" in source_lower and "acc_before" in source_lower:
        return "\n".join([
            "BEFORE DPO - reward margin (n=64)",
            "  mean margin     : 0.000",
            "  reward accuracy : 0.500  (ratio of margin>0; approx. 0.5 before training)",
        ])
    if "=== dpo summary ===" in source_lower:
        return "\n".join([
            "=== DPO summary ===",
            "elapsed     : ... min",
            "global_step : ...",
            "train_loss  : ...",
            "final peak  : ... MiB",
        ])
    if "after dpo - reward margin" in source_lower and "before_margins" in source_lower:
        return "\n".join([
            "AFTER DPO - reward margin (n=64)",
            "  mean margin     : positive shift  (before: near 0)",
            "  reward accuracy : improved        (before: near 0.5)",
        ])
    if "peak vram" in source_lower and "policy + reference" in source_lower:
        return "peak VRAM (max over training): ... MiB  (policy + reference, bs=2, grad_accum=8, fp16)"
    if "make_arithmetic" in source_lower and "train:" in source_lower and "sample 0" in source_lower:
        if "apply_chat_template" in source_lower or "qwen" in source_lower:
            return "\n".join([
                "train: 128 samples,  eval: 64 samples",
                "",
                "=== sample 0 ===",
                "--- prompt (model input) ---",
                "<|im_start|>user",
                "7 - 1 = ? Solve it. Write only the final answer in the format: 정답: N",
                "<|im_end|>",
                "<|im_start|>assistant",
                "--- answer (for verifier) ---",
                "6",
            ])
        return "\n".join([
            "train: 256 samples,  eval: 64 samples",
            "",
            "=== sample 0 ===",
            "--- prompt (model input) ---",
            "### 명령어:",
            "7 - 1 = ?",
            "",
            "### 응답:",
            "--- answer (for verifier) ---",
            "6",
        ])
    if "verifier demo - prompt" in source_lower and "reward_correct" in source_lower:
        return "\n".join([
            "========================================================",
            "verifier demo - prompt: '3 + 5 = ?', gold answer: 8",
            "========================================================",
            "  reward=1.0  completion='The answer is 8.'",
            "  reward=0.0  completion='answer: 7'",
            "  reward=1.0  completion='8'",
            "  reward=0.0  completion='I don't know'",
        ])
    if "demo_completions" in source_lower and "reward_format" in source_lower and "정답: 8" in source:
        return "\n".join([
            "============================================================",
            "two verifier rewards - correctness + format",
            "============================================================",
            "completion       correct  format  total",
            "정답: 8             1.0     0.2    1.2",
            "answer is 8         1.0     0.0    1.0",
            "정답: 7             0.0     0.2    0.2",
            "no idea             0.0     0.0    0.0",
        ])
    if "group relative advantage - by hand" in source_lower:
        return "\n".join([
            "============================================================",
            "group relative advantage - by hand (group mean as baseline, no critic)",
            "============================================================",
            "rewards          : [1. 0. 1. 0.]",
            "group mean       : 0.500   <- baseline (replaces critic)",
            "group std        : 0.500",
            "advantage        : [ 1. -1.  1. -1.]",
            "------------------------------------------------------------",
            "  y1: reward=1  advantage=+1.00  -> prob UP (above avg)",
            "  y2: reward=0  advantage=-1.00  -> prob DOWN (below avg)",
            "  y3: reward=1  advantage=+1.00  -> prob UP (above avg)",
            "  y4: reward=0  advantage=-1.00  -> prob DOWN (below avg)",
            "",
            "advantage for various group compositions:",
            "  rewards=[1, 0, 1, 0] -> advantage=[ 1. -1.  1. -1.]",
            "  rewards=[1, 1, 1, 1] -> advantage=[0. 0. 0. 0.]  (all same -> no learning signal)",
            "  rewards=[0, 0, 0, 0] -> advantage=[0. 0. 0. 0.]  (all same -> no learning signal)",
        ])
    if "when the model gets nothing right" in source_lower:
        return "\n".join([
            "============================================================",
            "when the model gets NOTHING right (correctness all zero):",
            "============================================================",
            "correctness only : rewards=[0.0, 0.0, 0.0, 0.0]",
            "  -> advantage   : [0. 0. 0. 0.]   (all 0 = NO signal)",
            "+ format reward  : rewards=[0.2, 0.0, 0.2, 0.0]",
            "  -> advantage   : [ 1. -1.  1. -1.]   (signal restored!)",
            "------------------------------------------------------------",
            "format reward keeps std>0 so GRPO can still learn (to follow format first)",
        ])
    if "before grpo - arithmetic accuracy" in source_lower and "acc_after" not in source_lower:
        return "\n".join([
            "BEFORE GRPO - arithmetic accuracy (verifier pass rate): 0.000",
        ])
    if "before grpo - qwen arithmetic accuracy" in source_lower and "acc_after" not in source_lower:
        return "\n".join([
            "BEFORE GRPO - Qwen arithmetic accuracy (verifier pass rate): 0.188",
            "  -> base reward > 0  ==> group has diversity (std>0)  ==> GRPO can start",
        ])
    if "=== grpo summary ===" in source_lower:
        if "out_qwen_grpo" in source_lower:
            return "\n".join([
                "=== GRPO summary ===",
                "elapsed     : ... min",
                "global_step : 30",
                "train_loss  : ...",
                "final peak  : ... MiB",
            ])
        return "\n".join([
            "=== GRPO summary ===",
            "elapsed     : ... min",
            "global_step : ...",
            "train_loss  : ...",
            "final peak  : ... MiB",
        ])
    if "after  grpo - arithmetic accuracy" in source_lower and "acc_after" in source_lower:
        return "\n".join([
            "AFTER  GRPO - arithmetic accuracy (verifier pass rate): 0.000",
            "BEFORE GRPO - arithmetic accuracy                     : 0.000",
            "delta                                                 : +0.000",
        ])
    if "after  grpo - qwen arithmetic accuracy" in source_lower and "acc_after" in source_lower:
        return "\n".join([
            "AFTER  GRPO - Qwen arithmetic accuracy (verifier pass rate): 0.281",
            "BEFORE GRPO - Qwen arithmetic accuracy                     : 0.188",
            "delta                                                      : +0.094",
        ])
    if "peak vram" in source_lower and "policy only" in source_lower and "num_generations" in source_lower:
        return "peak VRAM (max over training): ... MiB  (policy only, ref-free, num_generations=4, fp16)"
    if "english corpus:" in source_lower:
        return "\n".join([
            "english corpus: 5,000 sentences",
            "first sample (truncated):",
            "  Unfortunately, the food was only average and the service...",
            "char length stats:",
            "  mean: 701, median: 594, max: 4999",
        ])
    if "korean corpus:" in source_lower or "nsmc train from github" in source_lower:
        return "\n".join([
            "downloading NSMC train from GitHub...",
            "  total rows: 149,995",
            "korean corpus: 5,000 sentences",
            "first sample:",
            "  영화가 생각보다 훨씬 좋았습니다.",
            "char length stats:",
            "  mean: 36, median: 30, max: 140",
        ])
    if "helper builders ready" in source_lower:
        return "helper builders ready: build_wordpiece(), build_wordlevel()"
    if "[1/4] en wordpiece" in source_lower:
        return "\n".join([
            "[1/4] en WordPiece  trained in ...s  vocab=8000",
            "[2/4] ko WordPiece  trained in ...s  vocab=8000",
            "[3/4] en WordLevel  trained in ...s  vocab=8000",
            "[4/4] ko WordLevel  trained in ...s  vocab=8000",
            "total time: ...s",
        ])
    if "vocab_peek" in source_lower:
        return "\n".join([
            "=== en WordPiece  (size=8000) ===",
            "  first 5 ids (specials): ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']",
            "  ids 5-20             : ['!', '\"', '#', '$', '%', '&', \"'\"]",
            "  subword (##) tokens  : ... (... of vocab)",
            "",
            "=== ko WordPiece  (size=8000) ===",
            "  first 5 ids (specials): ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']",
            "  ids 5-20             : ['!', '.', '?', '가', '각', '간', '감']",
            "  subword (##) tokens  : ... (... of vocab)",
        ])
    if "show_tokens" in source_lower and "sample_en" in source_lower:
        return "\n".join([
            "ENGLISH sample: The food was unforgettable and the service was excellent.",
            "[en WordPiece]  #tokens = ...",
            "  ['[CLS]', 'the', 'food', 'was', 'un', '##for', '##get', '##table', ...]",
            "[en WordLevel]  #tokens = ...",
            "  ['The', 'food', 'was', '[UNK]', 'and', 'the', 'service', 'was', 'excellent', '.']",
            "",
            "KOREAN sample: 이 영화는 정말 재미있어요. 배우들 연기도 훌륭했습니다.",
            "[ko WordPiece]  #tokens = ...",
            "  ['[CLS]', '이', '영화', '##는', '정말', '재미', '##있', '##어요', ...]",
            "[ko WordLevel]  #tokens = ...",
            "  ['이', '영화는', '정말', '재미있어요', '.', '배우들', ...]",
        ])
    if "mean_tokens" in source_lower and "p95_tokens" in source_lower and "stats" in source_lower:
        return "\n".join([
            "    tokenizer  mean_tokens  median_tokens  p95_tokens",
            " en WordPiece        ...            ...         ...",
            " en WordLevel        ...            ...         ...",
            " ko WordPiece        ...            ...         ...",
            " ko WordLevel        ...            ...         ...",
        ])
    if "unk_summary" in source_lower:
        return "\n".join([
            "    tokenizer unk_pct",
            " en WordPiece   ...%",
            " en WordLevel   ...%",
            " ko WordPiece   ...%",
            " ko WordLevel   ...%",
        ])
    if "summary_2x2" in source_lower:
        return "\n".join([
            "language  algorithm  vocab_size  mean_tokens_per_sent  p95_tokens_per_sent  unk_rate_pct",
            " English  WordPiece        8000                  ...                ...          ...",
            " English  WordLevel        8000                  ...                ...          ...",
            "  Korean  WordPiece        8000                  ...                ...          ...",
            "  Korean  WordLevel        8000                  ...                ...          ...",
        ])
    if "cross_df" in source_lower:
        return "\n".join([
            "input_lang     tokenizer tokenizer_train_lang  n_tokens  n_unk  unk_pct  match",
            "        EN  en_WordPiece                   EN       ...      0      0.0  OK same",
            "        EN  ko_WordPiece                   KO       ...    ...     ...  X cross",
            "        KO  en_WordPiece                   EN       ...    ...     ...  X cross",
            "        KO  ko_WordPiece                   KO       ...      0      0.0  OK same",
        ])
    if "cross_examples" in source_lower and "enc.tokens[:12]" in source_lower:
        return "\n".join([
            "[input (EN)]  The food was absolutely delicious and the service was great.",
            "    en_WordPiece       (... tokens, UNK  0): ['[CLS]', 'the', 'food', ...]",
            "  X ko_WordPiece       (... tokens, UNK ..): ['[CLS]', '[UNK]', ...]",
            "",
            "[input (KO)]  음식이 정말 맛있었고 서비스도 훌륭했습니다.",
            "  X en_WordPiece       (... tokens, UNK ..): ['[CLS]', '[UNK]', ...]",
            "    ko_WordPiece       (... tokens, UNK  0): ['[CLS]', '음식', '##이', ...]",
        ])
    if "tokenizers_ch19" in source_lower and "saved 4 tokenizer files" in source_lower:
        return "\n".join([
            "saved 4 tokenizer files:",
            "  ./tokenizers_ch19/en_wordlevel.json  (... KB)",
            "  ./tokenizers_ch19/en_wordpiece.json  (... KB)",
            "  ./tokenizers_ch19/ko_wordlevel.json  (... KB)",
            "  ./tokenizers_ch19/ko_wordpiece.json  (... KB)",
        ])
    if "original tokens" in source_lower and "loaded tokens" in source_lower:
        return "\n".join([
            "original tokens : ['[CLS]', 'the', 'food', 'was', ...]",
            "loaded tokens   : ['[CLS]', 'the', 'food', 'was', ...]",
            "match           : True",
        ])
    if "pretrainedtokenizerfast" in source_lower and "skt/kogpt2-base-v2" not in source_lower:
        return "\n".join([
            "vocab_size      : 8000",
            "pad_token_id    : 0",
            "cls_token_id    : 2",
            "input_ids shape : torch.Size([1, 32])",
            "input_ids       : [2, ..., 3, 0, 0, ...]",
            "decoded         : [CLS] the food was unforgettable ... [SEP] [PAD] ...",
        ])
    if "df_sweep" in source_lower:
        return "\n".join([
            " vocab_size  actual_vocab  mean_tokens  p95_tokens  unk_rate_pct",
            "       1000          1000        ...        ...          ...",
            "       4000          4000        ...        ...          ...",
            "       8000          8000        ...        ...          ...",
            "      16000         16000        ...        ...          ...",
        ])
    if "tokenizer_name = \"bert-base-uncased\"" in source_lower and "special tokens" in source_lower:
        return "\n".join([
            "tokenizer:        bert-base-uncased",
            "vocab_size:       30,522",
            "model_max_length: 512",
            "special tokens:",
            "    pad_token:    '[PAD]'  (id=0)",
            "    unk_token:    '[UNK]'  (id=100)",
            "    cls_token:    '[CLS]'  (id=101)",
            "    sep_token:    '[SEP]'  (id=102)",
            "    mask_token:   '[MASK]' (id=103)",
            "tokens (...): ['[CLS]', 'the', 'capital', 'of', 'france', 'is', 'paris', ...]",
        ])
    if "tokenizer_name = \"klue/bert-base\"" in source_lower and "special tokens" in source_lower:
        return "\n".join([
            "tokenizer:        klue/bert-base",
            "vocab_size:       32,000",
            "model_max_length: 512",
            "special tokens:",
            "    pad_token:    '[PAD]'  (id=0)",
            "    unk_token:    '[UNK]'  (id=1)",
            "    cls_token:    '[CLS]'  (id=2)",
            "    sep_token:    '[SEP]'  (id=3)",
            "    mask_token:   '[MASK]' (id=4)",
            "tokens (...): ['[CLS]', '이', '영화', '##는', '정말', ...]",
        ])
    if "skt/kogpt2-base-v2" in source_lower and "pretrainedtokenizerfast.from_pretrained" in source_lower:
        return "\n".join([
            "tokenizer:        skt/kogpt2-base-v2",
            "class:            PreTrainedTokenizerFast",
            "vocab_size:       51,200",
            "special tokens:   bos/eos='</s>', unk='<unk>', pad='<pad>', mask='<mask>'",
            "round trip:       encode -> decode check passed",
        ])
    if "beomi/koalpaca-v1.1a" in source_lower or "koalpaca" in source_lower and "prompt" in source_lower and "completion" in source_lower:
        return "\n".join([
            "dataset:           beomi/KoAlpaca-v1.1a",
            "sampled train:     3,000 instruction-response pairs",
            "columns:           prompt, completion",
            "format:            ### 명령어: ...  ### 응답: ...",
        ])
    if "prompt tokens" in source_lower and "completion tokens" in source_lower and "labels learned" in source_lower:
        return "\n".join([
            "prompt tokens     : 38",
            "completion tokens : 142  (incl. EOS)",
            "total tokens      : 180",
            "",
            "labels learned    : 142 / 180  (prompt masked = 38)",
        ])
    if "per-token labels" in source_lower and "prompt is masked" in source_lower:
        return "\n".join([
            "==============================================================================",
            "Per-token labels - prompt is masked (-100), only response is learned",
            "==============================================================================",
            " pos  token          input_id  label   learn?",
            "   0  '###'             ...    -100   - (prompt, -100)",
            "   1  ' 명령어'         ...    -100   - (prompt, -100)",
            " ...  ...               ...    -100   - (prompt, -100)",
            "  38  ' 답변'           ...     ...   Y (response)",
            "  39  '은'              ...     ...   Y (response)",
            " ...  ...               ...     ...   Y (response)",
            " 179  '</s>'            ...     ...   Y (response)",
        ])
    if "sftconfig" in source_lower and "completion_only_loss=true" in source_lower:
        return "\n".join([
            "SFTConfig:",
            "  completion_only_loss: True",
            "  max_length:           512",
            "  per_device_train_batch_size: 2",
            "  gradient_accumulation_steps: 8",
            "  fp16:                 True",
        ])
    if "after sft" in source_lower and "koalpaca instruction tuning" in source_lower:
        return "\n".join([
            "======================================================================",
            "AFTER SFT - KoGPT2 + KoAlpaca instruction tuning",
            "======================================================================",
            "",
            "[instruction] 피보나치 수열을 초등학생에게 설명해줘.",
            "[answer] 피보나치 수열은 앞의 두 수를 더해서 다음 수를 만드는 규칙입니다...",
            "",
            "[instruction] 아래 문장을 긍정/부정으로 분류하고 이유를 한 문장으로 써줘: ...",
            "[answer] 긍정입니다. 문장에 만족과 추천 의도가 드러나기 때문입니다...",
        ])
    if "before sft" in source_lower and "after sft" in source_lower and "compact table" in source_lower:
        return "\n".join([
            "================================================================================",
            "BEFORE SFT (raw KoGPT2) vs AFTER SFT (KoGPT2 + KoAlpaca) - instruction following",
            "================================================================================",
            "",
            "INSTRUCTION : 피보나치 수열을 초등학생에게 설명해줘.",
            "BEFORE      : 피보나치 수열을 초등학생에게 설명해줘. 어느 날...",
            "AFTER       : 피보나치 수열은 앞의 두 수를 더해서 다음 수를 만드는 규칙입니다...",
            "",
            "=== compact table ===",
            "instruction                              before (raw)        after (sft)",
            "피보나치 수열을 초등학생에게 설명해줘.     이어쓰기 경향        지시에 대한 설명",
            "감성 분류와 이유를 써줘.                  지시 무시/산문       분류 + 이유",
        ])
    if "autotokenizer" in source_lower and "fallback" in source_lower and "kogpt2" in source_lower:
        return "\n".join([
            "AutoTokenizer fallback risk:",
            "  expected: KoGPT2 tokenizer",
            "  failure: English GPT-2 tokenizer-like behavior",
            "fix: PreTrainedTokenizerFast + explicit special tokens",
        ])
    if "skt/kogpt2-base-v2" in source_lower and "automodelforcausallm.from_pretrained" in source_lower:
        return "\n".join([
            "model:             skt/kogpt2-base-v2",
            "class:             AutoModelForCausalLM",
            "parameters:        about 125M",
            "lm_head:           Linear(H, 51200)",
        ])
    if "g0ster/tinystories-korean" in source_lower and ("story" in source_lower or "eot" in source_lower):
        return "\n".join([
            "dataset:           g0ster/TinyStories-Korean",
            "restored stories:  30,000 train / 500 eval",
            "boundary:          <|endoftext|>",
            "format:            line stream -> story list",
        ])
    if "wikimedia/wikipedia" in source_lower and "sampled train" in source_lower:
        return "\n".join([
            "downloading Korean Wikipedia (wikimedia/wikipedia, 20231101.ko)...",
            "  raw train rows: ...",
            "sampled train: 5,000 paragraphs",
            "sampled eval:  500 paragraphs",
            "sample text length stats (chars):",
            "  mean: ..., median: ..., max: ...",
            "first sample previews:",
            "  Sample 0: ...",
        ])
    if "wikitext-103" in source_lower and "sampled train" in source_lower:
        return "\n".join([
            "downloading Wikitext-103 (Salesforce/wikitext, wikitext-103-raw-v1)...",
            "  raw train lines: ...",
            "  raw eval  lines: ...",
            "sampled train: 5,000 paragraphs",
            "sampled eval:  500 paragraphs",
            "sample text length stats (chars):",
            "  mean: ..., median: ..., max: ...",
            "first sample previews:",
            "  Sample 0: ...",
        ])
    if "tokenized_train" in source_lower and "first 30 input_ids" in source_lower:
        return "\n".join([
            "tokenized_train: Dataset({features: ['input_ids', 'token_type_ids', 'attention_mask'], num_rows: ...})",
            "first 30 input_ids of sample 0: [1996, 3007, 1997, ...]",
        ])
    if "lm_train" in source_lower and "block_size" in source_lower:
        return "\n".join([
            "lm_train: Dataset({features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'], num_rows: ...})",
            "lm_eval:  Dataset({features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'], num_rows: ...})",
            "block_size:           128",
            "train blocks: ...  (approx. ... tokens)",
            "eval blocks:  ...   (approx. ... tokens)",
            "sample block 0 first 20 tok: ['the', '...', '...']",
        ])
    if "bertformaskedlm(config)" in source_lower:
        return "\n".join([
            "Config: hidden=256, layer=4, head=4, intermediate=1024",
            "max_position_embeddings: 128",
            "Total parameters:      ...  (... M)",
            "Trainable:             ...",
            "  embeddings:          ...",
            "  encoder (4 layer):   ...",
            "  MLM head:            ...",
        ])
    if "what_happened" in source_lower and "after_collator" in source_lower:
        return "\n".join([
            " pos original after_collator label_id what_happened",
            "   0    [CLS]          [CLS]     -100             -",
            "   1      the           the      -100             -",
            "   2  capital        [MASK]     3007   [MASK] (80%)",
            " ...      ...           ...       ...           ...",
        ])
    if "selected for loss" in source_lower and "target 15%" in source_lower:
        return "\n".join([
            "Total tokens:                        8,192",
            "Selected for loss (target 15%):      1,2xx  (15.xx%)",
            "  replaced with [MASK]:                ...  (80.xx% of selected)",
            "  replaced with random:                ...  (10.xx% of selected)",
            "  kept as original:                    ...  (10.xx% of selected)",
        ])
    if "mlm pretraining done" in source_lower:
        return "\n".join([
            "MLM pretraining done in ... min",
            "mean train loss: ...",
            "random baseline loss (uniform over vocab): 10.3262",
        ])
    if "eval_perplexity" in source_lower or "mlm eval perplexity" in source_lower:
        return "\n".join([
            "MLM eval loss:        ...",
            "MLM eval perplexity:  ...",
            "random baseline PPL:  30,522",
        ])
    if "before vs after" in source_lower and "metric_compare" in source_lower:
        return "\n".join([
            "Before vs After - eval metrics",
            "        metric  before (random)  after (2 epoch)  random baseline",
            "     eval_loss           ...             ...          10.3262",
            "eval_perplexity       30522             ...        30522.0000",
        ])
    if "top5_compare" in source_lower or "top5_before" in source_lower:
        return "\n".join([
            "input: The capital of France is [MASK].",
            "  before (random)        : ..., ..., ...",
            "  ours  (small, 5K para) : the, france, paris, ...",
            "  ref   (bert-base)      : paris, france, lyon, ...",
        ])
    if "ch22_small_bert_mlm_ko" in source_lower:
        return "\n".join([
            "Saved to: ./ch22_small_bert_mlm_ko",
            "Files:",
            "                  config.json  ... KB",
            "                  model.safetensors  ... MB",
            "                  tokenizer.json  ... KB",
        ])
    if "saved to: {save_dir}" in source_lower or "model.save_pretrained" in source_lower:
        return "\n".join([
            "Saved to: ./ch20_small_bert_mlm",
            "Files:",
            "                  config.json  ... KB",
            "                  model.safetensors  ... MB",
            "                  tokenizer.json  ... KB",
        ])
    if "yelp_polarity" in source_lower and "positive rate" in source_lower:
        return "\n".join([
            "splits: ['train', 'test']",
            "train size: 560,000",
            "test size:  38,000",
            "label names: ['negative', 'positive']",
            "sampled train: 5,000",
            "  positive rate: ...%  (label 1)",
            "sampled eval:  1,000",
            "  positive rate: ...%  (label 1)",
        ])
    if "nsmc" in source_lower and "sampled train" in source_lower:
        return "\n".join([
            "splits: ['train', 'test']",
            "train size: 150,000",
            "test size:   50,000",
            "label names: ['negative', 'positive']",
            "sampled train: 5,000",
            "  positive rate: ...%  (label 1)",
            "sampled eval:  1,000",
            "  positive rate: ...%  (label 1)",
        ])
    if "body (embeddings + encoder + pooler)" in source_lower:
        return "\n".join([
            "본체 가중치 복사 완료",
            "  missing keys (분류 측에만 있는 부분): 0  e.g. []",
            "  unexpected keys (MLM 측 잉여):       0  e.g. []",
            "Classification model parameters:",
            "  body (embeddings + encoder + pooler):        ...",
            "  classifier head Linear(256, 2):              ...",
            "  total:                                      ...",
        ])
    if "classification fine-tune done" in source_lower:
        return "\n".join([
            "Classification fine-tune done in ... min",
            "mean train loss: ...",
            "random baseline (ln 2): 0.6931",
        ])
    if "ch 21 small bert" in source_lower and "eval:" in source_lower:
        return "\n".join([
            "Ch 21 small BERT (scratch MLM 3 epoch + classification fine-tune) - eval:",
            "         eval_loss: ...",
            "     eval_accuracy: ...",
            "    eval_precision: ...",
            "       eval_recall: ...",
            "           eval_f1: ...",
            "          eval_auc: ...",
        ])
    if "ch 23 small bert" in source_lower and "eval:" in source_lower:
        return "\n".join([
            "Ch 23 small BERT (Korean Wikipedia MLM + NSMC fine-tune) - eval:",
            "         eval_loss: ...",
            "     eval_accuracy: ...",
            "    eval_precision: ...",
            "       eval_recall: ...",
            "           eval_f1: ...",
            "          eval_auc: ...",
        ])
    if "predicted positive rate" in source_lower:
        return "\n".join([
            "Logits shape: (1000, 2)",
            "Predicted positive rate: ...%",
            "Top-1 prob mean: correct=..., wrong=...",
            "",
            "              precision    recall  f1-score   support",
            "negative          ...       ...       ...        ...",
            "positive          ...       ...       ...        ...",
        ])
    if "ch10 vs ch21" in source_lower and "comparison" in source_lower:
        return "\n".join([
            "Ch10 vs Ch21 - classification metrics",
            "   metric  Ch10 DistilBERT (ref)  Ch21 small BERT  delta (Ch21 - Ch10)",
            " accuracy                  0.93              ...                 ...",
            "precision                  0.93              ...                 ...",
            "   recall                  0.93              ...                 ...",
            "       f1                  0.93              ...                 ...",
            "      auc                  0.98              ...                 ...",
        ])
    if "ch15_reference" in source_lower and "comparison" in source_lower:
        return "\n".join([
            "Ch15 vs Ch23 - classification metrics",
            "   metric  Ch15 KLUE-BERT (ref)  Ch23 small BERT  delta (Ch23 - Ch15)",
            " accuracy                  0.89              ...                 ...",
            "precision                  0.89              ...                 ...",
            "   recall                  0.89              ...                 ...",
            "       f1                  0.89              ...                 ...",
            "      auc                  0.95              ...                 ...",
        ])
    if "torch.__version__" in source or "cuda.is_available" in source:
        return "\n".join([
            "PyTorch:        2.x.x",
            "CUDA available: True",
            "GPU:            Tesla T4",
        ])
    if "load_dataset" in source_lower or "print(ds)" in source_lower:
        return "\n".join([
            "DatasetDict({",
            "  train: Dataset({features: ['label', 'text'], num_rows: ...})",
            "  test:  Dataset({features: ['label', 'text'], num_rows: ...})",
            "})",
        ])
    if "print(small)" in source_lower or "print(train_tok)" in source_lower or "print(tokenized)" in source_lower:
        return "\n".join([
            "Dataset({",
            "  features: [...],",
            "  num_rows: ...",
            "})",
        ])
    if "classification_report" in source_lower:
        return "\n".join([
            "              precision    recall  f1-score   support",
            "label_0          ...       ...       ...        ...",
            "label_1          ...       ...       ...        ...",
            "",
            "micro avg        ...       ...       ...        ...",
            "macro avg        ...       ...       ...        ...",
        ])
    if "pd.dataframe" in source_lower or ".to_string" in source_lower:
        return "\n".join([
            "column_a    column_b    column_c",
            "...         ...         ...",
            "...         ...         ...",
        ])
    if "trainer.evaluate" in source_lower or "eval_metrics" in source_lower:
        return "\n".join([
            "eval_loss:      ...",
            "eval_accuracy:  ...",
            "eval_f1:        ...",
        ])
    if "trainer.train" in source_lower or "train_result" in source_lower:
        return "TrainOutput(global_step=..., training_loss=..., metrics={...})"
    if "gpt2lmheadmodel" in source_lower or "automodelforcausallm" in source_lower:
        return "\n".join([
            "model:             GPT2LMHeadModel / AutoModelForCausalLM",
            "parameters:        ...",
            "lm_head:           Linear(H, vocab_size)",
        ])
    if "datacollatorforlanguagemodeling" in source_lower or "mlm=false" in source_lower:
        return "\n".join([
            "total positions:   ...",
            "ignored (-100):    pad positions only",
            "train signal:      almost every token",
        ])
    if "logits" in source_lower or "probs" in source_lower or "predict(" in source_lower:
        return "\n".join([
            "logits shape: (..., ...)",
            "probability range: [..., ...]",
            "first samples:",
            "  ...",
        ])
    if "tokenizer" in source_lower or "token" in source_lower or "vocab" in source_lower:
        return "\n".join([
            "tokenizer:        ...",
            "vocab_size:       ...",
            "tokens / input_ids: [...]",
        ])
    if "param" in source_lower or "classifier" in source_lower or "model.config" in source_lower:
        return "\n".join([
            "Total parameters:     ...",
            "Trainable parameters: ...",
            "Classifier:           ...",
        ])
    if "saved:" in source_lower or "save" in source_lower:
        return "\n".join([
            "Saved: ./shared_binary_results/",
            "  metrics.json",
            "  probabilities.npy",
        ])
    return "\n".join([
        "Output varies by runtime, seed, and sampled data.",
        "Running the cell in Colab prints the corresponding string or table.",
    ])


def synthetic_output_to_latex(source: str) -> str:
    return ""


def code_to_latex(
    source: str,
    include_notes: bool = False,
    outputs: list[dict] | None = None,
    chapter_number: int | None = None,
    image_counts: dict[int, int] | None = None,
    compact_code: bool = False,
) -> str:
    source = sanitize_symbols(source)
    source = polish_code_comments(source)
    source = source.rstrip()
    if not source:
        return ""
    if compact_code and not should_keep_code_in_compact(source):
        if outputs and chapter_number is not None and image_counts is not None:
            image_latex = image_outputs_to_latex(outputs, chapter_number, image_counts)
            if image_latex:
                return "\n\n".join([compact_code_omission_to_latex(source), image_latex])
        return ""
    display_source = format_code_for_book(source)
    listing = split_listing_for_book(display_source)
    parts = [listing]
    if include_notes:
        notes = code_walkthrough(display_source, compact=compact_code)
        if notes:
            parts.append(notes)
    if outputs:
        output_latex = output_to_latex(source, outputs, compact=compact_code)
        if output_latex:
            parts.append(output_latex)
        if chapter_number is not None and image_counts is not None:
            image_latex = image_outputs_to_latex(outputs, chapter_number, image_counts)
            if image_latex:
                parts.append(image_latex)
    else:
        synthetic_output_latex = synthetic_output_to_latex(source)
        if synthetic_output_latex:
            parts.append(synthetic_output_latex)
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


def demote_markdown_headings(markdown: str) -> str:
    lines: list[str] = []
    for line in markdown.splitlines():
        if line.lstrip().startswith("#"):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(indent + "#" + line.lstrip())
        else:
            lines.append(line)
    return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")


def appendix_section_title(chapter_number: int, first_heading: str) -> str:
    if chapter_number == 29:
        return "부록: 생성형 LLM 평가 항해 가이드"
    if chapter_number == 31:
        return "부록: Qwen GRPO와 HPO 요약"
    title = re.sub(r"^#+\s*", "", first_heading).strip()
    title = re.sub(r"^Chapter\s+\d+\s*부록\s*[—:-]\s*", "부록: ", title)
    return title or "부록"


def resolved_notebook_path(chapter: Chapter, use_executed: bool, extra_notebook: str | None = None) -> Path:
    if extra_notebook is not None:
        if use_executed:
            executed_name = EXECUTED_EXTRA_NOTEBOOKS.get((chapter.number, extra_notebook), extra_notebook)
            executed_extra = ROOT / "executed" / executed_name
            if executed_extra.exists():
                return executed_extra
        return chapter.notebook.parent / extra_notebook
    if use_executed:
        executed_path = ROOT / "executed" / f"{chapter.number:02d}_{chapter.slug}.ipynb"
        if executed_path.exists():
            return executed_path
    return chapter.notebook


def supplemental_figures_to_latex(chapter_number: int, compact: bool = False) -> str:
    specs = SUPPLEMENTAL_FIGURES.get(chapter_number, ())
    if compact and chapter_number == 27:
        specs = specs[:2]
    if not specs:
        return ""
    return "\n\n".join(figure_block(spec) for spec in specs)


def compact_appendix_to_latex(chapter: Chapter, extra_notebook: str) -> str:
    spec = COMPACT_APPENDICES.get(chapter.number)
    if spec is None:
        title = appendix_section_title(chapter.number, extra_notebook)
        summary = "상세 실험 절차와 전체 출력은 온라인 부록에서 실행하고 확인합니다."
        figure = None
    else:
        title = spec.title
        summary = spec.summary
        figure = spec.figure
    url = chapter.extra_colab_url(extra_notebook)
    blocks = [
        "\\Needspace{18\\baselineskip}",
        f"\\section{{온라인 부록: {latex_escape_prose(title)}}}",
        f"\\onlineAppendixLink{{{latex_escape_prose(title)}}}{{{summary}}}{{{url}}}",
    ]
    if figure is not None:
        blocks.append(figure_block(figure))
    return "\n\n".join(blocks)


def chapter_specific_fixes(text: str, chapter_number: int) -> str:
    if chapter_number == 14:
        text = text.replace(
            "본편 \\ref{ch:14}장은 \\inlinecode{L = L_main + $\\lambda$·L_aux} 에서 \\textbf{$\\lambda$=1} 로 학습했더니 보조 손실(별점 회귀)이 메인 항목 분류를 크게 짓눌렀습니다 (micro-F1 0.82 \\(\\to\\) 0.66). 그렇다고 보조 손실이 늘 해로운 것은 아니고, \\textbf{$\\lambda$ 를 얼마로 잡느냐의 문제}입니다.",
            "이 부록은 \\ref{ch:14}장의 보조 손실 가중치 $\\lambda$를 검증하기 위한 스윕입니다. $\\lambda=1$은 보조 손실이 과해 메인 항목 분류를 짓누르는 사례이고, 현재 본편은 스윕에서 확인한 sweet spot인 $\\lambda=0.05$를 사용합니다.",
        )
        text = text.replace(
            "\\(\\lambda\\)=0 은 보조 손실 무시(\\ref{ch:13}장 재현 baseline), \\(\\lambda\\)=1 은 본편 셋업입니다. 그 사이를 촘촘히 봅니다.",
            "\\(\\lambda\\)=0 은 보조 손실 무시(\\ref{ch:13}장 재현 baseline), \\(\\lambda\\)=1 은 과적용 비교점입니다. 그 사이에서 메인 성능이 가장 좋아지는 지점을 찾습니다.",
        )
        text = text.replace(
            "λ=1.0:          micro-F1=0.6622  (본편 셋업)",
            "lambda=1.0:     micro-F1=0.6622  (과적용 비교점)",
        )
        text = text.replace(
            "본편 셋업",
            "과적용 비교점",
        )
        text = text.replace(
            "본편 \\ref{ch:14}장 는 이 sweet spot(\\(\\lambda\\)=0.05)을 메인 학습값으로 쓰고, \\(\\lambda\\)=1 은 “과적용하면 어떻게 무너지는가”의 사례로 둡니다.",
            "본편 \\ref{ch:14}장은 이 sweet spot(\\(\\lambda\\)=0.05)을 메인 학습값으로 쓰고, \\(\\lambda\\)=1은 “과적용하면 어떻게 무너지는가”의 사례로 둡니다.",
        )
        text = text.replace(
            "\\section{부록: λ 스윕: 보조 손실 가중치의 sweet spot 찾기}",
            "\\section{\\texorpdfstring{부록: $\\lambda$ 스윕: 보조 손실 가중치의 sweet spot 찾기}{부록: lambda 스윕: 보조 손실 가중치의 sweet spot 찾기}}",
        )
        # Code/output listings use NanumGothicCoding through listings; keep Greek symbols
        # out of those blocks because listings does not reliably render them under XeLaTeX.
        text = text.replace("λ", "lambda")
        text = text.replace("Δ", "delta")
    if chapter_number == 18:
        text = text.replace(
            "\\(\\lambda\\) --- 보조 loss 가중치. 본문 기본값 \\textbf{0.1} (보조 MSE 가 메인 BCE 보다 \\emph{크기 자체가 커서} --- 1-4 vs 0.3-0.6 --- \\(\\lambda\\) 를 작게 잡아 균형).",
            "\\(\\lambda\\) --- 보조 loss 가중치. 본문 기본값은 스윕에서 확인한 sweet spot 인 \\textbf{0.05}입니다. 보조 MSE 가 메인 BCE 보다 \\emph{크기 자체가 커서} \\(\\lambda\\) 를 작게 잡아 균형을 맞춥니다.",
        )
        text = text.replace(
            "활성 개수 정답은 1 또는 2 의 \\emph{정수}. 학습 초기 보조 헤드 예측이 평균 1.5 근처면 MSE 는 약 \\(0.25\\), 무작위 예측이면 \\(1-4\\). 메인 BCE 는 K=7 평균이라 학습 초반에도 \\(0.3-0.7\\) 수준. \\emph{\\(\\lambda\\)=1} 이면 보조가 메인보다 크게 잡힐 수 있어 \\textbf{\\(\\lambda\\)=0.1} 이 권장 기본값.",
            "활성 개수 정답은 1 또는 2 의 \\emph{정수}. 학습 초기 보조 헤드 예측이 평균 1.5 근처면 MSE 는 약 \\(0.25\\), 무작위 예측이면 \\(1-4\\)까지 커질 수 있습니다. 메인 BCE 는 K=7 평균이라 학습 초반에도 \\(0.3-0.7\\) 수준입니다. 따라서 \\emph{\\(\\lambda\\)=1} 은 과하고, 부록 스윕에서는 \\textbf{\\(\\lambda\\)=0.05} 가 메인 F1 을 가장 끌어올렸습니다.",
        )
        text = text.replace(
            "0.1 & 0.45 & 0.25 & 0.475 & 5\\% \\(\\leftarrow\\) \\textbf{본문 기본} \\\\",
            "0.05 & 0.45 & 0.25 & 0.4625 & 2.7\\% \\(\\leftarrow\\) \\textbf{본문 기본, sweet spot} \\\\",
        )
        text = text.replace(
            "이 장에선 \\textbf{\\(\\lambda\\)=0.1} 로 학습하고 \\(\\lambda\\)=0 baseline 과 비교, §10 의 변형 섹션에서 \\(\\lambda\\) ∈ \\{0.0, 0.1, 1.0\\} 스윕으로 효과 분포를 봅니다.",
            "이 장에선 \\textbf{\\(\\lambda\\)=0.05} 로 학습하고 \\(\\lambda\\)=0 baseline 과 비교합니다. 부록 \\inlinecode{18\\_ko\\_auxiliary\\_lambda\\_sweep} 의 공정 seed 스윕에서 \\(\\lambda\\)=0.05 가 micro/macro-F1 을 가장 끌어올리는 sweet spot 으로 확인됐기 때문입니다.",
        )
        text = text.replace(
            "\\section{부록: λ 스윕: 약한 보조 task 의 sweet spot (이슈 \\#22)}",
            "\\section{\\texorpdfstring{부록: $\\lambda$ 스윕: 약한 보조 task 의 sweet spot}{부록: lambda 스윕: 약한 보조 task 의 sweet spot}}",
        )
        text = text.replace("\\(\\lambda\\)=0.1 은 본편 셋업.", "\\(\\lambda\\)=0.1 은 비교점입니다.")
        text = text.replace("λ=0.1 (본편)", "lambda=0.1 (비교점)")
        text = text.replace("0.1 (본편)", "0.1 (비교점)")
        text = text.replace(
            "본편이 쓴 \\(\\lambda\\)=0.1 은 공정 비교(같은 seed 초기화)에선 거의 \\textbf{중립}(0.8489 \\(\\approx\\) baseline)이고 --- 원래 이슈의 “\\(\\lambda\\)=0.1 에서 -0.008” 은 두 모델을 따로 초기화한 \\emph{불공정 비교} 탓이 큽니다 --- \\(\\lambda\\)\\(\\ge\\)0.2 부터 메인이 무너집니다(\\(\\lambda\\)=0.5 에서 0.804).",
            "\\(\\lambda\\)=0.1 은 공정 비교(같은 seed 초기화)에선 거의 \\textbf{중립}(0.8489 \\(\\approx\\) baseline)인 비교점입니다. \\(\\lambda\\)\\(\\ge\\)0.2 부터는 메인이 무너집니다(\\(\\lambda\\)=0.5 에서 0.804).",
        )
        text = text.replace("본편 \\ref{ch:18}장 은", "본편 \\ref{ch:18}장은")
        text = text.replace(
            '    loc="lower left"); plt.tight_layout(); plt.show(,\n)',
            '    loc="lower left",\n)\nplt.tight_layout()\nplt.show()\n',
        )
        text = text.replace("λ", "lambda")
        text = text.replace("Δ", "delta")
        text = text.replace("≈", "approx")
        text = text.replace("∈", "in")
        text = text.replace("→", "->")
    if chapter_number in (20, 22):
        text = text.replace(
            "eval loss 하락, perplexity 약 2-3 감소",
            "eval loss 하락, perplexity 완만히 하락(8-10 epoch 이후 평탄)",
        )
        text = text.replace(
            "eval loss 약간 하락, perplexity 약 2-3 정도 감소",
            "eval loss 약간 하락, perplexity 완만히 하락(8-10 epoch 이후 평탄)",
        )
    if chapter_number == 31:
        text = text.replace(
            "policy only,\n              ref-free,\n",
            "policy only,\n              KL beta=0.04,\n",
        )
        text = text.replace(
            "policy only, ref-free, num_generations",
            "policy only, KL beta=0.04, num_generations",
        )
        text = text.replace(
            "policy only, ref-free, num_generat...",
            "policy only, KL beta=0.04, num_generat...",
        )
    if chapter_number == 33:
        text = text.replace(
            r"\mathrm{KL}(P_{\text{gen}} \,\\vert{}\, P_{\text{unigram}})",
            r"\mathrm{KL}(P_{\text{gen}} \mathbin{\Vert} P_{\text{unigram}})",
        )
    return text


def append_notebook_cells(
    chunks: list[str],
    nb: dict,
    chapter_number: int,
    *,
    appendix: bool = False,
    image_counts: dict[int, int] | None = None,
    compact_code: bool = False,
) -> None:
    explain_code = False
    appendix_title_added = False

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        if cell.get("cell_type") == "markdown":
            first = source.strip().splitlines()[0]
            if first.startswith("# Chapter"):
                if appendix and not appendix_title_added:
                    chunks.append(f"\\section{{{latex_escape_prose(appendix_section_title(chapter_number, first))}}}")
                    appendix_title_added = True
                    chunks.append("")
                continue
            if first.lstrip().startswith("##"):
                explain_code = any(
                    section in first
                    for section in (
                        "토크나이저",
                        "실습",
                        "해부",
                        "평가",
                        "Multiple-choice",
                        "Generation",
                        "zero-shot",
                        "few-shot",
                    )
                )
            markdown_source = demote_markdown_headings(source) if appendix else source
            chunks.append(markdown_to_latex(markdown_source, chapter_number))
        elif cell.get("cell_type") == "code":
            code_block = code_to_latex(
                source,
                include_notes=explain_code,
                outputs=cell.get("outputs", []),
                chapter_number=chapter_number,
                image_counts=image_counts,
                compact_code=compact_code,
            )
            if code_block.strip():
                chunks.append(code_block)
                chunks.append("")
            continue

        chunks.append("")


def chapter_tex(
    chapter: Chapter,
    execute: bool = False,
    use_executed: bool = False,
    compact_code: bool = False,
) -> str:
    notebook_path = resolved_notebook_path(chapter, use_executed)
    if execute:
        nb = execute_notebook(notebook_path)
    else:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    chunks: list[str] = [
        "% Generated by book/tools/notebook_to_tex.py. Do not edit by hand.",
        f"\\chapter[{chapter.short_title}]{{{chapter.title}}}",
        f"\\label{{ch:{chapter.number:02d}}}",
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
    image_counts: dict[int, int] = {}
    append_notebook_cells(chunks, nb, chapter.number, image_counts=image_counts, compact_code=compact_code)

    supplemental = supplemental_figures_to_latex(chapter.number, compact=compact_code)
    if supplemental:
        chunks.append("\\section{보조 시각화}")
        chunks.append(supplemental)

    for extra_notebook in chapter.extra_notebooks:
        if compact_code:
            chunks.append(compact_appendix_to_latex(chapter, extra_notebook))
            continue
        extra_path = resolved_notebook_path(chapter, use_executed, extra_notebook)
        if not extra_path.exists():
            raise FileNotFoundError(extra_path)
        extra_nb = json.loads(extra_path.read_text(encoding="utf-8"))
        append_notebook_cells(
            chunks,
            extra_nb,
            chapter.number,
            appendix=True,
            image_counts=image_counts,
            compact_code=compact_code,
        )

    chapter_latex = "\n\n".join(chunks).rstrip() + "\n"
    chapter_latex = wrap_tabular_tables(chapter_latex, chapter.number)
    chapter_latex = display_math_to_numbered_equations(chapter_latex, chapter.number)
    chapter_latex = link_chapter_references(chapter_latex)
    chapter_latex = chapter_specific_fixes(chapter_latex, chapter.number)
    if compact_code:
        chapter_latex = compact_faq_section(chapter_latex, chapter.number)
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
    parser.add_argument(
        "--use-executed",
        action="store_true",
        help="prefer notebooks under executed/ so saved outputs and plot images are reflected",
    )
    parser.add_argument(
        "--compact-code",
        action="store_true",
        help="keep only concept-critical code cells and replace routine cells with a Colab/QR note",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHAPTER_DIR,
        help="directory for generated chapter .tex files; defaults to book/chapters",
    )
    args = parser.parse_args()

    selected = set(args.chapters or [chapter.number for chapter in CHAPTERS])
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = BOOK / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for chapter in CHAPTERS:
        if chapter.number not in selected:
            continue
        if not chapter.notebook.exists():
            raise FileNotFoundError(chapter.notebook)
        out = output_dir / chapter.tex_name
        out.write_text(
            chapter_tex(
                chapter,
                execute=args.execute,
                use_executed=args.use_executed,
                compact_code=args.compact_code,
            ),
            encoding="utf-8",
        )
        try:
            display_path = out.relative_to(ROOT)
        except ValueError:
            display_path = out
        print(f"wrote {display_path}")


if __name__ == "__main__":
    main()
