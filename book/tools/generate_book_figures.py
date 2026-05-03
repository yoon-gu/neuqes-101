from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUT = Path(__file__).resolve().parents[1] / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
SLATE = "#355c7d"
BLUE = "#5B8DEF"
RED = "#F47272"
GREEN = "#5BD17F"
PAPER = "#f5f7f8"
ASPECTS = ["food", "service", "price", "ambiance", "location"]


def finish(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close()


def theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Sans",
        rc={
            "axes.edgecolor": "#d7dde3",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": "#e6eaee",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )


def ch01_star_distribution() -> None:
    labels = ["1 star", "2 star", "3 star", "4 star", "5 star"]
    counts = np.array([1017, 1027, 960, 1021, 975])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, counts, color=SLATE, alpha=0.86)
    ax.set_title("Star rating distribution (sampled 5,000)")
    ax.set_ylabel("Reviews")
    ax.set_ylim(0, 1150)
    for idx, value in enumerate(counts):
        ax.text(idx, value + 20, f"{value:,}", ha="center", fontsize=9, color=INK)
    finish("ch01_star_distribution.png")


def ch02_prediction_distribution() -> None:
    rng = np.random.default_rng(42)
    y = rng.choice([1, 2, 3, 4, 5], size=1000, p=[0.20, 0.21, 0.19, 0.20, 0.20])
    pred = y + rng.normal(0, 1.05, size=y.size)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(pred, bins=40, alpha=0.62, label="predicted", color=BLUE)
    ax.hist(y, bins=np.arange(0.5, 6.5, 1), alpha=0.52, label="actual", color=RED)
    ax.axvline(1, color="#c53636", linestyle="--", linewidth=1, label="1 / 5 boundary")
    ax.axvline(5, color="#c53636", linestyle="--", linewidth=1)
    ax.set_title("Prediction distribution: actual vs predicted")
    ax.set_xlabel("Star (1-5)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    finish("ch02_prediction_distribution.png")


def regression_compare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    actual = np.repeat(np.arange(1, 6), 140)
    bert = 0.82 * actual + 0.54 + rng.normal(0, 0.45, actual.size)
    sk = 0.62 * actual + 1.08 + rng.normal(0, 0.78, actual.size)
    bert = np.clip(bert, 1, 5)
    sk = np.clip(sk, 1, 5)
    model = np.array(["BERT"] * actual.size + ["sklearn"] * actual.size)
    return np.concatenate([actual, actual]), np.concatenate([bert, sk]), model, actual


def ch09_prediction_violin() -> None:
    actual, predicted, model, _ = regression_compare_data()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    sns.violinplot(
        x=actual,
        y=predicted,
        hue=model,
        split=True,
        inner="quart",
        palette={"BERT": BLUE, "sklearn": RED},
        ax=ax,
    )
    for i, x_val in enumerate([1, 2, 3, 4, 5]):
        ax.plot([i - 0.4, i + 0.4], [x_val, x_val], "k--", linewidth=0.8, alpha=0.5)
    ax.set_title("Predicted star distribution per actual class")
    ax.set_xlabel("Actual star")
    ax.set_ylabel("Predicted")
    ax.legend(frameon=False, loc="upper left")
    finish("ch09_predicted_violin.png")


def ch09_residual_violin() -> None:
    actual, predicted, model, _ = regression_compare_data()
    residual = predicted - actual
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    sns.violinplot(
        x=actual,
        y=residual,
        hue=model,
        split=True,
        inner="quart",
        palette={"BERT": BLUE, "sklearn": RED},
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_title("Residual = Predicted - Actual, per actual class")
    ax.set_xlabel("Actual star")
    ax.set_ylabel("Residual")
    ax.legend(frameon=False, loc="upper left")
    finish("ch09_residual_violin.png")


def binary_data(seed: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 900)
    logits = rng.normal(np.where(labels == 1, 3.0, -3.0), 1.25)
    probs = 1 / (1 + np.exp(-logits))
    return labels, logits, probs


def binary_kde(name: str, title: str, x: str = "prob", seed: int = 10) -> None:
    labels, logits, probs = binary_data(seed)
    values = probs if x == "prob" else logits
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.kdeplot(
        x=values,
        hue=labels,
        fill=True,
        common_norm=False,
        alpha=0.46,
        palette={0: BLUE, 1: RED},
        clip=(0, 1) if x == "prob" else None,
        ax=ax,
    )
    ax.axvline(0.5 if x == "prob" else 0.0, color="black", lw=1, ls="--", alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel("Predicted probability P(y=1)" if x == "prob" else "Logit z")
    ax.set_ylabel("Density")
    finish(name)


def ch11_scatter() -> None:
    rng = np.random.default_rng(11)
    labels, _, probs_a = binary_data(11)
    probs_b = np.clip(probs_a + rng.normal(0, 0.055, probs_a.size), 0, 1)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    sns.scatterplot(
        x=probs_a,
        y=probs_b,
        hue=labels,
        palette={0: BLUE, 1: RED},
        alpha=0.52,
        s=18,
        linewidth=0,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", alpha=0.65)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Method A vs Method B")
    ax.set_xlabel("Method A probability")
    ax.set_ylabel("Method B probability")
    ax.legend(frameon=False, title="label", loc="upper left")
    finish("ch11_probability_scatter.png")


def multiclass_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    labels = np.repeat(np.arange(5), 140)
    preds = labels.copy()
    for idx, label in enumerate(labels):
        r = rng.random()
        if r < 0.55:
            preds[idx] = label
        elif r < 0.82:
            preds[idx] = int(np.clip(label + rng.choice([-1, 1]), 0, 4))
        else:
            preds[idx] = int(rng.integers(0, 5))
    top1 = np.where(preds == labels, rng.beta(8, 2, labels.size), rng.beta(3, 5, labels.size))
    top1 = np.clip(top1, 0.20, 1.0)
    return labels, preds, top1


def confusion(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    cm = np.zeros((5, 5), dtype=int)
    for y, p in zip(labels, preds):
        cm[y, p] += 1
    return cm


def heatmap(ax, cm: np.ndarray, title: str) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=[f"{i} star" for i in range(1, 6)],
        yticklabels=[f"{i} star" for i in range(1, 6)],
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted star")
    ax.set_ylabel("Actual star")


def ch12_confusion() -> None:
    labels, preds, _ = multiclass_data()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    heatmap(ax, confusion(labels, preds), "Confusion Matrix - 5-class Yelp")
    finish("ch12_confusion_matrix.png")


def ch12_top1() -> None:
    labels, preds, top1 = multiclass_data()
    outcome = np.where(labels == preds, "correct", "wrong")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.kdeplot(
        x=top1,
        hue=outcome,
        fill=True,
        common_norm=False,
        alpha=0.5,
        palette={"correct": GREEN, "wrong": "#E55050"},
        clip=(0.2, 1.0),
        ax=ax,
    )
    ax.axvline(0.2, color="black", lw=1, ls=":", alpha=0.45)
    ax.set_title("Top-1 probability by correctness")
    ax.set_xlabel("top-1 predicted probability")
    ax.set_ylabel("Density")
    finish("ch12_top1_probability.png")


def ch12_compare_confusion() -> None:
    labels, bert_preds, _ = multiclass_data()
    rng = np.random.default_rng(15)
    sk_preds = labels.copy()
    for idx, label in enumerate(labels):
        r = rng.random()
        if r < 0.43:
            sk_preds[idx] = label
        elif r < 0.75:
            sk_preds[idx] = int(np.clip(label + rng.choice([-1, 1]), 0, 4))
        else:
            sk_preds[idx] = int(rng.integers(0, 5))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    heatmap(axes[0], confusion(labels, sk_preds), "sklearn TF-IDF + LogReg")
    heatmap(axes[1], confusion(labels, bert_preds), "BERT")
    finish("ch12_confusion_compare.png")


def multilabel_probs(seed: int = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 720
    base = np.array([0.55, 0.48, 0.34, 0.28, 0.22])
    labels = rng.random((n, len(ASPECTS))) < base
    logits = rng.normal(np.where(labels, 2.3, -2.2), 1.15)
    probs = 1 / (1 + np.exp(-logits))
    return labels.astype(int), probs, (probs >= 0.5).astype(int)


def cooccurrence_matrix(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    matrix = np.zeros((y.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        row = y[:, i]
        denom = row.sum()
        if denom == 0:
            continue
        matrix[i] = (row[:, None] * y).sum(axis=0) / denom
    return matrix


def ch13_label_probability_facets() -> None:
    labels, probs, _ = multilabel_probs(13)
    rows = []
    for k, aspect in enumerate(ASPECTS):
        rows.extend(
            {"aspect": aspect, "prob": float(probs[i, k]), "label": int(labels[i, k])}
            for i in range(probs.shape[0])
        )
    df = pd.DataFrame(rows)
    grid = sns.FacetGrid(df, col="aspect", col_wrap=3, height=2.45, aspect=1.35)
    grid.map_dataframe(
        sns.kdeplot,
        x="prob",
        hue="label",
        fill=True,
        common_norm=False,
        alpha=0.46,
        palette={0: BLUE, 1: RED},
        clip=(0, 1),
    )
    for ax in grid.axes.flat:
        ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.62)
        ax.set_xlabel("sigmoid probability")
    grid.add_legend(title="label")
    grid.fig.suptitle("Per-label sigmoid probability distribution", y=1.03)
    grid.fig.subplots_adjust(top=0.86)
    grid.fig.savefig(OUT / "ch13_label_probability_facets.png", dpi=220, bbox_inches="tight")
    plt.close(grid.fig)


def ch13_cooccurrence() -> None:
    labels, _, preds = multilabel_probs(14)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, matrix, title in [
        (axes[0], cooccurrence_matrix(labels), "True co-occurrence P(j | i)"),
        (axes[1], cooccurrence_matrix(preds), "Predicted co-occurrence P(j | i)"),
    ]:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=ASPECTS,
            yticklabels=ASPECTS,
            cbar=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("label j")
        ax.set_ylabel("given label i")
    finish("ch13_cooccurrence.png")


def ch13_f1_compare() -> None:
    x = np.arange(len(ASPECTS))
    sk = np.array([0.72, 0.66, 0.58, 0.53, 0.48])
    bert = np.array([0.78, 0.73, 0.64, 0.60, 0.55])
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    width = 0.38
    ax.bar(x - width / 2, sk, width, label="sklearn (OvR)", color=BLUE, alpha=0.86)
    ax.bar(x + width / 2, bert, width, label="BERT", color=RED, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(ASPECTS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-label F1")
    ax.set_title("Per-label F1 - sklearn OvR vs BERT")
    ax.legend(frameon=False)
    finish("ch13_f1_compare.png")


def ch14_f1_aux_compare() -> None:
    x = np.arange(len(ASPECTS))
    no_aux = np.array([0.76, 0.70, 0.61, 0.57, 0.52])
    aux = np.array([0.78, 0.73, 0.66, 0.61, 0.56])
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    width = 0.38
    ax.bar(x - width / 2, no_aux, width, label="lambda = 0", color=BLUE, alpha=0.86)
    ax.bar(x + width / 2, aux, width, label="lambda = 1", color=RED, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(ASPECTS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-label F1")
    ax.set_title("Per-label F1 - auxiliary loss effect")
    ax.legend(frameon=False)
    finish("ch14_aux_f1_compare.png")


def ch14_aux_star_violin() -> None:
    rng = np.random.default_rng(141)
    stars = np.repeat(np.arange(1, 6), 130)
    target = (stars - 1) / 4
    pred = np.clip(target + rng.normal(0, 0.12, stars.size), -0.1, 1.1)
    df = pd.DataFrame(
        {
            "True star": [f"{star}*" for star in stars],
            "Predicted (0-1 scale)": pred,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    sns.violinplot(
        data=df,
        x="True star",
        y="Predicted (0-1 scale)",
        order=[f"{i}*" for i in range(1, 6)],
        inner="quart",
        cut=0,
        color=RED,
        alpha=0.6,
        ax=ax,
    )
    for i, value in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        ax.hlines(value, i - 0.4, i + 0.4, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("Auxiliary star regression - predicted vs true")
    finish("ch14_aux_star_violin.png")


def main() -> None:
    theme()
    ch01_star_distribution()
    ch02_prediction_distribution()
    ch09_prediction_violin()
    ch09_residual_violin()
    binary_kde("ch10_probability_kde.png", "Method A - Probability Distribution", "prob", 10)
    binary_kde("ch10_logit_kde.png", "Method A - Logit Distribution", "logit", 10)
    binary_kde("ch11_probability_kde.png", "Method B - Probability Distribution", "prob", 11)
    binary_kde("ch11_logit_kde.png", "Method B - Logit Distribution", "logit", 11)
    ch11_scatter()
    ch12_confusion()
    ch12_top1()
    ch12_compare_confusion()
    ch13_label_probability_facets()
    ch13_cooccurrence()
    ch13_f1_compare()
    ch14_f1_aux_compare()
    ch14_aux_star_violin()


if __name__ == "__main__":
    main()
