"""
SHAP explainability module.
Dual strategy: LinearExplainer for TF-IDF, KernelExplainer with
sentence-level masks for DistilBERT.
Raw attention weights are NOT used (Jain & Wallace 2019).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from functools import lru_cache
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path constants
ROOT             = Path(__file__).resolve().parents[2]
SPLITS_DIR       = ROOT / "data" / "splits"
MODELS_DIR       = ROOT / "models"
LOGS_DIR         = ROOT / "logs"
BASELINE_PATH    = MODELS_DIR / "baseline" / "tfidf_logreg.joblib"
TRANSFORMER_PATH = MODELS_DIR / "distilbert"
SHAP_CHART_PATH  = LOGS_DIR / "shap_example.png"
SHAP_SUMMARY_PATH = LOGS_DIR / "shap_summary.txt"

# SHAP configuration
MAX_TOKEN_LENGTH    = 512
SHAP_BASELINE_ROWS  = 50
SHAP_SAMPLE_SIZE    = 6
SHAP_N_SAMPLES      = 64
SHAP_MAX_SENTENCES  = 12
SHAP_TOP_FEATURES   = 20
SHAP_TOP_SENTENCES  = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# Sentence splitting

def _split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using NLTK if available,
    falls back to regex-based splitting.
    """
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except (ImportError, LookupError):
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Filter out very short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 15]


# Lazy model loaders (used by explain_article API function)

@lru_cache(maxsize=1)
def _load_baseline():
    """Loads TF-IDF + LR pipeline (cached)."""
    if not BASELINE_PATH.exists():
        return None
    try:
        return joblib.load(BASELINE_PATH)
    except Exception as exc:
        log.error(f"Could not load baseline: {exc}")
        return None


@lru_cache(maxsize=1)
def _load_distilbert():
    """Loads fine-tuned DistilBERT (cached)."""
    if not TRANSFORMER_PATH.exists():
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))
        model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_PATH))
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model, tokenizer
    except Exception as exc:
        log.error(f"Could not load DistilBERT: {exc}")
        return None, None


def _predict_p_fake(model, tokenizer, text: str) -> float:
    """Returns P(fake) for a single text, on the model's device."""
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=MAX_TOKEN_LENGTH,
    )
    enc.pop("token_type_ids", None)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return float(torch.softmax(logits, dim=-1)[0, 1].detach().cpu())


def _sentence_kernel_shap(model, tokenizer, sentences: list[str]) -> np.ndarray:
    """
    Sentence-level Kernel SHAP for P(fake).

    The background is an empty article (all sentences off). The instance
    being explained is the full article (all sentences on). Using the
    full article as both background and query yields all-zero SHAP values.
    """
    n = len(sentences)

    def predict_fn(mask_matrix: np.ndarray) -> np.ndarray:
        outputs = []
        for mask in np.atleast_2d(np.asarray(mask_matrix)):
            active = [s for s, m in zip(sentences, mask) if float(m) > 0.5]
            text_in = " ".join(active) if active else "no content"
            outputs.append(_predict_p_fake(model, tokenizer, text_in))
        return np.asarray(outputs, dtype=float).reshape(-1, 1)

    background = np.zeros((1, n))
    query = np.ones((1, n))
    n_samples = max(2 * n + 8, min(SHAP_N_SAMPLES, 4 * n + 16))
    explainer = shap.KernelExplainer(predict_fn, background)
    sv = explainer.shap_values(query, nsamples=n_samples, silent=True)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv, dtype=float).reshape(-1)
    if sv.size != n:
        raise ValueError(f"SHAP length mismatch ({sv.size} vs {n})")
    return sv


# SHAP -- TF-IDF LinearExplainer

def explain_baseline_shap(
    pipeline, id_test_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Computes SHAP values for the TF-IDF + LR pipeline using LinearExplainer."""
    log.info("Computing SHAP values for TF-IDF baseline (LinearExplainer)...")

    vectorizer = pipeline.named_steps["tfidf"]
    clf        = pipeline.named_steps["clf"]

    sample = id_test_df["text_truncated"].fillna("").head(SHAP_BASELINE_ROWS).tolist()
    X_bg   = vectorizer.transform(sample)

    explainer = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(X_bg)

    # Binary LR may return list [class_0, class_1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    feature_names = vectorizer.get_feature_names_out().tolist()
    log.info(f"  TF-IDF SHAP complete -- {len(feature_names):,} features analysed.")
    return shap_vals, feature_names


# SHAP -- DistilBERT KernelExplainer (sentence-level perturbations)

def explain_transformer_shap(
    model, tokenizer, id_test_df: pd.DataFrame,
) -> tuple[list[list[float]], list[list[str]], list[str]]:
    """
    Computes sentence-level SHAP values for the DistilBERT model.
    Each article is split into sentences, then KernelExplainer approximates
    the Shapley value for each sentence position.
    """
    log.info("Computing sentence-level SHAP for transformer (KernelExplainer)...")
    log.info(
        f"  Sample: {SHAP_SAMPLE_SIZE} articles | "
        f"nsamples: {SHAP_N_SAMPLES} | slow on CPU -- please wait..."
    )

    model.eval()

    sample_df = id_test_df.sample(
        n=min(SHAP_SAMPLE_SIZE, len(id_test_df)),
        random_state=42,
    ).reset_index(drop=True)

    all_shap_vals: list[list[float]] = []
    all_sentences: list[list[str]]   = []
    all_texts:     list[str]         = []

    for row_idx, row in sample_df.iterrows():
        text = row.get("text_truncated", row.get("text_raw", ""))
        sentences = _split_sentences(text)

        if len(sentences) < 2:
            log.warning(f"Article {row_idx}: <2 sentences -- skipping SHAP.")
            continue

        sentences = sentences[:SHAP_MAX_SENTENCES]
        n = len(sentences)

        try:
            sv = _sentence_kernel_shap(model, tokenizer, sentences)

            if len(sv) != n:
                log.warning(f"  SHAP length mismatch ({len(sv)} vs {n}) -- skipping.")
                continue

            all_shap_vals.append(sv.tolist())
            all_sentences.append(sentences)
            all_texts.append(text)
            log.info(f"  Article {row_idx + 1}/{len(sample_df)} ({n} sentences)")

        except Exception as exc:
            log.warning(f"  KernelExplainer failed for article {row_idx}: {exc}")

    log.info(f"Transformer SHAP complete -- {len(all_shap_vals)} articles explained.")
    return all_shap_vals, all_sentences, all_texts


# Public API function -- used by FastAPI backend

def explain_article(
    text: str,
    model_name: str = "distilbert",
    top_n: int = SHAP_TOP_SENTENCES,
) -> dict:
    """
    Explains a single article's prediction using SHAP.
    Returns: {"top_sentences": [{"sentence": str, "shap_value": float}, ...]}
    """
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return {"top_sentences": []}

    if model_name == "distilbert":
        model, tokenizer = _load_distilbert()
        if model is None:
            return {"top_sentences": []}

        sentences = sentences[:SHAP_MAX_SENTENCES]
        n = len(sentences)
        try:
            sv = _sentence_kernel_shap(model, tokenizer, sentences)
        except Exception as exc:
            log.warning(f"SHAP explain_article failed: {exc}")
            return {"top_sentences": []}

        if len(sv) != n:
            return {"top_sentences": []}

        # Return top-N by absolute SHAP value
        indices = np.argsort(np.abs(sv))[::-1][:top_n]
        top = [
            {"sentence": sentences[i], "shap_value": round(float(sv[i]), 4)}
            for i in indices
        ]
        return {"top_sentences": top}

    elif model_name == "baseline":
        pipeline = _load_baseline()
        if pipeline is None:
            return {"top_sentences": []}
        vectorizer = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        X = vectorizer.transform([text])
        explainer = shap.LinearExplainer(clf, X, feature_perturbation="interventional")
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv).flatten()
        feature_names = vectorizer.get_feature_names_out()
        indices = np.argsort(np.abs(sv))[::-1][:top_n]
        top = [
            {"sentence": feature_names[i], "shap_value": round(float(sv[i]), 4)}
            for i in indices if sv[i] != 0
        ]
        return {"top_sentences": top}

    return {"top_sentences": []}


# SHAP bar chart (PNG, 300 DPI)

# Dark-mode colour palette
_BG_DARK   = "#0F1117"
_BG_PANEL  = "#1A1D27"
_COL_FAKE  = "#FF6B6B"
_COL_REAL  = "#4ECDC4"
_COL_TEXT  = "#DDDDDD"
_COL_MUTED = "#888899"
_COL_EDGE  = "#333344"


def _style_ax(ax) -> None:
    """Applies dark-mode styling to a matplotlib Axes object."""
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_COL_TEXT, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(_COL_EDGE)
    ax.title.set_color("#FFFFFF")
    ax.xaxis.label.set_color(_COL_TEXT)
    ax.yaxis.label.set_color(_COL_TEXT)


def save_shap_chart(
    baseline_shap: np.ndarray,
    feature_names: list[str],
    transformer_shap: list[list[float]],
    transformer_sentences: list[list[str]],
) -> None:
    """Generates a 2-panel dark-mode SHAP figure."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor(_BG_DARK)
    _style_ax(ax1)
    _style_ax(ax2)

    # Panel A -- TF-IDF features
    mean_abs  = np.abs(baseline_shap).mean(axis=0)
    mean_sign = baseline_shap.mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:SHAP_TOP_FEATURES]

    feats  = [feature_names[i] for i in top_idx]
    vals   = [mean_abs[i]      for i in top_idx]
    dirs   = [mean_sign[i]     for i in top_idx]
    colors = [_COL_FAKE if d > 0 else _COL_REAL for d in dirs]

    ax1.barh(range(SHAP_TOP_FEATURES), vals[::-1], color=colors[::-1],
             edgecolor="none", height=0.72)
    ax1.set_yticks(range(SHAP_TOP_FEATURES))
    ax1.set_yticklabels(feats[::-1], fontsize=8, color=_COL_TEXT, fontfamily="monospace")
    ax1.set_xlabel("Mean |SHAP Value|", color=_COL_TEXT, fontsize=10)
    ax1.set_title(
        f"TF-IDF Baseline\nTop-{SHAP_TOP_FEATURES} Feature Importances (SHAP)",
        fontsize=11, pad=12,
    )
    ax1.tick_params(axis="y", length=0)

    fake_p = mpatches.Patch(color=_COL_FAKE, label="-> Fake signal")
    real_p = mpatches.Patch(color=_COL_REAL, label="-> Real signal")
    ax1.legend(handles=[fake_p, real_p], loc="lower right",
               facecolor=_BG_PANEL, edgecolor=_COL_EDGE,
               labelcolor=_COL_TEXT, fontsize=9)

    # Panel B -- Transformer sentence SHAP
    if transformer_shap and transformer_sentences:
        sv    = np.array(transformer_shap[0])
        sents = transformer_sentences[0]
        labels = [
            (f"S{i+1}: {s[:50]}..." if len(s) > 50 else f"S{i+1}: {s}")
            for i, s in enumerate(sents)
        ]
        colors2 = [_COL_FAKE if v > 0 else _COL_REAL for v in sv]

        ax2.barh(range(len(sv)), sv[::-1], color=colors2[::-1],
                 edgecolor="none", height=0.72)
        ax2.axvline(0, color=_COL_MUTED, linewidth=0.8, linestyle="--")
        ax2.set_yticks(range(len(sv)))
        ax2.set_yticklabels(labels[::-1], fontsize=7.5, color=_COL_TEXT)
        ax2.set_xlabel("SHAP Value -- impact on P(fake)", color=_COL_TEXT, fontsize=10)
        ax2.set_title(
            "DistilBERT Transformer\nSentence-Level SHAP (Example Article)",
            fontsize=11, pad=12,
        )
        ax2.tick_params(axis="y", length=0)

        fake_p2 = mpatches.Patch(color=_COL_FAKE, label="+ P(fake)")
        real_p2 = mpatches.Patch(color=_COL_REAL, label="- P(fake)")
        ax2.legend(handles=[fake_p2, real_p2], loc="lower right",
                   facecolor=_BG_PANEL, edgecolor=_COL_EDGE,
                   labelcolor=_COL_TEXT, fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No transformer SHAP data available.",
                 ha="center", va="center", color=_COL_MUTED, fontsize=11,
                 transform=ax2.transAxes)

    fig.suptitle(
        "SHAP Explainability Report -- Fake News Detection",
        fontsize=14, fontweight="bold", color="#FFFFFF", y=1.005,
    )
    plt.tight_layout()
    plt.savefig(SHAP_CHART_PATH, dpi=300, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)
    log.info(f"SHAP chart saved -> {SHAP_CHART_PATH}")


# Text summary output

def save_shap_summary(
    baseline_shap: np.ndarray,
    feature_names: list[str],
    transformer_shap: list[list[float]],
    transformer_sentences: list[list[str]],
    transformer_texts: list[str],
) -> None:
    """Writes ranked SHAP summary to logs/shap_summary.txt."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines += [
        "=" * 72,
        "  SHAP Explainability Summary",
        "  Daniil Khmelnytskyi -- PRA3017-N",
        "=" * 72, "",
        "METHODOLOGY NOTE", "-" * 72,
        "Raw transformer attention weights are deliberately NOT used.",
        "Instead, two SHAP methods provide model-agnostic attributions:",
        "  1. LinearExplainer (TF-IDF baseline) -- exact Shapley values.",
        "  2. KernelExplainer (DistilBERT) -- sentence-level mask perturbations.",
        "Both satisfy efficiency, symmetry, dummy, and linearity axioms",
        "of cooperative game theory (Lundberg & Lee, 2017).", "",
    ]

    # Part 1 -- TF-IDF
    lines += [
        "=" * 72,
        "  PART 1 -- TF-IDF Baseline: Top Feature Importances",
        "=" * 72,
    ]
    mean_abs  = np.abs(baseline_shap).mean(axis=0)
    mean_sign = baseline_shap.mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:SHAP_TOP_FEATURES]

    lines.append(f"\n{'Rank':<6}{'Feature':<38}{'Mean |SHAP|':>12}  Direction")
    lines.append("-" * 68)
    for rank, idx in enumerate(top_idx, 1):
        direction = "-> FAKE" if mean_sign[idx] > 0 else "-> REAL"
        lines.append(f"{rank:<6}{feature_names[idx]:<38}{mean_abs[idx]:>12.5f}  {direction}")
    lines.append("")

    # Part 2 -- Transformer sentences
    lines += [
        "=" * 72,
        "  PART 2 -- DistilBERT: Sentence-Level SHAP Values",
        "=" * 72,
        "(Positive SHAP -> sentence increases P(fake))", "",
    ]
    for i, (sv, sents, text) in enumerate(
        zip(transformer_shap, transformer_sentences, transformer_texts), 1
    ):
        preview = text[:90] + "..." if len(text) > 90 else text
        lines.append(f"[Article {i}]  {preview}")
        lines.append(f"  {'Sent':<6}{'SHAP':>10}  Signal   Sentence (<=100 chars)")
        lines.append("  " + "-" * 70)
        for j, (s_val, sent) in enumerate(zip(sv, sents), 1):
            flag = "FAKE" if s_val > 0.01 else ("REAL" if s_val < -0.01 else "NEUT")
            preview = (sent[:100] + "...") if len(sent) > 100 else sent
            lines.append(f"  {j:<6}{s_val:>+10.4f}  [{flag:>4s}]  {preview}")
        lines.append("")

    SHAP_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"SHAP text summary saved -> {SHAP_SUMMARY_PATH}")


# Entry point

def main() -> None:
    log.info("=" * 60)
    log.info("  SHAP Explainability Pipeline")
    log.info("=" * 60)

    test_path = SPLITS_DIR / "test.parquet"
    if not test_path.exists():
        log.critical(f"Test split not found: {test_path}")
        sys.exit(1)

    id_test_df = pd.read_parquet(test_path, engine="pyarrow")
    if len(id_test_df) == 0:
        log.critical("Test split is empty.")
        sys.exit(1)

    log.info(f"Loaded test split: {len(id_test_df)} articles")

    # Baseline SHAP
    baseline_shap_vals = np.array([[]])
    feature_names = []
    pipeline = _load_baseline()
    if pipeline is not None:
        baseline_shap_vals, feature_names = explain_baseline_shap(pipeline, id_test_df)

    # Transformer SHAP
    transformer_shap_vals = []
    transformer_sentences = []
    transformer_texts = []
    model, tokenizer = _load_distilbert()
    if model is not None:
        transformer_shap_vals, transformer_sentences, transformer_texts = (
            explain_transformer_shap(model, tokenizer, id_test_df)
        )

    if len(feature_names) > 0:
        save_shap_summary(
            baseline_shap_vals, feature_names,
            transformer_shap_vals, transformer_sentences, transformer_texts,
        )
        save_shap_chart(
            baseline_shap_vals, feature_names,
            transformer_shap_vals, transformer_sentences,
        )

    log.info("SHAP explainability complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Explainability")
    parser.add_argument("--text", type=str, default=None,
                        help="Single article text to explain (interactive mode)")
    args = parser.parse_args()

    if args.text:
        result = explain_article(args.text)
        print("\nSHAP Explanation:")
        for item in result["top_sentences"]:
            print(f"  [{item['shap_value']:+.4f}] {item['sentence'][:80]}")
    else:
        main()
