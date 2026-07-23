"""
Evaluation pipeline -- computes ID + OOD metrics for all models.
Compares TF-IDF baseline, SVM, and DistilBERT across test splits.
Saves results to logs/evaluation_metrics.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path constants
ROOT             = Path(__file__).resolve().parents[2]
SPLITS_DIR       = ROOT / "data" / "splits"
MODELS_DIR       = ROOT / "models"
LOGS_DIR         = ROOT / "logs"
BASELINE_PATH    = MODELS_DIR / "baseline" / "tfidf_logreg.joblib"
TRANSFORMER_PATH = MODELS_DIR / "distilbert"
METRICS_PATH     = LOGS_DIR / "evaluation_metrics.json"

MAX_TOKEN_LENGTH = 512
BATCH_SIZE       = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _encode_labels(series: pd.Series) -> list[int]:
    return [1 if l == "fake" else 0 for l in series]


def _metric_dict(y_true: list, y_pred: list, y_proba: list = None) -> dict:
    """Returns a comprehensive metrics dictionary."""
    n = len(y_true)
    result = {
        "n_samples": n,
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "recall_weighted": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "precision_per_class": {
            "real": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
            "fake": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        },
        "recall_per_class": {
            "real": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
            "fake": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        },
        "f1_per_class": {
            "real": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
            "fake": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None and len(set(y_true)) == 2:
        try:
            result["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except ValueError:
            result["roc_auc"] = None
    else:
        result["roc_auc"] = None

    return result


def _load_split(name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


def evaluate_baseline(splits: dict[str, pd.DataFrame]) -> dict:
    """Evaluates the TF-IDF + LogReg baseline on all test splits."""
    log.info("-" * 60)
    log.info("Evaluating TF-IDF + LR Baseline")
    log.info("-" * 60)

    if not BASELINE_PATH.exists():
        log.warning(f"Baseline not found at {BASELINE_PATH} -- skipping.")
        return {}

    pipeline = joblib.load(BASELINE_PATH)
    results = {}

    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            log.info(f"  {split_name}: SKIPPED (no data)")
            continue

        X = split_df["text_truncated"].fillna("").tolist()
        y_true = _encode_labels(split_df["label"])
        y_pred = pipeline.predict(X)

        y_proba = None
        try:
            proba = pipeline.predict_proba(X)
            y_proba = proba[:, 1].tolist()
        except Exception:
            pass

        m = _metric_dict(y_true, y_pred, y_proba)
        results[split_name] = m

        log.info(
            f"  {split_name:20s}: acc={m['accuracy']:.4f}  "
            f"f1={m['f1_weighted']:.4f}  roc_auc={m['roc_auc']}"
        )
        log.info(
            "\n" + classification_report(
                y_true, y_pred, target_names=["real", "fake"], zero_division=0
            )
        )

    return results


def evaluate_transformer(splits: dict[str, pd.DataFrame]) -> dict:
    """Evaluates the DistilBERT model on all test splits."""
    log.info("-" * 60)
    log.info("Evaluating DistilBERT Transformer")
    log.info("-" * 60)

    if not TRANSFORMER_PATH.exists():
        log.warning(f"Transformer not found at {TRANSFORMER_PATH} -- skipping.")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_PATH))
    model.eval()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.to("cuda")

    results = {}

    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            log.info(f"  {split_name}: SKIPPED (no data)")
            continue

        texts = split_df["text_truncated"].fillna("").tolist()
        y_true = _encode_labels(split_df["label"])

        all_preds = []
        all_proba = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=MAX_TOKEN_LENGTH,
            )
            if cuda_available:
                enc = {k: v.to("cuda") for k, v in enc.items()}

            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1).tolist()
            all_preds.extend(preds)
            all_proba.extend(probs[:, 1].tolist())

        m = _metric_dict(y_true, all_preds, all_proba)
        results[split_name] = m

        log.info(
            f"  {split_name:20s}: acc={m['accuracy']:.4f}  "
            f"f1={m['f1_weighted']:.4f}  roc_auc={m['roc_auc']}"
        )
        log.info(
            "\n" + classification_report(
                y_true, all_preds, target_names=["real", "fake"], zero_division=0
            )
        )

    return results


def main() -> None:
    log.info("=" * 60)
    log.info("  Evaluation Pipeline -- ID + OOD Metrics")
    log.info("=" * 60)

    splits = {}
    for name in ["val", "test", "ood_test"]:
        df = _load_split(name)
        if len(df) > 0:
            splits[name] = df
            log.info(f"  Loaded {name}: {len(df)} articles")
        else:
            log.info(f"  {name}: empty or not found")

    if not splits:
        log.critical("No test splits found.  Run the data pipeline first.")
        sys.exit(1)

    baseline_metrics    = evaluate_baseline(splits)
    transformer_metrics = evaluate_transformer(splits)

    all_metrics = {
        "baseline_tfidf_lr": baseline_metrics,
        "transformer_distilbert": transformer_metrics,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    log.info(f"Evaluation metrics saved -> {METRICS_PATH}")

    # Side-by-side comparison
    log.info("-" * 60)
    log.info("  SIDE-BY-SIDE COMPARISON")
    log.info("-" * 60)
    log.info(
        f"{'Split':<15} {'Model':<20} {'Acc':>8} {'F1':>8} {'AUC':>8}"
    )
    log.info("-" * 60)

    for split_name in splits:
        for model_name, metrics in [
            ("TF-IDF+LR", baseline_metrics),
            ("DistilBERT", transformer_metrics),
        ]:
            if split_name in metrics:
                m = metrics[split_name]
                auc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "  N/A "
                log.info(
                    f"{split_name:<15} {model_name:<20} "
                    f"{m['accuracy']:>8.4f} {m['f1_weighted']:>8.4f} {auc_str:>8}"
                )

    log.info("Evaluation complete.  Next: python -m src.explain.shap_explainer")


if __name__ == "__main__":
    main()
