"""
TF-IDF + Logistic Regression baseline model.
Establishes a performance floor for comparison with DistilBERT.
C parameter tuned via 5-fold GridSearchCV.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import json
import pandas as pd
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Path constants
ROOT          = Path(__file__).resolve().parents[2]
SPLITS_DIR    = ROOT / "data" / "splits"
MODELS_DIR    = ROOT / "models" / "baseline"
BASELINE_PATH = MODELS_DIR / "tfidf_logreg.joblib"

RANDOM_SEED   = 42
LOGS_DIR      = ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _load_split(name: str) -> pd.DataFrame:
    """Loads a split Parquet and validates it has required columns."""
    path = SPLITS_DIR / f"{name}.parquet"
    if not path.exists():
        log.critical(f"Split not found: {path}.  Run python -m src.data.split first.")
        sys.exit(1)
    df = pd.read_parquet(path, engine="pyarrow")
    if "text_truncated" not in df.columns:
        log.critical(
            f"Column 'text_truncated' missing in {name}.parquet.  "
            "Run python -m src.data.preprocess first."
        )
        sys.exit(1)
    return df


def _encode_labels(series: pd.Series) -> list[int]:
    return [1 if l == "fake" else 0 for l in series]


def main() -> None:
    log.info("=" * 60)
    log.info("  TF-IDF + Logistic Regression Baseline")
    log.info("=" * 60)

    train_df    = _load_split("train")
    val_df      = _load_split("val")
    test_df     = _load_split("test")
    ood_test_df = _load_split("ood_test")

    log.info(
        f"Splits loaded -- train: {len(train_df)} | val: {len(val_df)} | "
        f"test: {len(test_df)} | ood: {len(ood_test_df)}"
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])

    X_train = train_df["text_truncated"].fillna("").tolist()
    y_train = _encode_labels(train_df["label"])

    param_grid = {"clf__C": [0.1, 0.5, 1.0, 5.0, 10.0]}
    log.info("Running 5-fold GridSearchCV over C in [0.1, 0.5, 1.0, 5.0, 10.0]...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5,
        scoring="f1_weighted", n_jobs=-1, verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_C = grid_search.best_params_["clf__C"]
    log.info(f"Best C = {best_C}  |  CV F1 = {grid_search.best_score_:.4f}")
    pipeline = grid_search.best_estimator_

    # Evaluate on all splits
    eval_results: dict = {}
    for split_name, split_df in [
        ("val         (validation)", val_df),
        ("test        (ID-test)",    test_df),
        ("ood_test    (OOD-test)",   ood_test_df),
    ]:
        if len(split_df) == 0:
            log.info(f"  {split_name}: SKIPPED (no data)")
            continue

        X = split_df["text_truncated"].fillna("").tolist()
        y_true = _encode_labels(split_df["label"])
        y_pred = pipeline.predict(X)

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        log.info(f"  {split_name}: acc={acc:.4f}  f1={f1:.4f}")
        log.info(
            "\n" + classification_report(
                y_true, y_pred, target_names=["real", "fake"], zero_division=0
            )
        )
        short_name = split_name.split()[0].strip()
        eval_results[short_name] = {"accuracy": round(acc, 4), "f1": round(f1, 4)}

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    metric_log = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "model":      "TF-IDF + Logistic Regression (v2)",
        "config": {
            "max_features":   100_000,
            "ngram_range":    "(1, 2)",
            "sublinear_tf":   True,
            "best_C":         best_C,
            "cv_folds":       5,
            "class_weight":   "balanced",
        },
        "results": eval_results,
    }
    log_path = LOGS_DIR / "baseline_metrics.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metric_log, f, indent=2)
    log.info(f"  Metric log saved -> {log_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BASELINE_PATH)
    log.info(f"Baseline saved -> {BASELINE_PATH}")
    log.info("Baseline training complete.  Next: python -m src.models.distilbert_classifier")


if __name__ == "__main__":
    main()
