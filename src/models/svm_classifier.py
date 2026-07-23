"""
TF-IDF + Linear SVM classifier with probability calibration.
C parameter tuned via 5-fold GridSearchCV, then wrapped with
CalibratedClassifierCV to enable predict_proba().
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Path constants
ROOT       = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models" / "svm"
SVM_PATH   = MODELS_DIR / "tfidf_svm.joblib"
LOGS_DIR   = ROOT / "logs"

RANDOM_SEED = 42

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
    log.info("  TF-IDF + Linear SVM Classifier")
    log.info("=" * 60)

    train_df    = _load_split("train")
    val_df      = _load_split("val")
    test_df     = _load_split("test")

    ood_path = SPLITS_DIR / "ood_test.parquet"
    ood_test_df = pd.read_parquet(ood_path, engine="pyarrow") if ood_path.exists() else pd.DataFrame()

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
        ("clf", LinearSVC(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            dual="auto",
        )),
    ])

    X_train = train_df["text_truncated"].fillna("").tolist()
    y_train = _encode_labels(train_df["label"])

    param_grid = {"clf__C": [0.01, 0.1, 0.5, 1.0, 5.0]}
    log.info("Running 5-fold GridSearchCV over C in [0.01, 0.1, 0.5, 1.0, 5.0]...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5,
        scoring="f1_weighted", n_jobs=-1, verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_C = grid_search.best_params_["clf__C"]
    log.info(f"Best C = {best_C}  |  CV F1 = {grid_search.best_score_:.4f}")
    pipeline = grid_search.best_estimator_

    # Wrap with CalibratedClassifierCV so we get probability output
    log.info("Calibrating probabilities (CalibratedClassifierCV, 5-fold)...")
    calibrated = CalibratedClassifierCV(
        pipeline.named_steps["clf"],
        cv=5,
        method="sigmoid",
    )
    X_train_tfidf = pipeline.named_steps["tfidf"].transform(X_train)
    calibrated.fit(X_train_tfidf, y_train)

    from sklearn.pipeline import make_pipeline
    final_pipeline = make_pipeline(pipeline.named_steps["tfidf"], calibrated)

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
        y_pred = final_pipeline.predict(X)

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
        "model":      "TF-IDF + Linear SVM (CalibratedClassifierCV)",
        "config": {
            "max_features":   100_000,
            "ngram_range":    "(1, 2)",
            "sublinear_tf":   True,
            "best_C":         best_C,
            "cv_folds":       5,
            "class_weight":   "balanced",
            "calibration":    "sigmoid (5-fold)",
        },
        "results": eval_results,
    }
    log_path = LOGS_DIR / "svm_metrics.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metric_log, f, indent=2)
    log.info(f"  Metric log saved -> {log_path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, SVM_PATH)
    log.info(f"SVM saved -> {SVM_PATH}")
    log.info("SVM training complete.")


if __name__ == "__main__":
    main()
