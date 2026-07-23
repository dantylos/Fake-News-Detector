"""
Dataset augmentation -- merges WELFake with scraped FakeNewsNet articles.
Downloads WELFake from Kaggle and outputs an augmented processed_articles.parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Path constants
ROOT            = Path(__file__).resolve().parents[2]
DATA_DIR        = ROOT / "data"
EXISTING_PATH   = DATA_DIR / "processed_articles.parquet"
OUTPUT_PATH     = DATA_DIR / "processed_articles.parquet"
META_PATH       = DATA_DIR / "augment_metadata.json"
KAGGLE_DEST     = DATA_DIR / "raw" / "welFake"
WELFACE_CSV     = KAGGLE_DEST / "WELFake_Dataset.csv"

MAX_WORDS = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _truncate(text: str, max_words: int = MAX_WORDS) -> str:
    """Truncate to first max_words words, matching existing preprocessing."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join(words[:max_words])


def _download_welFake() -> Path:
    """Download WELFake CSV via the Kaggle API into KAGGLE_DEST."""
    import subprocess
    import shutil

    KAGGLE_DEST.mkdir(parents=True, exist_ok=True)

    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        venv_kaggle = Path(sys.executable).parent / "kaggle.exe"
        kaggle_bin = str(venv_kaggle) if venv_kaggle.exists() else None
    if kaggle_bin is None:
        log.critical("kaggle executable not found. Install via: pip install kaggle")
        sys.exit(1)

    log.info(f"Downloading WELFake from Kaggle using: {kaggle_bin}")
    result = subprocess.run(
        [
            kaggle_bin,
            "datasets", "download",
            "-d", "saurabhshahane/fake-news-classification",
            "-p", str(KAGGLE_DEST),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"Kaggle download failed:\n{result.stderr}")
        sys.exit(1)

    log.info(f"Downloaded to {KAGGLE_DEST}")

    csv_files = list(KAGGLE_DEST.glob("*.csv"))
    if not csv_files:
        log.critical(f"No CSV found in {KAGGLE_DEST} after download.")
        sys.exit(1)

    return csv_files[0]


def _load_welFake(csv_path: Path) -> pd.DataFrame:
    """Load WELFake CSV and normalise to the project schema."""
    log.info(f"Loading WELFake from {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    log.info(f"  Raw WELFake rows: {len(df):,}")

    required = {"title", "text", "label"}
    missing  = required - set(df.columns.str.lower())
    if missing:
        log.critical(f"WELFake missing columns: {missing}.  Got: {df.columns.tolist()}")
        sys.exit(1)

    df.columns = df.columns.str.lower()

    # WELFake uses 0=real, 1=fake
    df["label"] = df["label"].map({0: "real", 1: "fake"})
    df = df.dropna(subset=["label", "text"])

    df["title"]  = df["title"].fillna("").astype(str)
    df["text"]   = df["text"].fillna("").astype(str)
    df["full_text"] = (df["title"] + " " + df["text"]).str.strip()

    df["text_truncated"] = df["full_text"].apply(_truncate)

    df = df[df["text_truncated"].str.split().str.len() >= 20]
    df["source"] = "welFake"

    return df[["text_truncated", "label", "source"]]


def _load_existing() -> pd.DataFrame:
    """Load and normalise the existing processed_articles.parquet."""
    if not EXISTING_PATH.exists():
        log.warning("No existing processed_articles.parquet found -- using WELFake only.")
        return pd.DataFrame(columns=["text_truncated", "label", "source"])

    df = pd.read_parquet(EXISTING_PATH, engine="pyarrow")
    log.info(f"  Existing articles: {len(df):,}")

    if "text_truncated" not in df.columns:
        log.warning("Existing parquet lacks 'text_truncated'; skipping it.")
        return pd.DataFrame(columns=["text_truncated", "label", "source"])

    if "source" not in df.columns:
        df["source"] = "scraped"

    return df[["text_truncated", "label", "source"]]


def main(welFake_csv: Path | None = None) -> None:
    log.info("=" * 60)
    log.info("  WELFake Dataset Augmentation")
    log.info("=" * 60)

    if welFake_csv is None:
        if WELFACE_CSV.exists():
            log.info(f"Found cached WELFake CSV at {WELFACE_CSV}")
            welFake_csv = WELFACE_CSV
        else:
            welFake_csv = _download_welFake()

    existing_df = _load_existing()
    welFake_df  = _load_welFake(welFake_csv)

    combined = pd.concat([existing_df, welFake_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text_truncated"])
    combined = combined.dropna(subset=["text_truncated", "label"])

    log.info(f"\n  Combined size : {len(combined):,}")
    label_counts = combined["label"].value_counts()
    log.info(f"  Real articles : {label_counts.get('real', 0):,}")
    log.info(f"  Fake articles : {label_counts.get('fake', 0):,}")

    imbalance = label_counts.max() / label_counts.sum()
    log.info(f"  Class balance : {imbalance:.1%} majority class")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)
    log.info(f"\n  Saved -> {OUTPUT_PATH}")

    metadata = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "total_articles":   int(len(combined)),
        "real_count":       int(label_counts.get("real", 0)),
        "fake_count":       int(label_counts.get("fake", 0)),
        "imbalance_ratio":  float(round(imbalance, 4)),
        "sources": {
            s: int(n) for s, n in combined["source"].value_counts().items()
        },
        "welFake_csv_used": str(welFake_csv),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Metadata -> {META_PATH}")

    log.info("Augmentation complete.  Next: python -m src.data.split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset with WELFake")
    parser.add_argument(
        "--welFake-csv",
        type=Path,
        default=None,
        help="Path to WELFake_Dataset.csv (skip Kaggle download if provided)",
    )
    args = parser.parse_args()
    main(welFake_csv=args.welFake_csv)
