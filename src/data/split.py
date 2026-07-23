"""
Pre-feature data splitting pipeline.
Runs BEFORE tokenization to prevent data leakage.
Uses temporal split when possible, falls back to article ID ordering.
PolitiFact is always reserved as an out-of-distribution test set.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Path constants
ROOT         = Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT / "data"
SPLITS_DIR   = DATA_DIR / "splits"
PARQUET_PATH = DATA_DIR / "processed_articles.parquet"

# Split configuration
TRAIN_RATIO         = 0.70
VAL_RATIO           = 0.15
TEST_RATIO          = 0.15
DATE_COVERAGE_THRESH = 0.80
MIN_SAMPLES_PER_CLASS = 5
RANDOM_SEED         = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _extract_numeric_id(article_id: str) -> int:
    """Extracts trailing integer from FakeNewsNet ID (e.g. 'gossipcop-4000' -> 4000)."""
    match = re.search(r"\d+", str(article_id))
    return int(match.group()) if match else 0


def _parse_date(date_str) -> pd.Timestamp:
    """Attempts to parse a date string, returns NaT on failure."""
    if pd.isna(date_str) or not str(date_str).strip():
        return pd.NaT
    try:
        return pd.to_datetime(str(date_str), utc=True)
    except (ValueError, TypeError):
        return pd.NaT


def _stratified_temporal_split(
    df: pd.DataFrame, sort_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sorts df by sort_col, then splits 70/15/15.
    Falls back to stratified random if a class disappears from any partition.
    """
    df_sorted = df.sort_values(sort_col).reset_index(drop=True)
    n = len(df_sorted)
    cut1 = int(n * TRAIN_RATIO)
    cut2 = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df_sorted.iloc[:cut1]
    val   = df_sorted.iloc[cut1:cut2]
    test  = df_sorted.iloc[cut2:]

    for name, part in [("train", train), ("val", val), ("test", test)]:
        if part["label"].nunique() < 2 and df["label"].nunique() >= 2:
            log.warning(
                f"Sorted split left {name} with only class "
                f"{part['label'].unique().tolist()}.  "
                "Falling back to stratified random split."
            )
            return _stratified_random_split(df)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _stratified_random_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback: stratified random split preserving class ratio."""
    train, temp = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val, test = train_test_split(
        temp,
        test_size=(1 - relative_val),
        random_state=RANDOM_SEED,
        stratify=temp["label"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def main() -> None:
    log.info("=" * 60)
    log.info("  Data Splitting Pipeline (Pre-Feature)")
    log.info("=" * 60)

    if not PARQUET_PATH.exists():
        log.critical(
            f"Parquet not found at {PARQUET_PATH}.  "
            "Run python -m src.data.hydrate first."
        )
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    log.info(f"Loaded {len(df)} articles from {PARQUET_PATH}")
    log.info(f"Source x Label distribution:\n{df.groupby(['source', 'label']).size().to_string()}")

    if df["label"].nunique() < 2:
        log.critical(
            "The dataset contains only one class.  "
            "Both fake and real articles are required.  "
            "Re-run hydrate with a higher --max-per-source."
        )
        sys.exit(1)

    # Separate GossipCop (train/val/test) and PolitiFact (OOD)
    gossipcop  = df[df["source"] == "gossipcop"].copy()
    politifact = df[df["source"] == "politifact"].copy()

    metadata = {
        "total_articles": len(df),
        "gossipcop_count": len(gossipcop),
        "politifact_count": len(politifact),
        "split_ratios": f"{TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}",
        "strategy_used": None,
        "date_coverage_pct": None,
        "class_distributions": {},
    }

    # Determine split strategy for GossipCop
    gc_has_both = len(gossipcop) > 0 and gossipcop["label"].nunique() >= 2

    if not gc_has_both:
        log.warning(
            "GossipCop data missing or has only one class.  "
            "Falling back to stratified random split across all data."
        )
        train_df, val_df, test_df = _stratified_random_split(df)
        ood_df = pd.DataFrame(columns=df.columns)
        metadata["strategy_used"] = "stratified_random_all_data"
    else:
        gossipcop["_parsed_date"] = gossipcop["publish_date"].apply(_parse_date)
        date_coverage = gossipcop["_parsed_date"].notna().mean()
        metadata["date_coverage_pct"] = round(float(date_coverage) * 100, 1)

        if date_coverage >= DATE_COVERAGE_THRESH:
            log.info(
                f"publish_date coverage: {date_coverage:.1%} >= {DATE_COVERAGE_THRESH:.0%} "
                "-> using temporal split."
            )
            gossipcop_sorted = gossipcop.copy()
            gossipcop_sorted["_sort_key"] = gossipcop_sorted["_parsed_date"]
            train_df, val_df, test_df = _stratified_temporal_split(
                gossipcop_sorted.drop(columns=["_parsed_date"]), "_sort_key"
            )
            for split in [train_df, val_df, test_df]:
                if "_sort_key" in split.columns:
                    split.drop(columns=["_sort_key"], inplace=True)
            metadata["strategy_used"] = "temporal_publish_date"
        else:
            log.info(
                f"publish_date coverage: {date_coverage:.1%} < {DATE_COVERAGE_THRESH:.0%} "
                "-> falling back to article_id ordinal split."
            )
            gossipcop["_sort_key"] = gossipcop["article_id"].apply(_extract_numeric_id)
            train_df, val_df, test_df = _stratified_temporal_split(
                gossipcop.drop(columns=["_parsed_date"]), "_sort_key"
            )
            for split in [train_df, val_df, test_df]:
                if "_sort_key" in split.columns:
                    split.drop(columns=["_sort_key"], inplace=True)
            metadata["strategy_used"] = "article_id_ordinal_proxy"

        # PolitiFact is always the OOD set
        ood_df = politifact.copy()
        if "_parsed_date" in ood_df.columns:
            ood_df.drop(columns=["_parsed_date"], inplace=True)

    for name, split_df in [
        ("train", train_df), ("val", val_df),
        ("test", test_df), ("ood_test", ood_df)
    ]:
        dist = split_df["label"].value_counts().to_dict() if len(split_df) > 0 else {}
        metadata["class_distributions"][name] = {
            "count": len(split_df),
            "labels": {str(k): int(v) for k, v in dist.items()},
        }
        log.info(f"  {name:10s}: {len(split_df):>5} articles -- {dist}")

    # Persist splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(SPLITS_DIR / "train.parquet",    engine="pyarrow", index=False)
    val_df.to_parquet(SPLITS_DIR / "val.parquet",        engine="pyarrow", index=False)
    test_df.to_parquet(SPLITS_DIR / "test.parquet",      engine="pyarrow", index=False)
    ood_df.to_parquet(SPLITS_DIR / "ood_test.parquet",   engine="pyarrow", index=False)

    with open(SPLITS_DIR / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Splits persisted -> {SPLITS_DIR}/")
    log.info(f"Strategy used: {metadata['strategy_used']}")
    log.info("Splitting complete.  Next step: python -m src.data.preprocess")


if __name__ == "__main__":
    main()
