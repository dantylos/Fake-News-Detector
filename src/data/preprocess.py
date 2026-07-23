"""
Text cleaning and tokenization pipeline.
Runs AFTER split.py -- adds a text_truncated column to each split Parquet.
Truncation at 512 tokens is decoded back to plain text so the data
is not coupled to a specific tokenizer version.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

# Path constants
ROOT       = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"

# Configuration
TOKENIZER_NAME  = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 512

BOILERPLATE_PATTERNS = [
    r"subscribe to our newsletter.*",
    r"share on (facebook|twitter|linkedin).*",
    r"click here to.*",
    r"follow us on.*",
    r"sign up for.*",
    r"advertisement\s*",
    r"sponsored content\s*",
    r"related articles?:?\s*",
    r"read more:?\s*",
    r"copyright ©.*",
    r"all rights reserved.*",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def clean_article_text(raw: str) -> str:
    """Cleans raw scraped text: strips HTML, removes boilerplate, normalizes whitespace."""
    if not raw or not isinstance(raw, str):
        return ""

    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_truncate(text: str, tokenizer, max_length: int = MAX_TOKEN_LENGTH) -> str:
    """
    Tokenizes text with DistilBERT, truncates to max_length tokens,
    then decodes back to human-readable text so the stored data isn't
    tied to a specific tokenizer version.
    """
    if not text:
        return ""

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    truncated_text = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    return truncated_text


def process_split(split_path: Path, tokenizer) -> None:
    """Loads a split Parquet, cleans text, truncates, and saves back."""
    if not split_path.exists():
        log.warning(f"Split not found: {split_path} -- skipping.")
        return

    df = pd.read_parquet(split_path, engine="pyarrow")

    if len(df) == 0:
        log.info(f"  {split_path.stem}: 0 rows -- skipping.")
        df["text_truncated"] = pd.Series(dtype=str)
        df.to_parquet(split_path, engine="pyarrow", index=False)
        return

    df["text_raw"] = df["text_raw"].fillna("").apply(clean_article_text)

    df["text_truncated"] = df["text_raw"].apply(
        lambda t: tokenize_and_truncate(t, tokenizer)
    )

    df.to_parquet(split_path, engine="pyarrow", index=False)
    log.info(f"  {split_path.stem}: {len(df)} articles processed.")


def main() -> None:
    log.info("=" * 60)
    log.info("  Text Preprocessing Pipeline")
    log.info("=" * 60)

    if not SPLITS_DIR.exists():
        log.critical(
            f"Splits directory not found: {SPLITS_DIR}.  "
            "Run python -m src.data.split first."
        )
        sys.exit(1)

    log.info(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    log.info(f"Tokenizer loaded.  Max length: {MAX_TOKEN_LENGTH} tokens.")

    split_files = ["train.parquet", "val.parquet", "test.parquet", "ood_test.parquet"]

    for fname in split_files:
        process_split(SPLITS_DIR / fname, tokenizer)

    log.info("Preprocessing complete.  Next step: python -m src.models.baseline")


if __name__ == "__main__":
    main()
