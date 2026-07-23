"""
FakeNewsNet article hydration pipeline.
Scrapes articles from FakeNewsNet URLs, logs failures to CSV,
and writes results to data/processed_articles.parquet.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Path constants
ROOT              = Path(__file__).resolve().parents[2]
DATA_DIR          = ROOT / "data"
RAW_DIR           = DATA_DIR / "raw"
LOGS_DIR          = ROOT / "logs"
RAW_LINKS_PATH    = RAW_DIR / "raw_links.csv"
PARQUET_PATH      = DATA_DIR / "processed_articles.parquet"
LOSS_REPORT_PATH  = LOGS_DIR / "data_loss_report.csv"

# Scraping configuration
DEFAULT_MAX_PER_SOURCE = 300
REQUEST_TIMEOUT        = 12
POLITE_DELAY           = 0.5
MIN_TEXT_LENGTH        = 100

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Official FakeNewsNet GitHub raw CSV manifests
FAKENEWSNET_SOURCES: dict[str, str] = {
    "gossipcop_fake": (
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet"
        "/master/dataset/gossipcop_fake.csv"
    ),
    "gossipcop_real": (
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet"
        "/master/dataset/gossipcop_real.csv"
    ),
    "politifact_fake": (
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet"
        "/master/dataset/politifact_fake.csv"
    ),
    "politifact_real": (
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet"
        "/master/dataset/politifact_real.csv"
    ),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# Step 1 -- Download raw link manifests

def download_raw_links(max_per_source: int) -> pd.DataFrame:
    """
    Fetches the four FakeNewsNet CSV manifests from GitHub and merges them
    into a single DataFrame. Saves to data/raw/raw_links.csv for caching.
    """
    if RAW_LINKS_PATH.exists():
        log.info(
            "data/raw/raw_links.csv found on disk -- loading cached copy.  "
            "Delete to force a fresh download."
        )
        return pd.read_csv(RAW_LINKS_PATH)

    log.info("Downloading FakeNewsNet manifests from GitHub...")
    frames: list[pd.DataFrame] = []

    for source_name, url in FAKENEWSNET_SOURCES.items():
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
            resp.raise_for_status()

            df = pd.read_csv(StringIO(resp.text))
            df = df.rename(columns={"id": "article_id", "news_url": "url"})
            df = df[["article_id", "url"]].dropna(subset=["url"])

            # Shuffle to remove ordering artefacts, then cap
            df = (
                df.sample(frac=1, random_state=42)
                  .head(max_per_source)
                  .reset_index(drop=True)
            )

            dataset, polarity = source_name.split("_", 1)
            df["source"] = dataset
            df["label"]  = "fake" if polarity == "fake" else "real"

            frames.append(df)
            log.info(f"  {source_name}: {len(df)} articles")

        except requests.exceptions.RequestException as exc:
            log.error(f"  Failed to download manifest '{source_name}': {exc}")

    if not frames:
        log.critical("All manifest downloads failed.  Check network connectivity.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RAW_LINKS_PATH, index=False)
    log.info(f"Saved {len(combined)} raw links -> {RAW_LINKS_PATH}")
    return combined


# Step 2 -- Single-article scraper

def _scrape_article(url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fetches a news article URL and extracts title + body text.
    Returns (title, text, publish_date) -- any may be None on failure.
    """
    resp = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers=HEADERS,
        allow_redirects=True,
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "html.parser")

    for noise_tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "form"]
    ):
        noise_tag.decompose()

    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    paragraphs = soup.find_all("p")
    raw_text   = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
    cleaned    = re.sub(r"\s+", " ", raw_text).strip()

    text = cleaned if len(cleaned) >= MIN_TEXT_LENGTH else None

    publish_date = None
    for attr in ["article:published_time", "datePublished", "pubdate", "date"]:
        meta = soup.find("meta", attrs={"property": attr}) or \
               soup.find("meta", attrs={"name": attr})
        if meta and meta.get("content"):
            publish_date = meta["content"].strip()
            break
    if publish_date is None:
        time_el = soup.find("time", attrs={"datetime": True})
        if time_el:
            publish_date = time_el["datetime"].strip()

    return title, text, publish_date


# Step 3 -- Hydration loop with granular error handling

def hydrate_articles(
    raw_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """
    Iterates over every row in raw_df and attempts to scrape article text.
    Returns (successful, losses) -- both are lists of dicts.
    """
    successful: list[dict] = []
    losses: list[dict]     = []

    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Hydrating articles"):
        article_id = str(row["article_id"])
        url        = str(row["url"])
        label      = str(row["label"])
        source     = str(row["source"])
        now_iso    = datetime.now(timezone.utc).isoformat()

        try:
            title, text, publish_date = _scrape_article(url)
            if text is None:
                raise ValueError(
                    f"Extracted text below minimum length ({MIN_TEXT_LENGTH} chars)"
                )
            successful.append({
                "article_id":      article_id,
                "source":          source,
                "label":           label,
                "title":           title or "",
                "text_raw":        text,
                "publish_date":    publish_date,
                "scrape_timestamp": now_iso,
            })

        except requests.exceptions.Timeout:
            losses.append({
                "article_id":    article_id,
                "url":           url,
                "source":        source,
                "error_type":    "Timeout",
                "error_message": f"Request exceeded {REQUEST_TIMEOUT}s timeout.",
                "timestamp":     now_iso,
            })

        except requests.exceptions.HTTPError as exc:
            losses.append({
                "article_id":    article_id,
                "url":           url,
                "source":        source,
                "error_type":    "HTTPError",
                "error_message": str(exc)[:300],
                "timestamp":     now_iso,
            })

        except requests.exceptions.TooManyRedirects as exc:
            losses.append({
                "article_id":    article_id,
                "url":           url,
                "source":        source,
                "error_type":    "TooManyRedirects",
                "error_message": str(exc)[:300],
                "timestamp":     now_iso,
            })

        except requests.exceptions.ConnectionError as exc:
            losses.append({
                "article_id":    article_id,
                "url":           url,
                "source":        source,
                "error_type":    "ConnectionError",
                "error_message": str(exc)[:300],
                "timestamp":     now_iso,
            })

        except requests.exceptions.RequestException as exc:
            losses.append({
                "article_id":    article_id,
                "url":           url,
                "source":        source,
                "error_type":    "RequestException",
                "error_message": str(exc)[:300],
                "timestamp":     now_iso,
            })

        except ValueError as exc:
            losses.append({
                "article_id":    article_id,
                "url":           url,
                "source":        source,
                "error_type":    "EmptyContent",
                "error_message": str(exc),
                "timestamp":     now_iso,
            })

        except Exception as exc:
            losses.append({
                "article_id":    article_id,
                "url":           url,
                "source":        source,
                "error_type":    type(exc).__name__,
                "error_message": str(exc)[:300],
                "timestamp":     now_iso,
            })

        time.sleep(POLITE_DELAY)

    return successful, losses


# Step 4 -- Persist results

def save_parquet(records: list[dict]) -> None:
    """Writes successfully scraped articles to Parquet with Snappy compression."""
    if not records:
        log.warning("No successful articles -- skipping Parquet write.")
        return

    df = pd.DataFrame(records)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, engine="pyarrow", compression="snappy", index=False)
    log.info(f"Saved {len(df)} articles -> {PARQUET_PATH}")
    log.info(
        f"Label distribution:\n"
        f"{df.groupby(['source', 'label']).size().to_string()}"
    )


def save_loss_report(losses: list[dict]) -> None:
    """Writes all failed URL fetches to logs/data_loss_report.csv."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        losses if losses else [],
        columns=["article_id", "url", "source", "error_type", "error_message", "timestamp"],
    )
    df.to_csv(LOSS_REPORT_PATH, index=False)
    log.info(f"Loss report: {len(df)} failures -> {LOSS_REPORT_PATH}")

    if losses:
        breakdown = pd.Series([l["error_type"] for l in losses]).value_counts()
        log.info(f"Error breakdown:\n{breakdown.to_string()}")


# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FakeNewsNet hydration pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=DEFAULT_MAX_PER_SOURCE,
        help="Max articles to sample per source CSV.  Use 50 for a quick test run.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  FakeNewsNet Hydration Pipeline")
    log.info("=" * 60)
    log.info(f"  Articles per source  : {args.max_per_source}")
    log.info(f"  Request timeout      : {REQUEST_TIMEOUT}s")
    log.info(f"  Polite delay         : {POLITE_DELAY}s / request")
    log.info(
        f"  Est. runtime (worst) : "
        f"~{int(args.max_per_source * 4 * (POLITE_DELAY + 2) / 60)} min"
    )

    raw_df = download_raw_links(args.max_per_source)
    log.info(f"Total URLs to attempt: {len(raw_df)}")

    successful, losses = hydrate_articles(raw_df)

    total  = len(raw_df)
    n_ok   = len(successful)
    n_fail = len(losses)
    rate   = (n_fail / total * 100) if total else 0

    log.info(f"  Total attempted : {total:>5}")
    log.info(f"  Successful      : {n_ok:>5}")
    log.info(f"  Failed          : {n_fail:>5}  ({rate:.1f}% loss rate)")

    save_parquet(successful)
    save_loss_report(losses)

    log.info("Hydration complete.  Next step: python -m src.data.split")


if __name__ == "__main__":
    main()
