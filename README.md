# Fake News Detection — Browser-Based Assistant

**Author:** Daniil Khmelnytskyi  
**Module:** Computing Project
**Institution:** Teesside University  
**Type:** Proof-of-Concept Prototype

---

## Overview

A browser-based assistant that uses machine learning and NLP to detect potentially deceptive news articles. The system provides an automated credibility signal directly within the user's browsing context via a Chrome extension.

**Core research question:** Can an NLP-based browsing assistant help users identify potentially false information in a way that is quick, clear, and practical for everyday browsing?

## Architecture

| Component | Technology | Role |
|---|---|---|
| Browser Extension | JavaScript (Chrome Manifest V3) | Extracts article text, displays results |
| Backend API | FastAPI + Uvicorn | Preprocessing → inference → SHAP → response |
| Primary Model | DistilBERT (fine-tuned) | Binary classification (fake vs. real) |
| Baseline Model | TF-IDF + Logistic Regression | Performance floor for comparison |
| Storage | Parquet files (pandas + pyarrow) | **No SQL/NoSQL databases** |
| Explainability | SHAP | LinearExplainer + KernelExplainer |

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## Pipeline Execution Order

```bash
# Step 1 — Ingest and augment dataset (WELFake dataset)
python -m src.data.augment --welFake-csv data/raw/welFake/WELFake_Dataset.csv
# (Note: src.data.hydrate exists for scraping FakeNewsNet URLs, but due to link rot
# the project data pipeline uses WELFake as its primary source)

# Step 2 — Split dataset (BEFORE tokenization to prevent data leakage)
python -m src.data.split

# Step 3 — Preprocess (clean text + tokenize/truncate to 512 tokens)
python -m src.data.preprocess

# Step 4 — Train baseline and SVM models
python -m src.models.baseline
python -m src.models.svm_classifier

# Step 5 — Train DistilBERT (fine-tuned with 256-token truncation)
python -m src.models.distilbert_classifier

# Step 6 — Evaluate models
python -m src.evaluation.evaluate

# Step 7 — Generate SHAP explanations
python -m src.explain.shap_explainer

# Step 8 — Start API server
uvicorn src.api.main:app --reload --port 8000

# Step 9 — Load Chrome extension
# Open chrome://extensions → Developer Mode → Load unpacked → select extension/
```

> **Note on Model Weights:** Transformer weights (`models/distilbert/model.safetensors`) are not tracked in the git repository and are generated during Step 5 (DistilBERT training).

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Analyze article text → prediction + SHAP explanation |
| `GET` | `/history` | Past analyses from Parquet store |
| `GET` | `/health` | Service health + model loading status |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists discover that water is wet."}'
```


## Constraints (by design)

- **No databases** — flat Parquet files only
- **No attention-based explanations** — SHAP only (Jain & Wallace 2019)
- **No complex chunking** — preprocessing truncates text to 512 tokens, but DistilBERT training and the inference API truncate to 256 tokens
- **No real-time fact-checking** — text feature classification only
- **English only** — no multi-language support
- **No authentication** — prototype scope

## Data Source

- **WELFake** dataset (Word Embedding over Languages for Fake News Detection)
- The pipeline was originally prototyped to scrape articles from FakeNewsNet URLs (`src/data/hydrate.py`), but due to high web attrition (link rot), the full dataset was sourced and augmented from WELFake (`src/data/augment.py`, documented in `data/augment_metadata.json`).
- Total dataset contains 62,720 articles (34,753 real, 27,967 fake).

## Limitations

- **Single Dataset and No OOD Evaluation**: The system is trained and evaluated on a single dataset (WELFake) with a 70/15/15 stratified random split. Because PolitiFact URLs could not be hydrated, the out-of-distribution split (`ood_test.parquet`) contains 0 samples, meaning there is no out-of-distribution evaluation.
- **In-Distribution Performance Only**: The reported ~98.9% test accuracy reflects in-distribution performance only and does not measure generalization across different news domains or publishers.
- **Source-Style Shortcut Artifacts**: The dataset contains noticeable publisher stylistic shortcuts. For example, the string `(Reuters)` appears in 60.55% of real articles (21,044 / 34,753 across all splits: 60.50% in train, 61.25% in val, 60.08% in test) and only 0.04% of fake articles (11 / 27,967 across all splits: 0.03% in train, 0.02% in val, 0.10% in test). Classifiers can easily exploit this lexical signal rather than evaluating news credibility.
- **Heuristic Ensemble Weights**: The ensemble weights in `src/api/main.py` (0.50 DistilBERT, 0.25 TF-IDF Baseline, 0.25 SVM) are configured by hand rather than tuned or learned.
- **Approximate Sentence-Level SHAP**: Sentence-level SHAP explanation uses a limited permutation budget (`SHAP_N_SAMPLES = 64`, capped at `SHAP_MAX_SENTENCES = 12`) to maintain interactive response times, providing an approximation of true Shapley values.

## License

Academic project — not for commercial use.
