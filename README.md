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
# Step 1 — Hydrate dataset (scrape FakeNewsNet articles)
python -m src.data.hydrate --max-per-source 300    # ~15-20 min
python -m src.data.hydrate --max-per-source 50     # quick test (~2 min)

# Step 2 — Split dataset (BEFORE any feature extraction)
python -m src.data.split

# Step 3 — Preprocess (clean + tokenize/truncate)
python -m src.data.preprocess

# Step 4 — Train baseline model
python -m src.models.baseline

# Step 5 — Train DistilBERT (30-60 min on CPU)
python -m src.models.distilbert_classifier

# Step 6 — Evaluate both models
python -m src.evaluation.evaluate

# Step 7 — Generate SHAP explanations
python -m src.explain.shap_explainer

# Step 8 — Start API server
uvicorn src.api.main:app --reload --port 8000

# Step 9 — Load Chrome extension
# Open chrome://extensions → Developer Mode → Load unpacked → select extension/
```

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

## Project Structure

```
fake-new-detector/
├── data/                         # Data artifacts
├── logs/                         # Reports and metrics
├── models/                       # Trained model checkpoints
├── tests/                        # Tests
├── src/
│   ├── data/                     # Hydration, splitting, preprocessing
│   ├── models/                   # TF-IDF baseline + DistilBERT
│   ├── explain/                  # SHAP explainability
│   ├── api/                      # FastAPI backend
│   └── evaluation/               # Metrics computation
├── extension/                    # Chrome Extension (Manifest V3)
├── requirements.txt
└── README.md
```

## Constraints (by design)

- **No databases** — flat Parquet files only
- **No attention-based explanations** — SHAP only (Jain & Wallace 2019)
- **No complex chunking** — 512-token truncation
- **No real-time fact-checking** — text feature classification only
- **English only** — no multi-language support
- **No authentication** — prototype scope

## Data Source

- **FakeNewsNet** (https://github.com/KaiDMML/FakeNewsNet)
- Articles hydrated from original URLs (high attrition rate expected)
- Data loss documented in `logs/data_loss_report.csv`

## License

Academic project — not for commercial use.
