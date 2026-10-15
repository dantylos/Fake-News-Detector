"""
FastAPI backend for the fake news detection system.
Exposes /analyze, /history, and /health endpoints.
No database -- history stored in data/analysis_history.parquet.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Path constants
ROOT             = Path(__file__).resolve().parents[2]

# Ensure project root is on sys.path so `from src.*` imports work
# when running as `python src/api/main.py` from any directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR       = ROOT / "models"
DATA_DIR         = ROOT / "data"
BASELINE_PATH    = MODELS_DIR / "baseline" / "tfidf_logreg.joblib"
SVM_PATH         = MODELS_DIR / "svm" / "tfidf_svm.joblib"
TRANSFORMER_PATH = MODELS_DIR / "distilbert"
HISTORY_PATH     = DATA_DIR / "analysis_history.parquet"

# Configuration
MAX_TOKEN_LENGTH = 256
SHAP_TOP_N       = 5

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
)
log = logging.getLogger(__name__)

# Thread lock for safe Parquet writes
_history_lock = threading.Lock()


# Pydantic schemas

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Full article text or extracted content")
    url: Optional[str] = Field(None, description="Original article URL (for logging)")


class SentenceExplanation(BaseModel):
    sentence: str
    shap_value: float


class ExplanationPayload(BaseModel):
    top_sentences: list[SentenceExplanation]


class AnalyzeResponse(BaseModel):
    prediction: str = Field(..., description="'fake' or 'real'")
    confidence: float = Field(..., description="Confidence in the predicted class ∈ [0, 1]")
    explanation: ExplanationPayload
    machine_evaluation: str = Field(..., description="A natural language summary of the model's reasoning based on SHAP.")
    model_version: str
    timestamp: str
    disclaimer: str = (
        "This is an automated assessment and not a definitive truth judgment. "
        "Always cross-reference with trusted sources."
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict


# Lazy model loading (cached singletons)

@lru_cache(maxsize=1)
def _load_baseline():
    """Loads TF-IDF + LR pipeline."""
    if not BASELINE_PATH.exists():
        log.warning(f"Baseline model not found at {BASELINE_PATH}")
        return None
    try:
        import joblib
        pipeline = joblib.load(BASELINE_PATH)
        log.info(f"TF-IDF baseline loaded from {BASELINE_PATH}")
        return pipeline
    except Exception as exc:
        log.error(f"Failed to load baseline: {exc}")
        return None


@lru_cache(maxsize=1)
def _load_svm():
    """Loads TF-IDF + Linear SVM (calibrated) pipeline."""
    if not SVM_PATH.exists():
        log.warning(f"SVM model not found at {SVM_PATH}")
        return None
    try:
        import joblib
        pipeline = joblib.load(SVM_PATH)
        log.info(f"SVM loaded from {SVM_PATH}")
        return pipeline
    except Exception as exc:
        log.error(f"Failed to load SVM: {exc}")
        return None


@lru_cache(maxsize=1)
def _load_distilbert():
    """Loads fine-tuned DistilBERT model."""
    if not TRANSFORMER_PATH.exists():
        log.warning(f"DistilBERT model not found at {TRANSFORMER_PATH}")
        return None, None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))
        model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_PATH))
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        log.info(f"DistilBERT loaded from {TRANSFORMER_PATH}")
        return model, tokenizer
    except Exception as exc:
        log.error(f"Failed to load DistilBERT: {exc}")
        return None, None


# Text preprocessing

def _clean_text(text: str) -> str:
    """Basic text cleaning for incoming API requests."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Prediction logic

def _predict_baseline(text: str) -> Optional[float]:
    """Returns P(fake) from TF-IDF + LR."""
    pipeline = _load_baseline()
    if pipeline is None:
        return None
    try:
        proba = pipeline.predict_proba([text])[0]
        return float(proba[1])
    except Exception as exc:
        log.warning(f"Baseline prediction failed: {exc}")
        return None


def _predict_svm(text: str) -> Optional[float]:
    """Returns P(fake) from TF-IDF + SVM."""
    pipeline = _load_svm()
    if pipeline is None:
        return None
    try:
        proba = pipeline.predict_proba([text])[0]
        return float(proba[1])
    except Exception as exc:
        log.warning(f"SVM prediction failed: {exc}")
        return None


def _predict_distilbert(text: str) -> Optional[float]:
    """Returns P(fake) from DistilBERT."""
    model, tokenizer = _load_distilbert()
    if model is None:
        return None
    try:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=MAX_TOKEN_LENGTH,
        )
        enc.pop("token_type_ids", None)
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return float(proba[1])
    except Exception as exc:
        log.warning(f"DistilBERT prediction failed: {exc}")
        return None


# SHAP explanation

def _get_shap_explanation(text: str) -> list[dict]:
    """Gets SHAP sentence-level explanation for a given text."""
    try:
        from src.explain.shap_explainer import explain_article, _split_sentences
        sentences = _split_sentences(text)
        log.info(f"SHAP: text length={len(text)}, sentences found={len(sentences)}")
        if len(sentences) < 2:
            log.warning(f"SHAP: skipped — only {len(sentences)} sentence(s) after split")
            return []
        result = explain_article(text, model_name="distilbert", top_n=SHAP_TOP_N)
        top = result.get("top_sentences", [])
        if not top:
            result = explain_article(text, model_name="baseline", top_n=SHAP_TOP_N)
            top = result.get("top_sentences", [])
        log.info(f"SHAP: returned {len(top)} explanations")
        return top
    except Exception as exc:
        log.warning(f"SHAP explanation failed: {exc}", exc_info=True)
        return []


def _generate_machine_evaluation(prediction: str, confidence: float, shap_results: list[dict]) -> str:
    """Generates a dynamic natural language summary of the model's reasoning."""
    pct = round(confidence * 100)

    if prediction == "fake":
        if confidence >= 0.85:
            summary = f"The ensemble model is highly confident ({pct}%) that this article exhibits patterns of misinformation. "
        elif confidence >= 0.65:
            summary = f"The ensemble model suspects ({pct}%) this article may be unreliable. "
        else:
            summary = f"The ensemble model leans slightly towards categorizing this as fake ({pct}%), but remains largely inconclusive. "
    else:
        if confidence >= 0.85:
            summary = f"The ensemble model is highly confident ({pct}%) that this article follows credible reporting patterns. "
        elif confidence >= 0.65:
            summary = f"The ensemble model leans towards this article being credible ({pct}%), though some patterns are ambiguous. "
        else:
            summary = f"The ensemble model is uncertain, but slightly leans towards categorizing this as real ({pct}%). "

    if not shap_results:
        return summary + "However, specific explanatory passages could not be extracted."

    top_factor = shap_results[0]
    top_sentence = top_factor["sentence"].strip()
    if len(top_sentence) > 100:
        top_sentence = top_sentence[:97] + "..."

    sign = top_factor["shap_value"]

    if abs(sign) < 1e-4:
        summary += "Sentence-level SHAP did not isolate a dominant passage; the verdict comes from the article as a whole. "
        return summary

    is_aligned = (prediction == "fake" and sign > 0) or (prediction == "real" and sign < 0)

    if is_aligned:
        summary += f"The decision was strongly driven by the presence of phrases like: \"{top_sentence}\". "
    else:
        summary += f"Interestingly, despite the overall '{prediction}' prediction, passages like \"{top_sentence}\" contained conflicting signals. "

    if len(shap_results) > 1:
        summary += "The SHAP analysis identified several other key passages throughout the text that collectively pushed the probability towards the final verdict."

    return summary


# History persistence (Parquet, no databases)

def _append_to_history(record: dict) -> None:
    """Appends an analysis record to the Parquet history file."""
    with _history_lock:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            new_row = pd.DataFrame([record])

            if HISTORY_PATH.exists():
                existing = pd.read_parquet(HISTORY_PATH, engine="pyarrow")
                combined = pd.concat([existing, new_row], ignore_index=True)
            else:
                combined = new_row

            combined.to_parquet(HISTORY_PATH, engine="pyarrow",
                                compression="snappy", index=False)
        except Exception as exc:
            log.error(f"Failed to write history: {exc}")


# FastAPI application

app = FastAPI(
    title="Fake News Detection API",
    description=(
        "API for detecting fake news using ML models (TF-IDF + DistilBERT).  "
        "Part of the Computing Project by Daniil Khmelnytskyi — PRA3017-N."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(request: AnalyzeRequest):
    """
    Analyzes article text for fake news indicators.
    Pipeline: clean text -> model inference -> SHAP explanation -> response.
    """
    if not request.text or len(request.text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Article text must be at least 50 characters.",
        )

    cleaned = _clean_text(request.text)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Run predictions from all available models
    p_baseline   = _predict_baseline(cleaned)
    p_svm        = _predict_svm(cleaned)
    p_distilbert = _predict_distilbert(cleaned)

    # Weighted ensemble: DistilBERT gets most weight (highest accuracy)
    scores = []
    weights = []
    if p_distilbert is not None:
        scores.append(p_distilbert)
        weights.append(0.50)
    if p_svm is not None:
        scores.append(p_svm)
        weights.append(0.25)
    if p_baseline is not None:
        scores.append(p_baseline)
        weights.append(0.25)

    if scores:
        w_total = sum(weights)
        p_fake = sum(s * w / w_total for s, w in zip(scores, weights))
        n_models = len(scores)
        if n_models == 3:
            model_ver = "ensemble-v3 (DistilBERT + SVM + LR)"
        elif n_models == 2:
            model_ver = "ensemble-v3 (partial)"
        else:
            names = {p_distilbert: "distilbert-v2", p_svm: "svm-v1", p_baseline: "tfidf-baseline-v2"}
            model_ver = names.get(scores[0], "unknown")
    else:
        raise HTTPException(
            status_code=503,
            detail="No trained models available.  Run the training pipeline first.",
        )

    prediction = "fake" if p_fake >= 0.5 else "real"
    confidence = p_fake if prediction == "fake" else (1.0 - p_fake)

    # SHAP explanation and machine evaluation
    shap_results = _get_shap_explanation(cleaned)
    machine_eval = _generate_machine_evaluation(prediction, confidence, shap_results)

    # Log to history
    _append_to_history({
        "url":         request.url or "",
        "text_preview": cleaned[:200],
        "prediction":  prediction,
        "p_fake":      round(p_fake, 4),
        "confidence":  round(confidence, 4),
        "model_used":  model_ver,
        "timestamp":   now_iso,
    })

    return AnalyzeResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        explanation=ExplanationPayload(
            top_sentences=[
                SentenceExplanation(**s) for s in shap_results
            ]
        ),
        machine_evaluation=machine_eval,
        model_version=model_ver,
        timestamp=now_iso,
    )


@app.get("/history")
async def get_history(limit: int = 20):
    """Returns past analyses from the Parquet store."""
    if not HISTORY_PATH.exists():
        return []

    try:
        df = pd.read_parquet(HISTORY_PATH, engine="pyarrow")
        df = df.sort_values("timestamp", ascending=False).head(limit)
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read history: {exc}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns service health and model loading status."""
    baseline_ok = _load_baseline() is not None
    svm_ok = _load_svm() is not None
    model, _ = _load_distilbert()
    distilbert_ok = model is not None

    return HealthResponse(
        status="ok",
        models_loaded={
            "tfidf_baseline": baseline_ok,
            "svm": svm_ok,
            "distilbert": distilbert_ok,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
