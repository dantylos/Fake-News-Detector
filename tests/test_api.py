"""
Integration tests for the Fake News Detection API.
Covers /analyze, /history, and /health endpoints.

Run:
    pytest tests/ -v
    pytest tests/test_api.py -v -k "health"
"""

import pytest
from tests.conftest import (
    FAKE_ARTICLE,
    REAL_ARTICLE,
    SHORT_TEXT,
    EMPTY_TEXT,
    MINIMUM_VALID_TEXT,
)


# /health endpoint

class TestHealth:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_models_loaded_keys(self, client):
        data = client.get("/health").json()
        models = data["models_loaded"]
        assert "tfidf_baseline" in models
        assert "svm" in models
        assert "distilbert" in models

    def test_health_all_models_available(self, client):
        """All three models should report as loaded."""
        data = client.get("/health").json()
        models = data["models_loaded"]
        assert models["tfidf_baseline"] is True
        assert models["svm"] is True
        assert models["distilbert"] is True

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert set(data.keys()) == {"status", "models_loaded"}


# /analyze endpoint -- valid requests

class TestAnalyzeValid:

    def test_analyze_returns_200(self, client):
        response = client.post("/analyze", json={"text": FAKE_ARTICLE})
        assert response.status_code == 200

    def test_analyze_response_schema(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        required_keys = {
            "prediction", "confidence", "explanation",
            "machine_evaluation", "model_version", "timestamp", "disclaimer",
        }
        assert required_keys.issubset(set(data.keys()))

    def test_analyze_prediction_is_valid_label(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert data["prediction"] in ("fake", "real")

    def test_analyze_confidence_range(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_analyze_fake_article_detected(self, client):
        """Obviously fake text should be classified as fake."""
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert data["prediction"] == "fake"

    def test_analyze_fake_article_high_confidence(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert data["confidence"] > 0.7

    def test_analyze_real_article_detected(self, client):
        data = client.post("/analyze", json={"text": REAL_ARTICLE}).json()
        assert data["prediction"] == "real"

    def test_analyze_explanation_structure(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        explanation = data["explanation"]
        assert "top_sentences" in explanation
        assert isinstance(explanation["top_sentences"], list)

    def test_analyze_explanation_sentence_schema(self, client):
        """Each SHAP entry should have sentence + shap_value fields."""
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        sentences = data["explanation"]["top_sentences"]
        assert sentences
        for item in sentences:
            assert "sentence" in item
            assert "shap_value" in item
            assert isinstance(item["sentence"], str)
            assert isinstance(item["shap_value"], (int, float))

    def test_analyze_shap_values_not_all_zero(self, client):
        top = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()["explanation"]["top_sentences"]
        assert top, "explanation is empty"
        assert any(abs(s["shap_value"]) > 1e-4 for s in top)

    def test_analyze_model_version_present(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert len(data["model_version"]) > 0

    def test_analyze_ensemble_used(self, client):
        """With all models loaded, ensemble should be used."""
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert "ensemble" in data["model_version"].lower()

    def test_analyze_timestamp_present(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert len(data["timestamp"]) > 0
        assert "T" in data["timestamp"]

    def test_analyze_disclaimer_present(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 0

    def test_analyze_machine_evaluation_present(self, client):
        data = client.post("/analyze", json={"text": FAKE_ARTICLE}).json()
        assert isinstance(data["machine_evaluation"], str)
        assert len(data["machine_evaluation"]) > 20

    def test_analyze_with_url_field(self, client):
        response = client.post("/analyze", json={
            "text": FAKE_ARTICLE,
            "url": "https://example.com/article",
        })
        assert response.status_code == 200

    def test_analyze_minimum_valid_length(self, client):
        response = client.post("/analyze", json={"text": MINIMUM_VALID_TEXT})
        assert response.status_code == 200


# /analyze endpoint -- input validation

class TestAnalyzeValidation:

    def test_analyze_rejects_short_text(self, client):
        response = client.post("/analyze", json={"text": SHORT_TEXT})
        assert response.status_code == 400

    def test_analyze_short_text_error_message(self, client):
        data = client.post("/analyze", json={"text": SHORT_TEXT}).json()
        assert "50 characters" in data["detail"].lower() or "50" in data["detail"]

    def test_analyze_rejects_empty_text(self, client):
        response = client.post("/analyze", json={"text": EMPTY_TEXT})
        assert response.status_code == 400

    def test_analyze_rejects_whitespace_only(self, client):
        response = client.post("/analyze", json={"text": "     "})
        assert response.status_code == 400

    def test_analyze_rejects_missing_text_field(self, client):
        response = client.post("/analyze", json={"url": "https://example.com"})
        assert response.status_code == 422

    def test_analyze_rejects_empty_body(self, client):
        response = client.post("/analyze", json={})
        assert response.status_code == 422


# /history endpoint

class TestHistory:

    def test_history_returns_200(self, client):
        response = client.get("/history")
        assert response.status_code == 200

    def test_history_returns_list(self, client):
        data = client.get("/history").json()
        assert isinstance(data, list)

    def test_history_limit_parameter(self, client):
        data = client.get("/history?limit=1").json()
        assert len(data) <= 1

    def test_history_record_schema(self, client):
        """After analyzing, history should contain expected fields."""
        client.post("/analyze", json={"text": FAKE_ARTICLE})
        data = client.get("/history?limit=1").json()
        assert data
        record = data[0]
        expected_keys = {
            "text_preview", "prediction", "confidence",
            "model_used", "timestamp",
        }
        assert expected_keys.issubset(set(record.keys()))

    def test_history_prediction_values(self, client):
        data = client.get("/history").json()
        for record in data:
            assert record["prediction"] in ("fake", "real")

    def test_history_ordered_most_recent_first(self, client):
        data = client.get("/history").json()
        if len(data) >= 2:
            timestamps = [r["timestamp"] for r in data]
            assert timestamps == sorted(timestamps, reverse=True)

    def test_history_default_limit(self, client):
        data = client.get("/history").json()
        assert len(data) <= 20


# HTML cleaning

class TestTextCleaning:

    def test_analyze_strips_html_tags(self, client):
        html_text = (
            "<p>Scientists have just discovered that the moon is entirely "
            "made of green cheese.</p><script>alert('xss')</script> "
            "<b>This shocking revelation</b> was published by top conspiracy "
            "theorists today in a groundbreaking study that challenges "
            "everything we know about science and reality."
        )
        response = client.post("/analyze", json={"text": html_text})
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ("fake", "real")

    def test_analyze_handles_extra_whitespace(self, client):
        messy_text = (
            "Scientists   have   just   discovered   that   the   moon   "
            "is   entirely   made   of   green   cheese   and   this   is   "
            "a   shocking   revelation   published   by   conspiracy   "
            "theorists   today   in   a   groundbreaking   study."
        )
        response = client.post("/analyze", json={"text": messy_text})
        assert response.status_code == 200


# Wrong HTTP methods

class TestUnsupportedMethods:

    def test_analyze_rejects_get(self, client):
        response = client.get("/analyze")
        assert response.status_code == 405

    def test_history_rejects_post(self, client):
        response = client.post("/history")
        assert response.status_code == 405

    def test_health_rejects_post(self, client):
        response = client.post("/health")
        assert response.status_code == 405

    def test_nonexistent_route_returns_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404
