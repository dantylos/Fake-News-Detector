"""Shared fixtures and sample data for API tests."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient. Models load once via lru_cache
    and stay cached across all tests in the module.
    """
    with TestClient(app) as c:
        yield c


# Sample texts

FAKE_ARTICLE = (
    "BREAKING: Scientists have just discovered that the moon is entirely "
    "made of green cheese and the Earth is actually a giant flat disc "
    "supported by four elephants standing on the back of a giant turtle. "
    "This shocking revelation was published by top conspiracy theorists "
    "today in a groundbreaking study that challenges everything we know "
    "about science and reality. The government has been hiding this truth "
    "for decades, according to anonymous sources."
)

REAL_ARTICLE = (
    "The Federal Reserve announced today that it would maintain its "
    "benchmark interest rate at the current level, citing ongoing "
    "uncertainty in global economic conditions. Fed Chair Jerome Powell "
    "noted that inflation had shown signs of moderating in recent months, "
    "but emphasized the committee would remain data-dependent in its "
    "approach to future policy decisions. Markets reacted positively, "
    "with the S&P 500 rising 0.3 percent in afternoon trading. Economists "
    "widely anticipated the decision, with a Reuters poll showing 90 percent "
    "of respondents predicting a hold."
)

SHORT_TEXT = "Too short."

EMPTY_TEXT = ""

MINIMUM_VALID_TEXT = (
    "This is a test article that is exactly long enough to pass the "
    "fifty character minimum validation requirement for the API."
)
