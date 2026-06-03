"""
conftest.py — Pytest configuration and fixtures.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default="google/flan-t5-small",
        help="HuggingFace model for integration tests (e.g. google/flan-t5-base, distilgpt2)"
    )


@pytest.fixture(scope="session")
def model_name(request):
    return request.config.getoption("--model")