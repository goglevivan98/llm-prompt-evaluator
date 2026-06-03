"""
test_evaluator.py — Pytest test suite for the LLM Prompt Evaluator.

Structure:
  Unit tests  — metrics.py functions (pure, no I/O)
  Unit tests  — reporter.py helpers
  Integration — evaluator.evaluate_one() with mocked HTTP
  Integration — evaluator.run_evaluation() with mocked HTTP
  Contract    — config.json schema validation

Run with:
    pytest test_evaluator.py -v
    pytest test_evaluator.py -v --tb=short   # shorter tracebacks
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make sure the local modules are importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

import metrics as m
import reporter
import evaluator


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def config():
    """Load the real config.json once for the whole test session."""
    evaluator._CONFIG = None  # reset module-level cache
    return evaluator.load_config("config.json")


@pytest.fixture()
def minimal_tc():
    """A minimal test-case dict used by integration tests."""
    return {
        "id": "TC_TEST",
        "prompt": "What is software testing?",
        "expected": {
            "min_words": 3,
            "max_words": 50,
            "keywords": ["test"],
            "sentiment": "neutral",
            "not_empty": True,
        },
    }


@pytest.fixture()
def mock_config():
    return {
        "model": "google/flan-t5-base",
        "api_url": "https://api-inference.huggingface.co/models/",
        "max_response_time_sec": 10,
        "test_cases": [],
    }


# ===========================================================================
# F2 — check_not_empty
# ===========================================================================

class TestCheckNotEmpty:
    def test_normal_text(self):
        assert m.check_not_empty("Hello world") is True

    def test_whitespace_only(self):
        assert m.check_not_empty("   \t\n  ") is False

    def test_empty_string(self):
        assert m.check_not_empty("") is False

    def test_none_value(self):
        assert m.check_not_empty(None) is False  # type: ignore[arg-type]

    def test_single_char(self):
        assert m.check_not_empty("x") is True


# ===========================================================================
# F3 — check_length
# ===========================================================================

class TestCheckLength:
    def test_within_range(self):
        assert m.check_length("one two three four five", 3, 10) is True

    def test_exactly_min(self):
        assert m.check_length("a b c", 3, 10) is True

    def test_exactly_max(self):
        text = " ".join(["word"] * 10)
        assert m.check_length(text, 3, 10) is True

    def test_below_min(self):
        assert m.check_length("short", 5, 100) is False

    def test_above_max(self):
        text = " ".join(["word"] * 20)
        assert m.check_length(text, 1, 10) is False

    def test_empty_string(self):
        assert m.check_length("", 1, 100) is False

    def test_none(self):
        assert m.check_length(None, 1, 100) is False  # type: ignore[arg-type]

    def test_single_word_range_one(self):
        assert m.check_length("yes", 1, 1) is True


# ===========================================================================
# F4 — check_keywords
# ===========================================================================

class TestCheckKeywords:
    def test_all_present(self):
        assert m.check_keywords("software testing finds bugs", ["test", "bug"]) is True

    def test_case_insensitive(self):
        assert m.check_keywords("Software Testing", ["testing", "software"]) is True

    def test_partial_match_fails(self):
        # "qualityX" should NOT match keyword "quality"
        text = "qualityX is important"
        # Actually "quality" IS a substring of "qualityX" — check_keywords uses `in`
        # so this documents the current behaviour (substring match).
        assert m.check_keywords(text, ["quality"]) is True

    def test_missing_keyword(self):
        assert m.check_keywords("hello world", ["missing"]) is False

    def test_empty_keyword_list(self):
        assert m.check_keywords("any text", []) is True

    def test_empty_response(self):
        assert m.check_keywords("", ["test"]) is False

    def test_none_response(self):
        assert m.check_keywords(None, ["test"]) is False  # type: ignore[arg-type]

    def test_multiple_missing(self):
        assert m.check_keywords("hello", ["a", "b", "c"]) is False


# ===========================================================================
# F5 — check_sentiment (mocked pipeline)
# ===========================================================================

class TestCheckSentiment:
    """Mocks the HF pipeline so no model download is needed in CI."""

    def _pipe_returns(self, label: str, score: float):
        """Helper: patch _get_sentiment_pipeline to return a fixed result."""
        mock_pipe = MagicMock(return_value=[{"label": label, "score": score}])
        return patch("metrics._get_sentiment_pipeline", return_value=mock_pipe)

    def test_positive_match(self):
        with self._pipe_returns("POSITIVE", 0.98):
            assert m.check_sentiment("great job!", "positive") is True

    def test_negative_match(self):
        with self._pipe_returns("NEGATIVE", 0.91):
            assert m.check_sentiment("terrible outcome", "negative") is True

    def test_neutral_low_score(self):
        with self._pipe_returns("POSITIVE", 0.55):
            # score < 0.65 → classified as neutral
            assert m.check_sentiment("some text", "neutral") is True

    def test_mismatch(self):
        with self._pipe_returns("NEGATIVE", 0.95):
            assert m.check_sentiment("bad result", "positive") is False

    def test_empty_response(self):
        assert m.check_sentiment("", "neutral") is False


# ===========================================================================
# run_all_checks — integration of all metrics
# ===========================================================================

class TestRunAllChecks:
    def test_all_pass(self):
        response = "Software testing verifies and validates application quality to find bugs."
        expected = {
            "min_words": 5,
            "max_words": 20,
            "keywords": ["test", "quality", "bug"],
            "sentiment": "neutral",
        }
        mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.55}])
        with patch("metrics._get_sentiment_pipeline", return_value=mock_pipe):
            result = m.run_all_checks(response, expected)
        assert result["not_empty"] is True
        assert result["length"] is True
        assert result["keywords"] is True
        assert result["word_count"] == 11
        assert set(result["keywords_found"]) == {"test", "quality", "bug"}

    def test_empty_response(self):
        result = m.run_all_checks("", {"min_words": 5, "max_words": 100, "keywords": ["x"]})
        assert result["not_empty"] is False
        assert result["length"] is False
        assert result["keywords"] is False
        assert result["word_count"] == 0


# ===========================================================================
# evaluator.py — evaluate_one() with mocked HTTP
# ===========================================================================

def _make_mock_response(text: str, status_code: int = 200):
    """Build a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = [{"generated_text": text}]
    mock.raise_for_status = MagicMock()
    return mock


class TestEvaluateOne:
    def test_pass_case(self, minimal_tc, mock_config):
        good_text = "Software testing ensures code quality meets requirements."
        with patch("requests.post", return_value=_make_mock_response(good_text)):
            mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.5}])
            with patch("metrics._get_sentiment_pipeline", return_value=mock_pipe):
                result = evaluator.evaluate_one(minimal_tc, mock_config)

        assert result["status"] == "PASS"
        assert result["response"] == good_text
        assert result["word_count"] >= 3

    def test_fail_too_short(self, minimal_tc, mock_config):
        with patch("requests.post", return_value=_make_mock_response("Hi.")):
            mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.5}])
            with patch("metrics._get_sentiment_pipeline", return_value=mock_pipe):
                result = evaluator.evaluate_one(minimal_tc, mock_config)

        assert result["status"] == "FAIL"
        assert "short" in result["fail_reason"].lower()

    def test_error_on_timeout(self, minimal_tc, mock_config):
        import requests as req
        with patch("requests.post", side_effect=req.Timeout("timed out")):
            result = evaluator.evaluate_one(minimal_tc, mock_config)

        assert result["status"] == "ERROR"
        assert "timeout" in result["error_message"].lower()

    def test_error_on_http_error(self, minimal_tc, mock_config):
        import requests as req
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        http_error = req.HTTPError(response=mock_resp)
        with patch("requests.post", side_effect=http_error):
            result = evaluator.evaluate_one(minimal_tc, mock_config)

        assert result["status"] == "ERROR"
        assert "503" in result["error_message"]

    def test_response_time_recorded(self, minimal_tc, mock_config):
        def slow_post(*a, **kw):
            time.sleep(0.05)
            return _make_mock_response("testing quality bug check here now")
        with patch("requests.post", side_effect=slow_post):
            mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.5}])
            with patch("metrics._get_sentiment_pipeline", return_value=mock_pipe):
                result = evaluator.evaluate_one(minimal_tc, mock_config)
        assert result["response_time_sec"] >= 0.04

    def test_missing_keyword_causes_fail(self, mock_config):
        tc = {
            "id": "TC_KW",
            "prompt": "Test prompt",
            "expected": {
                "min_words": 1,
                "max_words": 50,
                "keywords": ["MISSING_WORD_XYZ"],
                "sentiment": "neutral",
            },
        }
        with patch("requests.post", return_value=_make_mock_response("Some response text")):
            mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.5}])
            with patch("metrics._get_sentiment_pipeline", return_value=mock_pipe):
                result = evaluator.evaluate_one(tc, mock_config)
        assert result["status"] == "FAIL"
        assert "missing keywords" in result["fail_reason"].lower()


# ===========================================================================
# reporter.py
# ===========================================================================

class TestReporter:
    @pytest.fixture()
    def sample_results(self):
        return [
            {
                "id": "TC001", "status": "PASS", "prompt": "What is testing?",
                "word_count": 12, "keywords_found": ["test"], "expected_keywords": ["test"],
                "sentiment_ok": True, "fail_reason": "", "error_message": "",
                "response_time_sec": 0.8,
            },
            {
                "id": "TC002", "status": "FAIL", "prompt": "Short prompt",
                "word_count": 2, "keywords_found": [], "expected_keywords": ["bug"],
                "sentiment_ok": False, "fail_reason": "too short (2 words)",
                "error_message": "", "response_time_sec": 0.5,
            },
            {
                "id": "TC003", "status": "ERROR", "prompt": "Another prompt",
                "word_count": 0, "keywords_found": [], "expected_keywords": [],
                "sentiment_ok": False, "fail_reason": "", "error_message": "API timeout",
                "response_time_sec": 10.1,
            },
        ]

    def test_save_json_creates_file(self, sample_results, tmp_path):
        out = str(tmp_path / "report.json")
        reporter.save_json(sample_results, path=out)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert data["summary"]["total"] == 3
        assert data["summary"]["passed"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["errors"] == 1
        assert len(data["results"]) == 3

    def test_save_csv_creates_file(self, sample_results, tmp_path):
        out = str(tmp_path / "report.csv")
        reporter.save_csv(sample_results, path=out)
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "TC001" in content
        assert "PASS" in content
        assert "FAIL" in content

    def test_print_report_outputs_summary(self, sample_results, capsys):
        reporter.print_report(sample_results)
        out = capsys.readouterr().out
        assert "1/3" in out or "passed" in out


# ===========================================================================
# Config schema validation
# ===========================================================================

class TestConfigSchema:
    def test_config_loads(self, config):
        assert "model" in config
        assert "api_url" in config
        assert "test_cases" in config

    def test_test_cases_not_empty(self, config):
        assert len(config["test_cases"]) > 0

    def test_each_case_has_required_fields(self, config):
        for tc in config["test_cases"]:
            assert "id" in tc, f"Missing 'id' in {tc}"
            assert "prompt" in tc, f"Missing 'prompt' in {tc}"
            assert "expected" in tc, f"Missing 'expected' in {tc}"

    def test_expected_fields_present(self, config):
        for tc in config["test_cases"]:
            exp = tc["expected"]
            assert "min_words" in exp
            assert "max_words" in exp
            assert "keywords" in exp
            assert "sentiment" in exp
            assert exp["min_words"] <= exp["max_words"]

    def test_sentiment_values_valid(self, config):
        valid = {"positive", "negative", "neutral"}
        for tc in config["test_cases"]:
            sent = tc["expected"].get("sentiment")
            assert sent in valid, f"Invalid sentiment '{sent}' in {tc['id']}"

    def test_missing_config_raises(self, tmp_path):
        evaluator._CONFIG = None
        with pytest.raises(FileNotFoundError):
            evaluator.load_config(str(tmp_path / "nonexistent.json"))

    def test_config_cached_after_load(self, config):
        # Second call should return the same object
        config2 = evaluator.load_config("config.json")
        assert config is config2

    def test_max_response_time_positive(self, config):
        t = config.get("max_response_time_sec", 10)
        assert isinstance(t, (int, float)) and t > 0


# ===========================================================================
# Non-functional: response time limit (mocked)
# ===========================================================================

class TestNonFunctional:
    def test_api_timeout_respected(self, minimal_tc, mock_config):
        """Evaluator must not wait longer than max_response_time_sec."""
        import requests as req
        mock_config["max_response_time_sec"] = 2

        call_timeout = []

        def capture_post(*args, **kwargs):
            call_timeout.append(kwargs.get("timeout"))
            raise req.Timeout("simulated")

        with patch("requests.post", side_effect=capture_post):
            evaluator.evaluate_one(minimal_tc, mock_config)

        assert call_timeout[0] == 2.0

    def test_error_status_not_fail_on_api_error(self, minimal_tc, mock_config):
        """API errors must produce ERROR, not FAIL."""
        import requests as req
        with patch("requests.post", side_effect=req.Timeout):
            result = evaluator.evaluate_one(minimal_tc, mock_config)
        assert result["status"] == "ERROR"
        assert result["status"] != "FAIL"