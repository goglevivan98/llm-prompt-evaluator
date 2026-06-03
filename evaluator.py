"""
evaluator.py — Core evaluation engine.

Responsibilities:
  1. Load config.json once at startup
  2. Send each prompt to the Hugging Face Inference API
  3. Run quality checks via metrics.py
  4. Collect structured results
  5. Delegate reporting to reporter.py

Usage:
    python evaluator.py                 # runs all test cases, writes reports
    python evaluator.py --config my.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

import metrics as m
import reporter

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG: Optional[dict] = None  # module-level cache


def load_config(path: str = "config.json") -> dict:
    """Read and validate config.json. Called once; result is cached."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    required_top = {"model", "api_url", "test_cases"}
    missing = required_top - data.keys()
    if missing:
        raise ValueError(f"config.json is missing keys: {missing}")

    _CONFIG = data
    return _CONFIG


# ---------------------------------------------------------------------------
# HuggingFace API
# ---------------------------------------------------------------------------

def _hf_token() -> Optional[str]:
    """Read HF_TOKEN from environment (optional for public models)."""
    return os.environ.get("HF_TOKEN")


def query_model(
        prompt: str,
        api_url: str,
        model: str,
        timeout: float = 10.0,
) -> str:
    """
    POST a prompt to the HF Inference API.

    Returns the response text.
    Raises requests.Timeout if the server doesn't respond in time.
    Raises requests.HTTPError on 4xx/5xx responses.
    """
    url = api_url.rstrip("/") + "/" + model
    headers = {"Content-Type": "application/json"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()

    # HF returns either a list of dicts or a plain dict depending on the model.
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return (
                    first.get("generated_text")
                    or first.get("translation_text")
                    or first.get("summary_text")
                    or str(first)
            )
        return str(first)
    if isinstance(data, dict):
        return (
                data.get("generated_text")
                or data.get("translation_text")
                or data.get("summary_text")
                or str(data)
        )
    return str(data)


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------

def evaluate_one(test_case: dict, config: dict) -> dict:
    """
    Run a single test case end-to-end.

    Returns a result dict with keys:
        id, prompt, status, response, word_count,
        keywords_found, expected_keywords, sentiment_ok,
        fail_reason, error_message, response_time_sec,
        checks (sub-dict of individual boolean results)
    """
    tc_id = test_case["id"]
    prompt = test_case["prompt"]
    expected = test_case["expected"]
    timeout = config.get("max_response_time_sec", 10)

    result: dict = {
        "id": tc_id,
        "prompt": prompt,
        "status": "ERROR",
        "response": None,
        "word_count": 0,
        "keywords_found": [],
        "expected_keywords": expected.get("keywords", []),
        "sentiment_ok": False,
        "fail_reason": "",
        "error_message": "",
        "response_time_sec": 0.0,
        "checks": {},
    }

    # --- Step 1: Call API -------------------------------------------------
    t0 = time.monotonic()
    try:
        response_text = query_model(
            prompt=prompt,
            api_url=config["api_url"],
            model=config["model"],
            timeout=float(timeout),
        )
    except requests.Timeout:
        result["error_message"] = f"API timeout after {timeout}s"
        result["response_time_sec"] = round(time.monotonic() - t0, 2)
        return result
    except requests.HTTPError as exc:
        result["error_message"] = f"HTTP {exc.response.status_code}: {exc}"
        result["response_time_sec"] = round(time.monotonic() - t0, 2)
        return result
    except Exception as exc:  # noqa: BLE001
        result["error_message"] = f"Unexpected error: {exc}"
        result["response_time_sec"] = round(time.monotonic() - t0, 2)
        return result

    result["response_time_sec"] = round(time.monotonic() - t0, 2)
    result["response"] = response_text

    # --- Step 2-7: Quality checks ----------------------------------------
    try:
        checks = m.run_all_checks(response_text, expected)
    except Exception as exc:  # noqa: BLE001
        result["error_message"] = f"Metrics error: {exc}"
        return result

    result["checks"] = checks
    result["word_count"] = checks["word_count"]
    result["keywords_found"] = checks["keywords_found"]
    result["sentiment_ok"] = checks["sentiment"]

    # --- Step 8: Determine PASS / FAIL -----------------------------------
    fail_reasons = []
    if not checks["not_empty"]:
        fail_reasons.append("response is empty")
    if not checks["length"]:
        wc = checks["word_count"]
        fail_reasons.append(
            f"{'too short' if wc < expected.get('min_words', 0) else 'too long'}"
            f" ({wc} words)"
        )
    if not checks["keywords"]:
        missing_kw = [
            kw for kw in expected.get("keywords", [])
            if kw not in checks["keywords_found"]
        ]
        fail_reasons.append(f"missing keywords: {missing_kw}")
    if not checks["sentiment"]:
        fail_reasons.append(
            f"sentiment mismatch (expected: {expected.get('sentiment', '?')})"
        )

    if fail_reasons:
        result["status"] = "FAIL"
        result["fail_reason"] = "; ".join(fail_reasons)
    else:
        result["status"] = "PASS"

    return result


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_evaluation(config_path: str = "config.json") -> list[dict]:
    """Run all test cases from the config and return the results list."""
    config = load_config(config_path)
    results = []

    for tc in config["test_cases"]:
        print(f"  Running {tc['id']}…", end=" ", flush=True)
        result = evaluate_one(tc, config)
        print(result["status"])
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Prompt Evaluator")
    parser.add_argument(
        "--config", default="config.json", help="Path to config JSON (default: config.json)"
    )
    parser.add_argument(
        "--json-out", default="report.json", help="Output path for JSON report"
    )
    parser.add_argument(
        "--csv-out", default="report.csv", help="Output path for CSV report"
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("LLM Prompt Evaluator")
    print(f"{'=' * 60}\n")

    results = run_evaluation(args.config)
    reporter.generate_report(results, json_path=args.json_out, csv_path=args.csv_out)

    # Exit code: 0 = all passed, 1 = some failed/errored
    all_passed = all(r["status"] == "PASS" for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()