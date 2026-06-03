"""
evaluator.py — Core evaluation engine (hybrid: API + local).

Supports two modes:
  - API mode: uses HuggingFace Inference API (requires internet + HF_TOKEN)
  - Local mode: uses downloaded model (no internet after first download)

Usage:
    python evaluator.py                        # auto-detect (API if token set, else local)
    python evaluator.py --mode api             # force API mode
    python evaluator.py --mode local           # force local mode
    python evaluator.py --mode local --model google/flan-t5-base
    python evaluator.py --config my.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

import metrics as m
import reporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CONFIG: Optional[dict] = None
_local_pipeline = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.json") -> dict:
    """Read and cache config.json. Called once; result is cached."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    _CONFIG = data
    return _CONFIG


# ---------------------------------------------------------------------------
# API mode
# ---------------------------------------------------------------------------

def _hf_token() -> Optional[str]:
    """Read HF_TOKEN from environment (optional for public models)."""
    return os.environ.get("HF_TOKEN")


def _query_api(prompt: str, api_url: str, model: str, timeout: float = 10.0) -> str:
    """POST a prompt to the HuggingFace Inference API and return the response text."""
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
# Local mode
# ---------------------------------------------------------------------------

def _get_local_pipeline(model_name: str = None):
    """Load a local HuggingFace model once and cache it. Uses T5/FLAN for text2text generation."""
    global _local_pipeline
    if _local_pipeline is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        if model_name is None:
            model_name = "google/flan-t5-small"
        print(f"Loading local model: {model_name} ...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        def generate(prompt, max_new_tokens=150):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Greedy decoding — fewer repetitions
                num_beams=3,              # Beam search — higher quality
                no_repeat_ngram_size=3,   # Prevent repeating trigrams
                early_stopping=True,
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        _local_pipeline = generate
        print("Model loaded.\n")
    return _local_pipeline


def _query_local(prompt: str, max_new_tokens: int = 150, model_name: str = None) -> str:
    pipe = _get_local_pipeline(model_name)
    result = pipe(prompt, max_new_tokens=max_new_tokens)
    # Handle both transformers v4 (returns list[dict]) and v5 (returns str)
    if isinstance(result, list) and len(result) > 0:
        first = result[0]
        if isinstance(first, dict):
            return first.get("generated_text", str(first))
        return str(first)
    if isinstance(result, dict):
        return result.get("generated_text", str(result))
    return str(result)


# ---------------------------------------------------------------------------
# Unified query
# ---------------------------------------------------------------------------

def query_model(prompt: str, config: dict, mode: str = "auto") -> str:
    """Send a prompt to the model (API, local, or auto-detect)."""
    if mode == "local":
        model_name = config.get("local_model", "google/flan-t5-small")
        return _query_local(prompt, model_name=model_name)
    elif mode == "api":
        return _query_api(prompt, config["api_url"], config["model"], config.get("max_response_time_sec", 10))
    else:  # auto
        if _hf_token():
            try:
                return _query_api(prompt, config["api_url"], config["model"], config.get("max_response_time_sec", 10))
            except Exception:
                print("  (API failed, falling back to local model)")
                model_name = config.get("local_model", "google/flan-t5-small")
                return _query_local(prompt, model_name=model_name)
        else:
            model_name = config.get("local_model", "google/flan-t5-small")
            return _query_local(prompt, model_name=model_name)


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------

def evaluate_one(test_case: dict, config: dict, mode: str = "auto") -> dict:
    """Run a single test case end-to-end and return a structured result dict."""
    tc_id = test_case["id"]
    prompt = test_case["prompt"]
    expected = test_case["expected"]

    result = {
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

    # Log the prompt
    logger.info("[%s] Prompt: %s", tc_id, prompt[:100])

    t0 = time.monotonic()
    try:
        response_text = query_model(prompt, config, mode)
    except Exception as exc:
        logger.error("[%s] Model error: %s", tc_id, exc)
        result["error_message"] = f"Model error: {exc}"
        result["response_time_sec"] = round(time.monotonic() - t0, 2)
        return result

    result["response_time_sec"] = round(time.monotonic() - t0, 2)
    result["response"] = response_text

    # Log the response
    logger.info("[%s] Response: %s", tc_id, response_text[:150])

    try:
        checks = m.run_all_checks(response_text, expected)
    except Exception as exc:
        logger.error("[%s] Metrics error: %s", tc_id, exc)
        result["error_message"] = f"Metrics error: {exc}"
        return result

    result["checks"] = checks
    result["word_count"] = checks["word_count"]
    result["keywords_found"] = checks["keywords_found"]
    result["sentiment_ok"] = checks["sentiment"]

    fail_reasons = []
    if not checks["not_empty"]:
        fail_reasons.append("response is empty")
    if not checks["length"]:
        wc = checks["word_count"]
        fail_reasons.append(
            f"{'too short' if wc < expected.get('min_words', 0) else 'too long'} ({wc} words)"
        )
    if not checks["keywords"]:
        missing_kw = [
            kw for kw in expected.get("keywords", [])
            if kw not in checks["keywords_found"]
        ]
        fail_reasons.append(f"missing keywords: {missing_kw}")
    if not checks["sentiment"]:
        fail_reasons.append(f"sentiment mismatch (expected: {expected.get('sentiment', '?')})")

    if fail_reasons:
        result["status"] = "FAIL"
        result["fail_reason"] = "; ".join(fail_reasons)
    else:
        result["status"] = "PASS"

    # Log the result
    logger.info("[%s] Result: %s", tc_id, result["status"])

    return result


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_evaluation(config_path: str = "config.json", mode: str = "auto") -> list[dict]:
    """Run all test cases from the config and return the results list."""
    config = load_config(config_path)
    results = []

    for tc in config["test_cases"]:
        print(f"  Running {tc['id']}…", end=" ", flush=True)
        result = evaluate_one(tc, config, mode)
        print(result["status"])
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Prompt Evaluator (hybrid: API + local)")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--json-out", default="report.json", help="JSON report path")
    parser.add_argument("--csv-out", default="report.csv", help="CSV report path")
    parser.add_argument("--mode", choices=["api", "local", "auto"], default="auto",
                        help="Run mode: api (internet), local (offline), auto (detect)")
    parser.add_argument("--model", default=None,
                        help="Override local model (e.g., 'google/flan-t5-base', 'distilgpt2')")
    args = parser.parse_args()

    # Override model from CLI argument
    config = load_config(args.config)
    if args.model:
        config["local_model"] = args.model

    print(f"\n{'=' * 60}")
    print(f"LLM Prompt Evaluator (mode: {args.mode})")
    print(f"Local model: {config.get('local_model', 'google/flan-t5-small')}")
    print(f"{'=' * 60}\n")

    results = run_evaluation(args.config, args.mode)
    reporter.generate_report(results, json_path=args.json_out, csv_path=args.csv_out)

    all_passed = all(r["status"] == "PASS" for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()