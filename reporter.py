"""
reporter.py — Format and persist evaluation results.

Outputs:
  1. Coloured console summary (PASS / FAIL / ERROR per test case)
  2. report.json  — machine-readable full detail
  3. report.csv   — one row per test case, importable into Excel / Sheets
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Sequence

# ANSI colours — disabled automatically when not writing to a real terminal
_USE_COLOUR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR", "0") != "0"

_GREEN = "\033[92m" if _USE_COLOUR else ""
_RED = "\033[91m" if _USE_COLOUR else ""
_YELLOW = "\033[93m" if _USE_COLOUR else ""
_RESET = "\033[0m" if _USE_COLOUR else ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _status_prefix(status: str) -> str:
    if status == "PASS":
        return f"{_GREEN}[PASS]{_RESET}"
    if status == "FAIL":
        return f"{_RED}[FAIL]{_RESET}"
    return f"{_YELLOW}[ERROR]{_RESET}"


def _format_line(result: dict) -> str:
    """Produce one human-readable console line for a result dict."""
    prefix = _status_prefix(result["status"])
    tc_id = result["id"]
    prompt = result["prompt"][:60] + ("…" if len(result["prompt"]) > 60 else "")

    if result["status"] == "ERROR":
        detail = result.get("error_message", "API error")
        return f'{prefix} {tc_id} — "{prompt}" — {detail}'

    wc = result.get("word_count", 0)
    kw_ok = len(result.get("keywords_found", []))
    kw_req = len(result.get("expected_keywords", []))
    reason = result.get("fail_reason", "")

    base = f'{prefix} {tc_id} — "{prompt}" — {wc} words, keywords: {kw_ok}/{kw_req}'
    if reason:
        base += f" | {reason}"
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_report(results: Sequence[dict]) -> None:
    """Print a coloured summary to stdout."""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    total = len(results)

    print()
    for r in results:
        print(_format_line(r))

    print()
    print(
        f"Results: {_GREEN}{passed}/{total} passed{_RESET}, "
        f"{_RED}{failed} failed{_RESET}, "
        f"{_YELLOW}{errors} error{'s' if errors != 1 else ''}{_RESET}"
    )


def save_json(results: Sequence[dict], path: str = "report.json") -> None:
    """Persist results to a JSON file."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "failed": sum(1 for r in results if r["status"] == "FAIL"),
            "errors": sum(1 for r in results if r["status"] == "ERROR"),
        },
        "results": list(results),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


def save_csv(results: Sequence[dict], path: str = "report.csv") -> None:
    """Persist results to a CSV file (one row per test case)."""
    fieldnames = [
        "id", "status", "prompt", "word_count",
        "keywords_found", "expected_keywords",
        "sentiment_ok", "fail_reason", "error_message",
        "response_time_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            # Flatten list fields for readability
            row["keywords_found"] = "; ".join(r.get("keywords_found", []))
            row["expected_keywords"] = "; ".join(r.get("expected_keywords", []))
            writer.writerow(row)
    print(f"Report saved to {path}")


def generate_report(
        results: Sequence[dict],
        json_path: str = "report.json",
        csv_path: str = "report.csv",
) -> None:
    """Convenience wrapper: print console + write both file formats."""
    print_report(results)
    save_json(results, json_path)
    save_csv(results, csv_path)