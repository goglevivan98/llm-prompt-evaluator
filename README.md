# LLM Prompt Evaluator
![Tests](https://github.com/goglevivan98/llm-prompt-evaluator/actions/workflows/tests.yml/badge.svg)

A QA tool for evaluating LLM responses. Supports both **HuggingFace Inference API** (online) and **local model** (offline) modes.

## What it does

Sends prompts to a language model and validates responses against expected criteria:

- **Not empty** — response must contain text
- **Length** — word count within min/max range
- **Keywords** — must include specified keywords (case-insensitive)
- **Sentiment** — expected tone (POSITIVE / NEGATIVE / NEUTRAL)

## Why I built this

As a QA Automation Engineer, I wanted to explore how testing principles apply to LLM outputs. This tool demonstrates systematic validation of non-deterministic systems — similar to testing complex distributed applications.

## Project structure
```
llm-prompt-evaluator/
├── config.json          # Test cases and expected criteria
├── evaluator.py         # Core engine: hybrid mode (API + local)
├── metrics.py           # Validation checks: length, keywords, sentiment
├── reporter.py          # Report generation: console, JSON, CSV
├── test_evaluator.py    # Pytest test suite (unit + integration tests)
├── conftest.py          # Pytest fixtures and --model CLI option
├── pytest.ini           # Pytest configuration
├── requirements.txt     # Python dependencies
├── report.json          # Generated test report (JSON)
├── report.csv           # Generated test report (CSV)
├── report.html          # Generated test report (HTML)
└── README.md            # This file
```


## Setup

### 1. Clone the repository

```bash
git clone https://github.com/goglevivan98/llm-prompt-evaluator.git
cd llm-prompt-evaluator
```

### 2. Create and activate virtual environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
#### Linux / Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run evaluator

```bash
# Auto mode: tries API first, falls back to local
python evaluator.py

# Local mode with default model (google/flan-t5-small)
python evaluator.py --mode local

# Local mode with a specific model
python evaluator.py --mode local --model google/flan-t5-base

# API mode (requires HF_TOKEN)
python evaluator.py --mode api
```

First run in local mode will download the model (~330 MB, one time only).

API mode requires a HuggingFace token:
```text
1. Get your token at https://huggingface.co/settings/tokens
2. Set environment variable:
   - Windows: $env:HF_TOKEN="your_token_here"
   - Linux/Mac: export HF_TOKEN="your_token"
```

### Run tests

```bash
# Unit tests only (fast, no model download)
pytest test_evaluator.py -m "not integration" -v

# Integration tests with real model
pytest test_evaluator.py -m integration --model google/flan-t5-base -v

# All tests
pytest test_evaluator.py --model google/flan-t5-base -v
```

### Generate HTML report

```bash
pytest test_evaluator.py --model google/flan-t5-base -v --html=report.html --self-contained-html
start report.html
```

## How it works
 - Reads test cases from `config.json`
 - Sends each prompt to the model (API or local)
 - Receives the model response
 - Validates response against expected criteria
 - Prints PASS / FAIL / ERROR for each test case
 - Saves full report to `report.json` and `report.csv`

## Modes
| **Mode** | **Command**                      | **Description** |
|----------|----------------------------------|-------------|
| auto     | python evaluator.py              |Tries API first; falls back to local if no token or API fails|
| local    | python evaluator.py --mode local | Uses downloaded model (no internet needed) |
|api | python evaluator.py --mode api | Uses HuggingFace Inference API (requires HF_TOKEN) |

## Supported models

Any HuggingFace model works. Tested with:

| Model | Size | Quality | Command |
|-------|------|---------|---------|
| google/flan-t5-small | ~300 MB | Basic | --model google/flan-t5-small |
| google/flan-t5-base | ~1 GB | Good | --model google/flan-t5-base |
| distilgpt2 | ~330 MB | Basic | --model distilgpt2 |

## Sample output

```text
============================================================
LLM Prompt Evaluator (mode: local)
============================================================

  Running TC001… PASS
  Running TC002… PASS
  Running TC003… FAIL
  Running TC004… PASS
  Running TC005… FAIL

[PASS] TC001 — "What is software testing?" — 8 words, keywords: 3/3
[PASS] TC002 — "Explain quantum computing briefly." — 12 words, keywords: 3/3
[FAIL] TC003 — "List benefits of agile methodology." — 5 words, keywords: 1/3 | too short (5 words); missing keywords: ['sprint', 'team']
[PASS] TC004 — "What causes software bugs?" — 7 words, keywords: 3/3
[FAIL] TC005 — "Describe CI/CD pipeline." — 4 words, keywords: 0/3 | too short (4 words); missing keywords: ['continuous', 'deploy', 'integration']

Results: 3/5 passed, 2 failed, 0 errors
Report saved to report.json
Report saved to report.csv
```

## Test cases configuration

Edit `config.json` to add your own test cases:

```json
{
  "id": "TC001",
  "prompt": "Your prompt here",
  "expected": {
    "not_empty": true,
    "min_words": 5,
    "max_words": 100,
    "keywords": ["word1", "word2"],
    "sentiment": "neutral"
  }
}
```

## Metrics explained

| **Metric** | **Description**                                     |
|----------|-----------------------------------------------------|
| _not_empty_	 | Response must contain text                          |
| _length_ | Word count must be between min_words and max_words  |
| _keywords_ | Response must contain all specified keywords (case-insensitive, substring match)| 
| _sentiment_	| Response tone must match expected sentiment (positive / negative / neutral) |

## Tech stack

- Python 3.10+
- HuggingFace Transformers — local models (FLAN-T5, distilgpt2)
- HuggingFace Inference API — remote models (optional)
- pytest — test framework (unit + integration tests)
- pytest-html — HTML report generation
- requests — HTTP client for API mode

# Author

Ivan Goglev - QA Automation Engineer
- GitHub: goglevivan98