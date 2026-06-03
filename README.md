# LLM Prompt Evaluator

A QA tool for evaluating LLM responses using Hugging Face Inference API.

## What it does

Sends prompts to a language model and validates responses against expected criteria:

- **Not empty** — response must contain text
- **Length** — word count within min/max range
- **Keywords** — must include specified keywords
- **Sentiment** — expected tone (POSITIVE / NEGATIVE / NEUTRAL)

## Why I built this

As a QA Automation Engineer, I wanted to explore how testing principles apply to LLM outputs. This tool demonstrates systematic validation of non-deterministic systems — similar to testing complex distributed applications.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-prompt-evaluator.git
cd llm-prompt-evaluator