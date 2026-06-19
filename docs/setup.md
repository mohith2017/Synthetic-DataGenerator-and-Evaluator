# Setup

## Requirements
- Python 3.10+ (developed/run on 3.12).
- The source PDF at `artifacts/source/Razorpay Terms & Conditions.pdf`.
- An Anthropic API key (`ANTHROPIC_API_KEY`) — required for generation.
- An OpenAI API key (`OPENAI_API_KEY`) — optional; enables GPT-5.5 as the judge. Falls back to Anthropic if absent.

## Install
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .            # deepeval + anthropic + openai (all runtime deps)
cp .env.example .env        # then fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### API keys
1. **`ANTHROPIC_API_KEY`** (required) — used by DeepEval `Synthesizer` for generation.
   Model: `claude-haiku-4-5-20251001` (configured under `models.anthropic` in `config/pipeline.yaml`).
2. **`OPENAI_API_KEY`** (optional) — used by DeepEval `GEval` + `DAGMetric` for judging.
   Model: `gpt-5.5` (configured under `models.judge` in `config/pipeline.yaml`).
   If absent, judging falls back to the Anthropic model.

## Dev tooling
```bash
pip install -e ".[dev]"     # ruff + pytest
ruff check src tests
pytest -q
```

## Configuration
All run settings live in `config/pipeline.yaml` (models, seed, counts, thresholds,
splits) and `config/taxonomy.yaml` (enum vocabularies). Nothing that affects outputs is
hard-coded in the source.
