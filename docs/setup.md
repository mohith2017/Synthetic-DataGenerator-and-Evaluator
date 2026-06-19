# Setup

## Requirements
- Python 3.10+ (developed/run on 3.12).
- The source PDF at `artifacts/source/Razorpay Terms & Conditions.pdf`.
- An Anthropic API key (`ANTHROPIC_API_KEY`).

## Install
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .            # deepeval + anthropic (required runtime deps)
cp .env.example .env        # then set ANTHROPIC_API_KEY
```

### Anthropic Claude (the only provider)
1. Set `ANTHROPIC_API_KEY` in `.env` (or export it in the shell).
2. Model is configured under `models.anthropic` in `config/pipeline.yaml`
   (default `claude-haiku-4-5-20251001`). Generation (DeepEval `Synthesizer`) and judging
   (DeepEval `GEval`) both use it.

## Dev tooling
```bash
pip install -e ".[dev]"     # ruff + pytest
ruff check src tests
pytest -q
```

## Configuration
All run settings live in `config/pipeline.yaml` (provider, seed, counts, thresholds,
splits) and `config/taxonomy.yaml` (enum vocabularies). Nothing that affects outputs is
hard-coded in the source.
