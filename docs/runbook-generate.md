# Runbook — Generate

```bash
# Prereq: ANTHROPIC_API_KEY set in the environment (or .env)
python -m razorpay_qa generate [--seed N]
```

## Stages

| # | Module | What happens |
|---|--------|-------------|
| 1 | `ingest/` (`load_source`) | Parse PDF, SHA-256 the bytes, cache cleaned text to `artifacts/source/` |
| 2 | `ingest/` (`build_clause_index`) | Build canonical clause index (`PartA/3.4` → verbatim text) |
| 3 | `generation/generate.py` | DeepEval `Synthesizer` generates question + response per seed; self-correction loop re-grounds failures (≤ `max_self_correct` re-prompts) |
| 4 | `generation/postprocess.py` | Verify citations verbatim; dedup; run guardrails (hard flags drop, soft flags annotate) |
| 5 | `generation/curate.py` | Balance check (≥15/cat), splits, freeze to `artifacts/dataset/vN/`, hash + registry + LATEST |

## Outputs

```
artifacts/dataset/
  vN/dataset.jsonl          frozen, versioned dataset
  vN/dataset_card.md        counts, distributions, limitations
  vN/run_manifest.json      model, seed, sha256, git provenance, drop stats, splits
  versions.json             append-only version registry
  LATEST                    newest version tag
  dataset.jsonl             latest convenience copy
```

## Notes

- `--seed` (default `7`) drives sampling/ordering/splits; recorded per record.
- Questions and answers are LLM-generated (non-deterministic). Grounding, citations, A/B/C categories, and verified C labels are deterministic (from `seeds.py`).
- 5 anchor seeds (`origin=seed_anchor`) carry Hyde's example questions as few-shot hints.
