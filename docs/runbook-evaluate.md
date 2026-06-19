# Runbook — Evaluate

```bash
# Prereq: ANTHROPIC_API_KEY set in the environment (or .env)
python -m razorpay_qa evaluate [--dataset PATH]
```

## What it does

| # | What |
|---|------|
| 1 | Load dataset + rebuild clause index |
| 2 | Deterministic citation checks (clause exists + verbatim quote) — hard pass/fail |
| 3 | LLM-as-judge: `GEval` for subjective dimensions + `DAGMetric` for Category-A (deterministic decision tree) |
| 4 | Eval guardrails per record (judge errors, missing dimensions) → `eval.guardrail_flags` |
| 5 | Write `artifacts/eval/vN/` — `eval_summary.md`, `dataset_evaluated.jsonl`, `eval_manifest.json` |

## Scoring dimensions

| Dimension | Applies | Measures |
|-----------|---------|----------|
| `citation_accuracy` | A B C | cited clause supports the answer |
| `faithfulness` | A B C | no claims beyond the clause context |
| `category_correctness` | A | DAGMetric: direct answer AND cites a clause |
| `question_realism` | A B C | a real engineer/CTO would ask this |
| `clarification_specificity` | B | targeted clarifying question, not vague |
| `ambiguity_honesty` | C | honestly flags uncertainty, no false certainty |

## Reading `eval_summary.md`

- **Overall composite + pass rate** (threshold: `evaluation.pass_threshold`).
- **Per-dimension means** and **stratified composite** (by category / topic / difficulty).
- **3 worst examples** with scores, `failure_stage`, and suggested fix.

## Note

Generation and judging use the same model — scores carry a self-preference bias. Treat `GEval` scores as directional; use the deterministic citation checks as the hard gate.
