"""LLM-as-judge evaluation: scoring dimensions + the run/render pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .utils.config import DATASET_DIR, EVAL_DIR, Settings
from .utils.guardrails import check_eval_result
from .utils.ingest import ClauseIndex, build_clause_index, load_source
from .utils.llm import PROVIDER, get_judge_model, get_model
from .utils.llm import judge_model_name as get_judge_model_name
from .utils.llm import model_name as get_model_name
from .utils.schema import DeterministicChecks, EvalBlock, QARecord, ReviewStatus


def _category_a_dag_metric(model):
    """Deterministic decision tree for the Category-A contract: direct answer AND cites a clause."""
    from deepeval.metrics import DAGMetric
    from deepeval.metrics.dag import (
        BinaryJudgementNode,
        DeepAcyclicGraph,
        VerdictNode,
    )
    from deepeval.test_case import LLMTestCaseParams as P

    params = [P.INPUT, P.ACTUAL_OUTPUT]
    cites_clause = BinaryJudgementNode(
        criteria="Does the response cite or reference a specific ToS clause to support the answer?",
        evaluation_params=params,
        children=[
            VerdictNode(verdict=False, score=0),
            VerdictNode(verdict=True, score=10),
        ],
    )
    direct_answer = BinaryJudgementNode(
        criteria=(
            "Does the response give a direct, confident answer to the question, rather than "
            "asking a clarifying question or hedging that the terms are ambiguous?"
        ),
        evaluation_params=params,
        children=[
            VerdictNode(verdict=False, score=0),
            VerdictNode(verdict=True, child=cites_clause),
        ],
    )
    dag = DeepAcyclicGraph(root_nodes=[direct_answer])
    return DAGMetric(name="Category Correctness", dag=dag, model=model)


def build_geval_metrics(model, category: str) -> list:
    from deepeval.metrics import GEval
    from deepeval.metrics.g_eval import Rubric
    from deepeval.test_case import SingleTurnParams as P

    metrics = [
        GEval(
            name="Citation Accuracy",
            model=model,
            evaluation_steps=[
                "Read the cited clause text provided as context / expected output.",
                "Check that the answer's claims are directly supported by that clause text.",
                "Reference the specific clause phrase you relied on.",
                "Penalise answers that cite a clause that does not actually support them.",
            ],
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
            rubric=[
                Rubric(score_range=(0, 3), expected_outcome="Cited clause does not support the answer."),
                Rubric(score_range=(4, 7), expected_outcome="Partially supported or imprecise."),
                Rubric(score_range=(8, 10), expected_outcome="Fully and precisely supported by the cited clause."),
            ],
        ),
        GEval(
            name="Faithfulness",
            model=model,
            evaluation_steps=[
                "Identify any factual claim in the answer not present in the clause context.",
                "Penalise hallucinated obligations, numbers, or timelines.",
            ],
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
            rubric=[
                Rubric(score_range=(0, 3), expected_outcome="Contains claims unsupported by the context."),
                Rubric(score_range=(4, 7), expected_outcome="Mostly grounded with minor drift."),
                Rubric(score_range=(8, 10), expected_outcome="Every claim grounded in the context."),
            ],
        ),
        GEval(
            name="Question Realism",
            model=model,
            evaluation_steps=[
                "Judge whether a fintech engineer or CTO would plausibly ask this in Slack.",
                "Penalise contrived, unnatural, or template-sounding questions.",
            ],
            evaluation_params=[P.INPUT],
            rubric=[
                Rubric(score_range=(0, 3), expected_outcome="Contrived or implausible."),
                Rubric(score_range=(4, 7), expected_outcome="Plausible but slightly unnatural."),
                Rubric(score_range=(8, 10), expected_outcome="Natural, realistic merchant question."),
            ],
        ),
    ]

    if category == "A":
        metrics.append(_category_a_dag_metric(model))
    elif category == "B":
        metrics.append(
            GEval(
                name="Clarification Specificity",
                model=model,
                evaluation_steps=[
                    "Check the response asks a specific, targeted clarifying question.",
                    "Penalise vague asks like 'can you tell me more?'.",
                    "Reward explaining what the missing detail would change about the answer.",
                ],
                evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
                rubric=[
                    Rubric(score_range=(0, 3), expected_outcome="Vague or no real clarifying question."),
                    Rubric(score_range=(4, 7), expected_outcome="Asks but does not explain the impact."),
                    Rubric(score_range=(8, 10), expected_outcome="Specific question + explains what it changes."),
                ],
            )
        )
    else:  # C
        metrics.append(
            GEval(
                name="Ambiguity Honesty",
                model=model,
                evaluation_steps=[
                    "Check the response honestly flags that the ToS is silent/vague/defers externally.",
                    "Penalise false certainty; reward stating what is known and recommending clarification.",
                ],
                evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
                rubric=[
                    Rubric(score_range=(0, 3), expected_outcome="Asserts false certainty."),
                    Rubric(score_range=(4, 7), expected_outcome="Hedges but weak on what's known / next steps."),
                    Rubric(score_range=(8, 10), expected_outcome="Honestly flags the gap + recommends clarification."),
                ],
            )
        )
    return metrics


def metric_key(name: str) -> str:
    return name.lower().replace(" ", "_")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_eval_manifest(records, version, provider, model, threshold, dataset_sha256, out_dir):
    """Mirror the dataset registry on the eval side: record what was judged + how."""
    comps = [r.eval.composite_score for r in records
             if r.eval and r.eval.composite_score is not None]
    n_pass = sum(1 for c in comps if c >= threshold)
    det_pass = sum(
        1 for r in records
        if r.eval and r.eval.deterministic_checks.citation_exists
        and r.eval.deterministic_checks.quote_matches
    )
    gr_breakdown: dict[str, int] = defaultdict(int)
    for r in records:
        for f in (r.eval.guardrail_flags if r.eval else []):
            gr_breakdown[f.split(":")[0]] += 1

    manifest = {
        "version": version,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": dataset_sha256,
        "provider": provider,
        "model": model,
        "threshold": threshold,
        "totals": {
            "records": len(records),
            "passed": n_pass,
            "deterministic_citation_passed": det_pass,
        },
        "guardrail_flags": dict(gr_breakdown),
    }
    path = out_dir / "eval_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _load_records(path: Path) -> list[QARecord]:
    return [
        QARecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _deterministic(rec: QARecord, index: ClauseIndex) -> DeterministicChecks:
    if not rec.citations:
        return DeterministicChecks(citation_exists=True, quote_matches=True)
    return DeterministicChecks(
        citation_exists=all(index.exists(c.clause_id) for c in rec.citations),
        quote_matches=all(index.quote_contains(c.clause_id, c.quoted_text) for c in rec.citations),
    )


def _failure_stage(rec: QARecord, checks: DeterministicChecks, scores: dict, threshold: float):
    if not (checks.citation_exists and checks.quote_matches):
        return "citation_mapping"
    if not scores:
        return None
    composite = mean(scores.values())
    if composite >= threshold:
        return None
    low = min(scores, key=lambda k: scores[k])
    if "clarification" in low or "ambiguity" in low:
        return "enrichment"
    return "generation"


async def _score_record(rec: QARecord, index: ClauseIndex, model, sem: asyncio.Semaphore, threshold: float):
    from deepeval.test_case import LLMTestCase

    actual = rec.answer or rec.response
    expected = (rec.citations[0].quoted_text if rec.citations else "") or rec.recommendation or ""
    context = [c.quoted_text for c in rec.citations] or [""]
    test_case = LLMTestCase(input=rec.question, actual_output=actual, expected_output=expected, context=context)

    metrics = build_geval_metrics(model, rec.category.value)
    expected_keys = {metric_key(m.name) for m in metrics}
    scores: dict[str, float] = {}
    reasons: list[str] = []

    async with sem:
        for metric in metrics:
            try:
                await metric.a_measure(test_case)
                scores[metric_key(metric.name)] = round(float(metric.score), 3)
                if metric.reason:
                    reasons.append(f"{metric.name}: {metric.reason}")
            except Exception as exc:  # keep going; record the failure
                scores[metric_key(metric.name)] = 0.0
                reasons.append(f"{metric.name}: ERROR {exc}")

    checks = _deterministic(rec, index)
    composite = round(mean(scores.values()), 3) if scores else 0.0
    rec.eval = EvalBlock(
        scores=scores,
        judge_rationale=" | ".join(reasons)[:1500],
        composite_score=composite,
        deterministic_checks=checks,
        failure_stage=_failure_stage(rec, checks, scores, threshold),
    )
    rec.eval.guardrail_flags = check_eval_result(rec, expected_keys)
    return rec


async def _run(records, index, model, max_conc: int, threshold: float):
    sem = asyncio.Semaphore(max_conc)
    await asyncio.gather(*(_score_record(r, index, model, sem, threshold) for r in records))


def evaluate(settings: Settings, dataset_path: Path | None = None) -> Path:
    provider = PROVIDER
    dataset_path = dataset_path or (DATASET_DIR / "dataset.jsonl")
    records = _load_records(dataset_path)

    doc = load_source(settings.pdf_path)
    index = build_clause_index(doc)
    models_cfg = settings.pipeline.get("models", {})
    model = get_judge_model(models_cfg)
    model_name = get_judge_model_name(models_cfg)
    dataset_sha256 = _sha256_file(dataset_path)
    max_conc = int(settings.pipeline.get("evaluation", {}).get("max_concurrent", 4))
    threshold = settings.pass_threshold

    print(f"Judging {len(records)} records with {model_name} (concurrency={max_conc})…")
    asyncio.run(_run(records, index, model, max_conc, threshold))

    version = (records[0].dataset_version if records and records[0].dataset_version else "v0")
    version_dir = EVAL_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    eval_out = version_dir / "dataset_evaluated.jsonl"
    with eval_out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")

    out_path = version_dir / "eval_summary.md"
    summary = _render(records, provider, threshold)
    out_path.write_text(summary, encoding="utf-8")
    _write_eval_manifest(records, version, provider, model_name, threshold, dataset_sha256, version_dir)
    (EVAL_DIR / "eval_summary.md").write_text(summary, encoding="utf-8")
    return out_path


def _agg_dims(records) -> dict:
    by: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.eval:
            for k, v in r.eval.scores.items():
                by[k].append(v)
    return {k: round(mean(v), 3) for k, v in sorted(by.items())}


def _strat(records, key) -> dict:
    b: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.eval and r.eval.composite_score is not None:
            b[str(key(r))].append(r.eval.composite_score)
    return {k: round(mean(v), 3) for k, v in sorted(b.items())}


def _fmt(d: dict) -> str:
    return ", ".join(f"`{k}`: {v}" for k, v in d.items()) if d else "n/a"


def _analysis(r: QARecord) -> str:
    stage = r.eval.failure_stage or "n/a"
    low = min(r.eval.scores.items(), key=lambda kv: kv[1]) if r.eval.scores else ("none", 0.0)
    fixes = {
        "citation_mapping": "re-map/verify the quote against the clause before write.",
        "generation": "tighten the faithfulness rubric so key facts must appear in the cited clause.",
        "enrichment": "require a concrete missing field / explicit uncertainty flag in enrichment.",
        "n/a": "add a targeted rubric check for the weakest dimension.",
    }
    return f"Weakest dim `{low[0]}`={low[1]}; likely **{stage}** stage. Fix: {fixes.get(stage, fixes['n/a'])}"


def _render(records, provider, threshold) -> str:
    total = len(records)
    comps = [r.eval.composite_score for r in records if r.eval and r.eval.composite_score is not None]
    overall = round(mean(comps), 3) if comps else 0.0
    n_pass = sum(1 for c in comps if c >= threshold)
    det_pass = sum(
        1 for r in records
        if r.eval and r.eval.deterministic_checks.citation_exists and r.eval.deterministic_checks.quote_matches
    )
    anchor = [r.eval.composite_score for r in records if r.review_status == ReviewStatus.human_verified and r.eval]
    rest = [r.eval.composite_score for r in records if r.review_status != ReviewStatus.human_verified and r.eval]
    worst = sorted([r for r in records if r.eval], key=lambda r: r.eval.composite_score or 1.0)[:3]

    gr_flagged = sum(1 for r in records if r.eval and r.eval.guardrail_flags)
    gr_breakdown: dict[str, int] = defaultdict(int)
    for r in records:
        for f in (r.eval.guardrail_flags if r.eval else []):
            gr_breakdown[f.split(":")[0]] += 1

    lines = [
        "# Evaluation Summary — Razorpay ToS Q&A Dataset",
        "",
        f"- **Judge:** `{provider}` (DeepEval G-Eval)",
        f"- **Records evaluated:** {total}",
        f"- **Overall mean composite:** {overall}",
        f"- **Pass rate (composite ≥ {threshold}):** {n_pass}/{total} ({n_pass / total:.0%})" if total else "",
        f"- **Deterministic citation checks passed:** {det_pass}/{total} ({det_pass / total:.0%})" if total else "",
        "",
        "## Scoring dimensions (0–1)",
        "- **citation_accuracy** — cited clause actually supports the answer (+ substring check).",
        "- **faithfulness** — no claims beyond the cited ToS context.",
        "- **category_correctness** — matches the A/B/C behaviour contract (A: deterministic DAG decision tree — direct answer AND cites a clause).",
        "- **question_realism** — a fintech engineer/CTO would plausibly ask it.",
        "- **clarification_specificity** — (B) targeted, not vague.",
        "- **ambiguity_honesty** — (C) flags uncertainty without false certainty.",
        "",
        "## Per-dimension means",
        _fmt(_agg_dims(records)),
        "",
        "## Stratified composite",
        f"- **By category:** {_fmt(_strat(records, lambda r: r.category.value))}",
        f"- **By topic:** {_fmt(_strat(records, lambda r: r.topic))}",
        f"- **By difficulty:** {_fmt(_strat(records, lambda r: r.difficulty.value))}",
        f"- **By question_type:** {_fmt(_strat(records, lambda r: r.question_type.value))}",
        "",
        "## Judge calibration (lightweight)",
        f"- Mean composite on human-verified seed anchors: {round(mean(anchor), 3) if anchor else 'n/a'} (n={len(anchor)})",
        f"- Mean composite on the rest: {round(mean(rest), 3) if rest else 'n/a'} (n={len(rest)})",
        "",
        "## Guardrails (judge-side)",
        f"- Records with eval guardrail flags: {gr_flagged}/{total}"
        + (f" — {_fmt(dict(gr_breakdown))}" if gr_breakdown else " (none)"),
        "",
        "## 3 worst examples",
        "",
    ]
    for i, r in enumerate(worst, 1):
        lines += [
            f"### {i}. `{r.id}` (category {r.category.value}, composite {r.eval.composite_score})",
            f"- **Question:** {r.question}",
            f"- **Scores:** {_fmt(r.eval.scores)}",
            f"- **Analysis (~50 words):** {_analysis(r)}",
            "",
        ]
    return "\n".join(line for line in lines if line is not None)


