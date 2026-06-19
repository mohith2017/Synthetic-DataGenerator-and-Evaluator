"""LLM generation via DeepEval's Synthesizer + schema-constrained enrichment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from .utils.config import Settings
from .utils.guardrails import HARD_FLAGS, check_generation_input, check_generation_output
from .utils.ingest import Clause, ClauseIndex, select_quote
from .utils.llm import PROVIDER, get_model
from .utils.llm import model_name as get_model_name
from .utils.llm import model_temperature as get_model_temperature
from .utils.schema import (
    AmbiguityType,
    Answerability,
    Category,
    Citation,
    CitationRole,
    DecisionFactor,
    DeterministicChecks,
    Difficulty,
    Escalation,
    EvalBlock,
    ExpectedBehavior,
    ExternalReference,
    GenerationMethod,
    MissingContext,
    Origin,
    Provenance,
    QARecord,
    QuestionType,
    ReviewStatus,
)
from .utils.seeds import SEEDS

PROMPT_VERSION = "synth-v1"

_BOILERPLATE = (
    "capitalised terms",
    "shall have the meaning ascribed",
)


def clause_quality(clause: Clause) -> float:
    """Heuristic 0-1 score: longer, well-structured, non-boilerplate clauses score higher."""
    text = clause.text.strip()
    n = len(text)
    length_score = min(n / 400.0, 1.0)  # saturates around 400 chars
    structure_score = 1.0 if text.endswith((".", ";")) else 0.7
    boiler_penalty = 0.5 if any(b in text.lower() for b in _BOILERPLATE) else 1.0
    return round(length_score * structure_score * boiler_penalty, 3)


@dataclass
class ClauseContext:
    clause_ids: list[str]
    text: str
    quality: float


def build_contexts(index: ClauseIndex, clause_ids: list[str]) -> ClauseContext:
    clauses = [index.get(cid) for cid in clause_ids if index.exists(cid)]
    if not clauses:
        return ClauseContext(clause_ids=[], text="", quality=0.0)
    text = "\n\n".join(c.text for c in clauses)
    quality = round(sum(clause_quality(c) for c in clauses) / len(clauses), 3)
    return ClauseContext(clause_ids=[c.clause_id for c in clauses], text=text, quality=quality)

_EN = " Write the question in clear, professional English."

_STYLING = {
    "A": {
        "scenario": "An engineer or CTO at a fintech asks a question the Razorpay ToS answers explicitly.",
        "task": "Produce a realistic question that the given clause answers clearly and unambiguously." + _EN,
        "input_format": "A single natural-language English question a merchant would ask in Slack.",
        "expected_output_format": (
            "A direct, confident answer to the question, grounded ONLY in the clause text, that "
            "cites the specific clause. Do not hedge and do not ask a clarifying question."
        ),
    },
    "B": {
        "scenario": "A merchant asks something that cannot be answered without more context.",
        "task": "Produce a realistic question whose answer depends on facts the user has not provided." + _EN,
        "input_format": "A single natural-language English question missing a key detail.",
        "expected_output_format": (
            "A response that does NOT answer directly. It asks ONE specific, targeted clarifying "
            "question (never a vague 'can you tell me more?') and explains what that missing detail "
            "would change about the answer, referencing the relevant clause."
        ),
    },
    "C": {
        "scenario": "A merchant asks something the ToS is silent/vague on or defers to external regulation.",
        "task": "Produce a realistic question that surfaces a genuine gap or external cross-reference." + _EN,
        "input_format": "A single natural-language English question about an under-specified area.",
        "expected_output_format": (
            "A response that honestly flags that the ToS is silent, vague, or defers to external "
            "regulation, states what IS known from the clause, and recommends seeking clarification "
            "rather than guessing. Do not assert false certainty."
        ),
    },
}


def _make_synthesizers(model, settings: Settings) -> dict:
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import (
        EvolutionConfig,
        FiltrationConfig,
        StylingConfig,
    )
    from deepeval.synthesizer.types import Evolution

    syn_cfg = settings.pipeline.get("synthesizer", {})
    filt = syn_cfg.get("filtration", {})
    grounded = syn_cfg.get("grounded_evolutions", ["CONCRETIZING", "CONSTRAINED"])
    evolutions = {getattr(Evolution, name): 1.0 / len(grounded) for name in grounded}
    max_conc = int(settings.pipeline.get("generation", {}).get("max_concurrent", 4))

    out = {}
    for cat in ("A", "B", "C"):
        out[cat] = Synthesizer(
            model=model,
            async_mode=True,
            max_concurrent=max_conc,
            styling_config=StylingConfig(**_STYLING[cat]),
            filtration_config=FiltrationConfig(
                critic_model=model,
                synthetic_input_quality_threshold=float(
                    filt.get("synthetic_input_quality_threshold", 0.5)
                ),
                max_quality_retries=int(filt.get("max_quality_retries", 1)),
            ),
            evolution_config=EvolutionConfig(
                evolutions=evolutions,
                num_evolutions=int(syn_cfg.get("num_evolutions", 1)),
            ),
        )
    return out


def _synthesize(synth, ctx_text: str, max_per_ctx: int):
    """Return (question, response, synthetic_input_quality, evolutions) or (None, None, None, []).

    DeepEval generates both the question (``golden.input``) and the full response
    (``golden.expected_output``, shaped by the per-category ``StylingConfig``).
    """
    try:
        goldens = synth.generate_goldens_from_contexts(
            contexts=[[ctx_text]],
            include_expected_output=True,
            max_goldens_per_context=max_per_ctx,
        )
    except Exception:
        return None, None, None, []
    if not goldens:
        return None, None, None, []
    g = goldens[0]
    meta = getattr(g, "additional_metadata", None) or {}
    response = (g.expected_output or "").strip() or None
    return g.input, response, meta.get("synthetic_input_quality"), list(meta.get("evolutions", []) or [])


def _citations(index: ClauseIndex, cids: list) -> list[Citation]:
    out: list[Citation] = []
    for cid in cids:
        clause_id, hint, role, relevance = cid
        clause = index.get(clause_id)
        if clause is None:
            continue
        out.append(
            Citation(
                part=clause.part,
                section=clause.section,
                clause_id=clause_id,
                clause_title=f"{clause.section_title} ({clause.number})",
                quoted_text=select_quote(index, clause_id, hint),
                role=CitationRole(role),
                relevance=relevance,
            )
        )
    return out


_CORRECTABLE_SOFT = {"hallucinated_clause"}


def _assemble_record(model, settings: Settings, index: ClauseIndex, seed: dict, cat: str,
                     question: str, response: str, citations: list[Citation], ctx,
                     model_name: str, provider: str, temperature: float,
                     generated_at: str, siq, evolutions: list) -> QARecord:
    primary_id = citations[0].clause_id
    answer = None
    clarifying_question = None
    missing_context: list[MissingContext] = []
    decision_factors: list[DecisionFactor] = []
    known_facts: list[str] = []
    uncertainty_reason = None
    recommendation = None
    ambiguity_type = None
    external_references: list[ExternalReference] = []
    assumptions: list[str] = []
    escalation = Escalation(required=False)
    method = GenerationMethod.synthesizer_direct

    if cat == "A":
        answer = response
    elif cat == "B":
        method = GenerationMethod.synthesizer_plus_enrichment
        b = decompose_b(model, question, response, ctx.text)
        clarifying_question = b.clarifying_question or "Could you confirm the specific facts of your situation?"
        missing_context = [MissingContext(field=m.field, why_needed=m.why_needed) for m in b.missing_context]
        decision_factors = [
            DecisionFactor(factor=d.factor, how_it_changes_the_answer=d.how_it_changes_the_answer)
            for d in b.decision_factors
        ]
        if not missing_context:
            missing_context = [MissingContext(field="situation_details", why_needed=f"the answer under {primary_id} depends on specifics not provided")]
        if not decision_factors:
            decision_factors = [
                DecisionFactor(factor="the relevant condition is met", how_it_changes_the_answer=f"the rule in {primary_id} would apply"),
                DecisionFactor(factor="it is not met", how_it_changes_the_answer=f"the rule in {primary_id} may not apply"),
            ]
        assumptions = b.assumptions_not_made or ["We did not assume the missing facts of your situation."]
    else:  # C
        method = GenerationMethod.synthesizer_plus_enrichment
        c = decompose_c(model, question, response, ctx.text)
        known_facts = c.known_facts or [f"{primary_id}: {select_quote(index, primary_id)}"]
        uncertainty_reason = c.uncertainty_reason or "the ToS does not fully resolve this and depends on external regulation."
        recommendation = c.recommendation or "Confirm with Razorpay in writing and seek legal advice before proceeding."
        ambiguity_type = AmbiguityType(seed["ambiguity_type"]) if seed.get("ambiguity_type") else AmbiguityType.vague
        external_references = [
            ExternalReference(name=n_, identifier=i_, relationship=r_)
            for (n_, i_, r_) in (seed.get("external_references") or [])
        ]
        if ambiguity_type == AmbiguityType.external_cross_reference and not external_references:
            ambiguity_type = AmbiguityType.vague
        assumptions = c.assumptions_not_made or ["We did not assume facts the ToS leaves open."]
        escalation = Escalation(required=True, reason=seed.get("escalation_reason") or "The ToS does not fully resolve this question.")

    try:
        negative_example = generate_negative_example(model, question, cat, response)
    except Exception:
        negative_example = None

    return QARecord(
        id=f"rzp-{seed['key']}",
        schema_version=settings.schema_version,
        question=question,
        response=response,
        answer=answer if cat != "C" else None,
        negative_example=negative_example,
        category=Category(cat),
        expected_behavior={
            "A": ExpectedBehavior.answer_directly,
            "B": ExpectedBehavior.ask_clarifying_question,
            "C": ExpectedBehavior.flag_ambiguity,
        }[cat],
        answerability={
            "A": Answerability.answerable,
            "B": Answerability.needs_clarification,
            "C": Answerability.indeterminate,
        }[cat],
        clarifying_question=clarifying_question,
        missing_context=missing_context,
        decision_factors=decision_factors,
        known_facts=known_facts,
        uncertainty_reason=uncertainty_reason,
        recommendation=recommendation,
        ambiguity_type=ambiguity_type,
        assumptions_not_made=assumptions,
        citations=citations,
        external_references=external_references,
        escalation=escalation,
        review_status=ReviewStatus(seed.get("review_status", "unreviewed")),
        persona=seed["persona"],
        topic=seed["topic"],
        difficulty=Difficulty(seed["difficulty"]),
        question_type=QuestionType(seed["question_type"]),
        origin=Origin(seed.get("origin", "synthesized")),
        generation_method=method,
        source_doc=index.source_filename,
        source_hash=index.source_sha256,
        source_parsed_at=generated_at,
        jurisdiction=settings.jurisdiction,
        tos_effective_date=settings.tos_effective_date,
        provenance=Provenance(
            model=model_name,
            provider=provider,
            prompt_version=PROMPT_VERSION,
            seed=settings.seed,
            temperature=temperature,
            generated_at=generated_at,
            evolutions=evolutions,
            context_quality=ctx.quality,
            synthetic_input_quality=siq,
        ),
    )


def _record_problems(record: QARecord, index: ClauseIndex) -> list[str]:
    """Correctable failures: citation quote mismatch + hard (and hallucinated-clause) flags."""
    checks = verify_citations(record, index)
    problems: list[str] = []
    if not (checks.citation_exists and checks.quote_matches):
        problems.append("citation_quote_mismatch")
    gr = check_generation_output(record, index)
    for f in gr.flags:
        head = f.split(":")[0]
        if head in HARD_FLAGS or head in _CORRECTABLE_SOFT:
            problems.append(f)
    return problems


def self_correct(model, cat: str, question: str, response: str, clause_text: str,
                 checker, max_attempts: int) -> tuple[str, list[str]]:
    problems = checker(response)
    attempts = 0
    while problems and attempts < max_attempts:
        response = correct_response(model, cat, question, response, clause_text, problems)
        problems = checker(response)
        attempts += 1
    return response, problems


def _build_seed_record(model, settings: Settings, index: ClauseIndex, seed: dict, cat: str,
                       citations: list[Citation], ctx, primary_id: str, question: str,
                       response: str, siq, evolutions: list, model_name: str, provider: str,
                       temperature: float, generated_at: str, max_self_correct: int) -> QARecord:
    last: dict[str, QARecord] = {}

    def checker(resp: str) -> list[str]:
        rec = _assemble_record(model, settings, index, seed, cat, question, resp, citations,
                               ctx, model_name, provider, temperature, generated_at, siq, evolutions)
        last["record"] = rec
        return _record_problems(rec, index)

    clause = index.get(primary_id)
    clause_text = clause.text if clause else ctx.text
    _, remaining = self_correct(model, cat, question, response, clause_text, checker, max_self_correct)
    if remaining:
        print(f"  [warn] {seed['key']} unresolved after {max_self_correct} self-corrections: {remaining}", flush=True)
    return last["record"]


def generate(index: ClauseIndex, settings: Settings) -> list[QARecord]:
    provider = PROVIDER
    models_cfg = settings.pipeline.get("models", {})
    model = get_model(models_cfg)
    model_name = get_model_name(models_cfg)
    temperature = get_model_temperature(models_cfg)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    synths = _make_synthesizers(model, settings)
    max_per_ctx = int(settings.pipeline.get("synthesizer", {}).get("max_goldens_per_context", 1))
    max_self_correct = int(settings.pipeline.get("generation", {}).get("max_self_correct", 2))

    records: list[QARecord] = []
    total = len(SEEDS)
    for n, seed in enumerate(SEEDS, 1):
        cat = seed["category"]
        cids = seed["cids"]
        gin = check_generation_input(seed, index)
        if not gin.ok:
            print(f"  [skip] {seed['key']} failed input guardrail: {gin.flags}", flush=True)
            continue
        citations = _citations(index, cids)
        if not citations:
            continue
        ctx = build_contexts(index, [c.clause_id for c in citations])
        primary_id = citations[0].clause_id

        print(f"  [{n}/{total}] {seed['key']} ({cat}) …", flush=True)
        question, response, siq, evolutions = _synthesize(synths[cat], ctx.text, max_per_ctx)
        if not question or not response:
            fb = generate_question_and_response(model, cat, ctx.text, seed.get("exemplar"))
            question = question or fb.question
            response = response or fb.response

        records.append(
            _build_seed_record(
                model, settings, index, seed, cat, citations, ctx, primary_id,
                question, response, siq, evolutions, model_name, provider,
                temperature, generated_at, max_self_correct,
            )
        )
    return records


class _MC(BaseModel):
    field: str
    why_needed: str


class _DF(BaseModel):
    factor: str
    how_it_changes_the_answer: str


class BDecomposition(BaseModel):
    clarifying_question: str
    missing_context: list[_MC] = Field(default_factory=list)
    decision_factors: list[_DF] = Field(default_factory=list)
    assumptions_not_made: list[str] = Field(default_factory=list)


class CDecomposition(BaseModel):
    known_facts: list[str] = Field(default_factory=list)
    uncertainty_reason: str = ""
    recommendation: str = ""
    assumptions_not_made: list[str] = Field(default_factory=list)


class QuestionAndResponse(BaseModel):
    question: str
    response: str


class NegativeExample(BaseModel):
    bad_answer: str


class CorrectedResponse(BaseModel):
    response: str


def _gen(model, prompt: str, schema):
    """Generate structured output from an LLM."""
    import os

    import anthropic as _anthropic
    from deepeval.models.llms.anthropic_model import AnthropicModel

    if isinstance(model, AnthropicModel):
        raw_key = (
            model.api_key.get_secret_value()
            if model.api_key is not None
            else os.environ.get("ANTHROPIC_API_KEY", "")
        )
        client = _anthropic.Anthropic(api_key=raw_key)
        tool_def = {
            "name": "output",
            "description": "Return the structured result",
            "input_schema": schema.model_json_schema(),
        }
        resp = client.messages.create(
            model=model.name,
            max_tokens=1024,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "output"},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_block = next(b for b in resp.content if b.type == "tool_use")
        return schema.model_validate(tool_block.input)

    out = model.generate(prompt, schema=schema)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, schema):
        return out
    return schema.model_validate_json(out) if isinstance(out, str) else schema.model_validate(out)


_EN_ENRICH = "Respond in clear, professional English only. "


def decompose_b(model, question: str, response: str, context: str) -> BDecomposition:
    """Structure a DeepEval-generated clarifying response into its B schema fields."""
    prompt = (
        _EN_ENRICH + "A compliance assistant produced the RESPONSE below to a question that cannot be "
        "answered without more context. Do NOT write a new answer — extract structure from the "
        "existing response: the single specific clarifying question it asks, the missing context "
        "fields it needs, and how each factor would change the answer.\n\n"
        f"Relevant clause(s):\n{context}\n\n"
        f"Question: {question}\n\n"
        f"RESPONSE: {response}"
    )
    return _gen(model, prompt, BDecomposition)


def decompose_c(model, question: str, response: str, context: str) -> CDecomposition:
    """Structure a DeepEval-generated ambiguity response into its C schema fields."""
    prompt = (
        _EN_ENRICH + "A compliance assistant produced the RESPONSE below to a question the ToS is silent, "
        "vague, or externally-deferring on. Do NOT write a new answer — extract structure from the "
        "existing response: what the ToS does say (known_facts), why it does not resolve the "
        "question (uncertainty_reason), and what the user should do (recommendation).\n\n"
        f"Relevant clause(s):\n{context}\n\n"
        f"Question: {question}\n\n"
        f"RESPONSE: {response}"
    )
    return _gen(model, prompt, CDecomposition)


def generate_negative_example(model, question: str, category: str, good_response: str) -> str:
    """Produce a plausible-but-WRONG answer that exemplifies this category's failure mode.

    Used as a contrastive training signal — what a bad assistant would do:
      * A: a confident answer with a wrong fact/number/obligation.
      * B: a confident guess that picks one interpretation instead of asking for context.
      * C: false certainty asserting a definitive answer the ToS does not actually support.
    """
    failure = {
        "A": "state a confident answer that gets a key fact, number, or obligation WRONG.",
        "B": "confidently guess and commit to one interpretation instead of asking for the "
        "missing context (the failure mode the assistant must avoid).",
        "C": "assert false certainty — give a definitive answer the ToS does not actually "
        "support, instead of honestly flagging the ambiguity.",
    }[category]
    prompt = (
        _EN_ENRICH + "For training a compliance assistant we need a NEGATIVE example: a realistic but "
        f"INCORRECT answer to the question below. Write a bad answer that would {failure} "
        "Keep it short and plausible-sounding (this is intentionally wrong, for contrast).\n\n"
        f"Question: {question}\n\n"
        f"The correct response was: {good_response}\n\n"
        'Return JSON {"bad_answer": "..."}.'
    )
    return _gen(model, prompt, NegativeExample).bad_answer.strip()


_CORRECT_CONTRACT = {
    "A": "a direct, confident answer grounded ONLY in the clause that cites it; do not hedge.",
    "B": "a response that does NOT answer directly but asks ONE specific clarifying question and "
    "explains what the missing detail would change, referencing the clause.",
    "C": "a response that honestly flags that the ToS is silent/vague/defers externally, states "
    "what IS known from the clause, and recommends seeking clarification; assert NO false certainty.",
}


def correct_response(
    model, category: str, question: str, response: str, clause_text: str, problems: list[str]
) -> str:
    """Re-prompt the model to fix specific deterministic/guardrail failures in a response.

    The model is given the exact clause text it must ground to and the named problems
    to repair (e.g. hallucinated clause refs, false certainty, too-short/refusal).
    """
    issues = "; ".join(problems)
    prompt = (
        _EN_ENRICH + "A compliance assistant's RESPONSE below FAILED automated checks and must be "
        f"rewritten to fix them. Problems to fix: {issues}.\n"
        "Rules: ground ONLY in the clause text provided; reference ONLY clause numbers that "
        "appear in it (do not invent 'Clause X.Y'); keep it substantive (not a refusal). "
        f"Produce {_CORRECT_CONTRACT[category]}\n\n"
        f"Clause text (the ONLY allowed grounding):\n{clause_text}\n\n"
        f"Question: {question}\n\n"
        f"Failed RESPONSE: {response}\n\n"
        'Return JSON {"response": "..."}.'
    )
    return _gen(model, prompt, CorrectedResponse).response.strip()


def generate_question_and_response(
    model, category: str, context: str, exemplar: Optional[str] = None
) -> QuestionAndResponse:
    """Fallback: produce a (question, response) pair when the Synthesizer drops a context."""
    contract = {
        "A": (
            "a question the clause answers clearly and unambiguously, and a direct, confident "
            "answer grounded only in the clause that cites the clause"
        ),
        "B": (
            "a question that cannot be answered without more context the user hasn't given, and a "
            "response that does NOT answer directly but asks one specific, targeted clarifying "
            "question and explains what that detail would change"
        ),
        "C": (
            "a question the clause is silent/vague on or defers to external regulation, and a "
            "response that honestly flags the gap, states what is known, and recommends seeking "
            "clarification rather than guessing"
        ),
    }[category]
    ex = f"\nExample question style: {exemplar}" if exemplar else ""
    prompt = (
        _EN_ENRICH + "You write realistic questions a fintech engineer/CTO would ask in Slack about their "
        f"Razorpay terms, with the assistant's response. Produce {contract}, grounded in the "
        f"clause below.{ex}\n\n"
        f"Clause(s):\n{context}\n\n"
        'Return JSON {"question": "...", "response": "..."}.'
    )
    return _gen(model, prompt, QuestionAndResponse)


@dataclass
class PostprocessReport:
    kept: int = 0
    dropped_citation: list[str] = field(default_factory=list)
    dropped_duplicate: list[str] = field(default_factory=list)
    dropped_language: list[str] = field(default_factory=list)
    dropped_guardrail: list[str] = field(default_factory=list)
    guardrail_flagged: list[str] = field(default_factory=list)


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().lower()


def _non_english_ratio(s: str) -> float:
    """Fraction of non-ASCII characters; flags language drift."""
    if not s:
        return 0.0
    return sum(1 for ch in s if ord(ch) > 127) / len(s)


def _is_non_english(rec: QARecord, threshold: float = 0.05) -> bool:
    return _non_english_ratio(rec.question) > threshold or _non_english_ratio(rec.response) > threshold


def verify_citations(record: QARecord, index: ClauseIndex) -> DeterministicChecks:
    if not record.citations:
        # Only A strictly requires citations; B/C may legitimately have them too.
        return DeterministicChecks(citation_exists=True, quote_matches=True)
    citation_exists = all(index.exists(c.clause_id) for c in record.citations)
    quote_matches = all(
        index.quote_contains(c.clause_id, c.quoted_text) for c in record.citations
    )
    return DeterministicChecks(citation_exists=citation_exists, quote_matches=quote_matches)


def postprocess(records: list[QARecord], index: ClauseIndex) -> tuple[list[QARecord], PostprocessReport]:
    report = PostprocessReport()
    seen_questions: set[str] = set()
    kept: list[QARecord] = []

    for rec in records:
        checks = verify_citations(rec, index)
        rec.eval = EvalBlock(deterministic_checks=checks)

        if rec.citations and not (checks.citation_exists and checks.quote_matches):
            report.dropped_citation.append(rec.id)
            continue

        if _is_non_english(rec):
            report.dropped_language.append(rec.id)
            continue

        key = _norm_q(rec.question)
        if key in seen_questions:
            report.dropped_duplicate.append(rec.id)
            continue

        gr = check_generation_output(rec, index)
        if gr.hard_flags:
            report.dropped_guardrail.append(rec.id)
            continue
        if gr.flags:
            report.guardrail_flagged.append(rec.id)

        seen_questions.add(key)
        kept.append(rec)

    report.kept = len(kept)
    return kept, report
