"""Deterministic guardrails for the synthetic pipeline (LLM-free)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import ClauseIndex
from .schema import QARecord

_PII_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b"),                      
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),                             
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                            
    re.compile(r"\b(?:\d[ \-]?){13,16}\b"),                           
]
_PROFANITY = {"fuck", "shit", "bastard", "asshole", "bitch", "damn"}
_FALSE_CERTAINTY = re.compile(
    r"\b(definitely|certainly|guaranteed|without (?:a )?doubt|absolutely|"
    r"there is no doubt|rest assured|100% certain)\b",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(
    r"\b(unclear|ambiguous|not (?:specified|defined|addressed)|silent|"
    r"does not (?:specify|define|state)|recommend|consult|confirm|clarif|"
    r"uncertain|depends|may|cannot be determined|seek (?:legal|further))\b",
    re.IGNORECASE,
)
_REFUSAL = re.compile(
    r"\b(i (?:can(?:not|'t)|am unable to|won't)|as an ai|i'm sorry,? but)\b",
    re.IGNORECASE,
)
_CLAUSE_REF = re.compile(r"(?:clause|section)\s+(\d{1,2}\.\d{1,2}[A-Z]?)", re.IGNORECASE)

HARD_FLAGS = {"empty_or_too_short", "refusal", "pii_in_output", "c_false_certainty"}


@dataclass
class GuardrailResult:
    ok: bool
    flags: list[str] = field(default_factory=list)

    @property
    def hard_flags(self) -> list[str]:
        return [f for f in self.flags if f.split(":")[0] in HARD_FLAGS]


def _contains_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


def _contains_profanity(text: str) -> bool:
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return bool(words & _PROFANITY)


def check_generation_input(seed: dict, index: ClauseIndex, min_chars: int = 40) -> GuardrailResult:
    """Validate a seed's grounding before it is handed to the synthesizer."""
    flags: list[str] = []
    cids = seed.get("cids") or []
    if not cids:
        flags.append("no_context")
    for cid in cids:
        clause_id = cid[0]
        clause = index.get(clause_id)
        if clause is None:
            flags.append(f"missing_clause:{clause_id}")
            continue
        if len(clause.text.strip()) < min_chars:
            flags.append(f"thin_context:{clause_id}")
        if _contains_pii(clause.text):
            flags.append(f"pii_in_context:{clause_id}")
    return GuardrailResult(ok=not flags, flags=flags)


def check_generation_output(rec: QARecord, index: ClauseIndex) -> GuardrailResult:
    """Validate a produced record and record verdicts in ``rec.safety_flags``."""
    flags: list[str] = []
    text = " ".join(
        t for t in (rec.response, rec.answer, rec.clarifying_question,
                    rec.recommendation, rec.uncertainty_reason) if t
    )

    if not rec.response or len(rec.response.strip()) < 20:
        flags.append("empty_or_too_short")
    if rec.response and _REFUSAL.search(rec.response):
        flags.append("refusal")
    if rec.response and len(rec.response) > 4000:
        flags.append("too_long")

    valid_numbers = {c.number for c in index.clauses.values()}
    for num in _CLAUSE_REF.findall(text):
        if num not in valid_numbers:
            flags.append(f"hallucinated_clause:{num}")

    if rec.category.value == "B":
        if "?" not in ((rec.clarifying_question or "") + " " + (rec.response or "")):
            flags.append("b_missing_question")
    if rec.category.value == "C":
        body = " ".join(t for t in (rec.response, rec.uncertainty_reason, rec.recommendation) if t)
        if not _UNCERTAINTY.search(body):
            flags.append("c_missing_uncertainty")
        if _FALSE_CERTAINTY.search(body):
            flags.append("c_false_certainty")

    pii = _contains_pii(text)
    tox = _contains_profanity(text)
    if pii:
        flags.append("pii_in_output")
    if tox:
        flags.append("toxicity_in_output")

    rec.safety_flags.pii_present = pii
    rec.safety_flags.toxicity = tox
    rec.safety_flags.notes = "; ".join(flags) or None
    return GuardrailResult(ok=not flags, flags=flags)


# Eval guardrail
def check_eval_result(rec: QARecord, expected_keys: set[str] | None = None) -> list[str]:
    """Catch judge-side failures: exceptions, missing dimensions, suspicious all-zero."""
    flags: list[str] = []
    ev = rec.eval
    if ev is None or not ev.scores:
        return ["no_scores"]
    if "ERROR" in (ev.judge_rationale or ""):
        flags.append("judge_error")
    if expected_keys:
        missing = sorted(expected_keys - set(ev.scores))
        if missing:
            flags.append("missing_dimensions:" + ",".join(missing))
    if all(v == 0.0 for v in ev.scores.values()):
        flags.append("all_zero_scores")
    return flags
