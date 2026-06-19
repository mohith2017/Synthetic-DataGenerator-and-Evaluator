"""Core pipeline tests — clause index, schema, enrich, seeds, and self-correction.

All LLM-free: fake models return canned responses so no API key is required.
"""

import json

import pytest
from pydantic import ValidationError

from razorpay_qa import generation as E
from razorpay_qa import generation as G
from razorpay_qa.generation import _citations
from razorpay_qa.utils.schema import (
    AmbiguityType,
    Answerability,
    Category,
    Citation,
    ExpectedBehavior,
    GenerationMethod,
    Origin,
    Provenance,
    QARecord,
)
from razorpay_qa.utils.seeds import SEEDS


def test_index_has_core_clauses(index):
    for cid in ["PartA/3.4", "PartA/7.2", "PartA/12.3", "PartA/16.1", "PartA/14.10"]:
        assert index.exists(cid), f"missing {cid}"


def test_no_spurious_inline_number_clause(index):
    assert not index.exists("PartA/19.40")


def test_quote_contains_is_verbatim(index):
    clause = index.get("PartA/3.4")
    assert index.quote_contains("PartA/3.4", clause.text[:30])
    assert not index.quote_contains("PartA/3.4", "this phrase is definitely not present xyz")


def test_section_titles_resolved(index):
    assert index.get("PartA/12.3").section_title == "DATA PROTECTION"
    assert index.get("PartA/7.2").section_title == "LIMITATION OF LIABILITY"


def _prov():
    return Provenance(
        model="m", provider="anthropic", prompt_version="v", seed=1,
        generated_at="2026-01-01T00:00:00+00:00",
    )


def _base(**over):
    data = dict(
        id="t1", question="What is the fee?", response="It is X.", answer="It is X.",
        category=Category.A, expected_behavior=ExpectedBehavior.answer_directly,
        answerability=Answerability.answerable,
        citations=[Citation(part="A", section="3", clause_id="PartA/3.1",
                            clause_title="FEES (3.1)", quoted_text="fees")],
        persona="cto", topic="fees", difficulty="easy", question_type="factual",
        origin=Origin.synthesized, generation_method=GenerationMethod.synthesizer_direct,
        source_doc="d.pdf", source_hash="h", source_parsed_at="2026-01-01T00:00:00+00:00",
        provenance=_prov(),
    )
    data.update(over)
    return data


def test_valid_a_record():
    assert QARecord(**_base()).category == Category.A


def test_category_behavior_mismatch_rejected():
    with pytest.raises(ValidationError):
        QARecord(**_base(expected_behavior=ExpectedBehavior.flag_ambiguity))


def test_b_requires_clarifying_fields():
    with pytest.raises(ValidationError):
        QARecord(**_base(
            category=Category.B,
            expected_behavior=ExpectedBehavior.ask_clarifying_question,
            answerability=Answerability.needs_clarification,
        ))


def test_c_requires_escalation_and_uncertainty():
    with pytest.raises(ValidationError):
        QARecord(**_base(
            category=Category.C, answer=None,
            expected_behavior=ExpectedBehavior.flag_ambiguity,
            answerability=Answerability.indeterminate,
        ))


def test_seeds_meet_minimum_per_category(settings):
    from collections import Counter
    counts = Counter(s["category"] for s in SEEDS)
    for cat in ("A", "B", "C"):
        assert counts[cat] >= settings.min_per_category, f"{cat}={counts[cat]}"


def test_all_seed_clauses_exist(index):
    for s in SEEDS:
        for cid in s["cids"]:
            assert index.exists(cid[0]), f"{s['key']} -> missing {cid[0]}"


def test_built_citations_are_verbatim_substrings(index):
    for s in SEEDS:
        citations = _citations(index, s["cids"])
        assert citations, s["key"]
        for c in citations:
            assert index.quote_contains(c.clause_id, c.quoted_text), f"{s['key']}:{c.clause_id}"


def test_c_seeds_have_valid_classification():
    for s in SEEDS:
        if s["category"] != "C":
            continue
        assert s.get("ambiguity_type") in {a.value for a in AmbiguityType}, s["key"]
        if s["ambiguity_type"] == AmbiguityType.external_cross_reference.value:
            assert s.get("external_references"), f"{s['key']} needs external_references"


class FakeModel:
    def __init__(self, payload: dict):
        self._payload = payload

    def generate(self, prompt, schema=None):
        return json.dumps(self._payload)


def test_decompose_b_returns_structured_fields():
    model = FakeModel({
        "clarifying_question": "At what stage is the dispute?",
        "missing_context": [{"field": "dispute_stage", "why_needed": "changes whether suspension applies"}],
        "decision_factors": [{"factor": "dispute is filed", "how_it_changes_the_answer": "suspension may apply"}],
        "assumptions_not_made": ["did not assume the dispute stage"],
    })
    out = E.decompose_b(model, "q", "It depends on the dispute stage…", "clause text")
    assert isinstance(out, E.BDecomposition)
    assert out.clarifying_question
    assert out.missing_context[0].field == "dispute_stage"


def test_decompose_c_returns_structured_fields():
    model = FakeModel({
        "known_facts": ["Clause 16.1 allows immediate suspension."],
        "uncertainty_reason": "no maximum hold duration is stated",
        "recommendation": "confirm timelines with Razorpay",
        "assumptions_not_made": ["did not assume a hold duration"],
    })
    out = E.decompose_c(model, "q", "The ToS is silent on duration…", "clause text")
    assert isinstance(out, E.CDecomposition)
    assert out.known_facts and out.uncertainty_reason and out.recommendation


def test_generate_fallback_and_negative_example():
    model = FakeModel({"question": "Do we still owe the fee after a refund?", "response": "Yes, per PartA/3.4."})
    qr = E.generate_question_and_response(model, "A", "clause text", exemplar="example")
    assert qr.question and qr.response

    model2 = FakeModel({"bad_answer": "No, refunded transactions are always free of fees."})
    neg = E.generate_negative_example(model2, "Do we owe fees after a refund?", "A", "Yes, per PartA/3.4.")
    assert isinstance(neg, str) and neg


class SeqFakeModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, prompt, schema=None):
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return json.dumps({"response": resp})


def _checker(resp: str) -> list[str]:
    return [] if "grounded" in resp.lower() else ["empty_or_too_short"]


def test_self_correct_recovers():
    model = SeqFakeModel(["A grounded, correct, sufficiently long answer."])
    final, remaining = G.self_correct(model, "A", "q?", "no", "Clause.", _checker, max_attempts=2)
    assert not remaining and "grounded" in final.lower() and model.calls == 1


def test_self_correct_bounded():
    model = SeqFakeModel(["still bad", "still bad", "still bad"])
    _, remaining = G.self_correct(model, "A", "q?", "no", "Clause.", _checker, max_attempts=2)
    assert remaining and model.calls == 2


def test_self_correct_noop_when_clean():
    model = SeqFakeModel(["unused"])
    final, remaining = G.self_correct(model, "A", "q?", "Already grounded.", "Clause.", _checker, max_attempts=2)
    assert not remaining and model.calls == 0
