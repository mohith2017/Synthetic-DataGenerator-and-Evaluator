"""Evaluation and guardrails tests — all LLM-free."""

from deepeval.models import DeepEvalBaseLLM

from razorpay_qa import evaluation as EV
from razorpay_qa.utils import guardrails as G
from razorpay_qa.utils.schema import (
    AmbiguityType,
    Answerability,
    Category,
    Citation,
    DeterministicChecks,
    Escalation,
    EvalBlock,
    ExpectedBehavior,
    GenerationMethod,
    Origin,
    Provenance,
    QARecord,
)


def _prov():
    return Provenance(model="m", provider="anthropic", prompt_version="v", seed=1,
                      generated_at="2026-01-01T00:00:00+00:00")


def _a(**over):
    data = dict(
        id="a1", question="Do we owe the fee after a refund?",
        response="Yes, fees remain payable per the FEES clause.", answer="Yes, fees remain payable.",
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
    return QARecord(**data)


def _c(**over):
    data = dict(
        id="c1", question="How long can funds be held?",
        response="The ToS is silent on this; we recommend confirming with Razorpay.",
        answer=None, category=Category.C, expected_behavior=ExpectedBehavior.flag_ambiguity,
        answerability=Answerability.indeterminate,
        uncertainty_reason="The document does not bound the hold duration.",
        recommendation="Confirm timelines directly with Razorpay.",
        ambiguity_type=AmbiguityType.silent, assumptions_not_made=["We did not assume a limit."],
        escalation=Escalation(required=True, reason="ToS silent on duration."),
        citations=[Citation(part="A", section="16", clause_id="PartA/16.1",
                            clause_title="SUSPENSION (16.1)", quoted_text="suspend")],
        persona="cto", topic="suspension_termination", difficulty="hard", question_type="factual",
        origin=Origin.synthesized, generation_method=GenerationMethod.synthesizer_direct,
        source_doc="d.pdf", source_hash="h", source_parsed_at="2026-01-01T00:00:00+00:00",
        provenance=_prov(),
    )
    data.update(over)
    return QARecord(**data)


class DummyModel(DeepEvalBaseLLM):
    def load_model(self): return self
    def generate(self, *a, **k): return ("{}", 0)
    async def a_generate(self, *a, **k): return ("{}", 0)
    def get_model_name(self): return "dummy"


def test_metric_dimensions_per_category():
    model = DummyModel()
    common = {"citation_accuracy", "faithfulness", "question_realism"}
    a = {EV.metric_key(m.name) for m in EV.build_geval_metrics(model, "A")}
    b = {EV.metric_key(m.name) for m in EV.build_geval_metrics(model, "B")}
    c = {EV.metric_key(m.name) for m in EV.build_geval_metrics(model, "C")}
    assert common <= a and "category_correctness" in a
    assert common <= b and "clarification_specificity" in b
    assert common <= c and "ambiguity_honesty" in c


def test_failure_stage_classification():
    rec = _a()
    bad = DeterministicChecks(citation_exists=False, quote_matches=False)
    assert EV._failure_stage(rec, bad, {"citation_accuracy": 0.9}, 0.6) == "citation_mapping"
    good = DeterministicChecks(citation_exists=True, quote_matches=True)
    assert EV._failure_stage(rec, good, {"faithfulness": 0.2}, 0.6) == "generation"
    assert EV._failure_stage(rec, good, {"clarification_specificity": 0.2}, 0.6) == "enrichment"
    assert EV._failure_stage(rec, good, {"faithfulness": 0.9}, 0.6) is None


def test_render_summary_smoke():
    rec = _a()
    rec.eval = EvalBlock(
        scores={"citation_accuracy": 0.8, "faithfulness": 0.9}, composite_score=0.85,
        deterministic_checks=DeterministicChecks(citation_exists=True, quote_matches=True),
    )
    md = EV._render([rec], "anthropic", 0.6)
    assert "Evaluation Summary" in md and "3 worst examples" in md and "a1" in md


def test_input_guardrail_passes_real_clause(index):
    res = G.check_generation_input({"cids": [["PartA/3.1", "fees", "primary", "x"]]}, index)
    assert res.ok and not res.flags


def test_input_guardrail_flags_missing_clause(index):
    res = G.check_generation_input({"cids": [["PartA/99.9", "x", "primary", "y"]]}, index)
    assert not res.ok and any(f.startswith("missing_clause") for f in res.flags)


def test_output_guardrail_clean_a(index):
    assert G.check_generation_output(_a(), index).ok


def test_output_guardrail_detects_pii(index):
    rec = _a(response="Yes, email us at ops@example.com for the refund.")
    res = G.check_generation_output(rec, index)
    assert "pii_in_output" in res.flags and res.hard_flags


def test_output_guardrail_detects_hallucinated_clause(index):
    rec = _a(response="Yes — see Clause 99.9 which is very clear on this.")
    res = G.check_generation_output(rec, index)
    assert any(f == "hallucinated_clause:99.9" for f in res.flags)


def test_output_guardrail_c_false_certainty(index):
    rec = _c(response="You are definitely safe; Razorpay absolutely cannot hold funds at all.")
    res = G.check_generation_output(rec, index)
    assert "c_false_certainty" in res.flags and res.hard_flags


def test_eval_guardrail_detects_judge_error():
    rec = _a()
    rec.eval = EvalBlock(scores={"faithfulness": 0.0}, judge_rationale="faithfulness: ERROR boom")
    assert "judge_error" in G.check_eval_result(rec, {"faithfulness"})


def test_eval_guardrail_detects_missing_dimension():
    rec = _a()
    rec.eval = EvalBlock(scores={"faithfulness": 0.8})
    assert any(f.startswith("missing_dimensions") for f in G.check_eval_result(rec, {"faithfulness", "citation_accuracy"}))
