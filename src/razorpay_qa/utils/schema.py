"""Pydantic schema for a single Q&A record and its sub-objects.

Design principle (see plan): every field has a concrete consumer — the model being
trained, the human compliance reviewer, the eval harness, or the pipeline itself.
Controlled vocabularies are enums so the dataset can be balanced and validated.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"


# Controlled vocabularies
class Category(str, Enum):
    A = "A"  # clear answer
    B = "B"  # clarification required
    C = "C"  # genuine ambiguity


class ExpectedBehavior(str, Enum):
    answer_directly = "answer_directly"
    ask_clarifying_question = "ask_clarifying_question"
    flag_ambiguity = "flag_ambiguity"


class Answerability(str, Enum):
    answerable = "answerable"
    needs_clarification = "needs_clarification"
    indeterminate = "indeterminate"


class AmbiguityType(str, Enum):
    silent = "silent"
    vague = "vague"
    external_cross_reference = "external_cross_reference"


class QuestionType(str, Enum):
    factual = "factual"
    multi_hop = "multi_hop"
    comparative = "comparative"
    conditional = "conditional"
    adversarial_false_premise = "adversarial_false_premise"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class CitationRole(str, Enum):
    primary = "primary"
    supporting = "supporting"


class ReviewStatus(str, Enum):
    unreviewed = "unreviewed"
    spot_checked = "spot_checked"
    human_verified = "human_verified"


class Origin(str, Enum):
    seed_anchor = "seed_anchor"
    synthesized = "synthesized"


class GenerationMethod(str, Enum):
    synthesizer_direct = "synthesizer_direct"
    synthesizer_plus_enrichment = "synthesizer_plus_enrichment"


class Citation(BaseModel):
    part: str
    section: str
    clause_id: str
    clause_title: str
    quoted_text: str
    role: CitationRole = CitationRole.primary
    relevance: str = ""
    source_locator: Optional[str] = None


class ExternalReference(BaseModel):
    name: str
    identifier: Optional[str] = None
    relationship: str = ""


class MissingContext(BaseModel):
    field: str
    why_needed: str


class DecisionFactor(BaseModel):
    factor: str
    how_it_changes_the_answer: str


class Escalation(BaseModel):
    required: bool = False
    reason: Optional[str] = None


class SafetyFlags(BaseModel):
    pii_present: bool = False
    toxicity: bool = False
    off_topic: bool = False
    notes: Optional[str] = None


class DeterministicChecks(BaseModel):
    citation_exists: Optional[bool] = None
    quote_matches: Optional[bool] = None


class EvalBlock(BaseModel):
    scores: dict[str, float] = Field(default_factory=dict)
    judge_rationale: Optional[str] = None
    composite_score: Optional[float] = None
    deterministic_checks: DeterministicChecks = Field(default_factory=DeterministicChecks)
    failure_stage: Optional[str] = None
    guardrail_flags: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    model: str
    provider: str
    prompt_version: str
    seed: int
    temperature: float = 0.0
    generated_at: str
    evolutions: list[str] = Field(default_factory=list)
    context_quality: Optional[float] = None
    synthetic_input_quality: Optional[float] = None


class QARecord(BaseModel):
    # Identity / versioning
    id: str
    schema_version: str = SCHEMA_VERSION
    dataset_version: Optional[str] = None

    # Behavior contract / training target
    question: str
    response: str
    answer: Optional[str] = None
    negative_example: Optional[str] = None
    category: Category
    expected_behavior: ExpectedBehavior
    answerability: Answerability

    # B-only
    clarifying_question: Optional[str] = None
    missing_context: list[MissingContext] = Field(default_factory=list)
    decision_factors: list[DecisionFactor] = Field(default_factory=list)

    # C-only
    known_facts: list[str] = Field(default_factory=list)
    uncertainty_reason: Optional[str] = None
    recommendation: Optional[str] = None
    ambiguity_type: Optional[AmbiguityType] = None

    # B/C
    assumptions_not_made: list[str] = Field(default_factory=list)

    # Grounding / reviewer trust
    citations: list[Citation] = Field(default_factory=list)
    external_references: list[ExternalReference] = Field(default_factory=list)
    escalation: Escalation = Field(default_factory=Escalation)
    review_status: ReviewStatus = ReviewStatus.unreviewed
    safety_flags: SafetyFlags = Field(default_factory=SafetyFlags)

    # Curation / coverage
    persona: str
    topic: str
    difficulty: Difficulty
    question_type: QuestionType

    # Origin / method
    origin: Origin
    generation_method: GenerationMethod

    # Source provenance / legal context
    source_doc: str
    source_hash: str
    source_parsed_at: str
    jurisdiction: str = "India"
    tos_effective_date: Optional[str] = None

    # Generation provenance + eval write-back
    provenance: Provenance
    eval: Optional[EvalBlock] = None

    @model_validator(mode="after")
    def _check_contract(self) -> "QARecord":
        cat = self.category

        expected_eb = {
            Category.A: ExpectedBehavior.answer_directly,
            Category.B: ExpectedBehavior.ask_clarifying_question,
            Category.C: ExpectedBehavior.flag_ambiguity,
        }[cat]
        if self.expected_behavior != expected_eb:
            raise ValueError(f"{self.id}: expected_behavior must be {expected_eb} for category {cat}")

        expected_ans = {
            Category.A: Answerability.answerable,
            Category.B: Answerability.needs_clarification,
            Category.C: Answerability.indeterminate,
        }[cat]
        if self.answerability != expected_ans:
            raise ValueError(f"{self.id}: answerability must be {expected_ans} for category {cat}")

        if cat == Category.A:
            if not self.citations:
                raise ValueError(f"{self.id}: Category A requires >=1 citation")
            if not self.answer:
                raise ValueError(f"{self.id}: Category A requires an answer")

        if cat == Category.B:
            if not self.clarifying_question:
                raise ValueError(f"{self.id}: Category B requires clarifying_question")
            if not self.missing_context:
                raise ValueError(f"{self.id}: Category B requires non-empty missing_context")
            if not self.decision_factors:
                raise ValueError(f"{self.id}: Category B requires non-empty decision_factors")

        if cat == Category.C:
            if not self.uncertainty_reason:
                raise ValueError(f"{self.id}: Category C requires uncertainty_reason")
            if not self.recommendation:
                raise ValueError(f"{self.id}: Category C requires recommendation")
            if self.ambiguity_type is None:
                raise ValueError(f"{self.id}: Category C requires ambiguity_type")
            if not self.escalation.required:
                raise ValueError(f"{self.id}: Category C requires escalation.required == true")

        if cat in (Category.B, Category.C) and not self.assumptions_not_made:
            raise ValueError(f"{self.id}: Category {cat} requires non-empty assumptions_not_made")

        if (
            self.ambiguity_type == AmbiguityType.external_cross_reference
            and not self.external_references
        ):
            raise ValueError(
                f"{self.id}: ambiguity_type=external_cross_reference requires external_references"
            )

        return self
