# Evaluation Summary — Razorpay ToS Q&A Dataset

- **Judge:** `anthropic` (DeepEval G-Eval)
- **Records evaluated:** 54
- **Overall mean composite:** 0.606
- **Pass rate (composite ≥ 0.6):** 27/54 (50%)
- **Deterministic citation checks passed:** 54/54 (100%)

## Scoring dimensions (0–1)
- **citation_accuracy** — cited clause actually supports the answer (+ substring check).
- **faithfulness** — no claims beyond the cited ToS context.
- **category_correctness** — matches the A/B/C behaviour contract (A: deterministic DAG decision tree — direct answer AND cites a clause).
- **question_realism** — a fintech engineer/CTO would plausibly ask it.
- **clarification_specificity** — (B) targeted, not vague.
- **ambiguity_honesty** — (C) flags uncertainty without false certainty.

## Per-dimension means
`ambiguity_honesty`: 0.956, `category_correctness`: 0.062, `citation_accuracy`: 0.585, `clarification_specificity`: 0.895, `faithfulness`: 0.496, `question_realism`: 0.674

## Stratified composite
- **By category:** `A`: 0.523, `B`: 0.621, `C`: 0.662
- **By topic:** `chargebacks_fraud`: 0.545, `cross_border`: 0.8, `data_protection`: 0.662, `fees`: 0.565, `gaming`: 0.7, `kyc_onboarding`: 0.675, `liability`: 0.644, `prohibited_businesses`: 0.53, `refunds`: 0.725, `settlements`: 0.592, `suspension_termination`: 0.5, `taxes_gst`: 0.575
- **By difficulty:** `easy`: 0.485, `hard`: 0.64, `medium`: 0.601
- **By question_type:** `adversarial_false_premise`: 0.619, `conditional`: 0.626, `factual`: 0.596, `multi_hop`: 0.512

## Judge calibration (lightweight)
- Mean composite on human-verified seed anchors: 0.555 (n=5)
- Mean composite on the rest: 0.611 (n=49)

## Guardrails (judge-side)
- Records with eval guardrail flags: 0/54 (none)

## 3 worst examples

### 1. `rzp-A08` (category A, composite 0.375)
- **Question:** Do we need to get approval from Razorpay before integrating a third-party payment orchestrator with our account, and what happens if we don't?
- **Scores:** `citation_accuracy`: 0.5, `faithfulness`: 0.2, `question_realism`: 0.8, `category_correctness`: 0.0
- **Analysis (~50 words):** Weakest dim `category_correctness`=0.0; likely **generation** stage. Fix: tighten the faithfulness rubric so key facts must appear in the cited clause.

### 2. `rzp-A05` (category A, composite 0.4)
- **Question:** What are our notification and cooperation obligations if we experience a customer data security incident in our merchant system?
- **Scores:** `citation_accuracy`: 0.6, `faithfulness`: 0.3, `question_realism`: 0.7, `category_correctness`: 0.0
- **Analysis (~50 words):** Weakest dim `category_correctness`=0.0; likely **generation** stage. Fix: tighten the faithfulness rubric so key facts must appear in the cited clause.

### 3. `rzp-A07` (category A, composite 0.425)
- **Question:** Are we allowed to add convenience fees to Razorpay payments, and what's our liability if customers dispute these charges?
- **Scores:** `citation_accuracy`: 0.5, `faithfulness`: 0.3, `question_realism`: 0.9, `category_correctness`: 0.0
- **Analysis (~50 words):** Weakest dim `category_correctness`=0.0; likely **generation** stage. Fix: tighten the faithfulness rubric so key facts must appear in the cited clause.
