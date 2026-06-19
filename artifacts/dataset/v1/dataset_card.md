# Dataset Card — Razorpay ToS Q&A (v1)

- **Records:** 54
- **Source:** `Razorpay Terms & Conditions.pdf` (sha256 `3a57aa417fab…`)
- **Reviewed fraction:** 14/54 (26%) carry a review_status above `unreviewed`

## Category distribution
`A`: 16, `B`: 20, `C`: 18

## Topic coverage
`chargebacks_fraud`: 5, `cross_border`: 2, `data_protection`: 6, `fees`: 10, `gaming`: 1, `kyc_onboarding`: 4, `liability`: 8, `prohibited_businesses`: 5, `refunds`: 2, `settlements`: 6, `suspension_termination`: 4, `taxes_gst`: 1

## Question-type distribution
`adversarial_false_premise`: 4, `conditional`: 20, `factual`: 28, `multi_hop`: 2

## Difficulty distribution
`easy`: 5, `hard`: 22, `medium`: 27

## Splits (stratified by category × question_type)
train: 38, val: 7, test: 9

## Known limitations
- Clause numbering in this PDF differs from Razorpay's live site; citations are grounded to the PDF's actual clauses (verified by substring check).
- Questions/answers are LLM-generated (DeepEval Synthesizer + enrichment); re-running with a different provider/model/seed yields different phrasings (see README determinism).
- Short verbatim quotes are used for grounding only (Razorpay's ToS is copyrighted).
