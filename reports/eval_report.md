# Project 06 — Offline Evaluation Report

Pipeline: schema-locked extraction (Pydantic, `extra="forbid"`), bounded
validation→repair loop, per-field confidence scoring with review routing, and
hallucination detection. Provider: `DeterministicProvider` (offline; zero API
keys, zero network — every number below is exactly reproducible).

**Reproduce:**

```bash
PYTHONPATH= .venv/bin/python main.py eval   # exit 1 on gate failure
PYTHONPATH= .venv/bin/python -m pytest tests
```

## Results (this run)

| metric | gate | value |
|---|---|---|
| field_accuracy | ≥ 0.85 | **1.0000** (17/17 fields) |
| schema_valid_first_try | ≥ 0.99 | **1.0000** (4/4 docs) |
| schema_valid_after_repair | = 1.00 | **1.0000** (1/1 scripted repair) |
| hallucination_rate | ≤ 0.06 | **0.0000** |
| review_precision | ≥ 0.50 | **1.0000** |
| documents_evaluated | — | 4 |
| docs_per_minute (measured, 4-doc corpus) | — | ~33k–44k (run-dependent) |
| review_fields_evaluated | — | 1 |
| cost_per_doc_usd (assumption) | — | $0.0005 |

Per-case field matching: clean_invoice 5/5 · eur_invoice 5/5 · receipt 4/4 ·
missing_total 3/3. One scripted schema+math repair case (bad `currency` enum +
broken `total_amount`) recovered in exactly 2 attempts.

## Metric definitions

- **Field accuracy** — exact match per hand-labeled field; floats pass within
  ±$0.01 ("close enough" for money). `line_items` is labeled as a count.
- **Schema-valid (first try)** — fraction of corpus docs whose first provider
  response passed Pydantic validation (shape/enums/types).
- **Schema-valid (after repair)** — scripted broken→fixed responses must be
  accepted after exactly one bounded repair retry.
- **Hallucination rate** — flagged fields whose values are absent from the
  source text, divided by all labeled fields. The detector is deliberately
  conservative (see Limitations): only text-like fields are checked literally.
- **Review precision** — of the fields the router flagged for human review,
  the fraction that are genuinely wrong per the hand labels (or absent from
  the source). This is the calibration metric: it answers "are low-confidence
  flags actually the wrong ones?"

## Confidence model (offline provider)

Declared per-field confidence: 0.99 directly-parsed money values;
evidence-based 0.95/0.45 for zeros (label present vs absent in source);
0.86 text fields; 0.80/0.97 line-item description/amount; 0.95 default for
undeclared fields. Review threshold: 0.7. In this corpus only the invented
`total_amount` on the missing-total case falls below threshold — and it is
also the hallucinated field — so review precision is 1.0 with 1 flag raised.

## Accuracy ladder (brief target vs this build)

| Config | field accuracy | schema-valid (1st try) | hallucination rate |
|---|---|---|---|
| naive prompt+parse (brief, reference) | 0.71 | 0.63 | 0.14 |
| + Instructor schema (brief, reference) | 0.86 | 0.99 | 0.06 |
| **+ validate/repair/confidence (this build, offline corpus)** | **1.00** | **1.00** | **0.00** |

The offline numbers are a *lower bound sanity check*, not a claim of parity
with a real LLM: the deterministic provider parses this corpus correctly by
construction. The ladder's purpose here is to prove the harness measures the
right things — gates, repair accounting, hallucination detection, and
calibration all fire and are unit-tested (`tests/test_evals.py`).

## Limitations (honest accounting)

1. **Hallucination metric skips numeric/date fields.** "28.05" vs "28.05 USD"
   vs "$28.05" formatting makes literal substring checks noisy, so
   `detect_hallucinations` only checks text-like fields (vendor, invoice
   number, descriptions, currency). A numeric hallucination on this corpus is
   caught indirectly: wrong numbers fail business-rule math or hand labels and
   route to review. On a real LLM provider, add numeric normalization
   (strip currency symbols, parse to Decimal) before extending literal checks.
2. **Corpus is 4 documents + 1 scripted repair.** The brief targets 50–200
   hand-labeled docs. The harness (`EvalSuite`) is corpus-size agnostic —
   scaling is a data-labeling task, not a code task.
3. **Throughput is measured; cost is an assumption.** `docs_per_minute` is timed
   over the 4-document corpus above, so it is a sanity measurement on a corpus
   far too small to extrapolate — read it as "the harness is not slow", not as a
   capacity claim. (It previously returned `n_cases * 60`, i.e. 240 for any
   machine; that figure was arithmetic, not a measurement.) `cost_per_doc_usd`
   remains an explicit $0.0005/doc *assumption* for the offline provider; real
   provider latency and pricing will differ (the OpenAI adapter reports its own
   cost accounting via `src/batch.py`).
4. **The OpenAI adapter is contract-tested but not live-fired.** The request
   payload matches the real Chat Completions contract (`tools` +
   `tool_choice`, forced function call) and response parsing is exercised with
   a scripted tool-call response; no live API call was made in this offline
   run (no API key required or used by the evals).

## Verification evidence (this run)

- `pytest tests` → **72 passed** (includes RED→GREEN regression test proving
  review-precision is computed per case, not globally).
- `python -m compileall src evals main.py tests scripts` → OK.
- Live HTTP probe (uvicorn 127.0.0.1:8097, real httpx client):
  `/health` 200 · `/extract` 200 with correct vendor/total · invalid
  `document_type` → 422 at the boundary · `/schema/invoice` 200.
- Eval gate: `main.py eval` exit code 0, GATES: PASS.
