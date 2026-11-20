# Project 06 — Structured Extraction Pipeline

Schema-locked invoice/receipt extraction with a bounded validation → repair
loop, per-field confidence scoring with human-review routing, hallucination
detection, batch throughput/cost accounting, and a deterministic offline eval
suite. Works fully offline (no API keys); a real OpenAI structured-outputs
adapter activates when `OPENAI_API_KEY` is set.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# extract a document
python main.py extract --file samples/invoice_acme.txt --type invoice

# batch a folder (throughput + cost stats)
python main.py batch --folder samples --type invoice

# run the offline eval suite (exit 1 on gate failure)
python main.py eval

# serve the HTTP API (port 8097, /health /extract /schema/{type})
python main.py serve

# dump the Pydantic JSON Schemas
python main.py config
```

## Architecture

```
document (.txt/.md/.pdf)
  │  src/parsing.py        normalize; PDFs via pypdf
  ▼
provider (DeterministicProvider offline | OpenAIProvider structured outputs)
  │  src/providers.py      returns raw JSON + declared field confidence
  ▼
schema lock (Pydantic, extra="forbid", closed enums)
  │  src/pipeline.py       bounded repair loop (default 2 retries)
  ▼
business validation  →  confidence scoring  →  review queue
  │  src/validation.py     src/confidence.py     needs_review[]
  ▼
hallucination detection (field values absent from source text)
  │  src/service.py        ExtractionService orchestrates everything
  ▼
transports: FastAPI (src/api.py) · CLI (main.py) · batch (src/batch.py)
```

**Repair contract.** Schema-invalid responses are retried with an error-annotated
repair prompt. A response that is schema-valid but fails business rules on the
final attempt is **kept** and routed to review — the pipeline surfaces
uncertainty instead of silently dropping data. Failures that produce no usable
document are returned with structured, repairable error messages.

**Review queue.** A field lands in `needs_review` when its confidence is below
the threshold (default 0.7), when a business-rule issue survived repair, or when
the value appears to be hallucinated (absent from the source text).

## Confidence model

The offline provider declares per-field confidence: 0.99 for directly parsed
money values, evidence-based (0.95/0.45) for zeros depending on whether the
label exists in the source text, 0.86 for text fields, 0.80/0.97 for line-item
description/amount. `score_document` merges declared values with a 0.95 default
for undeclared fields and computes the overall mean. Calibration quality is
measured as *review precision* in the eval suite: the fraction of flagged fields
that are actually wrong per the hand labels.

## Evaluation

`python main.py eval` runs a hand-labeled corpus (clean invoice, EUR invoice,
receipt, missing-total invoice, plus a scripted schema+math repair case) and
gates on:

| metric | gate | value |
|---|---|---|
| field_accuracy | ≥ 0.85 | 1.00 |
| schema_valid_first_try | ≥ 0.99 | 1.00 |
| schema_valid_after_repair | = 1.00 | 1.00 |
| hallucination_rate | ≤ 0.06 | 0.00 |
| review_precision | ≥ 0.50 | 1.00 |

Full metric definitions and the accuracy ladder are in `reports/eval_report.md`.

## Tests

```bash
pytest tests/          # 72 tests
python -m compileall src evals main.py tests
```

Test map: `test_models.py` schema locks · `test_validation.py` business rules ·
`test_pipeline.py` repair loop · `test_confidence.py` scoring/routing ·
`test_parsing.py` text+real PDF (hand-built minimal PDF, no fixtures) ·
`test_providers.py` offline provider + OpenAI adapter · `test_service.py`
orchestration + hallucinations · `test_batch.py` throughput/cost ·
`test_api.py` HTTP contract · `test_evals.py` eval harness.

## Providers

- **DeterministicProvider** (default): rule-based invoice/receipt extractor,
  zero network, conservative confidence — powers tests, offline evals, demo.
- **OpenAIProvider**: real `chat/completions` tool-calling adapter; the Pydantic
  JSON Schema is passed as the function schema with `tool_choice` forced.
  Requires `OPENAI_API_KEY` (optional `OPENAI_MODEL`, default `gpt-4o-mini`).
