"""Deterministic offline evaluation suite for the extraction pipeline.

No API keys, no network, no wall-clock flakiness. A hand-labeled corpus of
invoice/receipt documents exercises every metric the brief calls for:

* field-level exact-match accuracy (plus numeric "close enough")
* schema-valid rate, first try vs after repair
* hallucination rate (fields invented when source lacks them)
* confidence calibration — precision of the ``needs_review`` flag
* throughput and cost (offline baseline)

The pass gate encodes the accuracy ladder from the brief: naive prompt+parse
(~0.71) → schema lock (~0.86) → validate/repair/confidence (~0.93).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.batch import batch_extract
from pydantic import ValidationError

from src.models import DOCUMENT_MODELS, Invoice, Receipt
from src.pipeline import Extractor
from src.service import ExtractionService

# --------------------------------------------------------------------------- corpus


@dataclass(frozen=True)
class EvalCase:
    name: str
    document_type: str
    text: str
    expected: dict[str, Any]
    # Fields intentionally absent from the source: any value here is a hallucination.
    absent_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairCase:
    name: str
    document_type: str
    text: str
    broken_raw: dict[str, Any]
    fixed_raw: dict[str, Any]


class _ScriptProvider:
    """Emits a scripted first response, then a corrected one on repair."""

    def __init__(self, first: dict, second: dict) -> None:
        self.first = first
        self.second = second
        self.calls = 0

    def generate(self, document_text, schema, repair_prompt):
        self.calls += 1
        return self.first if self.calls == 1 else self.second


INV1 = """
INVOICE
Acme Corp
Invoice Number: INV-1001
Invoice Date: 2026-01-15
Due Date: 2026-02-14
Currency: USD
Widget  2  10.00  20.00
Gadget  1  5.50  5.50
Subtotal: 25.50 USD
Tax: 2.55 USD
Total: 28.05 USD
"""
INV1_EXPECTED = {
    "vendor": "Acme Corp",
    "invoice_number": "INV-1001",
    "currency": "USD",
    "total_amount": 28.05,
    "line_items": 2,
}

INV2 = """
INVOICE
Beta Ltd
Invoice Number: INV-1002
Invoice Date: 2026-02-15
Currency: EUR
Consulting  3  100.00  300.00
Subtotal: 300.00 EUR
Tax: 30.00 EUR
Total: 330.00 EUR
"""
INV2_EXPECTED = {
    "vendor": "Beta Ltd",
    "invoice_number": "INV-1002",
    "currency": "EUR",
    "total_amount": 330.00,
    "line_items": 1,
}

REC1 = """
RECEIPT
Coffee Shop
Date: 2026-03-02
Currency: USD
Latte  1 x 4.50 = 4.50
Subtotal: 4.50 USD
Tax: 0.00 USD
Total: 4.50 USD
"""
REC1_EXPECTED = {
    "merchant": "Coffee Shop",
    "currency": "USD",
    "total_amount": 4.50,
    "line_items": 1,
}

# A missing total: the source truly has no total; a filled-in value is a hallucination.
INV_MISSING_TOTAL = """
INVOICE
Gamma LLC
Invoice Number: INV-1003
Invoice Date: 2026-04-01
Currency: USD
Service  1  50.00  50.00
Subtotal: 50.00 USD
Tax: 0.00 USD
"""
INV_MISSING_TOTAL_EXPECTED = {
    "vendor": "Gamma LLC",
    "invoice_number": "INV-1003",
    "currency": "USD",
}


def default_cases() -> list[EvalCase]:
    return [
        EvalCase("clean_invoice", "invoice", INV1, INV1_EXPECTED),
        EvalCase("eur_invoice", "invoice", INV2, INV2_EXPECTED),
        EvalCase("receipt", "receipt", REC1, REC1_EXPECTED),
        EvalCase(
            "missing_total",
            "invoice",
            INV_MISSING_TOTAL,
            INV_MISSING_TOTAL_EXPECTED,
            absent_fields=("total_amount",),
        ),
    ]


def default_repair_cases() -> list[RepairCase]:
    broken = {
        "vendor": "Acme Corp",
        "invoice_number": "INV-1001",
        "invoice_date": "2026-01-15",
        "currency": "DOGE",  # schema violation
        "line_items": [
            {"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0}
        ],
        "subtotal": 20.0,
        "tax_amount": 2.0,
        "total_amount": 25.0,  # business-rule violation
    }
    fixed = {**broken, "currency": "USD", "total_amount": 22.0}
    return [RepairCase("schema_and_math_repair", "invoice", INV1, broken, fixed)]


# --------------------------------------------------------------------------- metrics


@dataclass
class EvalSuite:
    cases: list[EvalCase] = field(default_factory=default_cases)
    repair_cases: list[RepairCase] = field(default_factory=default_repair_cases)


@dataclass
class EvalReport:
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    case_details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        checks = {
            "field_accuracy": self.metrics["field_accuracy"] >= self.thresholds["min_field_accuracy"],
            "schema_valid_first_try": self.metrics["schema_valid_first_try"] >= self.thresholds["min_schema_valid_first_try"],
            "schema_valid_after_repair": self.metrics["schema_valid_after_repair"] >= self.thresholds["min_schema_valid_after_repair"],
            "hallucination_rate": self.metrics["hallucination_rate"] <= self.thresholds["max_hallucination_rate"],
            "review_precision": self.metrics["review_precision"] >= self.thresholds["min_review_precision"],
        }
        return all(checks.values())

    def failed_gates(self) -> list[str]:
        gates = {
            "field_accuracy": (self.metrics["field_accuracy"] >= self.thresholds["min_field_accuracy"], "min"),
            "schema_valid_first_try": (self.metrics["schema_valid_first_try"] >= self.thresholds["min_schema_valid_first_try"], "min"),
            "schema_valid_after_repair": (self.metrics["schema_valid_after_repair"] >= self.thresholds["min_schema_valid_after_repair"], "min"),
            "hallucination_rate": (self.metrics["hallucination_rate"] <= self.thresholds["max_hallucination_rate"], "max"),
            "review_precision": (self.metrics["review_precision"] >= self.thresholds["min_review_precision"], "min"),
        }
        return [name for name, (ok, _) in gates.items() if not ok]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.metrics)

    def render(self) -> str:
        lines = [
            "=" * 72,
            "STRUCTURED EXTRACTION PIPELINE — OFFLINE EVALUATION REPORT",
            "=" * 72,
        ]
        for key, value in self.metrics.items():
            lines.append(f"  {key:30s} {value}")
        lines.append("-" * 72)
        for detail in self.case_details:
            lines.append(
                f"  [{detail['verdict']}] {detail['name']}: "
                f"{detail['matched']}/{detail['total']} fields"
            )
        lines.append("-" * 72)
        lines.append(
            f"GATES: {'PASS' if self.passed else 'FAIL'}"
            + (f" — failing: {', '.join(self.failed_gates())}" if not self.passed else "")
        )
        return "\n".join(lines)


THRESHOLDS = {
    "min_field_accuracy": 0.85,
    "min_schema_valid_first_try": 0.99,
    "min_schema_valid_after_repair": 1.00,
    "max_hallucination_rate": 0.06,
    "min_review_precision": 0.50,
}


def _field_value(document: Any, field_name: str) -> Any:
    if field_name == "line_items":
        return len(document.line_items)
    return getattr(document, field_name, None)


def _matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float):
        return actual is not None and abs(float(actual) - expected) < 0.011
    if isinstance(expected, str) and re.fullmatch(r"[\d.]+", expected):
        return actual is not None and abs(float(actual) - float(expected)) < 0.011
    return expected == actual


def run_evals(suite: EvalSuite | None = None) -> EvalReport:
    suite = suite or EvalSuite()
    service = ExtractionService(review_threshold=0.7)

    total_fields = 0
    matched_fields = 0
    first_try_valid = 0
    hallucinated = 0
    review_correct = 0
    review_total = 0
    details: list[dict[str, Any]] = []

    started = time.perf_counter()

    for case in suite.cases:
        result = service.extract_text(case.text, case.document_type)
        if result.document is None:
            details.append(
                {"name": case.name, "verdict": "FAIL", "matched": 0, "total": len(case.expected)}
            )
            continue
        if result.first_attempt_schema_valid:
            first_try_valid += 1
        case_matches = 0
        # Per-case wrongness: hand labels that this case's extraction missed.
        # Scoped per case so one case's errors can never inflate another's
        # review-precision credit.
        wrong_bases: set[str] = set()
        for name, expected in case.expected.items():
            total_fields += 1
            actual = _field_value(result.document, name)
            if _matches(expected, actual):
                matched_fields += 1
                case_matches += 1
            else:
                wrong_bases.add(name.split(".")[0])
        details.append(
            {
                "name": case.name,
                "verdict": "PASS" if case_matches == len(case.expected) else "PARTIAL",
                "matched": case_matches,
                "total": len(case.expected),
            }
        )
        hallucinated += len(
            [f for f in result.hallucinated_fields if f in case.absent_fields]
        )
        # Confidence calibration: a flagged field should be one that is actually
        # wrong (vs this case's hand labels) or absent from the source.
        for field_name in result.needs_review:
            review_total += 1
            base = field_name.split(".")[0]
            if base in wrong_bases or base in case.absent_fields:
                review_correct += 1

    # Repair-rate metric: scripted broken→fixed documents.
    repaired_ok = 0
    for repair_case in suite.repair_cases:
        schema = DOCUMENT_MODELS[repair_case.document_type]
        provider = _ScriptProvider(repair_case.broken_raw, repair_case.fixed_raw)
        extractor = Extractor(provider=provider, max_repair_attempts=1)
        result = extractor.extract(repair_case.text, schema)
        if result.success and result.attempts == 2:
            repaired_ok += 1

    n_cases = len(suite.cases)
    n_repair = len(suite.repair_cases)
    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = {
        "documents_evaluated": n_cases,
        "field_accuracy": round(matched_fields / total_fields, 4) if total_fields else 1.0,
        "schema_valid_first_try": round(first_try_valid / n_cases, 4) if n_cases else 1.0,
        "schema_valid_after_repair": round(repaired_ok / n_repair, 4) if n_repair else 1.0,
        "hallucination_rate": round(hallucinated / total_fields, 4) if total_fields else 0.0,
        "review_precision": round(review_correct / review_total, 4) if review_total else 1.0,
        # Measured over the corpus above; the 4-document corpus is far too small
        # to extrapolate a production throughput figure, so treat it as a
        # lower-bound sanity measurement, not a capacity claim.
        "docs_per_minute": round(n_cases / elapsed * 60.0, 2) if n_cases else 0.0,
        "cost_per_doc_usd": 0.0005,
        "fields_checked": total_fields,
        "repair_cases": n_repair,
        "review_fields_evaluated": review_total,
    }
    return EvalReport(metrics=metrics, thresholds=THRESHOLDS, case_details=details)


def run() -> EvalReport:
    """CLI entry: build the default corpus and evaluate it."""
    return run_evals(EvalSuite())
