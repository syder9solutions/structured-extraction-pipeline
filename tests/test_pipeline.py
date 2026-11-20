"""Tests for schema extraction and the bounded self-repair loop.

The fake provider is a real object (not a MagicMock): it returns scripted raw
payloads in sequence, letting us assert the *behaviour* of the repair loop —
that it feeds validation errors back, retries a bounded number of times, and
gives up cleanly with structured diagnostics when it cannot repair.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import Invoice
from src.pipeline import Extractor, ExtractionAttempt, SchemaProvider


class ScriptedProvider(SchemaProvider):
    """Returns scripted raw dicts; records the repair prompts it receives."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.prompts: list = []

    def generate(self, document_text: str, schema: type, repair_prompt: str | None) -> dict:
        self.prompts.append(repair_prompt)
        if not self.responses:
            raise AssertionError("provider called more times than scripted")
        return self.responses.pop(0)


CLEAN_RAW = {
    "vendor": "Acme Corp",
    "invoice_number": "INV-1001",
    "invoice_date": "2026-01-15",
    "due_date": "2026-02-14",
    "currency": "USD",
    "line_items": [
        {"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0}
    ],
    "subtotal": 20.0,
    "tax_amount": 2.0,
    "total_amount": 22.0,
}


def test_clean_extraction_succeeds_on_first_attempt():
    provider = ScriptedProvider([CLEAN_RAW])
    extractor = Extractor(provider=provider, max_repair_attempts=2)
    result = extractor.extract("invoice text", Invoice)
    assert isinstance(result.document, Invoice)
    assert result.attempts == 1
    assert result.repairs == []
    assert provider.prompts == [None]


def test_schema_violation_triggers_repair_with_error_feedback():
    bad = {**CLEAN_RAW, "currency": "DOGE"}  # enum violation
    provider = ScriptedProvider([bad, CLEAN_RAW])
    extractor = Extractor(provider=provider, max_repair_attempts=2)
    result = extractor.extract("invoice text", Invoice)
    assert result.attempts == 2
    assert isinstance(result.document, Invoice)
    assert len(result.repairs) == 1
    assert "currency" in provider.prompts[1]  # error was fed back
    assert "DOGE" in provider.prompts[1]


def test_business_rule_violation_triggers_repair():
    bad_total = {**CLEAN_RAW, "total_amount": 21.0}
    provider = ScriptedProvider([bad_total, CLEAN_RAW])
    extractor = Extractor(provider=provider, max_repair_attempts=2)
    result = extractor.extract("invoice text", Invoice)
    assert result.attempts == 2
    assert isinstance(result.document, Invoice)
    assert "total_amount" in provider.prompts[1]


def test_repair_loop_is_bounded():
    bad = {**CLEAN_RAW, "currency": "DOGE"}
    provider = ScriptedProvider([bad, bad, bad])
    extractor = Extractor(provider=provider, max_repair_attempts=2)
    result = extractor.extract("invoice text", Invoice)
    assert result.document is None
    assert result.attempts == 3  # 1 initial + 2 repairs, never a 4th call
    assert len(result.repairs) == 2
    assert len(provider.prompts) == 3


def test_failure_result_carries_structured_errors():
    # Two schema-invalid responses exhaust the attempts: the final errors must
    # be structured and carry the offending values (repairable diagnostics).
    bad = {**CLEAN_RAW, "currency": "DOGE", "total_amount": "five"}
    provider = ScriptedProvider([bad])
    extractor = Extractor(provider=provider, max_repair_attempts=0)
    result = extractor.extract("invoice text", Invoice)
    assert result.document is None
    assert any("currency" in err for err in result.errors)
    assert any("total_amount" in err for err in result.errors)


def test_business_rule_errors_are_reported_alongside_kept_document():
    # Schema-valid but two independent business-rule violations: both must be
    # reported together so a reviewer (or repair prompt) can fix them at once.
    bad = {**CLEAN_RAW, "subtotal": 15.0, "total_amount": 5.0}
    provider = ScriptedProvider([bad])
    extractor = Extractor(provider=provider, max_repair_attempts=0)
    result = extractor.extract("invoice text", Invoice)
    assert result.document is not None  # kept for review, not dropped
    assert any("subtotal" in err for err in result.errors)
    assert any("total_amount" in err for err in result.errors)
    assert not result.succeeded


def test_business_rule_failure_on_final_attempt_routes_to_review_not_drop():
    bad = {**CLEAN_RAW, "total_amount": 21.0}
    provider = ScriptedProvider([bad])
    extractor = Extractor(provider=provider, max_repair_attempts=0)
    result = extractor.extract("invoice text", Invoice)
    assert result.document is not None  # schema-valid: never silently dropped
    assert result.document.total_amount == 21.0
    assert result.business_issues  # arithmetic issue recorded
    assert not result.succeeded  # but it is NOT a clean success
    assert any("total_amount" in e for e in result.errors)


def test_first_attempt_schema_validity_is_tracked_separately():
    # Business-rule failure on attempt 1 is still schema-valid on attempt 1.
    bad_math = {**CLEAN_RAW, "total_amount": 21.0}
    provider = ScriptedProvider([bad_math])
    result = Extractor(provider=provider, max_repair_attempts=0).extract("t", Invoice)
    assert result.first_attempt_schema_valid is True

    bad_schema = {**CLEAN_RAW, "currency": "DOGE"}
    provider2 = ScriptedProvider([bad_schema, bad_schema])
    result2 = Extractor(provider=provider2, max_repair_attempts=1).extract("t", Invoice)
    assert result2.first_attempt_schema_valid is False
    assert result2.document is None


def test_unknown_field_rejected_then_repaired():
    bad = {**CLEAN_RAW, " hallucinated_field": "oops"}
    provider = ScriptedProvider([bad, CLEAN_RAW])
    extractor = Extractor(provider=provider, max_repair_attempts=1)
    result = extractor.extract("invoice text", Invoice)
    assert isinstance(result.document, Invoice)
    assert "hallucinated_field" in provider.prompts[1]


def test_invalid_raw_type_from_provider_is_a_repairable_error():
    provider = ScriptedProvider(["not a dict", CLEAN_RAW])
    extractor = Extractor(provider=provider, max_repair_attempts=1)
    result = extractor.extract("invoice text", Invoice)
    assert isinstance(result.document, Invoice)
    assert result.attempts == 2


def test_extraction_attempt_is_a_typed_record():
    attempt = ExtractionAttempt(index=1, raw=CLEAN_RAW, errors=[])
    assert attempt.index == 1
    assert attempt.errors == []


def test_provider_raises_propagates_as_failure_not_crash():
    class ExplodingProvider(SchemaProvider):
        def generate(self, document_text, schema, repair_prompt):
            raise RuntimeError("provider down")

    extractor = Extractor(provider=ExplodingProvider(), max_repair_attempts=1)
    result = extractor.extract("text", Invoice)
    assert result.document is None
    assert any("provider down" in err for err in result.errors)
