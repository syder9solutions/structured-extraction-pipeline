"""Tests for the end-to-end extraction service and hallucination detection."""

from __future__ import annotations

import pytest

from src.models import Invoice, Receipt
from src.pipeline import Extractor
from src.service import (
    ExtractionService,
    assert_no_hallucinated_fields,
    detect_hallucinations,
    extract_file,
)

INVOICE_TEXT = """
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


def _service(**kwargs) -> ExtractionService:
    return ExtractionService(**kwargs)


def test_end_to_end_invoice_extraction_from_text():
    result = _service().extract_text(INVOICE_TEXT, "invoice")
    assert result.success
    assert result.document is not None
    assert isinstance(result.document, Invoice)
    assert result.document.vendor == "Acme Corp"
    assert result.document.total_amount == pytest.approx(28.05)
    assert result.validation_issues == []


def test_extract_text_dispatch_to_receipt_schema():
    text = """
RECEIPT
Coffee Shop
Date: 2026-03-02
Currency: USD
Latte  1 x 4.50 = 4.50
Subtotal: 4.50 USD
Tax: 0.00 USD
Total: 4.50 USD
"""
    result = _service().extract_text(text, "receipt")
    assert result.success
    assert isinstance(result.document, Receipt)
    assert result.document.merchant == "Coffee Shop"


def test_unknown_document_type_is_rejected():
    with pytest.raises(ValueError, match="document_type"):
        _service().extract_text(INVOICE_TEXT, "passport")


def test_result_carries_confidence_and_review_queue():
    result = _service(review_threshold=0.7).extract_text(INVOICE_TEXT, "invoice")
    assert result.confidence is not None
    assert 0 <= result.confidence.overall <= 1
    assert isinstance(result.needs_review, list)


def test_low_confidence_field_lands_in_review_queue():
    # Force all declared confidences low.
    class LowConfidence:
        def generate(self, document_text, schema, repair_prompt):
            from src.providers import DeterministicProvider

            raw = DeterministicProvider().generate(document_text, schema, repair_prompt)
            raw["_confidence"] = {k: 0.2 for k in raw if k != "line_items"}
            return raw

    service = ExtractionService(provider=LowConfidence(), review_threshold=0.7)
    result = service.extract_text(INVOICE_TEXT, "invoice")
    assert result.success
    assert "vendor" in result.needs_review


def test_extract_file_reads_and_extracts(tmp_path):
    path = tmp_path / "inv.txt"
    path.write_text(INVOICE_TEXT, encoding="utf-8")
    result = extract_file(path, "invoice")
    assert result.success
    assert result.document.total_amount == pytest.approx(28.05)


def test_detect_hallucinations_flags_fields_absent_from_source():
    document = Invoice.model_validate(
        {
            "vendor": "Acme Corp",
            "invoice_number": "INV-999",  # NOT in source
            "invoice_date": "2026-01-15",
            "currency": "USD",
            "line_items": [
                {"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0}
            ],
            "subtotal": 20.0,
            "tax_amount": 0.0,
            "total_amount": 20.0,
        }
    )
    source = "INVOICE\nAcme Corp\nInvoice Date: 2026-01-15\nWidget 2 10.00 20.00\nTotal: 20.00"
    flags = detect_hallucinations(document, source)
    assert "invoice_number" in flags


def test_hallucination_detection_ignores_numeric_formatting():
    document = Invoice.model_validate(
        {
            "vendor": "Acme Corp",
            "invoice_number": "INV-1001",
            "invoice_date": "2026-01-15",
            "currency": "USD",
            "line_items": [
                {"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0}
            ],
            "subtotal": 20.0,
            "tax_amount": 0.0,
            "total_amount": 20.0,
        }
    )
    source = "INV-1001 | 2026-01-15 | 20.00 | 20.00"
    flags = detect_hallucinations(document, source)
    assert "invoice_number" not in flags
    assert "total_amount" not in flags


def test_detect_hallucinations_handles_enum_and_date_values():
    # Enums (e.g. Currency) stringify as 'Currency.USD'; the detector must use
    # the .value so a legitimate currency is not flagged as invented.
    document = Invoice.model_validate(
        {
            "vendor": "Acme Corp",
            "invoice_number": "INV-1001",
            "invoice_date": "2026-01-15",
            "currency": "USD",
            "line_items": [
                {"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0}
            ],
            "subtotal": 20.0,
            "tax_amount": 2.0,
            "total_amount": 22.0,
        }
    )
    source = "INVOICE Acme Corp INV-1001 2026-01-15 USD Widget 2 10.00 20.00 Subtotal 20.00 Tax 2.00 Total 22.00"
    assert detect_hallucinations(document, source) == []


def test_assert_no_hallucinations_raises_when_found():
    document = Invoice.model_validate(
        {
            "vendor": "Acme Corp",
            "invoice_number": "INV-999",
            "invoice_date": "2026-01-15",
            "currency": "USD",
            "line_items": [],
            "subtotal": 0.0,
            "tax_amount": 0.0,
            "total_amount": 0.0,
        }
    )
    with pytest.raises(ValueError, match="[Hh]allucinat"):
        assert_no_hallucinated_fields(document, "Acme Corp | 2026-01-15")


def test_end_to_end_wrong_total_repair():
    text = """INVOICE
Acme Corp
Invoice Number: INV-1001
Invoice Date: 2026-01-15
Currency: USD
Widget 2 10.00 20.00
Subtotal: 20.00 USD
Tax: 2.00 USD
Total: 21.00 USD
"""
    res = ExtractionService(max_repair_attempts=2).extract_text(text, "invoice")
    assert res.success
    assert res.document.total_amount == pytest.approx(22.00)
    assert res.attempts == 2


def test_provider_failure_during_repair_does_not_pollute_business_issues():
    class ExplodingOnRepairProvider:
        name = "exploding"
        def __init__(self):
            self.calls = 0
        def generate(self, document_text, schema, repair_prompt):
            self.calls += 1
            if self.calls == 1:
                return {
                    "vendor": "Acme Corp",
                    "invoice_number": "INV-1001",
                    "invoice_date": "2026-01-15",
                    "currency": "USD",
                    "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0}],
                    "subtotal": 20.0,
                    "tax_amount": 2.0,
                    "total_amount": 21.0,
                }
            raise RuntimeError("provider down")

    service = ExtractionService(provider=ExplodingOnRepairProvider(), max_repair_attempts=1)
    res = service.extract_text(INVOICE_TEXT, "invoice")
    assert any("provider down" in err for err in res.errors)
    assert not any("provider down" in str(i) for i in res.validation_issues)
    assert res.document is not None
    assert res.document.total_amount == 21.0
    assert res.attempts == 2
    # The crash is reported in errors only — never as a business/validation issue —
    # but the kept document still violates arithmetic, so this is NOT a success.
    # Asserting the error-channel isolation AND the success contract together is
    # what the earlier rework omitted, which is why the suite stayed green while
    # `success` reported True on a wrong document.
    assert [i.code for i in res.validation_issues] == ["total_mismatch"]
    assert res.success is False

    # At the extractor level the residual business issue survives the crash and
    # carries only the genuine defect, never the provider text.
    raw_result = ExplodingOnRepairProvider()
    extraction = Extractor(provider=raw_result, max_repair_attempts=1).extract(
        INVOICE_TEXT, Invoice
    )
    assert extraction.business_issues
    assert not any("provider error" in issue for issue in extraction.business_issues)
    assert any("provider error" in err for err in extraction.errors)
    assert extraction.succeeded is False
