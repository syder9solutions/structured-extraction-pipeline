"""Tests for the deterministic offline provider and the OpenAI adapter."""

from __future__ import annotations

import pytest

from src.models import Invoice, Receipt
from src.providers import DeterministicProvider, OpenAIProvider, get_field_confidence

INVOICE_TEXT = """
INVOICE
Acme Corp
Invoice Number: INV-1001
Invoice Date: 2026-01-15
Due Date: 2026-02-14
Currency: USD
Description        Qty   Unit Price   Amount
Widget              2       10.00       20.00
Gadget              1        5.50        5.50
Subtotal: 25.50 USD
Tax: 2.55 USD
Total: 28.05 USD
"""

RECEIPT_TEXT = """
RECEIPT
Coffee Shop
Date: 2026-03-02
Currency: USD
Latte  1 x 4.50 = 4.50
Subtotal: 4.50 USD
Tax: 0.00 USD
Total: 4.50 USD
"""


def test_deterministic_provider_extracts_invoice():
    provider = DeterministicProvider()
    raw = provider.generate(INVOICE_TEXT, Invoice, None)
    invoice = Invoice.model_validate(raw)
    assert invoice.vendor == "Acme Corp"
    assert invoice.invoice_number == "INV-1001"
    assert invoice.total_amount == pytest.approx(28.05)
    assert len(invoice.line_items) == 2


def test_deterministic_provider_extracts_receipt():
    provider = DeterministicProvider()
    raw = provider.generate(RECEIPT_TEXT, Receipt, None)
    receipt = Receipt.model_validate(raw)
    assert receipt.merchant == "Coffee Shop"
    assert receipt.total_amount == pytest.approx(4.50)


def test_deterministic_provider_honors_repair_prompt_to_fix_field():
    provider = DeterministicProvider()
    # A repair prompt adds corrected values that the simulator applies.
    raw = provider.generate(INVOICE_TEXT, Invoice, "total_amount: expected 28.05")
    assert Invoice.model_validate(raw).total_amount == pytest.approx(28.05)


def test_declared_confidence_is_per_field():
    provider = DeterministicProvider()
    provider.generate(INVOICE_TEXT, Invoice, None)
    conf = provider.last_confidence
    assert 0 < conf["vendor"] <= 1
    assert conf["total_amount"] >= conf["vendor"]  # numeric fields more certain


def test_zero_values_get_evidence_based_confidence_from_source_text():
    provider = DeterministicProvider()
    with_tax_text = "Tax: 0.00 USD\nTotal: 4.50 USD"
    raw = provider.generate(with_tax_text, Receipt, None)
    # The label exists in the source, so the zero is evidenced — confident.
    assert provider.last_confidence["tax_amount"] >= 0.7
    # No 'Total' label anywhere in the source: the filled value is a guess.
    no_total_text = "Subtotal: 4.50 USD"
    provider.generate(no_total_text, Receipt, None)
    assert provider.last_confidence["total_amount"] <= 0.6


def test_get_field_confidence_parses_provider_response():
    response = {"vendor": 0.8, "total_amount": 0.99}
    assert get_field_confidence(response) == response


def test_openai_provider_requires_api_key():
    provider = OpenAIProvider(api_key="")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        provider.generate(INVOICE_TEXT, Invoice, None)


def test_openai_provider_builds_structured_output_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": '{"vendor": "Acme Corp"}'
                                    }
                                }
                            ]
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json, headers):
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr("src.providers.httpx", type("M", (), {"Client": FakeClient})())
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
    result = provider.generate(INVOICE_TEXT, Invoice, None)
    assert result["vendor"] == "Acme Corp"
    assert captured["payload"]["model"] == "gpt-4o-mini"
    # The request defines the extraction function under ``tools`` (the real
    # Chat Completions contract) and forces it via ``tool_choice``.
    tool_def = captured["payload"]["tools"][0]
    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == "Invoice"
    assert captured["payload"]["tool_choice"]["function"]["name"] == "Invoice"
    assert "tool_calls" not in captured["payload"]
