"""Tests for the strict, schema-locked extraction models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import Currency, Invoice, LineItem, Receipt

CLEAN_INVOICE = {
    "vendor": "Acme Corp",
    "invoice_number": "INV-1001",
    "invoice_date": "2026-01-15",
    "due_date": "2026-02-14",
    "currency": "USD",
    "line_items": [
        {"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0},
        {"description": "Gadget", "quantity": 1, "unit_price": 5.5, "amount": 5.5},
    ],
    "subtotal": 25.5,
    "tax_amount": 2.55,
    "total_amount": 28.05,
}


def test_clean_invoice_parses_into_typed_model():
    invoice = Invoice.model_validate(CLEAN_INVOICE)
    assert invoice.vendor == "Acme Corp"
    assert invoice.currency is Currency.USD
    assert invoice.line_items[0].quantity == 2
    assert invoice.total_amount == pytest.approx(28.05)


def test_unknown_fields_are_rejected():
    payload = {**CLEAN_INVOICE, "unexpected_field": "hallucinated"}
    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


def test_wrong_currency_enum_is_rejected():
    payload = {**CLEAN_INVOICE, "currency": "DOGE"}
    with pytest.raises(ValidationError):
        Invoice.model_validate(payload)


def test_line_item_rejects_non_positive_quantity():
    with pytest.raises(ValidationError):
        LineItem(description="x", quantity=0, unit_price=1.0, amount=0.0)


def test_receipt_is_a_distinct_strict_model():
    receipt = Receipt.model_validate(
        {
            "merchant": "Coffee Shop",
            "purchase_date": "2026-03-02",
            "currency": "USD",
            "line_items": [
                {"description": "Latte", "quantity": 1, "unit_price": 4.5, "amount": 4.5}
            ],
            "total_amount": 4.5,
        }
    )
    assert receipt.merchant == "Coffee Shop"
    assert receipt.total_amount == pytest.approx(4.5)
