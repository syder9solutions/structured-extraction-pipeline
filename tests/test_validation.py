"""Tests for business-rule validation (as opposed to schema-shape validation)."""

from __future__ import annotations

from src.models import Invoice, Receipt
from src.validation import ValidationIssue, validate_document, validate_invoice, validate_receipt

CLEAN = {
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


def _fields(issues):
    return {(issue.field, issue.code) for issue in issues}


def test_clean_invoice_has_no_business_issues():
    assert validate_invoice(Invoice.model_validate(CLEAN)) == []


def test_line_item_amount_must_equal_quantity_times_unit_price():
    payload = {**CLEAN, "line_items": [{**CLEAN["line_items"][0], "amount": 19.0}]}
    issues = validate_invoice(Invoice.model_validate(payload))
    assert ("line_items.0.amount", "line_item_math") in _fields(issues)


def test_subtotal_must_equal_sum_of_line_amounts():
    payload = {**CLEAN, "subtotal": 24.0}
    issues = validate_invoice(Invoice.model_validate(payload))
    assert ("subtotal", "subtotal_mismatch") in _fields(issues)


def test_total_must_equal_subtotal_plus_tax():
    payload = {**CLEAN, "total_amount": 25.5}  # forgot tax
    issues = validate_invoice(Invoice.model_validate(payload))
    assert ("total_amount", "total_mismatch") in _fields(issues)


def test_due_date_before_invoice_date_is_an_issue():
    payload = {**CLEAN, "due_date": "2026-01-01"}
    issues = validate_invoice(Invoice.model_validate(payload))
    assert ("due_date", "date_order") in _fields(issues)


def test_empty_line_items_is_an_issue():
    payload = {**CLEAN, "line_items": [], "subtotal": 0.0, "tax_amount": 0.0, "total_amount": 0.0}
    issues = validate_invoice(Invoice.model_validate(payload))
    assert ("line_items", "empty_line_items") in _fields(issues)


def test_floating_point_tolerance_does_not_create_false_positives():
    payload = {**CLEAN, "subtotal": 25.5000001, "total_amount": 28.0500001}
    assert validate_invoice(Invoice.model_validate(payload)) == []


def test_empty_line_items_is_a_business_violation_for_both_types():
    invoice = Invoice.model_validate(
        {
            "vendor": "X",
            "invoice_number": "INV-1",
            "invoice_date": "2026-01-01",
            "currency": "USD",
            "line_items": [],
            "subtotal": 0.0,
            "tax_amount": 0.0,
            "total_amount": 0.0,
        }
    )
    receipt = Receipt.model_validate(
        {
            "merchant": "Y",
            "purchase_date": "2026-01-01",
            "currency": "USD",
            "line_items": [],
            "subtotal": 0.0,
            "tax_amount": 0.0,
            "total_amount": 0.0,
        }
    )
    assert any("line item" in i.message.lower() for i in validate_invoice(invoice))
    assert any("line item" in i.message.lower() for i in validate_receipt(receipt))


def test_validate_document_dispatches_and_never_raises():
    receipt = Receipt.model_validate(
        {
            "merchant": "Shop",
            "purchase_date": "2026-03-02",
            "currency": "USD",
            "line_items": [
                {"description": "x", "quantity": 2, "unit_price": 3.0, "amount": 6.0}
            ],
            "subtotal": 6.0,
            "tax_amount": 0.0,
            "total_amount": 99.0,  # wrong on purpose
        }
    )
    issues = validate_document(receipt)
    assert ("total_amount", "total_mismatch") in _fields(issues)
