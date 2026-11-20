"""Tests for confidence scoring and human-in-the-loop routing."""

from __future__ import annotations

import pytest

from src.confidence import ConfidenceReport, field_confidence, score_document

CLEAN = {
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


def test_field_confidence_defaults_to_high_when_no_signal():
    report = field_confidence(CLEAN, declared={})
    assert report.get("vendor") == 0.95


def test_declared_field_confidence_is_preserved():
    report = field_confidence(CLEAN, declared={"vendor": 0.42})
    assert report["vendor"] == pytest.approx(0.42)


def test_score_document_flags_low_confidence_fields_for_review():
    report = score_document(CLEAN, declared={"total_amount": 0.3}, threshold=0.7)
    assert isinstance(report, ConfidenceReport)
    assert "total_amount" in report.needs_review
    assert "vendor" not in report.needs_review
    assert report.overall < 0.95  # dragged down by the one low field


def test_score_document_clean_when_all_fields_confident():
    report = score_document(CLEAN, declared={}, threshold=0.7)
    assert report.needs_review == []
    assert report.overall >= 0.9


def test_review_flag_is_deterministic_at_the_threshold():
    a = score_document(CLEAN, declared={"vendor": 0.7}, threshold=0.7)
    b = score_document(CLEAN, declared={"vendor": 0.7}, threshold=0.7)
    assert a.needs_review == b.needs_review
