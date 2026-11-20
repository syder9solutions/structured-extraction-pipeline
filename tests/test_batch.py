"""Tests for batch processing, throughput, and cost accounting."""

from __future__ import annotations

from src.batch import BatchResult, batch_extract

DOCS = [
    """INVOICE\nAcme Corp\nInvoice Number: INV-1001\nInvoice Date: 2026-01-15\nCurrency: USD\nWidget  2  10.00  20.00\nSubtotal: 20.00 USD\nTax: 2.00 USD\nTotal: 22.00 USD""",
    """INVOICE\nBeta Ltd\nInvoice Number: INV-1002\nInvoice Date: 2026-02-15\nCurrency: USD\nGadget  1  5.50  5.50\nSubtotal: 5.50 USD\nTax: 0.00 USD\nTotal: 5.50 USD""",
]


def test_batch_extract_processes_all_documents():
    result = batch_extract([(text, "invoice") for text in DOCS])
    assert isinstance(result, BatchResult)
    assert len(result.results) == 2
    assert all(r.success for r in result.results)
    assert result.processed == 2


def test_batch_reports_success_rate():
    result = batch_extract([(DOCS[0], "invoice"), (DOCS[1], "invoice")])
    assert result.success_rate == 1.0


def test_batch_tracks_failures_without_crashing():
    # Empty text yields a zero-everything document which the business rules
    # reject (no line items); batch must isolate it, not crash.
    result = batch_extract([(DOCS[0], "invoice"), ("", "invoice")])
    assert result.processed == 2
    assert result.failed >= 1
    assert 0 <= result.success_rate < 1


def test_batch_throughput_and_cost_are_nonnegative():
    result = batch_extract([(DOCS[0], "invoice")])
    assert result.docs_per_minute >= 0
    assert result.estimated_cost_usd >= 0


def test_batch_empty_input_is_handled():
    result = batch_extract([])
    assert result.processed == 0
    assert result.success_rate == 0.0
