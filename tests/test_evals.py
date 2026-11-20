"""Tests for the deterministic offline evaluation harness."""

from __future__ import annotations

import pytest

from evals.extraction_eval import EvalCase, EvalSuite, run_evals


def _case(name, text, fields):
    return EvalCase(name=name, document_type="invoice", text=text, expected=fields)


def test_eval_suite_runs_and_renders():
    suite = EvalSuite(cases=[], repair_cases=[])
    report = run_evals(suite)
    rendered = report.render()
    assert "STRUCTURED EXTRACTION" in rendered
    assert "GATES:" in rendered


def test_eval_report_has_required_metric_keys():
    suite = EvalSuite(cases=[], repair_cases=[])
    report = run_evals(suite)
    metrics = report.as_dict()
    for key in (
        "field_accuracy",
        "schema_valid_first_try",
        "hallucination_rate",
        "review_precision",
        "docs_per_minute",
        "cost_per_doc_usd",
    ):
        assert key in metrics


def test_review_precision_responds_to_a_ground_truth_change():
    """D4: the review_precision gate must be able to fail.

    Removing one ground-truth ``absent_fields`` label used to collapse the metric
    1.0 -> 0.0. Once the repair loop stopped routing anything to review, the
    denominator went to zero and the metric returned its 1.0 default *both*
    ways, so ``min_review_precision: 0.50`` passed without testing anything.
    This asserts the metric is falsifiable over the shipped corpus.
    """
    from evals.extraction_eval import default_cases

    baseline = run_evals(EvalSuite())
    perturbed_cases = [
        case
        if case.name != "missing_total"
        else EvalCase(
            case.name,
            case.document_type,
            case.text,
            case.expected,
            absent_fields=(),
        )
        for case in default_cases()
    ]
    perturbed = run_evals(EvalSuite(cases=perturbed_cases))

    assert baseline.metrics["review_fields_evaluated"] > 0, (
        "empty denominator: review_precision is computed over no fields"
    )
    assert baseline.metrics["review_precision"] == 1.0
    assert perturbed.metrics["review_precision"] < baseline.metrics["review_precision"], (
        "review_precision reported 1.0 with and without the label: the metric "
        "cannot fail and its gate proves nothing"
    )


def test_repair_cases_measure_schema_valid_after_repair():
    suite = EvalSuite(cases=[], repair_cases=[])
    report = run_evals(suite)
    assert "schema_valid_after_repair" in report.as_dict()


def test_docs_per_minute_is_measured_not_synthesised():
    """Throughput must come from the clock, not from ``n_cases * 60``.

    A synthesised figure made the reported rate identical for any corpus size
    and independent of the machine — it could never reveal a slowdown.
    """
    from evals.extraction_eval import default_cases

    small = run_evals(EvalSuite(cases=default_cases()[:1], repair_cases=[]))
    large = run_evals(EvalSuite(cases=default_cases(), repair_cases=[]))
    # Both are real measurements; each must be positive and finite.
    for report in (small, large):
        assert report.metrics["docs_per_minute"] > 0
        assert report.metrics["docs_per_minute"] != float("inf")
    # The old formula produced exactly 60 * n_cases, which is what this asserts
    # is no longer the case for at least one corpus size.
    assert not (
        small.metrics["docs_per_minute"] == 1 * 60
        and large.metrics["docs_per_minute"] == 4 * 60
    )


def test_batch_throughput_has_no_one_second_floor():
    """A fast batch must not be reported as if it took a full second."""
    from src.batch import BatchResult
    from src.service import ServiceResult

    def _result() -> ServiceResult:
        return ServiceResult(document=None, success=True, attempts=1)

    one_doc_fast = BatchResult(results=[_result()], elapsed_seconds=0.01)
    assert one_doc_fast.docs_per_minute == pytest.approx(1 / 0.01 * 60, rel=1e-6)

    many_fast = BatchResult(results=[_result() for _ in range(400)], elapsed_seconds=0.01)
    assert many_fast.docs_per_minute == pytest.approx(400 / 0.01 * 60, rel=1e-6)


def test_eval_gate_passes_on_clean_suite():
    suite = EvalSuite(cases=[], repair_cases=[])
    report = run_evals(suite)
    assert report.passed is True


def test_review_precision_credits_only_per_case_wrong_or_absent_flags(monkeypatch):
    """A flagged field counts toward review precision only when it is wrong in
    the SAME case's hand labels (or absent from that case's source)."""
    from types import SimpleNamespace

    from evals.extraction_eval import EvalCase, EvalSuite, run_evals

    class _FakeResult(SimpleNamespace):
        pass

    class _FakeService:
        def __init__(self, review_threshold: float = 0.7) -> None:
            self.review_threshold = review_threshold

        def extract_text(self, text: str, document_type: str) -> _FakeResult:
            scripted = {
                "a": _FakeResult(
                    document=SimpleNamespace(vendor="Right Vendor", total_amount=1.0),
                    first_attempt_schema_valid=True,
                    hallucinated_fields=[],
                    needs_review=[],  # correct extraction, nothing flagged
                ),
                "b": _FakeResult(
                    document=SimpleNamespace(vendor="Wrong Corp", total_amount=2.0),
                    first_attempt_schema_valid=True,
                    hallucinated_fields=[],
                    needs_review=["vendor"],  # flagged AND wrong -> credited
                ),
                "c": _FakeResult(
                    document=SimpleNamespace(vendor="Fine Ltd", total_amount=3.0),
                    first_attempt_schema_valid=True,
                    hallucinated_fields=[],
                    needs_review=["vendor"],  # flagged but correct -> NOT credited
                ),
            }
            return scripted[text]

    monkeypatch.setattr("evals.extraction_eval.ExtractionService", _FakeService)
    suite = EvalSuite(
        cases=[
            EvalCase("a", "invoice", "a", {"vendor": "Right Vendor", "total_amount": 1.0}),
            EvalCase("b", "invoice", "b", {"vendor": "Expected Inc", "total_amount": 2.0}),
            EvalCase("c", "invoice", "c", {"vendor": "Fine Ltd", "total_amount": 3.0}),
        ],
        repair_cases=[],
    )
    report = run_evals(suite)
    # 2 flags raised, only case b's was genuinely wrong -> precision 0.5.
    # (Global wrong-field accumulation would credit case c too and yield 1.0.)
    assert report.metrics["review_precision"] == 0.5
    assert report.metrics["field_accuracy"] == 0.8333  # 5/6: only b.vendor missed
