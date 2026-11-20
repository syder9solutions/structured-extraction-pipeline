"""End-to-end extraction service.

Wires together parsing → schema-locked extraction → bounded self-repair →
business validation → confidence scoring → review queue → hallucination check.
This is the single entry point used by the CLI, the API, and the eval harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.confidence import ConfidenceReport, field_confidence, score_document
from src.models import DOCUMENT_MODELS, Invoice, Receipt, StrictModel
from src.parsing import parse_document
from src.pipeline import Extractor, ExtractionResult
from src.providers import DeterministicProvider
from src.validation import ValidationIssue, validate_document

# Tokens whose values are numbers/formatting variants of what appears in text;
# they should not count as hallucinations when the literal differs.
NUMERIC_FIELDS = {
    "quantity",
    "unit_price",
    "amount",
    "subtotal",
    "tax_amount",
    "total_amount",
    "invoice_date",
    "due_date",
    "purchase_date",
}


@dataclass
class ServiceResult:
    document: Invoice | Receipt | None
    success: bool
    attempts: int
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    confidence: ConfidenceReport | None = None
    needs_review: list[str] = field(default_factory=list)
    hallucinated_fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    document_type: str = ""
    first_attempt_schema_valid: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "success": self.success,
            "document": self.document.model_dump(mode="json") if self.document else None,
            "attempts": self.attempts,
            "validation_issues": [i.as_dict() for i in self.validation_issues],
            "confidence": self.confidence.as_dict() if self.confidence else None,
            "needs_review": self.needs_review,
            "hallucinated_fields": self.hallucinated_fields,
            "errors": self.errors,
        }


def _leaf_values(value: Any) -> list[str]:
    if isinstance(value, Enum):
        # Enum leaves stringify as 'Currency.USD'; match on the value itself.
        return [str(value.value)]
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_leaf_values(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_leaf_values(item))
        return result
    if isinstance(value, (int, float)):
        return [str(value), f"{value:.2f}"]
    return [str(value)]


def detect_hallucinations(
    document: Invoice | Receipt, source_text: str
) -> list[str]:
    """Return dotted field paths whose value is absent from the source text.

    Conservative on purpose: only text-ish fields are checked literally, and
    numeric/date fields are skipped (formatting varies). This prioritises low
    false positives so the metric is trustworthy.
    """
    source_lower = source_text.lower()
    flags: list[str] = []
    payload = document.model_dump()
    for key, value in payload.items():
        if key in NUMERIC_FIELDS:
            continue
        if key == "line_items":
            for index, item in enumerate(value or []):
                for field_name, field_value in item.items():
                    if field_name == "description":
                        candidates = _leaf_values(field_value)
                        if not any(c.lower() in source_lower for c in candidates):
                            flags.append(f"line_items.{index}.description")
            continue
        candidates = [c for c in _leaf_values(value) if c]
        if candidates and not any(c.lower() in source_lower for c in candidates):
            flags.append(key)
    return sorted(flags)


def assert_no_hallucinated_fields(document: Invoice | Receipt, source_text: str) -> None:
    """Raise if any field value appears invented relative to the source."""
    flags = detect_hallucinations(document, source_text)
    if flags:
        raise ValueError(f"Hallucinated fields detected: {', '.join(flags)}")


class ExtractionService:
    """Orchestrate the full pipeline for one document."""

    def __init__(
        self,
        provider: Any | None = None,
        max_repair_attempts: int = 2,
        review_threshold: float = 0.7,
        check_hallucinations: bool = True,
    ) -> None:
        self.provider = provider or DeterministicProvider()
        self.extractor = Extractor(
            provider=self.provider, max_repair_attempts=max_repair_attempts
        )
        self.review_threshold = review_threshold
        self.check_hallucinations = check_hallucinations

    def extract_text(self, text: str, document_type: str) -> ServiceResult:
        if document_type not in DOCUMENT_MODELS:
            raise ValueError(
                f"Unsupported document_type {document_type!r}; "
                f"choose from {sorted(DOCUMENT_MODELS)}"
            )
        schema: type[StrictModel] = DOCUMENT_MODELS[document_type]
        extraction: ExtractionResult = self.extractor.extract(text, schema)
        if extraction.document is None:
            return ServiceResult(
                document=None,
                success=False,
                attempts=extraction.attempts,
                errors=extraction.errors,
                document_type=document_type,
            )

        issues = validate_document(extraction.document)
        # The repair loop may leave business-rule issues on a kept document;
        # surface them as structured issues alongside the re-validated ones.
        all_issues = list(issues)
        if extraction.business_issues:
            existing_fields = {i.field for i in issues}
            for bissue in extraction.business_issues:
                field_name = bissue.split(":")[0]
                if field_name not in existing_fields:
                    all_issues.append(ValidationIssue(field_name, "repair_residual", bissue))
        declared = field_confidence(extraction.document.model_dump(mode="json"))
        declared.update(extraction.declared_confidence)
        report = score_document(
            extraction.document.model_dump(mode="json"), declared, self.review_threshold
        )
        hallucinations = (
            detect_hallucinations(extraction.document, text)
            if self.check_hallucinations
            else []
        )
        needs_review = list(report.needs_review)
        # Business-rule residual issues and hallucinations force review too.
        for issue in all_issues:
            if issue.field not in needs_review:
                needs_review.append(issue.field)
        if hallucinations:
            for field_path in hallucinations:
                base = field_path.split(".")[0]
                if base not in needs_review:
                    needs_review.append(base)
        return ServiceResult(
            document=extraction.document,
            success=extraction.succeeded,  # clean pass only, no business residuals
            attempts=extraction.attempts,
            validation_issues=all_issues,
            confidence=report,
            needs_review=needs_review,
            hallucinated_fields=hallucinations,
            errors=extraction.errors,
            document_type=document_type,
            first_attempt_schema_valid=extraction.first_attempt_schema_valid,
        )

    def extract_file(self, path: str | Path, document_type: str) -> ServiceResult:
        return self.extract_text(parse_document(path), document_type)


def extract_file(path: str | Path, document_type: str) -> ServiceResult:
    """Convenience wrapper with the default service configuration."""
    return ExtractionService().extract_file(path, document_type)
