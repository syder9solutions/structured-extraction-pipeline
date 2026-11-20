"""Schema-locked extraction with validation-driven self-repair.

The pipeline is deliberately provider-agnostic: a ``SchemaProvider`` returns a
raw Python object, and this module is responsible for the trust boundaries —
schema coercion, business-rule validation, and a *bounded* repair loop that
feeds structured error feedback back to the provider. When the loop is
exhausted the result carries structured diagnostics rather than a plausible
but wrong document.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.models import Invoice, Receipt, StrictModel
from src.validation import validate_document

Document = Invoice | Receipt


class SchemaProvider:
    """Interface for a provider that emits a schema-shaped raw object."""

    name = "base"

    def generate(
        self, document_text: str, schema: type[StrictModel], repair_prompt: str | None
    ) -> Any:
        raise NotImplementedError


@dataclass
class ExtractionAttempt:
    index: int
    raw: Any
    errors: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    document: Document | None
    attempts: int
    repairs: list[ExtractionAttempt] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    declared_confidence: dict[str, float] = field(default_factory=dict)
    # Schema passed on attempt 1 (shape/enums/types) even if business rules failed.
    first_attempt_schema_valid: bool = False
    # Business-rule issues that survived every attempt (document kept for review).
    business_issues: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """Clean success: a document that passed schema AND business rules.

        The kept document is re-validated here so a stale or empty
        ``business_issues`` list can never turn a wrong document into a pass
        (a crashed repair must report failure, never success).
        """
        if self.document is None or self.business_issues:
            return False
        return not validate_document(self.document)

    @property
    def success(self) -> bool:
        return self.succeeded

    @property
    def schema_valid(self) -> bool:
        """A document exists and satisfies the schema (may still need review)."""
        return self.document is not None

    def as_dict(self) -> dict:
        return {
            "document": self.document.model_dump(mode="json") if self.document else None,
            "attempts": self.attempts,
            "repairs": [
                {"index": a.index, "errors": a.errors} for a in self.repairs
            ],
            "errors": self.errors,
            "first_attempt_schema_valid": self.first_attempt_schema_valid,
            "business_issues": self.business_issues,
        }


_MONEY_LABELS = {
    "subtotal": r"sub[\s\-]?total",
    "tax_amount": r"tax(?!\s*date)",
    "total_amount": r"total",
}

# `Qty x Unit Price = Amount` row as the deterministic provider reads them.
_LINE_ITEM_ROW = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 &-]*?)\s+(\d+)\s+(?:x\s+)?\$?\s*"
    r"([0-9]+\.[0-9]{2})\s*(?:=|\s)\s*\$?\s*([0-9]+\.[0-9]{2})\s*$",
    re.MULTILINE,
)


def _field_evidenced(field: str, document_text: str) -> bool:
    """True when the document itself carries the label/inputs for ``field``.

    A repair directive is only trustworthy when it corrects a value the source
    actually states. Deriving ``total_amount = subtotal + tax`` for a document
    with no total line would invent data, so such fields stay flagged instead.
    """
    if field.startswith("line_items."):
        try:
            index = int(field.split(".")[1])
        except (IndexError, ValueError):
            return False
        # The row's quantity and unit price are the printed inputs of the amount.
        return len(_LINE_ITEM_ROW.findall(document_text)) > index
    pattern = _MONEY_LABELS.get(field)
    if not pattern:
        return False
    return bool(
        re.search(rf"^\s*{pattern}", document_text, re.IGNORECASE | re.MULTILINE)
    )


def _format_repair_prompt(errors: list[str], document_text: str) -> str:
    lines = [
        "The previous extraction failed validation. Return corrected JSON with the "
        "same schema. Fix EVERY issue below; do not change unrelated fields.",
        "",
    ]
    lines.extend(f"- {error}" for error in errors)
    for error in errors:
        field_match = re.match(r"^([a-zA-Z0-9_.]+):", error)
        if not field_match:
            continue
        field = field_match.group(1)
        paren_match = re.search(r"\(([0-9.]+)\)\s*$", error)
        equals_match = re.search(r"=\s*([0-9.]+)\s*$", error)
        val = None
        if paren_match:
            val = paren_match.group(1)
        elif equals_match:
            val = equals_match.group(1)
        # Only correct a field the document evidences: a derived value for an
        # absent field would be an invention, not a repair.
        if val is not None and _field_evidenced(field, document_text):
            lines.append(f"CORRECT {field} = {val}")
    return "\n".join(lines)


def _strip_metadata(raw: Any) -> tuple[Any, dict[str, Any]]:
    """Remove provider metadata keys (e.g. ``_confidence``) before validation."""
    if not isinstance(raw, dict):
        return raw, {}
    clean = {k: v for k, v in raw.items() if not k.startswith("_")}
    return clean, {k: v for k, v in raw.items() if k.startswith("_")}


def _collect_outcome(raw: Any, schema: type[StrictModel]):
    """Coerce ``raw`` into ``schema``.

    Returns ``(document, errors, schema_ok)`` where ``schema_ok`` is False only
    for shape/enum/type violations. Business-rule failures keep the document
    (flagged for review) — the pipeline surfaces uncertainty, it never silently
    drops schema-valid data.
    """
    errors: list[str] = []
    raw, _metadata = _strip_metadata(raw)
    if not isinstance(raw, dict):
        return None, [f"Expected a JSON object, got {type(raw).__name__}: {raw!r}"], False
    try:
        document = schema.model_validate(raw)
    except ValidationError as exc:
        for item in exc.errors():
            loc = ".".join(str(part) for part in item.get("loc", ())) or "root"
            message = item.get("msg", "invalid value")
            # Pydantic's enum/literal messages omit the offending value; include it
            # so a repair provider sees exactly what it got wrong.
            bad_input = item.get("input")
            if bad_input is not None and str(bad_input) not in message:
                message = f"{message} (got {bad_input!r})"
            errors.append(f"{loc}: {message}")
        return None, errors, False
    business_errors = [f"{i.field}: {i.message}" for i in validate_document(document)]
    if business_errors:
        return document, business_errors, True
    return document, [], True


class Extractor:
    """Extract a validated document, repairing schema/business errors in a bounded loop."""

    def __init__(self, provider: SchemaProvider, max_repair_attempts: int = 2) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        self.provider = provider
        self.max_repair_attempts = max_repair_attempts

    def extract(
        self, document_text: str, schema: type[StrictModel]
    ) -> ExtractionResult:
        repairs: list[ExtractionAttempt] = []
        repair_prompt: str | None = None
        total_attempts = self.max_repair_attempts + 1
        last_errors: list[str] = []
        # Errors from the last attempt that produced an analysable outcome.
        # Kept separately so a later provider crash cannot erase the reason the
        # document is wrong (overwriting would let a crash masquerade as a pass).
        last_outcome_errors: list[str] = []
        last_document = None
        last_schema_ok = False
        first_attempt_schema_valid = False
        confidence_meta: dict = {}

        for index in range(1, total_attempts + 1):
            try:
                raw = self.provider.generate(document_text, schema, repair_prompt)
                # Snapshot declared confidence immediately after generation so
                # a later attempt cannot overwrite it.
                confidence_meta = {}
                if isinstance(raw, dict) and isinstance(raw.get("_confidence"), dict):
                    confidence_meta = dict(raw["_confidence"])
                provider_conf = getattr(self.provider, "last_confidence", None)
                if isinstance(provider_conf, dict):
                    confidence_meta = dict(provider_conf)
            except Exception as exc:  # provider failures are repairable diagnostics
                provider_error = f"provider error: {exc}"
                # Keep the diagnostics from the last analysable attempt: the
                # crash is reported, but it never erases the pre-existing defect.
                last_errors = list(last_outcome_errors) + [provider_error]
                if index < total_attempts:
                    repair_prompt = _format_repair_prompt(last_errors, document_text)
                    continue
                break

            document, errors, schema_ok = _collect_outcome(raw, schema)
            if index == 1:
                first_attempt_schema_valid = schema_ok
            last_document, last_schema_ok = document, schema_ok
            if document is not None and not errors:
                return ExtractionResult(
                    document=document,
                    attempts=index,
                    repairs=repairs,
                    declared_confidence={
                        str(k): float(v)
                        for k, v in confidence_meta.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    },
                    first_attempt_schema_valid=first_attempt_schema_valid,
                )

            last_errors = errors
            last_outcome_errors = list(errors)
            # A repair is an event: only record the failed attempt when a
            # retry actually follows it (the final failure is an error, not a repair).
            if index < total_attempts:
                repairs.append(ExtractionAttempt(index=index, raw=raw, errors=errors))
                repair_prompt = _format_repair_prompt(errors, document_text)

        if last_document is not None and last_schema_ok:
            # Schema-valid but business rules did not pass: keep the document,
            # route it to review, and never silently drop it.
            # Derive the residual business issues from the kept document itself —
            # not from the mutable error log — so no retry path (including a
            # provider crash) can empty this list and turn it into a false pass.
            residual = [f"{i.field}: {i.message}" for i in validate_document(last_document)]
            return ExtractionResult(
                document=last_document,
                attempts=total_attempts,
                repairs=repairs,
                errors=last_errors,
                declared_confidence={
                    str(k): float(v)
                    for k, v in confidence_meta.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                },
                first_attempt_schema_valid=first_attempt_schema_valid,
                business_issues=residual,
            )

        return ExtractionResult(
            document=None,
            attempts=total_attempts,
            repairs=repairs,
            errors=last_errors,
            first_attempt_schema_valid=first_attempt_schema_valid,
        )


def schema_as_json_schema(schema: type[StrictModel]) -> str:
    """Serialise a model's JSON Schema for provider prompt injection."""
    return json.dumps(schema.model_json_schema(), indent=2)
