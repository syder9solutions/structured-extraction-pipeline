"""Business-rule validation layered on top of schema validation.

Schema validation answers "is this a well-formed invoice?". These rules answer
"is this a *correct* invoice?" — arithmetic consistency, plausible dates, and
non-empty line items. Each issue carries a stable ``code`` so a repair loop can
feed a precise, machine-readable instruction back to the extractor.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import Invoice, Receipt, StrictModel

# Money is decimal-ish; allow sub-cent slack so float noise never raises a flag.
TOLERANCE = 0.01


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "code": self.code, "message": self.message}


def _sum_line_amounts(model: Invoice | Receipt) -> float:
    return sum(item.amount for item in model.line_items)


def _common_issues(model: Invoice | Receipt) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for index, item in enumerate(model.line_items):
        expected = item.quantity * item.unit_price
        if abs(item.amount - expected) > TOLERANCE:
            issues.append(
                ValidationIssue(
                    f"line_items.{index}.amount",
                    "line_item_math",
                    f"Line {index} amount {item.amount} should equal "
                    f"quantity {item.quantity} x unit_price {item.unit_price} = {expected:.2f}",
                )
            )
    if not model.line_items:
        issues.append(
            ValidationIssue(
                "line_items",
                "empty_line_items",
                "Document has no line items — a real invoice/receipt always has at least one",
            )
        )
    if abs(model.subtotal - _sum_line_amounts(model)) > TOLERANCE:
        issues.append(
            ValidationIssue(
                "subtotal",
                "subtotal_mismatch",
                f"subtotal {model.subtotal} should equal the sum of line amounts "
                f"({_sum_line_amounts(model):.2f})",
            )
        )
    if abs(model.total_amount - (model.subtotal + model.tax_amount)) > TOLERANCE:
        issues.append(
            ValidationIssue(
                "total_amount",
                "total_mismatch",
                f"total_amount {model.total_amount} should equal subtotal + tax "
                f"({model.subtotal + model.tax_amount:.2f})",
            )
        )
    return issues


def validate_invoice(invoice: Invoice) -> list[ValidationIssue]:
    """Return all business-rule issues found in an invoice."""
    issues = _common_issues(invoice)
    if invoice.due_date is not None and invoice.due_date < invoice.invoice_date:
        issues.append(
            ValidationIssue(
                "due_date",
                "date_order",
                f"due_date {invoice.due_date} is before invoice_date {invoice.invoice_date}",
            )
        )
    return issues


def validate_receipt(receipt: Receipt) -> list[ValidationIssue]:
    """Return all business-rule issues found in a receipt."""
    return _common_issues(receipt)


def validate_document(model: StrictModel) -> list[ValidationIssue]:
    """Dispatch to the right validator based on the model type."""
    if isinstance(model, Invoice):
        return validate_invoice(model)
    if isinstance(model, Receipt):
        return validate_receipt(model)
    # Unknown strict models have no business rules beyond the schema itself.
    return []
