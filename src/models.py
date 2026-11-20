"""Strict, schema-locked Pydantic models for document extraction.

Design notes (schema design is accuracy engineering):
- ``extra="forbid"`` makes hallucinated fields fail validation instead of
  silently flowing downstream.
- Field descriptions are not decoration: they are injected into the extractor
  prompt as accuracy levers, telling the model exactly what each field means.
- Enums and constrained numeric types make *wrong values* hard to produce,
  not merely wrong shapes.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


class StrictModel(BaseModel):
    """Base for all extraction models: reject unknown fields by default."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    PKR = "PKR"
    INR = "INR"
    JPY = "JPY"


class LineItem(StrictModel):
    description: str = Field(
        min_length=1, description="Human-readable description of the item or service"
    )
    quantity: PositiveInt = Field(description="Number of units; must be a positive integer")
    unit_price: PositiveFloat = Field(description="Price per single unit, excluding tax")
    amount: float = Field(ge=0, description="Line total = quantity x unit_price, excluding tax")


class Invoice(StrictModel):
    vendor: str = Field(min_length=1, description="Name of the company that issued the invoice")
    invoice_number: str = Field(min_length=1, description="Vendor's invoice identifier")
    invoice_date: date = Field(description="Date the invoice was issued (YYYY-MM-DD)")
    due_date: date | None = Field(
        default=None, description="Payment due date (YYYY-MM-DD), if stated"
    )
    currency: Currency = Field(description="Three-letter ISO currency code of all amounts")
    line_items: list[LineItem] = Field(
        default_factory=list, description="Every line item listed on the invoice"
    )
    subtotal: float = Field(ge=0, description="Sum of line item amounts before tax")
    tax_amount: float = Field(ge=0, description="Total tax added to the subtotal")
    total_amount: float = Field(ge=0, description="Final amount due = subtotal + tax_amount")


class Receipt(StrictModel):
    merchant: str = Field(min_length=1, description="Name of the merchant on the receipt")
    purchase_date: date = Field(description="Date of purchase (YYYY-MM-DD)")
    currency: Currency = Field(description="Three-letter ISO currency code of all amounts")
    line_items: list[LineItem] = Field(
        default_factory=list, description="Every item purchased on the receipt"
    )
    subtotal: float = Field(
        default=0.0, ge=0, description="Sum of line item amounts before tax"
    )
    tax_amount: float = Field(default=0.0, ge=0, description="Total tax on the purchase")
    total_amount: float = Field(ge=0, description="Final amount paid = subtotal + tax_amount")


DOCUMENT_MODELS = {"invoice": Invoice, "receipt": Receipt}
