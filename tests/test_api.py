"""Tests for the FastAPI HTTP transport (real ASGI app, no live port)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

INVOICE_TEXT = """
INVOICE
Acme Corp
Invoice Number: INV-1001
Invoice Date: 2026-01-15
Currency: USD
Widget  2  10.00  20.00
Subtotal: 20.00 USD
Tax: 2.00 USD
Total: 22.00 USD
"""


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_extract_endpoint_returns_validated_json():
    response = client.post(
        "/extract", json={"text": INVOICE_TEXT, "document_type": "invoice"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["document"]["vendor"] == "Acme Corp"
    assert body["document"]["total_amount"] == pytest.approx(22.0)


def test_extract_endpoint_rejects_unknown_document_type():
    response = client.post(
        "/extract", json={"text": INVOICE_TEXT, "document_type": "passport"}
    )
    assert response.status_code == 422  # pydantic enum validation at the boundary


def test_extract_endpoint_rejects_empty_text():
    response = client.post("/extract", json={"text": "", "document_type": "invoice"})
    assert response.status_code == 422


def test_extract_endpoint_rejects_unknown_fields():
    response = client.post(
        "/extract",
        json={"text": INVOICE_TEXT, "document_type": "invoice", "surprise": True},
    )
    assert response.status_code == 422


def test_schema_endpoint_returns_pydantic_schema():
    response = client.get("/schema/invoice")
    assert response.status_code == 200
    schema = response.json()
    assert schema["title"] == "Invoice"
    assert "vendor" in schema["properties"]
