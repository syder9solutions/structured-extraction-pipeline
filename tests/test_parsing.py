"""Tests for parsing: normalization, text files, and real PDF extraction."""

from __future__ import annotations

import zlib

import pytest

from src.parsing import normalize_text, parse_document, read_document


def test_normalize_text_collapses_whitespace_and_annotates_currency():
    raw = "Acme Corp\n\n\n\nInvoice\t\tTotal:  $1,234.56"
    result = normalize_text(raw)
    assert "Acme Corp" in result
    assert "$" not in result or "USD" in result
    assert "\n\n\n" not in result  # paragraph breaks preserved, runs collapsed


def test_read_document_from_text_path(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("hello world", encoding="utf-8")
    assert read_document(path) == "hello world"


def test_read_document_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(b"whatever")
    with pytest.raises(ValueError, match="Unsupported document type"):
        read_document(path)


def test_read_document_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_document(tmp_path / "nope.txt")


def test_parse_document_routes_pdf_to_pdf_parser(tmp_path, monkeypatch):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    called = {}

    def fake_pdf(p):
        called["path"] = p
        return "PDF TEXT"

    monkeypatch.setattr("src.parsing._parse_pdf", fake_pdf)
    assert parse_document(path) == "PDF TEXT"
    assert called["path"] == path


def _minimal_pdf(text_lines: list[str]) -> bytes:
    """Build a real one-page PDF whose text content is `text_lines`.

    Hand-rolled so the test suite has zero binary fixtures and no generator
    dependency: a valid PDF is just indexed objects with a content stream.
    """
    content_parts = ["BT"]
    y = 720
    for line in text_lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_parts.append(f"/F1 12 Tf 1 0 0 1 40 {y} Tm ({escaped}) Tj")
        y -= 16
    content_parts.append("ET")
    content = "\n".join(content_parts).encode("latin-1")

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")  # 1
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")  # 2
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )  # 3
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )  # 4
    compressed = zlib.compress(content)
    objects.append(
        b"<< /Length " + str(len(compressed)).encode() + b" /Filter /FlateDecode >>\nstream\n"
        + compressed + b"\nendstream"
    )  # 5

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


def test_parse_real_pdf_extracts_invoice_text(tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(
        _minimal_pdf(
            [
                "INVOICE",
                "Acme Corp",
                "Invoice Number: INV-2002",
                "Invoice Date: 2026-01-15",
                "Currency: USD",
                "Widget 2 10.00 20.00",
                "Subtotal: 20.00 USD",
                "Tax: 2.00 USD",
                "Total: 22.00 USD",
            ]
        )
    )
    # Real pypdf read — not a mock — plus normalization and extraction.
    text = parse_document(pdf_path)
    assert "INVOICE" in text
    assert "Acme Corp" in text

    from src.service import ExtractionService

    result = ExtractionService().extract_text(text, "invoice")
    assert result.success
    assert result.document is not None
    assert result.document.vendor == "Acme Corp"
    assert result.document.invoice_number == "INV-2002"
    assert result.document.total_amount == pytest.approx(22.0)


def test_pdf_without_pypdf_raises_actionable_error(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")
    with pytest.raises(RuntimeError, match="pip install pypdf"):
        parse_document(pdf_path)
