"""Document parsing: load raw bytes/text and normalize before extraction.

Garbage in → garbage out. This module owns the boundary: it converts PDFs and
plain text into a single normalized string and never invents content.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PLAIN_TEXT_SUFFIXES = {".txt", ".md", ".text"}
CURRENCY_SYMBOLS = {"$": " USD ", "€": " EUR ", "£": " GBP ", "₨": " PKR "}


def normalize_text(text: str) -> str:
    """Collapse noise and annotate currency symbols without changing meaning."""
    for symbol, code in CURRENCY_SYMBOLS.items():
        text = text.replace(symbol, code)
    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pypdf is required to parse PDFs (pip install pypdf)") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_document(path: str | Path) -> str:
    """Read a document from disk based on its extension."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")
    suffix = path.suffix.lower()
    if suffix in PLAIN_TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        # Call through the module so tests (and real deployments) can swap
        # the PDF backend by patching this single injection point.
        return sys.modules[__name__]._parse_pdf(path)
    raise ValueError(f"Unsupported document type: {suffix or '(none)'}; use .txt/.md/.pdf")


def parse_document(path: str | Path) -> str:
    """Parse + normalize a document into clean extraction input."""
    return normalize_text(read_document(path))
