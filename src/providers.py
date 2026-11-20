"""Extraction providers.

Two implementations share the ``SchemaProvider`` interface:

* ``DeterministicProvider`` — a rule-based extractor that works with zero API
  keys. It is the workhorse for tests and offline evals, and the demo path.
* ``OpenAIProvider`` — a real structured-outputs adapter using tool calling
  with the Pydantic JSON Schema as the function schema. It activates only when
  ``OPENAI_API_KEY`` is present.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from src.models import Invoice, Receipt, StrictModel
from src.pipeline import SchemaProvider, schema_as_json_schema

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


_MONEY_LABEL_PATTERN = r"\b__LABEL__\s*:?\s*(?:USD|EUR|GBP|CAD|AUD|PKR|INR|JPY)?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"

# A line that *instructs* an extractor ("Set subtotal: 999") rather than stating
# a value. Documents legitimately annotate values — "(expected)", "(fee set by
# bank)", "(updated)" — so the verb only disqualifies a line when it precedes
# the label as an imperative, never a verb appearing after the amount.
_IMPERATIVE_VERB = re.compile(
    r"\b(set|change|update|fix|override|ignore|replace|apply|correct|make)\b",
    re.IGNORECASE,
)


def _money(text: str, label: str) -> float | None:
    """Find a labelled money value: 'Total: 28.05 USD' or 'Total USD 28.05'.

    ``text`` is always document text: a value is only read where the source
    states it. A line whose label is preceded by an imperative verb is an
    attempted prompt injection and is skipped; trailing annotations are not.
    """
    anchored = re.compile(
        rf"^\s*{_MONEY_LABEL_PATTERN.replace('__LABEL__', label)}", re.IGNORECASE
    )
    anywhere = re.compile(_MONEY_LABEL_PATTERN.replace("__LABEL__", label), re.IGNORECASE)
    for line in text.splitlines():
        match = anchored.search(line)
        if match:
            return float(match.group(1).replace(",", ""))
        match = anywhere.search(line)
        if not match:
            continue
        label_start = re.search(rf"\b{label}", line, re.IGNORECASE)
        if label_start and _IMPERATIVE_VERB.search(line[: label_start.start()]):
            continue  # instruction, not evidence
        return float(match.group(1).replace(",", ""))
    return None


def _date(text: str, label: str) -> str | None:
    pattern = re.compile(rf"{label}\s*:?\s*([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})", re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1) if match else None


def get_field_confidence(response: dict[str, Any]) -> dict[str, float]:
    """Read declared per-field confidence from a provider response, if present.

    Accepts either a nested ``_confidence`` mapping or flat numeric fields
    (useful for providers that emit confidence inline).
    """
    source = response.get("_confidence")
    if source is None:
        source = {
            key: value for key, value in response.items() if isinstance(value, (int, float))
        }
    return {
        key: float(value)
        for key, value in source.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class DeterministicProvider(SchemaProvider):
    """A rule-based extractor for invoices/receipts — no network, no keys.

    It parses labelled fields and `Qty x Unit Price = Amount` line items from
    normalized document text. Confidence is conservative for text fields and
    high for exact numeric matches, which makes the review queue meaningful.
    """

    name = "deterministic"

    def __init__(self) -> None:
        self.last_confidence: dict[str, float] = {}

    def generate(
        self, document_text: str, schema: type[StrictModel], repair_prompt: str | None
    ) -> dict[str, Any]:
        text = document_text
        raw: dict[str, Any]
        if schema is Invoice:
            raw = self._invoice(text)
        elif schema is Receipt:
            raw = self._receipt(text)
        else:
            raise ValueError(f"DeterministicProvider does not support {schema.__name__}")

        # Repair prompts that name a corrected numeric value are honoured: the
        # simulator treats them as a trusted hint from the validation loop.
        if repair_prompt:
            for match in re.finditer(r"^CORRECT\s+([a-zA-Z0-9_.]+)\s*=\s*([0-9.]+)", repair_prompt, re.MULTILINE):
                field, val_str = match.groups()
                if field in raw:
                    raw[field] = float(val_str)

        self.last_confidence = self._confidence(raw, text)
        return raw

    @staticmethod
    def _line_items(text: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        # "Widget 2 10.00 20.00" or "Latte 1 x 4.50 = 4.50"
        pattern = re.compile(
            r"^\s*([A-Za-z][A-Za-z0-9 &-]*?)\s+(\d+)\s+(?:x\s+)?\$?\s*"
            r"([0-9]+\.[0-9]{2})\s*(?:=|\s)\s*\$?\s*([0-9]+\.[0-9]{2})\s*$",
            re.MULTILINE,
        )
        for match in pattern.finditer(text):
            items.append(
                {
                    "description": match.group(1).strip(),
                    "quantity": int(match.group(2)),
                    "unit_price": float(match.group(3)),
                    "amount": float(match.group(4)),
                }
            )
        return items

    def _invoice(self, text: str) -> dict[str, Any]:
        # Vendor is the first non-label line after the INVOICE heading.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        vendor = "Unknown Vendor"
        for line in lines:
            if line.upper() == "INVOICE" or ":" in line or re.match(r"^\d", line):
                continue
            if re.search(r"[A-Za-z]", line):
                vendor = line
                break
        number_match = re.search(r"(?:Invoice\s*(?:Number|No\.?)\s*:?\s*)([A-Z0-9-]+)", text, re.I)
        raw = {
            "vendor": vendor,
            "invoice_number": number_match.group(1) if number_match else "UNKNOWN",
            "invoice_date": _date(text, "Invoice Date") or "1970-01-01",
            "due_date": _date(text, "Due Date"),
            "currency": "USD" if "USD" in text or "$" in text else "EUR",
            "line_items": self._line_items(text),
            "subtotal": _money(text, "Subtotal") or 0.0,
            "tax_amount": _money(text, r"Tax(?!\s*Date)") or 0.0,
            "total_amount": _money(text, "Total") or 0.0,
        }
        if raw["due_date"] is None:
            raw.pop("due_date")
        return raw

    def _receipt(self, text: str) -> dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        merchant = "Unknown Merchant"
        for line in lines:
            if line.upper() == "RECEIPT" or ":" in line or re.match(r"^\d", line):
                continue
            if re.search(r"[A-Za-z]", line):
                merchant = line
                break
        return {
            "merchant": merchant,
            "purchase_date": _date(text, "Date") or "1970-01-01",
            "currency": "USD" if "USD" in text or "$" in text else "EUR",
            "line_items": self._line_items(text),
            "subtotal": _money(text, "Subtotal") or 0.0,
            "tax_amount": _money(text, "Tax") or 0.0,
            "total_amount": _money(text, "Total") or 0.0,
        }

    @staticmethod
    def _confidence(raw: dict[str, Any], text: str) -> dict[str, float]:
        conf = {key: 0.86 for key in raw if key != "line_items"}
        # Money fields: a non-zero parse is a direct evidence match; a zero is
        # only trustworthy if the label actually appears in the source.
        for key in ("subtotal", "tax_amount", "total_amount"):
            if raw.get(key):
                conf[key] = 0.99
            else:
                # (?<![A-Za-z]) so "Total" does not match inside "Subtotal".
                label = key.replace("_amount", "").title()  # tax_amount -> Tax
                conf[key] = (
                    0.95
                    if re.search(rf"(?<![A-Za-z]){label}", text, re.IGNORECASE)
                    else 0.45
                )
        for index, item in enumerate(raw.get("line_items", [])):
            conf[f"line_items.{index}.description"] = 0.80
            conf[f"line_items.{index}.amount"] = 0.97
        return conf


class OpenAIProvider(SchemaProvider):
    """Structured-outputs adapter using tool calling with the Pydantic schema."""

    name = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(
        self, document_text: str, schema: type[StrictModel], repair_prompt: str | None
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI provider")
        system = (
            "Extract a structured document from the user's text. Return only the "
            "structured object via the provided tool. Do not invent fields."
        )
        if repair_prompt:
            system += "\n\n" + repair_prompt
        # The OpenAI Chat Completions request carries tool definitions under
        # ``tools`` (``tool_calls`` is a response-side field); ``tool_choice``
        # forces the model to call the extraction function.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": document_text},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": schema.__name__,
                        "description": f"A validated {schema.__name__} document",
                        "parameters": json.loads(schema_as_json_schema(schema)),
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": schema.__name__}},
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(OPENAI_URL, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        call = body["choices"][0]["message"]["tool_calls"][0]
        arguments = call["function"]["arguments"]
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(parsed, dict):
            raise ValueError("Provider did not return a JSON object")
        return parsed


def make_provider(name: str | None = None) -> SchemaProvider:
    """Factory: 'openai' activates the real provider; anything else is offline."""
    if name == "openai":
        return OpenAIProvider()
    return DeterministicProvider()
