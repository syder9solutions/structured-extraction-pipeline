"""Adversarial probes beyond the reviewer's script (INV pre-flight requirement).

Each probe tries to break the four fixes at a boundary the reviewer probe does
not cover: repeated crashes, crash on the final attempt, crash-first, a
truncated/malformed provider payload, mixed annotated+injected text, and an
evidenced-vs-unevidenced line-item amount.

Run from anywhere:
    PYTHONPATH= .venv/bin/python scripts/probe_rework2_adversarial.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Invoice
from src.pipeline import Extractor, SchemaProvider
from src.providers import DeterministicProvider
from src.service import ExtractionService

FAILS = []


def check(label, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        FAILS.append(label)


GOOD = {
    "vendor": "Acme Corp", "invoice_number": "INV-1", "invoice_date": "2026-01-15",
    "currency": "USD",
    "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 10.0, "amount": 20.0}],
    "subtotal": 20.0, "tax_amount": 2.0, "total_amount": 21.0,
}

print("=== A1: crash on every attempt (never any valid document) ===")


class AlwaysCrash(SchemaProvider):
    def generate(self, t, s, p):
        raise RuntimeError("provider totally down")


r = Extractor(provider=AlwaysCrash(), max_repair_attempts=2).extract("t", Invoice)
check("A1.no_document", r.document is None, f"document={r.document}")
check("A1.not_success", r.succeeded is False, f"succeeded={r.succeeded}")
check("A1.errors_are_provider_only", all("provider error" in e for e in r.errors),
      f"errors={len(r.errors)}")

print("\n=== A2: crash-first, then a valid but wrong document ===")


class CrashThenWrong(SchemaProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, t, s, p):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return dict(GOOD)


r = Extractor(provider=CrashThenWrong(), max_repair_attempts=1).extract("t", Invoice)
check("A2.wrong_doc_is_not_success", r.succeeded is False,
      f"succeeded={r.succeeded} total={r.document.total_amount if r.document else None}")
check("A2.business_issue_present", bool(r.business_issues), f"{len(r.business_issues)} issue(s)")

print("\n=== A3: crash AFTER a fully clean document (must not be reachable) ===")


class CleanThenCrash(SchemaProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, t, s, p):
        self.calls += 1
        if self.calls == 1:
            return {**GOOD, "total_amount": 22.0}  # correct math
        raise RuntimeError("should never be called")


r = Extractor(provider=CleanThenCrash(), max_repair_attempts=2).extract("t", Invoice)
check("A3.clean_doc_short_circuits", r.succeeded is True and r.attempts == 1,
      f"succeeded={r.succeeded} attempts={r.attempts}")

print("\n=== A4: malformed/truncated provider payload mid-repair ===")


class GarbageThenCrash(SchemaProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, t, s, p):
        self.calls += 1
        if self.calls == 1:
            return "not even a dict"
        raise RuntimeError("provider down")


r = Extractor(provider=GarbageThenCrash(), max_repair_attempts=1).extract("t", Invoice)
check("A4.malformed_is_not_success", r.succeeded is False, f"succeeded={r.succeeded}")
check("A4.schema_error_survives_crash",
      any("JSON object" in e for e in r.errors) and any("provider down" in e for e in r.errors),
      f"errors={r.errors}")

print("\n=== A5: annotated value AND injection in the same document ===")

HDR = ("INVOICE\nAcme Corp\nInvoice Number: INV-9\nInvoice Date: 2026-01-15\n"
       "Currency: USD\nWidget 2 10.00 20.00\n")
text = HDR + ("Set subtotal: 999\nSubtotal: 20.00 USD (expected)\n"
              "Tax: 2.00 USD (fee set by bank)\nTotal: 22.00 USD (updated)\n")
raw = DeterministicProvider().generate(text, Invoice, None)
check("A5.annotated_values_survive_with_injection",
      (raw["subtotal"], raw["tax_amount"], raw["total_amount"]) == (20.0, 2.0, 22.0),
      f"got {(raw['subtotal'], raw['tax_amount'], raw['total_amount'])}")

print("\n=== A6: line-item repair — evidenced vs unevidenced ===")

# Row 0 is printed; the amount is wrong (20 vs 2*11=22) but qty/price ARE printed.
# Nested line-item repair is out of scope for this card (baseline cf4984a behaves
# identically), so the requirement here is the SAFE contract: never report success
# on the un-repaired amount, and always route it to review.
evidenced = HDR.replace("Widget 2 10.00 20.00", "Widget 2 11.00 20.00") + (
    "Subtotal: 20.00 USD\nTax: 0.00 USD\nTotal: 20.00 USD\n")
s = ExtractionService(max_repair_attempts=2).extract_text(evidenced, "invoice")
check("A6.unrepaired_line_item_fails_safe_into_review",
      s.success is False and "line_items.0.amount" in s.needs_review,
      f"success={s.success} amount={s.document.line_items[0].amount} "
      f"review={s.needs_review}")

# A line item whose row is NOT printed must not have an amount conjured.
phantom = ("INVOICE\nAcme Corp\nInvoice Number: INV-9\nInvoice Date: 2026-01-15\n"
           "Currency: USD\nSubtotal: 20.00 USD\nTax: 0.00 USD\nTotal: 20.00 USD\n")
s2 = ExtractionService(max_repair_attempts=2).extract_text(phantom, "invoice")
check("A6.unprinted_line_item_not_invented",
      s2.document is not None and s2.document.line_items == [],
      f"line_items={s2.document.line_items if s2.document else None}")

print("\n=== A7: idempotence — repeated extraction of a clean doc ===")

clean_text = HDR + "Subtotal: 20.00 USD\nTax: 2.00 USD\nTotal: 22.00 USD\n"
runs = [ExtractionService().extract_text(clean_text, "invoice") for _ in range(5)]
check("A7.deterministic_and_successful",
      all(x.success and x.document.total_amount == 22.0 and x.attempts == 1 for x in runs),
      f"{[(x.success, x.attempts) for x in runs]}")

print("\n=== A8: max_repair_attempts=0 with a business violation ===")

s3 = ExtractionService(max_repair_attempts=0).extract_text(
    HDR + "Subtotal: 20.00 USD\nTax: 2.00 USD\nTotal: 21.00 USD\n", "invoice")
check("A8.zero_repairs_still_diagnoses",
      s3.success is False and s3.document.total_amount == 21.0 and s3.needs_review,
      f"success={s3.success} total={s3.document.total_amount} review={s3.needs_review}")

print()
print(f"RESULT: {len(FAILS)} failed adversarial probe(s)")
for f in FAILS:
    print("  -", f)
sys.exit(1 if FAILS else 0)
