"""Live HTTP probe of the running API on 127.0.0.1:8097."""

import httpx

base = "http://127.0.0.1:8097"

health = httpx.get(f"{base}/health").json()
print("health:", health)

invoice = (
    "INVOICE\nAcme Corp\nInvoice Number: INV-1001\nInvoice Date: 2026-01-15\n"
    "Currency: USD\nWidget 2 10.00 20.00\nSubtotal: 20.00 USD\nTax: 2.00 USD\nTotal: 22.00 USD"
)
r = httpx.post(f"{base}/extract", json={"text": invoice, "document_type": "invoice"})
body = r.json()
print("extract status:", r.status_code)
print(
    "success:", body["success"],
    "| vendor:", body["document"]["vendor"],
    "| total:", body["document"]["total_amount"],
)

bad = httpx.post(f"{base}/extract", json={"text": invoice, "document_type": "passport"})
print("bad type status:", bad.status_code)

schema = httpx.get(f"{base}/schema/invoice")
print("schema status:", schema.status_code, "| title:", schema.json().get("title"))
