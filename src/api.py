"""FastAPI HTTP transport for the extraction service.

All inputs are strict Pydantic models: unknown fields, wrong enums, and empty
text are rejected at the boundary with a 422 before any extraction runs.
"""

from __future__ import annotations

from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.models import DOCUMENT_MODELS
from src.service import ExtractionService

app = FastAPI(
    title="Structured Extraction Pipeline",
    version="1.0.0",
    description="Schema-locked invoice/receipt extraction with validation + repair.",
)


class DocumentType(str, Enum):
    invoice = "invoice"
    receipt = "receipt"


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="Document text to extract")
    document_type: DocumentType = Field(description="Document schema to use")


class ExtractResponse(BaseModel):
    success: bool
    document: dict | None = None
    attempts: int
    needs_review: list[str]
    hallucinated_fields: list[str]
    errors: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/schema/{document_type}")
def get_schema(document_type: str) -> dict:
    if document_type not in DOCUMENT_MODELS:
        raise HTTPException(status_code=404, detail="Unknown document type")
    return DOCUMENT_MODELS[document_type].model_json_schema()


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    service = ExtractionService()
    result = service.extract_text(request.text, request.document_type.value)
    return ExtractResponse(
        success=result.success,
        document=result.document.model_dump(mode="json") if result.document else None,
        attempts=result.attempts,
        needs_review=result.needs_review,
        hallucinated_fields=result.hallucinated_fields,
        errors=result.errors,
    )
