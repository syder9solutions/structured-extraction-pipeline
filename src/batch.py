"""Batch extraction with throughput and cost accounting.

Failures are isolated: one bad document never aborts the batch. Cost is an
explicit, configurable estimate so it can be swapped for real provider billing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.parsing import parse_document
from src.service import ExtractionService, ServiceResult

DEFAULT_COST_PER_DOC_USD = 0.0005  # offline/deterministic baseline


@dataclass
class BatchResult:
    results: list[ServiceResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    cost_per_doc_usd: float = DEFAULT_COST_PER_DOC_USD

    @property
    def processed(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return self.processed - self.succeeded

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.processed if self.processed else 0.0

    @property
    def docs_per_minute(self) -> float:
        """Measured throughput for this batch.

        A small epsilon only guards against division by zero; it deliberately
        does not impose a 1-second floor, which would silently cap throughput
        at ``processed * 60`` and under-report fast offline runs.
        """
        elapsed = max(self.elapsed_seconds, 1e-9)
        return self.processed / elapsed * 60.0

    @property
    def estimated_cost_usd(self) -> float:
        return self.processed * self.cost_per_doc_usd

    def as_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "docs_per_minute": round(self.docs_per_minute, 2),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


def batch_extract(
    documents: list[tuple[str | Path, str]],
    service: ExtractionService | None = None,
    cost_per_doc_usd: float = DEFAULT_COST_PER_DOC_USD,
) -> BatchResult:
    """Extract a list of ``(text_or_path, document_type)`` pairs.

    A Path-like first element is parsed from disk; a string is treated as
    already-extracted document text.
    """
    service = service or ExtractionService()
    batch = BatchResult(cost_per_doc_usd=cost_per_doc_usd)
    started = time.perf_counter()
    for source, document_type in documents:
        try:
            if isinstance(source, Path):
                text = parse_document(source)
                result = service.extract_text(text, document_type)
            else:
                result = service.extract_text(str(source), document_type)
        except Exception as exc:
            result = ServiceResult(
                document=None, success=False, attempts=0, errors=[str(exc)]
            )
        batch.results.append(result)
    batch.elapsed_seconds = max(time.perf_counter() - started, 1e-9)
    return batch
