"""Per-field confidence scoring and human-in-the-loop routing.

An extraction pipeline that cannot say "I am unsure" is not trustworthy. The
extractor attaches a declared confidence per field (0..1); this module merges
those with a conservative default for undeclared fields, computes an overall
score, and routes low-confidence fields to a review queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class ConfidenceReport:
    fields: dict[str, float] = field(default_factory=dict)
    needs_review: list[str] = field(default_factory=list)
    overall: float = 1.0

    def as_dict(self) -> dict:
        return {
            "fields": dict(self.fields),
            "needs_review": list(self.needs_review),
            "overall": self.overall,
        }


def _flatten_keys(payload: dict, prefix: str = "") -> list[str]:
    """Return dotted paths for every leaf field in an extracted document."""
    keys: list[str] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(_flatten_keys(value, path))
        elif isinstance(value, list):
            if not value:
                keys.append(path)
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    keys.extend(_flatten_keys(item, f"{path}.{index}"))
                else:
                    keys.append(f"{path}.{index}")
        else:
            keys.append(path)
    return keys


def field_confidence(payload: dict, declared: dict[str, float] | None = None) -> dict[str, float]:
    """Merge declared per-field confidences with the default for all other fields."""
    declared = declared or {}
    return {
        key: float(declared.get(key, DEFAULT_CONFIDENCE)) for key in _flatten_keys(payload)
    }


def score_document(
    payload: dict, declared: dict[str, float] | None = None, threshold: float = 0.7
) -> ConfidenceReport:
    """Score every field, flag those below ``threshold`` for human review."""
    fields = field_confidence(payload, declared)
    needs_review = sorted(key for key, value in fields.items() if value < threshold)
    overall = sum(fields.values()) / len(fields) if fields else 1.0
    return ConfidenceReport(fields=fields, needs_review=needs_review, overall=round(overall, 4))
