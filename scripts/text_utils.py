from __future__ import annotations

import unicodedata

from scripts.pf_llm_models import normalize_slug


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def canonical_label(value: str) -> str:
    return normalize_slug(value)
