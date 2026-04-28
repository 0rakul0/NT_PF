from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

try:
    from pf_regex_classifier import (
        append_learned_rule,
        canonical_label,
        clean_learned_rules_file,
        fold_text,
    )
    from pf_regex_rules import BAD_LEARNED_PATTERN_TERMS, TERM_STOPWORDS
    from project_config import ANALYSIS_DIR
except ModuleNotFoundError:
    from scripts.pf_regex_classifier import (
        append_learned_rule,
        canonical_label,
        clean_learned_rules_file,
        fold_text,
    )
    from scripts.pf_regex_rules import BAD_LEARNED_PATTERN_TERMS, TERM_STOPWORDS
    from scripts.project_config import ANALYSIS_DIR


DEFAULT_CLUSTER_SUMMARY = ANALYSIS_DIR / "resumo_clusters.csv"


CLUSTER_LABEL_HINTS = {
    "pornografia_infantil": ("pornografia", "infantil"),
    "fraude_previdenciaria": ("fraude", "previdencia"),
    "corrupcao_desvio": ("corrupcao", "desvio", "licitacao", "recursos"),
    "contrabando_descaminho": ("contrabando", "descaminho"),
    "trafico_drogas": ("trafico", "drogas", "maconha", "cocaina"),
    "trafico_armas": ("trafico", "armas", "municoes"),
    "crimes_ambientais": ("ambiental", "ambientais", "garimpo", "desmatamento", "madeira"),
    "lavagem_dinheiro": ("lavagem", "dinheiro", "capitais"),
    "abuso_sexual_infantil": ("abuso", "sexual", "infantil", "infantojuvenil"),
    "organizacao_criminosa": ("organizacao", "criminosa", "crime", "organizado"),
    "falsidade_documental": ("falsidade", "documental", "documento", "falso", "moeda", "cedulas"),
}


def split_pipe_cell(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def useful_token(token: str) -> bool:
    return (
        len(token) >= 4
        and not token.isdigit()
        and token not in TERM_STOPWORDS
        and token not in BAD_LEARNED_PATTERN_TERMS
    )


def phrase_tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"\b[a-z0-9]{3,}\b", fold_text(value)) if useful_token(token)]


def label_terms(label: str) -> list[str]:
    canonical = canonical_label(label)
    return [term for term in canonical.split("_") if useful_token(term)]


def cluster_labels_from_row(row: pd.Series) -> tuple[list[str], list[str]]:
    crimes = [canonical_label(item) for item in split_pipe_cell(row.get("top_crimes"))]
    modus = [canonical_label(item) for item in split_pipe_cell(row.get("top_modus"))]

    cluster_text = fold_text(
        " | ".join(
            [
                str(row.get("top_terms", "")),
                str(row.get("top_tags", "")),
                str(row.get("cluster_label", "")),
                str(row.get("sample_titles", "")),
            ]
        )
    )
    for label, hints in CLUSTER_LABEL_HINTS.items():
        if sum(1 for hint in hints if re.search(rf"\b{re.escape(hint)}\w*\b", cluster_text)) >= 2:
            crimes.append(label)
    return sorted(set(crimes)), sorted(set(modus))


def ordered_cluster_terms(row: pd.Series) -> list[str]:
    terms: list[str] = []
    for cell_name in ("top_terms", "top_tags", "cluster_label", "sample_titles"):
        for phrase in split_pipe_cell(row.get(cell_name)):
            for token in phrase_tokens(phrase):
                if token not in terms:
                    terms.append(token)
    return terms


def pattern_from_terms(terms: list[str]) -> str:
    if len(terms) < 2:
        return ""
    return r".{0,240}".join(rf"\b{re.escape(term)}\w*\b" for term in terms[:4])


def cluster_pattern_for_label(label: str, row: pd.Series) -> str:
    terms = ordered_cluster_terms(row)
    wanted = label_terms(label)
    hints = list(CLUSTER_LABEL_HINTS.get(canonical_label(label), ()))
    evidence: list[str] = []

    for term in terms:
        if term in wanted or any(term.startswith(hint[:8]) or hint.startswith(term[:8]) for hint in [*wanted, *hints] if len(hint) >= 4):
            if term not in evidence:
                evidence.append(term)

    return pattern_from_terms(evidence)


def learn_from_clusters(
    cluster_summary: Path | str | None = None,
    min_cluster_size: int | None = None,
    max_labels_per_cluster: int | None = None,
) -> dict[str, int | str]:
    path = Path(cluster_summary or os.getenv("PF_CLUSTER_SUMMARY_CSV", "") or DEFAULT_CLUSTER_SUMMARY)
    min_size_raw = os.getenv("PF_CLUSTER_LEARN_MIN_SIZE", "").strip()
    max_labels_raw = os.getenv("PF_CLUSTER_LEARN_MAX_LABELS", "").strip()
    min_size = min_cluster_size if min_cluster_size is not None else int(min_size_raw) if min_size_raw.isdigit() else 80
    max_labels = max_labels_per_cluster if max_labels_per_cluster is not None else int(max_labels_raw) if max_labels_raw.isdigit() else 4

    df = pd.read_csv(path)
    attempted = 0
    accepted = 0
    for _, row in df.iterrows():
        if int(row.get("size", 0) or 0) < min_size:
            continue
        crimes, modus = cluster_labels_from_row(row)
        for kind, labels in (("crime", crimes[:max_labels]), ("modus", modus[:max_labels])):
            for label in labels:
                pattern = cluster_pattern_for_label(label, row)
                if not pattern:
                    continue
                attempted += 1
                if append_learned_rule(
                    kind=kind,
                    label=label,
                    pattern=pattern,
                    title=f"cluster_{row.get('cluster_id')}:{row.get('cluster_label')}",
                ):
                    accepted += 1

    clean_stats = clean_learned_rules_file()
    return {
        "source": str(path),
        "clusters": len(df),
        "attempted": attempted,
        "accepted": accepted,
        "rules_after_clean": int(clean_stats["after"]),
        "patterns_after_clean": int(clean_stats["after_patterns"]),
    }


def main() -> None:
    print(json.dumps(learn_from_clusters(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
