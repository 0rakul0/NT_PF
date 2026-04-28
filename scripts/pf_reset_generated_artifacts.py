from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from project_config import ANALYSIS_DIR, LLM_METADATA_CSV, LLM_METADATA_JSONL
except ModuleNotFoundError:
    from scripts.project_config import ANALYSIS_DIR, LLM_METADATA_CSV, LLM_METADATA_JSONL


RESET_LLM_METADATA = (
    LLM_METADATA_JSONL,
    LLM_METADATA_CSV,
)

RESET_ANALYSIS_ARTIFACTS = (
    ANALYSIS_DIR / "corpus_enriquecido.csv",
    ANALYSIS_DIR / "vizinhos_semelhantes.csv",
    ANALYSIS_DIR / "pares_recorrentes.csv",
    ANALYSIS_DIR / "resumo_clusters.csv",
    ANALYSIS_DIR / "recorrencia_temporal.csv",
    ANALYSIS_DIR / "crimes_por_ano.csv",
    ANALYSIS_DIR / "modus_operandi_por_ano.csv",
    ANALYSIS_DIR / "series_semanticas.csv",
    ANALYSIS_DIR / "clusters_por_ano.csv",
    ANALYSIS_DIR / "clusters_canonicos.csv",
    ANALYSIS_DIR / "clusters_canonicos_por_ano.csv",
    ANALYSIS_DIR / "recorrencia_temporal_clusters_canonicos.csv",
    ANALYSIS_DIR / "estados_por_ano.csv",
    ANALYSIS_DIR / "estados_por_cluster.csv",
    ANALYSIS_DIR / "analise_qualitativa.md",
)

OPTIONAL_REGEX_RULES = (
    ANALYSIS_DIR / "regex_classifier_rules.json",
)


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "sim"}


def unique_paths(paths: tuple[Path, ...]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def reset_generated_artifacts(
    reset_analysis: bool | None = None,
    reset_regex_rules: bool | None = None,
    dry_run: bool | None = None,
) -> dict[str, object]:
    include_analysis = env_flag("PF_RESET_ANALYSIS_ARTIFACTS") if reset_analysis is None else reset_analysis
    include_regex_rules = env_flag("PF_RESET_REGEX_RULES") if reset_regex_rules is None else reset_regex_rules
    only_preview = env_flag("PF_RESET_DRY_RUN") if dry_run is None else dry_run

    targets = list(RESET_LLM_METADATA)
    if include_analysis:
        targets.extend(RESET_ANALYSIS_ARTIFACTS)
    if include_regex_rules:
        targets.extend(OPTIONAL_REGEX_RULES)

    deleted: list[str] = []
    missing: list[str] = []
    for path in unique_paths(tuple(targets)):
        if not path.exists():
            missing.append(str(path))
            continue
        if not only_preview:
            path.unlink()
        deleted.append(str(path))

    return {
        "dry_run": only_preview,
        "deleted": deleted,
        "missing": missing,
        "preserved_by_default": [str(path) for path in OPTIONAL_REGEX_RULES if not include_regex_rules],
        "tips": {
            "reset_analysis_artifacts": "PF_RESET_ANALYSIS_ARTIFACTS=1",
            "reset_regex_rules": "PF_RESET_REGEX_RULES=1",
            "dry_run": "PF_RESET_DRY_RUN=1",
        },
    }


def main() -> None:
    print(json.dumps(reset_generated_artifacts(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
