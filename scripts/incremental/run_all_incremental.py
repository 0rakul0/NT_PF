from __future__ import annotations

import json

from scripts.incremental import agente1_temas, agente2_discriminadores, amostragem, clusterizacao_inicial, processar_lotes, relatorios, similaridade_cosseno
from scripts.incremental.common import LINGUISTIC_PREPROCESSING_JSON, read_json
from scripts.incremental.common import (
    CLUSTER_SUMMARY_CSV,
    RESERVE_CSV,
    RUN_DIR,
    RUN_MANIFEST_JSON,
    THEMES_JSON,
    WNN_FEATURE_BANK_PATH,
    RunConfig,
    append_event,
    read_json,
    reset_outputs,
    snapshot_existing_run,
    write_json,
)
from scripts.avaliar_crimes_por_tags import run_from_incremental_batches


def _foundation_from_artifacts(config: RunConfig, snapshot: dict[str, object] | None = None) -> dict[str, object]:
    """Build the report manifest without changing the frozen foundation."""
    sampling = read_json(RUN_DIR / "amostragem_result.json")
    clusters = read_json(RUN_DIR / "clusterizacao_result.json")
    themes = read_json(RUN_DIR / "agente1_result.json")
    cosine = read_json(RUN_DIR / "similaridade_cosseno_result.json")
    agent2 = read_json(RUN_DIR / "agente2_result.json")
    return {
        "base_docs": sampling["base_docs"],
        "sample_fraction": sampling["sample_fraction"],
        "sample_docs": sampling["sample_docs"],
        "reserve_docs": sampling["reserve_docs"],
        "clusters_total": clusters["clusters_total"],
        "noise_clusters": clusters["noise_clusters"],
        "themes_accepted": themes["themes_accepted"],
        "cosine_profiles": cosine["themes_profiled"],
        "wnn_discriminators": agent2.get("wnn_discriminators", 0),
        "agent2_discriminator_bank": agent2["wnn_feature_bank"],
        "wnn_feature_bank": str(WNN_FEATURE_BANK_PATH),
        "config": config.__dict__,
        "previous_run_snapshot": snapshot or {"created": False},
    }


def has_resume_checkpoint() -> bool:
    """A resume must reuse exactly the original reserve and WNN memory."""
    required = (
        RESERVE_CSV,
        WNN_FEATURE_BANK_PATH,
        RUN_DIR / "amostragem_result.json",
        RUN_DIR / "clusterizacao_result.json",
        RUN_DIR / "agente1_result.json",
        RUN_DIR / "similaridade_cosseno_result.json",
        RUN_DIR / "agente2_result.json",
        THEMES_JSON,
        CLUSTER_SUMMARY_CSV,
    )
    return all(path.exists() for path in required)


def run(config: RunConfig | None = None) -> dict[str, object]:
    config = config or RunConfig()
    snapshot = snapshot_existing_run("auto_before_reset") if config.reset and config.preserve_previous_run else {"created": False}
    deleted = reset_outputs() if config.reset else []
    append_event({"stage": "reset", "deleted": deleted, "snapshot": snapshot, "config": config.__dict__})

    sampling = amostragem.run(config)
    clusters = clusterizacao_inicial.run(config)
    themes = agente1_temas.run(config)
    cosine = similaridade_cosseno.run(config)
    agent2 = agente2_discriminadores.run(config)
    batches = processar_lotes.run(config)
    tag_evaluation = run_from_incremental_batches()

    foundation = _foundation_from_artifacts(config, snapshot)
    foundation["tag_evaluation"] = tag_evaluation
    write_json(RUN_MANIFEST_JSON, foundation)
    reports = relatorios.run(foundation)
    result = {
        "sampling": sampling,
        "clusters": clusters,
        "themes": themes,
        "cosine": cosine,
        "linguistic_preprocessing": read_json(LINGUISTIC_PREPROCESSING_JSON) if LINGUISTIC_PREPROCESSING_JSON.exists() else {},
        "agent2": agent2,
        "batches": batches,
        "tag_evaluation": tag_evaluation,
        "reports": reports,
    }
    append_event({"stage": "run_all_incremental", "result": result})
    return result


def resume(config: RunConfig) -> dict[str, object]:
    """Continue an interrupted batch run without rebuilding its foundation.

    The reserve CSV, theme taxonomy and WNN bank are the checkpoint.  Recreating
    any of them would invalidate the batch iteration numbers and learned rules.
    """
    if config.reset:
        raise ValueError("A continuacao exige RunConfig(reset=False).")
    if not has_resume_checkpoint():
        raise FileNotFoundError(
            "Nao ha uma fundacao incremental completa para continuar. "
            "Execute uma rodada nova com PF_RESUME_RUN=false."
        )

    append_event({"stage": "resume_incremental", "config": config.__dict__})
    batches = processar_lotes.run(config)
    tag_evaluation = run_from_incremental_batches()
    foundation = _foundation_from_artifacts(config)
    foundation["tag_evaluation"] = tag_evaluation
    foundation["resumed"] = True
    write_json(RUN_MANIFEST_JSON, foundation)
    reports = relatorios.run(foundation)
    result = {
        "resumed": True,
        "batches": batches,
        "tag_evaluation": tag_evaluation,
        "reports": reports,
    }
    append_event({"stage": "resume_incremental_complete", "result": result})
    return result


def main() -> None:
    result = run(RunConfig())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
