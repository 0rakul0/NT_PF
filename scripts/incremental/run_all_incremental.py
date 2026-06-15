from __future__ import annotations

import json

from scripts.incremental import agente1_temas, agente2_discriminadores, amostragem, clusterizacao_inicial, processar_lotes, relatorios, similaridade_cosseno
from scripts.incremental.common import RUN_MANIFEST_JSON, WNN_FEATURE_BANK_PATH, RunConfig, append_event, reset_outputs, snapshot_existing_run, write_json


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

    foundation = {
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
        "previous_run_snapshot": snapshot,
    }
    write_json(RUN_MANIFEST_JSON, foundation)
    reports = relatorios.run(foundation)
    result = {
        "sampling": sampling,
        "clusters": clusters,
        "themes": themes,
        "cosine": cosine,
        "agent2": agent2,
        "batches": batches,
        "reports": reports,
    }
    append_event({"stage": "run_all_incremental", "result": result})
    return result


def main() -> None:
    result = run(RunConfig())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
