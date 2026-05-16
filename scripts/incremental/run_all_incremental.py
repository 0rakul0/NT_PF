from __future__ import annotations

import json

from scripts.incremental import agente1_temas, agente2_regex_inicial, amostragem, clusterizacao_inicial, processar_lotes, relatorios, similaridade_cosseno
from scripts.incremental.common import ACTIVE_REGEX_BANK_PATH, RUN_MANIFEST_JSON, RunConfig, append_event, reset_outputs, write_json


def run(config: RunConfig | None = None) -> dict[str, object]:
    config = config or RunConfig()
    deleted = reset_outputs() if config.reset else []
    append_event({"stage": "reset", "deleted": deleted, "config": config.__dict__})

    sampling = amostragem.run(config)
    clusters = clusterizacao_inicial.run(config)
    themes = agente1_temas.run(config)
    cosine = similaridade_cosseno.run(config)
    initial_regex = agente2_regex_inicial.run(config)
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
        "initial_regex_accepted": initial_regex["initial_regex_accepted"],
        "agent2_regex_bank": initial_regex["agent2_regex_bank"],
        "active_regex_bank": str(ACTIVE_REGEX_BANK_PATH),
        "config": config.__dict__,
    }
    write_json(RUN_MANIFEST_JSON, foundation)
    reports = relatorios.run(foundation)
    result = {
        "sampling": sampling,
        "clusters": clusters,
        "themes": themes,
        "cosine": cosine,
        "initial_regex": initial_regex,
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
