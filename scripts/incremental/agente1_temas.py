from __future__ import annotations

import pandas as pd

from scripts.agentes.agente1_temas import generate_canonical_themes
from scripts.incremental.common import CLUSTER_SUMMARY_CSV, RUN_DIR, THEMES_JSON, RunConfig, append_event, write_json


def run(config: RunConfig) -> dict[str, object]:
    cluster_summary = pd.read_csv(CLUSTER_SUMMARY_CSV)
    themes = generate_canonical_themes(cluster_summary, config)
    write_json(THEMES_JSON, themes.model_dump())
    result = {
        "stage": "agente1_temas",
        "themes_json": str(THEMES_JSON),
        "themes_accepted": len([theme for theme in themes.themes if theme.decision == "accept"]),
        "quarantined_cluster_ids": themes.quarantined_cluster_ids,
        "discarded_cluster_ids": themes.discarded_cluster_ids,
    }
    write_json(RUN_DIR / "agente1_result.json", result)
    append_event(result)
    return result


def main() -> None:
    print(write_json(RUN_DIR / "agente1_result.json", run(RunConfig())))


if __name__ == "__main__":
    main()
