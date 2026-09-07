from __future__ import annotations

import pandas as pd

from scripts.incremental.common import (
    CLUSTER_ASSIGNMENTS_CSV,
    RUN_DIR,
    SAMPLE_CSV,
    THEMES_JSON,
    RunConfig,
    WNN_FEATURE_BANK_PATH,
    append_event,
    docs_by_manifest,
    read_json,
    write_json,
)
from scripts.pf_wnn_classifier import build_feature_bank


def run(config: RunConfig) -> dict[str, object]:
    themes_payload = read_json(THEMES_JSON)
    sample = docs_by_manifest(SAMPLE_CSV)
    cluster_rows = pd.read_csv(CLUSTER_ASSIGNMENTS_CSV)
    feature_bank = build_feature_bank(
        themes_payload,
        sample,
        cluster_rows,
        WNN_FEATURE_BANK_PATH,
        max_discriminators_per_theme=config.wnn_max_discriminators_per_theme,
    )
    result = {
        "stage": "agente2_discriminadores",
        "wnn_feature_bank": str(WNN_FEATURE_BANK_PATH),
        "wnn_discriminators": feature_bank.get("discriminator_count", 0),
        "wnn_labels": len(feature_bank.get("labels", [])),
        "wnn_crime_discriminators": len(
            [
                item
                for item in feature_bank.get("discriminators", [])
                if isinstance(item, dict) and str(item.get("kind", "crime")) == "crime"
            ]
        ),
        "discriminator_mode": "tokens_only",
        "generalization_policy": "crime_markers_from_texto_noticia_only",
        "retina_input": "x3_texto_noticia",
        "target_policy": "x2_tags_are_evaluation_labels_only",
    }
    write_json(RUN_DIR / "agente2_result.json", result)
    append_event(result)
    return result


def main() -> None:
    print(write_json(RUN_DIR / "agente2_result.json", run(RunConfig())))


if __name__ == "__main__":
    main()
