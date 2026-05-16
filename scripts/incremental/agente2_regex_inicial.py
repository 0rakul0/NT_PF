from __future__ import annotations

import pandas as pd

from scripts.agentes.agente2_regex import copy_regex_bank, generate_initial_regex, validate_regex
from scripts.incremental.common import (
    ACTIVE_REGEX_BANK_PATH,
    AGENT2_REGEX_BANK_PATH,
    CLUSTER_ASSIGNMENTS_CSV,
    INITIAL_REGEX_JSON,
    RUN_DIR,
    SAMPLE_CSV,
    THEMES_JSON,
    RunConfig,
    append_event,
    docs_by_manifest,
    read_json,
    write_json,
)


def run(config: RunConfig) -> dict[str, object]:
    ACTIVE_REGEX_BANK_PATH.write_text("[]\n", encoding="utf-8")
    themes_payload = read_json(THEMES_JSON)
    sample = docs_by_manifest(SAMPLE_CSV)
    cluster_rows = pd.read_csv(CLUSTER_ASSIGNMENTS_CSV)
    responses = generate_initial_regex(themes_payload, sample, cluster_rows, config, ACTIVE_REGEX_BANK_PATH)
    copy_regex_bank(ACTIVE_REGEX_BANK_PATH, AGENT2_REGEX_BANK_PATH)
    write_json(INITIAL_REGEX_JSON, responses)
    result = {
        "stage": "agente2_regex_inicial",
        "initial_regex_json": str(INITIAL_REGEX_JSON),
        "agent2_regex_bank": str(AGENT2_REGEX_BANK_PATH),
        "active_regex_bank": str(ACTIVE_REGEX_BANK_PATH),
        "initial_regex_accepted": sum(len(item["accepted_rules"]) for item in responses),
    }
    write_json(RUN_DIR / "agente2_result.json", result)
    append_event(result)
    return result


def main() -> None:
    print(write_json(RUN_DIR / "agente2_result.json", run(RunConfig())))


if __name__ == "__main__":
    main()
