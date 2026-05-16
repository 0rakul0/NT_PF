from __future__ import annotations

import pandas as pd

from scripts.incremental.common import (
    DOCS_JSONL,
    RESERVE_CSV,
    RUN_DIR,
    SAMPLE_CSV,
    RunConfig,
    append_event,
    load_docs,
    split_docs,
    temporal_bucket,
    write_json,
    write_jsonl,
)


def run(config: RunConfig) -> dict[str, object]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_docs(config.max_docs)
    sample, reserve = split_docs(docs, config.sample_fraction, config.seed, config.temporal_strata)
    write_jsonl(DOCS_JSONL, docs)
    sample_df = pd.DataFrame(
        {
            "arquivo": [doc["arquivo"] for doc in sample],
            "data_publicacao": [doc.get("parsed", {}).get("data_publicacao", "") for doc in sample],
            "estrato_temporal": [temporal_bucket(doc, config.temporal_strata) for doc in sample],
        }
    )
    reserve_df = pd.DataFrame(
        {
            "arquivo": [doc["arquivo"] for doc in reserve],
            "data_publicacao": [doc.get("parsed", {}).get("data_publicacao", "") for doc in reserve],
            "estrato_temporal": [temporal_bucket(doc, config.temporal_strata) for doc in reserve],
        }
    )
    sample_df.to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
    reserve_df.to_csv(RESERVE_CSV, index=False, encoding="utf-8-sig")
    strata_counts = sample_df["estrato_temporal"].value_counts().sort_index().to_dict() if not sample_df.empty else {}
    result = {
        "stage": "amostragem",
        "base_docs": len(docs),
        "sample_docs": len(sample),
        "reserve_docs": len(reserve),
        "sample_fraction": config.sample_fraction,
        "temporal_strata": config.temporal_strata,
        "sample_strata_counts": strata_counts,
        "docs_jsonl": str(DOCS_JSONL),
        "sample_csv": str(SAMPLE_CSV),
        "reserve_csv": str(RESERVE_CSV),
    }
    write_json(RUN_DIR / "amostragem_result.json", result)
    append_event(result)
    return result


def main() -> None:
    print(write_json(RUN_DIR / "amostragem_result.json", run(RunConfig())))


if __name__ == "__main__":
    main()
