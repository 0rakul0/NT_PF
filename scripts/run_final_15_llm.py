from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.incremental.common import RunConfig
from scripts.incremental.run_all_incremental import run


def main() -> None:
    config = RunConfig(
        sample_fraction=0.15,
        batch_size=10,
        model="llama3.2",
        reset=True,
        max_residual_llm_per_batch=None,
        max_batches=None,
        llm_timeout_seconds=180,
        initial_regex_target_per_theme=30,
        resume_batches=True,
        local_fallback_models=("gemma3n:e2b", "llama3:8b"),
    )
    result = run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
