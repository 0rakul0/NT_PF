from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.incremental import processar_lotes
from scripts.incremental.common import RunConfig


def main() -> None:
    config = RunConfig(
        sample_fraction=0.15,
        batch_size=500,
        model="llama3.2",
        reset=False,
        max_residual_llm_per_batch=None,
        max_batches=None,
        llm_timeout_seconds=180,
        resume_batches=True,
        local_fallback_models=("gemma3n:e2b", "llama3:8b"),
    )
    print(processar_lotes.run(config))


if __name__ == "__main__":
    main()
