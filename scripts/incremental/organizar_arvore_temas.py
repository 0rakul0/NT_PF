from __future__ import annotations

import json

from scripts.agentes.agente_organizador_arvore import run
from scripts.incremental.common import RUN_DIR, RunConfig, write_json


def main() -> None:
    result = run(RunConfig(reset=False))
    write_json(RUN_DIR / "agente_organizador_arvore_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
