from __future__ import annotations

import json

from scripts.incremental.common import RunConfig
from scripts.incremental.run_all_incremental import run


def main() -> None:
    result = run(RunConfig())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
