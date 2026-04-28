from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from pf_regex_classifier import clean_learned_rules_file
except ModuleNotFoundError:
    from scripts.pf_regex_classifier import clean_learned_rules_file


def main(path: Path | str | None = None) -> None:
    rules_path = path or os.getenv("PF_REGEX_LEARNED_RULES", "").strip() or None
    stats = clean_learned_rules_file(rules_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
