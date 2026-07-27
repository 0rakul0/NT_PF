from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.incremental.common import NEW_THEME_CANDIDATES_JSONL, RUN_DIR


OUTPUT = RUN_DIR / "candidatos_compostos_wnn.csv"


def run() -> dict[str, object]:
    """Export WNN multi-discriminator compositions to a reviewable CSV."""
    if not NEW_THEME_CANDIDATES_JSONL.exists():
        pd.DataFrame().to_csv(OUTPUT, index=False, encoding="utf-8-sig")
        return {"output": str(OUTPUT), "candidates": 0}

    records: list[dict[str, object]] = []
    for raw_line in NEW_THEME_CANDIDATES_JSONL.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        label = str(item.get("canonical_label", "") or "")
        summary = str(item.get("resumo_curto", "") or "")
        if not (label.startswith("composto_") or "Composicao multi-discriminador" in summary):
            continue
        markers = item.get("marcadores_secundarios", [])
        records.append(
            {
                "lote": item.get("iteration", ""),
                "arquivo": item.get("arquivo", ""),
                "titulo": item.get("titulo", ""),
                "tema_principal_wnn": item.get("tema_principal", ""),
                "classes_coativadas": " | ".join(str(marker) for marker in markers) if isinstance(markers, list) else str(markers),
                "candidato_composto": label,
                "confianca_wnn": item.get("confidence", ""),
                "evidencias": item.get("evidence_text", ""),
                "justificativa": item.get("rationale", ""),
            }
        )

    table = pd.DataFrame(records)
    if not table.empty:
        table = table.sort_values(["lote", "confianca_wnn"], ascending=[True, False])
    table.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    return {"output": str(OUTPUT), "candidates": len(table)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
