"""Add audited refinements for the weakest WNN crime classes without resetting memory."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.avaliar_crimes_por_tags import run as evaluate_by_tags
from scripts.incremental.common import RUN_SNAPSHOTS_DIR, WNN_FEATURE_BANK_PATH
from scripts.pf_wnn_classifier import CURATED_THEME_DISCRIMINATORS, load_feature_bank, sync_feature_memory


TARGET_LABELS = {
    "crime_organizado",
    "contrabando_descaminho",
    "corrupcao_desvio_recursos_publicos",
    "crimes_contra_criancas",
    "crimes_previdenciarios",
    "crimes_sistema_financeiro",
}


def _signature(label: str, tokens: list[object]) -> tuple[str, tuple[str, ...]]:
    normalized = tuple(sorted(str(token).strip().lower() for token in tokens if str(token).strip()))
    return label, normalized


def refine(feature_bank_path: Path = WNN_FEATURE_BANK_PATH) -> dict[str, Any]:
    """Append only missing curated rules and preserve every existing discriminator."""
    payload = load_feature_bank(feature_bank_path)
    if not payload:
        raise FileNotFoundError(f"Banco de discriminadores ausente: {feature_bank_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = RUN_SNAPSHOTS_DIR / f"{timestamp}_before_priority_discriminator_refinement"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = snapshot_dir / feature_bank_path.name
    shutil.copy2(feature_bank_path, snapshot_path)

    discriminators = payload.get("discriminators", [])
    if not isinstance(discriminators, list):
        raise ValueError("Banco de discriminadores invalido")
    existing = {
        _signature(str(item.get("label", "")), item.get("tokens", []))
        for item in discriminators
        if isinstance(item, dict) and isinstance(item.get("tokens", []), list)
    }
    added: list[dict[str, object]] = []
    for label in sorted(TARGET_LABELS):
        for rule in CURATED_THEME_DISCRIMINATORS.get(label, []):
            tokens = list(rule.get("tokens", []))
            signature = _signature(label, tokens)
            if not tokens or signature in existing:
                continue
            name = str(rule.get("name", "regra"))
            safe_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
            item = {
                "id": f"priority_refinement_{label}_{safe_name}",
                "kind": "crime",
                "label": label,
                "name": name,
                "tokens": tokens,
                "weight": float(rule.get("weight", 1.0)),
                "source": "priority_error_analysis_20260824",
                "strength": "strong",
                "confirmations": 4,
                "rationale": "Regra adicionada apos analise das confusoes por tags; exige todos os termos do padrao.",
            }
            discriminators.append(item)
            existing.add(signature)
            added.append(item)

    payload["discriminators"] = discriminators
    sync_feature_memory(payload)
    payload["priority_refinement"] = {
        "created_at": timestamp,
        "targets": sorted(TARGET_LABELS),
        "snapshot": str(snapshot_path),
        "added_discriminators": [item["id"] for item in added],
    }
    feature_bank_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"snapshot": str(snapshot_path), "added": len(added), "discriminators": added}


def main() -> None:
    result = refine()
    # This is a counterfactual reclassification of the unchanged document base;
    # original chronological batch artifacts remain intact for comparison.
    evaluation_dir = WNN_FEATURE_BANK_PATH.parent / "incremental" / "avaliacao_crime_tags" / "refinamento_prioritario"
    result["evaluation"] = evaluate_by_tags(output_dir=evaluation_dir, partition="all")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
