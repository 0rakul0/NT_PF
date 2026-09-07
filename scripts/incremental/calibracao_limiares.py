"""Calibração pós-decisão dos limiares de confiança da WNN.

As tags ``x2`` são consultadas apenas depois de um lote ter sido classificado.
O arquivo produzido é, portanto, uma configuração para lotes futuros e nunca
uma entrada da retina ou da decisão que originou a métrica.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


DEFAULT_MIN_REFERENCES = 10
DEFAULT_STEP = 0.02
TARGET_PRECISION = 0.80
# A conservative precision floor keeps the accepted decisions reliable; the
# lower recall target opens coverage only when that condition is already met.
TARGET_RECALL = 0.30
COMMON_BOUNDS = (0.35, 0.70)
ORGANIZED_CRIME_BOUNDS = (0.60, 0.80)


def _labels(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return _labels(parsed)


def load_active_thresholds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = payload.get("active_thresholds", {})
    return {str(label): float(value) for label, value in raw.items()}


def _bounds(label: str) -> tuple[float, float]:
    return ORGANIZED_CRIME_BOUNDS if label == "crime_organizado" else COMMON_BOUNDS


def calibrate_after_batch(
    rows: list[dict[str, object]],
    path: Path,
    iteration: int,
    default_threshold: float,
    enabled: bool = True,
    min_references: int = DEFAULT_MIN_REFERENCES,
    step: float = DEFAULT_STEP,
    base_thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Persist class-specific thresholds to be used from the next batch.

    A class only changes after enough x2 references. Precision protects against
    false positives; recall unlocks a small decrease only when precision is
    already high. Every update is capped and recorded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_active_thresholds(path)
    if not enabled:
        return {"enabled": False, "iteration": iteration, "changes": [], "active_thresholds": existing}

    labels = sorted(
        {
            label
            for row in rows
            for label in [*_labels(row.get("x2_target_labels")), str(row.get("wnn_top_label", "") or "")]
            if label
        }
    )
    changes: list[dict[str, object]] = []
    next_thresholds = dict(existing)
    for label in labels:
        references = [row for row in rows if label in _labels(row.get("x2_target_labels"))]
        predicted = [
            row
            for row in rows
            if bool(row.get("wnn_accepted")) and str(row.get("wnn_top_label", "") or "") == label
        ]
        true_positive = sum(1 for row in predicted if label in _labels(row.get("x2_target_labels")))
        reference_count = len(references)
        predicted_count = len(predicted)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / reference_count if reference_count else 0.0
        previous = float(existing.get(label, (base_thresholds or {}).get(label, default_threshold)))
        action = "mantido"
        candidate = previous
        if reference_count >= min_references:
            if predicted_count == 0 or (precision >= TARGET_PRECISION and recall < TARGET_RECALL):
                candidate = previous - step
                action = "reduzido_para_aumentar_revocacao"
            elif predicted_count > 0 and precision < TARGET_PRECISION:
                candidate = previous + step
                action = "elevado_para_proteger_precisao"
        else:
            action = "mantido_amostra_insuficiente"
        lower, upper = _bounds(label)
        current = round(min(upper, max(lower, candidate)), 4)
        next_thresholds[label] = current
        changes.append(
            {
                "label": label,
                "references": reference_count,
                "predicted": predicted_count,
                "true_positive": true_positive,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "previous_threshold": round(previous, 4),
                "new_threshold": current,
                "action": action,
                "bounds": [lower, upper],
            }
        )

    previous_history: list[object] = []
    if path.exists():
        try:
            raw_history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
            previous_history = raw_history if isinstance(raw_history, list) else []
        except (OSError, json.JSONDecodeError):
            previous_history = []
    payload = {
        "policy": "x2_pos_decisao; min10_referencias; precisao_ge80; revocacao_lt30_para_reduzir; efeito_a_partir_do_lote_seguinte; passo_limitado",
        "last_iteration": iteration,
        "min_references": min_references,
        "step": step,
        "active_thresholds": dict(sorted(next_thresholds.items())),
        "history": [*previous_history, {"iteration": iteration, "changes": changes}],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"enabled": True, "iteration": iteration, "changes": changes, "active_thresholds": payload["active_thresholds"]}
