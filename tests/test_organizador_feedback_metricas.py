from __future__ import annotations

import json

import pandas as pd

from scripts.agentes.agente_organizador_arvore import _apply_metric_feedback_to_wnn_bank


def test_metric_feedback_reinforces_correct_and_quarantines_repeated_false_positive(tmp_path) -> None:
    bank_path = tmp_path / "wnn_feature_bank.json"
    bank_path.write_text(
        json.dumps(
            {
                "discriminators": [
                    {"id": "right", "label": "trafico_drogas", "tokens": ["trafico", "drogas"], "confirmations": 1, "source": "agent3_learned_discriminator"},
                    {"id": "wrong", "label": "armas_municoes", "tokens": ["arma", "fogo"], "confirmations": 1, "source": "agent3_learned_discriminator"},
                    {"id": "curated", "label": "armas_municoes", "tokens": ["porte", "arma"], "confirmations": 1, "source": "agent2_curated_discriminator"},
                ]
            }
        ),
        encoding="utf-8",
    )
    active_right = "[{'id': 'right', 'label': 'trafico_drogas', 'mask_coverage': 1.0}]"
    active_wrong = "[{'id': 'wrong', 'label': 'armas_municoes', 'mask_coverage': 1.0}, {'id': 'curated', 'label': 'armas_municoes', 'mask_coverage': 1.0}]"
    pd.DataFrame(
        [
            {"wnn_accepted": True, "x2_target_labels": "['trafico_drogas']", "wnn_top_label": "trafico_drogas", "wnn_active_discriminators": active_right},
            {"wnn_accepted": True, "x2_target_labels": "['crimes_contra_criancas']", "wnn_top_label": "armas_municoes", "wnn_active_discriminators": active_wrong},
            {"wnn_accepted": True, "x2_target_labels": "['crimes_ambientais']", "wnn_top_label": "armas_municoes", "wnn_active_discriminators": active_wrong},
        ]
    ).to_csv(tmp_path / "lote_0001_classificacoes.csv", index=False)

    result = _apply_metric_feedback_to_wnn_bank(
        feature_bank_path=bank_path,
        lots_dir=tmp_path,
        snapshots_dir=tmp_path / "snapshots",
        report_path=tmp_path / "feedback.json",
    )

    updated = json.loads(bank_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in updated["discriminators"]}
    assert result["reinforced"] == 1
    assert result["quarantined"] == 1
    assert by_id["right"]["confirmations"] == 2
    assert by_id["wrong"]["quarantined"] is True
    assert "quarantined" not in by_id["curated"]
    assert (tmp_path / "feedback.json").exists()
