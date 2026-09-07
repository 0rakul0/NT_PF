from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.avaliar_crimes_por_tags import _metrics, crime_labels_from_tags, run_from_incremental_batches


class CrimeTagMappingTests(unittest.TestCase):
    def test_maps_criminal_tags_and_discards_metadata(self) -> None:
        labels = crime_labels_from_tags(
            ["Operação PF", "Bahia", "Tráfico internacional de drogas", "Destaque"]
        )
        self.assertEqual(labels, ["trafico_drogas"])

    def test_keeps_multiple_crimes_when_pf_tags_are_multi_label(self) -> None:
        labels = crime_labels_from_tags(["Combate organização criminosa", "Tráfico de drogas e armas"])
        self.assertEqual(labels, ["armas_municoes", "crime_organizado", "trafico_drogas"])

    def test_evaluates_saved_wnn_decisions_without_reclassification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs = root / "docs.jsonl"
            docs.write_text(
                "\n".join(
                    json.dumps(item)
                    for item in [
                        {"arquivo": "correta.md", "x2_tags": ["Tráfico de drogas"]},
                        {"arquivo": "abstencao.md", "x2_tags": ["Contrabando"]},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            lots = root / "lotes"
            lots.mkdir()
            pd.DataFrame(
                [
                    {"arquivo": "correta.md", "wnn_attempted": True, "wnn_accepted": True, "wnn_top_label": "trafico_drogas", "wnn_status": "accepted_crime"},
                    {"arquivo": "abstencao.md", "wnn_attempted": True, "wnn_accepted": False, "wnn_top_label": "", "wnn_status": "abstain_no_score"},
                ]
            ).to_csv(lots / "lote_001_classificacoes.csv", index=False)

            metrics = run_from_incremental_batches(docs, lots, root / "avaliacao")

            self.assertEqual(metrics["documents_with_mapped_crime_tag"], 2)
            self.assertEqual(metrics["correct_predictions"], 1)
            self.assertEqual(metrics["coverage"], 0.5)
            self.assertTrue(Path(metrics["outputs"]["confusion_matrix_csv"]).exists())

    def test_structural_parent_counts_when_emitted_as_secondary_label(self) -> None:
        metrics = _metrics(
            [
                {
                    "gold_labels": ["trafico_drogas", "crime_organizado"],
                    "predicted_label": "trafico_drogas",
                    "predicted_labels": ["trafico_drogas", "crime_organizado"],
                    "accepted": True,
                    "correct": True,
                }
            ]
        )

        organized = metrics["by_crime"]["crime_organizado"]
        self.assertEqual(organized["true_positive"], 1)
        self.assertEqual(organized["precision"], 1.0)
        self.assertEqual(organized["recall"], 1.0)
