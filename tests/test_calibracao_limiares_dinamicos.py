from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.incremental.calibracao_limiares import calibrate_after_batch, load_active_thresholds


def row(label: str, accepted: bool, predicted: str = "") -> dict[str, object]:
    return {"x2_target_labels": [label], "wnn_accepted": accepted, "wnn_top_label": predicted}


class DynamicThresholdCalibrationTests(unittest.TestCase):
    def test_high_precision_low_recall_reduces_threshold_for_next_batch(self) -> None:
        rows = [row("trafico_drogas", True, "trafico_drogas")] + [row("trafico_drogas", False) for _ in range(24)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = calibrate_after_batch(rows, Path(tmpdir) / "thresholds.json", 1, 0.50)
        change = next(item for item in result["changes"] if item["label"] == "trafico_drogas")
        self.assertEqual(change["action"], "reduzido_para_aumentar_revocacao")
        self.assertEqual(change["new_threshold"], 0.48)

    def test_high_precision_recall_at_or_above_30_percent_keeps_threshold(self) -> None:
        rows = [row("trafico_drogas", True, "trafico_drogas") for _ in range(8)]
        rows.extend(row("trafico_drogas", False) for _ in range(17))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = calibrate_after_batch(rows, Path(tmpdir) / "thresholds.json", 1, 0.50)
        change = next(item for item in result["changes"] if item["label"] == "trafico_drogas")
        self.assertEqual(change["action"], "mantido")
        self.assertEqual(change["new_threshold"], 0.50)

    def test_low_precision_elevates_threshold(self) -> None:
        rows = [row("trafico_drogas", True, "trafico_drogas") for _ in range(20)]
        rows.extend({"x2_target_labels": ["crimes_ambientais"], "wnn_accepted": True, "wnn_top_label": "trafico_drogas"} for _ in range(10))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = calibrate_after_batch(rows, Path(tmpdir) / "thresholds.json", 1, 0.50)
        change = next(item for item in result["changes"] if item["label"] == "trafico_drogas")
        self.assertEqual(change["action"], "elevado_para_proteger_precisao")
        self.assertEqual(change["new_threshold"], 0.52)

    def test_organized_crime_respects_restrictive_floor(self) -> None:
        rows = [row("crime_organizado", False) for _ in range(25)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "thresholds.json"
            result = calibrate_after_batch(
                rows,
                path,
                1,
                0.50,
                base_thresholds={"crime_organizado": 0.65},
            )
            self.assertEqual(load_active_thresholds(path)["crime_organizado"], 0.63)
        change = next(item for item in result["changes"] if item["label"] == "crime_organizado")
        self.assertGreaterEqual(change["new_threshold"], 0.60)

    def test_insufficient_reference_keeps_previous_threshold(self) -> None:
        rows = [row("trafico_drogas", True, "trafico_drogas") for _ in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = calibrate_after_batch(rows, Path(tmpdir) / "thresholds.json", 1, 0.50)
        change = next(item for item in result["changes"] if item["label"] == "trafico_drogas")
        self.assertEqual(change["action"], "mantido_amostra_insuficiente")
        self.assertEqual(change["new_threshold"], 0.50)


if __name__ == "__main__":
    unittest.main()
