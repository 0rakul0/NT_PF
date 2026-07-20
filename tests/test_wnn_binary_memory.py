from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.pf_wnn_classifier import (
    binary_memory_for_text,
    classify_with_wnn,
    suggest_discriminator_rules_from_review,
    sync_feature_memory,
)
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse


class WNNBinaryMemoryTests(unittest.TestCase):
    def test_sync_memory_keeps_positions_and_deduplicates_tokens(self) -> None:
        payload = {
            "discriminators": [
                {"id": "d1", "label": "trafico_drogas", "tokens": ["trafico", "drogas"]},
                {"id": "d2", "label": "crime_organizado", "tokens": ["trafico", "organizacao", "criminosa"]},
            ]
        }

        sync_feature_memory(payload)

        memory = payload["memory_vocab"]
        self.assertEqual(memory["tokens"], ["trafico", "droga", "organizacao", "criminosa"])
        self.assertEqual(payload["discriminators"][0]["memory_positions"], [0, 1])
        self.assertEqual(payload["discriminators"][1]["memory_positions"], [0, 2, 3])

    def test_binary_vector_uses_existing_positions(self) -> None:
        payload = {
            "memory_vocab": {"version": 1, "tokens": ["trafico", "drogas", "organizacao"]},
            "discriminators": [],
        }
        state = binary_memory_for_text("Apreensao de drogas durante apuracao de organizacao criminosa.", payload)

        self.assertEqual(state["binary"], "011")
        self.assertEqual(state["active_positions"], [1, 2])

    def test_new_token_extends_memory_on_the_right(self) -> None:
        payload = {
            "memory_vocab": {"version": 1, "tokens": ["trafico", "drogas"]},
            "discriminators": [
                {"id": "d1", "label": "trafico_drogas", "tokens": ["trafico", "drogas"]},
            ],
        }
        before = binary_memory_for_text("trafico de drogas", payload)

        payload["discriminators"].append(
            {"id": "d2", "label": "crimes_ambientais", "tokens": ["garimpo", "ilegal"]}
        )
        sync_feature_memory(payload)
        after = binary_memory_for_text("trafico de drogas", payload)

        self.assertEqual(before["binary"], "11")
        self.assertEqual(after["binary"], "1100")

    def test_wnn_separates_crime_from_modus_operandi(self) -> None:
        payload = {
            "version": 1,
            "source": "test",
            "discriminator_count": 3,
            "labels": ["roubo", "arma_fogo"],
            "themes": {},
            "discriminators": [
                {
                    "id": "crime-1",
                    "kind": "crime",
                    "label": "roubo",
                    "tokens": ["roubo", "celular"],
                    "weight": 1.2,
                    "source": "test",
                },
                {
                    "id": "crime-2",
                    "kind": "crime",
                    "label": "roubo",
                    "tokens": ["subtracao", "bem"],
                    "weight": 1.0,
                    "source": "test",
                },
                {
                    "id": "modus-1",
                    "kind": "modus",
                    "label": "arma_fogo",
                    "tokens": ["arma", "fogo"],
                    "weight": 1.1,
                    "source": "test",
                },
            ],
            "memories": {},
        }
        sync_feature_memory(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "Roubo com subtracao de celular e bem mediante uso de arma de fogo.",
                bank_path,
                confidence_threshold=0.10,
                margin_threshold=0.0,
                min_active_discriminators=1,
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "accepted_crime_with_modus")
        self.assertEqual(result.top_label, "roubo")
        self.assertEqual(result.inference.crimes_mais_presentes, ["roubo"])
        self.assertEqual(result.inference.modus_operandi, ["arma_fogo"])
        self.assertTrue(result.crime_autonomous)
        self.assertGreater(result.crime_confidence, 0.0)
        self.assertGreater(result.modus_confidence, 0.0)

    def test_residual_learning_suggests_crime_and_modus_discriminators(self) -> None:
        review = ResidualReviewAgentResponse(
            decision="classificar",
            canonical_label="trafico_drogas",
            confidence=0.9,
            evidence_text="Trafico de drogas com armazenamento digital e uso de arma de fogo em via publica.",
            rationale="Caso com crime principal e modo de execucao explicito.",
            resumo_curto="Drogas foram mantidas e negociadas com apoio de arma de fogo.",
            modus_operandi=["arma_fogo", "abordagem_via_publica"],
        )
        doc = {
            "context": "Trafico de drogas com armazenamento digital e uso de arma de fogo em via publica.",
            "body_text": "Trafico de drogas com armazenamento digital e uso de arma de fogo em via publica.",
            "parsed": {},
        }

        suggestions = suggest_discriminator_rules_from_review(doc, review)
        kinds = {(item["kind"], item["label"]) for item in suggestions}

        self.assertIn(("crime", "trafico_drogas"), kinds)
        self.assertIn(("modus", "arma_fogo"), kinds)


if __name__ == "__main__":
    unittest.main()
