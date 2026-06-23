from __future__ import annotations

import unittest

from scripts.pf_wnn_classifier import binary_memory_for_text, sync_feature_memory


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
        self.assertEqual(memory["tokens"], ["trafico", "drogas", "organizacao", "criminosa"])
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


if __name__ == "__main__":
    unittest.main()
