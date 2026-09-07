from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.pf_wnn_classifier import (
    append_discriminators_from_learned_rules,
    binary_memory_for_text,
    classify_with_wnn,
    mask_discriminators,
    migrate_feature_bank_to_weighted_masks,
    reconstruct_reverse_memory_prototype,
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
        bloom = memory["bloom_filter"]
        self.assertEqual(bloom["item_count"], 4)
        self.assertGreater(bloom["bit_size"], 0)

    def test_new_residual_token_requires_recurrence_before_appending_position(self) -> None:
        payload = {
            "memory_vocab": {"version": 1, "tokens": ["cartao"]},
            "discriminators": [
                {
                    "id": "foundation",
                    "kind": "crime",
                    "label": "crimes_sistema_financeiro",
                    "tokens": ["cartao", "fraude"],
                    "source": "agent1_evidence_term",
                    "confirmations": 1,
                },
                {
                    "id": "residual",
                    "kind": "crime",
                    "label": "crimes_sistema_financeiro",
                    "tokens": ["cartao", "perfurado"],
                    "source": "agent3_learned_discriminator",
                    "confirmations": 1,
                },
            ],
        }

        sync_feature_memory(payload)
        self.assertEqual(payload["memory_vocab"]["tokens"], ["cartao", "fraude"])

        payload["discriminators"][1]["confirmations"] = 2
        sync_feature_memory(payload)
        self.assertEqual(payload["memory_vocab"]["tokens"], ["cartao", "fraude", "perfurado"])
        self.assertEqual(payload["memory_vocab"]["token_to_position"]["cartao"], 0)

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

    def test_bloom_prefilter_requires_exact_vocabulary_lookup_to_activate_bit(self) -> None:
        payload = {
            "discriminators": [
                {"id": "d1", "label": "trafico_drogas", "tokens": ["trafico", "drogas"]},
            ]
        }
        sync_feature_memory(payload)

        state = binary_memory_for_text("Texto sem palavras do vocabulario: xylophone quasar.", payload)

        self.assertTrue(state["bloom_enabled"])
        self.assertEqual(state["active_positions"], [])
        self.assertEqual(state["binary"], "00")

    def test_reverse_memory_reconstructs_tokens_from_stable_positions(self) -> None:
        payload = {
            "memory_vocab": {"version": 3, "tokens": ["trafico", "droga", "apreensao", "arma"]},
            "reverse_memories": {
                "trafico_drogas": {
                    "sample_count": 4,
                    "active_position_counts": {"0": 4, "1": 4, "2": 3, "3": 1},
                    "source_counts": {"foundation_sample": 4},
                    "signatures": ["0 1 2"],
                }
            },
        }

        prototype = reconstruct_reverse_memory_prototype(payload, "trafico_drogas", limit=3)

        self.assertTrue(prototype["available"])
        self.assertEqual(prototype["token_sequence"], ["trafico", "droga", "apreensao"])
        self.assertEqual(prototype["prototype"][0]["position"], 0)
        self.assertEqual(prototype["prototype"][0]["support"], 1.0)

    def test_wnn_classifies_only_the_crime_axis(self) -> None:
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
        self.assertEqual(result.status, "accepted_crime")
        self.assertEqual(result.top_label, "roubo")
        self.assertEqual(result.inference.crimes_mais_presentes, ["roubo"])
        self.assertEqual(result.inference.modus_operandi, [])
        self.assertTrue(result.crime_autonomous)
        self.assertGreater(result.crime_confidence, 0.0)
        self.assertEqual(result.modus_confidence, 0.0)

    def test_wnn_uses_body_for_crime(self) -> None:
        payload = {
            "version": 1,
            "source": "test",
            "discriminators": [
                {"id": "crime-1", "kind": "crime", "label": "roubo", "tokens": ["roubo", "celular"], "weight": 1.2},
                {"id": "modus-1", "kind": "modus", "label": "arma_fogo", "tokens": ["arma", "fogo"], "weight": 1.1},
            ],
            "memories": {},
        }
        sync_feature_memory(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "Os suspeitos utilizaram arma de fogo durante a ação.",
                bank_path,
                confidence_threshold=0.10,
                margin_threshold=0.0,
                min_active_discriminators=1,
                crime_text="A investigação apura roubo de celular.",
                modus_text="Os suspeitos utilizaram arma de fogo durante a ação.",
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.top_label, "roubo")
        self.assertEqual(result.modus_operandi, [])
        self.assertEqual(result.crime_evidence_source, "body")
        self.assertEqual(
            {item["evidence_source"] for item in result.active_discriminators if item["kind"] == "crime"},
            {"body"},
        )

    def test_generic_organized_crime_context_is_not_emitted_as_secondary_label(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {"id": "organized", "kind": "crime", "label": "crime_organizado", "tokens": ["organizacao", "criminosa"], "weight": 1.2},
                {"id": "drugs", "kind": "crime", "label": "trafico_drogas", "tokens": ["trafico", "drogas"], "weight": 1.1},
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "",
                bank_path,
                confidence_threshold=0.10,
                margin_threshold=0.0,
                min_active_discriminators=1,
                crime_text="Organizacao criminosa ligada ao trafico de drogas.",
                modus_text="",
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.top_label, "trafico_drogas")
        self.assertNotIn("crime_organizado", result.inference.marcadores_secundarios)

    def test_incidental_weapon_is_suppressed_when_child_crime_mask_is_complete(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {
                    "id": "child-exploitation",
                    "kind": "crime",
                    "label": "crimes_contra_criancas",
                    "tokens": ["pornografia", "infantil"],
                    "weight": 1.0,
                    "strength": "strong",
                },
                {
                    "id": "incidental-weapon",
                    "kind": "crime",
                    "label": "armas_municoes",
                    "tokens": ["arma", "fogo"],
                    "weight": 1.2,
                    "strength": "strong",
                },
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "Foram encontradas imagens de pornografia infantil e uma arma de fogo.",
                bank_path,
                confidence_threshold=0.10,
                margin_threshold=0.0,
                min_active_discriminators=1,
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.top_label, "crimes_contra_criancas")
        self.assertTrue(
            any(item["reason"] == "incidental_weapon_context_with_protected_domain" for item in result.guard_rejected_discriminators)
        )

    def test_child_crime_uses_calibrated_class_threshold_below_global_threshold(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {
                    "id": "child",
                    "kind": "crime",
                    "label": "crimes_contra_criancas",
                    "tokens": ["abuso", "sexual"],
                    "weight": 1.1,
                    "strength": "strong",
                },
                {
                    "id": "cyber",
                    "kind": "crime",
                    "label": "crimes_ciberneticos",
                    "tokens": ["internet"],
                    "weight": 0.95,
                    "strength": "strong",
                },
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "Abuso sexual investigado na internet.",
                bank_path,
                confidence_threshold=0.50,
                margin_threshold=0.12,
                min_active_discriminators=1,
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.top_label, "crimes_contra_criancas")
        self.assertGreaterEqual(result.confidence, 0.52)

    def test_structural_organized_crime_evidence_is_emitted_as_secondary_label(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {
                    "id": "organized-structural",
                    "kind": "crime",
                    "label": "crime_organizado",
                    "tokens": ["organizacao", "criminosa", "integrantes"],
                    "weight": 1.2,
                    "strength": "strong",
                },
                {"id": "drugs", "kind": "crime", "label": "trafico_drogas", "tokens": ["trafico", "drogas"], "weight": 1.1},
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "",
                bank_path,
                confidence_threshold=0.10,
                margin_threshold=0.0,
                min_active_discriminators=1,
                crime_text="Integrantes de organização criminosa atuavam no tráfico de drogas.",
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.top_label, "trafico_drogas")
        self.assertIn("crime_organizado", result.inference.marcadores_secundarios)

    def test_known_cooccurrence_is_accepted_without_llm_tiebreaker(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {
                    "id": "currency",
                    "kind": "crime",
                    "label": "moeda_falsa",
                    "tokens": ["cedulas", "falsas"],
                    "weight": 1.0,
                },
                {
                    "id": "benefits",
                    "kind": "crime",
                    "label": "crimes_previdenciarios",
                    "tokens": ["beneficio", "previdenciario"],
                    "weight": 1.0,
                },
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "Foram apreendidas cedulas falsas em fraude de beneficio previdenciario.",
                bank_path,
                confidence_threshold=0.70,
                margin_threshold=0.12,
                min_active_discriminators=1,
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "accepted_known_cooccurrence")
        self.assertEqual(set(result.inference.crimes_mais_presentes), {"moeda_falsa", "crimes_previdenciarios"})

    def test_cosine_supported_theme_is_accepted_despite_low_wnn_confidence(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {"id": "drugs", "kind": "crime", "label": "trafico_drogas", "tokens": ["trafico", "droga"], "weight": 1.1},
                {"id": "laundering", "kind": "crime", "label": "lavagem_dinheiro", "tokens": ["lavagem", "dinheiro"], "weight": 0.9},
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "trafico de droga com lavagem de dinheiro",
                bank_path,
                confidence_threshold=0.80,
                margin_threshold=0.30,
                min_active_discriminators=1,
                cosine_candidates=[{"label": "trafico_drogas", "score": 0.40}],
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.top_label, "trafico_drogas")
        self.assertEqual(result.status, "accepted_cosine_supported")

    def test_cosine_breaks_two_class_ambiguity_in_favor_of_runner_up(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {"id": "alpha", "kind": "crime", "label": "classe_alpha", "tokens": ["alfa"], "weight": 1.1},
                {"id": "beta", "kind": "crime", "label": "classe_beta", "tokens": ["beta"], "weight": 1.0},
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "alfa beta",
                bank_path,
                confidence_threshold=0.80,
                margin_threshold=0.30,
                min_active_discriminators=1,
                cosine_candidates=[{"label": "classe_beta", "score": 0.40}],
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.top_label, "classe_beta")
        self.assertEqual(result.status, "accepted_cosine_tiebreak")

    def test_generic_generalized_pattern_is_context_not_decisive_evidence(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {
                    "id": "generic",
                    "kind": "crime",
                    "label": "contrabando_descaminho",
                    "tokens": ["associacao", "criminosa", "fraude", "organizacao"],
                    "weight": 1.0,
                    "strength": "strong",
                    "source": "agent2_generalized_micro_world",
                }
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "associacao criminosa e fraude em organizacao",
                bank_path,
                min_active_discriminators=1,
            )

        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "abstain_weak_marker_evidence")

    def test_tag_hint_cannot_create_crime_without_textual_evidence(self) -> None:
        payload = {
            "version": 1,
            "discriminators": [
                {"id": "drugs", "kind": "crime", "label": "trafico_drogas", "tokens": ["trafico", "drogas"], "weight": 1.1},
            ],
            "memories": {},
        }
        sync_feature_memory(payload)
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "Notícia sobre atividade institucional.",
                bank_path,
                crime_tag_hints=["trafico_drogas"],
            )

        self.assertFalse(result.accepted)

    def test_domain_guard_does_not_allow_generic_environmental_rule_to_override_arms(self) -> None:
        payload = {
            "discriminators": [
                {
                    "id": "environment-generic",
                    "kind": "crime",
                    "label": "crimes_ambientais",
                    "tokens": ["posse", "ilegal"],
                    "weight": 0.75,
                    "source": "agent2_generalized_micro_world",
                },
                {
                    "id": "arms-specific",
                    "kind": "crime",
                    "label": "armas_municoes",
                    "tokens": ["arma", "fogo"],
                    "weight": 1.15,
                    "source": "agent2_curated_discriminator",
                    "strength": "strong",
                },
            ]
        }
        sync_feature_memory(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = classify_with_wnn(
                "Posse ilegal de arma de fogo.",
                bank_path,
                confidence_threshold=0.10,
                margin_threshold=0.0,
                min_active_discriminators=1,
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.top_label, "armas_municoes")

    def test_mask_coverage_weights_generic_tokens_less_than_specific_tokens(self) -> None:
        payload = {
            "discriminators": [
                {
                    "id": "environment-mask",
                    "kind": "crime",
                    "label": "crimes_ambientais",
                    "tokens": ["garimpo", "ilegal"],
                    "weight": 1.0,
                }
            ]
        }

        candidates = mask_discriminators("Atividade ilegal foi apurada.", payload)

        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0]["mask_coverage"], 0.20)
        self.assertEqual(candidates[0]["matched_tokens"], ["ilegal"])

    def test_legacy_bank_migrates_tokens_to_weighted_mask(self) -> None:
        payload = {
            "discriminators": [
                {
                    "id": "environment-mask",
                    "kind": "crime",
                    "label": "crimes_ambientais",
                    "tokens": ["garimpo", "ilegal"],
                    "weight": 1.0,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps(payload), encoding="utf-8")
            result = migrate_feature_bank_to_weighted_masks(bank_path)
            migrated = json.loads(bank_path.read_text(encoding="utf-8"))

        mask = migrated["discriminators"][0]["mask"]
        self.assertTrue(result["migrated"])
        self.assertEqual(mask["garimpo"]["role"], "core")
        self.assertEqual(mask["ilegal"]["role"], "context")
        self.assertLess(mask["ilegal"]["weight"], mask["garimpo"]["weight"])

    def test_residual_learning_suggests_only_crime_discriminators(self) -> None:
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
        self.assertNotIn(("modus", "arma_fogo"), kinds)
        self.assertIn("arma", suggestions[0]["tokens"])

    def test_learned_discriminator_survives_when_foundation_class_is_at_capacity(self) -> None:
        foundation = [
            {
                "id": f"foundation-{index}",
                "kind": "crime",
                "label": "trafico_drogas",
                "tokens": [f"generico{index}", f"contexto{index}"],
                "source": "agent2_generalized_micro_world",
            }
            for index in range(35)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps({"discriminators": foundation}), encoding="utf-8")
            learned = append_discriminators_from_learned_rules(
                [
                    {
                        "kind": "crime",
                        "label": "trafico_drogas",
                        "tokens": ["trafico", "drogas", "aeronave", "clandestina"],
                        "source": "agent3_learned_discriminator",
                    }
                ],
                bank_path,
                max_discriminators_per_label=35,
            )
            saved = json.loads(bank_path.read_text(encoding="utf-8"))

        self.assertEqual(len(learned), 1)
        self.assertEqual(len(saved["discriminators"]), 35)
        self.assertTrue(
            any(item.get("source") == "agent3_learned_discriminator" for item in saved["discriminators"])
        )

    def test_frequent_class_expands_beyond_foundation_baseline_up_to_global_ceiling(self) -> None:
        foundation = [
            {
                "id": f"foundation-{index}",
                "kind": "crime",
                "label": "trafico_drogas",
                "tokens": [f"generico{index}", f"contexto{index}"],
                "source": "agent2_generalized_micro_world",
            }
            for index in range(35)
        ]
        rules = [
            {
                "kind": "crime",
                "label": "trafico_drogas",
                "tokens": ["trafico", "drogas", "aeronave", "clandestina"],
                "source": "agent3_learned_discriminator",
            },
            {
                "kind": "crime",
                "label": "trafico_drogas",
                "tokens": ["trafico", "drogas", "embarcacao", "maritima"],
                "source": "agent3_learned_discriminator",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "wnn_feature_bank.json"
            bank_path.write_text(json.dumps({"discriminators": foundation}), encoding="utf-8")
            learned = append_discriminators_from_learned_rules(rules, bank_path, max_discriminators_per_label=100)
            saved = json.loads(bank_path.read_text(encoding="utf-8"))

        self.assertEqual(len(learned), 2)
        self.assertEqual(len(saved["discriminators"]), 36)
        self.assertTrue(any(item.get("marker_variants") for item in saved["discriminators"]))


if __name__ == "__main__":
    unittest.main()
