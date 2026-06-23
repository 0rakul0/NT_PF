from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.incremental.preprocessamento_linguistico import preprocess_body_text


class LinguisticPreprocessingTests(unittest.TestCase):
    def test_preserves_domain_phrase_and_adjectives(self) -> None:
        with patch(
            "scripts.incremental.preprocessamento_linguistico.portuguese_pos_tagger",
            return_value=None,
        ):
            result = preprocess_body_text(
                "A operacao apurou abuso sexual infantojuvenil e pornografia infantil."
            )

        self.assertIn("abuso_sexual_infantojuvenil", result.semantic_features)
        self.assertIn("pornografia_infantil", result.semantic_features)
        self.assertIn("infantojuvenil", result.tokens)
        self.assertIn("infantil", result.tokens)

    def test_removes_generic_operational_verbs(self) -> None:
        with patch(
            "scripts.incremental.preprocessamento_linguistico.portuguese_pos_tagger",
            return_value=None,
        ):
            result = preprocess_body_text(
                "A Policia Federal deflagrou e realizou operacao contra garimpo ilegal."
            )

        self.assertNotIn("deflagrou", result.tokens)
        self.assertNotIn("realizou", result.tokens)
        self.assertIn("garimpo", result.tokens)
        self.assertIn("ilegal", result.tokens)

    def test_preserves_negation_and_domain_entities(self) -> None:
        with patch(
            "scripts.incremental.preprocessamento_linguistico.portuguese_pos_tagger",
            return_value=None,
        ):
            result = preprocess_body_text(
                "Radio funcionava sem autorizacao da Anatel e nao possuia licenca."
            )

        self.assertIn("sem", result.tokens)
        self.assertIn("nao", result.tokens)
        self.assertIn("anatel", result.tokens)


if __name__ == "__main__":
    unittest.main()
