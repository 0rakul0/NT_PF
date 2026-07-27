from __future__ import annotations

import unittest

from scripts.avaliar_crimes_por_tags import crime_labels_from_tags


class CrimeTagMappingTests(unittest.TestCase):
    def test_maps_criminal_tags_and_discards_metadata(self) -> None:
        labels = crime_labels_from_tags(
            ["Operação PF", "Bahia", "Tráfico internacional de drogas", "Destaque"]
        )
        self.assertEqual(labels, ["trafico_drogas"])

    def test_keeps_multiple_crimes_when_pf_tags_are_multi_label(self) -> None:
        labels = crime_labels_from_tags(["Combate organização criminosa", "Tráfico de drogas e armas"])
        self.assertEqual(labels, ["armas_municoes", "crime_organizado", "trafico_drogas"])
