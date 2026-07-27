from __future__ import annotations

import unittest

from scripts.incremental.common import split_docs


def doc(name: str, date: str) -> dict[str, object]:
    return {"arquivo": name, "parsed": {"data_publicacao": date}}


class ChronologicalFoundationSplitTests(unittest.TestCase):
    def test_foundation_contains_only_earliest_documents(self) -> None:
        docs = [
            doc("future.md", "10/01/2024"),
            doc("middle.md", "10/01/2022"),
            doc("early.md", "10/01/2020"),
            doc("later.md", "10/01/2023"),
        ]

        foundation, reserve = split_docs(docs, fraction=0.50, seed=42)

        self.assertEqual([item["arquivo"] for item in foundation], ["early.md", "middle.md"])
        self.assertEqual([item["arquivo"] for item in reserve], ["later.md", "future.md"])

    def test_undated_documents_are_not_selected_before_dated_history(self) -> None:
        docs = [
            doc("undated.md", ""),
            doc("early.md", "01/01/2019"),
            doc("later.md", "01/01/2020"),
        ]

        foundation, reserve = split_docs(docs, fraction=0.34, seed=42)

        self.assertEqual([item["arquivo"] for item in foundation], ["early.md"])
        self.assertEqual([item["arquivo"] for item in reserve], ["later.md", "undated.md"])

    def test_split_is_not_affected_by_seed_or_temporal_strata(self) -> None:
        docs = [doc("a.md", "01/01/2020"), doc("b.md", "01/01/2021"), doc("c.md", "01/01/2022")]

        first, _ = split_docs(docs, fraction=0.34, seed=1, temporal_granularity="year")
        second, _ = split_docs(docs, fraction=0.34, seed=999, temporal_granularity="month")

        self.assertEqual([item["arquivo"] for item in first], [item["arquivo"] for item in second])


if __name__ == "__main__":
    unittest.main()
