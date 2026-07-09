from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.project_config import ANALYSIS_DIR, PROJECT_ROOT


RUN_DIR = ANALYSIS_DIR / "incremental"
FIGURES_DIR = RUN_DIR / "figures"


CLASSIFICATIONS_FINAL_CSV = RUN_DIR / "classificacoes_incrementais_pos_quarentena.csv"
ARTICLE_MEDIA_DIR = PROJECT_ROOT / "artigo" / "media"
OUTPUT_ARTICLE = ARTICLE_MEDIA_DIR / "figura-8-arvore-wnn-crime-modus.png"
OUTPUT_RESULTS = FIGURES_DIR / "arvore_wnn_crime_modus.png"
OUTPUT_RESULTS_COMPLETE = FIGURES_DIR / "arvore_wnn_crime_modus_completa.png"


def _parse_jsonish_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text or text == "[]":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return []


def _parse_jsonish_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text or text == "{}":
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extract_crime_label(row: pd.Series) -> str:
    for column in ("label_pos_quarentena", "label_final_arvore_refinada", "agent3_canonical_label", "wnn_top_label"):
        value = str(row.get(column, "") or "").strip()
        if value and value.lower() != "nan":
            return value
    inference = _parse_jsonish_dict(row.get("inference", ""))
    for key in ("identidade_canonica", "tema_principal"):
        value = str(inference.get(key, "") or "").strip()
        if value:
            return value
    crimes = inference.get("crimes_mais_presentes", [])
    if isinstance(crimes, list):
        for item in crimes:
            value = str(item or "").strip()
            if value:
                return value
    return "sem_crime_definido"


def _extract_modus_labels(row: pd.Series) -> list[str]:
    for column in ("agent3_modus_operandi", "wnn_modus_operandi"):
        values = _parse_jsonish_list(row.get(column, []))
        labels = [str(item).strip() for item in values if str(item).strip()]
        if labels:
            return labels
    inference = _parse_jsonish_dict(row.get("inference", ""))
    values = inference.get("modus_operandi", [])
    if isinstance(values, list):
        labels = [str(item).strip() for item in values if str(item).strip()]
        if labels:
            return labels
    return []


def build_hierarchy_df(classifications_path: Path = CLASSIFICATIONS_FINAL_CSV) -> pd.DataFrame:
    if not classifications_path.exists():
        return pd.DataFrame(columns=["crime_label", "modus_label", "count", "crime_total"])
    frame = pd.read_csv(classifications_path)
    if frame.empty:
        return pd.DataFrame(columns=["crime_label", "modus_label", "count", "crime_total"])

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        crime_label = _extract_crime_label(row)
        modus_labels = _extract_modus_labels(row) or ["sem_modus_definido"]
        for modus_label in modus_labels:
            rows.append({"crime_label": crime_label, "modus_label": modus_label})
    if not rows:
        return pd.DataFrame(columns=["crime_label", "modus_label", "count", "crime_total"])

    hierarchy = (
        pd.DataFrame(rows)
        .groupby(["crime_label", "modus_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    crime_totals = (
        hierarchy.groupby("crime_label", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "crime_total"})
    )
    hierarchy = hierarchy.merge(crime_totals, on="crime_label", how="left")
    hierarchy = hierarchy.sort_values(
        ["crime_total", "count", "crime_label", "modus_label"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return hierarchy


def build_article_hierarchy(
    hierarchy: pd.DataFrame,
    *,
    top_crimes: int = 5,
    top_modus_per_crime: int = 3,
) -> pd.DataFrame:
    if hierarchy.empty:
        return hierarchy
    rows: list[pd.DataFrame] = []
    top_crime_labels = (
        hierarchy[["crime_label", "crime_total"]]
        .drop_duplicates()
        .sort_values(["crime_total", "crime_label"], ascending=[False, True])
        .head(top_crimes)["crime_label"]
        .tolist()
    )
    for crime_label in top_crime_labels:
        group = hierarchy.loc[hierarchy["crime_label"] == crime_label].copy()
        top_group = group.sort_values(["count", "modus_label"], ascending=[False, True]).head(top_modus_per_crime).copy()
        remainder = group.loc[~group["modus_label"].isin(top_group["modus_label"])].copy()
        if not remainder.empty:
            top_group = pd.concat(
                [
                    top_group,
                    pd.DataFrame(
                        [
                            {
                                "crime_label": crime_label,
                                "modus_label": "outros_modus",
                                "count": int(remainder["count"].sum()),
                                "crime_total": int(group["crime_total"].iloc[0]),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        rows.append(top_group)
    if not rows:
        return hierarchy.head(0).copy()
    article = pd.concat(rows, ignore_index=True)
    article["crime_total"] = article.groupby("crime_label")["count"].transform("sum")
    article = article.sort_values(
        ["crime_total", "count", "crime_label", "modus_label"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return article


def _display_label(value: str, *, width: int = 22) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = text.replace("modus operandi", "modus")
    text = text.replace("outros modus", "demais modos")
    text = text.replace("sem modus definido", "sem modus definido")
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width)) or text


def draw_tree(
    hierarchy: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Arvore WNN -> crime canonico -> modus operandi",
    subtitle: str = "Cada galho mostra ocorrencias finais por crime principal e modo de execucao; noticias com multiplos modos podem aparecer em mais de um ramo.",
) -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hierarchy.empty:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=180)
        ax.axis("off")
        ax.text(0.02, 0.62, title, fontsize=15, fontweight="bold", transform=ax.transAxes)
        ax.text(0.02, 0.40, "Sem classificacoes finais suficientes para montar a arvore.", fontsize=11, transform=ax.transAxes)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    crimes = hierarchy["crime_label"].drop_duplicates().tolist()
    row_gap = 0.84
    crime_gap = 0.56
    crime_positions: dict[str, float] = {}
    modus_positions: dict[tuple[str, str], float] = {}
    current_y = 0.0
    for crime_label in crimes:
        group = hierarchy.loc[hierarchy["crime_label"] == crime_label]
        start_y = current_y
        for item in group.itertuples():
            key = (str(item.crime_label), str(item.modus_label))
            modus_positions[key] = current_y
            current_y += row_gap
        end_y = current_y - row_gap
        crime_positions[crime_label] = (start_y + end_y) / 2
        current_y += crime_gap

    total_docs = int(hierarchy["count"].sum())
    root_y = (min(crime_positions.values()) + max(crime_positions.values())) / 2
    height = max(6.8, current_y * 0.42 + 1.8)
    fig, ax = plt.subplots(figsize=(15.2, height), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(current_y, -1.3)
    ax.axis("off")

    root_x = 0.04
    crime_x = 0.31
    modus_x = 0.68
    colors = plt.cm.tab20(range(max(len(crimes), 3)))

    ax.text(
        root_x,
        root_y,
        f"WNN\n{total_docs} ocorr.",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.52",
            "facecolor": "#dbeafe",
            "edgecolor": "#1d4ed8",
            "linewidth": 1.0,
        },
    )

    for index, crime_label in enumerate(crimes):
        crime_group = hierarchy.loc[hierarchy["crime_label"] == crime_label]
        crime_total = int(crime_group["crime_total"].iloc[0])
        crime_y = crime_positions[crime_label]
        color = colors[index % len(colors)]
        ax.plot([0.18, crime_x - 0.025], [root_y, crime_y], color="#94a3b8", linewidth=1.0, alpha=0.85)
        ax.text(
            crime_x,
            crime_y,
            f"{_display_label(crime_label, width=24)}\n{crime_total} ocorr. | {len(crime_group)} modus",
            ha="left",
            va="center",
            fontsize=9.4,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": color,
                "edgecolor": "#334155",
                "linewidth": 0.75,
                "alpha": 0.92,
            },
        )
        for item in crime_group.itertuples():
            modus_key = (str(item.crime_label), str(item.modus_label))
            modus_y = modus_positions[modus_key]
            ax.plot([0.53, modus_x - 0.03], [crime_y, modus_y], color="#cbd5e1", linewidth=0.9, alpha=0.9)
            ax.text(
                modus_x,
                modus_y,
                f"{_display_label(str(item.modus_label), width=18)} | {int(item.count)} ocorr.",
                ha="left",
                va="center",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "#f8fafc",
                    "edgecolor": "#cbd5e1",
                    "linewidth": 0.65,
                },
            )

    ax.text(0.04, -0.92, title, fontsize=15, fontweight="bold")
    ax.text(0.04, -0.52, subtitle, fontsize=9.0, color="#475569")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run() -> list[Path]:
    hierarchy = build_hierarchy_df()
    article_hierarchy = build_article_hierarchy(hierarchy)
    outputs = [
        draw_tree(
            article_hierarchy,
            OUTPUT_ARTICLE,
            title="Arvore WNN -> crime canonico -> modus operandi",
            subtitle="Estrutura de saida produzida pela classificacao final WNN: raiz da memoria, crime canonico principal e modos de execucao associados; a arvore completa permanece nos artefatos da execucao.",
        ),
        draw_tree(
            article_hierarchy,
            OUTPUT_RESULTS,
            title="Arvore WNN -> crime canonico -> modus operandi",
            subtitle="Estrutura de saida produzida pela classificacao final WNN em recorte representativo dos crimes mais frequentes.",
        ),
        draw_tree(
            hierarchy,
            OUTPUT_RESULTS_COMPLETE,
            title="Arvore WNN -> crime canonico -> modus operandi (completa)",
            subtitle="Arvore operacional completa; noticias com multiplos modos podem aparecer em mais de um ramo.",
        ),
    ]
    return outputs


def main() -> None:
    for path in run():
        print(path)


if __name__ == "__main__":
    main()
