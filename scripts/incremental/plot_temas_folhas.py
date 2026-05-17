from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.incremental.clusterizacao_inicial import domain_term_set, dominant_family
from scripts.incremental.common import CLUSTER_SUMMARY_CSV, FIGURES_DIR, PROJECT_ROOT


ARTICLE_MEDIA_DIR = PROJECT_ROOT / "artigo" / "media"
OUTPUT_ARTICLE = ARTICLE_MEDIA_DIR / "figura-temas-canonicos-folhas.png"
OUTPUT_RESULTS = FIGURES_DIR / "temas_canonicos_folhas.png"

THEME_LABELS = {
    "armas_municoes": "armas_municoes",
    "contrabando_descaminho": "contrabando_descaminho",
    "corrupcao_recursos_publicos": "corrupcao_desvio_recursos_publicos",
    "crime_organizado": "crime_organizado",
    "crimes_ambientais": "crimes_ambientais",
    "crimes_contra_criancas": "crimes_contra_criancas",
    "crimes_eleitorais": "crimes_eleitorais",
    "crimes_migratorios": "crimes_migratorios",
    "crimes_previdenciarios": "crimes_previdenciarios",
    "falsificacao_documental": "falsificacao_documental",
    "lavagem_dinheiro": "lavagem_dinheiro",
    "moeda_falsa": "moeda_falsa",
    "radiodifusao_clandestina": "radiodifusao_clandestina",
    "receptacao": "receptacao",
    "roubo_assalto": "roubo_assalto",
    "saude_publica": "crimes_contra_saude_publica",
    "trafico_drogas": "trafico_drogas",
    "indefinido": "tema_candidato_indefinido",
}


def compact_terms(value: object, limit: int = 3) -> str:
    selected: list[str] = []
    for term in str(value or "").split(" | "):
        term = term.strip()
        if term and term not in selected:
            selected.append(term)
        if len(selected) >= limit:
            break
    return ", ".join(selected)


def build_tree_rows() -> pd.DataFrame:
    summary = pd.read_csv(CLUSTER_SUMMARY_CSV)
    rows = []
    for row in summary.itertuples():
        terms = domain_term_set(getattr(row, "top_terms", ""))
        if not terms:
            terms = domain_term_set(getattr(row, "domain_terms", ""))
        family = dominant_family(terms)
        theme = THEME_LABELS.get(family, family)
        raw_ids = str(getattr(row, "raw_cluster_ids", "") or row.cluster_id)
        rows.append(
            {
                "theme": theme,
                "cluster_id": int(row.cluster_id),
                "raw_cluster_ids": raw_ids,
                "size": int(row.size),
                "leaf_label": f"cluster {int(row.cluster_id)} ({int(row.size)})",
                "leaf_detail": compact_terms(row.top_terms),
            }
        )
    return pd.DataFrame(rows).sort_values(["theme", "size"], ascending=[True, False]).reset_index(drop=True)


def draw_tree(rows: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    themes = rows["theme"].drop_duplicates().tolist()
    theme_counts = rows.groupby("theme")["size"].sum().to_dict()

    row_gap = 0.88
    theme_gap = 0.62
    y_positions: dict[int, float] = {}
    theme_positions: dict[str, float] = {}
    current_y = 0.0
    for theme in themes:
        group = rows.loc[rows["theme"] == theme]
        start_y = current_y
        for index in group.index:
            y_positions[int(index)] = current_y
            current_y += row_gap
        end_y = current_y - row_gap
        theme_positions[theme] = (start_y + end_y) / 2
        current_y += theme_gap

    height = max(8.5, current_y * 0.42 + 1.5)
    fig, ax = plt.subplots(figsize=(16, height), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(current_y, -1)
    ax.axis("off")

    theme_x = 0.04
    leaf_x = 0.46
    detail_x = 0.62
    colors = plt.cm.Set3(range(max(len(themes), 3)))

    for theme_index, theme in enumerate(themes):
        y = theme_positions[theme]
        color = colors[theme_index % len(colors)]
        ax.text(
            theme_x,
            y,
            f"{theme}\n{int(theme_counts[theme])} noticias",
            ha="left",
            va="center",
            fontsize=9.5,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#334155", "linewidth": 0.7, "alpha": 0.9},
        )

    for index, row in rows.iterrows():
        y = y_positions[int(index)]
        theme_y = theme_positions[row["theme"]]
        ax.plot([0.30, leaf_x - 0.025], [theme_y, y], color="#94a3b8", linewidth=0.8, alpha=0.75)
        raw = str(row["raw_cluster_ids"])
        raw_suffix = f" raw:{raw}" if " | " in raw else ""
        ax.text(
            leaf_x,
            y,
            f"{row['leaf_label']}{raw_suffix}",
            ha="left",
            va="center",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "linewidth": 0.6},
        )
        ax.text(detail_x, y, str(row["leaf_detail"]), ha="left", va="center", fontsize=7.8, color="#334155")

    ax.text(0.04, -0.65, "Arvore operacional: temas canonicos e folhas de clusters consolidados", fontsize=15, weight="bold")
    ax.text(0.04, -0.25, "Folhas agrupadas por familia criminal dominante; locais e entidades nao definem tema.", fontsize=8.8, color="#475569")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run() -> list[Path]:
    rows = build_tree_rows()
    draw_tree(rows, OUTPUT_ARTICLE)
    draw_tree(rows, OUTPUT_RESULTS)
    return [OUTPUT_ARTICLE, OUTPUT_RESULTS]


def main() -> None:
    for path in run():
        print(path)


if __name__ == "__main__":
    main()
