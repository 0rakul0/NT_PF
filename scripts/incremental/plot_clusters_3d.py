from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.incremental.common import FIGURES_DIR, RUN_DIR, THEMES_JSON


CLUSTER_ASSIGNMENTS_CSV = RUN_DIR / "cluster_assignments_amostra.csv"
OUTPUT_HTML = FIGURES_DIR / "clusters_3d_amostra.html"


def cluster_theme_map() -> dict[int, str]:
    if not THEMES_JSON.exists():
        return {}
    payload = json.loads(THEMES_JSON.read_text(encoding="utf-8"))
    mapping: dict[int, list[str]] = {}
    for theme in payload.get("themes", []):
        if theme.get("decision") != "accept":
            continue
        label = str(theme.get("canonical_theme", ""))
        for cluster_id in theme.get("included_cluster_ids", []):
            mapping.setdefault(int(cluster_id), []).append(label)
    return {cluster_id: " | ".join(sorted(set(labels))) for cluster_id, labels in mapping.items()}


def build_3d_dataframe(rows: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        max_features=25000,
    )
    text_column = "cluster_text" if "cluster_text" in rows.columns else "context"
    matrix = vectorizer.fit_transform(rows[text_column].fillna(""))
    components = TruncatedSVD(n_components=3, random_state=42).fit_transform(matrix)
    components = normalize(components)

    output = rows.copy()
    output["x"] = components[:, 0]
    output["y"] = components[:, 1]
    output["z"] = components[:, 2]
    theme_by_cluster = cluster_theme_map()
    output["tema_canonico"] = output["cluster_id"].map(lambda value: theme_by_cluster.get(int(value), "sem_tema_canonico"))
    output["cluster_label"] = output["cluster_id"].map(lambda value: f"cluster_{int(value)}")
    output["titulo_curto"] = output["titulo"].fillna("").str.slice(0, 120)
    output["tags_curta"] = output["tags"].fillna("").str.slice(0, 180)
    return output


def run() -> Path:
    rows = pd.read_csv(CLUSTER_ASSIGNMENTS_CSV)
    plot_df = build_3d_dataframe(rows)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = px.scatter_3d(
        plot_df,
        x="x",
        y="y",
        z="z",
        color="tema_canonico",
        symbol="cluster_label",
        hover_name="titulo_curto",
        hover_data={
            "arquivo": True,
            "cluster_id": True,
            "tema_canonico": True,
            "tags_curta": True,
            "x": False,
            "y": False,
            "z": False,
            "cluster_label": False,
        },
        title="Amostra inicial - clusters exploratorios e temas canonicos",
        height=820,
    )
    fig.update_traces(marker={"size": 4, "opacity": 0.72})
    fig.update_layout(
        legend_title_text="Tema canonico",
        scene={
            "xaxis_title": "SVD-1",
            "yaxis_title": "SVD-2",
            "zaxis_title": "SVD-3",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn", full_html=True)
    return OUTPUT_HTML


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
