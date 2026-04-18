from __future__ import annotations

import ast
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.project_config import (
    ANALYSIS_DIR,
    BRAZIL_STATES_GEOJSON,
    PROJECT_ROOT as BASE_DIR,
    STATE_POINTS,
    STREAMLIT_REQUIRED_DATA_FILES as REQUIRED_DATA_FILES,
)


CLUSTER_GRAPH_COLORS = [
    "#1d5c63",
    "#d64848",
    "#8a4b08",
    "#287271",
    "#5b8e7d",
    "#4f6d7a",
    "#9c6644",
    "#8d99ae",
    "#3d5a80",
    "#6d597a",
    "#b56576",
    "#2a9d8f",
]


def data_signature() -> tuple[tuple[str, int, int], ...]:
    """Build a cache signature from the required artifact files."""
    signature: list[tuple[str, int, int]] = []
    for path in REQUIRED_DATA_FILES:
        stat = path.stat()
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def missing_required_data_paths() -> list[Path]:
    """Return the generated artifacts that are still missing on disk."""
    return [path for path in REQUIRED_DATA_FILES if not path.exists()]


st.set_page_config(
    page_title="NT PF | Atlas Analitico",
    page_icon="PF",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Inject the custom CSS theme used by the Streamlit dashboard."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe4;
            --paper: #fffaf2;
            --ink: #1a1f1c;
            --muted: #5f675f;
            --accent: #1d5c63;
            --accent-2: #8a4b08;
            --line: rgba(29, 92, 99, 0.18);
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(138, 75, 8, 0.08), transparent 28%),
                radial-gradient(circle at left top, rgba(29, 92, 99, 0.10), transparent 32%),
                linear-gradient(180deg, #f7f1e6 0%, #f3ede1 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(29,92,99,0.95), rgba(24,38,43,0.96));
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 24px;
            padding: 1.5rem 1.6rem;
            color: #f9f4eb;
            box-shadow: 0 18px 40px rgba(20, 34, 36, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2.2rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.25rem 0;
            color: rgba(249, 244, 235, 0.88);
            max-width: 60rem;
        }
        .artifact-card {
            background: rgba(255, 250, 242, 0.88);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 150px;
            box-shadow: 0 10px 24px rgba(40, 42, 30, 0.06);
        }
        .artifact-card h4 {
            margin: 0 0 0.45rem 0;
            color: var(--ink);
        }
        .artifact-card p {
            margin: 0.2rem 0;
            color: var(--muted);
            font-size: 0.95rem;
        }
        .section-note {
            background: rgba(29,92,99,0.08);
            border-left: 4px solid rgba(29,92,99,0.85);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin: 0.4rem 0 1rem 0;
            color: #294246;
        }
        .metric-caption {
            color: var(--muted);
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()


def render_missing_data_state(missing_paths: list[Path]) -> None:
    """Render onboarding instructions when the generated data files are absent."""
    st.markdown(
        """
        <div class="hero">
            <h1>Atlas analitico pronto para ser gerado</h1>
            <p>
                Este repositorio foi enxugado para GitHub e nao inclui os artefatos gerados do pipeline.
                Para abrir o painel completo, gere os dados localmente com os comandos abaixo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("Arquivos de dados ausentes. Gere os artefatos antes de abrir o painel completo.")
    st.markdown("### Como preparar o ambiente")
    st.code(
        "\n".join(
            [
                "uv sync --group extraction",
                r".\\.venv\\Scripts\\python.exe .\\scripts\\pf_operacoes_pipeline.py sync --index-csv .\\data\\pf_operacoes_index.csv --content-csv .\\data\\pf_operacoes_conteudos.csv --markdown-dir .\\data\\noticias_markdown",
                r".\\.venv\\Scripts\\python.exe .\\scripts\\pf_analise_qualitativa.py --output-dir .\\data\\analise_qualitativa",
                r".\\.venv\\Scripts\\python.exe -m streamlit run .\\streamlit_app.py",
            ]
        ),
        language="powershell",
    )
    st.markdown("### O que ficou fora do GitHub")
    st.markdown(
        """
        - `data/noticias_markdown/`
        - `data/pf_operacoes_index.csv`
        - `data/pf_operacoes_conteudos.csv`
        - `data/analise_qualitativa/*.csv`
        - `data/analise_qualitativa/*.md`
        """
    )
    st.markdown("### Arquivos ausentes no momento")
    st.code("\n".join(str(path.relative_to(BASE_DIR)) for path in missing_paths))
    st.info("O arquivo `data/reference/brazil_states.geojson` continua versionado porque faz parte da referencia cartografica do app.")


MISSING_DATA_PATHS = missing_required_data_paths()
if MISSING_DATA_PATHS:
    render_missing_data_state(MISSING_DATA_PATHS)
    st.stop()


def parse_list_cell(value: object) -> list[str]:
    """Parse list-like CSV cells stored as Python lists or pipe-delimited strings."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    return [part.strip() for part in text.split("|") if part.strip()]


def fold_text(value: str) -> str:
    """Normalize text to lowercase ASCII for matching and fuzzy search."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(value))
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def build_canonical_cluster_ui_label(cluster_row: pd.Series) -> str:
    """Assemble the selector label shown for one canonical cluster."""
    base_label = str(cluster_row.get("cluster_canonico_label", "") or "").strip()
    first_date = pd.to_datetime(cluster_row.get("first_date"), errors="coerce")
    last_date = pd.to_datetime(cluster_row.get("last_date"), errors="coerce")
    first_year = int(first_date.year) if pd.notna(first_date) else 0
    last_year = int(last_date.year) if pd.notna(last_date) else 0
    size = int(cluster_row.get("size", 0) or 0)
    canonical_cluster_id = int(cluster_row.get("cluster_canonico_id", 0) or 0)
    return f"{base_label} | {first_year}-{last_year} | n={size} | id={canonical_cluster_id}"


@st.cache_data(show_spinner=False)
def load_data(_signature: tuple[tuple[str, int, int], ...]) -> dict[str, pd.DataFrame | str]:
    """Load all dashboard artifacts and derive helper columns used across views."""
    corpus = pd.read_csv(ANALYSIS_DIR / "corpus_enriquecido.csv", parse_dates=["data_publicacao_dt"])
    clusters = pd.read_csv(ANALYSIS_DIR / "resumo_clusters.csv", parse_dates=["first_date", "last_date"])
    temporal = pd.read_csv(ANALYSIS_DIR / "recorrencia_temporal.csv")
    canonical_clusters = pd.read_csv(ANALYSIS_DIR / "clusters_canonicos.csv", parse_dates=["first_date", "last_date"])
    canonical_cluster_year = pd.read_csv(ANALYSIS_DIR / "clusters_canonicos_por_ano.csv")
    canonical_temporal = pd.read_csv(ANALYSIS_DIR / "recorrencia_temporal_clusters_canonicos.csv")
    crimes = pd.read_csv(ANALYSIS_DIR / "crimes_por_ano.csv")
    modus = pd.read_csv(ANALYSIS_DIR / "modus_operandi_por_ano.csv")
    series = pd.read_csv(ANALYSIS_DIR / "series_semanticas.csv", parse_dates=["first_date", "last_date"])
    pairs = pd.read_csv(ANALYSIS_DIR / "pares_recorrentes.csv", parse_dates=["source_data", "target_data"])
    states_year = pd.read_csv(ANALYSIS_DIR / "estados_por_ano.csv")
    states_cluster = pd.read_csv(ANALYSIS_DIR / "estados_por_cluster.csv")
    report = (ANALYSIS_DIR / "analise_qualitativa.md").read_text(encoding="utf-8")

    corpus["crime_labels_list"] = corpus["crime_labels"].map(parse_list_cell)
    corpus["modus_labels_list"] = corpus["modus_labels"].map(parse_list_cell)
    corpus["tags_list"] = corpus["tags"].map(parse_list_cell)
    corpus["ufs_mencionadas_list"] = corpus["ufs_mencionadas"].map(parse_list_cell)
    corpus["estados_mencionados_list"] = corpus["estados_mencionados"].map(parse_list_cell)
    if "llm_tags" in corpus.columns:
        corpus["llm_tags_list"] = corpus["llm_tags"].map(parse_list_cell)
    if "llm_crimes_mais_presentes" in corpus.columns:
        corpus["llm_crimes_mais_presentes_list"] = corpus["llm_crimes_mais_presentes"].map(parse_list_cell)
    corpus["year"] = corpus["data_publicacao_dt"].dt.year
    corpus["month"] = corpus["data_publicacao_dt"].dt.to_period("M").astype(str)
    corpus["has_series"] = corpus["semantic_series_id"].notna()
    if "cluster_canonico_label" not in canonical_clusters.columns:
        canonical_clusters["cluster_canonico_label"] = ""
    if "cluster_canonico_tipo" not in canonical_clusters.columns:
        canonical_clusters["cluster_canonico_tipo"] = "Outras"
    if "cluster_canonico_display_label" not in canonical_clusters.columns:
        canonical_clusters["cluster_canonico_display_label"] = canonical_clusters.apply(build_canonical_cluster_ui_label, axis=1)
    canonical_clusters["cluster_canonico_ui_label"] = canonical_clusters.apply(build_canonical_cluster_ui_label, axis=1)

    return {
        "corpus": corpus,
        "clusters": clusters,
        "temporal": temporal,
        "canonical_clusters": canonical_clusters,
        "canonical_cluster_year": canonical_cluster_year,
        "canonical_temporal": canonical_temporal,
        "crimes": crimes,
        "modus": modus,
        "series": series,
        "pairs": pairs,
        "states_year": states_year,
        "states_cluster": states_cluster,
        "report": report,
    }


DATA = load_data(data_signature())
CORPUS = DATA["corpus"]
CLUSTERS = DATA["clusters"]
TEMPORAL = DATA["temporal"]
CANONICAL_CLUSTERS = DATA["canonical_clusters"]
CANONICAL_CLUSTER_YEAR = DATA["canonical_cluster_year"]
CANONICAL_TEMPORAL = DATA["canonical_temporal"]
CRIMES = DATA["crimes"]
MODUS = DATA["modus"]
SERIES = DATA["series"]
PAIRS = DATA["pairs"]
STATES_YEAR = DATA["states_year"]
STATES_CLUSTER = DATA["states_cluster"]
REPORT = DATA["report"]


def pretty_label(value: str) -> str:
    """Convert underscored identifiers into a more readable label."""
    return value.replace("_", " ").title()


def available_states() -> list[str]:
    """List the state names available in the current corpus."""
    states = sorted({STATE_POINTS[uf]["state"] for uf in STATE_POINTS})
    return ["Todos"] + states


def available_ufs() -> list[str]:
    """List the UF codes available in the current corpus."""
    return ["Todos"] + sorted(STATE_POINTS.keys())


def filter_corpus_by_state(df: pd.DataFrame, selected_state: str) -> pd.DataFrame:
    """Filter the corpus by one mentioned state name."""
    if selected_state == "Todos":
        return df
    uf = next((code for code, meta in STATE_POINTS.items() if meta["state"] == selected_state), None)
    if uf is None:
        return df.iloc[0:0]
    return df[df["ufs_mencionadas_list"].map(lambda items: uf in items)]


def filter_corpus_by_uf(df: pd.DataFrame, selected_uf: str) -> pd.DataFrame:
    """Filter the corpus by one mentioned UF code."""
    if selected_uf == "Todos":
        return df
    return df[df["ufs_mencionadas_list"].map(lambda items: selected_uf in items)]


def aggregate_states_from_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mentioned states into counts and map coordinates."""
    exploded = (
        df[["ufs_mencionadas_list"]]
        .explode("ufs_mencionadas_list")
        .dropna(subset=["ufs_mencionadas_list"])
        .rename(columns={"ufs_mencionadas_list": "uf"})
    )
    if exploded.empty:
        return pd.DataFrame(columns=["uf", "state", "lat", "lon", "noticias"])
    aggregated = exploded.groupby("uf").size().reset_index(name="noticias")
    aggregated["state"] = aggregated["uf"].map(lambda uf: STATE_POINTS[uf]["state"])
    aggregated["lat"] = aggregated["uf"].map(lambda uf: STATE_POINTS[uf]["lat"])
    aggregated["lon"] = aggregated["uf"].map(lambda uf: STATE_POINTS[uf]["lon"])
    max_count = max(float(aggregated["noticias"].max()), 1.0)
    aggregated["radius"] = aggregated["noticias"].clip(lower=1).pow(0.7) * 4000
    aggregated["share"] = aggregated["noticias"] / max_count
    aggregated["fill_color"] = aggregated["share"].map(color_scale)
    return aggregated.sort_values("noticias", ascending=False).reset_index(drop=True)


def aggregate_states_for_dual_map(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Align two state summaries for side-by-side map comparison."""
    map_a = aggregate_states_from_corpus(df_a)[["uf", "state", "lat", "lon", "noticias"]].rename(columns={"noticias": "noticias_a"})
    map_b = aggregate_states_from_corpus(df_b)[["uf", "state", "lat", "lon", "noticias"]].rename(columns={"noticias": "noticias_b"})
    merged = map_a.merge(map_b, on=["uf", "state", "lat", "lon"], how="outer").fillna(0)
    if merged.empty:
        return merged
    merged["noticias_a"] = merged["noticias_a"].astype(int)
    merged["noticias_b"] = merged["noticias_b"].astype(int)
    merged["total"] = merged["noticias_a"] + merged["noticias_b"]
    max_count = max(float(merged["noticias_a"].max()), float(merged["noticias_b"].max()), 1.0)
    merged["radius_a"] = (merged["noticias_a"] / max_count).clip(lower=0).pow(0.72) * 240000
    merged["radius_b"] = (merged["noticias_b"] / max_count).clip(lower=0).pow(0.72) * 240000
    return merged.sort_values("total", ascending=False).reset_index(drop=True)


def color_scale(value: float) -> list[int]:
    """Map a normalized value to the default RGBA color scale."""
    start = (220, 210, 181)
    end = (24, 88, 98)
    ratio = min(max(float(value), 0.0), 1.0)
    return [int(start[i] + (end[i] - start[i]) * ratio) for i in range(3)] + [210]


def crime_color_scale(value: float) -> list[int]:
    """Map crime intensity values to the choropleth RGBA scale."""
    start = (255, 237, 213)
    end = (153, 27, 27)
    ratio = min(max(float(value), 0.0), 1.0)
    return [int(start[i] + (end[i] - start[i]) * ratio) for i in range(3)] + [220]


def summarize_labels_from_corpus(df: pd.DataFrame, source_col: str, output_col: str) -> pd.DataFrame:
    """Explode label lists and aggregate them by year."""
    rows = []
    for _, row in df.iterrows():
        for label in row[source_col]:
            rows.append({"ano": row["year"], output_col: label})
    if not rows:
        return pd.DataFrame(columns=["ano", output_col, "noticias"])
    return (
        pd.DataFrame(rows)
        .groupby(["ano", output_col])
        .size()
        .reset_index(name="noticias")
        .sort_values(["ano", "noticias"], ascending=[True, False])
        .reset_index(drop=True)
    )


def crime_state_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize crime mentions by year and state."""
    rows = []
    for _, row in df.iterrows():
        if not row["crime_labels_list"] or not row["ufs_mencionadas_list"]:
            continue
        for crime_label in row["crime_labels_list"]:
            for uf in row["ufs_mencionadas_list"]:
                rows.append(
                    {
                        "ano": row["year"],
                        "crime_label": crime_label,
                        "uf": uf,
                        "state": STATE_POINTS[uf]["state"],
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["ano", "crime_label", "uf", "state", "noticias"])
    return (
        pd.DataFrame(rows)
        .groupby(["ano", "crime_label", "uf", "state"])
        .size()
        .reset_index(name="noticias")
        .sort_values(["ano", "crime_label", "noticias"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


@st.cache_resource(show_spinner=False)
def build_text_search_index(texts: tuple[str, ...]) -> tuple[TfidfVectorizer, object]:
    """Build the character n-gram index used by fuzzy text search."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


@st.cache_data(show_spinner=False)
def get_catalog_values(mode: str) -> list[str]:
    """Return the unique catalog values for tags, crimes, or modus."""
    if mode == "Tag":
        values = sorted({tag for tags in CORPUS["tags_list"] for tag in tags if tag})
    elif mode == "Crime":
        values = sorted({label for labels in CORPUS["crime_labels_list"] for label in labels if label})
    elif mode == "Modus":
        values = sorted({label for labels in CORPUS["modus_labels_list"] for label in labels if label})
    else:
        values = []
    return values


@st.cache_resource(show_spinner=False)
def build_catalog_index(values: tuple[str, ...]) -> tuple[list[str], TfidfVectorizer, object]:
    """Build a fuzzy-search index for a controlled catalog of values."""
    normalized = [fold_text(value) for value in values]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
    matrix = vectorizer.fit_transform(normalized) if normalized else None
    return normalized, vectorizer, matrix


def resolve_catalog_term(mode: str, value: str) -> tuple[str, float]:
    """Resolve free text to the closest known catalog value."""
    query = fold_text(value).strip()
    if not query:
        return "", 0.0
    values = get_catalog_values(mode)
    if not values:
        return value, 0.0
    normalized_values, vectorizer, matrix = build_catalog_index(tuple(values))
    if query in normalized_values:
        return values[normalized_values.index(query)], 1.0
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    best_idx = int(scores.argmax())
    return values[best_idx], float(scores[best_idx])


TEXT_VECTORIZER, TEXT_MATRIX = build_text_search_index(tuple(CORPUS["texto_busca_normalizado"].tolist()))


def fuzzy_text_mask(df: pd.DataFrame, value: str) -> tuple[pd.Series, str, float]:
    """Find corpus rows matching a free-text query with exact and fuzzy search."""
    query = fold_text(value).strip()
    if not query:
        return pd.Series(False, index=df.index), value, 0.0

    exact_mask = df["texto_busca_normalizado"].str.contains(query, regex=False, na=False)
    if exact_mask.any():
        return exact_mask, value, 1.0

    query_vector = TEXT_VECTORIZER.transform([query])
    all_scores = cosine_similarity(query_vector, TEXT_MATRIX).ravel()
    score_series = pd.Series(all_scores, index=CORPUS.index).reindex(df.index).fillna(0.0)
    mask = score_series >= 0.20
    if int(mask.sum()) < 12:
        top_idx = score_series.sort_values(ascending=False).head(150).index
        fallback = pd.Series(False, index=df.index)
        fallback.loc[top_idx] = score_series.loc[top_idx] >= 0.10
        mask = fallback
    top_score = float(score_series.max()) if not score_series.empty else 0.0
    return mask, f"{value} (aproximado)", top_score


def match_signal(df: pd.DataFrame, mode: str, value: str) -> tuple[pd.Series, str, float]:
    """Resolve one signal query and return its mask, resolved label, and score."""
    query = fold_text(value).strip()
    if not query:
        return pd.Series(False, index=df.index), value, 0.0
    if mode == "Texto livre":
        return fuzzy_text_mask(df, value)

    resolved_value, score = resolve_catalog_term(mode, value)
    if not resolved_value:
        return pd.Series(False, index=df.index), value, 0.0

    resolved_query = fold_text(resolved_value)
    if mode == "Tag":
        mask = df["tags_list"].map(lambda tags: resolved_query in [fold_text(tag) for tag in tags])
    elif mode == "Crime":
        mask = df["crime_labels_list"].map(lambda labels: resolved_query in [fold_text(label) for label in labels])
    elif mode == "Modus":
        mask = df["modus_labels_list"].map(lambda labels: resolved_query in [fold_text(label) for label in labels])
    else:
        mask = pd.Series(False, index=df.index)
    return mask, resolved_value, score


def build_signal_trend(df: pd.DataFrame, terms: list[str], period: str, mode: str) -> pd.DataFrame:
    """Compute relative signal frequency over time for the selected mode."""
    base_col = "year" if period == "Ano" else "month"
    denominator = df.groupby(base_col).size().rename("total_periodo")
    rows = []
    for term in terms:
        mask, resolved_value, score = match_signal(df, mode, term)
        mentions = df.loc[mask].groupby(base_col).size().rename("citacoes")
        trend = denominator.to_frame().join(mentions, how="left").fillna({"citacoes": 0}).reset_index()
        trend["termo"] = term
        trend["termo_resolvido"] = resolved_value
        trend["score_busca"] = round(score, 4)
        trend["citacoes"] = trend["citacoes"].astype(int)
        trend["share"] = trend["citacoes"] / trend["total_periodo"]
        peak = max(int(trend["citacoes"].max()), 1)
        trend["indice_google_like"] = (trend["citacoes"] / peak) * 100
        rows.append(trend)
    if not rows:
        return pd.DataFrame(
            columns=[base_col, "total_periodo", "citacoes", "termo", "termo_resolvido", "score_busca", "share", "indice_google_like"]
        )
    return pd.concat(rows, ignore_index=True)


def state_map(df: pd.DataFrame, title: str) -> None:
    """Render the bubble map of states mentioned in the current selection."""
    if df.empty:
        st.info("Nenhum estado identificado para este recorte.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="fill_color",
        get_line_color=[255, 250, 242, 180],
        line_width_min_pixels=1,
        pickable=True,
        stroked=True,
    )
    view_state = pdk.ViewState(latitude=-15.5, longitude=-52.5, zoom=3.3)
    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "{state}\nNoticias: {noticias}"},
        ),
        width='stretch',
    )
    st.caption(title)


def dual_state_map(df: pd.DataFrame, label_a: str, label_b: str) -> None:
    """Render the comparison map for two selected signals."""
    if df.empty:
        st.info("Nenhum estado identificado para este confronto.")
        return

    layer_a = pdk.Layer(
        "ScatterplotLayer",
        data=df[df["noticias_a"] > 0],
        get_position="[lon, lat]",
        get_radius="radius_a",
        get_fill_color=[45, 125, 210, 170],
        get_line_color=[15, 59, 117, 200],
        line_width_min_pixels=1,
        pickable=True,
        stroked=True,
    )
    layer_b = pdk.Layer(
        "ScatterplotLayer",
        data=df[df["noticias_b"] > 0],
        get_position="[lon, lat]",
        get_radius="radius_b",
        get_fill_color=[214, 72, 72, 170],
        get_line_color=[122, 23, 23, 200],
        line_width_min_pixels=1,
        pickable=True,
        stroked=True,
    )
    view_state = pdk.ViewState(latitude=-15.5, longitude=-52.5, zoom=3.3)
    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            initial_view_state=view_state,
            layers=[layer_a, layer_b],
            tooltip={"text": "{state}\nA: {noticias_a}\nB: {noticias_b}\nTotal: {total}"},
        ),
        width='stretch',
    )
    st.caption(f"Azul: {label_a} | Vermelho: {label_b}")


@st.cache_data(show_spinner=False)
def load_states_geojson() -> dict:
    """Load the static Brazil states GeoJSON reference file."""
    return json.loads(BRAZIL_STATES_GEOJSON.read_text(encoding="utf-8"))


def crime_map_geojson(summary_df: pd.DataFrame, crime_label: str, selected_year: str) -> tuple[dict, pd.DataFrame]:
    """Prepare the choropleth GeoJSON and summary table for one crime."""
    geojson = load_states_geojson()
    subset = summary_df[summary_df["crime_label"] == crime_label].copy()
    if selected_year != "Todos":
        subset = subset[subset["ano"] == int(selected_year)]
    else:
        subset = subset.groupby(["crime_label", "uf", "state"], as_index=False)["noticias"].sum()

    counts = subset.set_index("uf")["noticias"].to_dict()
    max_count = max(counts.values(), default=0)
    painted = {"type": "FeatureCollection", "features": []}
    for feature in geojson["features"]:
        uf = feature["properties"].get("estado_sigla")
        value = int(counts.get(uf, 0))
        ratio = value / max_count if max_count else 0.0
        color = crime_color_scale(ratio)
        feature_copy = {
            "type": feature["type"],
            "geometry": feature["geometry"],
            "properties": {
                **feature["properties"],
                "noticias": value,
                "state": STATE_POINTS.get(uf, {}).get("state", uf),
                "fill_color": color,
            },
        }
        painted["features"].append(feature_copy)
    return painted, subset


def crime_choropleth(geojson: dict, title: str) -> None:
    """Render the choropleth that shows crime intensity by state."""
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        stroked=True,
        filled=True,
        wireframe=False,
        pickable=True,
        get_fill_color="properties.fill_color",
        get_line_color=[250, 244, 235, 220],
        line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(latitude=-15.5, longitude=-52.5, zoom=3.2)
    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "{state}\nNoticias: {noticias}"},
        ),
        width="stretch",
    )
    st.caption(title)


def crime_state_trend_chart(summary_df: pd.DataFrame, crime_label: str) -> alt.Chart:
    """Build the time-series chart for crime mentions by state."""
    subset = summary_df[summary_df["crime_label"] == crime_label].copy()
    top_ufs = (
        subset.groupby("uf", as_index=False)["noticias"]
        .sum()
        .sort_values("noticias", ascending=False)
        .head(5)["uf"]
        .tolist()
    )
    subset = subset[subset["uf"].isin(top_ufs)]
    subset["ano"] = subset["ano"].astype(str)
    return (
        alt.Chart(subset)
        .mark_line(point=True, strokeWidth=2.2)
        .encode(
            x=alt.X("ano:O", title="Ano"),
            y=alt.Y("noticias:Q", title="Noticias"),
            color=alt.Color("uf:N", title="UF"),
            tooltip=["ano", "uf", "state", "noticias"],
        )
        .properties(height=260)
    )


def artifact_inventory() -> pd.DataFrame:
    """Describe the generated artifacts shown in the inventory section."""
    rows = [
        {
            "artefato": "corpus_enriquecido.csv",
            "papel": "Base mestre para navegar notícia por notícia.",
            "conta": "Título, data, cluster, série semântica, crimes, modus e caminho do markdown.",
            "pergunta": "Onde esta notícia entra na história geral?",
        },
        {
            "artefato": "resumo_clusters.csv",
            "papel": "Mapa dos grandes blocos temáticos.",
            "conta": "Tamanho, termos dominantes, crimes e modus mais frequentes por cluster.",
            "pergunta": "Quais universos de atuação aparecem com mais força?",
        },
        {
            "artefato": "recorrencia_temporal.csv",
            "papel": "Pulso temporal dos clusters.",
            "conta": "Meses de pico, anos ativos e ritmo médio entre notícias parecidas.",
            "pergunta": "Há sazonalidade ou repetição persistente?",
        },
        {
            "artefato": "crimes_por_ano.csv",
            "papel": "Linha do tempo criminal.",
            "conta": "Volume anual por tipo de crime identificado.",
            "pergunta": "Que crime cresce, cai ou muda de posição?",
        },
        {
            "artefato": "modus_operandi_por_ano.csv",
            "papel": "Linha do tempo operacional.",
            "conta": "Busca/apreensão, prisão, cooperação interagências, atuação online e outros.",
            "pergunta": "Como a forma de atuação muda ao longo do tempo?",
        },
        {
            "artefato": "clusters_canonicos.csv",
            "papel": "Identidades canônicas agregadas.",
            "conta": "Resumo dos agrupamentos principais guiados por identidade padronizada pela LLM.",
            "pergunta": "Que temas canônicos realmente estruturam o acervo?",
        },
        {
            "artefato": "recorrencia_temporal_clusters_canonicos.csv",
            "papel": "Pulso temporal dos clusters canônicos.",
            "conta": "Anos ativos, mês de pico, lacunas temporais e recorrência por identidade canônica.",
            "pergunta": "Quais clusters canônicos persistem, crescem ou reaparecem?",
        },
        {
            "artefato": "series_semanticas.csv",
            "papel": "Camada legada de recorrência.",
            "conta": "Conjuntos de notícias próximas mantidos como apoio histórico e comparativo.",
            "pergunta": "O que a lógica anterior agrupava como continuidade?",
        },
        {
            "artefato": "pares_recorrentes.csv",
            "papel": "Pares de notícias altamente próximos.",
            "conta": "Similaridade do cosseno, distância temporal e nome de operação quando existe.",
            "pergunta": "Quais casos reaparecem com textos quase gêmeos?",
        },
    ]
    return pd.DataFrame(rows)


def monthly_volume_chart(df: pd.DataFrame) -> alt.Chart:
    """Build the monthly publication volume chart."""
    monthly = df.groupby("month").size().reset_index(name="noticias")
    monthly["month_dt"] = pd.to_datetime(monthly["month"])
    return (
        alt.Chart(monthly)
        .mark_area(line={"color": "#1d5c63"}, color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="#1d5c63", offset=0), alt.GradientStop(color="#d8e9ea", offset=1)],
            x1=1, x2=1, y1=1, y2=0
        ))
        .encode(
            x=alt.X("month_dt:T", title="Mês"),
            y=alt.Y("noticias:Q", title="Notícias"),
            tooltip=["month", "noticias"],
        )
        .properties(height=280)
    )


def top_crime_chart(df: pd.DataFrame) -> alt.Chart:
    """Build the ranking chart for the most frequent crimes."""
    crime_totals = (
        df.groupby("crime_label", as_index=False)["noticias"]
        .sum()
        .sort_values("noticias", ascending=False)
        .head(10)
    )
    crime_totals["crime_label"] = crime_totals["crime_label"].map(pretty_label)
    return (
        alt.Chart(crime_totals)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color="#8a4b08")
        .encode(
            x=alt.X("noticias:Q", title="Notícias"),
            y=alt.Y("crime_label:N", sort="-x", title="Crime"),
            tooltip=["crime_label", "noticias"],
        )
        .properties(height=320)
    )


def cluster_size_chart(df: pd.DataFrame) -> alt.Chart:
    """Build the ranking chart for the largest clusters."""
    top_clusters = df.head(12).copy()
    top_clusters["cluster_name"] = top_clusters["cluster_id"].map(lambda x: f"Cluster {x}")
    return (
        alt.Chart(top_clusters)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color="#1d5c63")
        .encode(
            x=alt.X("size:Q", title="Notícias"),
            y=alt.Y("cluster_name:N", sort="-x", title="Cluster"),
            tooltip=["cluster_id", "size", "top_terms", "top_crimes"],
        )
        .properties(height=360)
    )


def label_heatmap(df: pd.DataFrame, label_col: str, title: str) -> alt.Chart:
    """Build the yearly heatmap for crime or modus labels."""
    if df.empty:
        return alt.Chart(pd.DataFrame({label_col: [], "ano": [], "noticias": []})).mark_rect()
    top_labels = (
        df.groupby(label_col, as_index=False)["noticias"]
        .sum()
        .sort_values("noticias", ascending=False)
        .head(8)[label_col]
        .tolist()
    )
    subset = df[df[label_col].isin(top_labels)].copy()
    subset["ano"] = subset["ano"].astype(str)
    subset[label_col] = subset[label_col].map(pretty_label)
    return (
        alt.Chart(subset)
        .mark_rect()
        .encode(
            x=alt.X("ano:O", title="Ano"),
            y=alt.Y(f"{label_col}:N", sort=top_labels, title=title),
            color=alt.Color("noticias:Q", scale=alt.Scale(scheme="teals"), title="Notícias"),
            tooltip=["ano", label_col, "noticias"],
        )
        .properties(height=280)
    )


def cluster_timeline_chart(corpus: pd.DataFrame, cluster_id: int) -> alt.Chart:
    """Build the publication timeline for one selected cluster."""
    subset = corpus[corpus["cluster_id"] == cluster_id]
    monthly = subset.groupby("month").size().reset_index(name="noticias")
    monthly["month_dt"] = pd.to_datetime(monthly["month"])
    return (
        alt.Chart(monthly)
        .mark_line(point=True, color="#8a4b08", strokeWidth=2.5)
        .encode(
            x=alt.X("month_dt:T", title="Mês"),
            y=alt.Y("noticias:Q", title="Notícias"),
            tooltip=["month", "noticias"],
        )
        .properties(height=260)
    )


def build_cluster_text_network(
    corpus: pd.DataFrame,
    clusters: pd.DataFrame,
    neighbors_per_cluster: int = 3,
    min_similarity: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the cluster network from aggregated cluster corpora."""
    edge_columns = [
        "source_cluster",
        "target_cluster",
        "source_x",
        "source_y",
        "source_z",
        "target_x",
        "target_y",
        "target_z",
        "similarity",
        "shared_count",
        "shared_terms",
    ]
    cluster_texts = (
        corpus.groupby("cluster_id")["texto_busca_normalizado"]
        .apply(lambda values: " ".join(str(value).strip() for value in values if str(value).strip()))
        .sort_index()
    )
    cluster_texts = cluster_texts[cluster_texts.astype(str).str.strip().ne("")]
    if cluster_texts.empty:
        return pd.DataFrame(), pd.DataFrame(columns=edge_columns)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.75,
        max_features=12000,
        token_pattern=r"(?u)\b[a-z0-9-]{2,}\b",
    )
    term_matrix = vectorizer.fit_transform(cluster_texts.tolist())
    if term_matrix.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame(columns=edge_columns)

    similarity = cosine_similarity(term_matrix)
    max_components = min(term_matrix.shape[0], term_matrix.shape[1]) - 1
    if max_components >= 1:
        reducer = TruncatedSVD(n_components=min(3, max_components), random_state=42)
        reduced = reducer.fit_transform(term_matrix)
    else:
        reduced = np.zeros((term_matrix.shape[0], 1))
    if reduced.shape[1] < 3:
        reduced = np.column_stack([reduced, np.zeros((reduced.shape[0], 3 - reduced.shape[1]))])

    coords = reduced[:, :3].astype(float)
    for axis in range(3):
        axis_values = coords[:, axis]
        spread = float(np.max(np.abs(axis_values))) if axis_values.size else 0.0
        if spread > 0:
            coords[:, axis] = axis_values / spread

    terms = np.asarray(vectorizer.get_feature_names_out())
    cluster_meta = clusters.set_index("cluster_id")
    dense_matrix = term_matrix.toarray()
    cluster_ids = cluster_texts.index.tolist()
    nodes = []
    for idx, cluster_id in enumerate(cluster_ids):
        meta = cluster_meta.loc[int(cluster_id)]
        weights = dense_matrix[idx]
        top_term_idx = weights.argsort()[::-1]
        dominant_terms = [terms[pos] for pos in top_term_idx if weights[pos] > 0][:6]
        nodes.append(
            {
                "cluster_id": int(cluster_id),
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "z": float(coords[idx, 2]),
                "size": int(meta.get("size", 0)),
                "active_years": int(meta.get("active_years", 0)),
                "top_terms": str(meta.get("top_terms", "")),
                "top_tags": str(meta.get("top_tags", "")),
                "top_crimes": str(meta.get("top_crimes", "")),
                "dominant_terms": " | ".join(dominant_terms),
                "color": CLUSTER_GRAPH_COLORS[idx % len(CLUSTER_GRAPH_COLORS)],
            }
        )
    nodes_df = pd.DataFrame(nodes).sort_values("cluster_id").reset_index(drop=True)

    node_lookup = nodes_df.set_index("cluster_id")
    edges = []
    seen_pairs: set[tuple[int, int]] = set()
    for i, source_cluster in enumerate(cluster_ids):
        candidate_pairs = []
        source_weights = dense_matrix[i]
        source_nonzero = source_weights > 0
        for j, target_cluster in enumerate(cluster_ids):
            if i == j:
                continue
            score = float(similarity[i, j])
            if score < min_similarity:
                continue
            target_weights = dense_matrix[j]
            overlap_mask = source_nonzero & (target_weights > 0)
            if not overlap_mask.any():
                continue
            overlap_strength = np.minimum(source_weights[overlap_mask], target_weights[overlap_mask])
            shared_terms = terms[overlap_mask][np.argsort(overlap_strength)[::-1]].tolist()
            candidate_pairs.append((target_cluster, score, shared_terms))

        candidate_pairs.sort(key=lambda item: item[1], reverse=True)
        for target_cluster, score, shared_terms in candidate_pairs[:neighbors_per_cluster]:
            pair_key = tuple(sorted((int(source_cluster), int(target_cluster))))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            source_row = node_lookup.loc[int(source_cluster)]
            target_row = node_lookup.loc[int(target_cluster)]
            edges.append(
                {
                    "source_cluster": int(source_cluster),
                    "target_cluster": int(target_cluster),
                    "source_x": float(source_row["x"]),
                    "source_y": float(source_row["y"]),
                    "source_z": float(source_row["z"]),
                    "target_x": float(target_row["x"]),
                    "target_y": float(target_row["y"]),
                    "target_z": float(target_row["z"]),
                    "similarity": round(score, 4),
                    "shared_count": len(shared_terms),
                    "shared_terms": " | ".join(shared_terms[:6]),
                }
            )

    edges_df = pd.DataFrame(edges, columns=edge_columns)
    if edges_df.empty:
        return nodes_df, edges_df
    edges_df = edges_df.sort_values(["similarity", "shared_count"], ascending=[False, False]).reset_index(drop=True)
    return nodes_df, edges_df


def isolated_cluster_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Return the cluster nodes that have no surviving edges in the network."""
    if nodes_df.empty:
        return nodes_df.copy()
    if edges_df.empty:
        return nodes_df.copy()
    connected_clusters = set(edges_df["source_cluster"].tolist()) | set(edges_df["target_cluster"].tolist())
    return nodes_df[~nodes_df["cluster_id"].isin(connected_clusters)].copy()


def cluster_network_3d_figure(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> go.Figure:
    """Render the Plotly 3D figure for the cluster network."""
    fig = go.Figure()

    for edge in edges_df.itertuples(index=False):
        fig.add_trace(
            go.Scatter3d(
                x=[edge.source_x, edge.target_x],
                y=[edge.source_y, edge.target_y],
                z=[edge.source_z, edge.target_z],
                mode="lines",
                line={"color": "rgba(88, 97, 109, 0.35)", "width": 2 + (edge.similarity * 8)},
                hovertemplate=(
                    f"Cluster {edge.source_cluster} ↔ Cluster {edge.target_cluster}<br>"
                    f"Similaridade: {edge.similarity:.3f}<br>"
                    f"Termos compartilhados: {edge.shared_terms}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    hover_custom = nodes_df[["cluster_id", "top_terms", "dominant_terms", "top_crimes", "size", "active_years"]].values
    fig.add_trace(
        go.Scatter3d(
            x=nodes_df["x"],
            y=nodes_df["y"],
            z=nodes_df["z"],
            mode="markers+text",
            text=nodes_df["cluster_id"].map(lambda cid: f"C{int(cid)}"),
            textposition="top center",
            customdata=hover_custom,
            hovertemplate=(
                "Cluster %{text}<br>"
                "Tamanho: %{customdata[4]}<br>"
                "Anos ativos: %{customdata[5]}<br>"
                "Top terms: %{customdata[1]}<br>"
                "Assinatura textual: %{customdata[2]}<br>"
                "Crimes dominantes: %{customdata[3]}<extra></extra>"
            ),
            marker={
                "size": np.clip(10 + (nodes_df["size"] / nodes_df["size"].max()) * 28, 10, 38),
                "color": nodes_df["color"],
                "line": {"color": "#f7f1e6", "width": 2},
                "opacity": 0.95,
            },
            showlegend=False,
        )
    )

    fig.update_layout(
        height=700,
        dragmode="orbit",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis": {"title": "Eixo semântico 1", "showbackground": False, "zeroline": False},
            "yaxis": {"title": "Eixo semântico 2", "showbackground": False, "zeroline": False},
            "zaxis": {"title": "Eixo semântico 3", "showbackground": False, "zeroline": False},
            "dragmode": "orbit",
            "camera": {"eye": {"x": 1.45, "y": 1.55, "z": 1.1}},
        },
    )
    return fig


def selected_cluster_from_network_event(event: object, nodes_df: pd.DataFrame, node_trace_index: int) -> int | None:
    """Extract the clicked cluster id from a Plotly selection event."""
    if not event:
        return None

    selection = None
    if isinstance(event, dict):
        selection = event.get("selection")
    else:
        selection = getattr(event, "selection", None)

    if not selection:
        return None

    if isinstance(selection, dict):
        points = selection.get("points", [])
    else:
        points = getattr(selection, "points", [])

    for point in points or []:
        if isinstance(point, dict):
            curve_number = point.get("curve_number", point.get("trace_index"))
            point_index = point.get("point_index", point.get("pointNumber"))
        else:
            curve_number = getattr(point, "curve_number", getattr(point, "trace_index", None))
            point_index = getattr(point, "point_index", getattr(point, "pointNumber", None))

        if curve_number != node_trace_index or point_index is None:
            continue
        if 0 <= int(point_index) < len(nodes_df):
            return int(nodes_df.iloc[int(point_index)]["cluster_id"])
    return None


def classify_series_type(series_row: pd.Series) -> str:
    """Classify a semantic series by naming pattern and content profile."""
    operation_names = str(series_row.get("operation_names", "") or "").strip()
    group_key = str(series_row.get("series_group_key", "") or "")
    llm_type = str(series_row.get("llm_type", "") or "").strip()
    llm_identity = str(series_row.get("llm_identity", "") or "").strip()
    if llm_type:
        return llm_type
    if operation_names and operation_names.lower() != "nan":
        return "Com operação nomeada"
    if llm_identity.startswith("crime_"):
        return "Por crime"
    if group_key.startswith("crime::"):
        return "Por crime"
    return "Outras"


def classify_series_strength(series_row: pd.Series) -> str:
    """Bucket a semantic series by recurrence strength."""
    size = int(series_row.get("size", 0) or 0)
    active_years = int(series_row.get("active_years", 0) or 0)
    strength_score = size * active_years
    if size >= 20 or strength_score >= 60 or (size >= 10 and active_years >= 4):
        return "Forte"
    if size >= 5 or active_years >= 3 or strength_score >= 15:
        return "Média"
    return "Fraca"


def prepare_series_catalog(series_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metadata used to browse the semantic series catalog."""
    catalog = series_df.copy()
    catalog["series_type"] = catalog.apply(classify_series_type, axis=1)
    catalog["strength_score"] = catalog["size"].fillna(0) * catalog["active_years"].fillna(0)
    catalog["strength_band"] = catalog.apply(classify_series_strength, axis=1)
    catalog["start_year"] = pd.to_datetime(catalog["first_date"], errors="coerce").dt.year
    catalog["end_year"] = pd.to_datetime(catalog["last_date"], errors="coerce").dt.year
    return catalog


def sort_series_catalog(series_df: pd.DataFrame, ranking_sort: str) -> pd.DataFrame:
    """Sort the semantic series catalog according to the selected ranking."""
    if ranking_sort == "Força":
        return series_df.sort_values(["strength_score", "size", "active_years"], ascending=[False, False, False])
    if ranking_sort == "Tamanho":
        return series_df.sort_values(["size", "active_years", "first_date"], ascending=[False, False, True])
    if ranking_sort == "Amplitude temporal":
        return series_df.sort_values(["active_years", "size", "first_date"], ascending=[False, False, True])
    return series_df.sort_values(["last_date", "size"], ascending=[False, False])


def series_timeline_chart(series_df: pd.DataFrame) -> alt.Chart:
    """Build the overview timeline of semantic series."""
    top_series = series_df.head(25).copy()
    top_series["serie"] = top_series["semantic_series_id"].map(lambda x: f"Série {int(x)}")
    return (
        alt.Chart(top_series)
        .mark_bar(size=16)
        .encode(
            x=alt.X("first_date:T", title="Início"),
            x2="last_date:T",
            y=alt.Y("serie:N", sort="-x", title="Série"),
            color=alt.Color(
                "strength_band:N",
                title="Força",
                scale=alt.Scale(
                    domain=["Forte", "Média", "Fraca"],
                    range=["#1d5c63", "#8a4b08", "#8d99ae"],
                ),
            ),
            tooltip=["semantic_series_id", "size", "active_years", "series_type", "strength_band", "operation_names", "crime_modes"],
        )
        .properties(height=420)
    )


def series_member_timeline_chart(members: pd.DataFrame) -> alt.Chart:
    """Build the member timeline for one semantic series."""
    monthly = members.groupby("month").size().reset_index(name="noticias")
    monthly["month_dt"] = pd.to_datetime(monthly["month"])
    return (
        alt.Chart(monthly)
        .mark_line(point=True, color="#1d5c63", strokeWidth=2.5)
        .encode(
            x=alt.X("month_dt:T", title="Mês"),
            y=alt.Y("noticias:Q", title="Notícias da série"),
            tooltip=["month", "noticias"],
        )
        .properties(height=220)
    )


def render_series_detail(series_row: pd.Series, members: pd.DataFrame, news_limit: int = 80) -> None:
    """Render metrics, charts, and examples for one semantic series."""
    a, b, c, d = st.columns(4)
    a.metric("Tamanho", int(series_row["size"]))
    b.metric("Anos ativos", int(series_row["active_years"]))
    c.metric("Início", pd.to_datetime(series_row["first_date"]).date().isoformat())
    d.metric("Fim", pd.to_datetime(series_row["last_date"]).date().isoformat())

    l1, l2 = st.columns([1.05, 0.95])
    with l1:
        st.markdown("**Leitura da série selecionada**")
        st.write(f"**Identidade da série:** {series_row['series_ui_label']}")
        st.write(f"**Tipo:** {series_row['series_type']}")
        st.write(f"**Força:** {series_row['strength_band']}")
        st.write(f"**Rótulo resumido:** {series_row['series_label']}")
        if str(series_row.get("llm_identity", "") or "").strip():
            st.write(f"**Identidade LLM:** {series_row['llm_identity']}")
        if str(series_row.get("llm_type", "") or "").strip():
            st.write(f"**Tipo LLM:** {series_row['llm_type']}")
        if str(series_row.get("llm_tags", "") or "").strip():
            st.write(f"**Tags LLM:** {series_row['llm_tags']}")
        st.write(f"**Nomes de operação encontrados:** {series_row['operation_names'] or 'Sem nome de operação explícito'}")
        st.write(f"**Crimes mais presentes:** {series_row['crime_modes']}")
        st.write(f"**Exemplos iniciais:** {series_row['sample_titles']}")
    with l2:
        st.markdown("**Pulso temporal da série**")
        st.altair_chart(series_member_timeline_chart(members), width='stretch')

    st.markdown("**Notícias da série**")
    st.dataframe(
        members[["data_publicacao_dt", "titulo", "cluster_id", "tags", "link"]].head(news_limit),
        width='stretch',
        hide_index=True,
    )


def classify_canonical_cluster_strength(cluster_row: pd.Series) -> str:
    """Bucket a canonical cluster by temporal and volumetric strength."""
    size = int(cluster_row.get("size", 0) or 0)
    active_years = int(cluster_row.get("active_years", 0) or 0)
    strength_score = size * active_years
    if size >= 30 or strength_score >= 80 or (size >= 12 and active_years >= 4):
        return "Forte"
    if size >= 8 or active_years >= 3 or strength_score >= 20:
        return "Media"
    return "Fraca"


def prepare_canonical_cluster_catalog(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metadata used to browse canonical clusters over time."""
    catalog = cluster_df.copy()
    catalog["strength_score"] = catalog["size"].fillna(0) * catalog["active_years"].fillna(0)
    catalog["strength_band"] = catalog.apply(classify_canonical_cluster_strength, axis=1)
    catalog["start_year"] = pd.to_datetime(catalog["first_date"], errors="coerce").dt.year
    catalog["end_year"] = pd.to_datetime(catalog["last_date"], errors="coerce").dt.year
    if "cluster_canonico_ui_label" not in catalog.columns:
        catalog["cluster_canonico_ui_label"] = catalog.apply(build_canonical_cluster_ui_label, axis=1)
    return catalog


def sort_canonical_cluster_catalog(cluster_df: pd.DataFrame, ranking_sort: str) -> pd.DataFrame:
    """Sort the canonical cluster catalog according to the selected ranking."""
    if ranking_sort == "Forca":
        return cluster_df.sort_values(["strength_score", "size", "active_years"], ascending=[False, False, False])
    if ranking_sort == "Tamanho":
        return cluster_df.sort_values(["size", "active_years", "first_date"], ascending=[False, False, True])
    if ranking_sort == "Amplitude temporal":
        return cluster_df.sort_values(["active_years", "size", "first_date"], ascending=[False, False, True])
    return cluster_df.sort_values(["last_date", "size"], ascending=[False, False])


def canonical_cluster_timeline_chart(cluster_df: pd.DataFrame) -> alt.Chart:
    """Build the overview timeline of canonical clusters."""
    top_clusters = cluster_df.head(25).copy()
    top_clusters["cluster"] = top_clusters["cluster_canonico_id"].map(lambda x: f"Cluster canonico {int(x)}")
    return (
        alt.Chart(top_clusters)
        .mark_bar(size=16)
        .encode(
            x=alt.X("first_date:T", title="Inicio"),
            x2="last_date:T",
            y=alt.Y("cluster:N", sort="-x", title="Cluster canonico"),
            color=alt.Color(
                "strength_band:N",
                title="Forca",
                scale=alt.Scale(
                    domain=["Forte", "Media", "Fraca"],
                    range=["#1d5c63", "#8a4b08", "#8d99ae"],
                ),
            ),
            tooltip=["cluster_canonico_id", "cluster_canonico_label", "size", "active_years", "cluster_canonico_tipo", "crime_modes"],
        )
        .properties(height=420)
    )


def canonical_cluster_member_timeline_chart(members: pd.DataFrame) -> alt.Chart:
    """Build the member timeline for one canonical cluster."""
    monthly = members.groupby("month").size().reset_index(name="noticias")
    monthly["month_dt"] = pd.to_datetime(monthly["month"])
    return (
        alt.Chart(monthly)
        .mark_line(point=True, color="#1d5c63", strokeWidth=2.5)
        .encode(
            x=alt.X("month_dt:T", title="Mes"),
            y=alt.Y("noticias:Q", title="Noticias do cluster canonico"),
            tooltip=["month", "noticias"],
        )
        .properties(height=220)
    )


def render_canonical_cluster_detail(cluster_row: pd.Series, members: pd.DataFrame, news_limit: int = 80) -> None:
    """Render metrics, charts, and examples for one canonical cluster."""
    a, b, c, d = st.columns(4)
    a.metric("Tamanho", int(cluster_row["size"]))
    b.metric("Anos ativos", int(cluster_row["active_years"]))
    c.metric("Inicio", pd.to_datetime(cluster_row["first_date"]).date().isoformat())
    d.metric("Fim", pd.to_datetime(cluster_row["last_date"]).date().isoformat())

    l1, l2 = st.columns([1.05, 0.95])
    with l1:
        st.markdown("**Leitura do cluster canonico**")
        st.write(f"**Identidade canonica:** {cluster_row['cluster_canonico_label']}")
        st.write(f"**Tipo:** {cluster_row['cluster_canonico_tipo']}")
        st.write(f"**Forca:** {cluster_row['strength_band']}")
        st.write(f"**Rotulo de interface:** {cluster_row['cluster_canonico_ui_label']}")
        if str(cluster_row.get("llm_tags", "") or "").strip():
            st.write(f"**Tags LLM dominantes:** {cluster_row['llm_tags']}")
        st.write(f"**Operacoes associadas:** {cluster_row['operation_names'] or 'Sem nome de operacao explicito'}")
        st.write(f"**Crimes mais presentes:** {cluster_row['crime_modes']}")
        st.write(f"**Exemplos iniciais:** {cluster_row['sample_titles']}")
    with l2:
        st.markdown("**Pulso temporal do cluster canonico**")
        st.altair_chart(canonical_cluster_member_timeline_chart(members), width='stretch')

    st.markdown("**Noticias do cluster canonico**")
    st.dataframe(
        members[["data_publicacao_dt", "titulo", "cluster_id", "tags", "link"]].head(news_limit),
        width='stretch',
        hide_index=True,
    )


def get_markdown_excerpt(path_value: str, limit: int = 6000) -> str:
    """Load and truncate the extracted markdown for one news item."""
    path = BASE_DIR / path_value
    if not path.exists():
        return "Conteúdo markdown não encontrado."
    text = path.read_text(encoding="utf-8")
    return text[:limit] + ("..." if len(text) > limit else "")


def render_metric_row() -> None:
    """Render the top metric row shown on the overview page."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Notícias", f"{len(CORPUS):,}".replace(",", "."))
    c2.metric("Clusters", int(CORPUS["cluster_id"].nunique()))
    c3.metric("Clusters canônicos", int(CANONICAL_CLUSTERS.shape[0]))
    c4.metric("Pares recorrentes", f"{len(PAIRS):,}".replace(",", "."))
    c5.metric("Período", f"{int(CORPUS['year'].min())}-{int(CORPUS['year'].max())}")


def render_story_overview() -> None:
    """Render the opening narrative and overview charts."""
    st.markdown(
        """
        <div class="hero">
            <h1>Atlas Analítico das Operações da PF</h1>
            <p>Este painel narra a jornada dos artefatos gerados no projeto: da coleta ao texto integral, da similaridade semântica aos ciclos de repetição no tempo.</p>
            <p>Em vez de ler milhares de notícias isoladamente, você pode percorrer camadas: panorama, clusters, crimes, modus operandi, tempo por clusters canônicos e pares quase gêmeos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_row()
    st.markdown(
        """
        <div class="section-note">
            A lógica do app é sequencial: primeiro ele mostra o pulso do acervo, depois aprofunda crimes e modus com filtros comparativos e por fim organiza o tempo por identidades canônicas, deixando o clustering não supervisionado como apoio exploratório.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("1. Quando as notícias se concentram")
        st.altair_chart(monthly_volume_chart(CORPUS), width='stretch')
    with right:
        st.subheader("2. Que crimes dominam o acervo")
        st.altair_chart(top_crime_chart(CRIMES), width='stretch')

    c1, c2 = st.columns([1.45, 0.75])
    with c1:
        st.subheader("3. Os maiores universos temáticos")
        st.altair_chart(cluster_size_chart(CLUSTERS), width='stretch')
    with c2:
        st.subheader("4. Como ler os artefatos")
        inventory = artifact_inventory()
        for _, row in inventory.head(4).iterrows():
            st.markdown(
                f"""
                <div class="artifact-card">
                    <h4>{row['artefato']}</h4>
                    <p><strong>Papel:</strong> {row['papel']}</p>
                    <p><strong>Conta:</strong> {row['conta']}</p>
                    <p><strong>Pergunta:</strong> {row['pergunta']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )



def render_crimes_modus() -> None:
    """Render the crimes and modus exploration section."""
    st.subheader("Crimes e Modus Operandi")
    st.markdown(
        """
        <div class="section-note">
            Esta camada mostra o que está sendo investigado e como a atuação aparece no texto. É a melhor entrada para análise qualitativa transversal.
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_uf = st.selectbox("Filtro geral por UF", available_ufs(), key="crime_state_filter")
    filtered_corpus = filter_corpus_by_uf(CORPUS, selected_uf)
    filtered_crimes = summarize_labels_from_corpus(filtered_corpus, "crime_labels_list", "crime_label")
    filtered_modus = summarize_labels_from_corpus(filtered_corpus, "modus_labels_list", "modus_label")

    if filtered_corpus.empty:
        st.info("Nenhuma noticia encontrada para o estado selecionado.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(label_heatmap(filtered_crimes, "crime_label", "Crime"), width='stretch')
    with c2:
        st.altair_chart(label_heatmap(filtered_modus, "modus_label", "Modus"), width='stretch')

    crime_totals = filtered_crimes.groupby("crime_label", as_index=False)["noticias"].sum().sort_values("noticias", ascending=False)
    modus_totals = filtered_modus.groupby("modus_label", as_index=False)["noticias"].sum().sort_values("noticias", ascending=False)
    cc1, cc2 = st.columns(2)
    with cc1:
        if crime_totals.empty:
            st.info("Sem crimes rotulados neste recorte.")
        else:
            crime_choice = st.selectbox(
                "Crime em foco",
                crime_totals["crime_label"].tolist(),
                format_func=pretty_label,
            )
            subset = filtered_crimes[filtered_crimes["crime_label"] == crime_choice].sort_values("ano")
            st.line_chart(subset.set_index("ano")["noticias"], height=260)
    with cc2:
        if modus_totals.empty:
            st.info("Sem modus rotulados neste recorte.")
        else:
            modus_choice = st.selectbox(
                "Modus em foco",
                modus_totals["modus_label"].tolist(),
                format_func=pretty_label,
            )
            subset = filtered_modus[filtered_modus["modus_label"] == modus_choice].sort_values("ano")
            st.line_chart(subset.set_index("ano")["noticias"], height=260)

    st.caption("Leitura rápida: crimes ajudam a entender o conteúdo do fato; modus operandi ajuda a entender o repertório de ação institucional.")

    st.subheader("Comparador temporal de termos")
    st.markdown(
        """
        <div class="section-note">
            Aqui a ideia é funcionar como um Google Trends interno do corpus. Você escolhe termos ou expressões e o painel mostra quando eles ganham força relativa.
        </div>
        """,
        unsafe_allow_html=True,
    )
    t1, t2, t3, t4 = st.columns(4)
    mode = t1.selectbox("Base da comparacao", ["Texto livre", "Tag", "Crime", "Modus"])
    termo_a = t2.text_input("Comparador A", value="lavagem de dinheiro")
    termo_b = t3.text_input("Comparador B", value="criptomoeda")
    period = t4.selectbox("Granularidade", ["Ano", "Mes"])
    corpus_a = filtered_corpus
    corpus_b = filtered_corpus
    trend_parts = []
    if termo_a.strip():
        trend_parts.append(build_signal_trend(corpus_a, [termo_a], period=period, mode=mode))
    if termo_b.strip():
        trend_parts.append(build_signal_trend(corpus_b, [termo_b], period=period, mode=mode))
    trend_df = pd.concat([part for part in trend_parts if not part.empty], ignore_index=True) if trend_parts else pd.DataFrame()
    if not trend_df.empty:
        time_col = "year" if period == "Ano" else "month"
        x_encoding = alt.X(f"{time_col}:O", title=period)
        color_scale_chart = alt.Scale(domain=[termo_a, termo_b], range=["#2d7dd2", "#d64848"])
        trend_chart = (
            alt.Chart(trend_df)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=x_encoding,
                y=alt.Y("indice_google_like:Q", title="Indice 0-100"),
                color=alt.Color("termo:N", title="Comparador", scale=color_scale_chart),
                tooltip=[time_col, "termo", "termo_resolvido", "citacoes", "share", "score_busca", "indice_google_like"],
            )
            .properties(height=320)
        )

        summary = (
            trend_df.sort_values(["termo", "citacoes"], ascending=[True, False])
            .groupby("termo", as_index=False)
            .first()[["termo", "termo_resolvido", "score_busca", time_col, "citacoes", "share"]]
            .rename(
                columns={
                    time_col: "pico_periodo",
                    "citacoes": "pico_citacoes",
                    "share": "share_no_pico",
                    "termo_resolvido": "busca_resolvida",
                }
            )
        )
        mask_a, resolved_a, score_a = match_signal(corpus_a, mode, termo_a)
        mask_b, resolved_b, score_b = match_signal(corpus_b, mode, termo_b)
        map_df = aggregate_states_for_dual_map(corpus_a[mask_a], corpus_b[mask_b])

        left, right = st.columns([1.2, 1])
        with left:
            st.altair_chart(trend_chart, width='stretch')
            st.dataframe(summary, width='stretch', hide_index=True)
            st.caption("A busca aproximada usa similaridade textual para tolerar erro de digitacao e tenta resolver o termo mais proximo no corpus.")
        with right:
            dual_state_map(
                map_df,
                f"{termo_a} | resolvido: {resolved_a} | score={score_a:.2f}",
                f"{termo_b} | resolvido: {resolved_b} | score={score_b:.2f}",
            )

    st.subheader("Mapa do Crime")
    st.markdown(
        """
        <div class="section-note">
            Este mapa mostra em que estados um crime aparece com mais força no corpus. Você pode escolher o tipo de crime e recortar por ano para acompanhar a intensidade territorial ao longo do tempo.
        </div>
        """,
        unsafe_allow_html=True,
    )
    crime_state_summary = crime_state_year_summary(filtered_corpus)
    crime_options = sorted(crime_state_summary["crime_label"].dropna().unique().tolist())
    if crime_options:
        c1, c2 = st.columns([1, 1])
        selected_crime_map = c1.selectbox("Crime no mapa", crime_options, format_func=pretty_label, key="crime_map_select")
        selected_year_map = c2.selectbox(
            "Ano do mapa",
            ["Todos"] + [str(int(year)) for year in sorted(crime_state_summary["ano"].dropna().unique().tolist())],
            key="crime_map_year",
        )
        geojson, map_subset = crime_map_geojson(crime_state_summary, selected_crime_map, selected_year_map)
        total_mentions = int(map_subset["noticias"].sum()) if not map_subset.empty else 0
        strongest_state = (
            map_subset.sort_values("noticias", ascending=False).iloc[0]["state"]
            if not map_subset.empty
            else "Sem dados"
        )
        strongest_count = (
            int(map_subset.sort_values("noticias", ascending=False).iloc[0]["noticias"])
            if not map_subset.empty
            else 0
        )

        left, right = st.columns([1.15, 0.85])
        with left:
            crime_choropleth(
                geojson,
                f"Darker = maior concentração de menções para {pretty_label(selected_crime_map)} em {selected_year_map}.",
            )
        with right:
            a, b = st.columns(2)
            a.metric("Estado lider", strongest_state)
            b.metric("Menções no pico", strongest_count)
            st.metric("Menções totais no recorte", total_mentions)
            st.altair_chart(crime_state_trend_chart(crime_state_summary, selected_crime_map), width="stretch")
    else:
        st.info("Nao ha dados suficientes para montar o mapa do crime neste recorte.")


def render_clusters(neighbors_per_cluster: int, min_similarity: float) -> None:
    """Render the cluster exploration section and its 3D network."""
    st.subheader("Clusters Semânticos")
    st.markdown(
        """
        <div class="section-note">
            Cada cluster agrupa noticias que compartilham vocabulario, temas e contextos muito proximos. Eles sao o mapa estrutural do acervo. A proximidade entre clusters na rede 3D e calculada a partir do corpus textual agregado de cada cluster, isto e, pela uniao dos textos das noticias que pertencem a ele.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Recorte atual da rede: ate {neighbors_per_cluster} vizinho(s) por cluster, com similaridade minima de {min_similarity:.2f}."
    )
    cluster_options = CLUSTERS["cluster_id"].tolist()
    selected_cluster = st.selectbox(
        "Escolha um cluster",
        cluster_options,
        format_func=lambda cid: f"Cluster {cid} | {CLUSTERS.loc[CLUSTERS['cluster_id'] == cid, 'top_terms'].iloc[0]}",
    )
    cluster_row = CLUSTERS[CLUSTERS["cluster_id"] == selected_cluster].iloc[0]
    subset = CORPUS[CORPUS["cluster_id"] == selected_cluster].sort_values("data_publicacao_dt", ascending=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tamanho", int(cluster_row["size"]))
    m2.metric("Anos ativos", int(cluster_row["active_years"]))
    m3.metric("Primeira notícia", pd.to_datetime(cluster_row["first_date"]).date().isoformat())
    m4.metric("Última notícia", pd.to_datetime(cluster_row["last_date"]).date().isoformat())

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            f"""
            <div class="artifact-card">
                <p><strong>Top terms:</strong> {cluster_row['top_terms']}</p>
                <p><strong>Crimes dominantes:</strong> {cluster_row['top_crimes']}</p>
                <p><strong>Modus dominantes:</strong> {cluster_row['top_modus']}</p>
                <p><strong>Como a proximidade e medida:</strong> a rede compara este cluster com os demais usando o corpus textual agregado do cluster, e nao apenas tags isoladas.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(cluster_timeline_chart(CORPUS, selected_cluster), width='stretch')
    with right:
        st.caption("A geografia foi removida da leitura semantica do cluster e aparece separadamente no mapa.")
        state_map(aggregate_states_from_corpus(subset), "Mapa de estados citados nas noticias deste cluster.")
    st.dataframe(
        subset[["data_publicacao_dt", "titulo", "tags", "cluster_canonico_label"]].head(30),
        width='stretch',
        hide_index=True,
    )

    network_nodes, network_edges = build_cluster_text_network(
        CORPUS[["cluster_id", "texto_busca_normalizado"]],
        CLUSTERS,
        neighbors_per_cluster=neighbors_per_cluster,
        min_similarity=min_similarity,
    )
    isolated_nodes = isolated_cluster_nodes(network_nodes, network_edges)
    st.markdown("### Rede 3D dos clusters")
    st.caption("Cada no e um cluster. A posicao vem da projecao tridimensional do corpus textual agregado de cada cluster, e as arestas ligam os vizinhos mais proximos por similaridade textual entre esses corpus. Ao clicar em um no, o painel abaixo tenta abrir as noticias desse cluster.")
    if not isolated_nodes.empty:
        isolated_labels = [f"C{int(cluster_id)}" for cluster_id in isolated_nodes["cluster_id"].tolist()]
        preview = ", ".join(isolated_labels[:8])
        if len(isolated_labels) > 8:
            preview += ", ..."
        st.info(
            f"Ha {len(isolated_nodes)} cluster(s) solto(s) nesta rede. Isso significa que, neste corte de similaridade e vizinhanca, eles nao formaram conexoes fortes o suficiente com outros clusters. Em geral isso acontece quando o vocabulario agregado e muito especifico, o cluster e pequeno, ou a semelhanca fica abaixo do limiar atual. Exemplos: {preview}."
        )
    if not network_nodes.empty and not network_edges.empty:
        left, right = st.columns([1.25, 0.75])
        figure = cluster_network_3d_figure(network_nodes, network_edges)
        node_trace_index = len(network_edges)
        with left:
            event = st.plotly_chart(
                figure,
                width='stretch',
                key="cluster_network_3d",
                on_select="rerun",
                selection_mode=("points"),
                config={
                    "scrollZoom": False,
                    "displaylogo": False,
                },
            )
        with right:
            st.info("As conexoes abaixo sao calculadas pela similaridade entre os corpus textuais agregados de cada cluster. Os termos exibidos ajudam a entender por que dois clusters ficaram proximos.")
            st.markdown("**Conexões mais fortes**")
            st.dataframe(
                network_edges.head(12)[["source_cluster", "target_cluster", "similarity", "shared_terms"]]
                .rename(
                    columns={
                        "source_cluster": "Cluster A",
                        "target_cluster": "Cluster B",
                        "similarity": "Similaridade",
                        "shared_terms": "Termos que conectam",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption("A rede mostra apenas os vizinhos mais fortes de cada cluster e aplica um limiar minimo de similaridade. Por isso, alguns clusters podem aparecer sem arestas nesta visualizacao.")

        clicked_cluster = selected_cluster_from_network_event(event, network_nodes, node_trace_index)
        manual_cluster = st.selectbox(
            "Ou escolha manualmente um cluster da rede",
            network_nodes["cluster_id"].tolist(),
            index=network_nodes["cluster_id"].tolist().index(clicked_cluster) if clicked_cluster in network_nodes["cluster_id"].tolist() else 0,
            format_func=lambda cid: f"Cluster {cid} | {CLUSTERS.loc[CLUSTERS['cluster_id'] == cid, 'top_terms'].iloc[0]}",
            key="cluster_network_manual_select",
        )
        detail_cluster = clicked_cluster if clicked_cluster is not None else manual_cluster
        detail_subset = CORPUS[CORPUS["cluster_id"] == detail_cluster].sort_values("data_publicacao_dt", ascending=False)
        st.markdown(f"**Notícias conectadas ao nó selecionado:** Cluster {detail_cluster}")
        st.dataframe(
            detail_subset[["data_publicacao_dt", "titulo", "tags", "cluster_canonico_label", "link"]].head(80),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Não houve sinal suficiente para montar a rede 3D de clusters neste recorte.")


def render_series() -> None:
    """Render the recurring semantic series section."""
    st.subheader("Séries Recorrentes")
    st.markdown(
        """
        <div class="section-note">
            Séries semânticas são cadeias de notícias muito próximas entre si ao longo do tempo. Aqui aparecem continuidades operacionais, fases, retomadas e padrões reincidentes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Como a série é feita: o pipeline encontra vizinhos semânticos por similaridade do cosseno, mantém pares recorrentes com distância temporal mínima, conecta esses pares em componentes e depois consolida componentes com o mesmo rótulo de série.")

    catalog = prepare_series_catalog(SERIES)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Séries totais", int(catalog.shape[0]))
    m2.metric("Com operação nomeada", int((catalog["series_type"] == "Com operação nomeada").sum()))
    m3.metric("Por crime", int((catalog["series_type"] == "Por crime").sum()))
    m4.metric("Séries fortes", int((catalog["strength_band"] == "Forte").sum()))
    st.caption("Leitura rápida: `Forte` combina volume e recorrência; `Média` indica continuidade moderada; `Fraca` costuma ser curta ou ainda pouco recorrente.")
    executive_tab, detail_tab = st.tabs(["Visão Executiva", "Exploração Detalhada"])

    with executive_tab:
        st.caption("Esta aba prioriza as séries mais robustas e recorrentes para apresentação e leitura rápida.")
        executive_catalog = sort_series_catalog(catalog[catalog["size"] >= 5].copy(), "Força")
        if executive_catalog.empty:
            executive_catalog = sort_series_catalog(catalog.copy(), "Força")

        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown("**Linha do tempo das séries prioritárias**")
            st.altair_chart(series_timeline_chart(executive_catalog), width='stretch')
        with right:
            st.markdown("**Ranking executivo**")
            st.dataframe(
                executive_catalog.head(15)[["semantic_series_id", "strength_band", "series_type", "size", "active_years", "series_ui_label"]]
                .rename(
                    columns={
                        "semantic_series_id": "Série",
                        "strength_band": "Força",
                        "series_type": "Tipo",
                        "size": "n",
                        "active_years": "Anos ativos",
                        "series_ui_label": "Identidade",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

        executive_series = st.selectbox(
            "Série em destaque",
            executive_catalog["semantic_series_id"].tolist(),
            format_func=lambda sid: f"Série {int(sid)} | {executive_catalog.loc[executive_catalog['semantic_series_id'] == sid, 'series_ui_label'].iloc[0]}",
            key="executive_series_select",
        )
        executive_row = executive_catalog[executive_catalog["semantic_series_id"] == executive_series].iloc[0]
        executive_members = CORPUS[CORPUS["semantic_series_id"] == executive_series].sort_values("data_publicacao_dt")
        render_series_detail(executive_row, executive_members, news_limit=40)

    with detail_tab:
        f1, f2, f3, f4 = st.columns([0.95, 0.8, 0.9, 1.15])
        series_type_filter = f1.selectbox("Tipo de série", ["Todas", "Com operação nomeada", "Por crime", "Outras"])
        min_size = int(f2.selectbox("Tamanho mínimo", [2, 3, 5, 8, 10], index=1))
        ranking_sort = f3.selectbox("Ordenar por", ["Força", "Tamanho", "Amplitude temporal", "Mais recente"])
        series_search = f4.text_input("Buscar na identidade da série", value="")

        filtered_catalog = catalog[catalog["size"] >= min_size].copy()
        if series_type_filter != "Todas":
            filtered_catalog = filtered_catalog[filtered_catalog["series_type"] == series_type_filter]
        if series_search.strip():
            lookup = series_search.strip()
            filtered_catalog = filtered_catalog[
                filtered_catalog["series_ui_label"].str.contains(lookup, case=False, na=False)
                | filtered_catalog["crime_modes"].fillna("").str.contains(lookup, case=False, na=False)
                | filtered_catalog["operation_names"].fillna("").str.contains(lookup, case=False, na=False)
            ]

        if filtered_catalog.empty:
            st.info("Nenhuma série atende a esse recorte. Tente reduzir o tamanho mínimo ou trocar o tipo.")
            return

        filtered_catalog = sort_series_catalog(filtered_catalog, ranking_sort)

        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown("**Linha do tempo das séries filtradas**")
            st.altair_chart(series_timeline_chart(filtered_catalog), width='stretch')
        with right:
            st.markdown("**Séries prioritárias no recorte**")
            st.dataframe(
                filtered_catalog.head(20)[["semantic_series_id", "strength_band", "series_type", "size", "active_years", "series_ui_label"]]
                .rename(
                    columns={
                        "semantic_series_id": "Série",
                        "strength_band": "Força",
                        "series_type": "Tipo",
                        "size": "n",
                        "active_years": "Anos ativos",
                        "series_ui_label": "Identidade",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

        selected_series = st.selectbox(
            "Escolha uma série",
            filtered_catalog["semantic_series_id"].tolist(),
            format_func=lambda sid: f"Série {int(sid)} | {filtered_catalog.loc[filtered_catalog['semantic_series_id'] == sid, 'series_ui_label'].iloc[0]}",
            key="detailed_series_select",
        )
        series_row = filtered_catalog[filtered_catalog["semantic_series_id"] == selected_series].iloc[0]
        members = CORPUS[CORPUS["semantic_series_id"] == selected_series].sort_values("data_publicacao_dt")
        render_series_detail(series_row, members, news_limit=80)


def render_canonical_clusters() -> None:
    """Render the time-by-canonical-cluster section."""
    st.subheader("Tempo por Clusters Canônicos")
    st.markdown(
        """
        <div class="section-note">
            Aqui a unidade principal não é mais a série semântica. O painel passa a acompanhar identidades canônicas estáveis ao longo do tempo, como crime_abuso_sexual_infantil, deixando o cluster não supervisionado como apoio exploratório.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("A lógica é direta: a LLM padroniza a identidade da notícia, o pipeline agrupa por essa identidade e a linha do tempo mostra volume, duração e repetição do cluster canônico.")

    catalog = prepare_canonical_cluster_catalog(CANONICAL_CLUSTERS)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters canônicos", int(catalog.shape[0]))
    m2.metric("Com operação nomeada", int((catalog["cluster_canonico_tipo"] == "Com operacao nomeada").sum()))
    m3.metric("Por crime", int((catalog["cluster_canonico_tipo"] == "Por crime").sum()))
    m4.metric("Clusters fortes", int((catalog["strength_band"] == "Forte").sum()))
    st.caption("Leitura rápida: `Forte` combina volume e persistência temporal; `Media` indica recorrência moderada; `Fraca` tende a ser pontual ou ainda pequena.")
    executive_tab, detail_tab = st.tabs(["Visão Executiva", "Exploração Detalhada"])

    with executive_tab:
        st.caption("Esta aba prioriza as identidades canônicas mais robustas e recorrentes para leitura rápida.")
        executive_catalog = sort_canonical_cluster_catalog(catalog[catalog["size"] >= 5].copy(), "Forca")
        if executive_catalog.empty:
            executive_catalog = sort_canonical_cluster_catalog(catalog.copy(), "Forca")

        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown("**Linha do tempo dos clusters canônicos prioritários**")
            st.altair_chart(canonical_cluster_timeline_chart(executive_catalog), width='stretch')
        with right:
            st.markdown("**Ranking executivo**")
            st.dataframe(
                executive_catalog.head(15)[["cluster_canonico_id", "strength_band", "cluster_canonico_tipo", "size", "active_years", "cluster_canonico_ui_label"]]
                .rename(
                    columns={
                        "cluster_canonico_id": "Cluster",
                        "strength_band": "Força",
                        "cluster_canonico_tipo": "Tipo",
                        "size": "n",
                        "active_years": "Anos ativos",
                        "cluster_canonico_ui_label": "Identidade",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

        executive_cluster = st.selectbox(
            "Cluster canônico em destaque",
            executive_catalog["cluster_canonico_id"].tolist(),
            format_func=lambda cid: f"Cluster {int(cid)} | {executive_catalog.loc[executive_catalog['cluster_canonico_id'] == cid, 'cluster_canonico_ui_label'].iloc[0]}",
            key="executive_canonical_cluster_select",
        )
        executive_row = executive_catalog[executive_catalog["cluster_canonico_id"] == executive_cluster].iloc[0]
        executive_members = CORPUS[CORPUS["cluster_canonico_id"] == executive_cluster].sort_values("data_publicacao_dt")
        render_canonical_cluster_detail(executive_row, executive_members, news_limit=40)

    with detail_tab:
        f1, f2, f3, f4 = st.columns([0.95, 0.8, 0.9, 1.15])
        cluster_type_filter = f1.selectbox("Tipo de cluster", ["Todas", "Com operacao nomeada", "Por crime", "Outras"])
        min_size = int(f2.selectbox("Tamanho mínimo", [2, 3, 5, 8, 10], index=1))
        ranking_sort = f3.selectbox("Ordenar por", ["Forca", "Tamanho", "Amplitude temporal", "Mais recente"])
        cluster_search = f4.text_input("Buscar na identidade canônica", value="")

        filtered_catalog = catalog[catalog["size"] >= min_size].copy()
        if cluster_type_filter != "Todas":
            filtered_catalog = filtered_catalog[filtered_catalog["cluster_canonico_tipo"] == cluster_type_filter]
        if cluster_search.strip():
            lookup = cluster_search.strip()
            filtered_catalog = filtered_catalog[
                filtered_catalog["cluster_canonico_ui_label"].str.contains(lookup, case=False, na=False)
                | filtered_catalog["crime_modes"].fillna("").str.contains(lookup, case=False, na=False)
                | filtered_catalog["operation_names"].fillna("").str.contains(lookup, case=False, na=False)
            ]

        if filtered_catalog.empty:
            st.info("Nenhum cluster canônico atende a esse recorte. Tente reduzir o tamanho mínimo ou trocar o tipo.")
            return

        filtered_catalog = sort_canonical_cluster_catalog(filtered_catalog, ranking_sort)

        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown("**Linha do tempo dos clusters canônicos filtrados**")
            st.altair_chart(canonical_cluster_timeline_chart(filtered_catalog), width='stretch')
        with right:
            st.markdown("**Clusters canônicos prioritários no recorte**")
            st.dataframe(
                filtered_catalog.head(20)[["cluster_canonico_id", "strength_band", "cluster_canonico_tipo", "size", "active_years", "cluster_canonico_ui_label"]]
                .rename(
                    columns={
                        "cluster_canonico_id": "Cluster",
                        "strength_band": "Força",
                        "cluster_canonico_tipo": "Tipo",
                        "size": "n",
                        "active_years": "Anos ativos",
                        "cluster_canonico_ui_label": "Identidade",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

        selected_cluster = st.selectbox(
            "Escolha um cluster canônico",
            filtered_catalog["cluster_canonico_id"].tolist(),
            format_func=lambda cid: f"Cluster {int(cid)} | {filtered_catalog.loc[filtered_catalog['cluster_canonico_id'] == cid, 'cluster_canonico_ui_label'].iloc[0]}",
            key="detailed_canonical_cluster_select",
        )
        cluster_row = filtered_catalog[filtered_catalog["cluster_canonico_id"] == selected_cluster].iloc[0]
        members = CORPUS[CORPUS["cluster_canonico_id"] == selected_cluster].sort_values("data_publicacao_dt")
        render_canonical_cluster_detail(cluster_row, members, news_limit=80)


def render_neighbors() -> None:
    """Render the nearest-neighbor exploration section."""
    st.subheader("Vizinhança Semântica e Casos Muito Próximos")
    st.markdown(
        """
        <div class="section-note">
            Esta camada permite sair do agregado e voltar ao caso. Ela mostra quais notícias ficam semanticamente mais próximas de uma notícia-fonte.
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input("Filtrar notícia por palavra no título", value="Nova Aliança")
    candidates = CORPUS[CORPUS["titulo"].str.contains(query, case=False, na=False)].copy()
    if candidates.empty:
        st.warning("Nenhuma notícia encontrada com esse filtro.")
        return

    selected_link = st.selectbox(
        "Selecione a notícia-fonte",
        candidates["link"].tolist(),
        format_func=lambda link: CORPUS.loc[CORPUS["link"] == link, "titulo"].iloc[0],
    )
    source = CORPUS[CORPUS["link"] == selected_link].iloc[0]
    neighbor_rows = (
        PAIRS[(PAIRS["source_link"] == selected_link) | (PAIRS["target_link"] == selected_link)]
        .sort_values(["cosine_similarity", "gap_days"], ascending=[False, False])
        .head(20)
    )

    c1, c2 = st.columns([1.15, 0.95])
    with c1:
        st.markdown("**Leitura do markdown**")
        st.caption("O texto abaixo respeita leitura vertical, com quebra de linha normal e rolagem para baixo.")
        with st.container(border=True, height=560):
            st.markdown(get_markdown_excerpt(source["markdown_path"]))
    with c2:
        st.markdown("**Metadados da notícia-fonte**")
        st.markdown(f"**Título:** {source['titulo']}")
        st.markdown(f"**Data:** {source['data_publicacao_dt']}")
        st.markdown(f"**Cluster:** {source['cluster_label']}")
        st.markdown(f"**Cluster canônico:** {source['cluster_canonico_label']}")
        st.markdown(f"**Tags:** {source['tags']}")
        st.markdown("**Vizinhos mais próximos**")
        st.dataframe(
            neighbor_rows[
                [
                    "source_titulo",
                    "target_titulo",
                    "cosine_similarity",
                    "gap_days",
                    "source_operation_name",
                    "target_operation_name",
                ]
            ].rename(
                columns={
                    "source_titulo": "Título-fonte",
                    "target_titulo": "Título vizinho",
                    "cosine_similarity": "Similaridade",
                    "gap_days": "Distância temporal",
                    "source_operation_name": "Operação-fonte",
                    "target_operation_name": "Operação vizinha",
                }
            ),
            width='stretch',
            hide_index=True,
        )


def render_artifacts() -> None:
    """Render the artifact inventory and consolidated report."""
    st.subheader("História dos Artefatos")
    st.markdown(
        """
        <div class="section-note">
            Os artefatos não são apenas arquivos. Eles formam uma narrativa de análise: base, síntese, recorrência, explicação e evidência.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(artifact_inventory(), width='stretch', hide_index=True)
    st.markdown("### Relatório narrativo consolidado")
    st.markdown(REPORT)


with st.sidebar:
    st.markdown("## Navegação")
    section = st.radio(
        "Percurso",
        [
            "Panorama",
            "Crimes e Modus",
            "Clusters",
            "Tempo por Clusters Canônicos",
            "Vizinhança Semântica",
            "Artefatos",
        ],
    )
    st.markdown("---")
    st.markdown("### Leitura sugerida")
    st.caption("1. Panorama")
    st.caption("2. Crimes e Modus")
    st.caption("3. Clusters")
    st.caption("4. Tempo por Clusters Canônicos")
    st.caption("5. Vizinhança Semântica")
    st.caption("6. Artefatos")
    cluster_neighbors_per_node = 3
    cluster_min_similarity = 0.12
    if section == "Clusters":
        st.markdown("---")
        st.markdown("### Parametros da Rede")
        cluster_neighbors_per_node = st.slider(
            "Vizinhos por cluster",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            help="Define quantas conexoes maximas cada cluster pode manter na rede 3D.",
        )
        cluster_min_similarity = st.slider(
            "Similaridade minima",
            min_value=0.05,
            max_value=0.60,
            value=0.12,
            step=0.01,
            help="Filtra conexoes fracas. Quanto maior o valor, mais seletiva fica a rede.",
        )



if section == "Panorama":
    render_story_overview()
elif section == "Crimes e Modus":
    render_crimes_modus()
elif section == "Clusters":
    render_clusters(
        neighbors_per_cluster=cluster_neighbors_per_node,
        min_similarity=cluster_min_similarity,
    )
elif section == "Tempo por Clusters Canônicos":
    render_canonical_clusters()
elif section == "Vizinhança Semântica":
    render_neighbors()
else:
    render_artifacts()
