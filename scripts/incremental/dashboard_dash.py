from __future__ import annotations

import ast
import json
import math
import os
import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from plotly.subplots import make_subplots

from scripts.incremental.common import ANALYSIS_DIR, EVENTS_JSONL, WNN_FEATURE_BANK_PATH
from scripts.incremental.dashboard_comparacao import (
    _classification_label,
    _compare_classifications,
    _latest_snapshot,
    _read_json,
    _read_lot_classifications,
    _read_metrics,
    _summarize_run,
)


REFRESH_MS = 10_000
THEMES_PATH = ANALYSIS_DIR / "incremental" / "temas_canonicos_agent1.json"
CLUSTER_SUMMARY_PATH = ANALYSIS_DIR / "incremental" / "resumo_clusters_amostra.csv"
LEGACY_RULE_TOKEN = "re" + "gex"
WNN_THEME_PARENTS: dict[str, str] = {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float_percent(value: Any) -> float:
    text = str(value or "").strip().replace("%", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def _card(title: str, value: Any, subtitle: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(str(value), className="kpi-value"),
            html.Div(subtitle, className="kpi-subtitle"),
        ],
        className="kpi",
    )


def _summary_cards(current: dict[str, Any], previous: dict[str, Any], comparison: dict[str, Any]) -> list[html.Div]:
    docs = _safe_int(current.get("docs_processados"))
    wnn = _safe_int(current.get("wnn_accepted"))
    llm = _safe_int(current.get("llm_processed"))
    learned = _safe_int(current.get("learned_rules"))
    composite = _safe_int(current.get("wnn_multi_discriminator_candidates"))
    reclassified = _safe_int(comparison.get("changed_docs")) if comparison.get("available") else 0
    stable = _safe_int(comparison.get("stable_docs")) if comparison.get("available") else 0
    previous_label = previous.get("execucao", "sem baseline")

    return [
        _card("Documentos Processados", docs, "execucao atual"),
        _card("WNN", wnn, "classificacoes aceitas pela camada 2"),
        _card("LLM Residual", llm, "casos enviados ao Agente 3"),
        _card("Aprendizado", learned, "regras incorporadas apos LLM residual"),
        _card("Candidatos Compostos", composite, "coacionamentos para revisar na arvore"),
        _card("Cobertura WNN", current.get("taxa_wnn", "sem taxa"), "resolvido por discriminadores"),
        _card("Reclassificados", reclassified, f"estaveis: {stable} | baseline: {previous_label}"),
    ]


def _metrics_figure(metrics: pd.DataFrame) -> go.Figure:
    if metrics.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Roteamento por lote",
            annotations=[
                {
                    "text": "Aguardando o primeiro lote concluir para exibir WNN, LLM residual e candidatos compostos.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14, "color": "#475467"},
                }
            ],
            xaxis={"visible": False},
            yaxis={"visible": False},
            margin={"l": 24, "r": 16, "t": 48, "b": 24},
        )
        return fig

    frame = metrics.copy()
    for column in ["wnn_accepted", "post_wnn_residual", "llm_processed", "wnn_multi_discriminator_candidates"]:
        if column not in frame:
            frame[column] = 0
    plot_df = frame[
        ["iteration", "wnn_accepted", "post_wnn_residual", "llm_processed", "wnn_multi_discriminator_candidates"]
    ].rename(
        columns={
            "wnn_accepted": "WNN aceitou",
            "post_wnn_residual": "Residual pos-WNN",
            "llm_processed": "LLM chamada",
            "wnn_multi_discriminator_candidates": "Candidatos compostos",
        }
    )
    long_df = plot_df.melt(id_vars="iteration", var_name="camada", value_name="noticias")
    fig = px.bar(
        long_df,
        x="iteration",
        y="noticias",
        color="camada",
        barmode="group",
        title="Roteamento operacional por lote",
        color_discrete_map={
            "WNN aceitou": "#059669",
            "Residual pos-WNN": "#f59e0b",
            "LLM chamada": "#dc2626",
            "Candidatos compostos": "#7c3aed",
        },
    )
    fig.update_layout(margin={"l": 24, "r": 16, "t": 48, "b": 24}, legend_title_text="")
    return fig


def _rate_figure(metrics: pd.DataFrame) -> go.Figure:
    if metrics.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Taxas e pressao de revisao",
            annotations=[
                {
                    "text": "Sem taxas ainda. Assim que houver lote, este grafico mostra aceitacao WNN, chamada LLM e coacionamentos compostos.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14, "color": "#475467"},
                }
            ],
            xaxis={"visible": False},
            yaxis={"visible": False},
            margin={"l": 24, "r": 16, "t": 48, "b": 24},
        )
        return fig
    frame = metrics.copy()
    for column in ["docs", "wnn_accepted", "llm_processed", "wnn_multi_discriminator_candidates"]:
        if column not in frame:
            frame[column] = 0
    if "docs" in frame and "llm_processed" in frame:
        frame["llm_rate"] = frame.apply(
            lambda row: (float(row["llm_processed"]) / float(row["docs"])) if float(row["docs"] or 0) else 0.0,
            axis=1,
        )
    else:
        frame["llm_rate"] = 0.0
    frame["wnn_rate"] = frame.apply(
        lambda row: (float(row["wnn_accepted"]) / float(row["docs"])) if float(row["docs"] or 0) else 0.0,
        axis=1,
    )
    frame["candidate_rate"] = frame.apply(
        lambda row: (float(row["wnn_multi_discriminator_candidates"]) / float(row["docs"])) if float(row["docs"] or 0) else 0.0,
        axis=1,
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=frame["iteration"], y=frame["wnn_rate"], mode="lines+markers", name="Aceitacao WNN"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["iteration"],
            y=frame["llm_rate"],
            mode="lines+markers",
            name="LLM residual",
            line={"dash": "dot", "color": "#dc2626"},
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["iteration"],
            y=frame["candidate_rate"],
            mode="lines+markers",
            name="Candidatos compostos",
            line={"dash": "dash", "color": "#7c3aed"},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Taxas e pressao de revisao por lote",
        margin={"l": 24, "r": 16, "t": 48, "b": 24},
    )
    left_ceiling = min(1.0, max(0.05, float(frame["wnn_rate"].max()) + 0.04))
    right_ceiling = min(
        1.0,
        max(0.05, float(frame[["llm_rate", "candidate_rate"]].max().max()) + 0.04),
    )
    fig.update_yaxes(title_text="Aceitacao WNN", tickformat=".0%", range=[0, left_ceiling], secondary_y=False)
    fig.update_yaxes(title_text="LLM / candidatos compostos", tickformat=".0%", range=[0, right_ceiling], secondary_y=True)
    return fig


def _summary_table(current: dict[str, Any], previous: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "base_docs",
        "sample_docs",
        "sample_fraction",
        "reserve_docs",
        "themes_accepted",
        "wnn_discriminators",
        "batches_done",
        "docs_processados",
        "wnn_accepted",
        "llm_processed",
        "learned_rules",
        "wnn_multi_discriminator_candidates",
        "rare_promoted_candidates",
        "taxa_wnn",
    ]
    return [
        {
            "metrica": key,
            "atual": current.get(key, ""),
            "baseline": previous.get(key, ""),
        }
        for key in keys
    ]


def _comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    if not comparison.get("available"):
        return [{"status": comparison.get("reason", "comparacao indisponivel")}]
    sample = comparison.get("changed_sample")
    if isinstance(sample, pd.DataFrame) and not sample.empty:
        visible_columns = [
            column
            for column in sample.columns
            if LEGACY_RULE_TOKEN not in str(column).lower()
            and not str(column).startswith("classification_source")
        ]
        cleaned = sample[visible_columns].copy()
        for column in cleaned.select_dtypes(include="object").columns:
            cleaned[column] = cleaned[column].map(
                lambda value: re.sub(
                    LEGACY_RULE_TOKEN,
                    "camada_legada",
                    str(value),
                    flags=re.IGNORECASE,
                )
            )
        return cleaned.to_dict("records")
    return [{"status": "sem reclassificacoes na amostra comparada"}]


def _parse_list_cell(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = str(value or "").strip()
    if not text or text == "[]":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _load_discriminator_bank() -> list[dict[str, Any]]:
    payload = _read_json(WNN_FEATURE_BANK_PATH)
    discriminators = payload.get("discriminators", []) if isinstance(payload, dict) else []
    return [item for item in discriminators if isinstance(item, dict)]


def _split_terms(value: Any, limit: int = 5) -> list[str]:
    if isinstance(value, list):
        terms = [str(item).strip() for item in value]
    else:
        terms = [part.strip() for part in str(value or "").split("|")]
    return [term for term in terms if term][:limit]


def _clip_label(value: str, max_chars: int = 82) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _theme_parent(label: str) -> str:
    normalized = str(label or "").strip()
    return WNN_THEME_PARENTS.get(normalized, normalized)


def _cluster_lookup() -> dict[int, dict[str, Any]]:
    if not CLUSTER_SUMMARY_PATH.exists():
        return {}
    try:
        frame = pd.read_csv(CLUSTER_SUMMARY_PATH)
    except Exception:
        return {}
    clusters: dict[int, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        cluster_id = _safe_int(row.get("cluster_id"))
        clusters[cluster_id] = {
            "size": _safe_int(row.get("size")),
            "top_terms": _split_terms(row.get("top_terms"), limit=5),
            "domain_terms": _split_terms(row.get("domain_terms"), limit=4),
        }
    return clusters


def _discriminators_by_label(limit_per_label: int = 5) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in _load_discriminator_bank():
        label = _theme_parent(str(item.get("label", "")).strip())
        if not label:
            continue
        grouped.setdefault(label, [])
        if len(grouped[label]) < limit_per_label:
            grouped[label].append(_discriminator_label(item))
    return grouped


def _discriminator_label(item: dict[str, Any]) -> str:
    name = str(item.get("name", "") or "").strip()
    if name:
        return name
    tokens = item.get("tokens")
    if isinstance(tokens, list) and tokens:
        return " + ".join(str(token) for token in tokens[:5])
    pattern = str(item.get("pattern", ""))
    extracted = re.findall(r"\\b([a-z0-9]{3,})", pattern)
    if extracted:
        return " + ".join(extracted[:5])
    return pattern[:80] or str(item.get("id", ""))


def _latest_active_from_events() -> tuple[set[str], str]:
    if not EVENTS_JSONL.exists():
        return set(), ""
    try:
        lines = EVENTS_JSONL.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set(), ""
    for line in reversed(lines[-1200:]):
        if '"stage": "wnn_classification"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        active = event.get("active_discriminators", [])
        if not isinstance(active, list):
            active = []
        ids = {
            str(item.get("id", ""))
            for item in active
            if isinstance(item, dict) and item.get("id")
        }
        label = str(event.get("arquivo", ""))
        return ids, label
    return set(), ""


def _canonical_discriminator_activity() -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], str]:
    bank = _load_discriminator_bank()
    by_id = {str(item.get("id", "")): item for item in bank if item.get("id")}
    counts = {disc_id: 0 for disc_id in by_id}
    latest_active: set[str] = set()
    latest_doc = ""
    classifications = _read_lot_classifications(ANALYSIS_DIR)
    if not classifications.empty and "wnn_active_discriminators" in classifications.columns:
        for _, row in classifications.iterrows():
            active = _parse_list_cell(row.get("wnn_active_discriminators", ""))
            ids = [str(item.get("id", "")) for item in active if item.get("id")]
            for disc_id in ids:
                counts[disc_id] = counts.get(disc_id, 0) + 1
            if ids:
                latest_active = set(ids)
                latest_doc = str(row.get("arquivo", ""))

    event_active, event_doc = _latest_active_from_events()
    if event_active:
        latest_active = event_active
        latest_doc = event_doc

    marker_rows = []
    for disc_id, item in by_id.items():
        marker_rows.append(
            {
                "id": disc_id,
                "label": str(item.get("label", "")),
                "descricao": _discriminator_label(item),
                "source": str(item.get("source", "")),
                "count": int(counts.get(disc_id, 0)),
                "active": disc_id in latest_active,
                "learned": str(item.get("source", "")).startswith("agent3_learned"),
            }
        )
    marker_rows.sort(key=lambda item: (-int(item["active"]), -int(item["count"]), item["label"], item["descricao"]))

    grouped: dict[str, dict[str, Any]] = {}
    for row in marker_rows:
        label = _theme_parent(str(row.get("label", "")).strip())
        if not label:
            continue
        bucket = grouped.setdefault(
            label,
            {
                "label": label,
                "markers_total": 0,
                "active_markers": 0,
                "count": 0,
                "learned_markers": 0,
                "active": False,
                "top_markers": [],
            },
        )
        bucket["markers_total"] += 1
        bucket["count"] += int(row.get("count", 0))
        if row.get("learned"):
            bucket["learned_markers"] += 1
        if row.get("active"):
            bucket["active"] = True
            bucket["active_markers"] += 1
        top_markers = bucket.setdefault("top_markers", [])
        if len(top_markers) < 6 and (row.get("active") or int(row.get("count", 0)) > 0):
            top_markers.append(str(row.get("descricao", "")))

    theme_rows = list(grouped.values())
    max_active = max([int(row.get("active_markers", 0)) for row in theme_rows] or [1])
    max_count = max([int(row.get("count", 0)) for row in theme_rows] or [1])
    for row in theme_rows:
        active_markers = int(row.get("active_markers", 0))
        count = int(row.get("count", 0))
        row["intensity"] = active_markers / max(1, max_active)
        row["coverage_intensity"] = count / max(1, max_count)
    theme_rows.sort(key=lambda item: (-int(item["active"]), -int(item["active_markers"]), -int(item["count"]), item["label"]))
    latest_active_labels = {str(row["label"]) for row in theme_rows if row.get("active")}
    return theme_rows, marker_rows, latest_active_labels, latest_doc


def _discriminator_lights(rows: list[dict[str, Any]], limit: int = 90) -> list[html.Div]:
    if not rows:
        return [html.Div("Sem discriminadores carregados ainda.", className="muted")]
    output = []
    for row in rows[:limit]:
        active = bool(row.get("active"))
        active_markers = int(row.get("active_markers", 0))
        markers_total = int(row.get("markers_total", 0))
        learned_markers = int(row.get("learned_markers", 0))
        count = int(row.get("count", 0))
        intensity = float(row.get("intensity", 0.0) or 0.0)
        coverage = float(row.get("coverage_intensity", 0.0) or 0.0)
        alpha = 0.12 + min(0.62, intensity * 0.62)
        if active:
            background = f"linear-gradient(90deg, rgba(34, 197, 94, {alpha:.2f}) 0%, rgba(34, 197, 94, {alpha:.2f}) {max(8, int(intensity * 100))}%, #ffffff {max(8, int(intensity * 100))}%)"
            border = "1px solid #86efac"
        else:
            background = f"linear-gradient(90deg, rgba(148, 163, 184, .16) 0%, rgba(148, 163, 184, .16) {max(3, int(coverage * 100))}%, #f8fafc {max(3, int(coverage * 100))}%)"
            border = "1px solid #e4e9f1"
        top_markers = row.get("top_markers", [])
        marker_hint = " | ".join(str(item) for item in top_markers[:4]) if isinstance(top_markers, list) else ""
        output.append(
            html.Div(
                [
                    html.Span(className=f"disc-dot {'on' if active else 'off'}"),
                    html.Div(
                        [
                            html.Div(f"discriminador {row.get('label', '')}", className="disc-desc"),
                            html.Div(
                                f"acesos={active_markers}/{markers_total} | acionamentos={count} | aprendidos={learned_markers}",
                                className="disc-meta",
                            ),
                            html.Div(marker_hint, className="disc-markers") if marker_hint else None,
                        ],
                        className="disc-text",
                    ),
                ],
                className=f"disc-light {'active' if active else ''}",
                style={"background": background, "border": border},
            )
        )
    return output


def _theme_cloud() -> list[html.Span]:
    themes_path = ANALYSIS_DIR / "incremental" / "temas_canonicos_agent1.json"
    payload = _read_json(themes_path)
    theme_names = [
        str(theme.get("canonical_theme", ""))
        for theme in payload.get("themes", [])
        if isinstance(theme, dict) and theme.get("decision") == "accept" and theme.get("canonical_theme")
    ]
    counts = {theme: 0 for theme in theme_names}
    classifications = _read_lot_classifications(ANALYSIS_DIR)
    if not classifications.empty:
        labels = classifications.apply(_classification_label, axis=1)
        for label, count in labels.value_counts().items():
            normalized = str(label or "").strip()
            if normalized.startswith("crime_"):
                normalized = normalized.removeprefix("crime_")
            if normalized in counts:
                counts[normalized] = int(count)
            elif normalized:
                counts[normalized] = int(count)

    if not counts:
        return [html.Span("Sem temas canonicos carregados ainda.", className="muted")]

    positive_counts = [value for value in counts.values() if value > 0]
    min_count = min(positive_counts) if positive_counts else 0
    max_count = max(positive_counts) if positive_counts else 1

    items: list[html.Span] = []
    for theme, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if max_count == min_count:
            size = 18
        else:
            size = 13 + int(((count - min_count) / max(1, max_count - min_count)) * 23)
        if count == 0:
            size = 12
        items.append(
            html.Span(
                theme,
                title=f"{theme}: {count} classificacoes",
                style={"fontSize": f"{size}px", "opacity": 1.0 if count else 0.45},
                className="theme-token",
            )
        )
    return items


def _classification_counts_by_theme() -> dict[str, int]:
    classifications = _read_lot_classifications(ANALYSIS_DIR)
    counts: dict[str, int] = {}
    if classifications.empty:
        return counts
    for _, row in classifications.iterrows():
        label = _theme_parent(_classification_label(row))
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _canonical_theme_rows() -> list[dict[str, Any]]:
    payload = _read_json(THEMES_PATH)
    raw_themes = [
        theme
        for theme in payload.get("themes", [])
        if isinstance(theme, dict) and theme.get("decision") == "accept" and theme.get("canonical_theme")
    ]
    themes_by_parent: dict[str, dict[str, Any]] = {}
    for theme in raw_themes:
        original_label = str(theme.get("canonical_theme", ""))
        label = _theme_parent(original_label)
        merged = themes_by_parent.setdefault(
            label,
            {
                "canonical_theme": label,
                "decision": "accept",
                "description": str(theme.get("description", label)),
                "included_cluster_ids": [],
                "included_subthemes": [],
                "evidence_terms": [],
                "child_themes": [],
            },
        )
        if original_label != label and original_label not in merged["child_themes"]:
            merged["child_themes"].append(original_label)
        for cluster_id in theme.get("included_cluster_ids", []) or []:
            if cluster_id not in merged["included_cluster_ids"]:
                merged["included_cluster_ids"].append(cluster_id)
        for subtheme in theme.get("included_subthemes", []) or []:
            if subtheme not in merged["included_subthemes"]:
                merged["included_subthemes"].append(subtheme)
        for term in theme.get("evidence_terms", []) or []:
            if term not in merged["evidence_terms"]:
                merged["evidence_terms"].append(term)
    return list(themes_by_parent.values())


def _theme_tree_figure() -> go.Figure:
    themes = _canonical_theme_rows()
    if not themes:
        fig = go.Figure()
        fig.update_layout(
            title="Arvore ainda nao criada",
            annotations=[
                {
                    "text": "Os temas canonicos ainda nao foram gerados pelo Agente 1.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
            height=420,
            margin={"l": 20, "r": 20, "t": 42, "b": 20},
        )
        return fig

    theme_rows, _marker_rows, _latest_active, _latest_doc = _canonical_discriminator_activity()
    theme_meta = {str(theme.get("canonical_theme", "")): theme for theme in themes}
    class_counts = _classification_counts_by_theme()
    rows: list[dict[str, Any]] = []
    for row in theme_rows:
        label = str(row.get("label", ""))
        meta = theme_meta.get(label, {})
        child_themes = meta.get("child_themes", []) if isinstance(meta, dict) else []
        rows.append(
            {
                "label": label,
                "discriminators": int(row.get("markers_total", 0)),
                "learned": int(row.get("learned_markers", 0)),
                "activations": int(row.get("count", 0)),
                "classified": int(class_counts.get(label, 0)),
                "active_markers": int(row.get("active_markers", 0)),
                "child_themes": child_themes,
            }
        )
    for label in sorted(set(theme_meta) - {row["label"] for row in rows}):
        rows.append(
            {
                "label": label,
                "discriminators": 0,
                "learned": 0,
                "activations": 0,
                "classified": int(class_counts.get(label, 0)),
                "active_markers": 0,
                "child_themes": theme_meta.get(label, {}).get("child_themes", []),
            }
        )
    rows.sort(key=lambda item: (-item["classified"], -item["activations"], -item["discriminators"], item["label"]))
    if not rows:
        fig = go.Figure()
        fig.update_layout(title="Mapa operacional de discriminadores ainda vazio")
        return fig

    labels = [row["label"] for row in rows]
    max_size_basis = max([max(row["classified"], row["activations"], 1) for row in rows])
    marker_sizes = [
        18 + int((max(row["classified"], row["activations"], 1) / max_size_basis) * 34)
        for row in rows
    ]
    hover = [
        (
            f"{row['label']}<br>"
            f"discriminadores={row['discriminators']}<br>"
            f"aprendidos={row['learned']}<br>"
            f"classificados nos lotes={row['classified']}<br>"
            f"acionamentos internos={row['activations']}<br>"
            f"marcadores acesos na ultima noticia={row['active_markers']}"
            + (f"<br>subtemas={', '.join(row['child_themes'])}" if row["child_themes"] else "")
        )
        for row in rows
    ]
    text = [
        (
            f"{row['label']}<br>"
            f"{row['discriminators']} discr. | {row['classified']} classif. | {row['activations']} acion."
        )
        for row in rows
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[row["discriminators"] for row in rows],
            y=labels,
            orientation="h",
            name="Discriminadores",
            marker={"color": "#dbeafe"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[row["discriminators"] for row in rows],
            y=labels,
            mode="markers+text",
            name="Tamanho atual",
            marker={
                "size": marker_sizes,
                "color": [row["classified"] for row in rows],
                "colorscale": "Viridis",
                "showscale": False,
                "line": {"color": "#1f2937", "width": 1},
            },
            text=text,
            textposition="middle right",
            hovertext=hover,
            hoverinfo="text",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Mapa operacional dos discriminadores canonicos",
        height=min(1400, max(560, len(rows) * 42 + 120)),
        margin={"l": 210, "r": 36, "t": 56, "b": 42},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis={"title": "quantidade de discriminadores no tema", "rangemode": "tozero"},
        yaxis={"title": "", "autorange": "reversed"},
        showlegend=False,
    )
    return fig


def _theme_tree_3d_figure() -> go.Figure:
    themes = _canonical_theme_rows()
    if not themes:
        fig = go.Figure()
        fig.update_layout(
            title="Arvore 3D ainda nao criada",
            height=620,
            scene={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False}},
        )
        return fig

    theme_rows, _marker_rows, latest_active, _latest_doc = _canonical_discriminator_activity()
    theme_meta = {str(theme.get("canonical_theme", "")): theme for theme in themes}
    class_counts = _classification_counts_by_theme()
    active_labels = {str(item.get("label", "")) for item in latest_active if isinstance(item, dict)}
    rows: list[dict[str, Any]] = []
    for row in theme_rows:
        label = str(row.get("label", ""))
        rows.append(
            {
                "label": label,
                "discriminators": int(row.get("markers_total", 0)),
                "learned": int(row.get("learned_markers", 0)),
                "activations": int(row.get("count", 0)),
                "classified": int(class_counts.get(label, 0)),
                "active_markers": int(row.get("active_markers", 0)),
                "active_now": label in active_labels,
                "child_themes": theme_meta.get(label, {}).get("child_themes", []),
            }
        )
    for label in sorted(set(theme_meta) - {row["label"] for row in rows}):
        rows.append(
            {
                "label": label,
                "discriminators": 0,
                "learned": 0,
                "activations": 0,
                "classified": int(class_counts.get(label, 0)),
                "active_markers": 0,
                "active_now": label in active_labels,
                "child_themes": theme_meta.get(label, {}).get("child_themes", []),
            }
        )
    rows.sort(key=lambda item: (-item["classified"], -item["activations"], -item["discriminators"], item["label"]))
    if not rows:
        fig = go.Figure()
        fig.update_layout(title="Arvore 3D de discriminadores ainda vazia", height=620)
        return fig

    max_disc = max(max(row["discriminators"], 1) for row in rows)
    max_volume = max(max(row["classified"], row["activations"], 1) for row in rows)
    radius = max(4.0, min(12.0, len(rows) * 0.75))
    root = {"x": 0.0, "y": 0.0, "z": 0.0}
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    sizes: list[float] = []
    colors: list[int] = []
    texts: list[str] = []
    hovers: list[str] = []
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []

    for index, row in enumerate(rows):
        angle = (2 * math.pi * index) / max(1, len(rows))
        radial = radius * (0.72 + 0.28 * (row["discriminators"] / max_disc))
        x = math.cos(angle) * radial
        y = math.sin(angle) * radial
        z = 0.8 + 5.2 * (max(row["classified"], row["activations"], 0) / max_volume)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        sizes.append(10 + 38 * (row["discriminators"] / max_disc))
        colors.append(row["learned"])
        texts.append(row["label"])
        hovers.append(
            f"{row['label']}<br>"
            f"discriminadores={row['discriminators']}<br>"
            f"aprendidos={row['learned']}<br>"
            f"classificados={row['classified']}<br>"
            f"acionamentos={row['activations']}<br>"
            f"ativos agora={'sim' if row['active_now'] else 'nao'}"
            + (f"<br>subtemas={', '.join(row['child_themes'])}" if row["child_themes"] else "")
        )
        edge_x.extend([root["x"], x, None])
        edge_y.extend([root["y"], y, None])
        edge_z.extend([root["z"], z, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={"color": "rgba(100,116,139,0.35)", "width": 3},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[root["x"]],
            y=[root["y"]],
            z=[root["z"]],
            mode="markers+text",
            marker={"size": 18, "color": "#0f766e", "line": {"color": "#064e3b", "width": 2}},
            text=["WNN"],
            textposition="bottom center",
            hovertext=["Raiz operacional da memoria WNN"],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            marker={
                "size": sizes,
                "color": colors,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "aprendidos", "len": 0.58},
                "opacity": 0.88,
                "line": {"color": "#1f2937", "width": 1},
            },
            text=texts,
            textposition="top center",
            hovertext=hovers,
            hoverinfo="text",
            showlegend=False,
        )
    )
    active_points = [
        (x, y, z, size, hover)
        for x, y, z, size, hover, row in zip(xs, ys, zs, sizes, hovers, rows, strict=False)
        if row["active_now"]
    ]
    if active_points:
        fig.add_trace(
            go.Scatter3d(
                x=[item[0] for item in active_points],
                y=[item[1] for item in active_points],
                z=[item[2] for item in active_points],
                mode="markers",
                marker={
                    "size": [item[3] + 8 for item in active_points],
                    "color": "rgba(34,197,94,0.18)",
                    "line": {"color": "#16a34a", "width": 5},
                },
                hovertext=[item[4] for item in active_points],
                hoverinfo="text",
                name="Ativo agora",
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Arvore 3D operacional dos discriminadores canonicos",
        height=680,
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
        paper_bgcolor="#ffffff",
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"title": "volume classificado", "showbackground": False, "gridcolor": "#e5e7eb"},
            "camera": {"eye": {"x": 1.45, "y": 1.55, "z": 1.05}},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.35, "y": 1.35, "z": 0.72},
        },
    )
    return fig


def _operational_map_legend() -> html.Div:
    return html.Div(
        [
            html.H3("Legenda"),
            html.Div(
                [
                    html.Span(className="legend-swatch legend-bar"),
                    html.Span("Galho: ligacao entre a memoria WNN e cada tema canonico."),
                ],
                className="legend-row",
            ),
            html.Div(
                [
                    html.Span(className="legend-bubble"),
                    html.Span("No: tamanho indica quantidade de discriminadores no tema."),
                ],
                className="legend-row",
            ),
            html.Div(
                [
                    html.Span(className="legend-gradient"),
                    html.Span("Cor: quantidade de marcadores aprendidos pela LLM residual."),
                ],
                className="legend-row legend-row-tall",
            ),
            html.Div(
                "Altura no eixo Z representa volume classificado/acionado. Borda verde indica discriminador aceso na ultima noticia WNN.",
                className="legend-note",
            ),
        ],
        className="operational-legend",
    )


def create_app() -> Dash:
    app = Dash(__name__)
    app.title = "NT_PF Dashboard"
    app.layout = html.Div(
        [
            dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),
            html.Div(
                [
                    html.H1("NT_PF - Acompanhamento WNN"),
                    html.Div(id="updated-at", className="muted"),
                ],
                className="header",
            ),
            html.Div(id="kpis", className="kpi-grid"),
            html.Div(
                [
                    dcc.Graph(id="metrics-graph", className="panel"),
                    html.Div(
                        [
                            dcc.Graph(id="rate-graph"),
                            html.Div(
                                [
                                    html.Strong("Como ler este grafico: "),
                                    html.Span(
                                        "Aceitacao WNN mostra a fracao de documentos resolvida pelos discriminadores canonicos. "
                                        "LLM residual mostra o que ainda precisou ir ao Agente 3. "
                                        "Candidatos compostos indicam coacionamentos de multiplos discriminadores sem fusao automatica; "
                                        "eles viram insumo para ajustar a arvore e os marcadores."
                                    ),
                                ],
                                className="chart-note",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="graph-grid",
            ),
            html.Div(
                [
                    html.H2("Discriminadores Canonicos WNN"),
                    html.Div(id="discriminator-status", className="muted cloud-help"),
                    html.Div(id="discriminator-lights", className="disc-grid"),
                    html.H2("Contagem de Acionamentos"),
                    dash_table.DataTable(
                        id="discriminator-table",
                        page_size=12,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"},
                        style_cell={"fontFamily": "Arial", "fontSize": 12, "padding": "7px", "textAlign": "left"},
                        style_header={"backgroundColor": "#e8edf4", "fontWeight": "bold"},
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.H2("Arvore Operacional de Temas e Folhas"),
                    html.Div(
                        "Visualizacao 3D da memoria WNN: raiz, temas canonicos e crescimento dos discriminadores conforme os lotes processados.",
                        className="muted cloud-help",
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="theme-tree-graph", className="tree-graph"),
                            _operational_map_legend(),
                        ],
                        className="tree-map-layout",
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Execucao Atual vs Baseline"),
                            dash_table.DataTable(
                                id="summary-table",
                                page_size=20,
                                style_table={"overflowX": "auto"},
                                style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "7px", "textAlign": "left"},
                                style_header={"backgroundColor": "#e8edf4", "fontWeight": "bold"},
                            ),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H2("Reclassificacoes"),
                            dash_table.DataTable(
                                id="changes-table",
                                page_size=12,
                                style_table={"overflowX": "auto", "maxHeight": "520px", "overflowY": "auto"},
                                style_cell={"fontFamily": "Arial", "fontSize": 12, "padding": "7px", "textAlign": "left"},
                                style_header={"backgroundColor": "#e8edf4", "fontWeight": "bold"},
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="table-grid",
            ),
        ],
        className="page",
    )

    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; background: #f5f7fb; color: #162033; font-family: Arial, sans-serif; }
            .page { padding: 22px; }
            .header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 16px; }
            h1 { font-size: 24px; margin: 0; }
            h2 { font-size: 16px; margin: 0 0 12px 0; }
            .muted { color: #667085; font-size: 13px; }
            .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 14px; }
            .kpi { background: #ffffff; border: 1px solid #d9e0ea; border-radius: 8px; padding: 14px; }
            .kpi-title { color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
            .kpi-value { font-size: 28px; font-weight: 700; margin-top: 6px; color: #111827; }
            .kpi-subtitle { color: #667085; font-size: 12px; margin-top: 4px; }
            .graph-grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, .8fr); gap: 14px; margin-bottom: 14px; }
            .table-grid { display: grid; grid-template-columns: minmax(0, .85fr) minmax(0, 1.15fr); gap: 14px; }
            .panel { background: #ffffff; border: 1px solid #d9e0ea; border-radius: 8px; padding: 12px; }
            .chart-note { border-top: 1px solid #e4e9f1; color: #475467; font-size: 13px; line-height: 1.45; padding: 10px 4px 2px 4px; }
            .hidden-note { display: none; }
            .cloud-help { margin-bottom: 10px; }
            .tree-map-layout { display: grid; grid-template-columns: minmax(0, 1fr) 260px; gap: 12px; align-items: start; }
            .tree-graph { min-height: 620px; border: 1px solid #e4e9f1; border-radius: 6px; background: #ffffff; }
            .operational-legend { border: 1px solid #e4e9f1; border-radius: 6px; padding: 12px; background: #f8fafc; position: sticky; top: 12px; }
            .operational-legend h3 { font-size: 14px; margin: 0 0 10px 0; color: #111827; }
            .legend-row { display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 9px; align-items: center; color: #475467; font-size: 12px; line-height: 1.35; margin-bottom: 10px; }
            .legend-row-tall { align-items: start; }
            .legend-swatch { display: inline-block; width: 26px; height: 14px; border-radius: 3px; border: 1px solid #bfdbfe; }
            .legend-bar { background: #dbeafe; }
            .legend-bubble { width: 20px; height: 20px; border-radius: 50%; background: #2fb47c; border: 1px solid #1f2937; box-shadow: 0 0 0 4px rgba(47, 180, 124, .12); }
            .legend-gradient { width: 26px; height: 54px; border-radius: 4px; background: linear-gradient(to top, #440154, #31688e, #35b779, #fde725); border: 1px solid #d0d5dd; }
            .legend-note { border-top: 1px solid #e4e9f1; padding-top: 10px; color: #667085; font-size: 12px; line-height: 1.4; }
            .disc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 8px; margin-bottom: 14px; }
            .disc-light { display: flex; align-items: flex-start; gap: 9px; min-height: 64px; border: 1px solid #e4e9f1; border-radius: 8px; padding: 9px; background: #f8fafc; opacity: .72; }
            .disc-light.active { opacity: 1; box-shadow: 0 0 0 1px rgba(22, 163, 74, .12); }
            .disc-dot { width: 14px; height: 14px; border-radius: 50%; flex: 0 0 14px; display: inline-block; }
            .disc-dot.off { background: #cbd5e1; box-shadow: inset 0 0 0 1px #94a3b8; }
            .disc-dot.on { background: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, .95), inset 0 0 0 1px #15803d; }
            .disc-desc { font-weight: 700; font-size: 13px; color: #111827; }
            .disc-meta { font-size: 11px; color: #667085; margin-top: 2px; }
            .disc-markers { color: #475467; font-size: 11px; line-height: 1.3; margin-top: 4px; }
            @media (max-width: 980px) {
                .graph-grid, .table-grid, .tree-map-layout { grid-template-columns: 1fr; }
                .header { display: block; }
                .operational-legend { position: static; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

    @app.callback(
        Output("updated-at", "children"),
        Output("kpis", "children"),
        Output("metrics-graph", "figure"),
        Output("rate-graph", "figure"),
        Output("summary-table", "data"),
        Output("summary-table", "columns"),
        Output("changes-table", "data"),
        Output("changes-table", "columns"),
        Output("discriminator-status", "children"),
        Output("discriminator-lights", "children"),
        Output("discriminator-table", "data"),
        Output("discriminator-table", "columns"),
        Output("theme-tree-graph", "figure"),
        Input("refresh", "n_intervals"),
    )
    def refresh(_n: int):
        previous = _latest_snapshot()
        current_summary = _summarize_run("atual", ANALYSIS_DIR)
        previous_summary = _summarize_run(previous.name, previous) if previous else {}
        comparison = _compare_classifications(previous, ANALYSIS_DIR)
        metrics = _read_metrics(ANALYSIS_DIR)
        learned_rules = int(metrics.get("learned_rules", pd.Series(dtype=int)).sum()) if not metrics.empty else 0
        current_summary["learned_rules"] = learned_rules
        current_summary["wnn_multi_discriminator_candidates"] = (
            int(metrics.get("wnn_multi_discriminator_candidates", pd.Series(dtype=int)).sum())
            if not metrics.empty
            else 0
        )
        current_summary["rare_promoted_candidates"] = (
            int(metrics.get("rare_promoted_candidates", pd.Series(dtype=int)).sum())
            if not metrics.empty
            else 0
        )

        summary_rows = _summary_table(current_summary, previous_summary)
        summary_columns = [{"name": key, "id": key} for key in ("metrica", "atual", "baseline")]
        change_rows = _comparison_rows(comparison)
        change_columns = [{"name": key, "id": key} for key in (change_rows[0].keys() if change_rows else ["status"])]
        discriminator_rows, marker_rows, latest_active, latest_doc = _canonical_discriminator_activity()
        discriminator_table_rows = [
            {
                "ativo_ultima": "sim" if row.get("active") else "nao",
                "discriminador_canonico": row.get("label", ""),
                "marcadores_acesos": row.get("active_markers", 0),
                "marcadores_total": row.get("markers_total", 0),
                "acionamentos": row.get("count", 0),
                "marcadores_aprendidos": row.get("learned_markers", 0),
                "principais_marcadores": " | ".join(str(item) for item in row.get("top_markers", [])[:5])
                if isinstance(row.get("top_markers", []), list)
                else "",
            }
            for row in discriminator_rows
        ]
        discriminator_columns = [
            {"name": key, "id": key}
            for key in [
                "ativo_ultima",
                "discriminador_canonico",
                "marcadores_acesos",
                "marcadores_total",
                "acionamentos",
                "marcadores_aprendidos",
                "principais_marcadores",
            ]
        ]
        discriminator_status = (
            f"{len(latest_active)} discriminadores canonicos acesos na ultima noticia WNN registrada "
            f"({sum(int(row.get('active_markers', 0)) for row in discriminator_rows)} marcadores internos). "
            f"Banco atual: {len(discriminator_rows)} temas canonicos e {len(marker_rows)} marcadores"
            + (f": {latest_doc}" if latest_doc else ".")
        )

        return (
            f"Atualiza a cada {REFRESH_MS // 1000}s | baseline: {previous.name if previous else 'nenhum'}",
            _summary_cards(current_summary, previous_summary, comparison),
            _metrics_figure(metrics),
            _rate_figure(metrics),
            summary_rows,
            summary_columns,
            change_rows,
            change_columns,
            discriminator_status,
            _discriminator_lights(discriminator_rows),
            discriminator_table_rows,
            discriminator_columns,
            _theme_tree_3d_figure(),
        )

    return app


def main() -> None:
    host = os.getenv("PF_DASH_HOST", "127.0.0.1")
    port = int(os.getenv("PF_DASH_PORT", "8050"))
    app = create_app()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
