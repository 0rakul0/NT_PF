from __future__ import annotations

import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from plotly.subplots import make_subplots

from scripts.incremental.common import ANALYSIS_DIR, EVENTS_JSONL, LINGUISTIC_PREPROCESSING_JSON, LOTS_DIR, WNN_FEATURE_BANK_PATH
from scripts.incremental.dashboard_comparacao import (
    _classification_label,
    _compare_classifications,
    _latest_snapshot,
    _read_json,
    _read_lot_classifications,
    _read_metrics,
    _summarize_run,
)
from scripts.incremental.wnn_crime_modus_tree import build_hierarchy_df


REFRESH_MS = int(os.getenv("PF_DASH_REFRESH_MS", "120000"))
AUTO_REFRESH = os.getenv("PF_DASH_AUTO_REFRESH", "false").strip().lower() in {"1", "true", "yes", "on"}
THEMES_PATH = ANALYSIS_DIR / "incremental" / "temas_canonicos_agent1.json"
CLUSTER_SUMMARY_PATH = ANALYSIS_DIR / "incremental" / "resumo_clusters_amostra.csv"
LEGACY_RULE_TOKEN = "re" + "gex"
WNN_THEME_PARENTS: dict[str, str] = {}
MAX_MEMORY_CELLS = 320
MAX_GRAPH_CRIMES = 10
MAX_GRAPH_MODUS_PER_GROUP = 6
MAX_SANKEY_CRIMES = 6
MAX_SANKEY_MODUS_PER_CRIME = 4
MAX_PARTICLE_POINTS = 180


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


def _card(title: str, value: Any, subtitle: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(str(value), className="kpi-value"),
            html.Div(subtitle, className="kpi-subtitle"),
        ],
        className="kpi",
    )


def _axis_overview_panel(current: dict[str, Any]) -> list[html.Div]:
    crime_docs = _safe_int(current.get("crime_docs"))
    modus_docs = _safe_int(current.get("modus_docs"))
    unique_crimes = _safe_int(current.get("unique_crime_labels"))
    unique_modus = _safe_int(current.get("unique_modus_labels"))
    learned_crime = _safe_int(current.get("learned_crime_discriminators"))
    learned_modus = _safe_int(current.get("learned_modus_discriminators"))
    return [
        html.Div(
            [
                html.Div("Eixo Crime", className="axis-title"),
                html.Div(
                    [
                        _card("Docs Com Crime", crime_docs, f"labels unicas: {unique_crimes}"),
                        _card("Discriminadores Aprendidos", learned_crime, "camada WNN de crime"),
                    ],
                    className="axis-card-grid",
                ),
            ],
            className="axis-panel axis-crime",
        ),
        html.Div(
            [
                html.Div("Eixo Modus Operandi", className="axis-title"),
                html.Div(
                    [
                        _card("Docs Com Modus", modus_docs, f"labels unicas: {unique_modus}"),
                        _card("Discriminadores Aprendidos", learned_modus, "camada WNN de modus"),
                    ],
                    className="axis-card-grid",
                ),
            ],
            className="axis-panel axis-modus",
        ),
    ]


def _summary_cards(current: dict[str, Any], previous: dict[str, Any], comparison: dict[str, Any]) -> list[html.Div]:
    docs = _safe_int(current.get("docs_processados"))
    wnn = _safe_int(current.get("wnn_accepted"))
    llm = _safe_int(current.get("llm_processed"))
    learned = _safe_int(current.get("learned_rules"))
    composite = _safe_int(current.get("wnn_multi_discriminator_candidates"))
    memory_size = _safe_int(current.get("wnn_memory_vocab_size"))
    crime_docs = _safe_int(current.get("crime_docs"))
    modus_docs = _safe_int(current.get("modus_docs"))
    unique_crimes = _safe_int(current.get("unique_crime_labels"))
    unique_modus = _safe_int(current.get("unique_modus_labels"))
    learned_crime = _safe_int(current.get("learned_crime_discriminators"))
    learned_modus = _safe_int(current.get("learned_modus_discriminators"))
    reclassified = _safe_int(comparison.get("changed_docs")) if comparison.get("available") else 0
    stable = _safe_int(comparison.get("stable_docs")) if comparison.get("available") else 0
    previous_label = previous.get("execucao", "sem baseline")
    semantic_reduction = current.get("semantic_reduction_ratio", "")
    try:
        semantic_value = f"{float(semantic_reduction):.2%}"
    except (TypeError, ValueError):
        semantic_value = "aguardando"

    return [
        _card("Documentos Processados", docs, "execucao atual"),
        _card("WNN", wnn, "classificacoes aceitas pela camada 2"),
        _card("Docs Com Crime", crime_docs, f"labels unicas: {unique_crimes}"),
        _card("Docs Com Modus", modus_docs, f"labels unicas: {unique_modus}"),
        _card("LLM Residual", llm, "casos enviados ao Agente 3"),
        _card("Aprendizado", learned, f"crime={learned_crime} | modus={learned_modus}"),
        _card("Candidatos Compostos", composite, "coacionamentos para revisar na arvore"),
        _card("Cobertura WNN", current.get("taxa_wnn", "sem taxa"), "resolvido por discriminadores"),
        _card("Memoria WNN", memory_size or "aguardando", "posicoes na matriz binaria"),
        _card("Reducao Semantica", semantic_value, "tokens mantidos: substantivos, verbos e adjetivos"),
        _card("Reclassificados", reclassified, f"estaveis: {stable} | baseline: {previous_label}"),
    ]


def _metrics_figure(metrics: pd.DataFrame, axis_metrics: pd.DataFrame | None = None) -> go.Figure:
    axis_metrics = axis_metrics if axis_metrics is not None else pd.DataFrame()
    if metrics.empty and axis_metrics.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Roteamento e cobertura por lote",
            annotations=[
                {
                    "text": "Aguardando o primeiro lote concluir para exibir roteamento, crime e modus operandi.",
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
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = {
        "WNN aceitou": "#059669",
        "Residual pos-WNN": "#f59e0b",
        "LLM chamada": "#dc2626",
        "Candidatos compostos": "#7c3aed",
    }
    for camada in ["WNN aceitou", "Residual pos-WNN", "LLM chamada", "Candidatos compostos"]:
        subset = long_df.loc[long_df["camada"] == camada]
        fig.add_trace(
            go.Bar(x=subset["iteration"], y=subset["noticias"], name=camada, marker_color=colors[camada]),
            secondary_y=False,
        )
    if not axis_metrics.empty:
        fig.add_trace(
            go.Scatter(
                x=axis_metrics["iteration"],
                y=axis_metrics["crime_rate"],
                mode="lines+markers",
                name="Cobertura crime",
                line={"color": "#0f172a", "width": 3},
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=axis_metrics["iteration"],
                y=axis_metrics["modus_rate"],
                mode="lines+markers",
                name="Cobertura modus",
                line={"color": "#2563eb", "width": 3, "dash": "dot"},
            ),
            secondary_y=True,
        )
    fig.update_layout(
        title="Roteamento operacional + cobertura por eixo",
        margin={"l": 24, "r": 16, "t": 48, "b": 24},
        legend_title_text="",
        barmode="group",
    )
    fig.update_yaxes(title_text="Documentos", secondary_y=False)
    fig.update_yaxes(title_text="Cobertura por eixo", tickformat=".0%", range=[0, 1], secondary_y=True)
    return fig


def _rate_figure(metrics: pd.DataFrame, axis_metrics: pd.DataFrame | None = None) -> go.Figure:
    axis_metrics = axis_metrics if axis_metrics is not None else pd.DataFrame()
    if metrics.empty and axis_metrics.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Taxas e diversidade dos classificadores",
            annotations=[
                {
                    "text": "Sem taxas ainda. Assim que houver lote, este grafico mostra crime, modus e pressao de revisao.",
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
    if not axis_metrics.empty:
        fig.add_trace(
            go.Scatter(
                x=axis_metrics["iteration"],
                y=axis_metrics["unique_crimes"],
                mode="lines+markers",
                name="Crimes unicos",
                line={"color": "#111827", "dash": "dash"},
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=axis_metrics["iteration"],
                y=axis_metrics["unique_modus"],
                mode="lines+markers",
                name="Modus unicos",
                line={"color": "#1d4ed8", "dash": "dash"},
            ),
            secondary_y=False,
        )
    fig.update_layout(
        title="Taxas e diversidade por lote",
        margin={"l": 24, "r": 16, "t": 48, "b": 24},
    )
    left_source = [float(frame["wnn_rate"].max()) if not frame.empty else 0.0]
    if not axis_metrics.empty:
        left_source.extend([float(axis_metrics["unique_crimes"].max()), float(axis_metrics["unique_modus"].max())])
    left_ceiling = max(1.0, max(left_source) + 1.0)
    right_ceiling = min(
        1.0,
        max(0.05, float(frame[["llm_rate", "candidate_rate"]].max().max()) + 0.04),
    )
    fig.update_yaxes(title_text="Qtd. de labels ativas / diversidade", range=[0, left_ceiling], secondary_y=False)
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
        "crime_docs",
        "modus_docs",
        "unique_crime_labels",
        "unique_modus_labels",
        "wnn_accepted",
        "llm_processed",
        "learned_rules",
        "learned_crime_discriminators",
        "learned_modus_discriminators",
        "wnn_multi_discriminator_candidates",
        "rare_promoted_candidates",
        "semantic_original_tokens",
        "semantic_kept_tokens",
        "semantic_reduction_ratio",
        "wnn_memory_vocab_size",
        "wnn_memory_active_avg",
        "taxa_wnn",
    ]


def _extract_modus_labels_from_row(row: pd.Series) -> list[str]:
    for column in ("agent3_modus_operandi", "wnn_modus_operandi"):
        values = _parse_jsonish_list(row.get(column, []))
        labels = [str(item).strip() for item in values if str(item).strip()]
        if labels:
            return labels
    raw = row.get("inference", "")
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            return []
        if isinstance(payload, dict):
            values = payload.get("modus_operandi", [])
            if isinstance(values, list):
                return [str(item).strip() for item in values if str(item).strip()]
    return []


def _crime_filter_options() -> list[dict[str, str]]:
    hierarchy = build_hierarchy_df()
    options = [{"label": "Top crimes", "value": "__top__"}]
    if hierarchy.empty:
        return options
    crime_rows = (
        hierarchy[["crime_label", "crime_total"]]
        .drop_duplicates()
        .sort_values(["crime_total", "crime_label"], ascending=[False, True])
    )
    for item in crime_rows.itertuples():
        label = str(item.crime_label)
        total = int(item.crime_total)
        options.append({"label": f"{label} ({total})", "value": label})
    return options


def _graph_mode_options() -> list[dict[str, str]]:
    return [
        {"label": "Exploracao", "value": "exploracao"},
        {"label": "Publicacao", "value": "publicacao"},
    ]


EXECUTION_MODUS = {
    "abordagem_via_publica",
    "arma_fogo",
    "atividade_clandestina",
    "caca_ilegal",
    "comercializacao_clandestina",
    "comercializacao_ilegal",
    "comercializacao_irregular",
    "compartilhamento_online",
    "contrabando",
    "contrabando_produtos",
    "corrupcao_ativa",
    "corrupcao_ativa_passiva",
    "corrupcao_passiva",
    "dano_qualificado",
    "desmatamento",
    "desvio_bens",
    "desvio_bens_publicos",
    "desvio_finalidade",
    "desvio_recursos",
    "desvio_sistematico",
    "desvio_valores",
    "distribuicao_drogas",
    "distribuicao_ilegal",
    "erradicacao_cultivos",
    "erradicacao_cultivo_ilicito",
    "erradicacao_plantacoes",
    "extracao_ilegal_madeira",
    "fraude_digital",
    "fraude_documental",
    "fraude_financeira",
    "fraude_pix",
    "furto_dados",
    "furto_qualificado",
    "furtos_qualificados",
    "garimpo_ilegal",
    "importacao_clandestina",
    "importacao_ilegal",
    "incendio_criminoso",
    "insercao_dados_falsos",
    "invasao_dispositivo",
    "invasao_terra_publica",
    "invasao_terras",
    "lavagem_financeira",
    "licitacao_fraudulenta",
    "maus_tratos_animais",
    "pagamento_propina",
    "pesca_ilegal",
    "porto_clandestino",
    "risco_ambiental",
    "roubo_carga",
    "roubo_organizacao",
    "sonegacao_fiscal",
    "tentativa_homicidio",
    "trafico_de_especies",
    "transporte_ilegal",
    "transporte_irregular",
    "uso_equipamentos_garimpo",
    "uso_ilegal_solo",
    "uso_redes_sociais",
    "venda_ilegal",
}
SUPPORT_MODUS = {
    "arma_digital",
    "armazenamento_bens",
    "armazenamento_clandestino",
    "armazenamento_digital",
    "armazenamento_ilegal",
    "armazenamento_irregular",
    "armazenamento_residencial",
    "bloqueio_ativos",
    "bloqueio_bancario",
    "bloqueio_bens",
    "bloqueio_contas",
    "bloqueio_judicial",
    "cambio_ilegal",
    "comercio_eletronico",
    "comercio_irregular",
    "conluio_empresarial",
    "dispensa_licitacao",
    "dispensa_licitacao_irregular",
    "encomenda_correios",
    "falsidade_documental",
    "falsidade_ideologica",
    "falsificacao_documentos",
    "falsificacao_identidade",
    "fraude_fiscalizacao",
    "fraude_interna",
    "movimentacao_em_especie",
    "notas_fiscais_falsas",
    "ocultacao_de_valores",
    "pagamento_dinheiro",
    "pagamento_em_especie",
    "participacao_funcionario_publico",
    "posse_falsa",
    "posse_irregular_arma",
    "posse_irregular_armas",
    "transporte_dinheiro",
    "transporte_oculto",
    "uso_documento_falso",
    "uso_empresas_fachada",
    "uso_empresa_fachada",
}
RESPONSE_MODUS = {
    "analise_pericial",
    "apreensao_arma_fogo",
    "apreensao_dinheiro",
    "apreensao_dispositivos_eletronicos",
    "apreensao_documentos",
    "apreensao_drogas",
    "apreensao_madeira",
    "apreensao_mercadoria_ilegal",
    "apreensao_mercadorias",
    "apreensao_veiculos",
    "busca_e_apreensao",
    "colaboracao_premiada",
    "cooperacao_internacional",
    "fiscalizacao_ambiental",
    "fiscalizacao_comercial",
    "medidas_cautelares",
    "monitoramento_eletronico",
    "pericia_criminal",
    "sequestro_bens",
}


def _modus_group(label: str) -> str:
    normalized = str(label or "").strip()
    if normalized in EXECUTION_MODUS:
        return "execucao_criminosa"
    if normalized in SUPPORT_MODUS:
        return "fraude_ocultacao_suporte"
    if normalized in RESPONSE_MODUS:
        return "resposta_operacional"
    return "outros_modus"


def _read_lot_classifications_with_batches(base: Path) -> pd.DataFrame:
    lots = base / "lotes"
    if not lots.exists() and base == ANALYSIS_DIR:
        lots = LOTS_DIR
    files = sorted(lots.glob("lote_*_classificacoes.csv"))
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        batch_id = path.stem.replace("_classificacoes", "")
        try:
            iteration = int(re.search(r"lote_(\d+)", batch_id).group(1))
        except Exception:
            iteration = len(frames) + 1
        frame["batch_id"] = batch_id
        frame["iteration"] = iteration
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _classifier_axis_metrics() -> tuple[dict[str, Any], pd.DataFrame]:
    classifications = _read_lot_classifications_with_batches(ANALYSIS_DIR)
    if classifications.empty:
        return {}, pd.DataFrame()

    frame = classifications.copy()
    frame["crime_label"] = frame.apply(_classification_label, axis=1)
    frame["modus_labels"] = frame.apply(_extract_modus_labels_from_row, axis=1)
    frame["has_crime"] = frame["crime_label"].map(lambda value: bool(str(value or "").strip()))
    frame["has_modus"] = frame["modus_labels"].map(lambda values: bool(values))

    summary = {
        "crime_docs": int(frame["has_crime"].sum()),
        "modus_docs": int(frame["has_modus"].sum()),
        "unique_crime_labels": int(frame.loc[frame["has_crime"], "crime_label"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
        "unique_modus_labels": int(
            len(
                {
                    str(label).strip()
                    for labels in frame["modus_labels"].tolist()
                    for label in labels
                    if str(label).strip()
                }
            )
        ),
    }

    per_batch_rows: list[dict[str, Any]] = []
    for iteration, batch_df in frame.groupby("iteration", sort=True):
        batch_docs = len(batch_df)
        batch_modus_labels = {
            str(label).strip()
            for labels in batch_df["modus_labels"].tolist()
            for label in labels
            if str(label).strip()
        }
        per_batch_rows.append(
            {
                "iteration": int(iteration),
                "batch_id": str(batch_df["batch_id"].iloc[0]),
                "docs": batch_docs,
                "crime_docs": int(batch_df["has_crime"].sum()),
                "modus_docs": int(batch_df["has_modus"].sum()),
                "crime_rate": (float(batch_df["has_crime"].sum()) / float(batch_docs)) if batch_docs else 0.0,
                "modus_rate": (float(batch_df["has_modus"].sum()) / float(batch_docs)) if batch_docs else 0.0,
                "unique_crimes": int(batch_df.loc[batch_df["has_crime"], "crime_label"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
                "unique_modus": int(len(batch_modus_labels)),
            }
        )
    return summary, pd.DataFrame(per_batch_rows)
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


def _memory_position_context() -> dict[int, dict[str, Any]]:
    payload = _read_json(WNN_FEATURE_BANK_PATH)
    if not isinstance(payload, dict):
        return {}
    memory = payload.get("memory_vocab", {})
    if not isinstance(memory, dict):
        return {}
    tokens = memory.get("tokens", [])
    if not isinstance(tokens, list):
        tokens = []
    context: dict[int, dict[str, Any]] = {
        index: {"token": str(token), "labels": set(), "discriminators": []}
        for index, token in enumerate(tokens)
    }
    token_to_position = {str(token): index for index, token in enumerate(tokens)}
    discriminators = payload.get("discriminators", [])
    if not isinstance(discriminators, list):
        discriminators = []
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        label = _theme_parent(str(item.get("label", "")))
        name = _discriminator_label(item)
        positions = item.get("memory_positions", [])
        if not isinstance(positions, list):
            positions = []
        if not positions:
            raw_tokens = item.get("tokens", [])
            if isinstance(raw_tokens, list):
                positions = [
                    token_to_position[str(token)]
                    for token in raw_tokens
                    if str(token) in token_to_position
                ]
        for raw_position in positions:
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                continue
            bucket = context.setdefault(position, {"token": "", "labels": set(), "discriminators": []})
            if label:
                bucket["labels"].add(label)
            disc_list = bucket.setdefault("discriminators", [])
            if isinstance(disc_list, list) and name and name not in disc_list:
                disc_list.append(name)
    return context


def _latest_memory_from_events() -> dict[str, Any]:
    if not EVENTS_JSONL.exists():
        return {}
    try:
        lines = EVENTS_JSONL.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines[-1200:]):
        if '"stage": "wnn_classification"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        binary = str(event.get("memory_binary", "") or "")
        if not binary and not event.get("memory_vocab_size"):
            continue
        return {
            "arquivo": str(event.get("arquivo", "")),
            "version": int(event.get("memory_version", 0) or 0),
            "vocab_size": int(event.get("memory_vocab_size", 0) or 0),
            "active_count": int(event.get("memory_active_count", 0) or 0),
            "active_positions": event.get("memory_active_positions", []),
            "binary": binary,
        }
    return {}


def _binary_memory_panel(max_cells: int = MAX_MEMORY_CELLS) -> html.Div:
    memory = _latest_memory_from_events()
    if not memory:
        return html.Div("Aguardando a primeira noticia passar pela memoria binaria.", className="muted")
    binary = str(memory.get("binary", ""))
    active_positions = memory.get("active_positions", [])
    if not isinstance(active_positions, list):
        active_positions = []
    position_preview = ", ".join(str(item) for item in active_positions[:24])
    if len(active_positions) > 24:
        position_preview += ", ..."
    position_context = _memory_position_context()

    def cell_title(index: int, bit: str) -> str:
        info = position_context.get(index, {})
        token = str(info.get("token", "") or "(sem token)")
        labels = sorted(str(item) for item in info.get("labels", set()) if str(item))
        discriminators = info.get("discriminators", [])
        if not isinstance(discriminators, list):
            discriminators = []
        lines = [
            f"posicao {index}: {'acesa' if bit == '1' else 'apagada'}",
            f"token: {token}",
        ]
        if labels:
            lines.append("temas: " + ", ".join(labels[:6]))
        if discriminators:
            lines.append("discriminadores: " + " | ".join(str(item) for item in discriminators[:5]))
        return "\n".join(lines)

    truncated = len(binary) > max_cells
    visible_binary = binary[:max_cells]
    bits = [
        html.Span(
            "",
            className=f"memory-cell {'on' if bit == '1' else 'off'}",
            title=cell_title(index, bit),
        )
        for index, bit in enumerate(visible_binary)
        if bit in {"0", "1"}
    ]
    return html.Div(
        [
            html.Div(
                f"versao={memory.get('version', 0)} | largura={memory.get('vocab_size', 0)} | bits acesos={memory.get('active_count', 0)}",
                className="disc-meta",
            ),
            html.Div(bits, className="memory-matrix"),
            html.Div(
                f"visualizacao resumida: primeiros {len(visible_binary)} de {len(binary)} bits da memoria."
                if truncated
                else "visualizacao completa da memoria binaria da ultima noticia.",
                className="disc-meta",
            ),
            html.Div(
                "Quadrado verde = palavra-chave da matriz encontrada na noticia; quadrado apagado = posicao conhecida, mas nao acionada.",
                className="disc-meta",
            ),
            html.Div(f"posicoes acesas: {position_preview or 'nenhuma'}", className="disc-markers"),
            html.Div(str(memory.get("arquivo", "")), className="muted"),
        ],
        className="memory-panel",
    )


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


def _discriminator_variants(item: dict[str, Any], limit: int = 6) -> list[str]:
    labels = [_discriminator_label(item)]
    raw_variants = item.get("marker_variants", [])
    if isinstance(raw_variants, list):
        for raw_variant in raw_variants:
            if not isinstance(raw_variant, dict):
                continue
            name = str(raw_variant.get("name", "") or "").strip()
            if not name:
                tokens = raw_variant.get("tokens", [])
                if isinstance(tokens, list) and tokens:
                    name = " + ".join(str(token) for token in tokens[:5])
            if name and name not in labels:
                labels.append(name)
            if len(labels) >= limit:
                break
    return labels[:limit]


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
        kind = str(item.get("kind", "crime") or "crime")
        marker_rows.append(
            {
                "id": disc_id,
                "kind": kind,
                "label": str(item.get("label", "")),
                "descricao": _discriminator_label(item),
                "variantes": _discriminator_variants(item),
                "variant_count": len(_discriminator_variants(item, limit=999)),
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
        kind = str(row.get("kind", "crime") or "crime")
        if not label:
            continue
        group_key = f"{kind}:{label}"
        bucket = grouped.setdefault(
            group_key,
            {
                "kind": kind,
                "label": label,
                "markers_total": 0,
                "active_markers": 0,
                "count": 0,
                "learned_markers": 0,
                "variants_total": 0,
                "active": False,
                "top_markers": [],
            },
        )
        bucket["markers_total"] += 1
        bucket["variants_total"] += max(1, int(row.get("variant_count", 1) or 1))
        bucket["count"] += int(row.get("count", 0))
        if row.get("learned"):
            bucket["learned_markers"] += 1
        if row.get("active"):
            bucket["active"] = True
            bucket["active_markers"] += 1
        top_markers = bucket.setdefault("top_markers", [])
        if len(top_markers) < 6 and (row.get("active") or int(row.get("count", 0)) > 0):
            for variant_name in row.get("variantes", []):
                if len(top_markers) >= 6:
                    break
                if variant_name and variant_name not in top_markers:
                    top_markers.append(str(variant_name))

    theme_rows = list(grouped.values())
    max_active = max([int(row.get("active_markers", 0)) for row in theme_rows] or [1])
    max_count = max([int(row.get("count", 0)) for row in theme_rows] or [1])
    for row in theme_rows:
        active_markers = int(row.get("active_markers", 0))
        count = int(row.get("count", 0))
        row["intensity"] = active_markers / max(1, max_active)
        row["coverage_intensity"] = count / max(1, max_count)
    theme_rows.sort(key=lambda item: (-int(item["active"]), -int(item["active_markers"]), -int(item["count"]), item["label"]))
    latest_active_labels = {f"{row.get('kind', 'crime')}:{row['label']}" for row in theme_rows if row.get("active")}
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
        variants_total = int(row.get("variants_total", markers_total) or markers_total)
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
                            html.Div(f"{row.get('kind', 'crime')}: {row.get('label', '')}", className="disc-desc"),
                            html.Div(
                                f"acesos={active_markers}/{markers_total} | variantes={variants_total} | acionamentos={count} | aprendidos={learned_markers}",
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


def _axis_discriminator_sections(rows: list[dict[str, Any]]) -> list[html.Div]:
    crime_rows = [row for row in rows if str(row.get("kind", "crime")) == "crime"]
    modus_rows = [row for row in rows if str(row.get("kind", "crime")) == "modus"]
    execution_rows = [
        row
        for row in modus_rows
        if _modus_group(str(row.get("label", ""))) == "execucao_criminosa"
    ]
    support_rows = [
        row
        for row in modus_rows
        if _modus_group(str(row.get("label", ""))) == "fraude_ocultacao_suporte"
    ]
    response_rows = [
        row
        for row in modus_rows
        if _modus_group(str(row.get("label", ""))) == "resposta_operacional"
    ]
    other_modus_rows = [
        row
        for row in modus_rows
        if _modus_group(str(row.get("label", ""))) == "outros_modus"
    ]
    return [
        html.Div(
            [
                html.H3("Discriminadores de Crime"),
                html.Div(
                    "Marcadores canonicos usados para definir o tipo de crime a partir das noticias.",
                    className="muted cloud-help",
                ),
                html.Div(_discriminator_lights(crime_rows, limit=24), className="disc-grid"),
            ],
            className="axis-discriminator-panel",
        ),
        html.Div(
            [
                html.H3("Discriminadores de Modus Operandi"),
                html.Div(
                    "Marcadores canonicos usados para descrever como a acao criminosa foi executada. O dashboard agrupa os modos em categorias operacionais reutilizaveis para todos os tipos de crime.",
                    className="muted cloud-help",
                ),
                html.H4("Execucao criminosa", className="subsection-title"),
                html.Div(_discriminator_lights(execution_rows, limit=14), className="disc-grid"),
                html.H4("Fraude, ocultacao e suporte", className="subsection-title"),
                html.Div(_discriminator_lights(support_rows, limit=14), className="disc-grid"),
                html.H4("Resposta operacional", className="subsection-title"),
                html.Div(_discriminator_lights(response_rows, limit=14), className="disc-grid"),
                html.H4("Outros modus", className="subsection-title"),
                html.Div(_discriminator_lights(other_modus_rows, limit=18), className="disc-grid"),
            ],
            className="axis-discriminator-panel",
        ),
    ]


def _humanize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().replace("_", " "))


def _hex_to_rgba(color: str, alpha: float) -> str:
    hex_color = str(color or "").strip().lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(148, 163, 184, {alpha:.2f})"
    try:
        red = int(hex_color[0:2], 16)
        green = int(hex_color[2:4], 16)
        blue = int(hex_color[4:6], 16)
    except ValueError:
        return f"rgba(148, 163, 184, {alpha:.2f})"
    return f"rgba({red}, {green}, {blue}, {alpha:.2f})"


def _theme_marker_lookup() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    theme_rows, _marker_rows, _latest_active, _latest_doc = _canonical_discriminator_activity()
    crime_markers: dict[str, list[str]] = {}
    modus_markers: dict[str, list[str]] = {}
    for row in theme_rows:
        label = str(row.get("label", "")).strip()
        markers = row.get("top_markers", [])
        if not label or not isinstance(markers, list):
            continue
        cleaned = [_humanize_label(item) for item in markers if str(item).strip()]
        if str(row.get("kind", "crime")) == "crime":
            crime_markers[label] = cleaned[:4]
        else:
            modus_markers[label] = cleaned[:4]
    return crime_markers, modus_markers


def _build_sankey_focus(selected_crime: str = "__top__") -> tuple[pd.DataFrame, dict[str, Any]]:
    hierarchy = build_hierarchy_df()
    if hierarchy.empty:
        return pd.DataFrame(columns=["crime_label", "modus_label", "count", "crime_total"]), {}

    crime_rows = (
        hierarchy[["crime_label", "crime_total"]]
        .drop_duplicates()
        .sort_values(["crime_total", "crime_label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selected = str(selected_crime or "__top__").strip()
    if selected and selected != "__top__":
        selected_crimes = [selected]
        top_modus_limit = 8
    else:
        selected_crimes = crime_rows.head(MAX_SANKEY_CRIMES)["crime_label"].tolist()
        top_modus_limit = MAX_SANKEY_MODUS_PER_CRIME

    focused_rows: list[dict[str, Any]] = []
    trimmed_modus = 0
    for crime_label in selected_crimes:
        crime_group = hierarchy.loc[hierarchy["crime_label"] == crime_label].copy()
        if crime_group.empty:
            continue
        crime_group = crime_group.sort_values(["count", "modus_label"], ascending=[False, True]).reset_index(drop=True)
        visible = crime_group.head(top_modus_limit).copy()
        remainder = crime_group.iloc[top_modus_limit:].copy()
        trimmed_modus += len(remainder)
        for item in visible.itertuples():
            focused_rows.append(
                {
                    "crime_label": str(item.crime_label),
                    "modus_label": str(item.modus_label),
                    "modus_display": str(item.modus_label),
                    "count": int(item.count),
                    "crime_total": int(item.crime_total),
                }
            )
        if not remainder.empty:
            focused_rows.append(
                {
                    "crime_label": str(crime_label),
                    "modus_label": f"demais_modus::{crime_label}",
                    "modus_display": "demais_modus",
                    "count": int(remainder["count"].sum()),
                    "crime_total": int(crime_group["crime_total"].iloc[0]),
                }
            )
    focused = pd.DataFrame(focused_rows)
    if focused.empty:
        return focused, {}

    focus_summary = {
        "selected_crime": selected if selected != "__top__" else "",
        "crime_count": int(focused["crime_label"].nunique()),
        "modus_count": int(focused["modus_label"].nunique()),
        "flow_total": int(focused["count"].sum()),
        "trimmed_modus": int(trimmed_modus),
        "top_modus_limit": int(top_modus_limit),
    }
    return focused, focus_summary


def _wnn_flow_sankey_figure(selected_crime: str = "__top__") -> tuple[go.Figure, str]:
    focused, summary = _build_sankey_focus(selected_crime)
    if focused.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Fluxo Sankey da Classificacao WNN",
            annotations=[
                {
                    "text": "Aguardando classificacoes finais para montar o fluxo de noticias, crime e modus operandi.",
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
            margin={"l": 24, "r": 24, "t": 56, "b": 24},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
        return fig, "Sem classificacoes finais suficientes para construir o Sankey."

    crime_markers, modus_markers = _theme_marker_lookup()
    palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Safe
    crime_order = (
        focused[["crime_label", "crime_total"]]
        .drop_duplicates()
        .sort_values(["crime_total", "crime_label"], ascending=[False, True])["crime_label"]
        .tolist()
    )
    color_map = {crime_label: palette[index % len(palette)] for index, crime_label in enumerate(crime_order)}

    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_customdata: list[str] = []
    node_index: dict[str, int] = {}
    stage_counts = {"news": 1, "crime_disc": 0, "crime_cls": 0, "modus_disc": 0, "modus_cls": 0}

    def add_node(node_id: str, label: str, color: str, stage: str, hover: str) -> int:
        existing = node_index.get(node_id)
        if existing is not None:
            return existing
        stage_position = {
            "news": 0.02,
            "crime_disc": 0.24,
            "crime_cls": 0.46,
            "modus_disc": 0.68,
            "modus_cls": 0.90,
        }[stage]
        stage_slot = stage_counts[stage]
        total_slots = max(
            1,
            {
                "news": 1,
                "crime_disc": int(focused["crime_label"].nunique()),
                "crime_cls": int(focused["crime_label"].nunique()),
                "modus_disc": int(focused["modus_label"].nunique()),
                "modus_cls": int(focused["modus_label"].nunique()),
            }[stage],
        )
        y_position = 0.5 if stage == "news" else min(0.97, (stage_slot + 0.5) / (total_slots + 0.1))
        index = len(node_labels)
        node_index[node_id] = index
        node_labels.append(label)
        node_colors.append(color)
        node_x.append(stage_position)
        node_y.append(y_position)
        node_customdata.append(hover)
        stage_counts[stage] += 1
        return index

    root_total = int(focused["count"].sum())
    root_idx = add_node(
        "root:news",
        "Noticias\nWNN",
        "#e9d5ff",
        "news",
        (
            "Entrada da visualizacao Sankey.<br>"
            f"fluxos classificados={root_total}<br>"
            "Uma mesma noticia pode aparecer em mais de um ramo de modus operandi."
        ),
    )

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []
    link_labels: list[str] = []

    crime_totals = (
        focused.groupby("crime_label", as_index=False)["count"]
        .sum()
        .sort_values(["count", "crime_label"], ascending=[False, True])
    )
    for item in crime_totals.itertuples():
        crime_label = str(item.crime_label)
        total = int(item.count)
        crime_color = color_map.get(crime_label, "#bfdbfe")
        crime_markers_preview = " | ".join(crime_markers.get(crime_label, [])[:3]) or _humanize_label(crime_label)
        crime_disc_idx = add_node(
            f"crime_disc::{crime_label}",
            _clip_label(crime_markers_preview, 28),
            _hex_to_rgba(crime_color, 0.86),
            "crime_disc",
            (
                f"Discriminadores de crime<br>{_humanize_label(crime_label)}<br>"
                f"fluxos={total}<br>"
                f"marcadores: {crime_markers_preview}"
            ),
        )
        crime_cls_idx = add_node(
            f"crime_cls::{crime_label}",
            _clip_label(_humanize_label(crime_label), 26),
            _hex_to_rgba(crime_color, 1.0),
            "crime_cls",
            f"Crime canonico<br>{_humanize_label(crime_label)}<br>fluxos={total}",
        )
        sources.extend([root_idx, crime_disc_idx])
        targets.extend([crime_disc_idx, crime_cls_idx])
        values.extend([total, total])
        link_colors.extend([_hex_to_rgba(crime_color, 0.24), _hex_to_rgba(crime_color, 0.38)])
        link_labels.extend(
            [
                f"Noticias -> discriminadores de crime: {_humanize_label(crime_label)}",
                f"Discriminadores de crime -> crime canonico: {_humanize_label(crime_label)}",
            ]
        )

    for item in focused.itertuples():
        crime_label = str(item.crime_label)
        modus_label = str(item.modus_label)
        modus_display = str(item.modus_display)
        count = int(item.count)
        crime_color = color_map.get(crime_label, "#bfdbfe")
        modus_markers_preview = " | ".join(modus_markers.get(modus_display, [])[:3]) or _humanize_label(modus_display)
        crime_cls_idx = node_index[f"crime_cls::{crime_label}"]
        modus_disc_idx = add_node(
            f"modus_disc::{crime_label}::{modus_label}",
            _clip_label(modus_markers_preview, 28),
            _hex_to_rgba(crime_color, 0.66),
            "modus_disc",
            (
                f"Discriminadores de modus<br>{_humanize_label(modus_display)}<br>"
                f"fluxos={count}<br>"
                f"marcadores: {modus_markers_preview}"
            ),
        )
        modus_cls_idx = add_node(
            f"modus_cls::{crime_label}::{modus_label}",
            _clip_label(_humanize_label(modus_display), 26),
            _hex_to_rgba(crime_color, 0.96),
            "modus_cls",
            f"Modus operandi classificado<br>{_humanize_label(modus_display)}<br>fluxos={count}",
        )
        sources.extend([crime_cls_idx, modus_disc_idx])
        targets.extend([modus_disc_idx, modus_cls_idx])
        values.extend([count, count])
        link_colors.extend([_hex_to_rgba(crime_color, 0.28), _hex_to_rgba(crime_color, 0.42)])
        link_labels.extend(
            [
                f"Crime -> discriminadores de modus: {_humanize_label(crime_label)} -> {_humanize_label(modus_display)}",
                f"Discriminadores de modus -> modus operandi: {_humanize_label(modus_display)}",
            ]
        )

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            valueformat="d",
            node={
                "pad": 18,
                "thickness": 18,
                "line": {"color": "rgba(15, 23, 42, 0.18)", "width": 0.6},
                "label": node_labels,
                "color": node_colors,
                "x": node_x,
                "y": node_y,
                "customdata": node_customdata,
                "hovertemplate": "%{customdata}<extra></extra>",
            },
            link={
                "source": sources,
                "target": targets,
                "value": values,
                "color": link_colors,
                "label": link_labels,
                "hovertemplate": "%{label}<br>fluxos=%{value}<extra></extra>",
            },
        )
    )
    fig.update_layout(
        title=(
            f"Fluxo Sankey WNN -> {selected_crime} -> modus operandi"
            if selected_crime and selected_crime != "__top__"
            else "Fluxo Sankey WNN -> crime canonico -> modus operandi"
        ),
        height=760,
        margin={"l": 10, "r": 10, "t": 68, "b": 36},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"size": 12, "color": "#162033"},
        annotations=[
            {
                "text": "Noticias",
                "xref": "paper",
                "yref": "paper",
                "x": 0.03,
                "y": 1.07,
                "showarrow": False,
                "font": {"size": 12, "color": "#667085"},
            },
            {
                "text": "Discriminadores de crime",
                "xref": "paper",
                "yref": "paper",
                "x": 0.24,
                "y": 1.07,
                "showarrow": False,
                "font": {"size": 12, "color": "#667085"},
            },
            {
                "text": "Crime canonico",
                "xref": "paper",
                "yref": "paper",
                "x": 0.46,
                "y": 1.07,
                "showarrow": False,
                "font": {"size": 12, "color": "#667085"},
            },
            {
                "text": "Discriminadores de modus",
                "xref": "paper",
                "yref": "paper",
                "x": 0.68,
                "y": 1.07,
                "showarrow": False,
                "font": {"size": 12, "color": "#667085"},
            },
            {
                "text": "Modus operandi",
                "xref": "paper",
                "yref": "paper",
                "x": 0.90,
                "y": 1.07,
                "showarrow": False,
                "font": {"size": 12, "color": "#667085"},
            },
        ],
    )

    status = (
        f"{summary.get('flow_total', 0)} fluxos classificados visiveis, "
        f"{summary.get('crime_count', 0)} crimes e {summary.get('modus_count', 0)} saidas de modus. "
        f"Cada crime passa por um bloco de discriminadores e depois por um bloco final de modus. "
        + (
            f"Filtro atual: {_humanize_label(summary.get('selected_crime', ''))}. "
            if summary.get("selected_crime")
            else f"Recorte atual: top {summary.get('crime_count', 0)} crimes e ate {summary.get('top_modus_limit', 0)} modos por crime. "
        )
        + (
            f"{summary.get('trimmed_modus', 0)} modos adicionais foram agregados como 'demais modos'. "
            if int(summary.get("trimmed_modus", 0)) > 0
            else ""
        )
        + "Uma mesma noticia pode aparecer em mais de um fluxo quando recebe multiplos modus operandi."
    )
    return fig, status


def _wnn_flow_mode_options() -> list[dict[str, str]]:
    return [
        {"label": "Sankey analitico", "value": "sankey"},
        {"label": "Fluxo animado", "value": "animado"},
    ]


def _wnn_flow_particle_srcdoc(selected_crime: str = "__top__") -> tuple[str, str]:
    focused, summary = _build_sankey_focus(selected_crime)
    if focused.empty:
        return (
            """
<!DOCTYPE html>
<html lang="pt-br">
<head><meta charset="UTF-8" /><style>body{margin:0;font-family:Arial,sans-serif;background:#fff;color:#475467;display:flex;align-items:center;justify-content:center;height:100vh;} .empty{padding:24px;text-align:center;}</style></head>
<body><div class="empty">Aguardando classificacoes finais para animar o fluxo WNN.</div></body>
</html>
""".strip(),
            "Sem classificacoes finais suficientes para montar o fluxo animado.",
        )

    crime_markers, modus_markers = _theme_marker_lookup()
    palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Safe
    crime_rows = (
        focused[["crime_label", "crime_total"]]
        .drop_duplicates()
        .sort_values(["crime_total", "crime_label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    color_map = {
        str(row.crime_label): palette[index % len(palette)]
        for index, row in enumerate(crime_rows.itertuples())
    }

    branch_rows = focused.sort_values(
        ["crime_total", "count", "crime_label", "modus_display"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    total_flow = max(1, int(branch_rows["count"].sum()))
    target_points = min(MAX_PARTICLE_POINTS, max(72, len(branch_rows) * 10))
    assigned_particles = 0
    branches: list[dict[str, Any]] = []
    for index, row in enumerate(branch_rows.itertuples()):
        crime_label = str(row.crime_label)
        modus_display = str(row.modus_display)
        branch_total = int(row.count)
        proportion = branch_total / total_flow
        particles = max(4, int(round(proportion * target_points)))
        assigned_particles += particles
        branches.append(
            {
                "id": f"branch_{index}",
                "crime": _humanize_label(crime_label),
                "crime_marker": " | ".join(crime_markers.get(crime_label, [])[:2]) or _humanize_label(crime_label),
                "modus": _humanize_label(modus_display),
                "modus_marker": " | ".join(modus_markers.get(modus_display, [])[:2]) or _humanize_label(modus_display),
                "count": branch_total,
                "color": color_map.get(crime_label, "#94a3b8"),
                "particles": particles,
            }
        )
    if branches and assigned_particles > MAX_PARTICLE_POINTS:
        overflow = assigned_particles - MAX_PARTICLE_POINTS
        for branch in sorted(branches, key=lambda item: item["particles"], reverse=True):
            if overflow <= 0:
                break
            removable = min(overflow, max(0, branch["particles"] - 4))
            branch["particles"] -= removable
            overflow -= removable

    stage_positions = {"source": 112, "crime_disc": 340, "modus_disc": 545, "modus": 760}
    top_margin = 108
    branch_row_gap = 118
    group_gap = 48
    layout_branches: list[dict[str, Any]] = []
    crime_groups: list[dict[str, Any]] = []
    current_y = top_margin
    grouped_branches: dict[str, list[dict[str, Any]]] = {}
    for branch in branches:
        group_key = str(branch.get("crime", ""))
        grouped_branches.setdefault(group_key, []).append(branch)
    for crime_label, group_branches in grouped_branches.items():
        branch_positions: list[float] = []
        group_start_y = current_y
        for branch in group_branches:
            branch_y = current_y
            branch_positions.append(branch_y)
            layout_branch = dict(branch)
            layout_branch["y"] = branch_y
            layout_branch["crime_y"] = 0.0
            layout_branches.append(layout_branch)
            current_y += branch_row_gap
        crime_y = sum(branch_positions) / max(1, len(branch_positions))
        crime_groups.append(
            {
                "crime": crime_label,
                "crime_marker": group_branches[0].get("crime_marker", crime_label),
                "y": crime_y,
                "start_y": group_start_y,
                "end_y": branch_positions[-1] if branch_positions else group_start_y,
                "count": int(sum(int(branch.get("count", 0)) for branch in group_branches)),
                "color": group_branches[0].get("color", "#94a3b8"),
                "branches": [str(branch.get("id", "")) for branch in group_branches],
            }
        )
        current_y += group_gap
    crime_y_by_branch = {
        branch_id: float(group["y"])
        for group in crime_groups
        for branch_id in group["branches"]
    }
    for branch in layout_branches:
        branch["crime_y"] = crime_y_by_branch.get(str(branch.get("id", "")), float(branch.get("y", 0.0)))

    canvas_height = max(620, int(current_y + 24))
    payload = {
        "width": 980,
        "height": canvas_height,
        "sourceLabel": "Noticias",
        "sourceSubtitle": "base WNN",
        "sourceBox": {"x": 42, "y": max(250, canvas_height - 240), "width": 132, "height": 154},
        "columns": [
            {"key": "crime_disc", "label": "Discriminadores de crime", "x": stage_positions["crime_disc"]},
            {"key": "modus_disc", "label": "Discriminadores de modus", "x": stage_positions["modus_disc"]},
            {"key": "modus", "label": "Modus operandi", "x": stage_positions["modus"]},
        ],
        "sourceX": stage_positions["source"],
        "branches": layout_branches,
        "crimeGroups": crime_groups,
    }
    payload_json = json.dumps(payload, ensure_ascii=True)
    status = (
        f"Fluxo animado editorial com {len(layout_branches)} ramos visiveis e {sum(int(branch['particles']) for branch in layout_branches)} particulas de demonstracao. "
        + (
            f"Filtro atual: {_humanize_label(summary.get('selected_crime', ''))}. "
            if summary.get("selected_crime")
            else "O recorte segue os crimes mais frequentes e os modos principais de cada um. "
        )
        + "A leitura segue o caminho Noticias -> Discriminadores de crime -> Discriminadores de modus -> Modus operandi."
    )
    srcdoc = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8" />
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; background: #ffffff; color: #162033; }}
  .wrap {{ padding: 10px 12px; }}
  .toolbar {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }}
  .title {{ font-size: 13px; font-weight: 700; color: #162033; }}
  .hint {{ font-size: 11px; color: #667085; max-width: 700px; line-height: 1.3; }}
  button {{ border: 1px solid #c7d2e2; background: #ffffff; color: #162033; border-radius: 8px; padding: 6px 10px; font-size: 11px; font-weight: 700; cursor: pointer; }}
  button:hover {{ background: #f8fafc; }}
  svg {{ width: 100%; height: auto; background: #ffffff; border: 1px solid #e4e9f1; border-radius: 8px; }}
  .stage-label {{ font-size: 10px; font-weight: 700; fill: #667085; text-anchor: middle; letter-spacing: .01em; }}
  .branch-label {{ font-size: 9px; font-weight: 700; fill: #162033; }}
  .branch-meta {{ font-size: 8px; fill: #667085; }}
  .source-title {{ font-size: 11px; font-weight: 700; fill: #344054; }}
  .source-subtitle {{ font-size: 9px; fill: #667085; }}
  .source-shell {{ fill: rgba(15, 23, 42, 0.035); }}
  .soft-path {{ fill: none; stroke: rgba(15, 23, 42, 0.08); stroke-linecap: round; }}
  .stage-pill {{ fill: rgba(255,255,255,.94); stroke: #e4e9f1; stroke-width: 1; }}
  .pixel-box {{ fill: none; stroke: rgba(15,23,42,.08); stroke-width: .9; }}
  .particle {{ opacity: .94; }}
  .note {{ font-size: 8px; fill: #98a2b3; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="toolbar">
    <div>
      <div class="title">Fluxo animado WNN</div>
      <div class="hint">Leitura editorial do fluxo: a nuvem de noticias atravessa discriminadores de crime, depois discriminadores de modus, e termina nos blocos finais de modus operandi.</div>
    </div>
    <button id="replay" type="button">Reiniciar animacao</button>
  </div>
  <svg id="viz" viewBox="0 0 980 {canvas_height}" preserveAspectRatio="xMidYMid meet"></svg>
</div>
<script>
const payload = {payload_json};
const svg = document.getElementById("viz");
const NS = "http://www.w3.org/2000/svg";
const particleState = [];
function el(tag, attrs = {{}}, parent = svg) {{
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  parent.appendChild(node);
  return node;
}}
function bezier(t, p0, p1, p2, p3) {{
  return Math.pow(1 - t, 3) * p0 + 3 * Math.pow(1 - t, 2) * t * p1 + 3 * (1 - t) * Math.pow(t, 2) * p2 + Math.pow(t, 3) * p3;
}}
function branchPath(branch) {{
  const y = branch.y;
  const crimeY = branch.crime_y || y;
  const sourceBox = payload.sourceBox;
  const sx = sourceBox.x + sourceBox.width - 8;
  const sourceY = sourceBox.y + sourceBox.height * 0.48;
  const x1 = payload.columns[0].x - 42;
  const x2 = payload.columns[1].x - 42;
  const ex = payload.columns[2].x - 84;
  return {{
    sourceToCrimeDisc: {{ p0: [sx, sourceY], p1: [220, sourceY], p2: [285, crimeY], p3: [x1, crimeY] }},
    crimeDiscToModusDisc: {{ p0: [payload.columns[0].x + 42, crimeY], p1: [430, crimeY], p2: [490, y], p3: [x2, y] }},
    modusDiscToOutput: {{ p0: [payload.columns[1].x + 42, y], p1: [650, y], p2: [715, y], p3: [ex, y] }},
  }};
}}
function pathD(segment) {{
  return `M ${{segment.p0[0]}} ${{segment.p0[1]}} C ${{segment.p1[0]}} ${{segment.p1[1]}}, ${{segment.p2[0]}} ${{segment.p2[1]}}, ${{segment.p3[0]}} ${{segment.p3[1]}}`;
}}
function pixelGrid(parent, x, y, size, color, count) {{
  const cols = 10;
  const gap = 2.6;
  const cell = 5.4;
  const total = Math.max(12, Math.min(100, count));
  for (let i = 0; i < total; i += 1) {{
    const col = i % cols;
    const row = Math.floor(i / cols);
    el("rect", {{
      x: x + col * (cell + gap),
      y: y + row * (cell + gap),
      width: cell,
      height: cell,
      rx: 1.1,
      ry: 1.1,
      fill: color,
      opacity: .9
    }}, parent);
  }}
}}
function drawBase() {{
  svg.innerHTML = "";
  particleState.length = 0;
  el("text", {{ x: payload.sourceBox.x + 78, y: 34, class: "stage-label" }}).textContent = "Noticias";
  payload.columns.forEach((column) => {{
    el("text", {{ x: column.x, y: 34, class: "stage-label" }}).textContent = column.label;
  }});
  const source = payload.sourceBox;
  el("rect", {{ x: source.x, y: source.y, width: source.width, height: source.height, rx: 18, ry: 18, class: "source-shell" }});
  el("text", {{ x: source.x + 6, y: source.y - 16, class: "source-title" }}).textContent = payload.sourceLabel;
  el("text", {{ x: source.x + 6, y: source.y + 4, class: "source-subtitle" }}).textContent = payload.sourceSubtitle;
  el("text", {{ x: source.x + 6, y: source.y + source.height + 22, class: "source-title" }}).textContent = `${{payload.branches.length}} ramos ativos`;
  payload.crimeGroups.forEach((group) => {{
    const crimeDiscX = payload.columns[0].x - 76;
    el("rect", {{ x: crimeDiscX, y: group.y - 12, width: 132, height: 22, rx: 11, ry: 11, class: "stage-pill" }});
    el("text", {{ x: crimeDiscX + 10, y: group.y + 1, class: "branch-label" }}).textContent = group.crime_marker;
  }});
  payload.branches.forEach((branch) => {{
    const y = branch.y;
    const segments = branchPath(branch);
    const branchWidth = Math.max(16, Math.min(36, 14 + branch.count * 0.82));
    Object.values(segments).forEach((segment, segmentIndex) => {{
      el("path", {{ d: pathD(segment), class: "soft-path", "stroke-width": branchWidth - segmentIndex * 5 }});
    }});
    const modusDiscX = payload.columns[1].x - 76;
    const outputGroup = el("g", {{}});
    el("rect", {{ x: modusDiscX, y: y - 12, width: 132, height: 22, rx: 11, ry: 11, class: "stage-pill" }});
    el("text", {{ x: modusDiscX + 10, y: y + 1, class: "branch-label" }}).textContent = branch.modus_marker;
    const outputX = payload.columns[2].x - 12;
    const outputY = y - 20;
    el("rect", {{ x: outputX - 6, y: outputY - 3, width: 58, height: 58, rx: 7, ry: 7, class: "pixel-box" }}, outputGroup);
    pixelGrid(outputGroup, outputX, outputY, 50, branch.color, Math.min(64, branch.particles * 2));
    el("text", {{ x: outputX + 62, y: y - 1, class: "branch-label" }}, outputGroup).textContent = branch.modus;
    el("text", {{ x: outputX + 62, y: y + 11, class: "branch-meta" }}, outputGroup).textContent = `${{branch.count}} fluxos`;
    for (let i = 0; i < branch.particles; i += 1) {{
      const dot = el("rect", {{
        x: source.x + 10 + ((i * 11 + Math.floor(Math.random() * 14)) % (source.width - 22)),
        y: source.y + 10 + ((i * 9 + Math.floor(Math.random() * 18)) % (source.height - 22)),
        width: 3.4,
        height: 3.4,
        rx: 1.1,
        ry: 1.1,
        fill: branch.color,
        class: "particle"
      }});
      particleState.push({{ el: dot, branch, delay: Math.random() * 700 + i * 22, speed: 5600 + Math.random() * 900 }});
    }}
  }});
}}
function animate() {{
  const start = performance.now();
  function frame(now) {{
    const elapsed = now - start;
    particleState.forEach((particle) => {{
      const segments = branchPath(particle.branch);
      const total = particle.speed;
      const t = ((elapsed - particle.delay) % total + total) % total;
      const phase = t / total;
      let segment;
      let local;
      if (phase < 0.34) {{ segment = segments.sourceToCrimeDisc; local = phase / 0.34; }}
      else if (phase < 0.67) {{ segment = segments.crimeDiscToModusDisc; local = (phase - 0.34) / 0.33; }}
      else {{ segment = segments.modusDiscToOutput; local = (phase - 0.67) / 0.33; }}
      const eased = 0.5 - Math.cos(local * Math.PI) / 2;
      const x = bezier(eased, segment.p0[0], segment.p1[0], segment.p2[0], segment.p3[0]);
      const y = bezier(eased, segment.p0[1], segment.p1[1], segment.p2[1], segment.p3[1]);
      particle.el.setAttribute("x", x);
      particle.el.setAttribute("y", y);
    }});
    requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}}
drawBase();
animate();
document.getElementById("replay").addEventListener("click", () => {{ drawBase(); animate(); }});
</script>
</body>
</html>
""".strip()
    return srcdoc, status


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
    theme_rows = [row for row in theme_rows if str(row.get("kind", "crime")) == "crime"]
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


def _wnn_crime_modus_figure(selected_crime: str = "__top__", graph_mode: str = "exploracao") -> go.Figure:
    hierarchy = build_hierarchy_df()
    if hierarchy.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Arvore WNN -> crime canonico -> modus operandi",
            annotations=[
                {
                    "text": "Aguardando classificacoes finais para montar os galhos da WNN.",
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
            height=620,
            margin={"l": 20, "r": 20, "t": 52, "b": 20},
        )
        return fig

    color_map: dict[str, str] = {}
    palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Safe
    for index, crime_label in enumerate(hierarchy["crime_label"].drop_duplicates().tolist()):
        color_map[crime_label] = palette[index % len(palette)]
    if selected_crime and selected_crime != "__top__":
        hierarchy = hierarchy.loc[hierarchy["crime_label"] == selected_crime].copy()
    else:
        top_crimes = (
            hierarchy[["crime_label", "crime_total"]]
            .drop_duplicates()
            .sort_values(["crime_total", "crime_label"], ascending=[False, True])
            .head(MAX_GRAPH_CRIMES)["crime_label"]
            .tolist()
        )
        hierarchy = hierarchy.loc[hierarchy["crime_label"].isin(top_crimes)].copy()
    root_total = int(hierarchy["count"].sum())
    group_titles = {
        "execucao_criminosa": "execucao criminosa",
        "fraude_ocultacao_suporte": "fraude / ocultacao / suporte",
        "resposta_operacional": "resposta operacional",
        "outros_modus": "outros modus",
    }
    crime_groups: list[dict[str, Any]] = []
    for crime_label, group in hierarchy.groupby("crime_label", sort=False):
        grouped_rows: dict[str, list[Any]] = {
            "execucao_criminosa": [],
            "fraude_ocultacao_suporte": [],
            "resposta_operacional": [],
            "outros_modus": [],
        }
        for item in group.itertuples():
            grouped_rows[_modus_group(str(item.modus_label))].append(item)
        for key, items in grouped_rows.items():
            grouped_rows[key] = sorted(items, key=lambda item: (-int(item.count), str(item.modus_label)))[
                : (MAX_GRAPH_MODUS_PER_GROUP if selected_crime == "__top__" else 999)
            ]
        crime_groups.append(
            {
                "crime_label": str(crime_label),
                "crime_total": int(group["crime_total"].iloc[0]),
                "groups": grouped_rows,
            }
        )
    crime_groups.sort(key=lambda item: (-item["crime_total"], item["crime_label"]))

    graph = nx.Graph()
    graph.add_node(
        "root:wnn",
        label="WNN",
        kind="root",
        value=root_total,
        color="#dbeafe",
        hover=f"WNN<br>ocorrencias={root_total}",
        crime_label="",
    )
    initial_pos: dict[str, tuple[float, float]] = {"root:wnn": (0.0, 0.0)}

    crime_count = max(1, len(crime_groups))
    for index, crime in enumerate(crime_groups):
        angle = (2 * math.pi * index) / crime_count
        crime_id = f"crime:{crime['crime_label']}"
        crime_color = color_map.get(crime["crime_label"], "#bfdbfe")
        graph.add_node(
            crime_id,
            label=crime["crime_label"],
            kind="crime",
            value=crime["crime_total"],
            color=crime_color,
            hover=f"{crime['crime_label']}<br>ocorrencias={crime['crime_total']}",
            crime_label=crime["crime_label"],
        )
        graph.add_edge("root:wnn", crime_id, weight=max(1.0, math.log1p(crime["crime_total"])))
        initial_pos[crime_id] = (math.cos(angle) * 2.0, math.sin(angle) * 2.0)

        branch_offset = 0
        for group_key, items in crime["groups"].items():
            if not items:
                continue
            group_total = int(sum(int(item.count) for item in items))
            group_id = f"modus-group:{crime['crime_label']}:{group_key}"
            graph.add_node(
                group_id,
                label=group_titles[group_key],
                kind="group",
                value=group_total,
                color=crime_color,
                hover=f"{group_titles[group_key]}<br>crime={crime['crime_label']}<br>ocorrencias={group_total}",
                crime_label=crime["crime_label"],
            )
            graph.add_edge(crime_id, group_id, weight=max(1.0, math.log1p(group_total)))
            group_angle = angle + (branch_offset * 0.42) - 0.35
            initial_pos[group_id] = (math.cos(group_angle) * 3.0, math.sin(group_angle) * 3.0)
            for item_index, item in enumerate(items):
                modus_id = f"modus:{crime['crime_label']}:{item.modus_label}"
                count = int(item.count)
                graph.add_node(
                    modus_id,
                    label=str(item.modus_label),
                    kind="modus",
                    value=count,
                    color=crime_color,
                    hover=f"{item.modus_label}<br>crime={crime['crime_label']}<br>ocorrencias={count}",
                    crime_label=crime["crime_label"],
                )
                graph.add_edge(group_id, modus_id, weight=max(0.6, math.log1p(count) * 0.7))
                leaf_angle = group_angle + ((item_index - max(0, len(items) - 1) / 2) * 0.18)
                leaf_radius = 4.1 + min(0.7, item_index * 0.04)
                initial_pos[modus_id] = (math.cos(leaf_angle) * leaf_radius, math.sin(leaf_angle) * leaf_radius)
            branch_offset += 1

    layout = nx.spring_layout(
        graph,
        seed=42,
        pos=initial_pos,
        fixed=["root:wnn"],
        k=1.45 / max(1.0, math.sqrt(max(1, graph.number_of_nodes())) / 3),
        iterations=300,
        weight="weight",
    )

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges():
        x0, y0 = layout[source]
        x1, y1 = layout[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    publication_mode = graph_mode == "publicacao"
    max_value = max(int(data.get("value", 1) or 1) for _, data in graph.nodes(data=True))
    root_x: list[float] = []
    root_y: list[float] = []
    root_text: list[str] = []
    root_hover: list[str] = []
    root_size: list[float] = []
    crime_x: list[float] = []
    crime_y: list[float] = []
    crime_text: list[str] = []
    crime_hover: list[str] = []
    crime_size: list[float] = []
    crime_color_values: list[str] = []
    group_x: list[float] = []
    group_y: list[float] = []
    group_text: list[str] = []
    group_hover: list[str] = []
    group_size: list[float] = []
    group_color_values: list[str] = []
    modus_x: list[float] = []
    modus_y: list[float] = []
    modus_text: list[str] = []
    modus_hover: list[str] = []
    modus_size: list[float] = []
    modus_color_values: list[str] = []

    for node_id, data in graph.nodes(data=True):
        x, y = layout[node_id]
        kind = str(data.get("kind", "modus"))
        value = int(data.get("value", 1) or 1)
        color = str(data.get("color", "#94a3b8"))
        label = str(data.get("label", "")).replace("_", " ")
        hover = str(data.get("hover", label))
        scale = value / max_value if max_value else 0.0
        if kind == "root":
            root_x.append(x)
            root_y.append(y)
            root_text.append(f"WNN<br>{value}")
            root_hover.append(hover)
            root_size.append((34 + 34 * scale) if publication_mode else (30 + 28 * scale))
        elif kind == "crime":
            crime_x.append(x)
            crime_y.append(y)
            crime_text.append(f"{label}<br>{value}")
            crime_hover.append(hover)
            crime_size.append((18 + 30 * scale) if publication_mode else (16 + 26 * scale))
            crime_color_values.append(color)
        elif kind == "group":
            group_x.append(x)
            group_y.append(y)
            group_text.append("" if publication_mode else (label if selected_crime != "__top__" else ""))
            group_hover.append(hover)
            group_size.append((8 + 12 * scale) if publication_mode else (10 + 18 * scale))
            group_color_values.append(color)
        else:
            modus_x.append(x)
            modus_y.append(y)
            modus_text.append("")
            modus_hover.append(hover)
            modus_size.append((3 + 8 * scale) if publication_mode else (4 + 12 * scale))
            modus_color_values.append(color)

    fig = go.Figure()
    if publication_mode:
        for crime in crime_groups:
            crime_label = str(crime["crime_label"])
            crime_color = color_map.get(crime_label, "#bfdbfe")
            branch_nodes = [
                (node_id, data)
                for node_id, data in graph.nodes(data=True)
                if str(data.get("crime_label", "")) == crime_label
            ]
            branch_edge_x: list[float | None] = []
            branch_edge_y: list[float | None] = []
            for source, target in graph.edges():
                source_label = str(graph.nodes[source].get("crime_label", ""))
                target_label = str(graph.nodes[target].get("crime_label", ""))
                if crime_label not in {source_label, target_label}:
                    continue
                x0, y0 = layout[source]
                x1, y1 = layout[target]
                branch_edge_x.extend([x0, x1, None])
                branch_edge_y.extend([y0, y1, None])
            if branch_edge_x:
                fig.add_trace(
                    go.Scatter(
                        x=branch_edge_x,
                        y=branch_edge_y,
                        mode="lines",
                        line={"color": "rgba(148,163,184,0.16)", "width": 0.65},
                        hoverinfo="skip",
                        legendgroup=crime_label,
                        showlegend=False,
                    )
                )
            branch_x: list[float] = []
            branch_y: list[float] = []
            branch_sizes: list[float] = []
            branch_hovers: list[str] = []
            branch_symbols: list[str] = []
            for node_id, data in branch_nodes:
                x, y = layout[node_id]
                kind = str(data.get("kind", "modus"))
                value = int(data.get("value", 1) or 1)
                scale = value / max_value if max_value else 0.0
                if kind == "crime":
                    size = 18 + 30 * scale
                    symbol = "circle"
                elif kind == "group":
                    size = 8 + 12 * scale
                    symbol = "circle"
                else:
                    size = 3 + 8 * scale
                    symbol = "circle"
                branch_x.append(x)
                branch_y.append(y)
                branch_sizes.append(size)
                branch_hovers.append(str(data.get("hover", "")))
                branch_symbols.append(symbol)
            fig.add_trace(
                go.Scatter(
                    x=branch_x,
                    y=branch_y,
                    mode="markers",
                    marker={
                        "size": branch_sizes,
                        "color": crime_color,
                        "opacity": 0.86,
                        "line": {"color": "rgba(71,84,103,0.65)", "width": 0.35},
                        "symbol": branch_symbols,
                    },
                    hovertext=branch_hovers,
                    hoverinfo="text",
                    legendgroup=crime_label,
                    name=f"{crime_label} ({crime['crime_total']})",
                    showlegend=True,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=root_x,
                y=root_y,
                mode="markers+text",
                text=root_text,
                textposition="middle center",
                marker={
                    "size": root_size,
                    "color": "#e9d5ff",
                    "line": {"color": "#6b21a8", "width": 2},
                    "opacity": 0.98,
                },
                hovertext=root_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "rgba(148,163,184,0.22)", "width": 0.8},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=modus_x,
                y=modus_y,
                mode="markers",
                marker={
                    "size": modus_size,
                    "color": modus_color_values,
                    "opacity": 0.72,
                    "line": {"color": "rgba(255,255,255,0.45)", "width": 0.35},
                },
                hovertext=modus_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group_x,
                y=group_y,
                mode="markers+text",
                text=group_text,
                textposition="top center",
                marker={
                    "size": group_size,
                    "color": group_color_values,
                    "opacity": 0.9,
                    "line": {"color": "#475467", "width": 0.8},
                },
                hovertext=group_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=crime_x,
                y=crime_y,
                mode="markers+text",
                text=crime_text,
                textposition="middle center",
                marker={
                    "size": crime_size,
                    "color": crime_color_values,
                    "opacity": 0.96,
                    "line": {"color": "#111827", "width": 1.4},
                },
                hovertext=crime_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=root_x,
                y=root_y,
                mode="markers+text",
                text=root_text,
                textposition="middle center",
                marker={
                    "size": root_size,
                    "color": "#e9d5ff",
                    "line": {"color": "#6b21a8", "width": 2},
                    "opacity": 0.98,
                },
                hovertext=root_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=(
            f"Grafo WNN -> {selected_crime} -> modus operandi"
            if selected_crime and selected_crime != "__top__"
            else "Grafo WNN -> crime canonico -> modus operandi"
        ),
        height=900 if publication_mode else 820,
        margin={"l": 20, "r": 220 if publication_mode else 20, "t": 52, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={"visible": False},
        yaxis={"visible": False},
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 11},
            "groupclick": "togglegroup",
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "#d0d5dd",
            "borderwidth": 1,
        } if publication_mode else {"orientation": "v"},
        annotations=[
            {
                "text": (
                    ""
                    if publication_mode
                    else (
                        "Rede por forcas: crimes maiores puxam seus grupos e modos; as cores seguem o crime canonico."
                        if selected_crime and selected_crime != "__top__"
                        else "Rede por forcas: a raiz WNN conecta crimes, grupos operacionais e modos em forma de constelacao."
                    )
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 1.03,
                "showarrow": False,
                "font": {"size": 12, "color": "#475467"},
            }
        ],
    )
    if publication_mode:
        fig.update_traces(textfont={"size": 11, "color": "#13294b"}, selector={"mode": "markers+text"})
    return fig


def create_app() -> Dash:
    app = Dash(__name__)
    app.title = "NT_PF Dashboard"
    app.layout = html.Div(
        [
            dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0, disabled=not AUTO_REFRESH),
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("NT_PF - Acompanhamento WNN"),
                            html.Div(id="updated-at", className="muted"),
                        ],
                        className="header-copy",
                    ),
                    html.Button("Atualizar dashboard", id="refresh-button", n_clicks=0, className="refresh-button"),
                ],
                className="header",
            ),
            html.Div(id="kpis", className="kpi-grid"),
            html.Div(
                [
                    html.H2("Eixos de Classificacao"),
                    html.Div(
                        "Separacao operacional entre o classificador de tipo de crime e o classificador de modus operandi.",
                        className="muted cloud-help",
                    ),
                    html.Div(id="axis-overview", className="axis-overview-grid"),
                ],
                className="panel",
            ),
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
                    html.H2("Fluxo Sankey da Classificacao WNN"),
                    html.Div(
                        "Inspirado em leituras editoriais de fluxo, o Sankey mostra como as noticias entram na WNN, atravessam o eixo de crime, passam pelo eixo de modus operandi e chegam na classificacao final.",
                        className="muted cloud-help",
                    ),
                    dcc.Dropdown(
                        id="wnn-flow-filter",
                        options=_crime_filter_options(),
                        value="__top__",
                        clearable=False,
                        style={"marginBottom": "10px"},
                    ),
                    dcc.RadioItems(
                        id="wnn-flow-view-mode",
                        options=_wnn_flow_mode_options(),
                        value="sankey",
                        inline=True,
                        style={"marginBottom": "10px"},
                        inputStyle={"marginRight": "6px", "marginLeft": "12px"},
                    ),
                    html.Div(id="wnn-flow-status", className="muted cloud-help"),
                    dcc.Graph(id="wnn-flow-sankey", className="tree-graph"),
                    html.Iframe(
                        id="wnn-flow-particles",
                        className="flow-frame",
                        style={"display": "none"},
                    ),
                    html.H2("Imagem Binaria da Ultima Noticia"),
                    html.Div(id="binary-memory-panel", className="memory-wrap"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.H2("Grafo WNN -> Crime -> Modus Operandi"),
                    html.Div(
                        "O grafo parte da raiz WNN, passa pelo crime canonico principal, organiza os modos por grupo operacional e termina no marcador final. As contagens representam ocorrencias por no.",
                        className="muted cloud-help",
                    ),
                    dcc.Dropdown(
                        id="wnn-crime-filter",
                        options=_crime_filter_options(),
                        value="__top__",
                        clearable=False,
                        style={"marginBottom": "10px"},
                    ),
                    dcc.RadioItems(
                        id="wnn-graph-mode",
                        options=_graph_mode_options(),
                        value="exploracao",
                        inline=True,
                        style={"marginBottom": "10px"},
                        inputStyle={"marginRight": "6px", "marginLeft": "12px"},
                    ),
                    dcc.Graph(id="wnn-crime-modus-graph", className="tree-graph"),
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
            .header { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 16px; }
            .header-copy { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
            h1 { font-size: 24px; margin: 0; }
            h2 { font-size: 16px; margin: 0 0 12px 0; }
            .muted { color: #667085; font-size: 13px; }
            .refresh-button { border: 1px solid #c7d2e2; background: #ffffff; color: #162033; border-radius: 8px; padding: 10px 14px; font-size: 13px; font-weight: 700; cursor: pointer; white-space: nowrap; }
            .refresh-button:hover { background: #f8fafc; }
            .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 14px; }
            .kpi { background: #ffffff; border: 1px solid #d9e0ea; border-radius: 8px; padding: 14px; }
            .kpi-title { color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
            .kpi-value { font-size: 28px; font-weight: 700; margin-top: 6px; color: #111827; }
            .kpi-subtitle { color: #667085; font-size: 12px; margin-top: 4px; }
            .graph-grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, .8fr); gap: 14px; margin-bottom: 14px; }
            .table-grid { display: grid; grid-template-columns: minmax(0, .85fr) minmax(0, 1.15fr); gap: 14px; }
            .panel { background: #ffffff; border: 1px solid #d9e0ea; border-radius: 8px; padding: 12px; }
            .axis-overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
            .axis-panel { border: 1px solid #d9e0ea; border-radius: 8px; padding: 12px; background: #f8fafc; }
            .axis-crime { background: linear-gradient(180deg, #f8fafc 0%, #eff6ff 100%); }
            .axis-modus { background: linear-gradient(180deg, #f8fafc 0%, #effaf5 100%); }
            .axis-title { font-size: 15px; font-weight: 700; color: #111827; margin-bottom: 10px; }
            .axis-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
            .chart-note { border-top: 1px solid #e4e9f1; color: #475467; font-size: 13px; line-height: 1.45; padding: 10px 4px 2px 4px; }
            .hidden-note { display: none; }
            .cloud-help { margin-bottom: 10px; }
            .tree-map-layout { display: grid; grid-template-columns: minmax(0, 1fr) 260px; gap: 12px; align-items: start; }
            .tree-graph { min-height: 620px; border: 1px solid #e4e9f1; border-radius: 6px; background: #ffffff; }
            .flow-frame { width: 100%; min-height: 680px; border: 1px solid #e4e9f1; border-radius: 6px; background: #ffffff; }
            .operational-legend { border: 1px solid #e4e9f1; border-radius: 6px; padding: 12px; background: #f8fafc; position: sticky; top: 12px; }
            .operational-legend h3 { font-size: 14px; margin: 0 0 10px 0; color: #111827; }
            .legend-row { display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 9px; align-items: center; color: #475467; font-size: 12px; line-height: 1.35; margin-bottom: 10px; }
            .legend-row-tall { align-items: start; }
            .legend-swatch { display: inline-block; width: 26px; height: 14px; border-radius: 3px; border: 1px solid #bfdbfe; }
            .legend-bar { background: #dbeafe; }
            .legend-bubble { width: 20px; height: 20px; border-radius: 50%; background: #2fb47c; border: 1px solid #1f2937; box-shadow: 0 0 0 4px rgba(47, 180, 124, .12); }
            .legend-gradient { width: 26px; height: 54px; border-radius: 4px; background: linear-gradient(to top, #440154, #31688e, #35b779, #fde725); border: 1px solid #d0d5dd; }
            .legend-note { border-top: 1px solid #e4e9f1; padding-top: 10px; color: #667085; font-size: 12px; line-height: 1.4; }
            .axis-discriminator-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; margin-bottom: 14px; }
            .axis-discriminator-panel { border: 1px solid #e4e9f1; border-radius: 8px; padding: 12px; background: #fbfcfe; }
            .axis-discriminator-panel h3 { font-size: 14px; margin: 0 0 8px 0; color: #111827; }
            .disc-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 0; align-items: stretch; }
            .disc-light { display: flex; align-items: flex-start; gap: 8px; min-height: 52px; flex: 1 1 240px; max-width: 320px; border: 1px solid #e4e9f1; border-radius: 8px; padding: 8px; background: #f8fafc; opacity: .72; }
            .disc-light.active { opacity: 1; box-shadow: 0 0 0 1px rgba(22, 163, 74, .12); }
            .disc-dot { width: 11px; height: 11px; border-radius: 50%; flex: 0 0 11px; display: inline-block; margin-top: 2px; }
            .disc-dot.off { background: #cbd5e1; box-shadow: inset 0 0 0 1px #94a3b8; }
            .disc-dot.on { background: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, .95), inset 0 0 0 1px #15803d; }
            .disc-text { min-width: 0; }
            .disc-desc { font-weight: 700; font-size: 12px; color: #111827; line-height: 1.25; word-break: break-word; }
            .disc-meta { font-size: 10px; color: #667085; margin-top: 2px; line-height: 1.25; }
            .disc-markers { color: #475467; font-size: 10px; line-height: 1.25; margin-top: 4px; word-break: break-word; }
            .memory-wrap { margin: 0 0 16px 0; }
            .memory-panel { border: 1px solid #d9e0ea; border-radius: 8px; padding: 12px; background: #f8fafc; }
            .memory-matrix {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(12px, 12px));
                gap: 5px;
                align-items: center;
                max-height: 260px;
                overflow: auto;
                margin: 10px 0 8px 0;
                padding: 12px;
                border-radius: 8px;
                background: #0b1218;
                border: 1px solid #14202a;
            }
            .memory-cell {
                width: 12px;
                height: 12px;
                border-radius: 3px;
                background: #16232c;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .025);
            }
            .memory-cell.on {
                background: #16c784;
                box-shadow: 0 0 10px rgba(22, 199, 132, .75), inset 0 0 0 1px rgba(255, 255, 255, .18);
            }
            .memory-cell.off { background: #16232c; }
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

    def _refresh_context() -> dict[str, Any]:
        previous = _latest_snapshot()
        current_summary = _summarize_run("atual", ANALYSIS_DIR)
        previous_summary = _summarize_run(previous.name, previous) if previous else {}
        comparison = _compare_classifications(previous, ANALYSIS_DIR)
        metrics = _read_metrics(ANALYSIS_DIR)
        axis_summary, axis_metrics = _classifier_axis_metrics()
        learned_rules = int(metrics.get("learned_rules", pd.Series(dtype=int)).sum()) if not metrics.empty else 0
        current_summary["learned_rules"] = learned_rules
        current_summary.update(axis_summary)
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
        if not metrics.empty and "wnn_memory_vocab_size" in metrics.columns:
            current_summary["wnn_memory_vocab_size"] = int(
                pd.to_numeric(metrics["wnn_memory_vocab_size"], errors="coerce").fillna(0).max()
            )
        if not metrics.empty and "wnn_memory_active_avg" in metrics.columns:
            current_summary["wnn_memory_active_avg"] = round(
                float(pd.to_numeric(metrics["wnn_memory_active_avg"], errors="coerce").fillna(0).mean()),
                4,
            )
        linguistic = _read_json(LINGUISTIC_PREPROCESSING_JSON)
        current_summary["semantic_original_tokens"] = linguistic.get("original_tokens", "")
        current_summary["semantic_kept_tokens"] = linguistic.get("kept_tokens", "")
        current_summary["semantic_reduction_ratio"] = linguistic.get("reduction_ratio", "")
        change_rows = _comparison_rows(comparison)
        change_columns = [{"name": key, "id": key} for key in (change_rows[0].keys() if change_rows else ["status"])]
        discriminator_rows, marker_rows, latest_active, latest_doc = _canonical_discriminator_activity()
        crime_axis_count = sum(1 for row in discriminator_rows if str(row.get("kind", "crime")) == "crime")
        modus_axis_count = sum(1 for row in discriminator_rows if str(row.get("kind", "crime")) == "modus")
        current_summary["learned_crime_discriminators"] = sum(
            int(row.get("learned_markers", 0))
            for row in discriminator_rows
            if str(row.get("kind", "crime")) == "crime"
        )
        current_summary["learned_modus_discriminators"] = sum(
            int(row.get("learned_markers", 0))
            for row in discriminator_rows
            if str(row.get("kind", "crime")) == "modus"
        )
        summary_rows = _summary_table(current_summary, previous_summary)
        summary_columns = [{"name": key, "id": key} for key in ("metrica", "atual", "baseline")]
        discriminator_table_rows = [
            {
                "ativo_ultima": "sim" if row.get("active") else "nao",
                "eixo": row.get("kind", "crime"),
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
                "eixo",
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
            f"Banco atual: {len(discriminator_rows)} discriminadores agregados "
            f"({crime_axis_count} de crime, {modus_axis_count} de modus) e {len(marker_rows)} marcadores"
            + (f": {latest_doc}" if latest_doc else ".")
        )
        return {
            "previous": previous,
            "current_summary": current_summary,
            "previous_summary": previous_summary,
            "comparison": comparison,
            "metrics": metrics,
            "axis_metrics": axis_metrics,
            "change_rows": change_rows,
            "change_columns": change_columns,
            "discriminator_rows": discriminator_rows,
            "marker_rows": marker_rows,
            "summary_rows": summary_rows,
            "summary_columns": summary_columns,
            "discriminator_table_rows": discriminator_table_rows,
            "discriminator_columns": discriminator_columns,
            "discriminator_status": discriminator_status,
        }

    @app.callback(
        Output("updated-at", "children"),
        Output("kpis", "children"),
        Output("axis-overview", "children"),
        Output("metrics-graph", "figure"),
        Output("rate-graph", "figure"),
        Output("summary-table", "data"),
        Output("summary-table", "columns"),
        Output("changes-table", "data"),
        Output("changes-table", "columns"),
        Input("refresh", "n_intervals"),
        Input("refresh-button", "n_clicks"),
    )
    def refresh_core(_n: int, _clicks: int):
        ctx = _refresh_context()
        previous = ctx["previous"]
        current_summary = ctx["current_summary"]
        previous_summary = ctx["previous_summary"]
        comparison = ctx["comparison"]
        metrics = ctx["metrics"]
        axis_metrics = ctx["axis_metrics"]
        return (
            (
                f"Atualiza automaticamente a cada {REFRESH_MS // 1000}s | baseline: {previous.name if previous else 'nenhum'}"
                if AUTO_REFRESH
                else f"Visualizacao estatica dos artefatos atuais | use o botao para recarregar | baseline: {previous.name if previous else 'nenhum'}"
            ),
            _summary_cards(current_summary, previous_summary, comparison),
            _axis_overview_panel(current_summary),
            _metrics_figure(metrics, axis_metrics),
            _rate_figure(metrics, axis_metrics),
            ctx["summary_rows"],
            ctx["summary_columns"],
            ctx["change_rows"],
            ctx["change_columns"],
        )

    @app.callback(
        Output("wnn-flow-status", "children"),
        Output("wnn-flow-sankey", "figure"),
        Output("wnn-flow-sankey", "style"),
        Output("wnn-flow-particles", "srcDoc"),
        Output("wnn-flow-particles", "style"),
        Output("binary-memory-panel", "children"),
        Input("refresh", "n_intervals"),
        Input("refresh-button", "n_clicks"),
        Input("wnn-flow-filter", "value"),
        Input("wnn-flow-view-mode", "value"),
    )
    def refresh_discriminators(_n: int, _clicks: int, selected_crime: str, view_mode: str):
        sankey_figure, sankey_status = _wnn_flow_sankey_figure(selected_crime or "__top__")
        particle_srcdoc, particle_status = _wnn_flow_particle_srcdoc(selected_crime or "__top__")
        selected_view = str(view_mode or "sankey")
        if selected_view == "animado":
            status = particle_status
            sankey_style = {"display": "none"}
            iframe_style = {"display": "block", "width": "100%", "minHeight": "680px"}
        else:
            status = sankey_status
            sankey_style = {"display": "block"}
            iframe_style = {"display": "none"}
        return (
            status,
            sankey_figure,
            sankey_style,
            particle_srcdoc,
            iframe_style,
            _binary_memory_panel(),
        )

    @app.callback(
        Output("wnn-crime-modus-graph", "figure"),
        Input("refresh", "n_intervals"),
        Input("refresh-button", "n_clicks"),
        Input("wnn-crime-filter", "value"),
        Input("wnn-graph-mode", "value"),
    )
    def refresh_graphs(_n: int, _clicks: int, selected_crime: str, graph_mode: str):
        return _wnn_crime_modus_figure(selected_crime or "__top__", graph_mode or "exploracao")

    return app


def main() -> None:
    host = os.getenv("PF_DASH_HOST", "127.0.0.1")
    port = int(os.getenv("PF_DASH_PORT", "8050"))
    app = create_app()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
