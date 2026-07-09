from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from scripts.incremental.wnn_crime_modus_tree import build_hierarchy_df
from scripts.project_config import ANALYSIS_DIR, PROJECT_ROOT


RUN_DIR = ANALYSIS_DIR / "incremental"
FIGURES_DIR = RUN_DIR / "figures"
ARTICLE_MEDIA_DIR = PROJECT_ROOT / "artigo" / "media"

OUTPUT_ARTICLE_PNG = ARTICLE_MEDIA_DIR / "figura-9-grafo-wnn-publicacao.png"
OUTPUT_ARTICLE_SVG = ARTICLE_MEDIA_DIR / "figura-9-grafo-wnn-publicacao.svg"
OUTPUT_RESULTS_PNG = FIGURES_DIR / "grafo_wnn_publicacao.png"
OUTPUT_RESULTS_SVG = FIGURES_DIR / "grafo_wnn_publicacao.svg"

TOP_CRIMES = 10
MAX_MODUS_PER_GROUP = 8

GROUP_TITLES = {
    "execucao_criminosa": "execucao criminosa",
    "fraude_ocultacao_suporte": "fraude / ocultacao / suporte",
    "resposta_operacional": "resposta operacional",
    "outros_modus": "outros modus",
}

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


def _crime_palette(crimes: list[str]) -> dict[str, str]:
    palette = [
        "#8ecae6",
        "#ffb703",
        "#219ebc",
        "#fb8500",
        "#cdb4db",
        "#f28482",
        "#90be6d",
        "#bde0fe",
        "#f9f7a1",
        "#84dcc6",
        "#c77dff",
        "#ffd6a5",
    ]
    return {crime: palette[index % len(palette)] for index, crime in enumerate(crimes)}


def build_publication_graph(hierarchy: pd.DataFrame) -> tuple[nx.Graph, dict[str, tuple[float, float]]]:
    top_crimes = (
        hierarchy[["crime_label", "crime_total"]]
        .drop_duplicates()
        .sort_values(["crime_total", "crime_label"], ascending=[False, True])
        .head(TOP_CRIMES)["crime_label"]
        .tolist()
    )
    frame = hierarchy.loc[hierarchy["crime_label"].isin(top_crimes)].copy()
    frame = frame.sort_values(["crime_total", "count", "crime_label", "modus_label"], ascending=[False, False, True, True])

    colors = _crime_palette(top_crimes)
    graph = nx.Graph()
    graph.add_node("root:wnn", kind="root", label="WNN", value=int(frame["count"].sum()), color="#ead7ff")
    initial_pos: dict[str, tuple[float, float]] = {"root:wnn": (0.0, 0.0)}

    for crime_index, crime_label in enumerate(top_crimes):
        group = frame.loc[frame["crime_label"] == crime_label].copy()
        if group.empty:
            continue
        angle = (2 * math.pi * crime_index) / max(1, len(top_crimes))
        crime_total = int(group["crime_total"].iloc[0])
        crime_id = f"crime:{crime_label}"
        graph.add_node(
            crime_id,
            kind="crime",
            label=crime_label,
            value=crime_total,
            color=colors[crime_label],
        )
        graph.add_edge("root:wnn", crime_id, weight=max(1.0, math.log1p(crime_total)))
        initial_pos[crime_id] = (math.cos(angle) * 2.2, math.sin(angle) * 2.2)

        grouped_rows: dict[str, list[pd.Series]] = {key: [] for key in GROUP_TITLES}
        for item in group.itertuples():
            grouped_rows[_modus_group(str(item.modus_label))].append(item)
        for offset, (group_key, items) in enumerate(grouped_rows.items()):
            if not items:
                continue
            items = sorted(items, key=lambda item: (-int(item.count), str(item.modus_label)))[:MAX_MODUS_PER_GROUP]
            group_total = int(sum(int(item.count) for item in items))
            group_id = f"group:{crime_label}:{group_key}"
            graph.add_node(
                group_id,
                kind="group",
                label=GROUP_TITLES[group_key],
                value=group_total,
                color=colors[crime_label],
            )
            graph.add_edge(crime_id, group_id, weight=max(0.8, math.log1p(group_total)))
            group_angle = angle + (offset * 0.40) - 0.45
            initial_pos[group_id] = (math.cos(group_angle) * 3.4, math.sin(group_angle) * 3.4)

            for item_index, item in enumerate(items):
                modus_id = f"modus:{crime_label}:{item.modus_label}"
                count = int(item.count)
                graph.add_node(
                    modus_id,
                    kind="modus",
                    label=str(item.modus_label),
                    value=count,
                    color=colors[crime_label],
                )
                graph.add_edge(group_id, modus_id, weight=max(0.3, math.log1p(count) * 0.5))
                leaf_angle = group_angle + ((item_index - max(0, len(items) - 1) / 2) * 0.16)
                initial_pos[modus_id] = (math.cos(leaf_angle) * 4.5, math.sin(leaf_angle) * 4.5)

    layout = nx.spring_layout(
        graph,
        seed=42,
        pos=initial_pos,
        fixed=["root:wnn"],
        k=1.55 / max(1.0, math.sqrt(max(1, graph.number_of_nodes())) / 3),
        iterations=320,
        weight="weight",
    )
    return graph, layout


def draw_publication_graph(graph: nx.Graph, layout: dict[str, tuple[float, float]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    edge_segments = [
        [layout[source], layout[target]]
        for source, target in graph.edges()
    ]
    from matplotlib.collections import LineCollection

    edge_collection = LineCollection(edge_segments, colors=[(0.58, 0.64, 0.72, 0.15)], linewidths=0.55, zorder=1)
    ax.add_collection(edge_collection)

    root_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "root"]
    crime_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "crime"]
    group_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "group"]
    modus_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "modus"]

    max_value = max(int(data.get("value", 1) or 1) for _, data in graph.nodes(data=True))

    def scatter_nodes(nodes: list[str], size_base: float, size_gain: float, alpha: float, line_width: float, zorder: int) -> None:
        if not nodes:
            return
        xs = [layout[node][0] for node in nodes]
        ys = [layout[node][1] for node in nodes]
        colors = [graph.nodes[node].get("color", "#94a3b8") for node in nodes]
        sizes = [
            size_base + size_gain * (int(graph.nodes[node].get("value", 1) or 1) / max_value)
            for node in nodes
        ]
        ax.scatter(xs, ys, s=sizes, c=colors, alpha=alpha, edgecolors="#475467", linewidths=line_width, zorder=zorder)

    scatter_nodes(modus_nodes, 10, 60, 0.78, 0.15, 2)
    scatter_nodes(group_nodes, 30, 90, 0.82, 0.45, 3)
    scatter_nodes(crime_nodes, 120, 520, 0.95, 0.9, 4)
    if root_nodes:
        root = root_nodes[0]
        ax.scatter(
            [layout[root][0]],
            [layout[root][1]],
            s=[1600],
            c=["#ead7ff"],
            alpha=0.98,
            edgecolors="#6b21a8",
            linewidths=1.8,
            zorder=5,
        )

    for node in crime_nodes:
        x, y = layout[node]
        label = str(graph.nodes[node].get("label", "")).replace("_", " ")
        value = int(graph.nodes[node].get("value", 0) or 0)
        ax.text(
            x,
            y,
            f"{label}\n{value}",
            ha="center",
            va="center",
            fontsize=11.5,
            color="#16325c",
            fontweight="medium",
            zorder=6,
        )

    if root_nodes:
        root = root_nodes[0]
        x, y = layout[root]
        value = int(graph.nodes[root].get("value", 0) or 0)
        ax.text(
            x,
            y,
            f"WNN\n{value}",
            ha="center",
            va="center",
            fontsize=12.5,
            color="#4c1d95",
            fontweight="bold",
            zorder=7,
        )

    ax.text(
        0.01,
        0.98,
        "Grafo WNN de crimes canonicos e modus operandi",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color="#12284b",
    )
    ax.text(
        0.01,
        0.945,
        "Modo de publicacao: raiz WNN, crimes centrais e constelacoes de grupos/modos com cor por crime canonico.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#5b6b80",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def run() -> list[Path]:
    hierarchy = build_hierarchy_df()
    if hierarchy.empty:
        return []
    graph, layout = build_publication_graph(hierarchy)
    outputs = []
    for output_path in (OUTPUT_ARTICLE_PNG, OUTPUT_ARTICLE_SVG, OUTPUT_RESULTS_PNG, OUTPUT_RESULTS_SVG):
        outputs.append(draw_publication_graph(graph, layout, output_path))
    return outputs


def main() -> None:
    for path in run():
        print(path)


if __name__ == "__main__":
    main()
