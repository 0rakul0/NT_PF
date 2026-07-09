from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
MEDIA = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "data" / "analise_qualitativa" / "incremental" / "metrics_batches.csv"
THEMES_PATH = ROOT / "data" / "analise_qualitativa" / "incremental" / "noticias_por_tema_pos_quarentena.csv"


def save(fig: plt.Figure, filename: str) -> None:
    out = MEDIA / filename
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


def box(ax, x, y, w, h, text, face, fs=9.2, edge="#303030"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.05",
        linewidth=1.1,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111111")
    return x, y, w, h


def right(b):
    x, y, w, h = b
    return x + w, y + h / 2


def left(b):
    x, y, w, h = b
    return x, y + h / 2


def top(b):
    x, y, w, h = b
    return x + w / 2, y + h


def bottom(b):
    x, y, w, h = b
    return x + w / 2, y


def arrow(ax, start, end, label="", rad=0.0, color="#222222"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )


def row_arrows(ax, boxes):
    for a, b in zip(boxes, boxes[1:]):
        arrow(ax, right(a), left(b))


def figure_1_cycle() -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(8, 9.55, "Ciclo completo da metodologia incremental WNN", ha="center", fontsize=18, weight="bold")
    ax.text(
        8,
        9.15,
        "Amostra descobre temas; Agente 2 cria discriminadores; a WNN classifica primeiro; apenas residuos e ambiguidades seguem para LLM.",
        ha="center",
        fontsize=10.2,
        color="#333333",
    )

    ax.text(0.35, 8.35, "1. Fundacao tematica", fontsize=12, weight="bold", color="#1f4e79")
    fundacao = [
        box(ax, 0.35, 7.55, 1.25, 0.66, "Base\ntextual", "#f4f4f4"),
        box(ax, 1.95, 7.55, 1.35, 0.66, "Amostra\ninicial", "#dff0ff"),
        box(ax, 3.65, 7.55, 1.55, 0.66, "Corpo da\nnoticia", "#dff0ff"),
        box(ax, 5.55, 7.55, 1.25, 0.66, "Embeddings", "#dff0ff"),
        box(ax, 7.15, 7.55, 1.35, 0.66, "HDBSCAN\nclusters", "#dff0ff"),
        box(ax, 8.85, 7.55, 1.5, 0.66, "Cosseno\nrefina folhas", "#dff0ff"),
        box(ax, 10.7, 7.55, 1.45, 0.66, "Agente 1\ntemas", "#e9e4ff"),
        box(ax, 12.5, 7.55, 1.55, 0.66, "Agente 2\ndiscriminadores", "#e9e4ff"),
    ]
    row_arrows(ax, fundacao)
    memoria = box(ax, 14.45, 7.55, 1.15, 0.66, "Memoria\nWNN", "#cdf0c6", fs=8.8)
    arrow(ax, right(fundacao[-1]), left(memoria))

    ax.text(0.35, 6.35, "2. Execucao incremental", fontsize=12, weight="bold", color="#2f6f3e")
    execucao = [
        box(ax, 0.35, 5.55, 1.55, 0.66, "Reserva\nincremental", "#f4f4f4"),
        box(ax, 2.25, 5.55, 1.15, 0.66, "Parser", "#e6f6df"),
        box(ax, 3.75, 5.55, 1.55, 0.66, "Pre-processa\ntexto", "#e6f6df"),
        box(ax, 5.65, 5.55, 1.45, 0.66, "Imagem\nbinaria", "#e6f6df"),
        box(ax, 7.45, 5.55, 1.65, 0.66, "Classificador\nWNN", "#cdf0c6"),
    ]
    row_arrows(ax, execucao)
    aceito = box(ax, 9.65, 5.9, 1.6, 0.66, "Classificado\npela WNN", "#cdf0c6")
    residual = box(ax, 9.65, 5.1, 1.6, 0.66, "Residual ou\nambiguo", "#ffe2bd")
    ag3 = box(ax, 11.75, 5.1, 1.35, 0.66, "Agente 3\nLLM", "#ffe2bd")
    decisao = box(ax, 13.45, 5.1, 1.55, 0.66, "Decisao\nestruturada", "#ffe2bd")
    eventos = box(ax, 13.45, 5.9, 1.55, 0.66, "Eventos e\nmetricas", "#ffffff")
    arrow(ax, right(execucao[-1]), left(aceito))
    arrow(ax, right(execucao[-1]), left(residual))
    arrow(ax, right(residual), left(ag3))
    arrow(ax, right(ag3), left(decisao))
    arrow(ax, right(aceito), left(eventos), rad=-0.08)
    arrow(ax, top(decisao), bottom(eventos))

    ax.text(0.35, 4.05, "3. Aprendizado e fechamento do ciclo", fontsize=12, weight="bold", color="#7a4b00")
    aprendizagem = [
        box(ax, 0.35, 3.25, 1.65, 0.66, "Decisao\nresidual", "#ffe2bd"),
        box(ax, 2.35, 3.25, 1.65, 0.66, "Extrai\nmarcadores", "#fff1d6"),
        box(ax, 4.35, 3.25, 1.75, 0.66, "Sanitizacao\nAgente 2", "#fff1d6"),
        box(ax, 6.45, 3.25, 1.8, 0.66, "Memoria WNN\nversionada", "#cdf0c6"),
    ]
    row_arrows(ax, aprendizagem)
    raro = box(ax, 2.35, 2.25, 1.85, 0.66, "Tema candidato\nou raro", "#eeeeee")
    arvore = box(ax, 4.65, 2.25, 1.9, 0.66, "Organizador\nda arvore", "#e9e4ff")
    refinada = box(ax, 6.95, 2.25, 1.8, 0.66, "Arvore\nrefinada", "#e9e4ff")
    retorno = box(
        ax,
        9.65,
        2.45,
        5.45,
        1.12,
        "Retroalimentacao do proximo lote\n"
        "- memoria WNN atualiza discriminadores\n"
        "- arvore refina labels e pesos de preferencia\n"
        "- raros recorrentes viram candidatos controlados",
        "#f8f8f8",
        fs=9.4,
    )
    arrow(ax, bottom(aprendizagem[0]), top(raro))
    arrow(ax, right(raro), left(arvore))
    arrow(ax, right(arvore), left(refinada))
    arrow(ax, right(aprendizagem[-1]), left(retorno))
    arrow(ax, right(refinada), left(retorno))

    ax.text(
        8,
        0.8,
        "Resultado: classificacao auditavel por discriminadores, menor chamada de LLM, memoria binaria ampliada e trilha de auditoria.",
        ha="center",
        fontsize=10.4,
        color="#333333",
    )
    save(fig, "figura-1-ciclo-completo-metodologia.png")


def figure_5_6_metrics() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    metrics["taxa_wnn_acumulada"] = metrics["wnn_accepted"].cumsum() / metrics["docs"].cumsum()

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(metrics["iteration"], metrics["wnn_accepted"], label="WNN", color="#009E73")
    ax.bar(
        metrics["iteration"],
        metrics["post_wnn_residual"],
        bottom=metrics["wnn_accepted"],
        label="Residual pos-WNN",
        color="#E69F00",
    )
    ax.set_title("WNN e residual por lote")
    ax.set_xlabel("Lote")
    ax.set_ylabel("Noticias")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xticks(metrics["iteration"])
    save(fig, "figura-3-wnn-vs-residual.png")

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(metrics["iteration"], metrics["wnn_rate"], marker="o", label="Taxa WNN por lote", color="#009E73")
    ax.plot(metrics["iteration"], metrics["taxa_wnn_acumulada"], marker="o", label="Taxa WNN acumulada", color="#0072B2")
    ax.set_ylim(max(0, metrics["wnn_rate"].min() - 0.06), min(1.0, metrics["wnn_rate"].max() + 0.04))
    ax.set_title("Cobertura WNN por lote")
    ax.set_xlabel("Lote")
    ax.set_ylabel("Proporcao")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xticks(metrics["iteration"])
    save(fig, "figura-4-taxa-wnn.png")


def figure_7_themes() -> None:
    df = pd.read_csv(THEMES_PATH)
    df = df[df["label_pos_quarentena"] != "noticias_raras"].copy()
    df = df.sort_values("total", ascending=False).head(10).sort_values("total")
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    left_vals = [0] * len(df)
    parts = [
        ("classificado_wnn", "WNN/memoria", "#009E73"),
        ("classificado_agent3", "Agente 3/LLM", "#E69F00"),
        ("candidato_merge_into_existing", "ajuste arvore", "#56B4E9"),
        ("candidato_promote_to_canonical", "tema promovido", "#CC79A7"),
        ("candidato_keep_as_leaf", "folha mantida", "#999999"),
    ]
    for col, label, color in parts:
        vals = df[col].fillna(0).astype(float).tolist()
        ax.barh(y, vals, left=left_vals, label=label, color=color)
        left_vals = [a + b for a, b in zip(left_vals, vals)]

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label_pos_quarentena"])
    ax.set_title("Top 10 temas apos classificacao WNN e revisao residual")
    ax.set_xlabel("Noticias")
    ax.set_ylabel("Tema final")
    ax.legend(loc="lower right", frameon=True, title="Origem/decisao")
    save(fig, "figura-5-temas-finais.png")


def main() -> None:
    figure_1_cycle()
    figure_5_6_metrics()
    figure_7_themes()


if __name__ == "__main__":
    main()
