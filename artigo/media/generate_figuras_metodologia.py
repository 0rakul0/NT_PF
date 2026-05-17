from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


MEDIA = Path(__file__).resolve().parent


BATCH_METRICS = pd.DataFrame(
    [
        ("lote_0001", 500, 487, 13, 1, 0.9740),
        ("lote_0002", 500, 469, 31, 4, 0.9380),
        ("lote_0003", 500, 479, 21, 4, 0.9580),
        ("lote_0004", 500, 486, 14, 3, 0.9720),
        ("lote_0005", 500, 466, 34, 7, 0.9320),
        ("lote_0006", 500, 474, 26, 3, 0.9480),
        ("lote_0007", 500, 481, 19, 1, 0.9620),
        ("lote_0008", 500, 470, 30, 4, 0.9400),
        ("lote_0009", 500, 477, 23, 2, 0.9540),
        ("lote_0010", 500, 462, 38, 4, 0.9240),
        ("lote_0011", 500, 477, 23, 3, 0.9540),
        ("lote_0012", 500, 464, 36, 2, 0.9280),
        ("lote_0013", 500, 472, 28, 6, 0.9440),
        ("lote_0014", 390, 363, 27, 7, 0.930769),
    ],
    columns=["lote", "docs", "regex", "residual", "aprendizados", "taxa_regex"],
)
BATCH_METRICS["iteracao"] = range(1, len(BATCH_METRICS) + 1)
BATCH_METRICS["taxa_regex_acumulada"] = BATCH_METRICS["regex"].cumsum() / BATCH_METRICS["docs"].cumsum()


THEME_COUNTS = pd.DataFrame(
    [
        ("trafico_drogas", 1264),
        ("crimes_contra_criancas", 1119),
        ("crime_organizado", 1067),
        ("corrupcao_desvio_recursos_publicos", 966),
        ("contrabando_descaminho", 552),
        ("crimes_ambientais", 490),
        ("armas_municoes", 273),
        ("crimes_previdenciarios", 241),
        ("crimes_eleitorais", 172),
        ("crimes_sistema_financeiro", 170),
        ("moeda_falsa", 139),
        ("fraudes_auxilios_beneficios", 112),
        ("radiodifusao_clandestina", 60),
        ("lavagem_dinheiro", 58),
        ("trabalho_escravo", 53),
        ("crimes_migratorios", 49),
        ("falsificacao_documental", 19),
        ("ameacas_e_terrorismo", 19),
        ("crimes_ciberneticos", 15),
        ("crimes_patrimoniais", 14),
        ("seguranca_privada_clandestina", 12),
        ("crimes_contra_saude_publica", 10),
        ("crimes_de_odio_e_extremismo", 9),
        ("noticias_raras", 7),
    ],
    columns=["tema", "noticias"],
)


CLUSTER_COUNTS = pd.DataFrame(
    [
        ("crimes_contra_criancas", 182),
        ("crime_organizado", 161),
        ("corrupcao_recursos_publicos", 125),
        ("trafico_drogas", 88),
        ("crimes_ambientais", 66),
        ("demais_clusters", 594),
    ],
    columns=["grupo", "noticias"],
)


def save_current(fig: plt.Figure, filename: str) -> None:
    out = MEDIA / filename
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


def draw_box(ax, xy, text, color="#eef0ff", width=2.6, height=0.72, fontsize=10.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.06",
        linewidth=1.15,
        edgecolor="#202020",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color="#111111")
    return x + width / 2, y + height / 2


def draw_arrow(ax, start, end, label="", rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.15,
        color="#222222",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(arr)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.15, label, ha="center", fontsize=9.5)


def ciclo_metodologia() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7, 7.55, "Ciclo autonomo de classificacao incremental", ha="center", fontsize=17, weight="bold")
    ax.text(
        7,
        7.17,
        "A LLM atua apenas no residual; regex domina o fluxo e noticias raras acumulam memoria ate haver recorrencia.",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )

    boxes = {
        "base": draw_box(ax, (0.5, 6.0), "Base completa\n8.106 noticias", "#f7f7f7"),
        "sample": draw_box(ax, (3.2, 6.0), "Amostra temporal\n15% / 1.216", "#e8f4f8"),
        "clusters": draw_box(ax, (5.9, 6.0), "Clusters + cosseno\n24 grupos", "#e8f4f8"),
        "a1": draw_box(ax, (8.6, 6.0), "Agente 1\n17 temas iniciais", "#eef0ff"),
        "a2": draw_box(ax, (11.3, 6.0), "Agente 2\n5.739 regex", "#eef0ff"),
        "reserve": draw_box(ax, (0.5, 4.35), "Reserva incremental\n6.890 noticias", "#f7f7f7"),
        "regex": draw_box(ax, (3.2, 4.35), "Banco regex\nclassifica primeiro", "#e4f7df"),
        "ok": draw_box(ax, (5.9, 4.35), "Aceita por regex\n6.527 noticias", "#e4f7df"),
        "a3": draw_box(ax, (3.2, 2.7), "Agente 3\n363 residuais", "#fff0df"),
        "learn": draw_box(ax, (5.9, 2.7), "Aprendiz regex\n51 regras", "#fff0df"),
        "tree": draw_box(ax, (8.6, 2.7), "Organizador da arvore\n6 macrotemas", "#eef0ff"),
        "rare": draw_box(ax, (11.3, 2.7), "noticias_raras\n7 casos finais", "#eeeeee"),
        "memory": draw_box(ax, (11.3, 1.15), "Memoria de assinaturas\npromove se recorrente", "#eeeeee"),
    }

    draw_arrow(ax, boxes["base"], boxes["sample"])
    draw_arrow(ax, boxes["sample"], boxes["clusters"])
    draw_arrow(ax, boxes["clusters"], boxes["a1"])
    draw_arrow(ax, boxes["a1"], boxes["a2"])
    draw_arrow(ax, boxes["base"], boxes["reserve"], rad=-0.22)
    draw_arrow(ax, boxes["reserve"], boxes["regex"])
    draw_arrow(ax, boxes["regex"], boxes["ok"], "match")
    draw_arrow(ax, (4.5, 4.35), (4.5, 3.42), "sem match")
    draw_arrow(ax, boxes["a3"], boxes["learn"])
    draw_arrow(ax, boxes["learn"], boxes["regex"], "incorpora", rad=-0.28)
    draw_arrow(ax, boxes["a3"], boxes["tree"], "candidatos", rad=0.12)
    draw_arrow(ax, boxes["tree"], boxes["regex"], "remapeia", rad=-0.28)
    draw_arrow(ax, boxes["a3"], boxes["rare"], "sem encaixe")
    draw_arrow(ax, boxes["rare"], boxes["memory"])
    draw_arrow(ax, boxes["memory"], boxes["tree"], "recorrencia", rad=-0.2)
    draw_arrow(ax, boxes["a2"], boxes["regex"], "inicializa", rad=0.15)

    save_current(fig, "figura-1-ciclo-metodologia-incremental.png")


def cluster_fundacao() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ordered = CLUSTER_COUNTS.sort_values("noticias")
    ax.barh(ordered["grupo"], ordered["noticias"], color="#4C78A8")
    ax.set_title("Fundacao tematica: principais grupos consolidados da amostra")
    ax.set_xlabel("Noticias na amostra")
    ax.set_ylabel("")
    for i, value in enumerate(ordered["noticias"]):
        ax.text(value + 6, i, str(value), va="center", fontsize=9)
    save_current(fig, "figura-2-clusters-fundacao.png")


def regex_residual() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(BATCH_METRICS["iteracao"], BATCH_METRICS["regex"], label="Regex", color="#4C78A8")
    ax.bar(
        BATCH_METRICS["iteracao"],
        BATCH_METRICS["residual"],
        bottom=BATCH_METRICS["regex"],
        label="Residual LLM",
        color="#F58518",
    )
    ax.set_title("Classificacao por regex e residual por lote")
    ax.set_xlabel("Lote")
    ax.set_ylabel("Noticias")
    ax.legend(loc="lower right", frameon=True)
    save_current(fig, "figura-3-regex-vs-residual.png")

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(BATCH_METRICS["iteracao"], BATCH_METRICS["taxa_regex"], marker="o", label="Taxa por lote", color="#4C78A8")
    ax.plot(
        BATCH_METRICS["iteracao"],
        BATCH_METRICS["taxa_regex_acumulada"],
        marker="o",
        label="Taxa acumulada",
        color="#54A24B",
    )
    ax.set_ylim(0.88, 1.0)
    ax.set_title("Cobertura regex por lote")
    ax.set_xlabel("Lote")
    ax.set_ylabel("Proporcao")
    ax.legend(loc="lower right", frameon=True)
    save_current(fig, "figura-4-taxa-regex.png")


def temas_finais() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    ordered = THEME_COUNTS.sort_values("noticias")
    colors = ["#7F7F7F" if tema == "noticias_raras" else "#4C78A8" for tema in ordered["tema"]]
    ax.barh(ordered["tema"], ordered["noticias"], color=colors)
    ax.set_title("Noticias por tema final apos organizacao da arvore")
    ax.set_xlabel("Noticias")
    ax.set_ylabel("")
    for i, value in enumerate(ordered["noticias"]):
        ax.text(value + 12, i, str(value), va="center", fontsize=8.5)
    save_current(fig, "figura-5-temas-finais.png")


def arvore_temas() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.2)
    ax.text(6.25, 6.85, "Arvore refinada: temas canonicos, macrotemas e noticias raras", ha="center", fontsize=15.5, weight="bold")

    root = draw_box(ax, (4.8, 5.85), "Arvore de temas\npos-incremental", "#f7f7f7", width=2.9)
    left = [
        ("Temas iniciais\n17", 0.8, 4.65, "#e8f4f8"),
        ("Macrotemas promovidos\n6", 4.8, 4.65, "#eef0ff"),
        ("Noticias raras\n7", 8.8, 4.65, "#eeeeee"),
    ]
    centers = []
    for text, x, y, color in left:
        c = draw_box(ax, (x, y), text, color, width=2.9)
        centers.append(c)
        draw_arrow(ax, root, c)

    macro_items = [
        "falsificacao_documental",
        "crimes_patrimoniais",
        "crimes_contra_saude_publica",
        "ameacas_e_terrorismo",
        "crimes_de_odio_e_extremismo",
        "seguranca_privada_clandestina",
    ]
    y = 3.65
    for i, item in enumerate(macro_items):
        c = draw_box(ax, (4.45, y - i * 0.55), item, "#fff0df", width=3.6, height=0.36, fontsize=8.8)
        draw_arrow(ax, centers[1], c)

    rare_items = [
        "execucao_mandado_prisional",
        "tortura_sequestro_carcere",
        "violencia_domestica_feminicidio",
        "assedio_sexual_coacao",
    ]
    y = 3.65
    for i, item in enumerate(rare_items):
        c = draw_box(ax, (8.45, y - i * 0.55), item, "#f2f2f2", width=3.5, height=0.36, fontsize=8.8)
        draw_arrow(ax, centers[2], c)

    ax.text(
        2.25,
        3.15,
        "Exemplos: trafico_drogas,\ncrimes_contra_criancas,\ncrime_organizado,\ncrimes_ambientais...",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#333333",
    )
    save_current(fig, "figura-6-arvore-temas-raras.png")


def main() -> None:
    ciclo_metodologia()
    cluster_fundacao()
    regex_residual()
    temas_finais()
    arvore_temas()


if __name__ == "__main__":
    main()
