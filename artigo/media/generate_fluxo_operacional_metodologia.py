from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("figura-2-fluxo-operacional-metodologia.png")


def box(ax, xy, wh, text, *, fc="#f8fafc", ec="#334155", size=8.9, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color="#0f172a")
    return patch


def arrow(ax, start, end, *, color="#64748b", rad=0.0, lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def label(ax, xy, text, *, color="#475569"):
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=7.8, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.955, "Fluxo operacional da metodologia incremental", fontsize=17, weight="bold", color="#0f172a")
    ax.text(
        0.04,
        0.918,
        "Representacao grafica do pseudocodigo: entradas, artefatos intermediarios, decisao regex-first e fechamento do ciclo.",
        fontsize=9.2,
        color="#475569",
    )

    # Faixa 1: preparacao da base.
    box(ax, (0.04, 0.76), (0.19, 0.105), "Entradas\nB: base textual\ny: alvo substantivo\np: amostra | L: lotes", fc="#e0f2fe", weight="bold")
    box(ax, (0.30, 0.76), (0.18, 0.105), "Ingerir e estruturar\nDefinir controles\nde dominio", fc="#ecfeff")
    box(ax, (0.55, 0.76), (0.18, 0.105), "Dividir base\nA: fundacao\nR: reserva incremental", fc="#ecfeff")

    # Faixa 2: fundacao tematica.
    box(
        ax,
        (0.06, 0.53),
        (0.24, 0.14),
        "Fundacao tematica\ntexto de dominio -> embeddings\nHDBSCAN -> clusters\ncosseno -> folhas consolidadas",
        fc="#fef9c3",
        weight="bold",
    )
    box(
        ax,
        (0.38, 0.53),
        (0.22, 0.14),
        "Agentes 1 e 2\nnomear temas canonicos\ngerar regex iniciais\nvalidar e versionar",
        fc="#fef3c7",
        weight="bold",
    )
    box(
        ax,
        (0.69, 0.53),
        (0.23, 0.14),
        "Artefatos operacionais\nTemas canonicos\nRegex versionadas\nCandidatos, raros e metricas",
        fc="#f1f5f9",
        weight="bold",
    )

    # Faixa 3: classificacao incremental e aprendizagem.
    box(ax, (0.05, 0.31), (0.17, 0.105), "Lote incremental\nparser + texto\nde dominio", fc="#dcfce7", weight="bold")
    box(ax, (0.30, 0.31), (0.16, 0.105), "Classificador\nregex-first", fc="#bbf7d0", weight="bold")
    box(ax, (0.54, 0.38), (0.17, 0.095), "Com label\nclassificacao\norigem: regex", fc="#d9f99d")
    box(ax, (0.54, 0.23), (0.17, 0.095), "Sem label\npacote residual\npara revisao", fc="#fee2e2")
    box(ax, (0.78, 0.23), (0.16, 0.095), "Agente 3\nrevisao residual\norigem: LLM", fc="#fecaca", weight="bold")
    box(ax, (0.78, 0.38), (0.16, 0.095), "Aprendizado\nnova regex\ntema candidato\nnoticia rara", fc="#ede9fe", weight="bold")

    box(ax, (0.28, 0.07), (0.25, 0.105), "Organizador\nrevisar arvore tematica\nregistrar metricas do lote", fc="#e0e7ff", weight="bold")
    box(
        ax,
        (0.64, 0.07),
        (0.30, 0.105),
        "Saidas\nclassificacoes finais | regex versionadas\narvore refinada | metricas | trilha de auditoria",
        fc="#f8fafc",
        weight="bold",
    )

    # Fluxo principal.
    arrow(ax, (0.23, 0.812), (0.30, 0.812))
    arrow(ax, (0.48, 0.812), (0.55, 0.812))
    arrow(ax, (0.64, 0.76), (0.18, 0.67), rad=0.10)
    arrow(ax, (0.30, 0.60), (0.38, 0.60))
    arrow(ax, (0.60, 0.60), (0.69, 0.60))
    arrow(ax, (0.80, 0.53), (0.38, 0.415), rad=-0.12)
    arrow(ax, (0.22, 0.362), (0.30, 0.362))
    arrow(ax, (0.46, 0.362), (0.54, 0.427))
    arrow(ax, (0.46, 0.362), (0.54, 0.277))
    arrow(ax, (0.71, 0.277), (0.78, 0.277))
    arrow(ax, (0.86, 0.325), (0.86, 0.38))
    arrow(ax, (0.625, 0.38), (0.405, 0.175), rad=0.05)
    arrow(ax, (0.86, 0.38), (0.455, 0.175), rad=0.05)
    arrow(ax, (0.53, 0.123), (0.64, 0.123))

    label(ax, (0.50, 0.416), "sim", color="#166534")
    label(ax, (0.50, 0.292), "nao", color="#991b1b")
    label(ax, (0.62, 0.50), "temas e regex")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
