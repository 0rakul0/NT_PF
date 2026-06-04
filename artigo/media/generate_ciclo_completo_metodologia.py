from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


MEDIA = Path(__file__).resolve().parent


def draw_box(ax, x, y, w, h, text, face, edge="#2b2b2b", fs=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.05",
        linewidth=1.15,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111111")
    return (x, y, w, h)


def left(box):
    x, y, w, h = box
    return x, y + h / 2


def right(box):
    x, y, w, h = box
    return x + w, y + h / 2


def top(box):
    x, y, w, h = box
    return x + w / 2, y + h


def bottom(box):
    x, y, w, h = box
    return x + w / 2, y


def arrow(ax, start, end, label="", rad=0.0, color="#303030"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.15,
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
            fontsize=8.6,
            color=color,
        )


def row_arrows(ax, boxes):
    for a, b in zip(boxes, boxes[1:]):
        arrow(ax, right(a), left(b))


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(8, 9.55, "Ciclo completo da metodologia incremental", ha="center", fontsize=18, weight="bold")
    ax.text(
        8,
        9.15,
        "A amostra inicial descobre temas e gera regras; os lotes usam regex primeiro; apenas resíduos seguem para LLM.",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )

    ax.text(0.35, 8.35, "1. Fundação temática", fontsize=12, weight="bold", color="#1f4e79")
    fundacao = [
        draw_box(ax, 0.35, 7.55, 1.35, 0.66, "Base\ntextual", "#f4f4f4"),
        draw_box(ax, 2.05, 7.55, 1.45, 0.66, "Amostra\ninicial", "#dff0ff"),
        draw_box(ax, 3.85, 7.55, 1.55, 0.66, "Texto de\ndomínio", "#dff0ff"),
        draw_box(ax, 5.75, 7.55, 1.35, 0.66, "Embeddings", "#dff0ff"),
        draw_box(ax, 7.45, 7.55, 1.45, 0.66, "HDBSCAN\nclusters", "#dff0ff"),
        draw_box(ax, 9.25, 7.55, 1.55, 0.66, "Cosseno\nrefina folhas", "#dff0ff"),
        draw_box(ax, 11.15, 7.55, 1.55, 0.66, "Agente 1\ntemas", "#e9e4ff"),
        draw_box(ax, 13.05, 7.55, 1.55, 0.66, "Agente 2\nregex", "#e9e4ff"),
    ]
    row_arrows(ax, fundacao)
    banco_inicial = draw_box(ax, 14.95, 7.55, 0.8, 0.66, "Banco\nregex", "#cdf0c6", fs=9)
    arrow(ax, right(fundacao[-1]), left(banco_inicial))

    ax.text(0.35, 6.35, "2. Execução incremental", fontsize=12, weight="bold", color="#2f6f3e")
    execucao = [
        draw_box(ax, 0.35, 5.55, 1.65, 0.66, "Reserva\nincremental", "#f4f4f4"),
        draw_box(ax, 2.35, 5.55, 1.25, 0.66, "Parser", "#e6f6df"),
        draw_box(ax, 3.95, 5.55, 1.55, 0.66, "Texto de\ndomínio", "#e6f6df"),
        draw_box(ax, 5.85, 5.55, 1.7, 0.66, "Classificador\nregex-first", "#cdf0c6"),
    ]
    row_arrows(ax, execucao)
    classificado = draw_box(ax, 8.05, 5.9, 1.65, 0.66, "Classificado\npor regex", "#cdf0c6")
    residual = draw_box(ax, 8.05, 5.1, 1.65, 0.66, "Residual\nsem regra", "#ffe2bd")
    arrow(ax, right(execucao[-1]), left(classificado))
    arrow(ax, right(execucao[-1]), left(residual))
    ag3 = draw_box(ax, 10.15, 5.1, 1.45, 0.66, "Agente 3\nLLM", "#ffe2bd")
    decisao = draw_box(ax, 12.05, 5.1, 1.6, 0.66, "Decisão\nestruturada", "#ffe2bd")
    arrow(ax, right(residual), left(ag3))
    arrow(ax, right(ag3), left(decisao))
    auditoria1 = draw_box(ax, 14.15, 5.5, 1.6, 0.66, "Eventos e\nmétricas", "#ffffff")
    arrow(ax, right(classificado), left(auditoria1), rad=-0.05)
    arrow(ax, right(decisao), left(auditoria1), rad=0.05)

    ax.text(0.35, 4.05, "3. Aprendizado e fechamento do ciclo", fontsize=12, weight="bold", color="#7a4b00")
    aprendizagem = [
        draw_box(ax, 0.35, 3.25, 1.75, 0.66, "Decisão\nresidual", "#ffe2bd"),
        draw_box(ax, 2.45, 3.25, 1.55, 0.66, "Aprendiz\nregex", "#fff1d6"),
        draw_box(ax, 4.35, 3.25, 1.65, 0.66, "Validação\nda regra", "#fff1d6"),
        draw_box(ax, 6.35, 3.25, 1.75, 0.66, "Banco regex\nversionado", "#cdf0c6"),
    ]
    row_arrows(ax, aprendizagem)
    candidato = draw_box(ax, 2.45, 2.25, 1.9, 0.66, "Tema candidato\nou raro", "#eeeeee")
    arvore = draw_box(ax, 4.75, 2.25, 1.85, 0.66, "Organizador\nda árvore", "#e9e4ff")
    arvore_saida = draw_box(ax, 7.0, 2.25, 1.75, 0.66, "Árvore\nrefinada", "#e9e4ff")
    arrow(ax, bottom(aprendizagem[0]), top(candidato))
    arrow(ax, right(candidato), left(arvore))
    arrow(ax, right(arvore), left(arvore_saida))

    retorno = draw_box(
        ax,
        9.55,
        2.55,
        5.95,
        1.05,
        "Retroalimentação do próximo lote\n"
        "- Banco regex versionado alimenta o classificador regex-first\n"
        "- Árvore refinada atualiza labels disponíveis ao Agente 3\n"
        "- Casos raros recorrentes podem virar temas candidatos",
        "#f8f8f8",
        fs=10,
    )
    arrow(ax, right(aprendizagem[-1]), left(retorno))
    arrow(ax, right(arvore_saida), left(retorno))

    ax.text(
        8,
        0.8,
        "Resultado: classificação auditável, menor uso de LLM, banco de regras ampliado, árvore temática estável e trilha de auditoria.",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )

    out = MEDIA / "figura-1-ciclo-completo-metodologia.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
