from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("figura-3-agente2-regex.png")


def box(ax, xy, wh, title, lines, *, fc="#f8fafc", ec="#334155"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="center", fontsize=11.0, weight="bold", color="#0f172a")
    ax.text(x + w / 2, y + h / 2 - 0.015, "\n".join(lines), ha="center", va="center", fontsize=9.0, color="#1e293b")
    return patch


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color="#64748b",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.94, "Contrato operacional do Agente 2", fontsize=17, weight="bold", color="#0f172a")
    ax.text(
        0.04,
        0.895,
        "Geracao de regex iniciais a partir de temas canonicos, folhas e evidencias observadas.",
        fontsize=9.4,
        color="#475569",
    )

    box(
        ax,
        (0.05, 0.50),
        (0.25, 0.28),
        "Entrada",
        [
            "tema canonico",
            "folhas associadas",
            "trechos positivos",
            "termos frequentes",
            "controles de dominio",
        ],
        fc="#e0f2fe",
    )
    box(
        ax,
        (0.38, 0.50),
        (0.24, 0.28),
        "Agente 2",
        [
            "gera padroes candidatos",
            "associa label",
            "ancora em evidencias",
            "marca sinais proibidos",
            "justifica a regra",
        ],
        fc="#fef3c7",
    )
    box(
        ax,
        (0.70, 0.50),
        (0.25, 0.28),
        "Saida",
        [
            "regex candidata",
            "label associada",
            "evidencias positivas",
            "sinais acidentais",
            "justificativa da regra",
        ],
        fc="#dcfce7",
    )
    box(
        ax,
        (0.18, 0.14),
        (0.64, 0.20),
        "Criterios de aceite",
        [
            "capturar evidencia substantiva",
            "rejeitar dependencia de localidade, orgao ou nome de operacao",
            "ser rastreavel ate exemplos observados",
            "nao ampliar excessivamente a classe",
        ],
        fc="#f1f5f9",
    )

    arrow(ax, (0.30, 0.64), (0.38, 0.64))
    arrow(ax, (0.62, 0.64), (0.70, 0.64))
    arrow(ax, (0.50, 0.50), (0.50, 0.34))
    arrow(ax, (0.82, 0.50), (0.73, 0.34))

    ax.text(0.50, 0.415, "valida", ha="center", va="center", fontsize=8.6, color="#475569")
    ax.text(0.76, 0.415, "aprova ou rejeita", ha="center", va="center", fontsize=8.6, color="#475569")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
