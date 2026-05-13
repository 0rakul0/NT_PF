from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUT = Path(__file__).resolve().parent / "figura-2-ciclo-aprendizado.png"

COLORS = {
    "input": "#f7f7f7",
    "process": "#eef0ff",
    "decision": "#fff8c9",
    "llm": "#fff0df",
    "output": "#e4f7df",
    "stroke": "#222222",
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=180)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str = "process", fontsize: float = 12) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.35,
            edgecolor=COLORS["stroke"],
            facecolor=COLORS[fc],
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#111111")

    def diamond(cx: float, cy: float, w: float, h: float, text: str, fontsize: float = 11.5) -> None:
        pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
        patch = Polygon(pts, closed=True, linewidth=1.35, edgecolor=COLORS["stroke"], facecolor=COLORS["decision"])
        ax.add_patch(patch)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color="#111111")

    def arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str | None = None,
        rad: float = 0.0,
        fontsize: float = 10.5,
    ) -> None:
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.35,
            color="#111111",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
        ax.add_patch(arr)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.18, label, ha="center", va="bottom", fontsize=fontsize, color="#111111")

    box(0.7, 5.55, 2.25, 0.75, "Notícia em\nMarkdown", "input")
    box(3.45, 5.42, 2.6, 1.0, "Classificador regex\n(regras estáticas\n+ aprendidas)", "process")
    diamond(7.15, 5.92, 1.75, 1.35, "Confiança\n≥ 0,85?")
    box(9.15, 5.55, 2.6, 0.75, "Aceitar classificação\nregex", "output")
    box(6.0, 3.72, 2.3, 0.82, "LLM com schema\nfechado", "llm")
    box(9.0, 3.72, 2.55, 0.82, "Validação e\nnormalização", "process")
    diamond(10.3, 2.45, 2.1, 1.25, "Evidência lexical\nsubstantiva?")
    box(6.0, 1.98, 2.35, 0.92, "Derivar regras\nregex candidatas", "process")
    box(3.25, 1.98, 2.1, 0.92, "Filtrar padrões\nfrágeis", "process")
    box(0.7, 1.98, 2.05, 0.92, "Salvar regras\naprendidas", "output")
    box(0.7, 3.55, 2.05, 0.85, "Próxima\nexecução", "input")
    box(9.05, 0.72, 2.75, 0.82, "Manter classificação LLM\nou reprocessar", "llm")

    arrow(2.95, 5.92, 3.45, 5.92)
    arrow(6.05, 5.92, 6.28, 5.92)
    arrow(8.02, 5.92, 9.15, 5.92, "sim")
    arrow(7.15, 5.24, 7.15, 4.54, "não")
    arrow(8.3, 4.13, 9.0, 4.13)
    arrow(10.28, 3.72, 10.28, 3.08)
    arrow(9.26, 2.45, 8.35, 2.45, "sim")
    arrow(6.0, 2.45, 5.35, 2.45)
    arrow(3.25, 2.45, 2.75, 2.45)
    arrow(1.72, 2.9, 1.72, 3.55)
    arrow(2.75, 3.98, 3.45, 5.45, "carrega")
    arrow(10.3, 1.82, 10.3, 1.55, "não")

    ax.text(
        6.75,
        6.86,
        "Ciclo de aprendizado contínuo: LLM resolve exceções e alimenta novas regras regex",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="#111111",
    )
    ax.text(
        6.75,
        0.25,
        "Casos recorrentes migram da camada LLM para a camada regex na execução seguinte, reduzindo custo e aumentando rastreabilidade.",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
