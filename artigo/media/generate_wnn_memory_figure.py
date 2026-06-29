from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MEDIA = Path(__file__).resolve().parent
BANK_PATH = ROOT / "data" / "analise_qualitativa" / "wnn_feature_bank.json"
OUT_PATH = MEDIA / "dashboard_wnn_memoria_binaria_linkedin.png"


def load_bank() -> dict:
    return json.loads(BANK_PATH.read_text(encoding="utf-8"))


def top_theme_counts(bank: dict, limit: int = 8) -> pd.DataFrame:
    rows = []
    for label, payload in bank.get("themes", {}).items():
        rows.append(
            {
                "label": label,
                "discriminator_count": int(payload.get("discriminator_count", 0)),
            }
        )
    df = pd.DataFrame(rows).sort_values("discriminator_count", ascending=False).head(limit)
    return df.sort_values("discriminator_count")


def pick_example(bank: dict) -> dict:
    discriminators = bank.get("discriminators", [])
    if not discriminators:
        raise ValueError("wnn_feature_bank.json sem discriminadores")
    return max(discriminators, key=lambda item: len(item.get("memory_positions", [])))


def draw_summary_box(ax, x: float, y: float, w: float, h: float, title: str, value: str, face: str) -> None:
    rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor="#d0d7de", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x + 0.04 * w, y + 0.63 * h, title, fontsize=10, color="#475467", va="center", ha="left")
    ax.text(x + 0.04 * w, y + 0.28 * h, value, fontsize=18, weight="bold", color="#101828", va="center", ha="left")


def build_figure() -> None:
    bank = load_bank()
    vocab = bank.get("memory_vocab", {})
    vocab_size = int(vocab.get("size", 0))
    labels = bank.get("labels", [])
    memories = bank.get("memories", {})
    example = pick_example(bank)
    top_themes = top_theme_counts(bank)

    fig = plt.figure(figsize=(14, 8), dpi=180, facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 1.8, 1.6], wspace=0.45, hspace=0.5)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.82,
        "Memoria binaria WNN e discriminadores canonicos",
        fontsize=20,
        weight="bold",
        ha="left",
        va="center",
        color="#101828",
    )
    ax_title.text(
        0.0,
        0.38,
        "A camada deterministica transforma marcadores sanitizados em posicoes binarias reutilizaveis. "
        "Cada noticia ativa um subconjunto dessas posicoes antes de qualquer chamada residual a LLM.",
        fontsize=10.5,
        ha="left",
        va="center",
        color="#475467",
    )

    ax_boxes = fig.add_subplot(gs[1, :2])
    ax_boxes.set_xlim(0, 1)
    ax_boxes.set_ylim(0, 1)
    ax_boxes.axis("off")
    draw_summary_box(ax_boxes, 0.00, 0.54, 0.48, 0.36, "Temas canonicos", str(len(labels)), "#ecfdf3")
    draw_summary_box(ax_boxes, 0.52, 0.54, 0.48, 0.36, "Discriminadores WNN", str(bank.get("discriminator_count", 0)), "#eff8ff")
    draw_summary_box(ax_boxes, 0.00, 0.08, 0.48, 0.36, "Largura da memoria", f"{vocab_size} posicoes", "#fff7ed")
    draw_summary_box(ax_boxes, 0.52, 0.08, 0.48, 0.36, "Memorias por tema", str(len(memories)), "#f5f3ff")

    ax_bar = fig.add_subplot(gs[1, 2:])
    ax_bar.barh(top_themes["label"], top_themes["discriminator_count"], color="#1570ef")
    ax_bar.set_title("Temas com mais discriminadores", fontsize=12, loc="left")
    ax_bar.set_xlabel("Discriminadores")
    ax_bar.set_ylabel("")
    for idx, value in enumerate(top_themes["discriminator_count"]):
        ax_bar.text(value + 2, idx, str(int(value)), va="center", fontsize=9, color="#344054")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    ax_matrix = fig.add_subplot(gs[2, :3])
    positions = example.get("memory_positions", [])
    row = np.zeros(vocab_size, dtype=int)
    for pos in positions:
        if 0 <= int(pos) < vocab_size:
            row[int(pos)] = 1
    matrix = np.vstack([row])
    ax_matrix.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    ax_matrix.set_title(
        f"Exemplo de assinatura binaria: {example.get('label', '')} / {example.get('name', '')}",
        fontsize=12,
        loc="left",
    )
    ax_matrix.set_yticks([0])
    ax_matrix.set_yticklabels(["discriminador"])
    ax_matrix.set_xticks(np.linspace(0, max(vocab_size - 1, 1), 9, dtype=int))
    ax_matrix.set_xlabel("Posicoes na memoria binaria")
    ax_matrix.set_ylabel("")

    ax_text = fig.add_subplot(gs[2, 3])
    ax_text.axis("off")
    token_lines = "\n".join(f"- {token}" for token in example.get("tokens", [])[:8])
    pos_lines = ", ".join(str(pos) for pos in positions[:10])
    ax_text.text(
        0,
        1.0,
        "Marcadores do exemplo",
        fontsize=12,
        weight="bold",
        ha="left",
        va="top",
        color="#101828",
    )
    ax_text.text(
        0,
        0.82,
        token_lines or "- sem tokens",
        fontsize=10,
        ha="left",
        va="top",
        color="#344054",
    )
    ax_text.text(
        0,
        0.32,
        "Posicoes ativas",
        fontsize=12,
        weight="bold",
        ha="left",
        va="top",
        color="#101828",
    )
    ax_text.text(
        0,
        0.16,
        pos_lines or "sem posicoes",
        fontsize=10,
        ha="left",
        va="top",
        color="#344054",
        wrap=True,
    )

    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    build_figure()
