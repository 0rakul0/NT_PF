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
OUT_OVERVIEW = MEDIA / "dashboard_wnn_memoria_binaria_overview.png"
OUT_CRIME = MEDIA / "dashboard_wnn_memoria_binaria_crime.png"
OUT_MODUS = MEDIA / "dashboard_wnn_memoria_binaria_modus.png"


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


def pick_example(bank: dict, kind: str) -> dict:
    discriminators = [
        item
        for item in bank.get("discriminators", [])
        if str(item.get("kind", "crime") or "crime") == kind
    ]
    if not discriminators:
        raise ValueError(f"wnn_feature_bank.json sem discriminadores do tipo {kind}")
    return max(discriminators, key=lambda item: len(item.get("memory_positions", [])))


def invert_vocab(vocab: dict) -> dict[int, str]:
    token_to_position = vocab.get("token_to_position", {})
    inverted: dict[int, str] = {}
    for token, pos in token_to_position.items():
        try:
            inverted[int(pos)] = str(token)
        except (TypeError, ValueError):
            continue
    return inverted


def draw_summary_box(ax, x: float, y: float, w: float, h: float, title: str, value: str, face: str) -> None:
    rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor="#d0d7de", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x + 0.04 * w, y + 0.63 * h, title, fontsize=10, color="#475467", va="center", ha="left")
    ax.text(x + 0.04 * w, y + 0.28 * h, value, fontsize=18, weight="bold", color="#101828", va="center", ha="left")


def _compact_bits(row: np.ndarray, width: int = 36) -> str:
    bits = "".join(str(int(bit)) for bit in row)
    if len(bits) <= width:
        return bits or "0"
    half = max(8, width // 2 - 2)
    return f"{bits[:half]}...{bits[-half:]}"


def _active_positions_text(row: np.ndarray, limit: int = 6) -> str:
    positions = [str(int(pos)) for pos in np.where(row == 1)[0][:limit]]
    if not positions:
        return "nenhuma"
    suffix = " ..." if int(row.sum()) > limit else ""
    return ", ".join(positions) + suffix


def _plot_axis_row(ax, row: np.ndarray, color: str, label: str) -> None:
    positions = np.where(row == 1)[0]
    ax.set_facecolor("#f8fbff")
    ax.set_xlim(0, len(row) - 1 if len(row) else 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Posicoes na memoria WNN")
    ax.set_title(label, fontsize=12, loc="left")
    if len(positions):
        ax.vlines(positions, 0.18, 0.82, colors=color, linewidth=2.2)
        ax.scatter(positions, np.full(len(positions), 0.5), color=color, s=28, zorder=3)
    ax.hlines(0.5, 0, len(row) - 1 if len(row) else 1, colors="#d6e1f0", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def save_overview_figure(
    labels: list[str],
    memories: dict,
    vocab_size: int,
    bank: dict,
    crime_example: dict,
    modus_example: dict,
    crime_row: np.ndarray,
    modus_row: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(13.8, 6.6), dpi=180, facecolor="white")
    gs = fig.add_gridspec(4, 12, height_ratios=[0.9, 0.8, 1.2, 1.5], hspace=0.45, wspace=0.28)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.82,
        "Memoria WNN em duas secoes: crime e modus operandi",
        fontsize=20,
        weight="bold",
        ha="left",
        va="center",
        color="#101828",
    )
    ax_title.text(
        0.0,
        0.38,
        "A imagem binaria pode ser lida em dois blocos: primeiro o crime canonico e depois o modus operandi. "
        "Cada bloco mostra apenas posicoes 0/1 ligadas ao seu respectivo eixo.",
        fontsize=10.5,
        ha="left",
        va="center",
        color="#475467",
    )

    ax_boxes = fig.add_subplot(gs[1, :])
    ax_boxes.set_xlim(0, 1)
    ax_boxes.set_ylim(0, 1)
    ax_boxes.axis("off")
    draw_summary_box(ax_boxes, 0.00, 0.18, 0.23, 0.52, "Temas canonicos", str(len(labels)), "#ecfdf3")
    draw_summary_box(ax_boxes, 0.26, 0.18, 0.23, 0.52, "Discriminadores WNN", str(bank.get("discriminator_count", 0)), "#eff8ff")
    draw_summary_box(ax_boxes, 0.52, 0.18, 0.23, 0.52, "Largura da memoria", f"{vocab_size} posicoes", "#fff7ed")
    draw_summary_box(ax_boxes, 0.78, 0.18, 0.22, 0.52, "Memorias por tema", str(len(memories)), "#f5f3ff")

    crime_bits = _compact_bits(crime_row, width=26)
    modus_bits = _compact_bits(modus_row, width=26)
    crime_active_text = _active_positions_text(crime_row)
    modus_active_text = _active_positions_text(modus_row)

    ax_crime = fig.add_subplot(gs[2, :6])
    ax_crime.set_xlim(0, 1)
    ax_crime.set_ylim(0, 1)
    ax_crime.axis("off")
    crime_box = plt.Rectangle((0.00, 0.02), 0.98, 0.94, facecolor="#eef4ff", edgecolor="#bfd4ff", linewidth=1.1)
    ax_crime.add_patch(crime_box)
    ax_crime.text(0.04, 0.90, "Sessao 1 - Crime", fontsize=13, weight="bold", color="#101828", ha="left", va="top")
    ax_crime.text(0.04, 0.74, "Bloco principal da decisao autonoma.\nA WNN procura aqui o crime canonico mais estavel.", fontsize=10, color="#475467", ha="left", va="top")
    ax_crime.text(0.04, 0.48, "Bloco 0/1 (recorte)", fontsize=9.6, color="#475467", ha="left", va="top")
    ax_crime.text(0.04, 0.33, crime_bits, fontsize=13, family="monospace", color="#1570ef", ha="left", va="top")
    ax_crime.text(
        0.04,
        0.13,
        f"label: {crime_example.get('label', '')}\nposicoes ativas: {crime_active_text}",
        fontsize=9.6,
        color="#344054",
        ha="left",
        va="top",
    )

    ax_modus = fig.add_subplot(gs[2, 6:])
    ax_modus.set_xlim(0, 1)
    ax_modus.set_ylim(0, 1)
    ax_modus.axis("off")
    modus_box = plt.Rectangle((0.00, 0.02), 0.98, 0.94, facecolor="#eefbf3", edgecolor="#bde5ca", linewidth=1.1)
    ax_modus.add_patch(modus_box)
    ax_modus.text(0.04, 0.90, "Sessao 2 - Modus operandi", fontsize=13, weight="bold", color="#101828", ha="left", va="top")
    ax_modus.text(0.04, 0.74, "Bloco complementar da classificacao.\nAqui a WNN descreve a forma de execucao sem derrubar o crime.", fontsize=10, color="#475467", ha="left", va="top")
    ax_modus.text(0.04, 0.48, "Bloco 0/1 (recorte)", fontsize=9.6, color="#475467", ha="left", va="top")
    ax_modus.text(0.04, 0.33, modus_bits, fontsize=13, family="monospace", color="#16a34a", ha="left", va="top")
    ax_modus.text(
        0.04,
        0.13,
        f"label: {modus_example.get('label', '')}\nposicoes ativas: {modus_active_text}",
        fontsize=9.6,
        color="#344054",
        ha="left",
        va="top",
    )

    ax_matrix = fig.add_subplot(gs[3, :9])
    matrix = np.vstack([crime_row, modus_row])
    ax_matrix.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    ax_matrix.set_title("Visualizacao central da memoria", fontsize=12, loc="left")
    ax_matrix.set_yticks([0, 1])
    ax_matrix.set_yticklabels(["crime", "modus"])
    ax_matrix.set_xticks(np.linspace(0, max(vocab_size - 1, 1), 9, dtype=int))
    ax_matrix.set_xlabel("Posicoes na memoria WNN")
    ax_matrix.set_ylabel("")
    ax_matrix.text(
        0.0,
        -0.18,
        "Cada linha mostra apenas as posicoes ativadas em sua propria sessao.",
        transform=ax_matrix.transAxes,
        fontsize=9.2,
        color="#344054",
        ha="left",
        va="top",
    )

    fig.savefig(OUT_OVERVIEW, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_axis_detail_figure(kind: str, label: str, example: dict, row: np.ndarray, color: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(12.5, 4.8), dpi=180, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.2], width_ratios=[1.1, 1.4, 1.0], wspace=0.28, hspace=0.35)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.78, label, fontsize=20, weight="bold", color="#101828", ha="left", va="center")
    ax_title.text(
        0.0,
        0.30,
        "Leitura detalhada de uma unica sessao da memoria WNN. "
        "O objetivo aqui e isolar o eixo para facilitar a interpretacao.",
        fontsize=10.5,
        color="#475467",
        ha="left",
        va="center",
    )

    ax_bits = fig.add_subplot(gs[1, 0])
    ax_bits.set_xlim(0, 1)
    ax_bits.set_ylim(0, 1)
    ax_bits.axis("off")
    box = plt.Rectangle((0.00, 0.04), 0.98, 0.90, facecolor="#f8fbff", edgecolor="#d0d7de", linewidth=1.0)
    ax_bits.add_patch(box)
    preview_width = min(len(row), 64)
    bitstring = "".join(str(int(bit)) for bit in row[:preview_width]) or "0"
    ax_bits.text(0.05, 0.86, "Primeiras posicoes 0/1", fontsize=11, weight="bold", color="#101828", ha="left", va="top")
    ax_bits.text(0.05, 0.62, bitstring, fontsize=13, family="monospace", color=color, ha="left", va="top")
    ax_bits.text(0.05, 0.26, f"label: {example.get('label', '')}\nbits ativos: {int(row.sum())}", fontsize=10, color="#344054", ha="left", va="top")

    ax_plot = fig.add_subplot(gs[1, 1])
    _plot_axis_row(ax_plot, row, color, "Distribuicao dos bits ativos")

    ax_tokens = fig.add_subplot(gs[1, 2])
    ax_tokens.set_xlim(0, 1)
    ax_tokens.set_ylim(0, 1)
    ax_tokens.axis("off")
    token_lines = "\n".join(f"- {token}" for token in example.get("tokens", [])[:6])
    positions = np.where(row == 1)[0][:10]
    pos_line = ", ".join(str(int(pos)) for pos in positions) or "nenhuma"
    ax_tokens.text(0.0, 0.92, "Marcadores", fontsize=11, weight="bold", color="#101828", ha="left", va="top")
    ax_tokens.text(0.0, 0.76, token_lines or "- sem tokens", fontsize=10, color="#344054", ha="left", va="top")
    ax_tokens.text(0.0, 0.28, "Posicoes ativas", fontsize=11, weight="bold", color="#101828", ha="left", va="top")
    ax_tokens.text(0.0, 0.12, pos_line, fontsize=10, color="#344054", ha="left", va="top", wrap=True)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figure() -> None:
    bank = load_bank()
    vocab = bank.get("memory_vocab", {})
    vocab_size = int(vocab.get("size", 0))
    position_to_token = invert_vocab(vocab)
    labels = bank.get("labels", [])
    memories = bank.get("memories", {})
    crime_example = pick_example(bank, "crime")
    modus_example = pick_example(bank, "modus")
    top_themes = top_theme_counts(bank)
    crime_positions = [int(pos) for pos in crime_example.get("memory_positions", [])]
    modus_positions = [int(pos) for pos in modus_example.get("memory_positions", [])]
    crime_row = np.zeros(vocab_size, dtype=int)
    modus_row = np.zeros(vocab_size, dtype=int)
    for pos in crime_positions:
        if 0 <= pos < vocab_size:
            crime_row[pos] = 1
    for pos in modus_positions:
        if 0 <= pos < vocab_size:
            modus_row[pos] = 1

    save_overview_figure(labels, memories, vocab_size, bank, crime_example, modus_example, crime_row, modus_row)
    save_axis_detail_figure(
        "crime",
        "Sessao 1 - Crime",
        crime_example,
        crime_row,
        "#1570ef",
        OUT_CRIME,
    )
    save_axis_detail_figure(
        "modus",
        "Sessao 2 - Modus operandi",
        modus_example,
        modus_row,
        "#16a34a",
        OUT_MODUS,
    )
    print(OUT_PATH)


if __name__ == "__main__":
    build_figure()
