from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data" / "analise_qualitativa" / "incremental" / "resumo_clusters_amostra.csv"
OUT = ROOT / "artigo" / "media" / "figura-2-clusters-fundacao.png"


def main() -> None:
    df = pd.read_csv(SUMMARY)
    df = df.sort_values("size", ascending=False).head(10).copy()
    df["grupo"] = df["top_terms"].fillna("").str.split(" | ", regex=False).str[:2].str.join(", ")
    df.loc[df["grupo"].str.len() == 0, "grupo"] = "cluster " + df["cluster_id"].astype(str)
    df["label"] = df["grupo"] + " (" + df["cluster_id"].astype(str) + ")"
    ordered = df.sort_values("size")

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    ax.barh(ordered["label"], ordered["size"], color="#4C78A8")
    ax.set_title("Principais grupos consolidados da amostra inicial")
    ax.set_xlabel("Notícias na amostra")
    ax.set_ylabel("")
    for i, value in enumerate(ordered["size"]):
        ax.text(value + 3, i, str(int(value)), va="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
