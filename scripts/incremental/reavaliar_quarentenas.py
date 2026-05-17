from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.incremental.common import RARE_NEWS_LABEL, RUN_DIR
from scripts.incremental.noticias_raras import rare_signature


INPUT_CSV = RUN_DIR / "classificacoes_incrementais_arvore_refinada.csv"
OUTPUT_CSV = RUN_DIR / "classificacoes_incrementais_pos_quarentena.csv"
QUARANTINE_REVIEW_CSV = RUN_DIR / "quarentenas_reavaliacao.csv"
SUMMARY_CSV = RUN_DIR / "status_pos_quarentena.csv"
THEME_COUNTS_CSV = RUN_DIR / "noticias_por_tema_pos_quarentena.csv"
FIGURE_PATH = RUN_DIR / "figures" / "noticias_por_tema_pos_quarentena.png"


RESCUE_RULES: list[tuple[str, list[str]]] = [
    (
        "crimes_patrimoniais",
        [
            "roubo",
            "assalto",
            "furto",
            "subtraid",
            "correios",
            "agencia postal",
            "bens subtraidos",
            "veiculos clonados",
        ],
    ),
    (
        "falsificacao_documental",
        [
            "falsificacao",
            "falsidade",
            "documento",
            "diploma",
            "vistos falsos",
            "falso pretor",
            "se passava por policial",
            "sinal publico",
            "simbolos nacionais",
            "exercicio irregular de profissao",
        ],
    ),
    (
        "ameacas_e_terrorismo",
        [
            "ameaca",
            "ameacar",
            "ameacas",
            "terrorismo",
            "atentado",
            "stalking",
            "perseguicao",
            "violencia contra universidade",
            "atos preparatorios",
        ],
    ),
    (
        "crimes_de_odio_e_extremismo",
        ["odio", "nazismo", "nazista", "apologia", "racismo", "discriminacao", "ideologias extremistas"],
    ),
    (
        "seguranca_privada_clandestina",
        ["seguranca privada", "vigilante", "atividade de seguranca privada"],
    ),
    (
        "crimes_ambientais",
        ["terra indigena", "indigena", "conflito indigena", "homicidio de indigena", "reforma agraria"],
    ),
    (
        "crimes_sistema_financeiro",
        ["apostas", "mercado de cartoes", "combustiveis", "setor seguros"],
    ),
    (
        "corrupcao_desvio_recursos_publicos",
        [
            "servidor publico",
            "cargo",
            "exploracao de prestigio",
            "administracao da justica",
            "obstrucao da justica",
            "servicos de utilidade publica",
        ],
    ),
]


def fold_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", text).strip()


def suggest_label(row: pd.Series) -> tuple[str, str]:
    text = fold_text(
        " ".join(
            [
                str(row.get("titulo", "")),
                str(row.get("agent3_rationale", "")),
                str(row.get("agent3_evidence_text", "")),
            ]
        )
    )
    for label, terms in RESCUE_RULES:
        matched = [term for term in terms if term in text]
        if matched:
            return label, "; ".join(matched[:4])
    return RARE_NEWS_LABEL, "sem evidencia suficiente para macrotema; noticia rara"


def run() -> dict[str, object]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df["label_pos_quarentena"] = df["label_final_arvore_refinada"]
    df["status_pos_quarentena"] = df["status_final_arvore_refinada"]
    df["motivo_reavaliacao_quarentena"] = ""

    quarantine_mask = df["label_final_arvore_refinada"].eq("quarentena")
    quarantine = df[quarantine_mask].copy()
    suggestions = quarantine.apply(lambda row: pd.Series(suggest_label(row), index=["sugestao_label", "sugestao_motivo"]), axis=1)
    quarantine = pd.concat([quarantine, suggestions], axis=1)
    rare_mask = quarantine["sugestao_label"].eq(RARE_NEWS_LABEL)
    if rare_mask.any():
        signatures = quarantine.loc[rare_mask].apply(
            lambda row: pd.Series(
                rare_signature(
                    str(row.get("titulo", "")),
                    str(row.get("agent3_evidence_text", "")),
                    str(row.get("agent3_rationale", "")),
                ),
                index=["rare_signature", "rare_signature_reason"],
            ),
            axis=1,
        )
        quarantine.loc[rare_mask, "rare_signature"] = signatures["rare_signature"]
        quarantine.loc[rare_mask, "rare_signature_reason"] = signatures["rare_signature_reason"]
    QUARANTINE_REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    quarantine.to_csv(QUARANTINE_REVIEW_CSV, index=False, encoding="utf-8-sig")

    for index, row in quarantine.iterrows():
        suggestion = str(row.get("sugestao_label", "quarentena"))
        df.loc[index, "label_pos_quarentena"] = suggestion
        df.loc[index, "status_pos_quarentena"] = (
            "noticia_rara" if suggestion == RARE_NEWS_LABEL else "quarentena_reclassificada_pos_arvore"
        )
        df.loc[index, "motivo_reavaliacao_quarentena"] = str(row.get("sugestao_motivo", ""))

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    status = df.groupby("status_pos_quarentena").size().reset_index(name="noticias")
    status.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    counts = df.groupby(["label_pos_quarentena", "status_pos_quarentena"]).size().reset_index(name="noticias")
    pivot = counts.pivot_table(index="label_pos_quarentena", columns="status_pos_quarentena", values="noticias", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot.to_csv(THEME_COUNTS_CSV, encoding="utf-8-sig")

    plot_df = pivot.drop(columns=["total"], errors="ignore")
    colors = {
        "classificado_regex": "#4C78A8",
        "classificado_agent3": "#F58518",
        "candidato_merge_into_existing": "#54A24B",
        "candidato_promote_to_canonical": "#72B7B2",
        "candidato_quarantine": "#B279A2",
        "quarentena_agent3": "#9D755D",
        "quarentena_reclassificada_pos_arvore": "#E45756",
        "noticia_rara": "#7F7F7F",
    }
    columns = [column for column in colors if column in plot_df.columns] + [column for column in plot_df.columns if column not in colors]
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ax = plot_df[columns].plot(
        kind="barh",
        stacked=True,
        figsize=(12, max(7, len(plot_df) * 0.34)),
        color=[colors.get(column) for column in columns],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Noticias")
    ax.set_ylabel("Tema final")
    ax.set_title("Noticias por tema apos classificacao das noticias raras")
    ax.legend(title="Origem/decisao", loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=180)
    plt.close()

    rescued = int((df["status_pos_quarentena"] == "quarentena_reclassificada_pos_arvore").sum())
    remaining = int((df["label_pos_quarentena"] == RARE_NEWS_LABEL).sum())
    return {
        "stage": "reavaliar_quarentenas",
        "quarantines_reviewed": int(len(quarantine)),
        "rescued": rescued,
        "rare_news": remaining,
        "review_csv": str(QUARANTINE_REVIEW_CSV),
        "output_csv": str(OUTPUT_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "theme_counts_csv": str(THEME_COUNTS_CSV),
        "figure": str(FIGURE_PATH),
    }


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
