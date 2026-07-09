from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.incremental.common import LOTS_DIR, RARE_NEWS_LABEL, REFINED_THEME_TREE_JSON, RUN_DIR, read_json
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


DISPLAY_LABEL_ALIASES = {
    "crime_armas_municoes": "armas_municoes",
    "crime_contrabando_descaminho": "contrabando_descaminho",
    "crime_corrupcao_desvio_recursos_publicos": "corrupcao_desvio_recursos_publicos",
    "crime_fraudes_auxilios_beneficios": "fraudes_auxilios_beneficios",
    "crime_lavagem_dinheiro": "lavagem_dinheiro",
    "crime_moeda_falsa": "moeda_falsa",
    "crime_radiodifusao_clandestina": "radiodifusao_clandestina",
    "crime_trabalho_escravo": "trabalho_escravo",
    "crime_trafico_drogas": "trafico_drogas",
}


def display_label(label: object) -> str:
    value = str(label or "").strip()
    return DISPLAY_LABEL_ALIASES.get(value, value)


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


def _safe_literal_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _refined_tree_mapping() -> tuple[dict[str, str], dict[str, str]]:
    if not REFINED_THEME_TREE_JSON.exists():
        return {}, {}
    payload = read_json(REFINED_THEME_TREE_JSON)
    label_map: dict[str, str] = {}
    status_map: dict[str, str] = {}
    for decision in payload.get("decisions", []) if isinstance(payload, dict) else []:
        candidate = str(decision.get("candidate_label", "")).strip()
        if not candidate:
            continue
        tree_decision = str(decision.get("decision", "")).strip()
        parent = str(decision.get("parent_theme", "")).strip()
        promoted = str(decision.get("promoted_theme", "")).strip()
        if tree_decision == "merge_into_existing" and parent:
            label_map[candidate] = parent
        elif tree_decision == "promote_to_canonical" and promoted:
            label_map[candidate] = promoted
        elif tree_decision == "keep_as_leaf" and parent:
            label_map[candidate] = parent
        elif tree_decision in {"discard", "quarantine"}:
            label_map[candidate] = "quarentena"
        status_map[candidate] = f"candidato_{tree_decision}" if tree_decision else "candidato_sem_decisao_arvore"
    return label_map, status_map


def _row_label_and_status(row: pd.Series, label_map: dict[str, str], status_map: dict[str, str]) -> tuple[str, str]:
    if bool(row.get("wnn_accepted", False)):
        inference = _safe_literal_dict(row.get("inference", {}))
        label = str(
            row.get("wnn_top_label", "")
            or inference.get("identidade_canonica", "")
            or inference.get("canonical_label", "")
        ).strip()
        return label or "quarentena", "classificado_wnn" if label else "quarentena_wnn_sem_label"

    decision = str(row.get("agent3_decision", "") or "").strip()
    label = str(row.get("agent3_canonical_label", "") or "").strip()
    if decision == "classificar":
        return label or "quarentena", "noticia_rara" if label == RARE_NEWS_LABEL else "classificado_agent3"
    if decision == "novo_tema_candidato":
        refined_label = label_map.get(label, label or "quarentena")
        return refined_label, status_map.get(label, "candidato_sem_decisao_arvore")
    if decision == "quarentena":
        return "quarentena", "quarentena_agent3"
    if decision == "error":
        return "quarentena", "erro_revisao_agent3"
    return "quarentena", "residual_nao_revisado"


def build_refined_classification_csv() -> Path:
    batch_files = sorted(LOTS_DIR.glob("lote_*_classificacoes.csv"))
    if not batch_files:
        raise FileNotFoundError(f"Nenhum lote de classificacao encontrado em: {LOTS_DIR}")

    df = pd.concat((pd.read_csv(path) for path in batch_files), ignore_index=True)
    label_map, status_map = _refined_tree_mapping()
    labels_status = df.apply(
        lambda row: pd.Series(
            _row_label_and_status(row, label_map, status_map),
            index=["label_final_arvore_refinada", "status_final_arvore_refinada"],
        ),
        axis=1,
    )
    df = pd.concat([df, labels_status], axis=1)
    INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")
    return INPUT_CSV


def run() -> dict[str, object]:
    build_refined_classification_csv()

    df = pd.read_csv(INPUT_CSV)
    df["label_pos_quarentena"] = df["label_final_arvore_refinada"]
    df["status_pos_quarentena"] = df["status_final_arvore_refinada"]
    df["motivo_reavaliacao_quarentena"] = ""

    quarantine_mask = df["label_final_arvore_refinada"].eq("quarentena")
    quarantine = df[quarantine_mask].copy()
    if quarantine.empty:
        suggestions = pd.DataFrame(columns=["sugestao_label", "sugestao_motivo"], index=quarantine.index)
    else:
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

    df["label_pos_quarentena_exibicao"] = df["label_pos_quarentena"].map(display_label)
    counts = df.groupby(["label_pos_quarentena_exibicao", "status_pos_quarentena"]).size().reset_index(name="noticias")
    pivot = counts.pivot_table(
        index="label_pos_quarentena_exibicao",
        columns="status_pos_quarentena",
        values="noticias",
        fill_value=0,
    )
    pivot.index.name = "label_pos_quarentena"
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot.to_csv(THEME_COUNTS_CSV, encoding="utf-8-sig")

    plot_df = pivot.head(10).drop(columns=["total"], errors="ignore")
    colors = {
        "classificado_agent3": "#F58518",
        "candidato_merge_into_existing": "#54A24B",
        "candidato_promote_to_canonical": "#72B7B2",
        "candidato_quarantine": "#B279A2",
        "quarentena_agent3": "#9D755D",
        "quarentena_reclassificada_pos_arvore": "#E45756",
        "noticia_rara": "#7F7F7F",
    }
    columns = [column for column in colors if column in plot_df.columns] + [column for column in plot_df.columns if column not in colors]
    columns = [column for column in columns if plot_df[column].sum() > 0]
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ax = plot_df[columns].plot(
        kind="barh",
        stacked=True,
        figsize=(11.5, 6.4),
        color=[colors.get(column, "#BAB0AC") for column in columns],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Notícias")
    ax.set_ylabel("Tema final")
    ax.set_title("Top 10 temas após classificação das notícias raras")
    ax.legend(title="Origem/decisão", loc="lower right", frameon=True)
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
