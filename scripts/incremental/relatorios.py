from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from scripts.incremental.common import FIGURES_DIR, METRICS_CSV, PROJECT_ROOT, REFINED_THEME_TREE_JSON, RUN_DIR, RUN_RESULT_JSON, append_event, read_json, write_json


def plot_metrics(metrics: pd.DataFrame) -> list[object]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures = []
    fig, ax = plt.subplots(figsize=(11, 5))
    wnn_accepted = metrics["wnn_accepted"] if "wnn_accepted" in metrics else pd.Series([0] * len(metrics), index=metrics.index)
    post_wnn_residual = metrics["post_wnn_residual"] if "post_wnn_residual" in metrics else pd.Series([0] * len(metrics), index=metrics.index)
    composite = metrics.get("wnn_multi_discriminator_candidates", pd.Series([0] * len(metrics), index=metrics.index))
    ax.bar(metrics["iteration"], wnn_accepted, label="WNN")
    ax.bar(metrics["iteration"], post_wnn_residual, bottom=wnn_accepted, label="Residual pos-WNN")
    ax.bar(metrics["iteration"], composite, label="Candidatos compostos", alpha=0.7)
    ax.set_title("WNN, residuais e candidatos compostos por iteracao")
    ax.set_xlabel("Iteracao")
    ax.set_ylabel("Noticias")
    ax.legend()
    fig.tight_layout()
    output = FIGURES_DIR / "wnn_residual_candidatos_por_iteracao.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    figures.append(output)

    fig, ax = plt.subplots(figsize=(11, 5))
    wnn_rate = metrics.get("wnn_rate", metrics["wnn_accepted"] / metrics["docs"])
    llm_rate = metrics["llm_processed"] / metrics["docs"]
    candidate_rate = metrics.get("wnn_multi_discriminator_candidates", pd.Series([0] * len(metrics), index=metrics.index)) / metrics["docs"]
    ax.plot(metrics["iteration"], wnn_rate, marker="o", label="Taxa WNN")
    ax.plot(metrics["iteration"], llm_rate, marker="o", label="Taxa LLM residual")
    ax.plot(metrics["iteration"], candidate_rate, marker="o", label="Taxa candidatos compostos")
    ax.set_ylim(0, 1)
    ax.set_title("Taxas WNN, LLM e candidatos por iteracao")
    ax.set_xlabel("Iteracao")
    ax.set_ylabel("Proporcao")
    ax.legend()
    fig.tight_layout()
    output = FIGURES_DIR / "taxas_wnn_llm_candidatos_por_iteracao.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    figures.append(output)
    return figures


def build_report_lines(metrics: pd.DataFrame, foundation: dict[str, object], figures: list[object]) -> list[str]:
    lines = ["# Execucao da metodologia incremental", "", "## Fundacao", ""]
    lines.extend(
        [
            f"- Base: {foundation['base_docs']}",
            f"- Amostra inicial: {foundation['sample_docs']} ({foundation['sample_fraction']:.0%})",
            f"- Reserva incremental: {foundation['reserve_docs']}",
            f"- Clusters gerados: {foundation['clusters_total']}",
            f"- Clusters de ruido: {foundation['noise_clusters']}",
            f"- Temas canonicos aceitos: {foundation['themes_accepted']}",
            f"- Discriminadores WNN: {foundation.get('wnn_discriminators', 0)}",
            "",
        ]
    )
    if not metrics.empty:
        total_docs = int(metrics["docs"].sum())
        total_wnn = int(metrics.get("wnn_accepted", pd.Series(dtype=int)).sum())
        total_post_wnn_residual = int(metrics.get("post_wnn_residual", pd.Series(dtype=int)).sum())
        total_llm = int(metrics["llm_processed"].sum())
        total_learned = int(metrics["learned_rules"].sum())
        total_composite = int(metrics.get("wnn_multi_discriminator_candidates", pd.Series(dtype=int)).sum())
        total_new_theme_candidates = int(metrics.get("agent3_new_theme_candidates", pd.Series(dtype=int)).sum())
        total_rare_promoted = int(metrics.get("rare_promoted_candidates", pd.Series(dtype=int)).sum())
        total_agent3_quarantined = int(metrics.get("agent3_quarantined", pd.Series(dtype=int)).sum())
        total_rare_news = int(metrics.get("agent3_rare_news", pd.Series(dtype=int)).sum())
        total_agent3_errors = int(metrics.get("agent3_errors", pd.Series(dtype=int)).sum())
        lines.extend(
            [
                "## Lotes",
                "",
                f"- Iteracoes documentadas: {len(metrics)}",
                f"- Noticias nos lotes: {total_docs}",
                f"- Capturadas por WNN: {total_wnn}",
                f"- Residuais apos WNN: {total_post_wnn_residual}",
                f"- Processadas pela LLM residual: {total_llm}",
                f"- Marcadores aprendidos para WNN: {total_learned}",
                f"- Candidatos compostos WNN: {total_composite}",
                f"- Novos temas candidatos: {total_new_theme_candidates}",
                f"- Noticias raras promovidas a candidato: {total_rare_promoted}",
                f"- Noticias raras identificadas pelo Agente 3: {total_rare_news}",
                f"- Quarentenas tecnicas do Agente 3: {total_agent3_quarantined}",
                f"- Erros do Agente 3: {total_agent3_errors}",
                f"- Taxa WNN acumulada: {total_wnn / total_docs:.2%}",
                "",
                "## Interacoes",
                "",
            ]
        )
        for _, row in metrics.iterrows():
            lines.append(
                f"- {row['batch_id']}: docs={int(row['docs'])}, residual={int(row.get('post_wnn_residual', 0))}, "
                f"wnn={int(row.get('wnn_accepted', 0))}, llm={int(row['llm_processed'])}, aprendizados={int(row['learned_rules'])}, "
                f"candidatos_compostos={int(row.get('wnn_multi_discriminator_candidates', 0))}, taxa_wnn={row.get('wnn_rate', 0):.2%}"
            )
    if figures:
        lines.extend(["", "## Graficos", ""])
        for figure in figures:
            lines.append(f"- ![]({figure.relative_to(PROJECT_ROOT).as_posix()})")
    if REFINED_THEME_TREE_JSON.exists():
        refined = read_json(REFINED_THEME_TREE_JSON)
        lines.extend(
            [
                "",
                "## Agente Organizador da Arvore",
                "",
                f"- Candidatos avaliados: {len(refined.get('decisions', []))}",
                f"- Absorvidos por temas existentes: {refined.get('merged_into_existing_count', 0)}",
                f"- Promovidos a novos temas canonicos: {len(set(refined.get('promoted_canonical_themes', [])))}",
                f"- Mantidos como folhas: {refined.get('kept_as_leaf_count', 0)}",
                f"- Arvore refinada: `{REFINED_THEME_TREE_JSON.relative_to(PROJECT_ROOT).as_posix()}`",
            ]
        )
    return lines


def run(foundation: dict[str, object]) -> dict[str, object]:
    metrics = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame()
    figures = plot_metrics(metrics) if not metrics.empty else []
    report = RUN_DIR / "relatorio_execucao_metodologia.md"
    readme = RUN_DIR / "README_METRICAS.md"
    lines = build_report_lines(metrics, foundation, figures)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme_lines = list(lines)
    readme_lines[0] = "# Metricas da execucao incremental"
    readme_lines.extend(
        [
            "",
            "## Arquivos",
            "",
            "- `metrics_batches.csv`: metricas por iteracao.",
            "- `relatorio_execucao_metodologia.md`: relatorio narrativo da execucao.",
            "- `events.jsonl`: trilha completa de eventos.",
            "- `insumo_agente_organizador_arvore.json`: insumo completo do Agente Organizador da Arvore.",
            "- `arvore_temas_agent1_refinada.json`: reorganizacao global dos temas candidatos.",
            "- `figures/`: graficos WNN, LLM residual e candidatos compostos.",
        ]
    )
    readme.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    result = {
        "stage": "relatorios",
        "metrics": str(METRICS_CSV),
        "metrics_readme": str(readme),
        "figures": [str(path) for path in figures],
        "report": str(report),
    }
    write_json(RUN_RESULT_JSON, result)
    append_event(result)
    return result
