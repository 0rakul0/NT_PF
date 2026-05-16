from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from scripts.incremental.common import FIGURES_DIR, METRICS_CSV, PROJECT_ROOT, RUN_DIR, RUN_RESULT_JSON, append_event, read_json, write_json


def plot_metrics(metrics: pd.DataFrame) -> list[object]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures = []
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(metrics["iteration"], metrics["regex_accepted"], label="Regex")
    ax.bar(metrics["iteration"], metrics["regex_residual"], bottom=metrics["regex_accepted"], label="Residual")
    ax.set_title("Regex e residuos por iteracao")
    ax.set_xlabel("Iteracao")
    ax.set_ylabel("Noticias")
    ax.legend()
    fig.tight_layout()
    output = FIGURES_DIR / "regex_vs_residual_por_iteracao.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    figures.append(output)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(metrics["iteration"], metrics["regex_rate"], marker="o", label="Taxa regex")
    ax.plot(metrics["iteration"], metrics["cumulative_regex_rate"], marker="o", label="Taxa acumulada")
    ax.set_ylim(0, 1)
    ax.set_title("Cobertura regex por iteracao")
    ax.set_xlabel("Iteracao")
    ax.set_ylabel("Proporcao")
    ax.legend()
    fig.tight_layout()
    output = FIGURES_DIR / "taxa_regex_por_iteracao.png"
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
            f"- Regex iniciais aceitas: {foundation['initial_regex_accepted']}",
            "",
        ]
    )
    if not metrics.empty:
        total_docs = int(metrics["docs"].sum())
        total_regex = int(metrics["regex_accepted"].sum())
        total_residual = int(metrics["regex_residual"].sum())
        total_llm = int(metrics["llm_processed"].sum())
        total_learned = int(metrics["learned_rules"].sum())
        lines.extend(
            [
                "## Lotes",
                "",
                f"- Iteracoes documentadas: {len(metrics)}",
                f"- Noticias nos lotes: {total_docs}",
                f"- Capturadas por regex: {total_regex}",
                f"- Residuais: {total_residual}",
                f"- Processadas pela LLM residual: {total_llm}",
                f"- Regras aprendidas pelo Agente 3: {total_learned}",
                f"- Taxa regex acumulada: {total_regex / total_docs:.2%}",
                "",
                "## Interacoes",
                "",
            ]
        )
        for _, row in metrics.iterrows():
            lines.append(
                f"- {row['batch_id']}: docs={int(row['docs'])}, regex={int(row['regex_accepted'])}, residual={int(row['regex_residual'])}, "
                f"llm={int(row['llm_processed'])}, aprendizados={int(row['learned_rules'])}, taxa_regex={row['regex_rate']:.2%}"
            )
    if figures:
        lines.extend(["", "## Graficos", ""])
        for figure in figures:
            lines.append(f"- ![]({figure.relative_to(PROJECT_ROOT).as_posix()})")
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
            "- `figures/`: graficos da cobertura regex e residuos.",
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
