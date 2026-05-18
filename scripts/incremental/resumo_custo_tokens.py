from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.incremental.common import EVENTS_JSONL, METRICS_CSV, RUN_DIR, write_json


OUTPUT_JSON = RUN_DIR / "resumo_custo_tokens.json"
OUTPUT_MD = RUN_DIR / "resumo_custo_tokens.md"


def _read_events() -> list[dict[str, Any]]:
    if not EVENTS_JSONL.exists():
        return []
    rows = []
    with EVENTS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize() -> dict[str, Any]:
    metrics = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame()
    events = [event for event in _read_events() if event.get("stage") == "llm_residual"]

    if not metrics.empty and "tokens_total" in metrics.columns:
        prompt_tokens = int(metrics.get("prompt_tokens_total", pd.Series(dtype=int)).fillna(0).sum())
        completion_tokens = int(metrics.get("completion_tokens_total", pd.Series(dtype=int)).fillna(0).sum())
        total_tokens = int(metrics["tokens_total"].fillna(0).sum())
        llm_processed = int(metrics.get("llm_processed", pd.Series(dtype=int)).fillna(0).sum())
    else:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        llm_processed = 0

    if total_tokens == 0 and events:
        prompt_tokens = sum(int(event.get("tokens", {}).get("prompt_tokens", 0) or 0) for event in events)
        completion_tokens = sum(int(event.get("tokens", {}).get("completion_tokens", 0) or 0) for event in events)
        total_tokens = sum(int(event.get("tokens", {}).get("total_tokens", 0) or 0) for event in events)
        llm_processed = sum(1 for event in events if event.get("status") != "error")

    return {
        "llm_residual_calls": llm_processed,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "tokens_total": total_tokens,
        "avg_tokens_per_llm": round(total_tokens / llm_processed, 4) if llm_processed else 0,
        "source_metrics_csv": str(METRICS_CSV),
        "source_events_jsonl": str(EVENTS_JSONL),
    }


def write_markdown(summary: dict[str, Any], path: Path = OUTPUT_MD) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resumo de custo por tokens",
        "",
        "| Indicador | Valor |",
        "|---|---:|",
        f"| Chamadas LLM residuais | {summary['llm_residual_calls']} |",
        f"| Prompt tokens | {summary['prompt_tokens_total']} |",
        f"| Completion tokens | {summary['completion_tokens_total']} |",
        f"| Tokens totais | {summary['tokens_total']} |",
        f"| Media de tokens por residual | {summary['avg_tokens_per_llm']} |",
        "",
        "O custo operacional variavel deve ser estimado sobre os tokens residuais, pois noticias classificadas por regex nao consomem inferencia LLM.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run() -> dict[str, Any]:
    summary = summarize()
    write_json(OUTPUT_JSON, summary)
    write_markdown(summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
