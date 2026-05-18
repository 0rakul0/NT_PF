from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.incremental.common import RunConfig
from scripts.incremental.organizar_arvore_temas import run as run_theme_tree_organizer
from scripts.incremental.reavaliar_quarentenas import run as run_rare_news_review
from scripts.incremental.resumo_custo_tokens import run as run_token_cost_summary
from scripts.incremental.run_all_incremental import run
from scripts.project_config import CONTENT_CSV, INDEX_CSV, NEWS_MARKDOWN_DIR, PROJECT_ROOT


def run_command(command: list[str], label: str, skip: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    if skip:
        return {"label": label, "skipped": True, "elapsed_seconds": 0.0, "returncode": 0}
    print(f"[rodar_sistema] iniciando: {label}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    elapsed = round(time.perf_counter() - started, 4)
    if completed.returncode != 0:
        raise RuntimeError(f"Etapa falhou ({label}) com codigo {completed.returncode}")
    return {"label": label, "skipped": False, "elapsed_seconds": elapsed, "returncode": completed.returncode}


def ensure_base_inputs() -> None:
    missing = [
        path
        for path in (NEWS_MARKDOWN_DIR, INDEX_CSV, CONTENT_CSV)
        if not path.exists()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Entradas obrigatorias ausentes:\n{formatted}")


def main() -> None:
    steps: list[dict[str, object]] = []
    steps.append(
        run_command(
            [sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "pf_operacoes_pipeline.py")],
            "sincronizacao da base",
        )
    )
    ensure_base_inputs()

    config = RunConfig(
        sample_fraction=0.15,
        batch_size=500,
        seed=42,
        regex_threshold=0.85,
        temporal_strata="year",
        model="llama3.2",
        base_url="http://localhost:11434",
        max_docs=None,
        reset=True,
        max_residual_llm_per_batch=None,
    )
    started = time.perf_counter()
    result = run(config)
    steps.append(
        {
            "label": "metodologia incremental",
            "skipped": False,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "returncode": 0,
        }
    )
    started = time.perf_counter()
    tree_result = run_theme_tree_organizer(RunConfig(reset=False))
    steps.append(
        {
            "label": "agente organizador da arvore de temas",
            "skipped": False,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "returncode": 0,
        }
    )
    started = time.perf_counter()
    rare_news_result = run_rare_news_review()
    steps.append(
        {
            "label": "consolidacao de noticias raras",
            "skipped": False,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "returncode": 0,
        }
    )
    started = time.perf_counter()
    token_cost_summary = run_token_cost_summary()
    steps.append(
        {
            "label": "resumo de custo por tokens",
            "skipped": False,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "returncode": 0,
        }
    )
    result["theme_tree_organizer"] = tree_result
    result["rare_news_review"] = rare_news_result
    result["token_cost_summary"] = token_cost_summary
    result["steps"] = steps

    output = PROJECT_ROOT / "data" / "analise_qualitativa" / "incremental" / "rodar_sistema_resultado.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
