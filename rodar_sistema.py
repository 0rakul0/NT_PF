from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.incremental.common import RunConfig
from scripts.incremental.dashboard_comparacao import run as run_comparison_dashboard
from scripts.incremental.organizar_arvore_temas import run as run_theme_tree_organizer
from scripts.incremental.reavaliar_quarentenas import run as run_rare_news_review
from scripts.incremental.resumo_custo_tokens import run as run_token_cost_summary
from scripts.incremental.run_all_incremental import has_resume_checkpoint, resume as resume_incremental, run
from scripts.project_config import CONTENT_CSV, INDEX_CSV, NEWS_MARKDOWN_DIR, PROJECT_ROOT, get_llm_settings


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


def env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def env_optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "sim", "s"}


def build_run_config(reset: bool = True) -> RunConfig:
    llm_settings = get_llm_settings()
    local_fallback_models = tuple(
        model
        for model in ("llama3.1:latest", "gemma3n:e2b", "llama3:8b")
        if model != llm_settings.ollama.model_name
    )
    return RunConfig(
        sample_fraction=env_float("PF_SAMPLE_FRACTION", 0.10),
        batch_size=env_int("PF_BATCH_SIZE", 100),
        seed=env_int("PF_RANDOM_SEED", 42),
        wnn_enabled=env_bool("PF_WNN_ENABLED", True),
        wnn_confidence_threshold=env_float("PF_WNN_CONFIDENCE_THRESHOLD", 0.50),
        wnn_margin_threshold=env_float("PF_WNN_MARGIN_THRESHOLD", 0.12),
        wnn_min_active_discriminators=env_int("PF_WNN_MIN_ACTIVE_DISCRIMINATORS", 2),
        wnn_max_discriminators_per_theme=env_int("PF_WNN_MAX_DISCRIMINATORS_PER_THEME", 200),
        temporal_strata=os.getenv("PF_TEMPORAL_STRATA", "year").strip() or "year",
        model=llm_settings.ollama.model_name,
        base_url=llm_settings.ollama.base_url,
        max_docs=env_optional_int("PF_MAX_DOCS"),
        reset=reset,
        max_residual_llm_per_batch=env_optional_int("PF_MAX_RESIDUAL_LLM_PER_BATCH"),
        max_batches=env_optional_int("PF_MAX_BATCHES"),
        llm_timeout_seconds=env_int("PF_LLM_TIMEOUT_SECONDS", 180),
        ollama_num_ctx=env_int("PF_OLLAMA_NUM_CTX", 131072),
        ollama_num_predict=env_int("PF_OLLAMA_NUM_PREDICT", 1024),
        agent3_min_confidence=env_float("PF_AGENT3_MIN_CONFIDENCE", 0.55),
        resume_batches=env_bool("PF_RESUME_BATCHES", True),
        dynamic_class_thresholds=env_bool("PF_DYNAMIC_CLASS_THRESHOLDS", True),
        dynamic_threshold_min_references=env_int("PF_DYNAMIC_THRESHOLD_MIN_REFERENCES", 10),
        dynamic_threshold_step=env_float("PF_DYNAMIC_THRESHOLD_STEP", 0.02),
        # Each full execution starts from a clean incremental workspace. Prior
        # runs meant for comparison must be archived under comparacoes/ first.
        preserve_previous_run=env_bool("PF_PRESERVE_PREVIOUS_RUN", False),
        theme_tree_review_interval_batches=env_int("PF_THEME_TREE_REVIEW_INTERVAL_BATCHES", 0),
        dashboard_update_interval_batches=env_int("PF_DASHBOARD_UPDATE_INTERVAL_BATCHES", 0),
        wnn_compaction_interval_batches=env_int("PF_WNN_COMPACTION_INTERVAL_BATCHES", 1),
        local_fallback_models=local_fallback_models,
    )


def main() -> None:
    steps: list[dict[str, object]] = []
    requested_resume = env_bool("PF_RESUME_RUN", True)
    requested_reset = env_bool("PF_RESET_RUN", False)
    resume_run = not requested_reset and requested_resume and has_resume_checkpoint()

    if resume_run:
        # The source base is deliberately not synchronized here.  The existing
        # reserve CSV is the immutable checkpoint for this execution.
        steps.append(
            {
                "label": "sincronizacao da base",
                "skipped": True,
                "reason": "continuacao preserva a reserva original",
                "elapsed_seconds": 0.0,
                "returncode": 0,
            }
        )
        print("[rodar_sistema] continuando a execucao interrompida pelos checkpoints dos lotes")
    else:
        if requested_reset:
            print("[rodar_sistema] reset solicitado; iniciando nova rodada e reconstruindo a fundacao")
        elif requested_resume:
            print("[rodar_sistema] nenhum checkpoint completo encontrado; iniciando nova rodada")
        steps.append(
            run_command(
                [sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "pf_operacoes_pipeline.py")],
                "sincronizacao da base",
                skip=env_bool("PF_SKIP_SYNC", False),
            )
        )
    ensure_base_inputs()

    config = build_run_config(reset=not resume_run)
    started = time.perf_counter()
    result = resume_incremental(config) if resume_run else run(config)
    steps.append(
        {
            "label": "continuacao incremental" if resume_run else "metodologia incremental",
            "skipped": False,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "returncode": 0,
        }
    )
    started = time.perf_counter()
    tree_result = run_theme_tree_organizer(build_run_config(reset=False))
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
    result["comparison_dashboard"] = run_comparison_dashboard()
    result["steps"] = steps

    output = PROJECT_ROOT / "data" / "analise_qualitativa" / "incremental" / "rodar_sistema_resultado.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
