from __future__ import annotations

import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_ollama import ChatOllama
from scripts.incremental.preprocessamento_linguistico import preprocess_body_text

try:
    from scripts.pf_llm_metadata import build_llm_context, parse_news_markdown
    from scripts.project_config import ANALYSIS_DIR, NEWS_MARKDOWN_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from pf_llm_metadata import build_llm_context, parse_news_markdown
    from project_config import ANALYSIS_DIR, NEWS_MARKDOWN_DIR, PROJECT_ROOT


RUN_DIR = ANALYSIS_DIR / "incremental"
LOTS_DIR = ANALYSIS_DIR / "lotes"
FIGURES_DIR = RUN_DIR / "figures"
EVENTS_JSONL = RUN_DIR / "events.jsonl"
RUN_SNAPSHOTS_DIR = ANALYSIS_DIR / "run_snapshots"
DASHBOARD_HTML = ANALYSIS_DIR / "dashboard_comparacao.html"

ACTIVE_REGEX_BANK_PATH = ANALYSIS_DIR / "regex_classifier_rules.json"
AGENT2_REGEX_BANK_PATH = RUN_DIR / "regex_banco_agent2.json"
WNN_FEATURE_BANK_PATH = ANALYSIS_DIR / "wnn_feature_bank.json"

DOCS_JSONL = RUN_DIR / "documentos_base.jsonl"
SAMPLE_CSV = RUN_DIR / "amostra_inicial.csv"
RESERVE_CSV = RUN_DIR / "reserva_incremental.csv"
CLUSTER_ASSIGNMENTS_CSV = RUN_DIR / "cluster_assignments_amostra.csv"
CLUSTER_SUMMARY_CSV = RUN_DIR / "resumo_clusters_amostra.csv"
THEMES_JSON = RUN_DIR / "temas_canonicos_agent1.json"
INITIAL_REGEX_JSON = RUN_DIR / "regex_iniciais_agent2.json"
COSINE_PROFILE_PKL = RUN_DIR / "perfis_cosseno_temas.pkl"
COSINE_PROFILE_JSON = RUN_DIR / "perfis_cosseno_temas.json"
NEW_THEME_CANDIDATES_JSONL = RUN_DIR / "temas_candidatos_agent3.jsonl"
RARE_NEWS_JSONL = RUN_DIR / "noticias_raras_observacoes.jsonl"
REFINED_THEME_TREE_JSON = RUN_DIR / "arvore_temas_agent1_refinada.json"
THEME_REFINEMENT_INPUT_JSON = RUN_DIR / "insumo_agente_organizador_arvore.json"
METRICS_CSV = RUN_DIR / "metrics_batches.csv"
RUN_MANIFEST_JSON = RUN_DIR / "run_manifest.json"
RUN_RESULT_JSON = RUN_DIR / "run_result.json"
LINGUISTIC_PREPROCESSING_JSON = RUN_DIR / "preprocessamento_linguistico.json"

RARE_NEWS_LABEL = "noticias_raras"
RARE_NEWS_DESCRIPTION = "Noticias residuais raras, sem encaixe defensavel nos temas canonicos ou macrotemas existentes."
RARE_NEWS_PROMOTION_THRESHOLD = 2


def news_body_text(parsed: dict[str, Any]) -> str:
    body = str(parsed.get("corpo", "") or "").strip()
    if body:
        return body
    return build_llm_context(parsed)


@dataclass(frozen=True)
class RunConfig:
    sample_fraction: float = 0.10
    batch_size: int = 500
    seed: int = 42
    regex_threshold: float = 0.85
    regex_enabled: bool = False
    wnn_enabled: bool = True
    wnn_confidence_threshold: float = 0.50
    wnn_margin_threshold: float = 0.12
    wnn_min_active_discriminators: int = 2
    wnn_max_discriminators_per_theme: int = 35
    temporal_strata: str = "year"
    model: str = "llama3.1:latest"
    base_url: str = "http://localhost:11434"
    max_docs: int | None = None
    reset: bool = True
    max_residual_llm_per_batch: int | None = None
    max_batches: int | None = None
    llm_timeout_seconds: int = 180
    ollama_num_ctx: int = 131072
    ollama_num_predict: int = 1024
    agent3_min_confidence: float = 0.55
    initial_regex_target_per_theme: int | None = 0
    resume_batches: bool = True
    preserve_previous_run: bool = True
    theme_tree_review_interval_batches: int = 1
    local_fallback_models: tuple[str, ...] = ("llama3.1:latest", "gemma3n:e2b", "llama3:8b")


def workspace_path(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if not str(resolved).lower().startswith(str(root).lower()):
        raise ValueError(f"Caminho fora do workspace: {resolved}")
    return resolved


def reset_outputs() -> list[str]:
    deleted: list[str] = []
    for path in (RUN_DIR, LOTS_DIR):
        if path.exists():
            resolved = workspace_path(path)
            shutil.rmtree(resolved)
            deleted.append(str(resolved))
    for path in ANALYSIS_DIR.glob("*"):
        if path.is_file():
            resolved = workspace_path(path)
            resolved.unlink()
            deleted.append(str(resolved))
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_REGEX_BANK_PATH.write_text("[]\n", encoding="utf-8")
    WNN_FEATURE_BANK_PATH.write_text("{}\n", encoding="utf-8")
    return deleted


def snapshot_existing_run(label: str = "") -> dict[str, object]:
    """Preserve current run artifacts before a reset so experiments remain comparable."""

    candidates = [
        RUN_DIR,
        LOTS_DIR,
        ACTIVE_REGEX_BANK_PATH,
        WNN_FEATURE_BANK_PATH,
        DASHBOARD_HTML,
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {"created": False, "reason": "no_previous_artifacts"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip()) if label.strip() else "previous"
    snapshot_dir = RUN_SNAPSHOTS_DIR / f"{timestamp}_{suffix}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    copied: list[str] = []
    for path in existing:
        destination = snapshot_dir / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
        copied.append(str(destination))

    manifest = {
        "created_at": timestamp,
        "label": label or "previous",
        "snapshot_dir": str(snapshot_dir),
        "copied": copied,
    }
    write_json(snapshot_dir / "snapshot_manifest.json", manifest)
    return {"created": True, "snapshot_dir": str(snapshot_dir), "copied": copied}


def append_event(event: dict[str, Any]) -> None:
    EVENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_docs(max_docs: int | None = None) -> list[dict[str, Any]]:
    files = sorted(NEWS_MARKDOWN_DIR.glob("*.md"))
    if max_docs is not None:
        files = files[:max_docs]
    docs: list[dict[str, Any]] = []
    preprocessing_started = time.perf_counter()
    preprocessing_totals = {
        "original_tokens": 0,
        "kept_tokens": 0,
        "elapsed_seconds": 0.0,
        "backends": {},
    }
    for path in files:
        parsed = parse_news_markdown(path.read_text(encoding="utf-8"))
        body_text = news_body_text(parsed)
        linguistic = preprocess_body_text(body_text)
        preprocessing_totals["original_tokens"] += linguistic.original_tokens
        preprocessing_totals["kept_tokens"] += linguistic.kept_tokens
        preprocessing_totals["elapsed_seconds"] += linguistic.elapsed_seconds
        backends = preprocessing_totals["backends"]
        backends[linguistic.backend] = int(backends.get(linguistic.backend, 0)) + 1
        docs.append(
            {
                "arquivo": path.name,
                "markdown_path": str(path),
                "titulo": str(parsed.get("titulo", "")),
                "tags": parsed.get("tags", []),
                "parsed": parsed,
                "body_text": body_text,
                "semantic_features": linguistic.semantic_features,
                "semantic_tokens": linguistic.tokens,
                "semantic_phrases": linguistic.phrases,
                "linguistic_metrics": linguistic.metrics(),
                "context": build_llm_context(parsed),
            }
        )
    if not docs:
        raise FileNotFoundError(f"Nenhum markdown encontrado em {NEWS_MARKDOWN_DIR}")
    original_tokens = int(preprocessing_totals["original_tokens"])
    kept_tokens = int(preprocessing_totals["kept_tokens"])
    write_json(
        LINGUISTIC_PREPROCESSING_JSON,
        {
            "stage": "preprocessamento_linguistico",
            "documents": len(docs),
            "original_tokens": original_tokens,
            "kept_tokens": kept_tokens,
            "removed_tokens": max(0, original_tokens - kept_tokens),
            "reduction_ratio": round(1.0 - (kept_tokens / original_tokens), 6) if original_tokens else 0.0,
            "linguistic_elapsed_seconds": round(float(preprocessing_totals["elapsed_seconds"]), 4),
            "total_elapsed_seconds": round(time.perf_counter() - preprocessing_started, 4),
            "backends": preprocessing_totals["backends"],
            "representation": "substantivos_verbos_adjetivos_e_expressoes_de_dominio",
        },
    )
    return docs


def parse_br_date(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%Y %Hh%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def temporal_bucket(doc: dict[str, Any], granularity: str) -> str:
    parsed = doc.get("parsed", {})
    published = parsed.get("data_publicacao") if isinstance(parsed, dict) else ""
    parsed_date = parse_br_date(published)
    if parsed_date is None:
        return "sem_data"
    if granularity == "month":
        return parsed_date.strftime("%Y-%m")
    return parsed_date.strftime("%Y")


def temporal_stratified_sample_names(
    docs: list[dict[str, Any]],
    fraction: float,
    seed: int,
    granularity: str = "year",
) -> set[str]:
    target_size = max(1, round(len(docs) * fraction))
    rng = random.Random(seed)
    strata: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        strata.setdefault(temporal_bucket(doc, granularity), []).append(doc)

    selected: set[str] = set()
    ordered_buckets = sorted(strata)

    for bucket in ordered_buckets:
        bucket_docs = list(strata[bucket])
        rng.shuffle(bucket_docs)
        if bucket_docs and len(selected) < target_size:
            selected.add(bucket_docs[0]["arquivo"])

    remaining_slots = target_size - len(selected)
    if remaining_slots <= 0:
        return set(sorted(selected)[:target_size])

    weighted_candidates: list[dict[str, Any]] = []
    for bucket in ordered_buckets:
        bucket_docs = [doc for doc in strata[bucket] if doc["arquivo"] not in selected]
        if not bucket_docs:
            continue
        proportional = round((len(strata[bucket]) / len(docs)) * target_size)
        already = sum(1 for name in selected if any(doc["arquivo"] == name for doc in strata[bucket]))
        quota = max(0, proportional - already)
        rng.shuffle(bucket_docs)
        for doc in bucket_docs[:quota]:
            if len(selected) >= target_size:
                break
            selected.add(doc["arquivo"])
        weighted_candidates.extend(doc for doc in bucket_docs if doc["arquivo"] not in selected)

    rng.shuffle(weighted_candidates)
    for doc in weighted_candidates:
        if len(selected) >= target_size:
            break
        selected.add(doc["arquivo"])
    return selected


def split_docs(
    docs: list[dict[str, Any]],
    fraction: float,
    seed: int,
    temporal_granularity: str = "year",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_names = temporal_stratified_sample_names(docs, fraction, seed, temporal_granularity)
    sample = [item for item in docs if item["arquivo"] in sample_names]
    reserve = [item for item in docs if item["arquivo"] not in sample_names]
    return sample, reserve


def docs_by_manifest(manifest_csv: Path) -> list[dict[str, Any]]:
    names = set(pd.read_csv(manifest_csv)["arquivo"].astype(str))
    return [doc for doc in read_jsonl(DOCS_JSONL) if doc["arquivo"] in names]


def llm(model: str, base_url: str, timeout_seconds: int = 180, json_schema: dict[str, Any] | None = None) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
        format=json_schema or "json",
        num_ctx=int(os.getenv("PF_OLLAMA_NUM_CTX", "131072")),
        num_predict=1024,
        keep_alive="10m",
        sync_client_kwargs={"timeout": timeout_seconds},
    )


def slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"\W+", "_", value.lower())).strip("_")
