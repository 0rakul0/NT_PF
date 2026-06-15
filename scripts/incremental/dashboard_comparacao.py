from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.incremental.common import (
    ANALYSIS_DIR,
    DASHBOARD_HTML,
    LOTS_DIR,
    METRICS_CSV,
    RUN_DIR,
    RUN_MANIFEST_JSON,
    RUN_SNAPSHOTS_DIR,
    WNN_FEATURE_BANK_PATH,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_metrics(base: Path) -> pd.DataFrame:
    path = base / "incremental" / "metrics_batches.csv"
    if not path.exists() and base == ANALYSIS_DIR:
        path = METRICS_CSV
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_lot_classifications(base: Path) -> pd.DataFrame:
    lots = base / "lotes"
    if not lots.exists() and base == ANALYSIS_DIR:
        lots = LOTS_DIR
    files = sorted(lots.glob("lote_*_classificacoes.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def _latest_snapshot() -> Path | None:
    if not RUN_SNAPSHOTS_DIR.exists():
        return None
    snapshots = [path for path in RUN_SNAPSHOTS_DIR.iterdir() if path.is_dir()]
    return sorted(snapshots, key=lambda path: path.name)[-1] if snapshots else None


def _summarize_run(name: str, base: Path) -> dict[str, Any]:
    incremental = base / "incremental"
    manifest = _read_json(incremental / "run_manifest.json")
    sampling = _read_json(incremental / "amostragem_result.json")
    agent2 = _read_json(incremental / "agente2_result.json")
    metrics = _read_metrics(base)
    wnn_bank = base / "wnn_feature_bank.json"
    if base == ANALYSIS_DIR:
        wnn_bank = WNN_FEATURE_BANK_PATH

    summary: dict[str, Any] = {
        "execucao": name,
        "base_docs": sampling.get("base_docs", manifest.get("base_docs", "")),
        "sample_docs": sampling.get("sample_docs", manifest.get("sample_docs", "")),
        "sample_fraction": sampling.get("sample_fraction", manifest.get("sample_fraction", "")),
        "reserve_docs": sampling.get("reserve_docs", manifest.get("reserve_docs", "")),
        "themes_accepted": manifest.get("themes_accepted", ""),
        "wnn_discriminators": agent2.get("wnn_discriminators", manifest.get("wnn_discriminators", "")),
        "wnn_bank_bytes": wnn_bank.stat().st_size if wnn_bank.exists() else 0,
        "batches_done": len(metrics),
    }
    if not metrics.empty:
        total_docs = int(metrics["docs"].sum())
        total_wnn = int(metrics.get("wnn_accepted", pd.Series(dtype=int)).sum())
        total_llm = int(metrics["llm_processed"].sum())
        total_composite_candidates = int(metrics.get("wnn_multi_discriminator_candidates", pd.Series(dtype=int)).sum())
        total_rare_promoted = int(metrics.get("rare_promoted_candidates", pd.Series(dtype=int)).sum())
        summary.update(
            {
                "docs_processados": total_docs,
                "wnn_accepted": total_wnn,
                "llm_processed": total_llm,
                "wnn_multi_discriminator_candidates": total_composite_candidates,
                "rare_promoted_candidates": total_rare_promoted,
                "taxa_wnn": f"{(total_wnn / total_docs):.2%}" if total_docs else "",
            }
        )
    return summary


def _classification_label(row: pd.Series) -> str:
    for column in ("agent3_canonical_label", "wnn_top_label"):
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    raw = row.get("inference", "")
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw.replace("'", '"'))
            if isinstance(payload, dict):
                crimes = payload.get("crimes_mais_presentes", [])
                if isinstance(crimes, list) and crimes:
                    return str(crimes[0])
                return str(payload.get("identidade_canonica", ""))
        except json.JSONDecodeError:
            return raw[:80]
    return ""


def _compare_classifications(previous_base: Path | None, current_base: Path) -> dict[str, Any]:
    if previous_base is None:
        return {"available": False, "reason": "sem snapshot anterior"}
    previous = _read_lot_classifications(previous_base)
    current = _read_lot_classifications(current_base)
    if previous.empty or current.empty:
        return {"available": False, "reason": "classificacoes de lote ainda ausentes em uma das execucoes"}
    if "arquivo" not in previous or "arquivo" not in current:
        return {"available": False, "reason": "coluna arquivo ausente"}

    prev = previous.copy()
    cur = current.copy()
    prev["label_anterior"] = prev.apply(_classification_label, axis=1)
    cur["label_atual"] = cur.apply(_classification_label, axis=1)
    prev_cols = ["arquivo", "classification_source", "label_anterior"]
    cur_cols = ["arquivo", "classification_source", "label_atual"]
    merged = prev[prev_cols].merge(cur[cur_cols], on="arquivo", how="inner", suffixes=("_anterior", "_atual"))
    if merged.empty:
        return {"available": False, "reason": "sem arquivos em comum"}
    changed = merged.loc[merged["label_anterior"] != merged["label_atual"]].copy()
    return {
        "available": True,
        "common_docs": len(merged),
        "changed_docs": len(changed),
        "stable_docs": len(merged) - len(changed),
        "changed_sample": changed.head(100),
    }


def _table(rows: list[dict[str, Any]] | pd.DataFrame) -> str:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return "<p>Sem dados ainda.</p>"
    return df.to_html(index=False, escape=True, classes="data-table")


def run(output_path: Path = DASHBOARD_HTML) -> dict[str, object]:
    previous = _latest_snapshot()
    current_summary = _summarize_run("atual", ANALYSIS_DIR)
    previous_summary = _summarize_run(previous.name, previous) if previous else {}
    comparison = _compare_classifications(previous, ANALYSIS_DIR)
    current_metrics = _read_metrics(ANALYSIS_DIR)

    changed_html = ""
    if comparison.get("available"):
        changed_html = _table(comparison["changed_sample"])
        comparison_summary = {
            "documentos_em_comum": comparison["common_docs"],
            "classificacao_estavel": comparison["stable_docs"],
            "reclassificados": comparison["changed_docs"],
        }
    else:
        comparison_summary = {"status": comparison.get("reason", "comparacao indisponivel")}

    html_text = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <title>Dashboard NT_PF - comparacao de execucoes</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; background: #f6f8fb; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    .muted {{ color: #5d6d7e; font-size: 13px; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
    .data-table th, .data-table td {{ border: 1px solid #d8dee9; padding: 6px 8px; text-align: left; vertical-align: top; }}
    .data-table th {{ background: #eaf0f6; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Dashboard NT_PF</h1>
  <p class="muted">Atualizado em {html.escape(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))}. A pagina recarrega a cada 30 segundos.</p>
  <div class="grid">
    <div class="card">
      <h2>Execucao Atual</h2>
      {_table([current_summary])}
    </div>
    <div class="card">
      <h2>Baseline Anterior</h2>
      {_table([previous_summary] if previous_summary else [])}
    </div>
    <div class="card">
      <h2>Comparacao</h2>
      {_table([comparison_summary])}
    </div>
  </div>
  <h2>Metricas Por Lote</h2>
  {_table(current_metrics)}
  <h2>Reclassificacoes Detectadas</h2>
  {changed_html or "<p>Sem comparacao de classificacoes disponivel ainda.</p>"}
  <p class="muted">Arquivo: <code>{html.escape(str(output_path))}</code></p>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return {"dashboard": str(output_path), "previous_snapshot": str(previous) if previous else ""}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
