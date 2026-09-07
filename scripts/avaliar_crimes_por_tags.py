"""Evaluate crime classification against the criminal tags published by the PF.

The PF tag set also contains locations and institutional metadata. Only explicit,
curated criminal tags are used as reference labels; unmapped tags are preserved in
the detailed output but excluded from the metric denominator.
"""

from __future__ import annotations

import json
import hashlib
import ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.incremental.preprocessamento_linguistico import preprocess_body_text
from scripts.pf_wnn_classifier import classify_with_wnn, load_feature_bank, sync_feature_memory
from scripts.text_utils import canonical_label


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analise_qualitativa"
RUN_DIR = ANALYSIS_DIR / "incremental"
LOTS_DIR = ANALYSIS_DIR / "lotes"
DOCS_JSONL = RUN_DIR / "documentos_base.jsonl"
WNN_FEATURE_BANK_PATH = ANALYSIS_DIR / "wnn_feature_bank.json"
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_MARGIN_THRESHOLD = 0.12
DEFAULT_MIN_ACTIVE_DISCRIMINATORS = 2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


TAG_TO_CRIME: dict[str, tuple[str, ...]] = {
    "trabalho escravo": ("trabalho_escravo",),
    "condicoes analogas a escravidao": ("trabalho_escravo",),
    "trafico de drogas": ("trafico_drogas",),
    "trafico internacional de drogas": ("trafico_drogas",),
    "trafico de drogas e armas": ("trafico_drogas", "armas_municoes"),
    "lavagem de dinheiro": ("lavagem_dinheiro",),
    "pornografia infantil": ("crimes_contra_criancas",),
    "abuso sexual infantojuvenil": ("crimes_contra_criancas",),
    "abuso sexual infantil": ("crimes_contra_criancas",),
    "exploracao sexual infantil": ("crimes_contra_criancas",),
    "contrabando": ("contrabando_descaminho",),
    "descaminho": ("contrabando_descaminho",),
    "corrupcao": ("corrupcao_desvio_recursos_publicos",),
    "desvio de recursos publicos": ("corrupcao_desvio_recursos_publicos",),
    "organizacao criminosa": ("crime_organizado",),
    "organizacoes criminosas": ("crime_organizado",),
    "combate organizacao criminosa": ("crime_organizado",),
    "crime organizado": ("crime_organizado",),
    "arma de fogo ilegal": ("armas_municoes",),
    "armas": ("armas_municoes",),
    "radio clandestina": ("radiodifusao_clandestina",),
    "radiodifusao clandestina": ("radiodifusao_clandestina",),
    "moeda falsa": ("moeda_falsa",),
    "crimes ambientais": ("crimes_ambientais",),
    "desmatamento": ("crimes_ambientais",),
    "garimpo ilegal": ("crimes_ambientais",),
    "mineracao ilegal": ("crimes_ambientais",),
    "trafico de animais silvestres": ("crimes_ambientais",),
    "crimes ciberneticos": ("crimes_ciberneticos",),
    "crime cibernetico": ("crimes_ciberneticos",),
    "crimes eleitorais": ("crimes_eleitorais",),
    "crimes previdenciarios": ("crimes_previdenciarios",),
    "fraudes previdenciarias": ("crimes_previdenciarios",),
    "crimes contra o sistema financeiro": ("crimes_sistema_financeiro",),
    "fraude ao auxilio emergencial": ("fraudes_auxilios_beneficios",),
    "fraude em beneficios sociais": ("fraudes_auxilios_beneficios",),
    "crimes migratorios": ("crimes_migratorios",),
}


def crime_labels_from_tags(tags: list[object] | None) -> list[str]:
    """Map only curated PF criminal tags to canonical crime labels."""
    labels: set[str] = set()
    for tag in tags or []:
        normalized = canonical_label(str(tag)).replace("_", " ")
        for key, mapped_labels in TAG_TO_CRIME.items():
            if normalized == key or key in normalized:
                labels.update(mapped_labels)
    return sorted(labels)


def classification_input(doc: dict[str, Any]) -> str:
    """Return x3 only; x2 (tags) is the gold target, never model input."""
    parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
    body = str(doc.get("semantic_features", "") or "").strip()
    if not body:
        raw_body = str(doc.get("x3_texto_noticia", "") or doc.get("body_text", "") or parsed.get("corpo", "")).strip()
        body = preprocess_body_text(raw_body).semantic_features
    return body


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    reference_rows = [row for row in rows if row["gold_labels"]]
    accepted_rows = [row for row in reference_rows if bool(row["accepted"])]
    correct_primary_rows = [
        row for row in accepted_rows if row["predicted_label"] and row["predicted_label"] in row["gold_labels"]
    ]
    correct_any_label_rows = [row for row in accepted_rows if bool(row["correct"])]
    precision = len(correct_primary_rows) / len(accepted_rows) if accepted_rows else 0.0
    recall = len(correct_primary_rows) / len(reference_rows) if reference_rows else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    multilabel_tp = sum(
        len(set(row.get("predicted_labels", [])).intersection(row["gold_labels"]))
        for row in reference_rows
    )
    multilabel_fp = sum(
        len(set(row.get("predicted_labels", [])).difference(row["gold_labels"]))
        for row in reference_rows
    )
    multilabel_fn = sum(
        len(set(row["gold_labels"]).difference(row.get("predicted_labels", [])))
        for row in reference_rows
    )
    multilabel_precision = multilabel_tp / (multilabel_tp + multilabel_fp) if multilabel_tp + multilabel_fp else 0.0
    multilabel_recall = multilabel_tp / (multilabel_tp + multilabel_fn) if multilabel_tp + multilabel_fn else 0.0
    multilabel_f1 = (
        2 * multilabel_precision * multilabel_recall / (multilabel_precision + multilabel_recall)
        if multilabel_precision + multilabel_recall
        else 0.0
    )

    # Global precision/recall/F1 remain tied to the primary decision. Per-crime
    # values use every predicted label, so a structural parent such as
    # crime_organizado is credited when emitted as a secondary parent marker.
    labels = sorted(
        {label for row in reference_rows for label in row["gold_labels"]}
        | {str(label) for row in accepted_rows for label in row.get("predicted_labels", []) if str(label)}
    )
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(label in row.get("predicted_labels", []) and label in row["gold_labels"] for row in reference_rows)
        fp = sum(label in row.get("predicted_labels", []) and label not in row["gold_labels"] for row in accepted_rows)
        fn = sum(label in row["gold_labels"] and label not in row.get("predicted_labels", []) for row in reference_rows)
        label_precision = tp / (tp + fp) if tp + fp else 0.0
        label_recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {
            "reference": tp + fn,
            "predicted": tp + fp,
            "true_positive": tp,
            "precision": round(label_precision, 4),
            "recall": round(label_recall, 4),
            "f1": round(2 * label_precision * label_recall / (label_precision + label_recall), 4)
            if label_precision + label_recall
            else 0.0,
        }

    return {
        "documents_with_mapped_crime_tag": len(reference_rows),
        "accepted_by_wnn": len(accepted_rows),
        "correct_predictions": len(correct_primary_rows),
        "coverage": round(len(accepted_rows) / len(reference_rows), 4) if reference_rows else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "document_match_any_label": len(correct_any_label_rows),
        "document_match_any_label_rate": round(len(correct_any_label_rows) / len(reference_rows), 4) if reference_rows else 0.0,
        "multilabel": {
            "true_positive": multilabel_tp,
            "false_positive": multilabel_fp,
            "false_negative": multilabel_fn,
            "precision": round(multilabel_precision, 4),
            "recall": round(multilabel_recall, 4),
            "f1": round(multilabel_f1, 4),
        },
        "by_crime": per_label,
    }


def _confusion_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return a long-form confusion matrix for possibly multi-label tags.

    A row with two reference tags contributes once to each corresponding gold
    label. Abstentions are kept explicit, so they are visible in the report.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        prediction = str(row["predicted_label"] or "abstencao_wnn")
        for gold in row["gold_labels"]:
            counts[(str(gold), prediction)] += 1
    return [
        {"tag_referencia": gold, "predicao_wnn": prediction, "quantidade": count}
        for (gold, prediction), count in sorted(counts.items())
    ]


def _as_label_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return _as_label_list(parsed)


def _write_evaluation_outputs(
    rows: list[dict[str, object]], output_dir: Path, scope: dict[str, object]
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "predicoes_crime_por_tags.csv"
    pd.DataFrame(rows).to_csv(details_path, index=False, encoding="utf-8-sig")
    confusion_path = output_dir / "matriz_confusao_crime_por_tags.csv"
    pd.DataFrame(_confusion_rows(rows)).to_csv(confusion_path, index=False, encoding="utf-8-sig")
    metrics = _metrics(rows)
    metrics["evaluation_scope"] = scope
    metrics["tag_mapping"] = dict(sorted(TAG_TO_CRIME.items()))
    metrics_path = output_dir / "metricas_crime_por_tags.json"
    metrics["outputs"] = {
        "details_csv": str(details_path),
        "confusion_matrix_csv": str(confusion_path),
        "metrics_json": str(metrics_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def run_from_incremental_batches(
    docs_path: Path = DOCS_JSONL,
    lots_dir: Path = LOTS_DIR,
    output_dir: Path = RUN_DIR / "avaliacao_crime_tags",
) -> dict[str, object]:
    """Evaluate each WNN decision recorded during chronological processing.

    This is the operational metric: it compares the discriminator output saved
    before residual learning for a reserve document with its x2 tags. It does not
    reclassify documents using the final, later-updated feature bank.
    """
    docs_by_name = {str(doc.get("arquivo", "")): doc for doc in read_jsonl(docs_path)}
    batch_paths = sorted(lots_dir.glob("lote_*_classificacoes.csv"))
    rows: list[dict[str, object]] = []
    for batch_path in batch_paths:
        for item in pd.read_csv(batch_path).fillna("").to_dict(orient="records"):
            if str(item.get("wnn_attempted", "")).lower() not in {"true", "1"}:
                continue
            document_id = str(item.get("arquivo", ""))
            doc = docs_by_name.get(document_id, {})
            parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
            tags = doc.get("x2_tags", doc.get("tags") or parsed.get("tags") or [])
            gold_labels = crime_labels_from_tags(tags if isinstance(tags, list) else [])
            if not gold_labels:
                continue
            accepted = str(item.get("wnn_accepted", "")).lower() in {"true", "1"}
            predicted = str(item.get("wnn_top_label", "")) if accepted else ""
            secondary = _as_label_list(item.get("wnn_marcadores_secundarios", [])) if accepted else []
            predicted_labels = list(dict.fromkeys([label for label in [predicted, *secondary] if label]))
            rows.append(
                {
                    "arquivo": document_id,
                    "titulo": str(doc.get("x1_titulo", doc.get("titulo", ""))),
                    "tags_pf": tags,
                    "gold_labels": gold_labels,
                    "predicted_label": predicted,
                    "predicted_labels": predicted_labels,
                    "accepted": accepted,
                    "correct": bool(set(predicted_labels).intersection(gold_labels)),
                    "status": str(item.get("wnn_status", "")),
                    "crime_evidence_source": str(item.get("wnn_crime_evidence_source", "")),
                    "confidence": item.get("wnn_crime_confidence", ""),
                    "margin": item.get("wnn_crime_margin", ""),
                }
            )
    return _write_evaluation_outputs(
        rows,
        output_dir,
        {"mode": "chronological_incremental_wnn_decisions", "batches_evaluated": len(batch_paths)},
    )


def run(
    docs_path: Path = DOCS_JSONL,
    feature_bank_path: Path = WNN_FEATURE_BANK_PATH,
    output_dir: Path = RUN_DIR / "avaliacao_crime_tags",
    max_per_crime: int | None = None,
    partition: str = "dev",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    min_active_discriminators: int = DEFAULT_MIN_ACTIVE_DISCRIMINATORS,
) -> dict[str, object]:
    if partition not in {"dev", "test", "all"}:
        raise ValueError("partition deve ser 'dev', 'test' ou 'all'")
    rows: list[dict[str, object]] = []
    feature_bank_payload = load_feature_bank(feature_bank_path)
    sync_feature_memory(feature_bank_payload)
    sampled_by_crime: Counter[str] = Counter()
    for doc in read_jsonl(docs_path):
        document_id = str(doc.get("arquivo", ""))
        bucket = int(hashlib.sha1(document_id.encode("utf-8")).hexdigest(), 16) % 5
        if partition == "dev" and bucket == 0:
            continue
        if partition == "test" and bucket != 0:
            continue
        parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
        tags = doc.get("x2_tags", doc.get("tags") or parsed.get("tags") or [])
        gold_labels = crime_labels_from_tags(tags if isinstance(tags, list) else [])
        if not gold_labels:
            continue
        if max_per_crime is not None:
            eligible_labels = [label for label in gold_labels if sampled_by_crime[label] < max_per_crime]
            if not eligible_labels:
                continue
            for label in eligible_labels:
                sampled_by_crime[label] += 1
        crime_text = classification_input(doc)
        result = classify_with_wnn(
            crime_text,
            feature_bank_path,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            min_active_discriminators=min_active_discriminators,
            crime_text=crime_text,
            feature_bank_payload=feature_bank_payload,
            sync_memory=False,
        )
        predicted = result.top_label if result.accepted else ""
        secondary = result.inference.marcadores_secundarios if result.accepted and result.inference else []
        predicted_labels = list(dict.fromkeys([label for label in [predicted, *secondary] if label]))
        rows.append(
            {
                "arquivo": document_id,
                "titulo": str(doc.get("titulo", "")),
                "tags_pf": tags,
                "gold_labels": gold_labels,
                "predicted_label": predicted,
                "predicted_labels": predicted_labels,
                "accepted": result.accepted,
                "correct": bool(set(predicted_labels).intersection(gold_labels)),
                "status": result.status,
                "crime_evidence_source": result.crime_evidence_source,
                "confidence": round(result.crime_confidence, 4),
                "margin": round(result.crime_margin, 4),
                "active_discriminators": result.active_discriminators,
            }
        )

    scope = (
        {"mode": "stratified_sample", "max_per_crime": max_per_crime, "partition": partition}
        if max_per_crime is not None
        else {"mode": "full_mapped_tag_set", "partition": partition}
    )
    return _write_evaluation_outputs(rows, output_dir, scope)


if __name__ == "__main__":
    print(json.dumps(run(max_per_crime=10, partition="dev"), ensure_ascii=False, indent=2))
