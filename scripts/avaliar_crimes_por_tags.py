"""Evaluate crime classification against the criminal tags published by the PF.

The PF tag set also contains locations and institutional metadata. Only explicit,
curated criminal tags are used as reference labels; unmapped tags are preserved in
the detailed output but excluded from the metric denominator.
"""

from __future__ import annotations

import json
import hashlib
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


def classification_inputs(doc: dict[str, Any]) -> tuple[str, str]:
    title = str(doc.get("titulo", "") or "").strip()
    title_features = preprocess_body_text(title).semantic_features if title else ""
    parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
    body = str(doc.get("semantic_features", "") or "").strip()
    if not body:
        raw_body = str(doc.get("body_text", "") or parsed.get("corpo", "") or doc.get("context", "")).strip()
        body = preprocess_body_text(raw_body).semantic_features
    return title_features or title, body


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    reference_rows = [row for row in rows if row["gold_labels"]]
    accepted_rows = [row for row in reference_rows if bool(row["accepted"])]
    correct_rows = [row for row in accepted_rows if bool(row["correct"])]
    precision = len(correct_rows) / len(accepted_rows) if accepted_rows else 0.0
    recall = len(correct_rows) / len(reference_rows) if reference_rows else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    labels = sorted({label for row in reference_rows for label in row["gold_labels"]} | {str(row["predicted_label"]) for row in accepted_rows})
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(row["predicted_label"] == label and label in row["gold_labels"] for row in reference_rows)
        fp = sum(row["predicted_label"] == label and label not in row["gold_labels"] for row in accepted_rows)
        fn = sum(label in row["gold_labels"] and row["predicted_label"] != label for row in reference_rows)
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
        "correct_predictions": len(correct_rows),
        "coverage": round(len(accepted_rows) / len(reference_rows), 4) if reference_rows else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "by_crime": per_label,
    }


def run(
    docs_path: Path = DOCS_JSONL,
    feature_bank_path: Path = WNN_FEATURE_BANK_PATH,
    output_dir: Path = RUN_DIR / "avaliacao_crime_tags",
    max_per_crime: int | None = None,
    partition: str = "dev",
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
        tags = doc.get("tags") or parsed.get("tags") or []
        gold_labels = crime_labels_from_tags(tags if isinstance(tags, list) else [])
        if not gold_labels:
            continue
        if max_per_crime is not None:
            eligible_labels = [label for label in gold_labels if sampled_by_crime[label] < max_per_crime]
            if not eligible_labels:
                continue
            for label in eligible_labels:
                sampled_by_crime[label] += 1
        crime_text, title_fallback_text = classification_inputs(doc)
        result = classify_with_wnn(
            title_fallback_text,
            feature_bank_path,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            margin_threshold=DEFAULT_MARGIN_THRESHOLD,
            min_active_discriminators=DEFAULT_MIN_ACTIVE_DISCRIMINATORS,
            crime_text=crime_text,
            feature_bank_payload=feature_bank_payload,
            sync_memory=False,
        )
        predicted = result.top_label if result.accepted else ""
        rows.append(
            {
                "arquivo": document_id,
                "titulo": str(doc.get("titulo", "")),
                "tags_pf": tags,
                "gold_labels": gold_labels,
                "predicted_label": predicted,
                "accepted": result.accepted,
                "correct": bool(predicted and predicted in gold_labels),
                "status": result.status,
                "crime_evidence_source": result.crime_evidence_source,
                "confidence": round(result.crime_confidence, 4),
                "margin": round(result.crime_margin, 4),
                "active_discriminators": result.active_discriminators,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "predicoes_crime_por_tags.csv"
    pd.DataFrame(rows).to_csv(details_path, index=False, encoding="utf-8-sig")
    metrics = _metrics(rows)
    metrics["evaluation_scope"] = (
        {"mode": "stratified_sample", "max_per_crime": max_per_crime, "partition": partition}
        if max_per_crime is not None
        else {"mode": "full_mapped_tag_set", "partition": partition}
    )
    metrics["tag_mapping"] = dict(sorted(TAG_TO_CRIME.items()))
    metrics_path = output_dir / "metricas_crime_por_tags.json"
    metrics["outputs"] = {"details_csv": str(details_path), "metrics_json": str(metrics_path)}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(max_per_crime=10, partition="dev"), ensure_ascii=False, indent=2))
