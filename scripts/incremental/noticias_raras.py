from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Any

from scripts.incremental.common import RARE_NEWS_JSONL, RARE_NEWS_PROMOTION_THRESHOLD


RARE_SIGNATURE_RULES: list[tuple[str, list[str]]] = [
    ("violencia_politica_atos_antidemocraticos", ["atos antidemocraticos", "ordem politica", "instituicoes do poder publico"]),
    ("ameacas_stalking_perseguicao", ["stalking", "perseguicao", "ameaca", "ameacas", "ameacar"]),
    ("terrorismo_atos_preparatorios", ["terrorismo", "atos preparatorios", "atentado"]),
    ("tortura_sequestro_carcere", ["tortura", "sequestro", "carcere privado"]),
    ("violencia_domestica_feminicidio", ["violencia domestica", "feminicidio"]),
    ("assedio_sexual_coacao", ["assedio sexual", "coacao"]),
    ("execucao_mandado_prisional", ["mandado de prisao", "condenado", "foragido", "foragida"]),
    ("operacao_sem_contexto", ["nao apresenta informacoes", "nao ha detalhes", "sem informacoes suficientes"]),
]


def fold_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", text).strip()


def rare_signature(title: str, evidence: str = "", rationale: str = "") -> tuple[str, str]:
    text = fold_text(" ".join([title, evidence, rationale]))
    for label, terms in RARE_SIGNATURE_RULES:
        matched = [term for term in terms if term in text]
        if matched:
            return label, "; ".join(matched[:4])
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]{4,}", text)
        if token
        not in {
            "policia",
            "federal",
            "operacao",
            "deflagra",
            "investiga",
            "noticia",
            "crime",
            "crimes",
            "tema",
            "claro",
            "canonicas",
            "fornecidas",
        }
    ]
    if not tokens:
        return "rara_sem_assinatura", "sem termos suficientes"
    return "rara_" + "_".join(tokens[:3]), "assinatura lexical automatica"


def _read_rows() -> list[dict[str, Any]]:
    if not RARE_NEWS_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in RARE_NEWS_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_rare_news_observation(
    doc: dict[str, Any],
    iteration: int,
    evidence_text: str,
    rationale: str,
    confidence: float,
) -> dict[str, Any]:
    signature, reason = rare_signature(str(doc.get("titulo", "")), evidence_text, rationale)
    previous = _read_rows()
    count_before = sum(1 for row in previous if row.get("rare_signature") == signature)
    count_after = count_before + 1
    promoted_label = signature if signature != "operacao_sem_contexto" and count_after >= RARE_NEWS_PROMOTION_THRESHOLD else ""
    row = {
        "iteration": iteration,
        "arquivo": doc.get("arquivo", ""),
        "titulo": doc.get("titulo", ""),
        "rare_signature": signature,
        "signature_reason": reason,
        "signature_count": count_after,
        "promoted_label": promoted_label,
        "confidence": confidence,
        "evidence_text": evidence_text,
        "rationale": rationale,
    }
    RARE_NEWS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RARE_NEWS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def rare_news_summary() -> dict[str, Any]:
    rows = _read_rows()
    counts = Counter(str(row.get("rare_signature", "")) for row in rows)
    promoted = sorted({str(row.get("promoted_label", "")) for row in rows if row.get("promoted_label")})
    return {
        "rare_observations": len(rows),
        "rare_signatures": dict(counts),
        "promoted_from_rare": promoted,
    }
