from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from scripts.pf_llm_models import NoticiaLLMInference
    from scripts.pf_regex_classifier import canonical_label, fold_text
except ModuleNotFoundError:
    from pf_llm_models import NoticiaLLMInference
    from pf_regex_classifier import canonical_label, fold_text


DEFAULT_FEATURE_WEIGHT = 1.0
EVIDENCE_FEATURE_WEIGHT = 0.75
WEAK_SIGNAL_WEIGHT = 0.20
MAX_SIGNATURES_PER_LABEL = 250
WNN_THEME_PARENTS: dict[str, str] = {}
STRENGTH_WEIGHTS = {
    "strong": 1.15,
    "medium": 0.75,
    "weak": WEAK_SIGNAL_WEIGHT,
}
ORGANIZED_CRIME_BRIDGE_TOKENS = {
    "associacao",
    "criminosa",
    "crime",
    "faccao",
    "faccoes",
    "organizacao",
    "organizado",
    "quadrilha",
}
STRONG_ORGANIZED_CRIME_BRIDGE_TOKENS = ORGANIZED_CRIME_BRIDGE_TOKENS - {"crime"}
ORGANIZED_OPERATIONAL_SUBTHEMES = {
    "trafico_drogas",
    "lavagem_dinheiro",
    "armas_municoes",
}
ENVIRONMENTAL_SUBTHEMES = {
    "crimes_ambientais",
    "mineracao_ilegal",
    "garimpo_ilegal",
}
PROTECTED_DOMAIN_PRIORITY = {
    "crimes_contra_criancas": 100,
    "crimes_ambientais": 90,
    "trabalho_escravo": 85,
}
COSINE_SUPPORT_MIN_SCORE = 0.18
COSINE_SUSPICION_MIN_SCORE = 0.22
COSINE_ASSISTED_MARGIN_THRESHOLD = 0.04
CONFIRMED_MEDIUM_THRESHOLD = 2
CONFIRMED_STRONG_THRESHOLD = 4

CURATED_THEME_DISCRIMINATORS: dict[str, list[dict[str, object]]] = {
    "crimes_contra_criancas": [
        {"name": "pornografia_infantil", "tokens": ["pornografia", "infantil"], "weight": 1.2},
        {"name": "abuso_sexual_infantil", "tokens": ["abuso", "sexual", "infantil"], "weight": 1.2},
        {"name": "estupro_vulneravel", "tokens": ["estupro", "vulneravel"], "weight": 1.15},
        {"name": "exploracao_sexual_infantil", "tokens": ["exploracao", "sexual", "infantil"], "weight": 1.1},
        {"name": "material_abuso_infantil", "tokens": ["material", "abuso", "infantil"], "weight": 1.0},
        {"name": "armazenamento_pornografia_infantil", "tokens": ["armazenamento", "pornografia", "infantil"], "weight": 0.9},
        {"name": "compartilhamento_pornografia_infantil", "tokens": ["compartilhamento", "pornografia", "infantil"], "weight": 0.9},
        {"name": "aliciamento_menor", "tokens": ["aliciamento", "menor"], "weight": 0.9},
    ],
    "trafico_drogas": [
        {"name": "trafico_drogas", "tokens": ["trafico", "drogas"], "weight": 1.2},
        {"name": "trafico_internacional_entorpecentes", "tokens": ["trafico", "internacional", "entorpecentes"], "weight": 1.1},
        {"name": "apreensao_cocaina", "tokens": ["apreensao", "cocaina"], "weight": 0.9},
        {"name": "apreensao_maconha", "tokens": ["apreensao", "maconha"], "weight": 0.9},
        {"name": "plantio_maconha", "tokens": ["plantio", "maconha"], "weight": 0.9},
    ],
    "crimes_ambientais": [
        {"name": "garimpo_ilegal", "tokens": ["garimpo", "ilegal"], "weight": 1.15},
        {"name": "extracao_ilegal_ouro", "tokens": ["extracao", "ilegal", "ouro"], "weight": 1.1},
        {"name": "desmatamento_ilegal", "tokens": ["desmatamento", "ilegal"], "weight": 1.1},
        {"name": "madeira_ilegal", "tokens": ["madeira", "ilegal"], "weight": 1.0},
        {"name": "trafico_animais_silvestres", "tokens": ["trafico", "animais", "silvestres"], "weight": 0.95},
    ],
    "contrabando_descaminho": [
        {"name": "contrabando_descaminho", "tokens": ["contrabando", "descaminho"], "weight": 1.2},
        {"name": "cigarros_ilegais", "tokens": ["cigarros", "ilegais"], "weight": 1.05},
        {"name": "mercadoria_estrangeira_irregular", "tokens": ["mercadoria", "estrangeira", "irregular"], "weight": 0.95},
    ],
    "lavagem_dinheiro": [
        {"name": "lavagem_dinheiro", "tokens": ["lavagem", "dinheiro"], "weight": 1.2},
        {"name": "ocultacao_bens", "tokens": ["ocultacao", "bens"], "weight": 1.0},
        {"name": "dissimulacao_valores", "tokens": ["dissimulacao", "valores"], "weight": 1.0},
    ],
    "corrupcao_desvio_recursos_publicos": [
        {"name": "desvio_recursos_publicos", "tokens": ["desvio", "recursos", "publicos"], "weight": 1.2},
        {"name": "fraude_licitacao", "tokens": ["fraude", "licitacao"], "weight": 1.1},
        {"name": "contratacao_fraudulenta", "tokens": ["contratacao", "fraudulenta"], "weight": 0.95},
    ],
    "armas_municoes": [
        {"name": "arma_fogo", "tokens": ["arma", "fogo"], "weight": 1.1},
        {"name": "porte_ilegal_arma", "tokens": ["porte", "ilegal", "arma"], "weight": 1.1},
        {"name": "posse_ilegal_arma", "tokens": ["posse", "ilegal", "arma"], "weight": 1.1},
    ],
    "crime_organizado": [
        {"name": "organizacao_criminosa", "tokens": ["organizacao", "criminosa"], "weight": 1.15},
        {"name": "crime_organizado", "tokens": ["crime", "organizado"], "weight": 1.1},
        {"name": "associacao_criminosa", "tokens": ["associacao", "criminosa"], "weight": 1.0},
        {"name": "faccao_criminosa", "tokens": ["faccao", "criminosa"], "weight": 1.0},
    ],
    "radiodifusao_clandestina": [
        {"name": "radio_clandestina", "tokens": ["radio", "clandestina"], "weight": 1.15},
        {"name": "radiodifusao_clandestina", "tokens": ["radiodifusao", "clandestina"], "weight": 1.15},
        {"name": "telecomunicacao_irregular", "tokens": ["telecomunicacao", "irregular"], "weight": 0.9},
    ],
    "moeda_falsa": [
        {"name": "moeda_falsa", "tokens": ["moeda", "falsa"], "weight": 1.2},
        {"name": "cedulas_falsas", "tokens": ["cedulas", "falsas"], "weight": 1.1},
        {"name": "notas_falsas", "tokens": ["notas", "falsas"], "weight": 1.0},
    ],
    "trabalho_escravo": [
        {"name": "trabalho_escravo", "tokens": ["trabalho", "escravo"], "weight": 1.25},
        {"name": "condicoes_analogas_escravidao", "tokens": ["condicoes", "analogas", "escravidao"], "weight": 1.25},
        {"name": "condicao_analoga_escravo", "tokens": ["condicao", "analoga", "escravo"], "weight": 1.2},
        {"name": "trabalho_analogo_escravidao", "tokens": ["trabalho", "analogo", "escravidao"], "weight": 1.2},
        {"name": "trabalhadores_resgatados", "tokens": ["trabalhadores", "resgatados"], "weight": 1.05},
        {"name": "resgate_trabalhadores", "tokens": ["resgate", "trabalhadores"], "weight": 1.0},
    ],
}

THEME_MICRO_WORLD_ANCHORS: dict[str, set[str]] = {
    "crimes_contra_criancas": {
        "adolescente",
        "adolescentes",
        "aliciamento",
        "armazenamento",
        "abuso",
        "crianca",
        "criancas",
        "compartilhamento",
        "estupro",
        "exploracao",
        "infantil",
        "infantojuvenil",
        "material",
        "menor",
        "pornografia",
        "sexual",
        "vulneravel",
    },
    "trafico_drogas": {"apreensao", "cocaina", "droga", "drogas", "entorpecentes", "maconha", "plantio", "trafico"},
    "crimes_ambientais": {
        "ambiental",
        "ambientais",
        "animais",
        "desmatamento",
        "extracao",
        "fauna",
        "garimpo",
        "ilegal",
        "madeira",
        "ouro",
        "pesca",
        "silvestres",
    },
    "contrabando_descaminho": {
        "cigarro",
        "cigarros",
        "contrabando",
        "descaminho",
        "eletronicos",
        "estrangeira",
        "ilegais",
        "mercadoria",
        "produtos",
    },
    "lavagem_dinheiro": {"bens", "dissimulacao", "dinheiro", "lavagem", "ocultacao", "valores"},
    "corrupcao_desvio_recursos_publicos": {
        "contratacao",
        "corrupcao",
        "desvio",
        "fraude",
        "fraudulenta",
        "licitacao",
        "publicos",
        "recursos",
    },
    "armas_municoes": {"arma", "armamento", "fogo", "municao", "municoes", "porte", "posse"},
    "crime_organizado": {"associacao", "criminosa", "faccao", "faccoes", "organizacao", "organizado", "quadrilha"},
    "radiodifusao_clandestina": {"anatel", "clandestina", "radio", "radiodifusao", "telecomunicacao"},
    "moeda_falsa": {"cedula", "cedulas", "falsa", "falsas", "moeda", "notas"},
    "trabalho_escravo": {
        "analoga",
        "analogas",
        "analogo",
        "condicao",
        "condicoes",
        "escravidao",
        "escravo",
        "resgate",
        "resgatados",
        "trabalhadores",
        "trabalho",
    },
    "crimes_ciberneticos": {"cibernetico", "ciberneticos", "dados", "dispositivos", "internet", "invasao", "sistemas"},
    "crimes_migratorios": {"imigracao", "migracao", "migrantes", "passaportes"},
    "crimes_previdenciarios": {"aposentadoria", "beneficio", "beneficios", "inss", "previdencia", "previdenciario"},
    "crimes_eleitorais": {"compra", "eleicoes", "eleitoral", "eleitorais", "votos"},
    "crimes_sistema_financeiro": {"bancaria", "financeiro", "sistema", "emprestimos"},
    "fraudes_auxilios_beneficios": {"auxilio", "brasil", "beneficio", "beneficios", "emergencial", "fraude"},
}

THEME_MICRO_WORLD_BRIDGES: dict[str, set[str]] = {
    "crime_organizado": ORGANIZED_OPERATIONAL_SUBTHEMES,
}

BLOCKED_SENSOR_TERMS = {
    "acao",
    "apoio",
    "busca",
    "cumpre",
    "deflagra",
    "deflagrou",
    "delegacia",
    "federal",
    "investiga",
    "investigacao",
    "mandado",
    "mandados",
    "operacao",
    "policia",
    "prisao",
    "regional",
    "suspeito",
    "suspeitos",
    "destaque",
    "objetivo",
    "batalhao",
    "combate",
    "dominus",
    "feira",
    "forca",
    "icmbio",
    "nesta",
    "quinta",
    "quarta",
    "segunda",
    "terca",
    "tarefa",
}


@dataclass(frozen=True)
class WNNClassification:
    inference: NoticiaLLMInference | None
    status: str
    confidence: float
    margin: float
    top_label: str
    active_discriminators: list[dict[str, object]]
    scores: list[dict[str, object]]
    feature_bank: str
    theme_candidate: dict[str, object] | None = None

    @property
    def accepted(self) -> bool:
        return self.inference is not None and self.status == "accepted"

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "top_label": self.top_label,
            "feature_bank": self.feature_bank,
            "active_discriminators": self.active_discriminators[:20],
            "scores": self.scores[:5],
            "theme_candidate": self.theme_candidate,
            "inference": self.inference.model_dump() if self.inference else None,
        }


def _stable_id(label: str, pattern: str) -> str:
    digest = hashlib.sha1(f"{label}:{pattern}".encode("utf-8")).hexdigest()[:12]
    return f"{canonical_label(label)}__{digest}"


def parent_theme(label: str) -> str:
    normalized = canonical_label(label)
    return WNN_THEME_PARENTS.get(normalized, normalized)


def _with_parent_theme(item: dict[str, object], original_label: str) -> dict[str, object]:
    parent = parent_theme(original_label)
    item["label"] = parent
    if parent != canonical_label(original_label):
        item["subtheme"] = canonical_label(original_label)
        item["parent_theme"] = parent
    return item


def _active_label(item: dict[str, object]) -> str:
    subtheme = canonical_label(str(item.get("subtheme", "")))
    if subtheme:
        return subtheme
    return canonical_label(str(item.get("label", "")))


def _active_tokens(active: list[dict[str, object]]) -> set[str]:
    tokens: set[str] = set()
    for item in active:
        raw_tokens = item.get("tokens", [])
        if isinstance(raw_tokens, list):
            tokens.update(canonical_label(str(token)) for token in raw_tokens if str(token).strip())
    return tokens


def _dedupe_labels(labels: Iterable[str]) -> list[str]:
    output: list[str] = []
    for label in labels:
        normalized = canonical_label(str(label))
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _domain_secondaries(primary: str, ranked: list[str], scores_by_label: dict[str, float]) -> list[str]:
    return [
        label
        for label in ranked
        if label != primary and scores_by_label.get(label, 0.0) >= 0.75
    ]


def _preferred_protected_domain(active_labels: set[str], scores_by_label: dict[str, float]) -> str:
    candidates = [
        label
        for label in active_labels
        if label in PROTECTED_DOMAIN_PRIORITY and scores_by_label.get(label, 0.0) >= 0.5
    ]
    if not candidates:
        return ""
    return sorted(
        candidates,
        key=lambda label: (PROTECTED_DOMAIN_PRIORITY.get(label, 0), scores_by_label.get(label, 0.0)),
        reverse=True,
    )[0]


def _cosine_top_candidate(cosine_candidates: list[dict[str, object]] | None) -> tuple[str, float]:
    for candidate in cosine_candidates or []:
        if not isinstance(candidate, dict):
            continue
        label = canonical_label(str(candidate.get("label", "")))
        try:
            score = float(candidate.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if label and score > 0:
            return label, score
    return "", 0.0


def _cosine_supports_decision(
    top_label: str,
    secondary: list[str],
    cosine_candidates: list[dict[str, object]] | None,
) -> bool:
    cosine_label, cosine_score = _cosine_top_candidate(cosine_candidates)
    return bool(
        cosine_label
        and cosine_score >= COSINE_SUPPORT_MIN_SCORE
        and cosine_label in {top_label, *secondary}
    )


def _cosine_suspicion(
    top_label: str,
    secondary: list[str],
    cosine_candidates: list[dict[str, object]] | None,
) -> tuple[bool, str, float]:
    cosine_label, cosine_score = _cosine_top_candidate(cosine_candidates)
    if not cosine_label or cosine_score < COSINE_SUSPICION_MIN_SCORE:
        return False, cosine_label, cosine_score
    decision_labels = {top_label, *secondary}
    if cosine_label in decision_labels:
        return False, cosine_label, cosine_score
    if cosine_label in PROTECTED_DOMAIN_PRIORITY or top_label in PROTECTED_DOMAIN_PRIORITY:
        return True, cosine_label, cosine_score
    return False, cosine_label, cosine_score


def _operational_decision(
    scores_by_label: dict[str, float],
    active: list[dict[str, object]],
) -> tuple[str, list[str], list[str], str]:
    ranked = [label for label, score in sorted(scores_by_label.items(), key=lambda item: item[1], reverse=True) if score > 0]
    if not ranked:
        return "", [], [], ""

    active_labels = {_active_label(item) for item in active}
    tokens = _active_tokens(active)
    has_organization_bridge = (
        bool(tokens.intersection(STRONG_ORGANIZED_CRIME_BRIDGE_TOKENS))
        or "crime_organizado" in active_labels
    )
    organized_subthemes = sorted(active_labels.intersection(ORGANIZED_OPERATIONAL_SUBTHEMES))

    protected_domain = _preferred_protected_domain(active_labels, scores_by_label)
    if protected_domain:
        secondary = _domain_secondaries(protected_domain, ranked, scores_by_label)
        crimes = _dedupe_labels([protected_domain, *secondary])
        relation = "dominio_preferencial"
        if has_organization_bridge and "crime_organizado" in secondary:
            relation = "dominio_preferencial_com_organizacao"
        return protected_domain, crimes, [label for label in crimes if label != protected_domain], relation

    if has_organization_bridge and (organized_subthemes or len(active_labels) > 1):
        operational_domains = [
            label
            for label in sorted(active_labels)
            if label != "crime_organizado" and scores_by_label.get(label, 0.0) >= 0.75
        ]
        crimes = _dedupe_labels(["crime_organizado", *operational_domains])
        secondary = [label for label in crimes if label != "crime_organizado"]
        relation = "crime_organizado_multidominio" if len(secondary) > 1 else "cadeia_operacional"
        return "crime_organizado", crimes, secondary, relation

    top_label = ranked[0]
    secondary = [label for label in ranked[1:] if scores_by_label.get(label, 0.0) >= 0.75]
    if secondary:
        return top_label, _dedupe_labels([top_label, *secondary]), secondary, "coocorrencia_sem_fusao"
    return top_label, [top_label], [], "tema_unico"


def _multi_discriminator_candidate(
    scores_by_label: dict[str, float],
    active: list[dict[str, object]],
    relation: str,
) -> dict[str, object] | None:
    if relation in {
        "cadeia_operacional",
        "crime_organizado_multidominio",
        "dominio_preferencial",
        "dominio_preferencial_com_organizacao",
    }:
        return None
    active_labels = {
        _active_label(item)
        for item in active
        if _active_label(item) and scores_by_label.get(_active_label(item), 0.0) >= 1.0
    }
    if len(active_labels) < 2:
        return None
    ranked_scores = sorted(
        [float(scores_by_label.get(label, 0.0) or 0.0) for label in active_labels],
        reverse=True,
    )
    if len(ranked_scores) < 2 or ranked_scores[0] <= 0:
        return None
    margin = (ranked_scores[0] - ranked_scores[1]) / ranked_scores[0]
    if margin > 0.25:
        return None
    labels = sorted(active_labels)
    marker_terms: list[str] = []
    marker_ids: list[str] = []
    for item in active:
        label = _active_label(item)
        if label not in active_labels:
            continue
        marker_id = str(item.get("id", ""))
        if marker_id and marker_id not in marker_ids:
            marker_ids.append(marker_id)
        raw_tokens = item.get("tokens", [])
        tokens = raw_tokens if isinstance(raw_tokens, list) else []
        for token in tokens:
            token = canonical_label(str(token))
            if token not in marker_terms:
                marker_terms.append(token)
    return {
        "candidate_label": "composto_" + "__".join(labels[:4]),
        "kind": "multi_discriminator_composition",
        "labels": labels,
        "relation": relation or "coocorrencia_sem_fusao",
        "marker_terms": marker_terms[:20],
        "marker_ids": marker_ids[:20],
        "rationale": "Multiplos discriminadores fortes foram acionados; registrar composicao para ajuste da arvore/discriminadores sem fusao automatica.",
    }


def _compile_pattern(pattern: str) -> re.Pattern[str] | None:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


def _tokens_from_text(value: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]{4,}\b", fold_text(value))


def _token_set_from_text(value: str) -> set[str]:
    return set(_tokens_from_text(value))


def _tokens_match_text(tokens: Iterable[object], words: set[str]) -> bool:
    normalized = [
        canonical_label(str(token))
        for token in tokens
        if canonical_label(str(token)) and len(canonical_label(str(token))) >= 4
    ]
    if not normalized:
        return False
    for token in normalized:
        if token in words:
            continue
        if not any(word.startswith(token) or token.startswith(word) for word in words if len(word) >= 4):
            return False
    return True


def _is_sensor_candidate(value: str) -> bool:
    tokens = _tokens_from_text(value)
    if len(tokens) < 2:
        return False
    if all(token in BLOCKED_SENSOR_TERMS for token in tokens):
        return False
    return bool(set(tokens) - BLOCKED_SENSOR_TERMS)


def _literal_pattern(value: str) -> str:
    tokens = _tokens_from_text(value)
    if len(tokens) < 2:
        return ""
    return r"\s+".join(rf"\b{re.escape(token)}\w*\b" for token in tokens)


def _tokens_from_regex_pattern(pattern: str) -> list[str]:
    tokens = re.findall(r"\\b([a-z0-9]{3,})", fold_text(pattern))
    if not tokens:
        tokens = _tokens_from_text(pattern)
    selected: list[str] = []
    for token in tokens:
        if len(token) < 4 or token in BLOCKED_SENSOR_TERMS:
            continue
        if token not in selected:
            selected.append(token)
    return selected


def _label_tokens(label: str) -> list[str]:
    return [
        token
        for token in canonical_label(label).split("_")
        if len(token) >= 4 and token not in {"crime", "crimes"}
    ]


def _label_hint_tokens(label: str) -> list[str]:
    hints: list[str] = []
    for item in CURATED_THEME_DISCRIMINATORS.get(canonical_label(label), []):
        raw_tokens = item.get("tokens", [])
        if not isinstance(raw_tokens, list):
            continue
        for token in raw_tokens:
            normalized = canonical_label(str(token))
            if normalized and len(normalized) >= 4 and normalized not in BLOCKED_SENSOR_TERMS and normalized not in hints:
                hints.append(normalized)
    return hints


def _theme_anchor_tokens(label: str) -> set[str]:
    normalized = canonical_label(label)
    anchors = set(THEME_MICRO_WORLD_ANCHORS.get(normalized, set()))
    anchors.update(_label_tokens(normalized))
    anchors.update(_label_hint_tokens(normalized))
    return {token for token in anchors if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS}


def _marker_matches_micro_world(label: str, tokens: Iterable[object]) -> tuple[bool, str]:
    normalized_label = canonical_label(label)
    clean_tokens = {
        canonical_label(str(token))
        for token in tokens
        if canonical_label(str(token)) and len(canonical_label(str(token))) >= 4
    }
    if len(clean_tokens) < 2:
        return False, "menos de dois tokens substantivos"
    anchors = _theme_anchor_tokens(normalized_label)
    if not anchors:
        return True, "sem vocabulario de guarda para o tema"
    anchor_hits = clean_tokens.intersection(anchors)
    foreign_hits: set[str] = set()
    for other_label, other_anchors in THEME_MICRO_WORLD_ANCHORS.items():
        if other_label == normalized_label:
            continue
        foreign_hits.update(clean_tokens.intersection(other_anchors - anchors))
    label_terms = set(_label_tokens(normalized_label))
    strong_anchor_hits = anchor_hits - label_terms - {"contra", "crime", "crimes"}
    if foreign_hits and not strong_anchor_hits:
        return False, f"contaminacao cruzada: {', '.join(sorted(foreign_hits)[:4])}"
    if anchor_hits:
        return True, f"ancoras do tema: {', '.join(sorted(anchor_hits)[:4])}"
    return False, f"sem ancora do micromundo {normalized_label}"


def _marker_strength(
    label: str,
    tokens: Iterable[object],
    source: str = "",
    confirmations: int = 0,
) -> str:
    clean_tokens = [
        canonical_label(str(token))
        for token in tokens
        if canonical_label(str(token)) and len(canonical_label(str(token))) >= 4
    ]
    if len(clean_tokens) < 2:
        return "weak"
    if str(source).startswith("weak_signal:"):
        return "weak"
    if confirmations >= CONFIRMED_STRONG_THRESHOLD:
        return "strong"
    normalized_label = canonical_label(label)
    anchors = _theme_anchor_tokens(normalized_label)
    anchor_hits = set(clean_tokens).intersection(anchors)
    if source == "agent2_curated_discriminator":
        return "strong"
    if normalized_label in PROTECTED_DOMAIN_PRIORITY and len(anchor_hits) >= 2:
        return "strong"
    if confirmations >= CONFIRMED_MEDIUM_THRESHOLD:
        return "medium"
    if len(anchor_hits) >= 2:
        return "medium"
    if source in {"agent2_generalized_micro_world", "agent1_evidence_term"} and anchor_hits:
        return "medium"
    return "weak"


def _apply_strength_metadata(item: dict[str, object]) -> dict[str, object]:
    label = parent_theme(str(item.get("label", "")))
    tokens = item.get("tokens", [])
    source = str(item.get("source", ""))
    try:
        confirmations = int(item.get("confirmations", 0) or 0)
    except (TypeError, ValueError):
        confirmations = 0
    strength = _marker_strength(label, tokens if isinstance(tokens, list) else [], source, confirmations)
    item["strength"] = strength
    item["weight"] = max(
        float(item.get("weight", 0.0) or 0.0),
        STRENGTH_WEIGHTS.get(strength, WEAK_SIGNAL_WEIGHT),
    )
    if strength == "weak":
        item["weight"] = min(float(item.get("weight", WEAK_SIGNAL_WEIGHT) or WEAK_SIGNAL_WEIGHT), WEAK_SIGNAL_WEIGHT)
    return item


def _unordered_pattern(tokens: list[str]) -> str:
    if not tokens:
        return ""
    lookaheads = "".join(rf"(?=.*\b{re.escape(token)}\w*\b)" for token in tokens)
    return rf"{lookaheads}.*"


def _substantive_tokens(value: str, min_len: int = 4) -> list[str]:
    output: list[str] = []
    for token in _tokens_from_text(value):
        if len(token) < min_len or token in BLOCKED_SENSOR_TERMS:
            continue
        if token not in output:
            output.append(token)
    return output


def _token_stem(value: str) -> str:
    token = canonical_label(value)
    for suffix in ("mente", "coes", "cao", "oes", "adas", "ados", "ada", "ado", "ais", "al", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token[:7] if len(token) > 7 else token


def _token_related(left: str, right: str) -> bool:
    left_stem = _token_stem(left)
    right_stem = _token_stem(right)
    if not left_stem or not right_stem:
        return False
    return left_stem == right_stem or left_stem in right_stem or right_stem in left_stem


def _compact_marker_tokens(tokens: Iterable[str], max_tokens: int = 4) -> list[str]:
    selected: list[str] = []
    for token in tokens:
        cleaned = canonical_label(str(token))
        if len(cleaned) < 4 or cleaned in BLOCKED_SENSOR_TERMS:
            continue
        if cleaned not in selected:
            selected.append(cleaned)
        if len(selected) >= max_tokens:
            break
    return selected


def _agent2_generalized_marker_sets(label: str, terms: Iterable[str]) -> list[list[str]]:
    """Create order-independent marker families from the theme micro-world.

    The goal is to generalize like:
    "trabalho escravo" -> "condicoes analogas escravidao", "trabalhadores resgatados",
    without turning every isolated word into a deterministic rule.
    """

    label_terms = _label_tokens(label)
    hint_terms = _label_hint_tokens(label)
    label_or_hint = [*label_terms, *hint_terms]
    markers: list[list[str]] = []

    def add(tokens: Iterable[str]) -> None:
        marker = _compact_marker_tokens(tokens)
        if len(marker) < 2:
            return
        signature = "|".join(sorted(marker))
        if signature not in {"|".join(sorted(item)) for item in markers}:
            markers.append(marker)

    for term in terms:
        tokens = _substantive_tokens(str(term))
        if len(tokens) >= 2:
            add(tokens[:4])
        if len(tokens) >= 3:
            add(tokens[-3:])
        if label_terms:
            related = [
                token
                for token in tokens
                if token in hint_terms or any(_token_related(token, label_token) for label_token in label_or_hint)
            ]
            if related:
                add([*label_terms[:2], *related[:2]])

    return markers


def _marker_signature(label: str, tokens: Iterable[object]) -> str:
    normalized = sorted(
        {
            token
            for token in (canonical_label(str(value)) for value in tokens)
            if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
        }
    )
    if not normalized:
        return ""
    return f"{canonical_label(label)}:{'|'.join(normalized)}"


def _marker_signature_from_discriminator(item: dict[str, object]) -> str:
    label = canonical_label(str(item.get("label", "")))
    tokens = item.get("tokens")
    if isinstance(tokens, list) and tokens:
        return _marker_signature(label, tokens)
    return _marker_signature(label, _tokens_from_regex_pattern(str(item.get("pattern", ""))))


def learned_rule_to_discriminator(rule: dict[str, object]) -> dict[str, object] | None:
    original_label = canonical_label(str(rule.get("label", "")))
    label = parent_theme(original_label)
    if not original_label:
        return None

    raw_tokens = rule.get("tokens", [])
    if isinstance(raw_tokens, list):
        tokens = [
            token
            for token in (canonical_label(str(value)) for value in raw_tokens)
            if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
        ]
    else:
        tokens = []
    if not tokens:
        legacy_pattern = str(rule.get("pattern", "")).strip()
        if not legacy_pattern:
            return None
        tokens = _tokens_from_regex_pattern(legacy_pattern)
    label_terms = _label_tokens(original_label)
    hint_terms = _label_hint_tokens(original_label)
    priority = [token for token in tokens if token in label_terms]
    micro_world = [token for token in tokens if token in hint_terms and token not in priority]
    support = [token for token in tokens if token not in priority and token not in micro_world]
    selected = [*priority, *micro_world, *support]
    selected = selected[:4]
    if len(selected) < 2:
        return None
    guard_ok, _guard_reason = _marker_matches_micro_world(label, selected)
    if not guard_ok:
        return None

    unordered = _unordered_pattern(selected)
    signature = _marker_signature(label, selected)
    discriminator = {
        "id": _stable_id(label, signature),
        "label": label,
        "name": canonical_label(str(rule.get("name", ""))) or "_".join(selected),
        "pattern": unordered,
        "weight": EVIDENCE_FEATURE_WEIGHT,
        "source": str(rule.get("source", "agent3_learned_discriminator")),
        "rationale": str(rule.get("rationale", "discriminador WNN aprendido no residual por tokens substantivos")),
        "tokens": selected,
        "marker_signature": signature,
        "confirmations": 1,
    }
    return _with_parent_theme(_apply_strength_metadata(discriminator), original_label)


def suggest_discriminator_rules_from_review(doc: dict[str, Any], review: Any) -> list[dict[str, object]]:
    label = canonical_label(str(getattr(review, "canonical_label", "") or ""))
    if not label:
        return []
    parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
    evidence = str(
        getattr(review, "evidence_text", "")
        or doc.get("body_text", "")
        or parsed.get("corpo", "")
        or doc.get("context", "")
    )
    label_terms = _label_tokens(label)
    hint_terms = _label_hint_tokens(label)
    evidence_tokens = [
        token
        for token in _tokens_from_text(evidence)
        if token not in BLOCKED_SENSOR_TERMS
    ]
    if label == "crime_organizado" and not set(evidence_tokens).intersection(ORGANIZED_CRIME_BRIDGE_TOKENS):
        return []
    selected: list[str] = []
    for token in [*label_terms, *[token for token in evidence_tokens if token in hint_terms], *evidence_tokens]:
        if token not in selected:
            selected.append(token)
        if len(selected) >= 5:
            break
    if len(selected) < 2:
        return []
    return [
        {
            "label": label,
            "name": "_".join(selected[:4]),
            "tokens": selected,
            "source": "agent3_learned_discriminator",
            "rationale": "marcador WNN aprendido a partir de classificacao residual do Agente 3",
        }
    ]


def append_discriminators_from_learned_rules(
    rules: list[dict[str, object]],
    feature_bank_path: Path | str,
) -> list[dict[str, object]]:
    payload = load_feature_bank(feature_bank_path)
    if not payload:
        payload = {
            "version": 1,
            "source": "agent2_discriminators",
            "discriminator_count": 0,
            "labels": [],
            "discriminators": [],
            "memories": {},
        }

    discriminators = payload.setdefault("discriminators", [])
    if not isinstance(discriminators, list):
        discriminators = []
        payload["discriminators"] = discriminators

    existing_by_signature = {
        str(item.get("marker_signature") or _marker_signature_from_discriminator(item)): item
        for item in discriminators
        if isinstance(item, dict)
    }
    added: list[dict[str, object]] = []
    for rule in rules:
        discriminator = learned_rule_to_discriminator(rule)
        if discriminator is None:
            continue
        signature = str(discriminator.get("marker_signature") or _marker_signature_from_discriminator(discriminator))
        existing = existing_by_signature.get(signature)
        if existing is not None:
            existing["confirmations"] = int(existing.get("confirmations", 0) or 0) + 1
            existing["last_confirmed_source"] = str(rule.get("source", "agent3_learned_discriminator"))
            _apply_strength_metadata(existing)
            added.append(existing)
            continue
        discriminators.append(discriminator)
        existing_by_signature[signature] = discriminator
        added.append(discriminator)

    labels = sorted(
        {
            canonical_label(str(item.get("label", "")))
            for item in discriminators
            if isinstance(item, dict) and item.get("label")
        }
    )
    themes: dict[str, dict[str, object]] = {}
    for label in labels:
        theme_discriminators = [
            item
            for item in discriminators
            if isinstance(item, dict) and canonical_label(str(item.get("label", ""))) == label
        ]
        themes[label] = {
            "canonical_theme": label,
            "discriminator_count": len(theme_discriminators),
            "discriminators": theme_discriminators,
        }
    payload["labels"] = labels
    payload["discriminator_count"] = len(discriminators)
    payload["themes"] = themes

    resolved = Path(feature_bank_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return added


def compact_feature_bank(feature_bank_path: Path | str) -> dict[str, int]:
    payload = load_feature_bank(feature_bank_path)
    discriminators = payload.get("discriminators", []) if isinstance(payload, dict) else []
    if not isinstance(discriminators, list):
        return {"before": 0, "after": 0, "removed": 0}

    compacted: list[dict[str, object]] = []
    seen: set[str] = set()
    weak_signals: list[dict[str, object]] = []
    rejected_by_guard: list[dict[str, object]] = []
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        original_label = canonical_label(str(item.get("subtheme") or item.get("label", "")))
        parent = parent_theme(original_label)
        tokens = item.get("tokens")
        if isinstance(tokens, list) and tokens:
            clean_tokens = [
                token
                for token in (canonical_label(str(value)) for value in tokens)
                if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
            ]
        else:
            clean_tokens = _tokens_from_regex_pattern(str(item.get("pattern", "")))
        if len(clean_tokens) < 2:
            item["tokens"] = clean_tokens
            item["weight"] = min(float(item.get("weight", WEAK_SIGNAL_WEIGHT) or WEAK_SIGNAL_WEIGHT), WEAK_SIGNAL_WEIGHT)
            item["strength"] = "weak"
            item["source"] = f"weak_signal:{item.get('source', '')}"
            weak_signals.append(item)
            continue
        guard_ok, guard_reason = _marker_matches_micro_world(parent, clean_tokens)
        if not guard_ok:
            item["guard_rejection"] = guard_reason
            rejected_by_guard.append(item)
            continue
        item["tokens"] = clean_tokens
        item["label"] = parent
        if parent != original_label:
            item["subtheme"] = original_label
            item["parent_theme"] = parent
        else:
            item.pop("subtheme", None)
            item.pop("parent_theme", None)
        signature = _marker_signature(parent, clean_tokens)
        if not signature:
            continue
        if signature in seen:
            continue
        item["marker_signature"] = signature
        item["pattern"] = _unordered_pattern(clean_tokens)
        _apply_strength_metadata(item)
        seen.add(signature)
        compacted.append(item)

    labels = sorted(
        {
            canonical_label(str(item.get("label", "")))
            for item in compacted
            if item.get("label")
        }
    )
    themes: dict[str, dict[str, object]] = {}
    for label in labels:
        theme_discriminators = [
            item
            for item in compacted
            if canonical_label(str(item.get("label", ""))) == label
        ]
        themes[label] = {
            "canonical_theme": label,
            "discriminator_count": len(theme_discriminators),
            "discriminators": theme_discriminators,
        }

    payload["discriminators"] = compacted
    payload["weak_signals"] = weak_signals
    payload["guard_rejected"] = rejected_by_guard
    payload["labels"] = labels
    payload["themes"] = themes
    payload["discriminator_count"] = len(compacted)
    payload["theme_parents"] = WNN_THEME_PARENTS

    resolved = Path(feature_bank_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "before": len(discriminators),
        "after": len(compacted),
        "removed": len(discriminators) - len(compacted),
        "weak_signals": len(weak_signals),
        "guard_rejected": len(rejected_by_guard),
    }


def _curated_discriminator(label: str, item: dict[str, object]) -> dict[str, object] | None:
    original_label = canonical_label(label)
    label = parent_theme(original_label)
    name = canonical_label(str(item.get("name", "")))
    raw_tokens = item.get("tokens", [])
    if not isinstance(raw_tokens, list):
        return None
    tokens = [
        token
        for token in (canonical_label(str(value)) for value in raw_tokens)
        if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
    ]
    if len(tokens) < 2:
        return None
    guard_ok, _guard_reason = _marker_matches_micro_world(label, tokens)
    if not guard_ok:
        return None
    pattern = _unordered_pattern(tokens)
    if not pattern:
        return None
    discriminator = {
        "id": _stable_id(label, _marker_signature(label, tokens)),
        "label": label,
        "name": name or "_".join(tokens),
        "pattern": pattern,
        "tokens": tokens,
        "marker_signature": _marker_signature(label, tokens),
        "weight": float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT),
        "source": "agent2_curated_discriminator",
        "rationale": f"discriminador substantivo do tema canonico {label}",
    }
    return _with_parent_theme(_apply_strength_metadata(discriminator), original_label)


def build_feature_bank(
    themes_payload: dict[str, object],
    sample: list[dict[str, object]],
    cluster_rows: pd.DataFrame,
    output_path: Path,
    max_discriminators_per_theme: int = 35,
) -> dict[str, object]:
    """Builds an auditable discriminator bank and a simple WNN-style associative memory."""

    discriminators_by_label: dict[str, list[dict[str, object]]] = {}
    seen: set[str] = set()

    for theme in themes_payload.get("themes", []):
        if not isinstance(theme, dict) or theme.get("decision") != "accept":
            continue
        original_label = canonical_label(str(theme.get("canonical_theme", "")))
        label = parent_theme(original_label)
        if not original_label:
            continue
        for curated in CURATED_THEME_DISCRIMINATORS.get(original_label, []):
            discriminator = _curated_discriminator(original_label, curated)
            if discriminator is None:
                continue
            key = str(discriminator.get("marker_signature") or _marker_signature_from_discriminator(discriminator))
            if key in seen:
                continue
            seen.add(key)
            discriminators_by_label.setdefault(label, []).append(discriminator)
        theme_terms: list[str] = []
        for value in theme.get("evidence_terms", []):
            text = str(value).strip()
            if text and text not in theme_terms:
                theme_terms.append(text)
        included_cluster_ids = [int(item) for item in theme.get("included_cluster_ids", []) if str(item).lstrip("-").isdigit()]
        if included_cluster_ids and "cluster_id" in cluster_rows.columns:
            subset = cluster_rows.loc[cluster_rows["cluster_id"].isin(included_cluster_ids)]
            for column in ("cluster_domain_terms", "cluster_text"):
                if column not in subset.columns:
                    continue
                for raw_value in subset[column].fillna("").astype(str).head(200).tolist():
                    separators = r"\s*\|\s*|[,;]\s*"
                    for part in re.split(separators, raw_value):
                        text = part.replace("_", " ").strip()
                        if text and len(text) >= 5 and text not in theme_terms:
                            theme_terms.append(text)
                        if len(theme_terms) >= 250:
                            break
                    if len(theme_terms) >= 250:
                        break
                if len(theme_terms) >= 250:
                    break
        for tokens in _agent2_generalized_marker_sets(original_label, theme_terms):
            guard_ok, _guard_reason = _marker_matches_micro_world(label, tokens)
            if not guard_ok:
                continue
            key = _marker_signature(label, tokens)
            if not key or key in seen:
                continue
            seen.add(key)
            discriminator = {
                "id": _stable_id(label, key),
                "label": label,
                "name": "_".join(tokens[:4]),
                "pattern": _unordered_pattern(tokens),
                "tokens": tokens,
                "marker_signature": key,
                "weight": EVIDENCE_FEATURE_WEIGHT,
                "source": "agent2_generalized_micro_world",
                "rationale": f"marcador generalizado pelo Agente 2 para o micromundo do tema {label}",
            }
            discriminators_by_label.setdefault(label, []).append(_with_parent_theme(_apply_strength_metadata(discriminator), original_label))
        for term in theme.get("evidence_terms", []):
            term_text = str(term).strip()
            if not _is_sensor_candidate(term_text):
                continue
            pattern = _literal_pattern(term_text)
            if not pattern:
                continue
            tokens = _tokens_from_text(term_text)
            key = _marker_signature(label, tokens)
            if key in seen:
                continue
            seen.add(key)
            discriminator = {
                    "id": _stable_id(label, key),
                    "label": label,
                    "pattern": pattern,
                    "tokens": tokens,
                    "marker_signature": key,
                    "weight": EVIDENCE_FEATURE_WEIGHT,
                    "source": "agent1_evidence_term",
                    "rationale": f"termo de evidencia do tema {label}: {term_text}",
                }
            discriminators_by_label.setdefault(label, []).append(_with_parent_theme(_apply_strength_metadata(discriminator), original_label))

    discriminators: list[dict[str, object]] = []
    for label, items in sorted(discriminators_by_label.items()):
        discriminators.extend(items[:max_discriminators_per_theme])

    docs_by_name = {item["arquivo"]: item for item in sample}
    label_by_doc: dict[str, str] = {}
    for theme in themes_payload.get("themes", []):
        if not isinstance(theme, dict) or theme.get("decision") != "accept":
            continue
        label = parent_theme(str(theme.get("canonical_theme", "")))
        for cluster_id in theme.get("included_cluster_ids", []):
            if "cluster_id" not in cluster_rows.columns:
                continue
            names = cluster_rows.loc[cluster_rows["cluster_id"] == cluster_id, "arquivo"].tolist()
            for name in names:
                if name in docs_by_name and name not in label_by_doc:
                    label_by_doc[name] = label

    memories: dict[str, dict[str, object]] = {}
    for name, label in label_by_doc.items():
        doc = docs_by_name.get(name)
        if not doc:
            continue
        parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
        text = str(doc.get("body_text", "") or parsed.get("corpo", "") or doc.get("context", ""))
        words = _token_set_from_text(text)
        active_ids = sorted(
            str(item["id"])
            for item in discriminators
            if _tokens_match_text(item.get("tokens", []), words)
        )
        if not active_ids:
            continue
        memory = memories.setdefault(label, {"sample_count": 0, "active_counts": {}, "signatures": []})
        memory["sample_count"] = int(memory.get("sample_count", 0) or 0) + 1
        active_counts = memory.setdefault("active_counts", {})
        if isinstance(active_counts, dict):
            for feature_id in active_ids:
                active_counts[feature_id] = int(active_counts.get(feature_id, 0) or 0) + 1
        signatures = memory.setdefault("signatures", [])
        signature = " ".join(active_ids)
        if isinstance(signatures, list) and signature not in signatures and len(signatures) < MAX_SIGNATURES_PER_LABEL:
            signatures.append(signature)

    payload = {
        "version": 1,
        "source": "agent2_discriminators",
        "discriminator_count": len(discriminators),
        "labels": sorted(discriminators_by_label),
        "themes": {
            label: {
                "canonical_theme": label,
                "discriminator_count": len(items[:max_discriminators_per_theme]),
                "discriminators": items[:max_discriminators_per_theme],
            }
            for label, items in sorted(discriminators_by_label.items())
        },
        "discriminators": discriminators,
        "memories": memories,
        "theme_parents": WNN_THEME_PARENTS,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_feature_bank(path: Path | str) -> dict[str, object]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def active_discriminators(text: str, feature_bank: dict[str, object]) -> list[dict[str, object]]:
    words = _token_set_from_text(text)
    active: list[dict[str, object]] = []
    for item in feature_bank.get("discriminators", []):
        if not isinstance(item, dict):
            continue
        raw_tokens = item.get("tokens", [])
        if not isinstance(raw_tokens, list) or not _tokens_match_text(raw_tokens, words):
            continue
        active.append(
            {
                "id": str(item.get("id", "")),
                "label": parent_theme(str(item.get("label", ""))),
                "subtheme": canonical_label(str(item.get("subtheme", ""))),
                "tokens": item.get("tokens", []) if isinstance(item.get("tokens", []), list) else [],
                "weight": float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT),
                "strength": str(item.get("strength", "medium") or "medium"),
                "confirmations": int(item.get("confirmations", 0) or 0),
                "source": str(item.get("source", "")),
            }
        )
    return active


def _label_evidence_counts(active: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in active:
        label = _active_label(item)
        strength = str(item.get("strength", "medium") or "medium")
        bucket = counts.setdefault(label, {"strong": 0, "medium": 0, "weak": 0, "confirmed": 0})
        if strength not in {"strong", "medium", "weak"}:
            strength = "medium"
        bucket[strength] += 1
        if int(item.get("confirmations", 0) or 0) >= CONFIRMED_MEDIUM_THRESHOLD:
            bucket["confirmed"] += 1
    return counts


def _has_enough_marker_evidence(
    top_label: str,
    secondary: list[str],
    active: list[dict[str, object]],
    cosine_supported: bool,
    memory_supported: bool,
) -> bool:
    counts = _label_evidence_counts(active)
    labels = [top_label, *secondary]
    total = {"strong": 0, "medium": 0, "weak": 0, "confirmed": 0}
    for label in labels:
        for key, value in counts.get(label, {}).items():
            total[key] = total.get(key, 0) + value
    if total["strong"] >= 1:
        return True
    if total["medium"] >= 2:
        return True
    if total["medium"] >= 1 and (cosine_supported or memory_supported or total["confirmed"] >= 1):
        return True
    return False


def classify_with_wnn(
    text: str,
    feature_bank_path: Path | str,
    confidence_threshold: float = 0.50,
    margin_threshold: float = 0.12,
    min_active_discriminators: int = 2,
    cosine_candidates: list[dict[str, object]] | None = None,
) -> WNNClassification:
    feature_bank = load_feature_bank(feature_bank_path)
    active = active_discriminators(text, feature_bank)
    if len(active) < min_active_discriminators:
        return WNNClassification(None, "abstain_insufficient_features", 0.0, 0.0, "", active, [], str(feature_bank_path))

    active_ids = {str(item["id"]) for item in active}
    scores_by_label: dict[str, float] = {}
    for item in active:
        label = _active_label(item)
        scores_by_label[label] = scores_by_label.get(label, 0.0) + float(item["weight"])

    memory_scores_by_label: dict[str, float] = {}
    for candidate in cosine_candidates or []:
        if not isinstance(candidate, dict):
            continue
        label = canonical_label(str(candidate.get("label", "")))
        try:
            cosine_score = float(candidate.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            cosine_score = 0.0
        if not label or cosine_score < 0.08:
            continue
        scores_by_label[label] = scores_by_label.get(label, 0.0) + min(0.9, cosine_score * 1.4)

    memories = feature_bank.get("memories", {})
    if isinstance(memories, dict):
        for label, memory in memories.items():
            if not isinstance(memory, dict):
                continue
            active_counts = memory.get("active_counts", {})
            if not isinstance(active_counts, dict):
                continue
            sample_count = max(1, int(memory.get("sample_count", 0) or 0))
            memory_score = 0.0
            for feature_id in active_ids:
                memory_score += min(1.0, float(active_counts.get(feature_id, 0) or 0) / sample_count)
            if memory_score:
                normalized_label = parent_theme(str(label))
                scores_by_label[normalized_label] = scores_by_label.get(normalized_label, 0.0) + memory_score
                memory_scores_by_label[normalized_label] = memory_scores_by_label.get(normalized_label, 0.0) + memory_score

    scores = [
        {"label": label, "score": round(score, 4)}
        for label, score in sorted(scores_by_label.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ]
    if not scores:
        return WNNClassification(None, "abstain_no_score", 0.0, 0.0, "", active, [], str(feature_bank_path))

    top = scores[0]
    second_score = float(scores[1]["score"]) if len(scores) > 1 else 0.0
    total_score = sum(float(item["score"]) for item in scores)
    top_score = float(top["score"])
    top_label = canonical_label(str(top["label"]))
    operational_label, crimes, secondary, relation = _operational_decision(scores_by_label, active)
    if operational_label and operational_label != top_label:
        top_label = operational_label
    theme_candidate = _multi_discriminator_candidate(scores_by_label, active, relation)

    if relation in {"cadeia_operacional", "crime_organizado_multidominio"} and top_label == "crime_organizado":
        operational_set = set(crimes)
        top_score = sum(float(scores_by_label.get(label, 0.0) or 0.0) for label in operational_set)
        second_score = max(
            [float(score) for label, score in scores_by_label.items() if label not in operational_set] or [0.0]
        )
    elif relation in {"dominio_preferencial", "dominio_preferencial_com_organizacao"}:
        preferred_set = set(crimes or [top_label])
        top_score = sum(float(scores_by_label.get(label, 0.0) or 0.0) for label in preferred_set)
        second_score = max(
            [float(score) for label, score in scores_by_label.items() if label not in preferred_set] or [0.0]
        )
    confidence = top_score / total_score if total_score else 0.0
    margin = (top_score - second_score) / top_score if top_score else 0.0
    cosine_supported = _cosine_supports_decision(top_label, secondary, cosine_candidates)
    memory_supported = any(memory_scores_by_label.get(label, 0.0) >= 0.25 for label in [top_label, *secondary])
    cosine_suspect, cosine_suspect_label, cosine_suspect_score = _cosine_suspicion(
        top_label,
        secondary,
        cosine_candidates,
    )
    effective_margin_threshold = (
        min(margin_threshold, COSINE_ASSISTED_MARGIN_THRESHOLD)
        if cosine_supported
        else margin_threshold
    )

    if cosine_suspect:
        return WNNClassification(
            None,
            f"abstain_cosine_suspicion:{cosine_suspect_label}:{cosine_suspect_score:.4f}",
            confidence,
            margin,
            top_label,
            active,
            scores,
            str(feature_bank_path),
            theme_candidate,
        )

    if not _has_enough_marker_evidence(top_label, secondary, active, cosine_supported, memory_supported):
        return WNNClassification(
            None,
            "abstain_weak_marker_evidence",
            confidence,
            margin,
            top_label,
            active,
            scores,
            str(feature_bank_path),
            theme_candidate,
        )

    if confidence < confidence_threshold or margin < effective_margin_threshold:
        return WNNClassification(
            None,
            "abstain_ambiguous" if not cosine_supported else "abstain_cosine_supported_but_low_confidence",
            confidence,
            margin,
            top_label,
            active,
            scores,
            str(feature_bank_path),
            theme_candidate,
        )

    identity = top_label if top_label.startswith(("crime_", "crimes_")) else f"crime_{top_label}"
    if not crimes:
        crimes = [top_label]
    inference = NoticiaLLMInference(
        identidade_canonica=identity,
        classificacao="Por crime",
        crimes_mais_presentes=crimes,
        tema_principal=top_label,
        marcadores_secundarios=secondary,
        relacao_operacional=relation,
        modus_operandi=[],
    )
    return WNNClassification(
        inference,
        "accepted",
        confidence,
        margin,
        top_label,
        active,
        scores,
        str(feature_bank_path),
        theme_candidate,
    )
