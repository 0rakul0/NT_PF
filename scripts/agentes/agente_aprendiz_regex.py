from __future__ import annotations

import re
from typing import Any

from scripts.agentes.agente2_regex import pattern_has_crime_modus_anchor, validate_regex
from scripts.incremental.common import ACTIVE_REGEX_BANK_PATH, append_event
from scripts.pf_llm_models import NoticiaLLMInference
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse

try:
    from scripts.pf_regex_classifier import append_learned_rule, clean_learned_rules_file, fold_text, suggest_regex_from_llm
except ModuleNotFoundError:
    from pf_regex_classifier import append_learned_rule, clean_learned_rules_file, fold_text, suggest_regex_from_llm


RESIDUAL_REGEX_STOPWORDS = {
    "policia",
    "federal",
    "operacao",
    "contra",
    "para",
    "com",
    "houve",
    "meio",
    "uso",
    "documentos",
    "falsos",
}

RESIDUAL_LABEL_KEYWORDS = {
    "armas_municoes": {"arma", "armas", "armamento", "armamentos", "fogo", "municao", "municoes"},
    "radiodifusao_clandestina": {"radio", "radios", "radiodifusao", "anatel", "telecomunicacao", "telecomunicacoes", "clandestina"},
    "trafico_drogas": {"trafico", "drogas", "droga", "maconha", "cocaina", "entorpecente", "entorpecentes"},
    "crimes_contra_criancas": {"infantil", "infantojuvenil", "crianca", "criancas", "adolescente", "adolescentes", "pornografia", "abuso"},
    "contrabando_descaminho": {"contrabando", "descaminho", "cigarro", "cigarros", "mercadoria", "mercadorias"},
    "crimes_ambientais": {"garimpo", "madeira", "desmatamento", "ambiental", "ambientais", "indigena", "ouro"},
}


def review_to_inference(review: ResidualReviewAgentResponse) -> NoticiaLLMInference:
    return NoticiaLLMInference(
        identidade_canonica=review.canonical_label,
        classificacao="Por crime",
        crimes_mais_presentes=[review.canonical_label],
        modus_operandi=[],
        resumo_curto=review.resumo_curto,
        resumo_estruturado={},
        evidencia_textual=review.evidence_text,
        atores_mencionados=[],
        setor_afetado="",
        precisa_reprocessamento=review.decision != "classificar",
    )


def residual_tokens(term: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]{4,}", fold_text(term)) if token not in RESIDUAL_REGEX_STOPWORDS]


def residual_term_pattern(term: str) -> str:
    tokens = residual_tokens(term)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return rf"\b{re.escape(tokens[0])}\w*\b"
    return r".{0,80}".join(rf"\b{re.escape(token)}\w*\b" for token in tokens[:5])


def evidence_phrases(text: str) -> list[str]:
    folded = fold_text(text)
    phrases: list[str] = []
    for size in (6, 5, 4, 3, 2):
        for match in re.finditer(r"\b[a-z0-9]{4,}(?:\s+[a-z0-9]{2,}){" + str(size - 1) + r"}\b", folded):
            phrase = match.group(0)
            tokens = residual_tokens(phrase)
            if len(tokens) < 2:
                continue
            if phrase.startswith(("policia federal", "operacao")):
                continue
            if phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= 10:
                return phrases
    return phrases


def phrase_matches_label(phrase: str, label: str) -> bool:
    keywords = RESIDUAL_LABEL_KEYWORDS.get(label)
    if not keywords:
        return True
    tokens = set(residual_tokens(phrase))
    return bool(tokens.intersection(keywords))


def agent_regex_learning_suggestions(doc: dict[str, Any], review: ResidualReviewAgentResponse) -> list[dict[str, str]]:
    suggestions = suggest_regex_from_llm(doc["context"], review_to_inference(review))
    if suggestions:
        return suggestions

    if review.canonical_label == "radiodifusao_clandestina":
        return [
            {"kind": "crime", "label": review.canonical_label, "pattern": r"\banatel\w*\b.{0,120}\bclandestina\w*\b"},
            {"kind": "crime", "label": review.canonical_label, "pattern": r"\bclandestina\w*\b.{0,120}\bautori\w*\b"},
        ]

    evidence_parts = [review.evidence_text, review.resumo_curto, doc["titulo"]]
    phrase_candidates = [
        phrase
        for phrase in evidence_phrases(" | ".join(part for part in evidence_parts if part))
        if phrase_matches_label(phrase, review.canonical_label)
    ]
    output: list[dict[str, str]] = []
    seen_patterns: set[str] = set()
    for phrase in phrase_candidates:
        pattern = residual_term_pattern(phrase)
        if not pattern or pattern in seen_patterns:
            continue
        seen_patterns.add(pattern)
        output.append({"kind": "crime", "label": review.canonical_label, "pattern": pattern})
        if len(output) >= 3:
            break
    return output


def generate_regex_from_review(doc: dict[str, Any], review: ResidualReviewAgentResponse, negatives: list[str]) -> list[dict[str, Any]]:
    canonical_label = review.canonical_label
    if not canonical_label:
        append_event({"stage": "agente_aprendiz_regex", "arquivo": doc["arquivo"], "decision": "quarentena", "reason": "sem_label_canonico"})
        return []
    suggestions = agent_regex_learning_suggestions(doc, review)
    incorporated: list[dict[str, Any]] = []
    for suggestion in suggestions:
        suggestion["label"] = canonical_label
        if not pattern_has_crime_modus_anchor(canonical_label, suggestion["pattern"]):
            record = suggestion | {"validation": "sem ancora suficiente de crime/modus da label", "decision": "quarentena"}
            append_event({"stage": "agente_aprendiz_regex", "arquivo": doc["arquivo"], "candidate": record})
            continue
        ok, reason = validate_regex(suggestion["pattern"], [doc["context"]], negatives)
        record = suggestion | {"validation": reason, "decision": "incorporar" if ok else "quarentena"}
        if ok and append_learned_rule(
            suggestion["kind"],
            canonical_label,
            suggestion["pattern"],
            title=doc["titulo"],
            path=ACTIVE_REGEX_BANK_PATH,
            source="agent_regex_learning_from_agent3_review",
        ):
            incorporated.append(record)
        else:
            append_event({"stage": "agente_aprendiz_regex", "arquivo": doc["arquivo"], "candidate": record})
    if incorporated:
        clean_learned_rules_file(ACTIVE_REGEX_BANK_PATH)
    return incorporated
