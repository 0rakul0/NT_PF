from __future__ import annotations

import json
import re
from typing import Any

from scripts.agentes.agente2_regex import pattern_has_crime_modus_anchor, validate_regex
from scripts.incremental.common import ACTIVE_REGEX_BANK_PATH, NEW_THEME_CANDIDATES_JSONL, RunConfig, append_event
from scripts.incremental.llm_api import invoke_json_with_fallback
from scripts.pf_llm_models import NoticiaLLMInference, normalize_slug
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse

try:
    from scripts.pf_regex_classifier import append_learned_rule, clean_learned_rules_file, fold_text, suggest_regex_from_llm
except ModuleNotFoundError:
    from pf_regex_classifier import append_learned_rule, clean_learned_rules_file, fold_text, suggest_regex_from_llm


def map_to_canonical_label(inference_labels: list[str], allowed_labels: list[str]) -> str:
    folded = {normalize_slug(label): label for label in allowed_labels}
    for label in inference_labels:
        normalized_label = normalize_slug(label)
        if normalized_label in folded:
            return folded[normalized_label]
    for allowed in allowed_labels:
        allowed_terms = {term for term in allowed.lower().split("_") if len(term) >= 4}
        for label in inference_labels:
            normalized_label = normalize_slug(label)
            if allowed_terms.intersection({term for term in normalized_label.split("_") if len(term) >= 4}):
                return allowed
    return allowed_labels[0] if allowed_labels else ""


def deterministic_new_theme_candidate(doc: dict[str, Any], allowed_labels: list[str]) -> ResidualReviewAgentResponse | None:
    text = fold_text(" ".join([doc.get("titulo", ""), " ".join(str(tag) for tag in doc.get("parsed", {}).get("tags", [])), doc.get("context", "")[:1200]]))
    if "radiodifusao_clandestina" not in allowed_labels and re.search(r"\b(radio|radios|radiodifusao|anatel)\w*\b", text):
        return ResidualReviewAgentResponse(
            decision="novo_tema_candidato",
            canonical_label="radiodifusao_clandestina",
            confidence=0.9,
            evidence_text="radio clandestina; anatel; radiodifusao sem autorizacao",
            rationale="Residual contem sinais substantivos de radio/radiodifusao clandestina e nao ha label canonica adequada na fundacao.",
            resumo_curto="Noticia sobre atividade de radiodifusao clandestina ou radio irregular.",
        )
    return None


def review_residual(
    doc: dict[str, Any],
    allowed_labels: list[str],
    config: RunConfig,
    cosine_candidates: list[dict[str, Any]],
) -> tuple[ResidualReviewAgentResponse, str, str]:
    deterministic = deterministic_new_theme_candidate(doc, allowed_labels)
    if deterministic is not None:
        return deterministic, "deterministic", "theme_candidate_rules"

    labels_block = "\n".join(f"- {label}" for label in allowed_labels)
    tags = ", ".join(str(tag) for tag in doc.get("parsed", {}).get("tags", []) if str(tag).strip()) or "sem_tags"
    cosine_block = "\n".join(
        f"- {item['label']} score={item['score']} clusters={item.get('cluster_ids', [])} termos={', '.join(item.get('top_terms', [])[:6])}"
        for item in cosine_candidates
    ) or "sem_sugestoes"
    prompt = f"""
Voce e o Agente 3 de revisao residual.

Tarefa:
- classifique a noticia usando exclusivamente uma das labels canonicas abaixo, geradas pelo Agente 1;
- use "novo_tema_candidato" quando nenhuma label canonica for defensavel, mas houver tema substantivo claro;
- use "quarentena" quando nenhuma label canonica for defensavel e tambem nao houver tema claro;
- escolha a label pelo sentido substantivo do caso, nao apenas por palavra solta;
- use as sugestoes por similaridade do cosseno como apoio, nao como verdade obrigatoria;
- responda somente um objeto JSON valido com as chaves:
  decision, canonical_label, confidence, evidence_text, rationale, resumo_curto;
- se decision for novo_tema_candidato, canonical_label deve ser uma nova label em lowercase_com_underscores.

Labels canonicas permitidas:
{labels_block}

Sugestoes por similaridade do cosseno:
{cosine_block}

Titulo:
{doc["titulo"]}

Tags:
{tags}

Texto:
{doc["context"][:1400]}
""".strip()
    review, provider, model_name = invoke_json_with_fallback(prompt, ResidualReviewAgentResponse, config, "agente3_review")
    if review.decision == "novo_tema_candidato":
        candidate_label = normalize_slug(review.canonical_label)
        if not candidate_label:
            return review.model_copy(update={"decision": "quarentena", "canonical_label": "", "rationale": "Novo tema candidato sem label valida."}), provider, model_name
        return review.model_copy(update={"canonical_label": candidate_label}), provider, model_name
    if review.decision != "classificar":
        return review, provider, model_name
    canonical_label = map_to_canonical_label([review.canonical_label], allowed_labels)
    if not canonical_label:
        quarantined = ResidualReviewAgentResponse(
            decision="quarentena",
            canonical_label="",
            confidence=review.confidence,
            evidence_text=review.evidence_text,
            rationale="Label retornada fora da lista canonica permitida.",
            resumo_curto=review.resumo_curto,
        )
        return quarantined, provider, model_name
    return review.model_copy(update={"canonical_label": canonical_label}), provider, model_name


def append_new_theme_candidate(doc: dict[str, Any], review: ResidualReviewAgentResponse, iteration: int) -> None:
    NEW_THEME_CANDIDATES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with NEW_THEME_CANDIDATES_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "iteration": iteration,
                    "arquivo": doc["arquivo"],
                    "titulo": doc["titulo"],
                    "canonical_label": review.canonical_label,
                    "confidence": review.confidence,
                    "evidence_text": review.evidence_text,
                    "rationale": review.rationale,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


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


def agent2_residual_suggestions(doc: dict[str, Any], review: ResidualReviewAgentResponse) -> list[dict[str, str]]:
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
        append_event({"stage": "agente2_regex_pos_agent3", "arquivo": doc["arquivo"], "decision": "quarentena", "reason": "sem_label_canonico"})
        return []
    suggestions = agent2_residual_suggestions(doc, review)
    incorporated: list[dict[str, Any]] = []
    for suggestion in suggestions:
        suggestion["label"] = canonical_label
        if not pattern_has_crime_modus_anchor(canonical_label, suggestion["pattern"]):
            record = suggestion | {"validation": "sem ancora suficiente de crime/modus da label", "decision": "quarentena"}
            append_event({"stage": "agente2_regex_pos_agent3", "arquivo": doc["arquivo"], "candidate": record})
            continue
        ok, reason = validate_regex(suggestion["pattern"], [doc["context"]], negatives)
        record = suggestion | {"validation": reason, "decision": "incorporar" if ok else "quarentena"}
        if ok and append_learned_rule(
            suggestion["kind"],
            canonical_label,
            suggestion["pattern"],
            title=doc["titulo"],
            path=ACTIVE_REGEX_BANK_PATH,
            source="agent2_from_agent3_review",
        ):
            incorporated.append(record)
        else:
            append_event({"stage": "agente2_regex_pos_agent3", "arquivo": doc["arquivo"], "candidate": record})
    if incorporated:
        clean_learned_rules_file(ACTIVE_REGEX_BANK_PATH)
    return incorporated
