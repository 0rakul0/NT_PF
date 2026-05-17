from __future__ import annotations

import json
import re
from typing import Any

from scripts.incremental.common import NEW_THEME_CANDIDATES_JSONL, RARE_NEWS_LABEL, RunConfig
from scripts.incremental.llm_api import invoke_json_with_fallback
from scripts.pf_llm_models import normalize_slug
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse

try:
    from scripts.pf_regex_classifier import fold_text
except ModuleNotFoundError:
    from pf_regex_classifier import fold_text


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
    return ""


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


def rare_news_review(review: ResidualReviewAgentResponse, rationale: str | None = None) -> ResidualReviewAgentResponse:
    reason = rationale or review.rationale or "Residual sem encaixe defensavel nos temas canonicos ou nos candidatos substantivos."
    return review.model_copy(
        update={
            "decision": "classificar",
            "canonical_label": RARE_NEWS_LABEL,
            "rationale": f"Classificado como noticia rara: {reason}",
        }
    )


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
- use "{RARE_NEWS_LABEL}" quando nenhuma label canonica for defensavel e tambem nao houver tema claro; nesse caso retorne decision="classificar" e canonical_label="{RARE_NEWS_LABEL}";
- use "quarentena" apenas para texto insuficiente, corrompido ou impossivel de analisar;
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
            return rare_news_review(review, "Novo tema candidato sem label valida."), provider, model_name
        return review.model_copy(update={"canonical_label": candidate_label}), provider, model_name
    if review.decision != "classificar":
        return rare_news_review(review), provider, model_name
    canonical_label = map_to_canonical_label([review.canonical_label], allowed_labels)
    if not canonical_label:
        return rare_news_review(review, "Label retornada fora da lista canonica permitida."), provider, model_name
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
