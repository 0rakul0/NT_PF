from __future__ import annotations

import json
import re
from typing import Any

from scripts.incremental.common import NEW_THEME_CANDIDATES_JSONL, RARE_NEWS_LABEL, RunConfig
from scripts.incremental.llm_api import TokenUsage, ZERO_TOKEN_USAGE, invoke_json_with_fallback
from scripts.pf_llm_models import normalize_slug
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse
from scripts.agentes.agente1_temas import normalize_theme_label

try:
    from scripts.pf_regex_classifier import fold_text
except ModuleNotFoundError:
    from pf_regex_classifier import fold_text


def body_text(doc: dict[str, Any]) -> str:
    parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
    return str(doc.get("body_text", "") or parsed.get("corpo", "") or doc.get("context", "")).strip()


def map_to_canonical_label(inference_labels: list[str], allowed_labels: list[str]) -> str:
    folded = {normalize_slug(label): label for label in allowed_labels}
    for label in inference_labels:
        normalized_label = normalize_slug(label)
        alias = normalize_theme_label(normalized_label)
        if alias in folded:
            return folded[alias]
        if normalized_label in folded:
            return folded[normalized_label]
    for allowed in allowed_labels:
        allowed_terms = {term for term in allowed.lower().split("_") if len(term) >= 4}
        for label in inference_labels:
            normalized_label = normalize_slug(label)
            if allowed_terms.intersection({term for term in normalized_label.split("_") if len(term) >= 4}):
                return allowed
    return ""


def map_secondary_labels(labels: list[str], allowed_labels: list[str], primary: str) -> list[str]:
    mapped: list[str] = []
    for label in labels:
        canonical = map_to_canonical_label([label], allowed_labels)
        if canonical and canonical != primary and canonical not in mapped:
            mapped.append(canonical)
    return mapped[:8]


def deterministic_new_theme_candidate(doc: dict[str, Any], allowed_labels: list[str]) -> ResidualReviewAgentResponse | None:
    text = fold_text(body_text(doc)[:1600])
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
) -> tuple[ResidualReviewAgentResponse, str, str, TokenUsage]:
    deterministic = deterministic_new_theme_candidate(doc, allowed_labels)
    if deterministic is not None:
        return deterministic, "deterministic", "theme_candidate_rules", ZERO_TOKEN_USAGE

    labels_block = "\n".join(f"- {label}" for label in allowed_labels)
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
- quando houver varios marcadores, nao force fusao automaticamente:
  * dominios sensiveis/materialmente especificos tem preferencia sobre crime_organizado quando houver evidencia direta: crimes_contra_criancas vence se houver infantil/crianca/adolescente/menor/pornografia/abuso/estupro/exploracao; crimes_ambientais vence se houver mineracao/garimpo/madeira/desmatamento/extracao/ouro/animais silvestres; trabalho_escravo vence se houver trabalho analogo/condicoes analogas/trabalhadores resgatados;
  * trafico de drogas + organizacao/faccao/associacao criminosa deve virar canonical_label="crime_organizado" com marcadores_secundarios=["trafico_drogas"];
  * trafico de drogas sem ponte organizacional deve ficar em canonical_label="trafico_drogas";
  * trafico de drogas + mineracao/garimpo/madeira/desmatamento/extracao deve ficar em canonical_label="crimes_ambientais", com trafico_drogas como marcador secundario se relevante;
  * crime_organizado + extracao de madeira/garimpo/mineracao/desmatamento deve ficar em canonical_label="crimes_ambientais", com crime_organizado como marcador secundario;
  * se houver relacao operacional clara entre organizacao/faccao/associacao, trafico, lavagem, armas ou outros eixos sem dominio preferencial mais especifico, use canonical_label="crime_organizado", relacao_operacional="crime_organizado_multidominio" ou "cadeia_operacional";
  * se houver dominios distintos sem ponte operacional clara, escolha o tema dominante em canonical_label, preencha marcadores_secundarios e use relacao_operacional="coocorrencia_sem_fusao";
  * exemplo: mineracao ilegal + trafico de drogas sem grupo organizado explicito nao deve virar crime_organizado automaticamente;
  * exemplo: trafico de drogas + lavagem de dinheiro + organizacao criminosa deve virar crime_organizado com marcadores_secundarios=["trafico_drogas","lavagem_dinheiro"];
- analise somente o corpo da noticia informado em Texto; ignore titulo, tags, slug do arquivo e metadados externos;
- responda somente um objeto JSON valido com as chaves:
  decision, canonical_label, confidence, evidence_text, rationale, resumo_curto, tema_principal, marcadores_secundarios, relacao_operacional;
- se decision for novo_tema_candidato, canonical_label deve ser uma nova label em lowercase_com_underscores.

Labels canonicas permitidas:
{labels_block}

Sugestoes por similaridade do cosseno:
{cosine_block}

Texto:
{body_text(doc)[:1800]}
""".strip()
    review, provider, model_name, token_usage = invoke_json_with_fallback(prompt, ResidualReviewAgentResponse, config, "agente3_review")
    if review.decision == "novo_tema_candidato":
        candidate_label = normalize_slug(review.canonical_label)
        if not candidate_label:
            return rare_news_review(review, "Novo tema candidato sem label valida."), provider, model_name, token_usage
        return review.model_copy(update={"canonical_label": candidate_label}), provider, model_name, token_usage
    if review.decision != "classificar":
        return rare_news_review(review), provider, model_name, token_usage
    canonical_label = map_to_canonical_label([review.canonical_label], allowed_labels)
    if not canonical_label:
        return rare_news_review(review, "Label retornada fora da lista canonica permitida."), provider, model_name, token_usage
    secondary = map_secondary_labels(review.marcadores_secundarios, allowed_labels, canonical_label)
    relation = review.relacao_operacional or ("coocorrencia_sem_fusao" if secondary else "tema_unico")
    return review.model_copy(
        update={
            "canonical_label": canonical_label,
            "tema_principal": review.tema_principal or canonical_label,
            "marcadores_secundarios": secondary,
            "relacao_operacional": relation,
        }
    ), provider, model_name, token_usage


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
                    "tema_principal": review.tema_principal,
                    "marcadores_secundarios": review.marcadores_secundarios,
                    "relacao_operacional": review.relacao_operacional,
                    "resumo_curto": review.resumo_curto,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
