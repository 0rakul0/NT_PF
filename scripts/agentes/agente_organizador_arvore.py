from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from scripts.incremental.common import (
    ACTIVE_REGEX_BANK_PATH,
    NEW_THEME_CANDIDATES_JSONL,
    RARE_NEWS_DESCRIPTION,
    RARE_NEWS_LABEL,
    REFINED_THEME_TREE_JSON,
    THEME_REFINEMENT_INPUT_JSON,
    THEMES_JSON,
    RunConfig,
    append_event,
    read_json,
    read_jsonl,
    write_json,
)
from scripts.incremental.llm_api import invoke_json_with_fallback
from scripts.incremental.similaridade_cosseno import top_k_similar_themes
from scripts.schemas.pf_incremental_agent_schemas import ThemeCandidateDecision, ThemeTreeRefinementResponse


WEAK_THEME_TOKENS = {"crime", "crimes", "contra", "publico", "publica", "publicos", "ilegal", "ilegais", "fraude", "fraudes"}

EXISTING_THEME_ABSORPTION = {
    "crimes_ambientais": {
        "ambientais",
        "ambiental",
        "animais",
        "fauna",
        "grilagem",
        "indigenas",
        "indigena",
        "pesca",
        "reforma",
        "terra",
        "terras",
    },
    "crimes_ciberneticos": {
        "ciberneticos",
        "comunicacao",
        "dados",
        "equipamentos",
        "internet",
        "sigilosas",
        "sinal",
        "transmissao",
        "tv",
        "vazamento",
    },
    "crimes_eleitorais": {"eleitoral", "eleitorais", "eleicoes", "politico", "politica"},
    "crimes_migratorios": {"migrantes", "migratoria", "regularizacao", "refugiados", "extradicao"},
    "crimes_sistema_financeiro": {"cambio", "capitais", "divisas", "financeiro", "mercado", "seguros"},
    "contrabando_descaminho": {"descaminho", "contrabando", "mercadorias", "encomendas"},
    "corrupcao_desvio_recursos_publicos": {
        "cotas",
        "fiscalizacao",
        "imoveis",
        "publicos",
        "universitarias",
    },
}

PROMOTION_FAMILIES = {
    "falsificacao_documental": {
        "diplomas",
        "documental",
        "documentos",
        "falsidade",
        "falsificacao",
        "ideologica",
    },
    "crimes_patrimoniais": {
        "assalto",
        "bancarios",
        "bens",
        "correios",
        "furto",
        "furtos",
        "patrimonio",
        "receptacao",
        "roubo",
    },
    "crimes_contra_saude_publica": {
        "bebidas",
        "medicamentos",
        "medicamento",
        "produto",
        "produtos",
        "quimicos",
        "saude",
        "tatuagem",
        "terapeutico",
        "terapeuticos",
    },
    "seguranca_privada_clandestina": {"privada", "seguranca", "vigilante", "vigilantes"},
    "crimes_de_odio_e_extremismo": {
        "discriminacao",
        "extremistas",
        "ideologias",
        "nazistas",
        "odio",
        "racismo",
    },
    "ameacas_e_terrorismo": {"ameacas", "atentado", "preparatorios", "terrorismo", "terroristas"},
}


def _tokens(label: str) -> set[str]:
    return {token for token in label.split("_") if token and token not in WEAK_THEME_TOKENS}


def _regex_by_label() -> dict[str, list[dict[str, Any]]]:
    if not ACTIVE_REGEX_BANK_PATH.exists():
        return {}
    payload = read_json(ACTIVE_REGEX_BANK_PATH)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload if isinstance(payload, list) else []:
        label = str(item.get("classificador", "") or item.get("label", "")).strip()
        if not label:
            continue
        for pattern in item.get("patterns_do_classificador", []) or []:
            if isinstance(pattern, dict):
                grouped[label].append(
                    {
                        "pattern": pattern.get("pattern_do_classificador", ""),
                        "source": pattern.get("source", item.get("source", "")),
                        "uses": pattern.get("uses", 0),
                        "examples": pattern.get("examples", [])[:3],
                    }
                )
    return grouped


def _candidate_groups() -> list[dict[str, Any]]:
    rows = read_jsonl(NEW_THEME_CANDIDATES_JSONL)
    regex_lookup = _regex_by_label()
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("canonical_label", "")).strip()
        if not label:
            continue
        item = grouped.setdefault(
            label,
            {
                "candidate_label": label,
                "count": 0,
                "examples": [],
                "evidence_terms": [],
                "iterations": [],
                "learned_regex": [],
                "cosine_candidates": [],
            },
        )
        item["count"] += 1
        if row.get("iteration") not in item["iterations"]:
            item["iterations"].append(row.get("iteration"))
        example = {
            "arquivo": row.get("arquivo", ""),
            "titulo": row.get("titulo", ""),
            "evidence_text": row.get("evidence_text", ""),
            "rationale": row.get("rationale", ""),
            "confidence": row.get("confidence", 0),
        }
        if len(item["examples"]) < 5:
            item["examples"].append(example)
        for value in (row.get("evidence_text", ""), row.get("rationale", ""), row.get("titulo", "")):
            for term in str(value).replace(";", ".").replace(",", ".").split("."):
                term = " ".join(term.split()).strip()
                if 8 <= len(term) <= 90 and term not in item["evidence_terms"]:
                    item["evidence_terms"].append(term)
                if len(item["evidence_terms"]) >= 12:
                    break
    for label, item in grouped.items():
        item["learned_regex"] = regex_lookup.get(label, [])[:12]
        similarity_text = " ".join(
            [
                label.replace("_", " "),
                *[str(value) for value in item.get("evidence_terms", [])[:8]],
                *[str(example.get("titulo", "")) for example in item.get("examples", [])[:5]],
                *[str(example.get("evidence_text", "")) for example in item.get("examples", [])[:3]],
            ]
        )
        item["cosine_candidates"] = top_k_similar_themes(similarity_text, top_k=5)
    return sorted(grouped.values(), key=lambda item: (-int(item["count"]), str(item["candidate_label"])))


def _active_themes() -> list[dict[str, Any]]:
    payload = read_json(THEMES_JSON)
    regex_lookup = _regex_by_label()
    themes = []
    for theme in payload.get("themes", []):
        if theme.get("decision") != "accept":
            continue
        label = str(theme.get("canonical_theme"))
        themes.append(
            {
                "canonical_theme": label,
                "description": theme.get("description", ""),
                "included_cluster_ids": theme.get("included_cluster_ids", []),
                "included_subthemes": theme.get("included_subthemes", []),
                "evidence_terms": theme.get("evidence_terms", []),
                "learned_regex_count": len(regex_lookup.get(label, [])),
                "learned_regex_sample": regex_lookup.get(label, [])[:8],
            }
        )
    if RARE_NEWS_LABEL not in {str(theme["canonical_theme"]) for theme in themes}:
        themes.append(
            {
                "canonical_theme": RARE_NEWS_LABEL,
                "description": RARE_NEWS_DESCRIPTION,
                "included_cluster_ids": [],
                "included_subthemes": [],
                "evidence_terms": ["residual sem encaixe", "caso raro", "sem tema canonico defensavel"],
                "learned_regex_count": 0,
                "learned_regex_sample": [],
            }
        )
    return themes


def _family_for_tokens(tokens: set[str]) -> str:
    best_family = ""
    best_overlap = 0
    for family, terms in PROMOTION_FAMILIES.items():
        overlap = len(tokens.intersection(terms))
        if overlap > best_overlap:
            best_family = family
            best_overlap = overlap
    return best_family if best_overlap else ""


def _existing_parent_for_tokens(tokens: set[str], active_labels: set[str]) -> tuple[str, int]:
    best_parent = ""
    best_overlap = 0
    for label in active_labels:
        overlap = len(tokens.intersection(_tokens(label)))
        if overlap > best_overlap:
            best_parent = label
            best_overlap = overlap
    for label, terms in EXISTING_THEME_ABSORPTION.items():
        if label not in active_labels:
            continue
        overlap = len(tokens.intersection(terms))
        if overlap > best_overlap:
            best_parent = label
            best_overlap = overlap
    return best_parent, best_overlap


def _taxonomy_policy_decisions(active_themes: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> ThemeTreeRefinementResponse:
    """Consolida candidatos raros antes de promover temas.

    O Agente 3 pode enxergar uma novidade local e sugerir uma label muito especifica.
    Esta etapa olha o conjunto inteiro e evita que essas folhas virem temas finais
    quando cabem em macrofamilias ou em temas canonicos existentes.
    """
    active_labels = {str(theme["canonical_theme"]) for theme in active_themes}
    candidate_by_label = {str(item["candidate_label"]): item for item in candidates}
    family_counts: Counter[str] = Counter()
    family_members: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        label = str(candidate["candidate_label"])
        family = _family_for_tokens(_tokens(label))
        if not family:
            continue
        family_counts[family] += int(candidate.get("count", 0) or 0)
        family_members[family].append(label)

    decisions: list[ThemeCandidateDecision] = []
    promoted: list[str] = []
    promoted_seen: set[str] = set()
    for candidate in candidates:
        label = str(candidate["candidate_label"])
        tokens = _tokens(label)
        count = int(candidate.get("count", 0) or 0)
        parent, overlap = _existing_parent_for_tokens(tokens, active_labels)
        family = _family_for_tokens(tokens)

        if parent and overlap >= 1 and not family:
            decision = "merge_into_existing"
            promoted_theme = ""
            rationale = f"Candidata absorvida por tema canonico existente: {parent}."
        elif family and family_counts[family] >= 3:
            decision = "promote_to_canonical"
            parent = ""
            promoted_theme = family
            rationale = f"Candidata consolidada na macrofamilia recorrente {family}."
            if family not in promoted_seen:
                promoted.append(family)
                promoted_seen.add(family)
        elif parent and overlap >= 1:
            decision = "merge_into_existing"
            promoted_theme = ""
            rationale = f"Candidata rara absorvida por tema canonico existente: {parent}."
        elif count >= 3:
            decision = "promote_to_canonical"
            promoted_theme = label
            rationale = "Candidata recorrente sem tema maior defensavel."
            if promoted_theme not in promoted_seen:
                promoted.append(promoted_theme)
                promoted_seen.add(promoted_theme)
        elif family:
            decision = "keep_as_leaf"
            parent = family
            promoted_theme = ""
            rationale = f"Candidata rara mantida apenas como folha da macrofamilia {family}."
        else:
            decision = "merge_into_existing"
            parent = RARE_NEWS_LABEL
            promoted_theme = ""
            rationale = "Candidata unitaria sem massa e sem tema maior confiavel; agregada ao tema operacional noticias_raras."

        decisions.append(
            ThemeCandidateDecision(
                candidate_label=label,
                decision=decision,
                parent_theme=parent if decision in {"merge_into_existing", "keep_as_leaf"} else "",
                promoted_theme=promoted_theme if decision == "promote_to_canonical" else "",
                merged_candidate_labels=[],
                evidence_terms=list(candidate.get("evidence_terms", []))[:6],
                rationale=rationale,
                confidence=0.8 if decision != "quarantine" else 0.65,
            )
        )

    promoted_groups: dict[str, list[str]] = defaultdict(list)
    for decision in decisions:
        if decision.decision == "promote_to_canonical" and decision.promoted_theme:
            promoted_groups[decision.promoted_theme].append(decision.candidate_label)
    for index, decision in enumerate(decisions):
        if decision.decision == "promote_to_canonical" and decision.promoted_theme:
            merged = [label for label in promoted_groups[decision.promoted_theme] if label != decision.candidate_label]
            decisions[index] = decision.model_copy(update={"merged_candidate_labels": merged})

    counter = Counter(decision.decision for decision in decisions)
    return ThemeTreeRefinementResponse(
        decisions=decisions,
        promoted_canonical_themes=promoted,
        merged_into_existing_count=counter["merge_into_existing"],
        promoted_count=counter["promote_to_canonical"],
        kept_as_leaf_count=counter["keep_as_leaf"],
        discarded_count=counter["discard"],
        quarantined_count=counter["quarantine"],
        notes=[
            "Politica taxonomica deterministica aplicada apos a revisao residual.",
            "Candidatos unitarios nao viram temas finais salvo quando entram em macrofamilia recorrente.",
        ],
    )


def _refined_label_map(response: ThemeTreeRefinementResponse) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for decision in response.decisions:
        if decision.decision == "merge_into_existing" and decision.parent_theme:
            mapping[decision.candidate_label] = decision.parent_theme
        elif decision.decision == "promote_to_canonical" and decision.promoted_theme:
            mapping[decision.candidate_label] = decision.promoted_theme
        elif decision.decision == "keep_as_leaf" and decision.parent_theme:
            mapping[decision.candidate_label] = decision.parent_theme
        elif decision.decision in {"discard", "quarantine"}:
            mapping[decision.candidate_label] = ""
    return mapping


def _apply_refined_tree_to_regex_bank(response: ThemeTreeRefinementResponse) -> dict[str, int]:
    if not ACTIVE_REGEX_BANK_PATH.exists():
        return {"remapped_classifiers": 0, "removed_classifiers": 0, "merged_classifiers": 0}
    mapping = _refined_label_map(response)
    if not mapping:
        return {"remapped_classifiers": 0, "removed_classifiers": 0, "merged_classifiers": 0}

    payload = read_json(ACTIVE_REGEX_BANK_PATH)
    if not isinstance(payload, list):
        return {"remapped_classifiers": 0, "removed_classifiers": 0, "merged_classifiers": 0}

    grouped: dict[str, dict[str, Any]] = {}
    remapped = 0
    removed = 0
    merged = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        original_label = str(item.get("classificador", "") or item.get("label", "")).strip()
        target_label = mapping.get(original_label, original_label)
        if target_label in {"", RARE_NEWS_LABEL}:
            removed += 1
            continue
        if target_label != original_label:
            remapped += 1
        updated = dict(item)
        updated["classificador"] = target_label
        updated["source"] = "mixed" if target_label != original_label else updated.get("source", "")
        patterns = []
        for pattern in updated.get("patterns_do_classificador", []) or []:
            if not isinstance(pattern, dict):
                continue
            candidate = dict(pattern)
            examples = list(candidate.get("examples", []) or [])
            if original_label != target_label:
                examples.append(f"remapped_from:{original_label}")
            candidate["examples"] = examples[:8]
            patterns.append(candidate)
        updated["patterns_do_classificador"] = patterns
        if target_label not in grouped:
            grouped[target_label] = updated
            continue
        merged += 1
        current = grouped[target_label]
        current["uses"] = int(current.get("uses", 0) or 0) + int(updated.get("uses", 0) or 0)
        current["source"] = "mixed"
        current_examples = list(current.get("examples", []) or [])
        current["examples"] = (current_examples + list(updated.get("examples", []) or []))[:12]
        seen_patterns = {str(pattern.get("pattern_do_classificador", "")) for pattern in current.get("patterns_do_classificador", []) or []}
        for pattern in patterns:
            key = str(pattern.get("pattern_do_classificador", ""))
            if key and key not in seen_patterns:
                current.setdefault("patterns_do_classificador", []).append(pattern)
                seen_patterns.add(key)

    ordered = [grouped[label] for label in sorted(grouped)]
    write_json(ACTIVE_REGEX_BANK_PATH, ordered)
    return {"remapped_classifiers": remapped, "removed_classifiers": removed, "merged_classifiers": merged}


def _fallback_decisions(active_themes: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> ThemeTreeRefinementResponse:
    return _taxonomy_policy_decisions(active_themes, candidates)


def refine_theme_tree(config: RunConfig) -> ThemeTreeRefinementResponse:
    active_themes = _active_themes()
    candidates = _candidate_groups()
    write_json(
        THEME_REFINEMENT_INPUT_JSON,
        {
            "active_canonical_themes": active_themes,
            "agent3_candidate_themes": candidates,
            "notes": [
                "Insumo completo do Agente Organizador da Arvore.",
                "Cada candidato inclui contagem, evidencias, regex aprendidas quando houver e sugestoes por similaridade do cosseno.",
            ],
        },
    )
    if not candidates:
        response = ThemeTreeRefinementResponse(notes=["Nenhum tema candidato gerado pelo Agente 3."])
        write_json(REFINED_THEME_TREE_JSON, response.model_dump())
        return response

    prompt = (
        "Voce e o Agente Organizador da Arvore de Temas. Corrija a arvore global olhando TODOS os temas canonicos ativos "
        "e TODOS os candidatos surgidos no residual. Sua funcao e evitar bagunca taxonomica: nao promova candidato "
        "que pode ser absorvido por um tema existente; funda candidatos equivalentes; promova apenas subtemas claros, "
        "recorrentes e nao cobertos pelos pais atuais. Use zero intervencao humana.\n\n"
        "Decisoes permitidas por candidata:\n"
        "- merge_into_existing: candidata e folha/subtema de tema canonico existente;\n"
        "- promote_to_canonical: candidata recorrente e substantiva que merece novo no canonico;\n"
        "- keep_as_leaf: candidata ainda rara, mas pode ficar como folha observada;\n"
        "- discard: ruido ou tema operacional fraco;\n"
        "- quarantine: incerta.\n\n"
        "Temas canonicos ativos:\n"
        + json.dumps(active_themes, ensure_ascii=False)
        + "\n\nCandidatos do Agente 3 agrupados:\n"
        + json.dumps(candidates, ensure_ascii=False)
    )
    try:
        response, provider, model_name, _token_usage = invoke_json_with_fallback(prompt, ThemeTreeRefinementResponse, config, "agente_organizador_arvore")
        append_event({"stage": "agente_organizador_arvore", "status": "llm_ok", "provider": provider, "model": model_name})
        if not response.decisions and candidates:
            append_event({"stage": "agente_organizador_arvore", "status": "fallback_empty_decisions"})
            response = _fallback_decisions(active_themes, candidates)
    except Exception as exc:
        append_event({"stage": "agente_organizador_arvore", "status": "fallback", "error": str(exc)})
        response = _fallback_decisions(active_themes, candidates)

    response = _taxonomy_policy_decisions(active_themes, candidates)
    bank_result = _apply_refined_tree_to_regex_bank(response)
    append_event({"stage": "agente_organizador_arvore", "status": "taxonomy_policy_applied"})
    append_event({"stage": "agente_organizador_arvore", "status": "regex_bank_remapped", **bank_result})

    write_json(REFINED_THEME_TREE_JSON, response.model_dump())
    return response


def run(config: RunConfig | None = None) -> dict[str, object]:
    """Organiza globalmente a arvore de temas apos a etapa incremental."""
    config = config or RunConfig(reset=False)
    response = refine_theme_tree(config)
    result = {
        "stage": "agente_organizador_arvore",
        "theme_refinement_input_json": str(THEME_REFINEMENT_INPUT_JSON),
        "refined_theme_tree_json": str(REFINED_THEME_TREE_JSON),
        "decisions": len(response.decisions),
        "promoted_count": response.promoted_count,
        "promoted_unique_count": len(set(response.promoted_canonical_themes)),
        "promoted_canonical_themes": sorted(set(response.promoted_canonical_themes)),
        "merged_into_existing_count": response.merged_into_existing_count,
        "kept_as_leaf_count": response.kept_as_leaf_count,
        "discarded_count": response.discarded_count,
        "quarantined_count": response.quarantined_count,
    }
    append_event(result)
    return result
