from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from scripts.incremental.common import ACTIVE_REGEX_BANK_PATH, RunConfig, append_event
from scripts.incremental.llm_api import invoke_json_with_fallback
from scripts.schemas.pf_incremental_agent_schemas import InitialRegexBatchResponse, InitialRegexResponse, RegexRuleProposal

try:
    from scripts.pf_regex_classifier import LEARNED_RULE_WEIGHT, canonical_label, clean_learned_rules_file, clear_learned_rules_cache, fold_text, group_learned_rules
except ModuleNotFoundError:
    from pf_regex_classifier import LEARNED_RULE_WEIGHT, canonical_label, clean_learned_rules_file, clear_learned_rules_cache, fold_text, group_learned_rules


GENERIC_TERMS = {
    "para",
    "bahia",
    "rio janeiro",
    "janeiro",
    "sao paulo",
    "mato",
    "grosso",
    "mato grosso",
    "mato grosso sul",
    "rio grande",
    "rio grande sul",
    "rio grande norte",
    "minas gerais",
    "tocantins",
    "santa catarina",
    "espirito santo",
    "distrito federal",
    "ceara",
    "maranhao",
    "goias",
    "parana",
    "acre",
    "amapa",
    "amazonas",
    "grupo",
    "mandado",
    "mandados",
    "prisao",
    "investigados",
    "empresas",
    "dpf gov",
    "srrj gov",
    "federal rio",
    "gov 2203",
    "publicado",
    "subtitulo",
    "titulo",
    "tags",
    "corpo",
    "objetivo",
    "suspeito",
    "suspeitos",
    "cumpre",
    "cumpriu",
    "busca",
    "apreensao",
    "flagrante",
    "superintendencia",
    "regional",
    "delegacia",
    "ministerio",
    "secretaria",
    "fazenda",
    "veiculo",
    "pelos",
    "pelo",
    "foram",
    "ocorreu",
    "ocasiao",
    "objetivo",
    "identifica",
    "identificou",
    "lote",
}

PREFERRED_TERMS = {
    "crimes_contra_criancas": ["abuso sexual", "pornografia infantil", "pornografia infantojuvenil", "material de abuso infantil", "exploracao sexual"],
    "crimes_ambientais": ["garimpo ilegal", "extracao ilegal", "madeira ilegal", "desmatamento ilegal", "mineracao ilegal", "caca ilegal"],
    "trafico_drogas": ["trafico drogas", "trafico internacional", "maconha", "cocaina", "entorpecentes", "drogas"],
    "crimes_eleitorais": ["corrupcao eleitoral", "compra votos", "fraude eleitoral", "crimes eleitorais", "eleicoes"],
    "corrupcao_desvio_recursos_publicos": ["desvio recursos", "recursos publicos", "licitacao fraudulenta", "fraude licitacao", "peculato"],
    "crimes_sistema_financeiro": ["sistema financeiro", "gestao fraudulenta", "fraude bancaria", "emprestimos fraudulentos", "instituicao financeira"],
    "contrabando_descaminho": ["contrabando", "descaminho", "cigarros contrabandeados", "mercadoria estrangeira", "produtos eletronicos"],
    "lavagem_dinheiro": ["lavagem dinheiro", "ocultacao bens", "dissimular valores", "valores ilicitos"],
    "radiodifusao_clandestina": ["radio clandestina", "radio irregular", "radiodifusao clandestina", "anatel", "telecomunicacoes"],
    "armas_municoes": ["arma fogo", "armamento ilegal", "municoes", "posse ilegal arma", "porte ilegal arma"],
    "trabalho_escravo": ["trabalho escravo", "condicoes analogas", "trabalhadores resgatados"],
    "crimes_ciberneticos": ["crime cibernetico", "invasao sistemas", "fraude eletronica", "dispositivo informatico", "internet"],
    "crimes_migratorios": ["migracao ilegal", "imigracao ilegal", "promocao migracao ilegal", "passaporte falso", "documentos migratorios"],
    "crimes_previdenciarios": ["fraude previdenciaria", "beneficios previdenciarios", "estelionato previdenciario", "beneficio inss"],
    "moeda_falsa": ["moeda falsa", "cedulas falsas", "notas falsas"],
    "crime_organizado": ["crime organizado", "organizacao criminosa", "faccao criminosa", "associacao criminosa"],
    "fraudes_auxilios_beneficios": ["fraude auxilio", "auxilio emergencial", "auxilio brasil", "beneficio fraudulento"],
}

GLOBAL_DOMAIN_TERMS = {
    "fraude",
    "fraudes",
    "falso",
    "falsa",
    "falsas",
    "falsificacao",
    "ilegal",
    "ilegais",
    "irregular",
    "irregulares",
    "ilicito",
    "ilicitos",
    "criminosa",
    "criminoso",
    "criminosa",
    "trafico",
    "contrabando",
    "descaminho",
    "lavagem",
    "ocultacao",
    "desvio",
    "corrupcao",
    "abuso",
    "pornografia",
    "exploracao",
    "roubo",
    "furtado",
    "furto",
    "clandestina",
    "adulterado",
    "adulterada",
    "favorecer",
    "vantagem",
    "indevida",
}

REQUIRED_LABEL_TOKENS = {
    "armas_municoes": {"arma", "armas", "armamento", "municoes", "municao", "porte", "posse"},
    "contrabando_descaminho": {"contrabando", "descaminho", "cigarros", "cigarro", "mercadoria", "mercadorias"},
    "corrupcao_desvio_recursos_publicos": {
        "desvio",
        "recursos",
        "licitacao",
        "licitacoes",
        "fraude",
        "peculato",
        "contratacao",
        "corrupcao",
        "vantagem",
        "indevida",
        "propina",
        "favorecer",
        "pagamento",
    },
    "crime_organizado": {"organizado", "organizacao", "faccao", "faccoes", "associacao", "criminosa", "quadrilha"},
    "crimes_ambientais": {"garimpo", "extracao", "madeira", "desmatamento", "mineracao", "caca", "ambiental", "ambientais"},
    "crimes_ciberneticos": {"cibernetico", "ciberneticos", "invasao", "sistemas", "eletronica", "informatico", "internet"},
    "crimes_contra_criancas": {"abuso", "pornografia", "infantil", "infantojuvenil", "criancas", "adolescentes", "exploracao"},
    "crimes_eleitorais": {"eleitoral", "eleitorais", "eleicoes", "votos", "campanha", "campanhas"},
    "crimes_migratorios": {"migracao", "imigracao", "migratorios", "passaporte", "passaportes", "migrantes"},
    "crimes_previdenciarios": {"previdenciaria", "previdenciarias", "previdenciario", "previdenciarios", "inss"},
    "crimes_sistema_financeiro": {"financeiro", "financeira", "bancaria", "bancario", "emprestimos", "instituicao", "gestao"},
    "fraudes_auxilios_beneficios": {"auxilio", "beneficio", "beneficios", "emergencial", "brasil"},
    "lavagem_dinheiro": {"lavagem", "ocultacao", "dissimular", "valores", "bens", "dinheiro"},
    "moeda_falsa": {"moeda", "cedulas", "cedula", "notas"},
    "trabalho_escravo": {"trabalho", "escravo", "analogas", "trabalhadores"},
    "trafico_drogas": {"trafico", "drogas", "droga", "maconha", "cocaina", "entorpecentes", "sinteticas"},
    "radiodifusao_clandestina": {"radio", "radiodifusao", "clandestina", "telecomunicacoes", "irregular"},
}

LOCATION_ENTITY_TERMS = {
    "acre", "amapa", "amazonas", "bahia", "ceara", "distrito", "federal", "espirito", "santo",
    "goias", "maranhao", "mato", "grosso", "minas", "gerais", "para", "parana", "janeiro",
    "paulo", "santa", "catarina", "tocantins", "estados", "unidos", "caixa", "economica",
    "receita", "anatel", "correios", "ficco", "horus", "advenus", "ponto", "final",
    "onipresente", "recorrencia", "relapsus", "panela", "ferro", "recobro", "ousadia",
}

OPERATIONAL_TERMS = {
    "operacao", "acao", "conjunta", "integrada", "apoio", "deflagra", "deflagrou", "combate",
    "cumpre", "mandado", "mandados", "busca", "apreensao", "prisao", "flagrante", "investiga",
    "investigados", "suspeito", "suspeitos", "policia", "federal",
}


def unique_keep_order(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", fold_text(value)).strip()
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return output


def meaningful_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"\b[a-z0-9]{4,}\b", fold_text(value))
        if token not in GENERIC_TERMS and token not in OPERATIONAL_TERMS
    ]


def regex_pattern_tokens(value: str) -> list[str]:
    pattern_terms = re.findall(r"\\b([a-z0-9]{3,})", value.lower())
    if pattern_terms:
        return [token for token in pattern_terms if token not in GENERIC_TERMS and token not in OPERATIONAL_TERMS]
    return meaningful_tokens(value)


def label_domain_tokens(label: str) -> set[str]:
    tokens: set[str] = set()
    for term in PREFERRED_TERMS.get(label, []):
        tokens.update(meaningful_tokens(term))
    tokens.update(token for token in label.split("_") if len(token) >= 4 and token not in {"crimes", "crime"})
    return tokens


def is_location_or_entity_only(tokens: list[str]) -> bool:
    return bool(tokens) and all(token in LOCATION_ENTITY_TERMS for token in tokens)


def term_is_domain_candidate(label: str, term: str) -> bool:
    tokens = regex_pattern_tokens(term)
    if len(tokens) < 2 or is_location_or_entity_only(tokens):
        return False
    token_set = set(tokens)
    required = REQUIRED_LABEL_TOKENS.get(label, label_domain_tokens(label))
    return bool(token_set.intersection(required))


def pattern_has_crime_modus_anchor(label: str, pattern: str) -> bool:
    tokens = regex_pattern_tokens(pattern)
    if len(tokens) < 2 or is_location_or_entity_only(tokens):
        return False
    token_set = set(tokens)
    required = REQUIRED_LABEL_TOKENS.get(label, label_domain_tokens(label))
    required_hits = token_set.intersection(required)
    domain_hits = token_set.intersection(required | GLOBAL_DOMAIN_TERMS)
    noisy_hits = token_set.intersection(LOCATION_ENTITY_TERMS | OPERATIONAL_TERMS | GENERIC_TERMS)
    if not required_hits:
        return False
    if len(required_hits) >= 2:
        return True
    if len(domain_hits) >= 2 and len(noisy_hits) < len(domain_hits):
        return True
    return False


def term_pattern(term: str) -> str:
    tokens = meaningful_tokens(term)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return rf"\b{re.escape(tokens[0])}\w*\b"
    return r"\s+".join(rf"\b{re.escape(token)}\w*\b" for token in tokens)


def proximity_pattern(left: str, right: str, distance: int = 90) -> str:
    left_tokens = meaningful_tokens(left)
    right_tokens = meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return ""
    left_token = left_tokens[0]
    right_token = right_tokens[-1]
    if left_token == right_token:
        return ""
    return rf"\b{re.escape(left_token)}\w*\b.{{0,{distance}}}\b{re.escape(right_token)}\w*\b"


def phrase_candidates_from_text(text: str, limit: int = 80) -> list[str]:
    folded = fold_text(text)
    candidates: list[str] = []
    for line in folded.splitlines():
        line_tokens = meaningful_tokens(line)
        if 2 <= len(line_tokens) <= 8:
            candidates.append(" ".join(line_tokens))
    sentences = [part.strip() for part in re.split(r"[.!?;\n|:]+", folded) if part.strip()]
    for sentence in sentences:
        tokens = meaningful_tokens(sentence)
        for size in (5, 4, 3, 2):
            if len(tokens) < size:
                continue
            for index in range(0, len(tokens) - size + 1):
                candidates.append(" ".join(tokens[index : index + size]))
                if len(candidates) >= limit:
                    return unique_keep_order(candidates)
    return unique_keep_order(candidates)


def regex_rule_candidates_for_theme(
    label: str,
    evidence_terms: list[str],
    positives: list[str] | None = None,
    tags: list[str] | None = None,
    target: int | None = None,
) -> list[RegexRuleProposal]:
    positives = positives or []
    tags = tags or []
    preferred = PREFERRED_TERMS.get(label, [])
    selected_terms = [term for term in unique_keep_order([*preferred, *evidence_terms, *tags]) if term_is_domain_candidate(label, term)]
    for text in positives:
        selected_terms.extend(phrase for phrase in phrase_candidates_from_text(text, limit=40) if term_is_domain_candidate(label, phrase))
    selected_terms = [term for term in unique_keep_order(selected_terms) if term_is_domain_candidate(label, term)]

    patterns: list[tuple[str, str]] = []
    for term in selected_terms:
        if len(meaningful_tokens(term)) >= 2:
            pattern = term_pattern(term)
            if pattern:
                patterns.append((pattern, term))

    anchors = [token for term in preferred for token in meaningful_tokens(term)]
    if not anchors:
        anchors = [token for token in meaningful_tokens(label.replace("_", " "))]
    for term in selected_terms:
        term_tokens = meaningful_tokens(term)
        if not term_is_domain_candidate(label, term):
            continue
        for anchor in anchors[:8]:
            for token in term_tokens[:8]:
                pattern = proximity_pattern(anchor, token)
                if pattern:
                    patterns.append((pattern, f"{anchor} + {token}"))
                reverse = proximity_pattern(token, anchor)
                if reverse:
                    patterns.append((reverse, f"{token} + {anchor}"))

    rules: list[RegexRuleProposal] = []
    seen: set[str] = set()
    for pattern, source_term in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        if len(re.findall(r"\\b([a-z0-9]{3,})", pattern.lower())) < 2:
            continue
        rules.append(RegexRuleProposal(kind="crime", label=label, pattern=pattern, rationale=f"Candidato automatico por evidencia/folha: {source_term}", risk="medio"))
        if target is not None and len(rules) >= max(target * 4, target):
            break
    return rules


def fallback_rules_for_theme(label: str, evidence_terms: list[str]) -> list[RegexRuleProposal]:
    selected = [term for term in unique_keep_order([*PREFERRED_TERMS.get(label, []), *evidence_terms]) if term_is_domain_candidate(label, term)][:8]
    rules: list[RegexRuleProposal] = []
    for index, term in enumerate(selected, start=1):
        if len(re.findall(r"[a-z0-9]{3,}", term)) < 2:
            continue
        pattern = term_pattern(term)
        if pattern:
            rules.append(RegexRuleProposal(kind="crime", label=label, pattern=pattern, rationale=f"Fallback automatico da folha {index}: {term}", risk="medio"))
    return rules


def validate_regex(pattern: str, positives: list[str], negatives: list[str]) -> tuple[bool, str]:
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return False, f"regex invalida: {exc}"
    positive_hits = sum(1 for text in positives if compiled.search(fold_text(text)))
    negative_hits = sum(1 for text in negatives if compiled.search(fold_text(text)))
    pattern_terms = re.findall(r"\\b([a-z0-9]{3,})", pattern.lower())
    if is_location_or_entity_only(pattern_terms):
        return False, "padrao baseado apenas em localidade/entidade"
    if positives and positive_hits == 0:
        return False, "sem acerto positivo"
    if negatives and negative_hits > max(2, int(0.15 * len(negatives))):
        return False, f"muitos falsos positivos negativos={negative_hits}"
    return True, f"positivos={positive_hits}; negativos={negative_hits}"


def matches_preferred_leaf(doc: dict[str, object], label: str) -> bool:
    terms = PREFERRED_TERMS.get(label, [])
    if not terms:
        return True
    parsed = doc.get("parsed", {})
    tags = parsed.get("tags", []) if isinstance(parsed, dict) else []
    haystack = fold_text(" ".join([str(doc.get("titulo", "")), str(doc.get("context", "")), " ".join(str(tag) for tag in tags)]))
    return any(re.search(term_pattern(term), haystack, re.IGNORECASE) for term in terms)


def copy_regex_bank(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_pending_rule(records: list[dict[str, object]], kind: str, label: str, pattern: str, title: str, source: str) -> bool:
    normalized_label = canonical_label(label)
    if not normalized_label or not pattern:
        return False
    for record in records:
        if record.get("kind") == kind and record.get("label") == normalized_label and record.get("pattern") == pattern:
            examples = record.setdefault("examples", [])
            if isinstance(examples, list) and title and title not in examples:
                examples.append(title)
            record["uses"] = int(record.get("uses", 0) or 0) + 1
            return False
    records.append(
        {
            "kind": kind,
            "label": normalized_label,
            "pattern": pattern,
            "weight": LEARNED_RULE_WEIGHT,
            "source": source,
            "uses": 1,
            "examples": [title] if title else [],
        }
    )
    return True


def generate_initial_regex(
    themes_payload: dict[str, object],
    sample: list[dict[str, object]],
    cluster_rows: pd.DataFrame,
    config: RunConfig,
    active_regex_bank_path: Path = ACTIVE_REGEX_BANK_PATH,
) -> list[dict[str, object]]:
    docs_by_name = {item["arquivo"]: item for item in sample}
    responses: list[dict[str, object]] = []
    pending_bank_records: list[dict[str, object]] = []
    prompt_themes = []
    contexts_by_theme: dict[str, tuple[list[str], list[str], list[str]]] = {}

    for theme in themes_payload.get("themes", []):
        if theme.get("decision") != "accept":
            continue
        positives = []
        leaf_positives = []
        for cluster_id in theme.get("included_cluster_ids", []):
            names = cluster_rows.loc[cluster_rows["cluster_id"] == cluster_id, "arquivo"].head(20).tolist()
            for name in names:
                if name not in docs_by_name:
                    continue
                doc = docs_by_name[name]
                positives.append(doc["context"])
                if matches_preferred_leaf(doc, theme["canonical_theme"]):
                    leaf_positives.append(doc["context"])
        if leaf_positives:
            positives = leaf_positives
        tag_values = []
        for cluster_id in theme.get("included_cluster_ids", []):
            cluster_subset = cluster_rows.loc[cluster_rows["cluster_id"] == cluster_id]
            for raw_tags in cluster_subset.get("tags", pd.Series(dtype=str)).fillna("").tolist():
                for tag in str(raw_tags).split(" | "):
                    tag = tag.strip()
                    if tag and tag not in tag_values:
                        tag_values.append(tag)
        negatives = [item["context"] for item in sample[:50] if item["context"] not in positives]
        contexts_by_theme[theme["canonical_theme"]] = (positives, negatives, tag_values)
        prompt_themes.append(
            {
                "canonical_theme": theme["canonical_theme"],
                "included_cluster_ids": theme.get("included_cluster_ids", []),
                "evidence_terms": theme.get("evidence_terms", [])[:10],
                "tags": tag_values[:20],
                "positive_snippets": [text[:360] for text in positives[:4]],
            }
        )

    prompt = (
        "Voce e o Agente 2. Gere regex iniciais para todos os temas canonicos abaixo em uma unica resposta. "
        "A label de cada regex deve ser exatamente o canonical_theme recebido do Agente 1. "
        "Use regex curtas, auditaveis e sem acentos quando possivel. "
        + (
            f"Gere ate {config.initial_regex_target_per_theme} regex candidatas por tema quando houver evidencias suficientes. "
            if config.initial_regex_target_per_theme is not None
            else "Gere todas as regex candidatas uteis quando houver evidencias suficientes; nao aplique teto artificial por tema. "
        )
        + "Cubra folhas diferentes do mesmo tema usando apenas crimes e modus operandi como pistas. "
        "Nao use entidades, orgaos, cidades, estados, paises, nomes de operacao ou localidades como regex. "
        "Evite termos genericos como policia, federal, operacao, crime, investigacao isolados. "
        "Retorne uma lista themes no schema InitialRegexBatchResponse.\n\n"
        + json.dumps(prompt_themes, ensure_ascii=False)
    )
    try:
        batch_response, provider, model_name = invoke_json_with_fallback(prompt, InitialRegexBatchResponse, config, "agente2_regex_inicial")
        append_event({"stage": "agente2_regex_inicial", "status": "llm_ok", "provider": provider, "model": model_name})
        response_by_theme = {item.canonical_theme: item for item in batch_response.themes}
    except Exception as exc:
        append_event({"stage": "agente2_regex_inicial", "status": "fallback_batch", "error": str(exc)})
        response_by_theme = {}

    for theme in themes_payload.get("themes", []):
        if theme.get("decision") != "accept":
            continue
        positives, negatives, tag_values = contexts_by_theme.get(theme["canonical_theme"], ([], [], []))
        response = response_by_theme.get(theme["canonical_theme"])
        if response is None:
            response = InitialRegexResponse(canonical_theme=theme["canonical_theme"], accepted_rules=[])
        canonical_theme = theme["canonical_theme"]
        response.accepted_rules = [rule for rule in response.accepted_rules if rule.label == canonical_theme]
        response.accepted_rules = [
            rule for rule in response.accepted_rules if len(re.findall(r"\\b([a-z0-9]{3,})", rule.pattern.lower())) >= 2
        ]
        deterministic_candidates = regex_rule_candidates_for_theme(
            canonical_theme,
            theme.get("evidence_terms", []),
            positives=positives,
            tags=tag_values,
            target=config.initial_regex_target_per_theme,
        )
        response.accepted_rules = [*response.accepted_rules, *deterministic_candidates]

        accepted = []
        quarantined = []
        seen_patterns: set[str] = set()
        for rule in response.accepted_rules:
            if config.initial_regex_target_per_theme is not None and len(accepted) >= config.initial_regex_target_per_theme:
                break
            if rule.pattern in seen_patterns:
                continue
            seen_patterns.add(rule.pattern)
            if not term_is_domain_candidate(canonical_theme, rule.pattern):
                quarantined.append(rule.model_dump() | {"validation": "sem ancora de crime/modus da label"})
                continue
            ok, reason = validate_regex(rule.pattern, positives, negatives)
            record = rule.model_dump() | {"validation": reason}
            if ok and append_pending_rule(pending_bank_records, rule.kind, theme["canonical_theme"], rule.pattern, f"agent2:{theme['canonical_theme']}", "agent2_initial_regex"):
                accepted.append(record | {"label": theme["canonical_theme"]})
            else:
                quarantined.append(record)
        responses.append(
            {
                "canonical_theme": theme["canonical_theme"],
                "included_cluster_ids": theme.get("included_cluster_ids", []),
                "accepted_rules": accepted,
                "rejected_rules": [rule.model_dump() for rule in response.rejected_rules],
                "quarantined_rules": [rule.model_dump() for rule in response.quarantined_rules] + quarantined,
                "notes": response.notes,
            }
        )

    active_regex_bank_path.parent.mkdir(parents=True, exist_ok=True)
    active_regex_bank_path.write_text(json.dumps(group_learned_rules(pending_bank_records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    clear_learned_rules_cache(active_regex_bank_path)
    clean_learned_rules_file(active_regex_bank_path)
    return responses
