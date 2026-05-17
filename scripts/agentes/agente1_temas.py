from __future__ import annotations

import json
import re
import unicodedata

import pandas as pd

from scripts.incremental.common import RunConfig, append_event, slug
from scripts.incremental.llm_api import invoke_json_with_fallback
from scripts.schemas.pf_incremental_agent_schemas import OperationalCanonicalTheme, OperationalThemeBifurcationResponse


THEME_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("radiodifusao_clandestina", ("radio clandestina", "radio irregular", "radiodifusao", "anatel", "telecomunicacao", "telecomunicacoes")),
    ("crimes_contra_criancas", ("infantil", "infantojuvenil", "criancas", "adolescentes", "pornografia infantil", "abuso sexual")),
    ("trafico_drogas", ("trafico drogas", "trafico", "drogas", "maconha", "cocaina")),
    ("crimes_ambientais", ("garimpo", "ouro", "madeira", "indigena", "extracao", "ambiental", "desmatamento")),
    ("corrupcao_desvio_recursos_publicos", ("recursos publicos", "desvio", "licitacao", "licitacoes", "fraude", "publicos")),
    ("crimes_eleitorais", ("eleitoral", "eleitorais", "eleicoes", "compra votos")),
    ("crimes_sistema_financeiro", ("sistema financeiro", "emprestimos", "financeiro", "bancaria", "fraudes bancarias")),
    ("contrabando_descaminho", ("contrabando", "cigarros", "descaminho", "produtos eletronicos")),
    ("lavagem_dinheiro", ("lavagem", "dinheiro", "ocultacao", "milhoes")),
    ("armas_municoes", ("arma", "armas", "fogo", "municoes", "arma fogo")),
    ("trabalho_escravo", ("trabalho escravo", "trabalhadores", "condicoes analogas", "resgate")),
    ("crimes_ciberneticos", ("cibernetico", "internet", "invasao", "sistemas", "dispositivos")),
    ("crimes_migratorios", ("migracao ilegal", "imigracao", "migrantes", "passaportes")),
    ("crimes_previdenciarios", ("previdencia", "inss", "beneficio", "beneficios", "aposentadoria")),
    ("moeda_falsa", ("moeda falsa", "cedulas falsas", "cedulas", "notas falsas")),
    ("crime_organizado", ("ficco", "crime organizado", "organizacao criminosa", "associacao criminosa", "faccao")),
    ("fraudes_auxilios_beneficios", ("auxilio", "emergencial", "beneficios", "auxilio brasil")),
]

PREFERRED_EVIDENCE: dict[str, list[str]] = {label: list(needles[:4]) for label, needles in THEME_RULES}

GENERIC_EVIDENCE_SLUGS = {
    "acre",
    "amapa",
    "amazonas",
    "bahia",
    "ceara",
    "distrito_federal",
    "espirito_santo",
    "goias",
    "maranhao",
    "mato_grosso",
    "mato_grosso_sul",
    "minas_gerais",
    "para",
    "parana",
    "rio_de_janeiro",
    "rio_grande_norte",
    "rio_grande_sul",
    "santa_catarina",
    "sao_paulo",
    "tocantins",
    "operacao",
    "operacao_pf",
    "policia_federal",
    "mandado",
    "mandados",
    "busca_apreensao",
    "prisao",
    "flagrante",
    "foram",
    "cumpre",
    "cumpriu",
}


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def needle_matches(text: str, needle: str) -> bool:
    tokens = [token for token in re.findall(r"[a-z0-9]{3,}", fold_text(needle))]
    if not tokens:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(token) + r"\w*" for token in tokens) + r"\b"
    return re.search(pattern, text) is not None


def infer_canonical_themes(terms: list[str]) -> list[str]:
    text = fold_text(" ".join(terms))
    labels = [label for label, needles in THEME_RULES if any(needle_matches(text, needle) for needle in needles)]
    if labels:
        return labels
    candidates = [
        slug(term)
        for term in terms
        if len(term) >= 5 and slug(term) not in {"gerais", "geral", "para", "bahia", "sao_paulo", "minas_gerais", "rio_janeiro", "janeiro", "ilegal"}
    ]
    return [candidates[0]] if candidates else ["tema_operacional"]


def evidence_matches_label(label: str, term: str) -> bool:
    term_slug = slug(term)
    if not term_slug or term_slug in GENERIC_EVIDENCE_SLUGS:
        return False
    text = fold_text(term)
    return any(needle_matches(text, needle) for needle in PREFERRED_EVIDENCE.get(label, []))


def fallback_themes(cluster_summary: pd.DataFrame) -> OperationalThemeBifurcationResponse:
    groups: dict[str, dict[str, object]] = {}
    for _, row in cluster_summary.loc[cluster_summary["cluster_id"] != -1].iterrows():
        terms = [term.strip() for term in str(row["top_terms"]).split(" | ") if term.strip()]
        tags = [tag.strip() for tag in str(row.get("sample_tags", "")).split(" | ") if tag.strip()]
        titles = [title.strip() for title in str(row.get("sample_titles", "")).split(" | ") if title.strip()]
        labels = infer_canonical_themes([*terms, *tags, *titles])
        for label in labels:
            group = groups.setdefault(label, {"cluster_ids": [], "evidence_terms": [], "subthemes": []})
            group["cluster_ids"].append(int(row["cluster_id"]))
            for term in [*PREFERRED_EVIDENCE.get(label, []), *terms, *tags[:8]]:
                if term not in PREFERRED_EVIDENCE.get(label, []) and not evidence_matches_label(label, term):
                    continue
                if term and term not in group["evidence_terms"]:
                    group["evidence_terms"].append(term)
            if len(labels) > 1:
                group["subthemes"].append(f"cluster_{int(row['cluster_id'])}_folha")
    themes = [
        OperationalCanonicalTheme(
            canonical_theme=label,
            description=f"Tema canonico automatico agregado a partir dos clusters {', '.join(map(str, data['cluster_ids']))}",
            included_cluster_ids=list(data["cluster_ids"]),
            included_subthemes=list(dict.fromkeys(data.get("subthemes", []))),
            evidence_terms=list(data["evidence_terms"])[:12],
            confidence=0.62,
            decision="accept",
        )
        for label, data in sorted(groups.items())
        if label != "tema_operacional"
    ]
    noise = [-1] if -1 in set(cluster_summary["cluster_id"]) else []
    return OperationalThemeBifurcationResponse(themes=themes, quarantined_cluster_ids=noise)


def generate_canonical_themes(cluster_summary: pd.DataFrame, config: RunConfig) -> OperationalThemeBifurcationResponse:
    payload = cluster_summary.to_dict(orient="records")
    prompt = (
        "Voce e o Agente 1. Agrupe clusters exploratorios em temas canonicos amplos. "
        "Os clusters foram gerados com texto focado em crimes e modus operandi; trate locais, orgaos, telefones, siglas regionais e nomes de operacao apenas como metadados, nunca como tema canonico. "
        "Exemplo: abuso infantil, pornografia infantil e compartilhamento de material devem virar crimes_contra_criancas. "
        "Se um cluster misto tiver subtema claro, bifurque em mais de um tema quando necessario. "
        "Nao gere regex. Nao peca revisao humana; use accept, discard ou quarantine. "
        "Clusters:\n"
        + json.dumps(payload, ensure_ascii=False)[:24000]
    )
    try:
        themes, provider, model_name = invoke_json_with_fallback(prompt, OperationalThemeBifurcationResponse, config, "agente1_temas")
        append_event({"stage": "agente1_temas", "status": "llm_ok", "provider": provider, "model": model_name})
        if not any(theme.decision == "accept" for theme in themes.themes):
            append_event({"stage": "agente1_temas", "status": "fallback_empty_llm"})
            return fallback_themes(cluster_summary)
        return themes
    except Exception as exc:
        append_event({"stage": "agente1_temas", "status": "fallback", "error": str(exc)})
        return fallback_themes(cluster_summary)
