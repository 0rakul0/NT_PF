from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any


DOMAIN_PHRASES: tuple[str, ...] = (
    "abuso sexual",
    "abuso sexual infantil",
    "abuso sexual infantojuvenil",
    "adulteracao",
    "agrotoxico ilegal",
    "ameaca",
    "anabolizante",
    "arma de fogo",
    "arma ilegal",
    "armamento",
    "assalto",
    "associacao criminosa",
    "auxilio emergencial",
    "beneficio previdenciario",
    "caixa eletronico",
    "cedula falsa",
    "certificado falso",
    "cigarro contrabandeado",
    "comercio ilegal",
    "compartilhamento de material",
    "contrabando",
    "corrupcao",
    "corrupcao eleitoral",
    "crime ambiental",
    "crime cibernetico",
    "crime contra crianca",
    "crime eleitoral",
    "crime organizado",
    "crime previdenciario",
    "crime transfronteirico",
    "crimes contra o sistema financeiro",
    "descaminho",
    "desmatamento",
    "desvio de recurso publico",
    "documento falso",
    "droga",
    "estelionato",
    "evasao de divisas",
    "exploracao ilegal",
    "exploracao sexual",
    "extracao ilegal",
    "faccao criminosa",
    "falsidade ideologica",
    "falsificacao",
    "fraude",
    "fraude bancaria",
    "fraude contra o inss",
    "fraude em licitacao",
    "fraude previdenciaria",
    "garimpo ilegal",
    "gestao fraudulenta",
    "importacao ilegal",
    "invasao de sistema",
    "lavagem de dinheiro",
    "licitacao",
    "madeira ilegal",
    "material pornografico",
    "medicamento ilegal",
    "migracao ilegal",
    "mineracao ilegal",
    "moeda falsa",
    "municao",
    "organizacao criminosa",
    "ouro ilegal",
    "pesca ilegal",
    "pirataria",
    "pornografia infantil",
    "posse ilegal",
    "produto falsificado",
    "radio clandestina",
    "radio irregular",
    "radiodifusao clandestina",
    "receptacao",
    "recurso publico",
    "roubo",
    "sonegação",
    "sonegacao",
    "terrorismo",
    "trafico",
    "trafico de drogas",
    "trafico de armas",
    "trafico internacional",
    "trabalho escravo",
    "uso de documento falso",
)

DOMAIN_ANCHORS: tuple[str, ...] = (
    "abus",
    "adulter",
    "anabol",
    "arma",
    "assalt",
    "benefici",
    "cedula",
    "clandestin",
    "contraband",
    "corrup",
    "crime",
    "crimin",
    "descaminh",
    "desmat",
    "desvio",
    "documento fals",
    "droga",
    "estelionat",
    "explor",
    "extracao",
    "facc",
    "fals",
    "fraud",
    "garimp",
    "ilegal",
    "ilic",
    "inss",
    "lavagem",
    "licit",
    "madeira",
    "medicamento",
    "migracao",
    "moeda",
    "munic",
    "organizacao crimin",
    "ouro",
    "pornograf",
    "previdenci",
    "radio",
    "recept",
    "roubo",
    "soneg",
    "trafic",
    "trabalho escrav",
)

DOMAIN_STEM_LABELS: tuple[tuple[str, str], ...] = (
    ("abus", "abuso sexual"),
    ("adulter", "adulteracao"),
    ("anabol", "anabolizante"),
    ("armament", "armamento"),
    ("arma de fogo", "arma de fogo"),
    ("arma ilegal", "arma ilegal"),
    ("assalt", "assalto"),
    ("associacao crimin", "crime organizado"),
    ("benefici", "beneficio previdenciario"),
    ("cedul", "cedula falsa"),
    ("clandestin", "radiodifusao clandestina"),
    ("contraband", "contrabando"),
    ("corrup", "corrupcao"),
    ("descaminh", "descaminho"),
    ("desmat", "desmatamento"),
    ("desvio", "desvio de recurso publico"),
    ("estelionat", "estelionato"),
    ("exploracao sexual", "exploracao sexual"),
    ("extracao", "extracao ilegal"),
    ("facc", "faccao criminosa"),
    ("fals", "falsificacao"),
    ("fraud", "fraude"),
    ("garimp", "garimpo ilegal"),
    ("lavagem", "lavagem de dinheiro"),
    ("licitac", "licitacao"),
    ("madeira", "madeira ilegal"),
    ("medicamento", "medicamento ilegal"),
    ("migracao", "migracao ilegal"),
    ("miner", "mineracao ilegal"),
    ("moeda", "moeda falsa"),
    ("organizacao crimin", "crime organizado"),
    ("ouro", "ouro ilegal"),
    ("pornograf", "pornografia infantil"),
    ("previdenci", "crime previdenciario"),
    ("radio", "radiodifusao clandestina"),
    ("recept", "receptacao"),
    ("roubo", "roubo"),
    ("soneg", "sonegacao"),
    ("trafic", "trafico"),
    ("trabalho escrav", "trabalho escravo"),
)

LOCATION_ENTITY_STOPWORDS: set[str] = {
    "acre",
    "alagoas",
    "amapa",
    "amazonas",
    "anapolis",
    "aracaju",
    "bahia",
    "belem",
    "belo",
    "boa",
    "brasil",
    "brasilia",
    "catarina",
    "ceara",
    "cgcs",
    "distrito",
    "espirito",
    "federal",
    "ficco",
    "goias",
    "horizonte",
    "janeiro",
    "joao",
    "luis",
    "macapa",
    "maranhao",
    "mato",
    "minas",
    "natal",
    "norte",
    "para",
    "paraiba",
    "parana",
    "paulo",
    "pernambuco",
    "piaui",
    "policia",
    "porto",
    "recife",
    "ribeirao",
    "rio",
    "rondonia",
    "roraima",
    "salvador",
    "santa",
    "santo",
    "sao",
    "sergipe",
    "sres",
    "srmg",
    "srpb",
    "srrj",
    "srrn",
    "srrr",
    "srsp",
    "superintendencia",
    "tocantins",
    "vitoria",
}

GENERIC_STOPWORDS: set[str] = {
    "acao",
    "apoio",
    "apreensao",
    "associacao",
    "busca",
    "combate",
    "comunicacao",
    "contra",
    "coordenacao",
    "criminal",
    "criminalidade",
    "criminosa",
    "criminosas",
    "criminoso",
    "criminosos",
    "crimes",
    "cumpre",
    "cumprimento",
    "deflagra",
    "deflagrada",
    "destaque",
    "durante",
    "estado",
    "foram",
    "geral",
    "ilegal",
    "ilegais",
    "irregular",
    "irregulares",
    "mandado",
    "mandados",
    "operacao",
    "organizacao",
    "organizacoes",
    "preso",
    "prisao",
    "sexta",
    "social",
    "suspeito",
}


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def token_slug(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]{3,}", fold_text(value)))


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", fold_text(phrase))
    return re.compile(r"\b" + r"\W+".join(re.escape(token) + r"\w*" for token in tokens) + r"\b")


DOMAIN_PATTERNS = tuple((phrase, phrase_pattern(phrase)) for phrase in DOMAIN_PHRASES)


def flatten_tags(tags: object) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        values = re.split(r"\s*\|\s*|,\s*", tags)
    elif isinstance(tags, Iterable):
        values = [str(item) for item in tags]
    else:
        values = [str(tags)]
    return [value.strip() for value in values if value and value.strip()]


def clean_domain_tokens(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]{3,}", fold_text(text))
    kept = [
        token
        for token in tokens
        if token not in LOCATION_ENTITY_STOPWORDS
        and token not in GENERIC_STOPWORDS
        and not token.isdigit()
        and not re.fullmatch(r"\d+[a-z]*", token)
    ]
    return " ".join(kept)


def domain_sentences(text: str, limit: int = 12) -> list[str]:
    folded = fold_text(text)
    parts = [part.strip() for part in re.split(r"[.!?;\n]+", folded) if part.strip()]
    selected: list[str] = []
    for part in parts:
        if any(anchor in part for anchor in DOMAIN_ANCHORS):
            cleaned = clean_domain_tokens(part)
            if cleaned and cleaned not in selected:
                selected.append(cleaned)
        if len(selected) >= limit:
            break
    return selected


def domain_terms_from_text(text: str) -> list[str]:
    folded = fold_text(text)
    terms: list[str] = []
    for phrase, pattern in DOMAIN_PATTERNS:
        if pattern.search(folded) and phrase not in terms:
            terms.append(fold_text(phrase))
    for stem, label in DOMAIN_STEM_LABELS:
        if stem in folded and label not in terms:
            terms.append(label)
    return terms


def informative_domain_tags(tags: object) -> list[str]:
    selected: list[str] = []
    for tag in flatten_tags(tags):
        slug = token_slug(tag)
        if not slug or slug in LOCATION_ENTITY_STOPWORDS or slug in GENERIC_STOPWORDS:
            continue
        folded = fold_text(tag)
        if domain_terms_from_text(folded) or any(anchor in folded for anchor in DOMAIN_ANCHORS):
            selected.append(clean_domain_tokens(folded))
    return [tag for tag in selected if tag]


def build_domain_cluster_text(doc: dict[str, Any] | str) -> tuple[str, list[str]]:
    if isinstance(doc, str):
        text = doc
        tags: object = []
    else:
        parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
        parts = [
            str(doc.get("titulo", "")),
            str(parsed.get("subtitulo", "")),
            " ".join(flatten_tags(doc.get("tags", []))),
            str(parsed.get("corpo", "")),
            str(doc.get("context", "")),
        ]
        text = "\n".join(part for part in parts if part)
        tags = doc.get("tags", [])

    terms = domain_terms_from_text(text)
    tag_terms = informative_domain_tags(tags)
    sentences = domain_sentences(text)
    weighted_terms = [token_slug(term) for term in terms] * 8 + [token_slug(term) for term in tag_terms] * 4
    cluster_text = " ".join([*weighted_terms, *sentences[:4]]).strip()
    if not cluster_text:
        cluster_text = "tema_criminal_indefinido " + clean_domain_tokens(text)[:1200]
    return cluster_text, terms
