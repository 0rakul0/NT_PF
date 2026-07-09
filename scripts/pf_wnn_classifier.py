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
    from scripts.text_utils import canonical_label, fold_text
except ModuleNotFoundError:
    from pf_llm_models import NoticiaLLMInference
    from text_utils import canonical_label, fold_text


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
BLOCKED_MODUS_LABELS = {
    "atuacao_clandestina",
    "falta_de_fiscalizacao",
}
MODUS_LABEL_ALIASES = {
    "madeira_ilegal": "extracao_ilegal_madeira",
    "exploracao_ilegal_madeira": "extracao_ilegal_madeira",
    "extracao_ilegal_de_madeira": "extracao_ilegal_madeira",
    "exploracao_ilicita_madeira": "extracao_ilegal_madeira",
    "mineracao_ilegal": "garimpo_ilegal",
    "garimpo_clandestino": "garimpo_ilegal",
    "trafico_animais_silvestres": "trafico_de_especies",
    "trafico_fauna_silvestre": "trafico_de_especies",
    "trafico_vida_silvestre": "trafico_de_especies",
    "pesca_clandestina": "pesca_ilegal",
}


def normalize_modus_label(value: object) -> str:
    normalized = canonical_label(str(value or ""))
    if not normalized:
        return ""
    normalized = MODUS_LABEL_ALIASES.get(normalized, normalized)
    if normalized in BLOCKED_MODUS_LABELS:
        return ""
    if normalized.startswith("fiscalizacao_"):
        if normalized != "fiscalizacao_ambiental":
            return ""
    return normalized

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
        {"name": "produto_descaminhado", "tokens": ["produto", "descaminhado"], "weight": 0.95},
        {"name": "importacao_irregular", "tokens": ["importacao", "irregular"], "weight": 0.9},
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
        {"name": "peculato_recursos_publicos", "tokens": ["peculato", "recursos", "publicos"], "weight": 1.0},
        {"name": "desvio_verbas_publicas", "tokens": ["desvio", "verbas", "publicas"], "weight": 1.0},
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
    "crimes_ciberneticos": [
        {"name": "invasao_dispositivo_informatico", "tokens": ["invasao", "dispositivo", "informatico"], "weight": 1.15},
        {"name": "dados_cadastrais_violados", "tokens": ["dados", "cadastrais", "violados"], "weight": 0.95},
        {"name": "fraude_digital_sistemas", "tokens": ["fraude", "digital", "sistemas"], "weight": 0.9},
    ],
    "crimes_migratorios": [
        {"name": "migracao_ilegal", "tokens": ["migracao", "ilegal"], "weight": 1.1},
        {"name": "passaporte_fraudulento", "tokens": ["passaporte", "fraudulento"], "weight": 1.0},
        {"name": "visto_irregular", "tokens": ["visto", "irregular"], "weight": 0.95},
    ],
    "fraudes_auxilios_beneficios": [
        {"name": "auxilio_emergencial_fraude", "tokens": ["auxilio", "emergencial", "fraude"], "weight": 1.15},
        {"name": "beneficio_social_irregular", "tokens": ["beneficio", "social", "irregular"], "weight": 1.0},
        {"name": "saque_beneficio_indevido", "tokens": ["saque", "beneficio", "indevido"], "weight": 0.95},
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

CURATED_MODUS_DISCRIMINATORS: dict[str, list[dict[str, object]]] = {
    "extracao_ilegal_madeira": [
        {"name": "extracao_ilegal_madeira", "tokens": ["extracao", "ilegal", "madeira"], "weight": 1.15},
        {"name": "madeira_ilegal", "tokens": ["madeira", "ilegal"], "weight": 1.05},
        {"name": "exploracao_ilegal_madeira", "tokens": ["exploracao", "ilegal", "madeira"], "weight": 1.0},
    ],
    "garimpo_ilegal": [
        {"name": "garimpo_ilegal", "tokens": ["garimpo", "ilegal"], "weight": 1.15},
        {"name": "mineracao_ilegal", "tokens": ["mineracao", "ilegal"], "weight": 1.05},
        {"name": "extracao_ilegal_ouro", "tokens": ["extracao", "ilegal", "ouro"], "weight": 1.0},
    ],
    "desmatamento": [
        {"name": "desmatamento", "tokens": ["desmatamento"], "weight": 1.1},
        {"name": "desmatamento_ilegal", "tokens": ["desmatamento", "ilegal"], "weight": 1.05},
        {"name": "queimada_desmatamento", "tokens": ["queimada", "desmatamento"], "weight": 0.95},
    ],
    "trafico_de_especies": [
        {"name": "trafico_de_especies", "tokens": ["trafico", "especies"], "weight": 1.1},
        {"name": "animais_silvestres", "tokens": ["animais", "silvestres"], "weight": 1.0},
        {"name": "fauna_silvestre", "tokens": ["fauna", "silvestre"], "weight": 0.95},
    ],
    "caca_ilegal": [
        {"name": "caca_ilegal", "tokens": ["caca", "ilegal"], "weight": 1.0},
        {"name": "abate_animal_silvestre", "tokens": ["abate", "animal", "silvestre"], "weight": 0.95},
    ],
    "pesca_ilegal": [
        {"name": "pesca_ilegal", "tokens": ["pesca", "ilegal"], "weight": 1.0},
        {"name": "uso_redes_pesca", "tokens": ["redes", "pesca"], "weight": 0.95},
    ],
    "uso_ilegal_solo": [
        {"name": "uso_ilegal_solo", "tokens": ["uso", "ilegal", "solo"], "weight": 1.0},
        {"name": "invasao_terra_publica", "tokens": ["invasao", "terra", "publica"], "weight": 0.95},
    ],
    "comercializacao_ilegal": [
        {"name": "comercializacao_ilegal", "tokens": ["comercializacao", "ilegal"], "weight": 1.0},
        {"name": "comercio_irregular", "tokens": ["comercio", "irregular"], "weight": 0.95},
        {"name": "venda_ilegal", "tokens": ["venda", "ilegal"], "weight": 0.95},
    ],
    "transporte_ilegal": [
        {"name": "transporte_ilegal", "tokens": ["transporte", "ilegal"], "weight": 1.0},
        {"name": "transporte_irregular", "tokens": ["transporte", "irregular"], "weight": 0.95},
    ],
    "apreensao_madeira": [
        {"name": "apreensao_madeira", "tokens": ["apreensao", "madeira"], "weight": 1.0},
        {"name": "apreensao_madeira_irregular", "tokens": ["apreensao", "madeira", "irregular"], "weight": 0.95},
    ],
    "fiscalizacao_ambiental": [
        {"name": "fiscalizacao_ambiental", "tokens": ["fiscalizacao", "ambiental"], "weight": 1.0},
        {"name": "acao_fiscalizacao_ambiental", "tokens": ["acao", "fiscalizacao", "ambiental"], "weight": 0.95},
    ],
    "atividade_clandestina": [
        {"name": "atividade_clandestina", "tokens": ["atividade", "clandestina"], "weight": 1.0},
        {"name": "funcionamento_clandestino", "tokens": ["funcionamento", "clandestino"], "weight": 0.95},
    ],
    "risco_ambiental": [
        {"name": "risco_ambiental", "tokens": ["risco", "ambiental"], "weight": 1.0},
        {"name": "dano_ambiental", "tokens": ["dano", "ambiental"], "weight": 0.95},
    ],
    "arma_fogo": [
        {"name": "arma_fogo", "tokens": ["arma", "fogo"], "weight": 1.2},
        {"name": "porte_arma", "tokens": ["porte", "arma"], "weight": 1.0},
    ],
    "fraude_documental": [
        {"name": "documento_falso", "tokens": ["documento", "falso"], "weight": 1.15},
        {"name": "documentos_falsos", "tokens": ["documentos", "falsos"], "weight": 1.1},
        {"name": "falsificacao_documental", "tokens": ["falsificacao", "documental"], "weight": 1.0},
    ],
    "fraude_digital": [
        {"name": "perfil_falso", "tokens": ["perfil", "falso"], "weight": 1.05},
        {"name": "aplicativo_mensagens", "tokens": ["aplicativo", "mensagens"], "weight": 0.95},
        {"name": "rede_social", "tokens": ["rede", "social"], "weight": 0.9},
    ],
    "fraude_pix": [
        {"name": "fraude_pix", "tokens": ["fraude", "pix"], "weight": 1.2},
        {"name": "transferencia_pix", "tokens": ["transferencia", "pix"], "weight": 1.0},
    ],
    "arrombamento": [
        {"name": "arrombamento", "tokens": ["arrombamento"], "weight": 1.1},
        {"name": "porta_arrombada", "tokens": ["porta", "arrombada"], "weight": 1.0},
    ],
    "abordagem_via_publica": [
        {"name": "abordagem_rua", "tokens": ["abordagem", "rua"], "weight": 1.0},
        {"name": "via_publica", "tokens": ["via", "publica"], "weight": 0.95},
    ],
    "invasao_dispositivo": [
        {"name": "invasao_dispositivo", "tokens": ["invasao", "dispositivo"], "weight": 1.05},
        {"name": "dispositivo_eletronico", "tokens": ["dispositivo", "eletronico"], "weight": 0.9},
    ],
    "lavagem_financeira": [
        {"name": "ocultacao_valores", "tokens": ["ocultacao", "valores"], "weight": 1.0},
        {"name": "dissimulacao_bens", "tokens": ["dissimulacao", "bens"], "weight": 0.95},
    ],
}

MODUS_HINT_TERMS: dict[str, set[str]] = {
    "extracao_ilegal_madeira": {"extracao", "exploracao", "ilegal", "madeira"},
    "garimpo_ilegal": {"garimpo", "mineracao", "ilegal", "ouro"},
    "desmatamento": {"desmatamento", "queimada", "queimadas", "floresta"},
    "trafico_de_especies": {"trafico", "especies", "animais", "silvestres", "fauna"},
    "caca_ilegal": {"caca", "ilegal", "animais", "silvestres"},
    "pesca_ilegal": {"pesca", "ilegal", "redes", "arrasto"},
    "uso_ilegal_solo": {"solo", "terra", "ocupacao", "invasao"},
    "comercializacao_ilegal": {"comercializacao", "ilegal", "comercio", "venda"},
    "transporte_ilegal": {"transporte", "ilegal", "irregular", "carga", "embarcacao"},
    "apreensao_madeira": {"apreensao", "madeira", "toras"},
    "fiscalizacao_ambiental": {"fiscalizacao", "ambiental", "ibama", "funai"},
    "atividade_clandestina": {"atividade", "clandestina", "funcionamento", "ilegal"},
    "risco_ambiental": {"risco", "ambiental", "degradacao", "contaminacao"},
    "arma_fogo": {"arma", "fogo", "armas", "municao", "municoes"},
    "fraude_documental": {"documento", "documentos", "falso", "falsos", "falsificacao"},
    "fraude_digital": {"perfil", "rede", "social", "mensagens", "aplicativo", "conta"},
    "fraude_pix": {"pix", "transferencia", "chave", "bancaria"},
    "arrombamento": {"arrombamento", "arrombada", "rompimento", "porta", "janela"},
    "abordagem_via_publica": {"abordagem", "via", "publica", "rua"},
    "invasao_dispositivo": {"invasao", "dispositivo", "celular", "eletronico", "eletronicos"},
    "lavagem_financeira": {"ocultacao", "dissimulacao", "valores", "bens", "contas"},
    "armazenamento_digital": {"armazenamento", "arquivos", "conteudo", "dispositivo"},
    "compartilhamento_online": {"compartilhamento", "internet", "online", "rede", "social"},
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
    "crimes_migratorios": {"imigracao", "migracao", "migrantes", "passaportes", "vistos", "estrangeiros"},
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
    modus_operandi: list[str]
    active_discriminators: list[dict[str, object]]
    scores: list[dict[str, object]]
    feature_bank: str
    theme_candidate: dict[str, object] | None = None
    memory_binary: str = ""
    memory_active_positions: list[int] | None = None
    memory_version: int = 0
    memory_vocab_size: int = 0

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
            "modus_operandi": self.modus_operandi[:8],
            "feature_bank": self.feature_bank,
            "active_discriminators": self.active_discriminators[:20],
            "scores": self.scores[:5],
            "theme_candidate": self.theme_candidate,
            "memory_binary": self.memory_binary,
            "memory_active_positions": (self.memory_active_positions or [])[:120],
            "memory_active_count": len(self.memory_active_positions or []),
            "memory_version": self.memory_version,
            "memory_vocab_size": self.memory_vocab_size,
            "inference": self.inference.model_dump() if self.inference else None,
        }


def _stable_id(label: str, pattern: str) -> str:
    digest = hashlib.sha1(f"{label}:{pattern}".encode("utf-8")).hexdigest()[:12]
    return f"{canonical_label(label)}__{digest}"


def _rule_kind(value: object, default: str = "crime") -> str:
    kind = canonical_label(str(value or "")).strip()
    return kind if kind in {"crime", "modus"} else default


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


def _token_present_in_words(token: str, words: set[str]) -> bool:
    normalized = canonical_label(token)
    if not normalized or len(normalized) < 4:
        return False
    if normalized in words:
        return True
    return any(word.startswith(normalized) or normalized.startswith(word) for word in words if len(word) >= 4)


def _normalize_memory_token(token: object) -> str:
    normalized = canonical_label(str(token))
    if len(normalized) < 4:
        return ""
    plural_rules = (
        ("coes", "cao"),
        ("oes", "ao"),
        ("ais", "al"),
        ("eis", "el"),
        ("is", "il"),
        ("ns", "m"),
        ("s", ""),
    )
    for suffix, replacement in plural_rules:
        if len(normalized) > len(suffix) + 3 and normalized.endswith(suffix):
            candidate = normalized[: -len(suffix)] + replacement
            if len(candidate) >= 4:
                return candidate
    return normalized


def _memory_tokens_from_discriminator(item: dict[str, object]) -> list[str]:
    output: list[str] = []
    for tokens in _discriminator_token_variants(item):
        for token in tokens:
            normalized = _normalize_memory_token(token)
            if len(normalized) < 4 or normalized in BLOCKED_SENSOR_TERMS:
                continue
            if normalized not in output:
                output.append(normalized)
    return output


def _variant_signature(tokens: Iterable[object]) -> str:
    normalized = sorted(
        {
            token
            for token in (canonical_label(str(value)) for value in tokens)
            if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
        }
    )
    return "|".join(normalized)


def _discriminator_variants(item: dict[str, object]) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    raw_tokens = item.get("tokens", [])
    base_tokens = raw_tokens if isinstance(raw_tokens, list) else _tokens_from_pattern_payload(str(item.get("pattern", "")))
    base_clean = [
        token
        for token in (canonical_label(str(value)) for value in base_tokens)
        if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
    ]
    if base_clean:
        variants.append(
            {
                "name": str(item.get("name", "") or "_".join(base_clean[:4])),
                "tokens": base_clean,
                "source": str(item.get("source", "")),
                "confirmations": int(item.get("confirmations", 0) or 0),
                "variant_signature": _variant_signature(base_clean),
                "is_primary": True,
            }
        )
    raw_variants = item.get("marker_variants", [])
    if isinstance(raw_variants, list):
        for raw_variant in raw_variants:
            if not isinstance(raw_variant, dict):
                continue
            tokens = raw_variant.get("tokens", [])
            if not isinstance(tokens, list):
                continue
            clean = [
                token
                for token in (canonical_label(str(value)) for value in tokens)
                if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
            ]
            if len(clean) < 2:
                continue
            signature = _variant_signature(clean)
            if not signature or any(signature == str(item.get("variant_signature", "")) for item in variants):
                continue
            variants.append(
                {
                    "name": str(raw_variant.get("name", "") or "_".join(clean[:4])),
                    "tokens": clean,
                    "source": str(raw_variant.get("source", "")),
                    "confirmations": int(raw_variant.get("confirmations", 0) or 0),
                    "variant_signature": signature,
                    "is_primary": False,
                }
            )
    return variants


def _discriminator_token_variants(item: dict[str, object]) -> list[list[str]]:
    return [
        [str(token) for token in variant.get("tokens", []) if str(token)]
        for variant in _discriminator_variants(item)
        if isinstance(variant.get("tokens", []), list)
    ]


def _find_compatible_discriminator_owner(
    candidate: dict[str, object],
    discriminators: list[dict[str, object]],
) -> dict[str, object] | None:
    label = canonical_label(str(candidate.get("label", "")))
    kind = _rule_kind(candidate.get("kind", "crime"))
    candidate_tokens = {
        canonical_label(str(token))
        for token in candidate.get("tokens", [])
        if canonical_label(str(token))
    }
    if len(candidate_tokens) < 2:
        return None
    best_owner: dict[str, object] | None = None
    best_score = 0
    label_hints = set(_label_hint_tokens(label)).union(_label_tokens(label))
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        if canonical_label(str(item.get("label", ""))) != label:
            continue
        if _rule_kind(item.get("kind", "crime")) != kind:
            continue
        item_variant_tokens = {
            canonical_label(str(token))
            for variant_tokens in _discriminator_token_variants(item)
            for token in variant_tokens
            if canonical_label(str(token))
        }
        overlap = candidate_tokens.intersection(item_variant_tokens)
        hint_overlap = overlap.intersection(label_hints)
        score = len(hint_overlap) * 10 + len(overlap)
        if score > best_score:
            best_score = score
            best_owner = item
    return best_owner if best_score >= 1 else None


def _append_marker_variant(owner: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    variants = owner.setdefault("marker_variants", [])
    if not isinstance(variants, list):
        variants = []
        owner["marker_variants"] = variants
    tokens = [
        token
        for token in (canonical_label(str(value)) for value in candidate.get("tokens", []))
        if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
    ]
    signature = _variant_signature(tokens)
    if not signature:
        return owner
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        if str(variant.get("variant_signature", "")) == signature:
            variant["confirmations"] = int(variant.get("confirmations", 0) or 0) + 1
            owner["confirmations"] = int(owner.get("confirmations", 0) or 0) + 1
            return owner
    variants.append(
        {
            "name": str(candidate.get("name", "") or "_".join(tokens[:4])),
            "tokens": tokens,
            "source": str(candidate.get("source", "agent3_learned_discriminator")),
            "confirmations": int(candidate.get("confirmations", 1) or 1),
            "variant_signature": signature,
            "rationale": str(candidate.get("rationale", "")),
        }
    )
    owner["confirmations"] = int(owner.get("confirmations", 0) or 0) + 1
    owner["variant_count"] = len([item for item in variants if isinstance(item, dict)]) + 1
    return owner


def sync_feature_memory(payload: dict[str, object]) -> dict[str, object]:
    """Keep a versioned WNN vocabulary where each token has a stable binary position."""
    memory = payload.get("memory_vocab", {})
    if not isinstance(memory, dict):
        memory = {}
    previous_tokens = memory.get("tokens", [])
    vocab: list[str] = []
    if isinstance(previous_tokens, list):
        for token in previous_tokens:
            normalized = _normalize_memory_token(token)
            if len(normalized) >= 4 and normalized not in BLOCKED_SENSOR_TERMS and normalized not in vocab:
                vocab.append(normalized)

    changed = False
    discriminators = payload.get("discriminators", [])
    if not isinstance(discriminators, list):
        discriminators = []
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        for token in _memory_tokens_from_discriminator(item):
            if token not in vocab:
                vocab.append(token)
                changed = True

    token_to_position = {token: index for index, token in enumerate(vocab)}
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        positions = [
            token_to_position[token]
            for token in _memory_tokens_from_discriminator(item)
            if token in token_to_position
        ]
        item["memory_positions"] = positions
        item["memory_width"] = len(vocab)

    old_version = int(memory.get("version", 0) or 0) if isinstance(memory, dict) else 0
    version = old_version if old_version and not changed and len(vocab) == len(previous_tokens or []) else old_version + 1
    if version <= 0:
        version = 1
    payload["memory_vocab"] = {
        "version": version,
        "size": len(vocab),
        "tokens": vocab,
        "token_to_position": token_to_position,
        "padding_policy": "right_zero_fill_for_previous_documents",
    }
    payload["memory_model"] = {
        "kind": "binary_keyword_matrix",
        "unit": "sanitized_discriminator_token",
        "description": "Cada posicao representa uma palavra-chave discriminativa; a noticia vira um vetor binario 1/0.",
    }
    return payload


def binary_memory_for_text(text: str, feature_bank: dict[str, object]) -> dict[str, object]:
    memory = feature_bank.get("memory_vocab", {})
    if not isinstance(memory, dict):
        memory = {}
    tokens = memory.get("tokens", [])
    if not isinstance(tokens, list):
        tokens = []
    normalized_tokens = [
        canonical_label(str(token))
        for token in tokens
        if len(canonical_label(str(token))) >= 4 and canonical_label(str(token)) not in BLOCKED_SENSOR_TERMS
    ]
    words = _token_set_from_text(text)
    active_positions = [
        index
        for index, token in enumerate(normalized_tokens)
        if _token_present_in_words(token, words)
    ]
    active_set = set(active_positions)
    return {
        "version": int(memory.get("version", 0) or 0),
        "vocab_size": len(normalized_tokens),
        "active_count": len(active_positions),
        "active_positions": active_positions,
        "binary": "".join("1" if index in active_set else "0" for index in range(len(normalized_tokens))),
        "active_tokens": [normalized_tokens[index] for index in active_positions[:120]],
    }


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


def _tokens_from_pattern_payload(pattern: str) -> list[str]:
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


def _agent2_modus_marker_sets(terms: Iterable[str]) -> dict[str, list[str]]:
    normalized_terms = [
        token
        for term in terms
        for token in _substantive_tokens(str(term))
    ]
    token_set = set(normalized_terms)
    output: dict[str, list[str]] = {}
    for modus_label, hints in MODUS_HINT_TERMS.items():
        selected = [token for token in normalized_terms if token in hints]
        if len(set(selected)) >= 2:
            deduped: list[str] = []
            for token in selected:
                if token not in deduped:
                    deduped.append(token)
            output[modus_label] = deduped[:4]
            continue
        overlap = [token for token in hints if token in token_set]
        if len(overlap) >= 2:
            output[modus_label] = overlap[:4]
    return output


def _marker_signature(label: str, tokens: Iterable[object], kind: str = "crime") -> str:
    normalized = sorted(
        {
            token
            for token in (canonical_label(str(value)) for value in tokens)
            if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
        }
    )
    if not normalized:
        return ""
    return f"{_rule_kind(kind)}:{canonical_label(label)}:{'|'.join(normalized)}"


def _marker_signature_from_discriminator(item: dict[str, object]) -> str:
    label = canonical_label(str(item.get("label", "")))
    kind = _rule_kind(item.get("kind", "crime"))
    tokens = item.get("tokens")
    if isinstance(tokens, list) and tokens:
        return _marker_signature(label, tokens, kind=kind)
    return _marker_signature(label, _tokens_from_pattern_payload(str(item.get("pattern", ""))), kind=kind)


def learned_rule_to_discriminator(rule: dict[str, object]) -> dict[str, object] | None:
    kind = _rule_kind(rule.get("kind", "crime"))
    original_label = canonical_label(str(rule.get("label", "")))
    label = parent_theme(original_label) if kind == "crime" else normalize_modus_label(original_label)
    if not original_label:
        return None
    if kind == "modus" and not label:
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
        tokens = _tokens_from_pattern_payload(legacy_pattern)
    label_terms = _label_tokens(original_label)
    hint_terms = _label_hint_tokens(original_label)
    priority = [token for token in tokens if token in label_terms]
    micro_world = [token for token in tokens if token in hint_terms and token not in priority]
    support = [token for token in tokens if token not in priority and token not in micro_world]
    selected = [*priority, *micro_world, *support] if kind == "crime" else tokens
    selected = selected[:4]
    if len(selected) < 2:
        return None
    if kind == "crime":
        guard_ok, _guard_reason = _marker_matches_micro_world(label, selected)
        if not guard_ok:
            return None

    unordered = _unordered_pattern(selected)
    signature = _marker_signature(label, selected, kind=kind)
    discriminator = {
        "id": _stable_id(label, signature),
        "kind": kind,
        "label": label,
        "name": canonical_label(str(rule.get("name", ""))) or "_".join(selected),
        "pattern": unordered,
        "weight": EVIDENCE_FEATURE_WEIGHT,
        "source": str(rule.get("source", "agent3_learned_discriminator")),
        "rationale": str(rule.get("rationale", "discriminador WNN aprendido no residual por tokens substantivos")),
        "tokens": selected,
        "marker_signature": signature,
        "confirmations": 1,
        "variant_count": 1,
    }
    discriminator = _apply_strength_metadata(discriminator)
    if kind == "crime":
        return _with_parent_theme(discriminator, original_label)
    return discriminator


def suggest_discriminator_rules_from_review(doc: dict[str, Any], review: Any) -> list[dict[str, object]]:
    label = canonical_label(str(getattr(review, "canonical_label", "") or ""))
    review_modus = getattr(review, "modus_operandi", []) or []
    if not label and not review_modus:
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
        label = ""

    output: list[dict[str, object]] = []
    if label:
        selected: list[str] = []
        prioritized_tokens = [
            *label_terms,
            *[token for token in evidence_tokens if token in hint_terms],
            *[token for token in evidence_tokens if token in label_terms],
        ]
        for token in prioritized_tokens:
            if token not in selected:
                selected.append(token)
            if len(selected) >= 5:
                break
        if len(selected) >= 2 and set(selected).intersection(set(label_terms).union(hint_terms)):
            output.append(
                {
                    "kind": "crime",
                    "label": label,
                    "name": "_".join(selected[:4]),
                    "tokens": selected,
                    "source": "agent3_learned_discriminator",
                    "rationale": "marcador WNN aprendido a partir da classificacao residual do Agente 3",
                }
            )

    for raw_modus in review_modus:
        modus_label = normalize_modus_label(raw_modus)
        if not modus_label:
            continue
        modus_terms = [token for token in modus_label.split("_") if len(token) >= 4]
        selected_modus: list[str] = []
        prioritized_modus = [
            *modus_terms,
            *[token for token in evidence_tokens if token in modus_terms],
        ]
        for token in prioritized_modus:
            if token not in selected_modus:
                selected_modus.append(token)
            if len(selected_modus) >= 5:
                break
        if len(selected_modus) < 2 or not set(selected_modus).intersection(modus_terms):
            continue
        output.append(
            {
                "kind": "modus",
                "label": modus_label,
                "name": "_".join(selected_modus[:4]),
                "tokens": selected_modus,
                "source": "agent3_learned_modus_discriminator",
                "rationale": "marcador WNN de modus operandi aprendido a partir da classificacao residual do Agente 3",
            }
        )
    return output


def _discriminator_priority(item: dict[str, object]) -> tuple[int, int, float, str]:
    source = str(item.get("source", "") or "")
    curated = 1 if "curated" in source else 0
    confirmations = int(item.get("confirmations", 0) or 0)
    variant_count = int(item.get("variant_count", 1) or 1)
    weight = float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT)
    return (curated, variant_count, confirmations, weight, source)


def _cap_discriminators_per_label(
    discriminators: list[dict[str, object]],
    max_per_label: int,
) -> list[dict[str, object]]:
    if max_per_label <= 0:
        return discriminators
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for item in discriminators:
        key = (
            _rule_kind(item.get("kind", "crime")),
            canonical_label(str(item.get("label", ""))),
        )
        grouped.setdefault(key, []).append(item)
    limited: list[dict[str, object]] = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=_discriminator_priority, reverse=True)
        limited.extend(items[:max_per_label])
    return limited


def append_discriminators_from_learned_rules(
    rules: list[dict[str, object]],
    feature_bank_path: Path | str,
    max_discriminators_per_label: int = 35,
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
    existing_variant_owner: dict[str, dict[str, object]] = {}
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        for variant in _discriminator_variants(item):
            signature = str(variant.get("variant_signature", ""))
            if signature:
                existing_variant_owner[signature] = item
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
        variant_signature = _variant_signature(discriminator.get("tokens", []))
        variant_owner = existing_variant_owner.get(variant_signature)
        if variant_owner is not None:
            _append_marker_variant(variant_owner, discriminator)
            _apply_strength_metadata(variant_owner)
            added.append(variant_owner)
            continue
        compatible_owner = _find_compatible_discriminator_owner(discriminator, discriminators)
        if compatible_owner is not None:
            _append_marker_variant(compatible_owner, discriminator)
            existing_variant_owner[variant_signature] = compatible_owner
            _apply_strength_metadata(compatible_owner)
            added.append(compatible_owner)
            continue
        discriminators.append(discriminator)
        existing_by_signature[signature] = discriminator
        existing_variant_owner[variant_signature] = discriminator
        added.append(discriminator)

    discriminators = _cap_discriminators_per_label(
        [item for item in discriminators if isinstance(item, dict)],
        max_per_label=max_discriminators_per_label,
    )
    payload["discriminators"] = discriminators

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
    sync_feature_memory(payload)

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
            clean_tokens = _tokens_from_pattern_payload(str(item.get("pattern", "")))
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
        kind = _rule_kind(item.get("kind", "crime"))
        item["tokens"] = clean_tokens
        item["kind"] = kind
        item["label"] = parent
        if kind == "crime" and parent != original_label:
            item["subtheme"] = original_label
            item["parent_theme"] = parent
        else:
            item.pop("subtheme", None)
            item.pop("parent_theme", None)
        signature = _marker_signature(parent, clean_tokens, kind=kind)
        if not signature:
            continue
        if signature in seen:
            continue
        item["marker_signature"] = signature
        item["pattern"] = _unordered_pattern(clean_tokens)
        cleaned_variants: list[dict[str, object]] = []
        for variant in _discriminator_variants(item):
            if bool(variant.get("is_primary")):
                continue
            variant_tokens = [
                token
                for token in (canonical_label(str(value)) for value in variant.get("tokens", []))
                if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
            ]
            variant_signature = _variant_signature(variant_tokens)
            if not variant_signature or variant_signature == signature:
                continue
            if any(str(existing.get("variant_signature", "")) == variant_signature for existing in cleaned_variants):
                continue
            cleaned_variants.append(
                {
                    "name": str(variant.get("name", "") or "_".join(variant_tokens[:4])),
                    "tokens": variant_tokens,
                    "source": str(variant.get("source", item.get("source", ""))),
                    "confirmations": int(variant.get("confirmations", 0) or 0),
                    "variant_signature": variant_signature,
                    "rationale": str(variant.get("rationale", "")),
                }
            )
        if cleaned_variants:
            item["marker_variants"] = cleaned_variants
            item["variant_count"] = len(cleaned_variants) + 1
        else:
            item.pop("marker_variants", None)
            item["variant_count"] = 1
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

    compacted = _cap_discriminators_per_label(compacted, max_per_label=35)
    labels = sorted(
        {
            canonical_label(str(item.get("label", "")))
            for item in compacted
            if item.get("label")
        }
    )
    themes = {}
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
    sync_feature_memory(payload)

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
        "id": _stable_id(label, _marker_signature(label, tokens, kind="crime")),
        "kind": "crime",
        "label": label,
        "name": name or "_".join(tokens),
        "pattern": pattern,
        "tokens": tokens,
        "marker_signature": _marker_signature(label, tokens, kind="crime"),
        "weight": float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT),
        "source": "agent2_curated_discriminator",
        "rationale": f"discriminador substantivo do tema canonico {label}",
        "variant_count": 1,
    }
    return _with_parent_theme(_apply_strength_metadata(discriminator), original_label)


def _curated_modus_discriminator(label: str, item: dict[str, object]) -> dict[str, object] | None:
    label = normalize_modus_label(label)
    if not label:
        return None
    name = canonical_label(str(item.get("name", "")))
    raw_tokens = item.get("tokens", [])
    if not isinstance(raw_tokens, list):
        return None
    tokens = [
        token
        for token in (canonical_label(str(value)) for value in raw_tokens)
        if token and len(token) >= 4 and token not in BLOCKED_SENSOR_TERMS
    ]
    if not tokens:
        return None
    pattern = _unordered_pattern(tokens)
    if not pattern:
        return None
    signature = _marker_signature(label, tokens, kind="modus")
    return _apply_strength_metadata(
        {
            "id": _stable_id(label, signature),
            "kind": "modus",
            "label": label,
            "name": name or "_".join(tokens),
            "pattern": pattern,
            "tokens": tokens,
            "marker_signature": signature,
            "weight": float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT),
            "source": "agent2_curated_modus_discriminator",
            "rationale": f"discriminador de modus operandi {label}",
            "variant_count": 1,
        }
    )


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
            key = _marker_signature(label, tokens, kind="crime")
            if not key or key in seen:
                continue
            seen.add(key)
            discriminator = {
                "id": _stable_id(label, key),
                "kind": "crime",
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
            key = _marker_signature(label, tokens, kind="crime")
            if key in seen:
                continue
            seen.add(key)
            discriminator = {
                    "id": _stable_id(label, key),
                    "kind": "crime",
                    "label": label,
                    "pattern": pattern,
                    "tokens": tokens,
                    "marker_signature": key,
                    "weight": EVIDENCE_FEATURE_WEIGHT,
                    "source": "agent1_evidence_term",
                    "rationale": f"termo de evidencia do tema {label}: {term_text}",
                }
            discriminators_by_label.setdefault(label, []).append(_with_parent_theme(_apply_strength_metadata(discriminator), original_label))

        for modus_label, tokens in _agent2_modus_marker_sets(theme_terms).items():
            discriminator = _curated_modus_discriminator(
                modus_label,
                {"name": modus_label, "tokens": tokens, "weight": EVIDENCE_FEATURE_WEIGHT},
            )
            if discriminator is None:
                continue
            key = str(discriminator.get("marker_signature") or _marker_signature_from_discriminator(discriminator))
            if key in seen:
                continue
            seen.add(key)
            discriminators_by_label.setdefault(modus_label, []).append(discriminator)

    for modus_label, items in sorted(CURATED_MODUS_DISCRIMINATORS.items()):
        for curated in items:
            discriminator = _curated_modus_discriminator(modus_label, curated)
            if discriminator is None:
                continue
            key = str(discriminator.get("marker_signature") or _marker_signature_from_discriminator(discriminator))
            if key in seen:
                continue
            seen.add(key)
            discriminators_by_label.setdefault(modus_label, []).append(discriminator)

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
            if _rule_kind(item.get("kind", "crime")) == "crime" and _tokens_match_text(item.get("tokens", []), words)
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
    sync_feature_memory(payload)
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
    if not isinstance(payload, dict):
        return {}
    discriminators = payload.get("discriminators", [])
    if isinstance(discriminators, list):
        cleaned: list[dict[str, object]] = []
        seen: set[str] = set()
        for item in discriminators:
            if not isinstance(item, dict):
                continue
            if _rule_kind(item.get("kind", "crime")) == "modus":
                normalized = normalize_modus_label(item.get("label", ""))
                if not normalized:
                    continue
                item = dict(item)
                item["label"] = normalized
            signature = str(item.get("marker_signature") or _marker_signature_from_discriminator(item))
            if signature in seen:
                continue
            seen.add(signature)
            cleaned.append(item)
        payload["discriminators"] = cleaned
    return payload


def active_discriminators(text: str, feature_bank: dict[str, object]) -> list[dict[str, object]]:
    words = _token_set_from_text(text)
    active: list[dict[str, object]] = []
    for item in feature_bank.get("discriminators", []):
        if not isinstance(item, dict):
            continue
        matched_variant: dict[str, object] | None = None
        for variant in _discriminator_variants(item):
            variant_tokens = variant.get("tokens", [])
            if isinstance(variant_tokens, list) and _tokens_match_text(variant_tokens, words):
                matched_variant = variant
                break
        if matched_variant is None:
            continue
        active.append(
            {
                "id": str(item.get("id", "")),
                "kind": _rule_kind(item.get("kind", "crime")),
                "label": (
                    parent_theme(str(item.get("label", "")))
                    if _rule_kind(item.get("kind", "crime")) == "crime"
                    else normalize_modus_label(item.get("label", ""))
                ),
                "subtheme": canonical_label(str(item.get("subtheme", ""))),
                "tokens": matched_variant.get("tokens", []) if isinstance(matched_variant.get("tokens", []), list) else [],
                "matched_variant_name": str(matched_variant.get("name", "")),
                "variant_count": len(_discriminator_variants(item)),
                "weight": float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT),
                "strength": str(item.get("strength", "medium") or "medium"),
                "confirmations": int(item.get("confirmations", 0) or 0),
                "source": str(item.get("source", "")),
            }
        )
    return active


def _modus_scores(active: list[dict[str, object]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in active:
        if _rule_kind(item.get("kind", "crime")) != "modus":
            continue
        label = canonical_label(str(item.get("label", "")))
        if not label:
            continue
        scores[label] = scores.get(label, 0.0) + float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT)
    return scores


def _top_modus_operandi(active: list[dict[str, object]], limit: int = 6) -> list[str]:
    scores = _modus_scores(active)
    return [
        label
        for label, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


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
    sync_feature_memory(feature_bank)
    memory_state = binary_memory_for_text(text, feature_bank)
    active = active_discriminators(text, feature_bank)
    modus_operandi = _top_modus_operandi(active)
    if not active:
        return WNNClassification(
            None,
            "abstain_insufficient_features",
            0.0,
            0.0,
            "",
            [],
            active,
            [],
            str(feature_bank_path),
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
        )

    crime_active = [item for item in active if _rule_kind(item.get("kind", "crime")) == "crime"]
    if not crime_active:
        status = "abstain_only_modus" if modus_operandi else "abstain_insufficient_crime_discriminators"
        return WNNClassification(
            None,
            status,
            0.0,
            0.0,
            "",
            modus_operandi,
            active,
            [],
            str(feature_bank_path),
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
        )
    active_ids = {str(item["id"]) for item in active}

    scores_by_label: dict[str, float] = {}
    for item in crime_active:
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
        return WNNClassification(
            None,
            "abstain_no_score",
            0.0,
            0.0,
            "",
            modus_operandi,
            active,
            [],
            str(feature_bank_path),
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
        )

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
            modus_operandi,
            active,
            scores,
            str(feature_bank_path),
            theme_candidate,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
        )

    if not _has_enough_marker_evidence(top_label, secondary, active, cosine_supported, memory_supported):
        return WNNClassification(
            None,
            "abstain_weak_marker_evidence",
            confidence,
            margin,
            top_label,
            modus_operandi,
            active,
            scores,
            str(feature_bank_path),
            theme_candidate,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
        )

    if confidence < confidence_threshold or margin < effective_margin_threshold:
        return WNNClassification(
            None,
            "abstain_ambiguous" if not cosine_supported else "abstain_cosine_supported_but_low_confidence",
            confidence,
            margin,
            top_label,
            modus_operandi,
            active,
            scores,
            str(feature_bank_path),
            theme_candidate,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
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
        modus_operandi=modus_operandi,
    )
    return WNNClassification(
        inference,
        "accepted",
        confidence,
        margin,
        top_label,
        modus_operandi,
        active,
        scores,
        str(feature_bank_path),
        theme_candidate,
        memory_binary=str(memory_state.get("binary", "")),
        memory_active_positions=list(memory_state.get("active_positions", [])),
        memory_version=int(memory_state.get("version", 0) or 0),
        memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
    )
