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
GENERIC_CONTEXT_TOKENS = {
    "associacao",
    "criminosa",
    "crime",
    "documento",
    "falso",
    "fraude",
    "ilegal",
    "organizacao",
    "organizado",
}
MAX_SIGNATURES_PER_LABEL = 250
# Preserve room for genuinely new patterns learned from verified residuals.
# Otherwise an initial Agent 2 bank at its cap can discard every new marker.
INITIAL_DISCRIMINATOR_BASELINE = 35
BLOOM_BITS_PER_TOKEN = 12
BLOOM_HASH_COUNT = 7
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
ORGANIZED_CRIME_STRUCTURAL_TOKENS = {
    "integrante",
    "integrantes",
    "lideranca",
    "liderancas",
    "hierarquia",
    "nucleo",
    "membro",
    "membros",
    "divisao",
    "tarefas",
    "estruturada",
}
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
MIN_CONFIRMATIONS_FOR_NEW_LEARNED_MEMORY_TOKEN = 2
CRIME_CONFIDENCE_THRESHOLDS = {
    "crime_organizado": 0.65,
    "corrupcao_desvio_recursos_publicos": 0.50,
    "contrabando_descaminho": 0.50,
    "crimes_contra_criancas": 0.50,
    "crimes_previdenciarios": 0.45,
    "trafico_drogas": 0.45,
}
CRIME_MARGIN_THRESHOLDS = {
    "crime_organizado": 0.28,
    "corrupcao_desvio_recursos_publicos": 0.16,
    "contrabando_descaminho": 0.14,
    "crimes_contra_criancas": 0.12,
    "crimes_previdenciarios": 0.10,
    "trafico_drogas": 0.10,
}
BODY_FALLBACK_CONFIDENCE_BONUS = 0.08
BODY_FALLBACK_MARGIN_BONUS = 0.06
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
        {"name": "divulgacao_pornografia_infantil", "tokens": ["divulgacao", "pornografia", "infantil"], "weight": 1.05},
        {"name": "imagens_abuso_infantil", "tokens": ["imagens", "abuso", "infantil"], "weight": 1.05},
        {"name": "violencia_sexual_infantil", "tokens": ["violencia", "sexual", "infantil"], "weight": 1.1},
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
        {"name": "cigarros_contrabandeados", "tokens": ["cigarros", "contrabandeados"], "weight": 1.15},
        {"name": "mercadorias_contrabandeadas", "tokens": ["mercadorias", "contrabandeadas"], "weight": 1.1},
        {"name": "carga_descaminhada", "tokens": ["carga", "descaminhada"], "weight": 1.05},
        {"name": "cigarros_ilegais", "tokens": ["cigarros", "ilegais"], "weight": 1.05},
        {"name": "mercadoria_estrangeira_irregular", "tokens": ["mercadoria", "estrangeira", "irregular"], "weight": 0.95},
        {"name": "produto_descaminhado", "tokens": ["produto", "descaminhado"], "weight": 0.95},
        {"name": "importacao_irregular", "tokens": ["importacao", "irregular"], "weight": 0.9},
        {"name": "cigarros_origem_estrangeira", "tokens": ["cigarros", "origem", "estrangeira"], "weight": 1.0},
        {"name": "eletronicos_importados_irregulares", "tokens": ["eletronicos", "importados", "irregulares"], "weight": 1.0},
        {"name": "mercadorias_sem_documentacao", "tokens": ["mercadorias", "sem", "documentacao"], "weight": 0.95},
        {"name": "fraude_aduaneira", "tokens": ["fraude", "aduaneira"], "weight": 1.0},
    ],
    "lavagem_dinheiro": [
        {"name": "lavagem_dinheiro", "tokens": ["lavagem", "dinheiro"], "weight": 1.2},
        {"name": "ocultacao_bens", "tokens": ["ocultacao", "bens"], "weight": 1.0},
        {"name": "dissimulacao_valores", "tokens": ["dissimulacao", "valores"], "weight": 1.0},
    ],
    "corrupcao_desvio_recursos_publicos": [
        {"name": "desvio_recursos_publicos", "tokens": ["desvio", "recursos", "publicos"], "weight": 1.2},
        {"name": "vantagem_indevida", "tokens": ["vantagem", "indevida"], "weight": 1.15},
        {"name": "pagamento_propina", "tokens": ["pagamento", "propina"], "weight": 1.15},
        {"name": "corrupcao_ativa_passiva", "tokens": ["corrupcao", "ativa"], "weight": 1.05},
        {"name": "fraude_licitacao", "tokens": ["fraude", "licitacao"], "weight": 1.1},
        {"name": "contratacao_fraudulenta", "tokens": ["contratacao", "fraudulenta"], "weight": 0.95},
        {"name": "peculato_recursos_publicos", "tokens": ["peculato", "recursos", "publicos"], "weight": 1.0},
        {"name": "desvio_verbas_publicas", "tokens": ["desvio", "verbas", "publicas"], "weight": 1.0},
        {"name": "superfaturamento_licitacao", "tokens": ["superfaturamento", "licitacao"], "weight": 1.1},
        {"name": "fraude_contrato_publico", "tokens": ["fraude", "contrato", "publico"], "weight": 1.1},
        {"name": "desvio_emendas_parlamentares", "tokens": ["desvio", "emendas", "parlamentares"], "weight": 1.1},
        {"name": "recursos_federais_desviados", "tokens": ["recursos", "federais", "desviados"], "weight": 1.05},
    ],
    "armas_municoes": [
        {"name": "arma_fogo", "tokens": ["arma", "fogo"], "weight": 1.1},
        {"name": "porte_ilegal_arma", "tokens": ["porte", "ilegal", "arma"], "weight": 1.1},
        {"name": "posse_ilegal_arma", "tokens": ["posse", "ilegal", "arma"], "weight": 1.1},
    ],
    "crimes_previdenciarios": [
        {"name": "fraude_beneficio_previdenciario", "tokens": ["fraude", "beneficio", "previdenciario"], "weight": 1.15},
        {"name": "fraude_inss", "tokens": ["fraude", "inss"], "weight": 1.15},
        {"name": "beneficio_inss_indevido", "tokens": ["beneficio", "inss", "indevido"], "weight": 1.1},
        {"name": "aposentadoria_irregular", "tokens": ["aposentadoria", "irregular"], "weight": 1.05},
        {"name": "pensao_irregular", "tokens": ["pensao", "irregular"], "weight": 1.0},
    ],
    "crime_organizado": [
        {"name": "organizacao_criminosa", "tokens": ["organizacao", "criminosa"], "weight": 1.15},
        {"name": "organizacao_criminosa_integrantes", "tokens": ["organizacao", "criminosa", "integrantes"], "weight": 1.2},
        {"name": "associacao_criminosa_membros", "tokens": ["associacao", "criminosa", "membros"], "weight": 1.2},
        {"name": "faccao_criminosa_liderancas", "tokens": ["faccao", "criminosa", "liderancas"], "weight": 1.2},
        {"name": "quadrilha_estruturada", "tokens": ["quadrilha", "estruturada"], "weight": 1.1},
        {"name": "crime_organizado", "tokens": ["crime", "organizado"], "weight": 1.1},
        {"name": "associacao_criminosa", "tokens": ["associacao", "criminosa"], "weight": 1.0},
        {"name": "faccao_criminosa", "tokens": ["faccao", "criminosa"], "weight": 1.0},
    ],
    "crimes_sistema_financeiro": [
        {"name": "instituicao_financeira", "tokens": ["instituicao", "financeira"], "weight": 1.15},
        {"name": "gestao_fraudulenta", "tokens": ["gestao", "fraudulenta"], "weight": 1.15},
        {"name": "evasao_divisas", "tokens": ["evasao", "divisas"], "weight": 1.15},
        {"name": "operacao_cambio", "tokens": ["operacao", "cambio"], "weight": 1.05},
        {"name": "fraude_bancaria", "tokens": ["fraude", "bancaria"], "weight": 1.05},
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
NON_SPECIFIC_DOMAIN_ANCHORS = {
    "acao",
    "atividade",
    "ambiental",
    "ambientais",
    "apreensao",
    "armazenamento",
    "clandestina",
    "clandestino",
    "comercio",
    "comercializacao",
    "ilegal",
    "ilegais",
    "irregular",
    "irregulares",
    "posse",
    "transporte",
    "uso",
}
MASK_MIN_COVERAGE = 0.67
BLEACHING_COVERAGE_STEPS = (0.75, 0.85, 1.0)
MASK_GENERIC_TOKEN_WEIGHT = 0.20
MASK_SPECIFIC_TOKEN_WEIGHT = 1.0


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
    crime_confidence: float = 0.0
    crime_margin: float = 0.0
    crime_autonomous: bool = False
    modus_confidence: float = 0.0
    modus_evidence_count: int = 0
    theme_candidate: dict[str, object] | None = None
    memory_binary: str = ""
    memory_active_positions: list[int] | None = None
    memory_version: int = 0
    memory_vocab_size: int = 0
    guard_rejected_discriminators: list[dict[str, object]] | None = None
    crime_evidence_source: str = ""
    modus_evidence_source: str = ""
    bleaching_coverage_threshold: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.inference is not None and self.status.startswith("accepted_")

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "top_label": self.top_label,
            "modus_operandi": self.modus_operandi[:8],
            "crime_confidence": round(self.crime_confidence, 4),
            "crime_margin": round(self.crime_margin, 4),
            "crime_autonomous": self.crime_autonomous,
            "modus_confidence": round(self.modus_confidence, 4),
            "modus_evidence_count": int(self.modus_evidence_count),
            "feature_bank": self.feature_bank,
            "active_discriminators": self.active_discriminators[:20],
            "scores": self.scores[:5],
            "theme_candidate": self.theme_candidate,
            "memory_binary": self.memory_binary,
            "memory_active_positions": (self.memory_active_positions or [])[:120],
            "memory_active_count": len(self.memory_active_positions or []),
            "memory_version": self.memory_version,
            "memory_vocab_size": self.memory_vocab_size,
            "guard_rejected_discriminators": (self.guard_rejected_discriminators or [])[:20],
            "crime_evidence_source": self.crime_evidence_source,
            "modus_evidence_source": self.modus_evidence_source,
            "bleaching_coverage_threshold": self.bleaching_coverage_threshold,
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
        raw_tokens = item.get("matched_tokens", item.get("tokens", []))
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


def _qualified_secondary_labels(
    primary: str,
    ranked: list[str],
    scores_by_label: dict[str, float],
    active: list[dict[str, object]],
) -> list[str]:
    """Return conservative secondary labels backed by a complete discriminator.

    Secondary labels are evaluated as a multilabel output. They must therefore
    have stricter evidence than a contextual co-occurrence: a full mask and a
    score of at least one complete, specific discriminator. ``crime_organizado``
    additionally requires an explicit structural signal of the investigated
    group, avoiding labels triggered merely by an institutional mention.
    """
    selected: list[str] = []
    for label in ranked:
        if label == primary or scores_by_label.get(label, 0.0) < 1.0:
            continue
        label_active = [
            item
            for item in active
            if _active_label(item) == label and float(item.get("mask_coverage", 0.0) or 0.0) >= 0.999
        ]
        if not label_active:
            continue
        if label == "crime_organizado":
            organization_tokens = _active_tokens(label_active)
            if not organization_tokens.intersection(ORGANIZED_CRIME_STRUCTURAL_TOKENS):
                continue
        selected.append(label)
    return selected


def _preferred_protected_domain(active_labels: set[str], scores_by_label: dict[str, float]) -> str:
    """Choose a protected domain only when it is competitive with the score leader.

    Domain priority resolves genuine near-ties; it must not override a much
    stronger class because of one incidental marker.
    """
    top_score = max((float(score) for score in scores_by_label.values()), default=0.0)
    candidates = [
        label
        for label in active_labels
        if label in PROTECTED_DOMAIN_PRIORITY
        and scores_by_label.get(label, 0.0) >= 0.5
        and scores_by_label.get(label, 0.0) >= top_score * 0.85
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
    has_organization_bridge = bool(tokens.intersection(STRONG_ORGANIZED_CRIME_BRIDGE_TOKENS))
    has_structural_organization_evidence = has_organization_bridge and bool(
        tokens.intersection(ORGANIZED_CRIME_STRUCTURAL_TOKENS)
    )
    organized_subthemes = sorted(active_labels.intersection(ORGANIZED_OPERATIONAL_SUBTHEMES))

    protected_domain = _preferred_protected_domain(active_labels, scores_by_label)
    if protected_domain:
        secondary = _qualified_secondary_labels(protected_domain, ranked, scores_by_label, active)
        crimes = _dedupe_labels([protected_domain, *secondary])
        relation = "dominio_preferencial"
        if has_organization_bridge and "crime_organizado" in secondary:
            relation = "dominio_preferencial_com_organizacao"
        return protected_domain, crimes, [label for label in crimes if label != protected_domain], relation

    if has_organization_bridge and organized_subthemes:
        operational_domains = [
            label
            for label in active_labels
            if label != "crime_organizado" and scores_by_label.get(label, 0.0) >= 0.75
        ]
        if operational_domains:
            primary = max(operational_domains, key=lambda label: scores_by_label.get(label, 0.0))
            secondary = _qualified_secondary_labels(primary, ranked, scores_by_label, active)
            if has_structural_organization_evidence and "crime_organizado" not in secondary:
                secondary.append("crime_organizado")
            crimes = _dedupe_labels([primary, *secondary])
            relation = (
                "dominio_preferencial_com_organizacao"
                if "crime_organizado" in secondary
                else "dominio_preferencial_contexto_organizacao_suprimida"
            )
            return primary, crimes, [label for label in crimes if label != primary], relation

    top_label = ranked[0]
    secondary = _qualified_secondary_labels(top_label, ranked, scores_by_label, active)
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


def _can_resolve_known_cooccurrence(
    scores_by_label: dict[str, float],
    active: list[dict[str, object]],
    top_label: str,
    secondary: list[str],
    relation: str,
) -> bool:
    """Accept a well-evidenced multi-label case without an LLM tiebreaker.

    This is intentionally stricter than merely registering a composite
    candidate: every co-active class must have a high-coverage lexical marker.
    The result keeps one primary class and records the others as secondary,
    rather than inventing a fused crime label.
    """
    if relation != "coocorrencia_sem_fusao" or not secondary:
        return False
    decision_labels = _dedupe_labels([top_label, *secondary])
    if len(decision_labels) < 2:
        return False
    ranked_scores = sorted(
        [float(scores_by_label.get(label, 0.0) or 0.0) for label in decision_labels],
        reverse=True,
    )
    if len(ranked_scores) < 2 or ranked_scores[0] <= 0:
        return False
    relative_margin = (ranked_scores[0] - ranked_scores[1]) / ranked_scores[0]
    if ranked_scores[1] < 1.0 or relative_margin > 0.25:
        return False
    for label in decision_labels:
        high_coverage_marker = any(
            _active_label(item) == label
            and float(item.get("mask_coverage", 0.0) or 0.0) >= 0.75
            and len(item.get("matched_tokens", item.get("tokens", [])) or []) >= 2
            for item in active
        )
        if not high_coverage_marker:
            return False
    return True


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


def _variant_is_memory_approved(variant: dict[str, object], owner: dict[str, object]) -> bool:
    """Keep one-off residual wording out of the append-only WNN vocabulary.

    Curated and foundation markers are admitted immediately. A lexical variant
    learned from the residual route needs recurrence before it can allocate a
    new binary position. It may still be kept as an auditable pending variant.
    """
    source = str(variant.get("source", owner.get("source", "")) or "")
    if "agent3_learned" not in source:
        return True
    return int(variant.get("confirmations", 0) or 0) >= MIN_CONFIRMATIONS_FOR_NEW_LEARNED_MEMORY_TOKEN


def _memory_tokens_from_discriminator(item: dict[str, object]) -> list[str]:
    output: list[str] = []
    for variant in _discriminator_variants(item):
        if not _variant_is_memory_approved(variant, item):
            continue
        tokens = variant.get("tokens", [])
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            normalized = _normalize_memory_token(token)
            if len(normalized) < 4 or normalized in BLOCKED_SENSOR_TERMS:
                continue
            if normalized not in output:
                output.append(normalized)
    return output


def _bloom_indexes(token: str, bit_size: int, hash_count: int) -> list[int]:
    """Return deterministic Bloom-filter positions without relying on Python's salted hash."""
    if bit_size <= 0 or hash_count <= 0:
        return []
    encoded = token.encode("utf-8")
    return [
        int.from_bytes(hashlib.blake2b(encoded, digest_size=8, person=index.to_bytes(8)).digest(), "big") % bit_size
        for index in range(hash_count)
    ]


def _build_bloom_filter(tokens: Iterable[str], *, vocab_version: int) -> dict[str, object]:
    unique_tokens = sorted({str(token) for token in tokens if str(token)})
    bit_size = max(64, len(unique_tokens) * BLOOM_BITS_PER_TOKEN)
    bits = bytearray((bit_size + 7) // 8)
    for token in unique_tokens:
        for index in _bloom_indexes(token, bit_size, BLOOM_HASH_COUNT):
            bits[index // 8] |= 1 << (index % 8)
    return {
        "algorithm": "blake2b_double_hash",
        "bit_size": bit_size,
        "hash_count": BLOOM_HASH_COUNT,
        "item_count": len(unique_tokens),
        "vocab_version": vocab_version,
        "bits_hex": bytes(bits).hex(),
    }


def _valid_bloom_filter(bloom: object, *, item_count: int, vocab_version: int) -> bool:
    if not isinstance(bloom, dict):
        return False
    try:
        bit_size = int(bloom.get("bit_size", 0) or 0)
        hash_count = int(bloom.get("hash_count", 0) or 0)
        encoded = str(bloom.get("bits_hex", ""))
        return (
            bloom.get("algorithm") == "blake2b_double_hash"
            and bit_size > 0
            and hash_count > 0
            and int(bloom.get("item_count", -1)) == item_count
            and int(bloom.get("vocab_version", -1)) == vocab_version
            and len(bytes.fromhex(encoded)) >= (bit_size + 7) // 8
        )
    except (TypeError, ValueError):
        return False


def _bloom_might_contain(token: str, bloom: object) -> bool:
    if not isinstance(bloom, dict):
        return True
    try:
        bit_size = int(bloom.get("bit_size", 0) or 0)
        hash_count = int(bloom.get("hash_count", 0) or 0)
        bits = bytes.fromhex(str(bloom.get("bits_hex", "")))
    except (TypeError, ValueError):
        return True
    if bit_size <= 0 or hash_count <= 0 or len(bits) < (bit_size + 7) // 8:
        return True
    return all(bits[index // 8] & (1 << (index % 8)) for index in _bloom_indexes(token, bit_size, hash_count))


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
    sync_discriminator_masks(payload)
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
    existing_bloom = memory.get("bloom_filter") if isinstance(memory, dict) else None
    bloom_filter = (
        existing_bloom
        if _valid_bloom_filter(existing_bloom, item_count=len(vocab), vocab_version=version)
        else _build_bloom_filter(vocab, vocab_version=version)
    )
    payload["memory_vocab"] = {
        "version": version,
        "size": len(vocab),
        "tokens": vocab,
        "token_to_position": token_to_position,
        "padding_policy": "right_zero_fill_for_previous_documents",
        "bloom_filter": bloom_filter,
    }
    payload["memory_model"] = {
        "kind": "binary_keyword_matrix",
        "unit": "sanitized_discriminator_token",
        "description": "Cada posicao representa uma palavra-chave discriminativa; a noticia vira um vetor binario 1/0.",
        "position_policy": "posicoes imutaveis; novos tokens aprovados sao acrescentados ao final; variantes residuais exigem recorrencia",
        "new_learned_token_min_confirmations": MIN_CONFIRMATIONS_FOR_NEW_LEARNED_MEMORY_TOKEN,
        "reverse_memory": "textual_drasiw_position_counters_by_label",
        "bloom_filter": "prefiltro probabilistico seguido de consulta exata token_para_posicao",
    }
    return payload


def binary_memory_for_text(text: str, feature_bank: dict[str, object]) -> dict[str, object]:
    memory = feature_bank.get("memory_vocab", {})
    if not isinstance(memory, dict):
        memory = {}
    tokens = memory.get("tokens", [])
    if not isinstance(tokens, list):
        tokens = []
    normalized_tokens = [canonical_label(str(token)) for token in tokens]
    token_to_position = memory.get("token_to_position", {})
    if not isinstance(token_to_position, dict) or not token_to_position:
        token_to_position = {token: index for index, token in enumerate(normalized_tokens)}
    bloom_filter = memory.get("bloom_filter")
    active_set: set[int] = set()
    bloom_positive_queries = 0
    for word in _token_set_from_text(text):
        candidates = {canonical_label(word), _normalize_memory_token(word)}
        for candidate in candidates:
            if len(candidate) < 4 or candidate in BLOCKED_SENSOR_TERMS:
                continue
            if not _bloom_might_contain(candidate, bloom_filter):
                continue
            bloom_positive_queries += 1
            position = token_to_position.get(candidate)
            if isinstance(position, int) and 0 <= position < len(normalized_tokens):
                active_set.add(position)
    active_positions = sorted(active_set)
    return {
        "version": int(memory.get("version", 0) or 0),
        "vocab_size": len(normalized_tokens),
        "active_count": len(active_positions),
        "active_positions": active_positions,
        "binary": "".join("1" if index in active_set else "0" for index in range(len(normalized_tokens))),
        "active_tokens": [normalized_tokens[index] for index in active_positions[:120]],
        "bloom_enabled": isinstance(bloom_filter, dict),
        "bloom_positive_queries": bloom_positive_queries,
    }


def _record_reverse_memory_observation(
    payload: dict[str, object],
    label: str,
    text: str,
    *,
    source: str,
) -> dict[str, object]:
    """Record a labelled binary input so it can be reconstructed as a class prototype.

    This is the textual DRASiW layer: positions are the stable positions of the
    WNN vocabulary, rather than word offsets in the original document.
    """
    normalized_label = parent_theme(label)
    if not normalized_label:
        return {"recorded": False, "reason": "empty_label"}
    sync_feature_memory(payload)
    state = binary_memory_for_text(text, payload)
    positions = [int(position) for position in state.get("active_positions", [])]
    if not positions:
        return {"recorded": False, "reason": "no_active_positions", "label": normalized_label}

    reverse_memories = payload.setdefault("reverse_memories", {})
    if not isinstance(reverse_memories, dict):
        reverse_memories = {}
        payload["reverse_memories"] = reverse_memories
    memory = reverse_memories.setdefault(
        normalized_label,
        {"sample_count": 0, "active_position_counts": {}, "signatures": [], "source_counts": {}},
    )
    if not isinstance(memory, dict):
        memory = {"sample_count": 0, "active_position_counts": {}, "signatures": [], "source_counts": {}}
        reverse_memories[normalized_label] = memory
    memory["sample_count"] = int(memory.get("sample_count", 0) or 0) + 1
    counts = memory.setdefault("active_position_counts", {})
    if not isinstance(counts, dict):
        counts = {}
        memory["active_position_counts"] = counts
    for position in positions:
        key = str(position)
        counts[key] = int(counts.get(key, 0) or 0) + 1
    signatures = memory.setdefault("signatures", [])
    signature = " ".join(str(position) for position in positions)
    if isinstance(signatures, list) and signature not in signatures and len(signatures) < MAX_SIGNATURES_PER_LABEL:
        signatures.append(signature)
    source_counts = memory.setdefault("source_counts", {})
    if isinstance(source_counts, dict):
        source_counts[source] = int(source_counts.get(source, 0) or 0) + 1
    return {
        "recorded": True,
        "label": normalized_label,
        "active_positions": positions,
        "memory_version": int(state.get("version", 0) or 0),
    }


def record_reverse_memory_observation(
    feature_bank_path: Path | str,
    label: str,
    text: str,
    *,
    source: str = "verified_residual",
) -> dict[str, object]:
    """Persist a verified document as a DRASiW-style textual memory observation."""
    payload = load_feature_bank(feature_bank_path)
    if not payload:
        return {"recorded": False, "reason": "feature_bank_not_found"}
    result = _record_reverse_memory_observation(payload, label, text, source=source)
    if not result.get("recorded"):
        return result
    resolved = Path(feature_bank_path)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def reconstruct_reverse_memory_prototype(
    feature_bank: dict[str, object],
    label: str,
    *,
    limit: int = 12,
) -> dict[str, object]:
    """Reverse stable binary positions into the most representative tokens of a label."""
    normalized_label = parent_theme(label)
    reverse_memories = feature_bank.get("reverse_memories", {})
    if not isinstance(reverse_memories, dict):
        return {"label": normalized_label, "available": False, "reason": "no_reverse_memory"}
    memory = reverse_memories.get(normalized_label, {})
    if not isinstance(memory, dict):
        return {"label": normalized_label, "available": False, "reason": "label_not_observed"}
    vocab = feature_bank.get("memory_vocab", {})
    tokens = vocab.get("tokens", []) if isinstance(vocab, dict) else []
    counts = memory.get("active_position_counts", {})
    if not isinstance(tokens, list) or not isinstance(counts, dict):
        return {"label": normalized_label, "available": False, "reason": "invalid_reverse_memory"}
    sample_count = max(1, int(memory.get("sample_count", 0) or 0))
    ranked: list[dict[str, object]] = []
    for raw_position, raw_count in counts.items():
        try:
            position = int(raw_position)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if 0 <= position < len(tokens) and count > 0:
            ranked.append(
                {
                    "position": position,
                    "token": str(tokens[position]),
                    "count": count,
                    "support": round(count / sample_count, 4),
                }
            )
    ranked.sort(key=lambda item: (-int(item["count"]), int(item["position"])))
    return {
        "label": normalized_label,
        "available": bool(ranked),
        "sample_count": sample_count,
        "memory_version": int(vocab.get("version", 0) or 0) if isinstance(vocab, dict) else 0,
        "prototype": ranked[: max(1, limit)],
        "token_sequence": [str(item["token"]) for item in ranked[: max(1, limit)]],
        "source_counts": memory.get("source_counts", {}),
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
    if normalized_label not in THEME_MICRO_WORLD_ANCHORS:
        return True, "sem vocabulario de guarda para o tema"
    anchors = _theme_anchor_tokens(normalized_label)
    if not anchors:
        return True, "sem vocabulario de guarda para o tema"
    anchor_hits = clean_tokens.intersection(anchors)
    foreign_hits: set[str] = set()
    for other_label, other_anchors in THEME_MICRO_WORLD_ANCHORS.items():
        if other_label == normalized_label:
            continue
        foreign_hits.update(clean_tokens.intersection(other_anchors - anchors))
    specific_anchor_hits = anchor_hits - NON_SPECIFIC_DOMAIN_ANCHORS - {"contra", "crime", "crimes"}
    if foreign_hits and not specific_anchor_hits:
        return False, f"contaminacao cruzada: {', '.join(sorted(foreign_hits)[:4])}"
    if specific_anchor_hits:
        return True, f"ancoras especificas do tema: {', '.join(sorted(specific_anchor_hits)[:4])}"
    if anchor_hits:
        return False, f"somente ancoras genericas: {', '.join(sorted(anchor_hits)[:4])}"
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


def _default_mask_entry(label: str, token: str, confirmations: int = 0) -> dict[str, object]:
    """Create an interpretable weight and role for a word in a discriminator mask."""
    normalized = canonical_label(token)
    if normalized in NON_SPECIFIC_DOMAIN_ANCHORS:
        role, weight = "context", MASK_GENERIC_TOKEN_WEIGHT
    elif normalized in _theme_anchor_tokens(label):
        role, weight = "core", MASK_SPECIFIC_TOKEN_WEIGHT
    else:
        role, weight = "support", 0.55
    reinforcement = min(0.25, max(0, confirmations) * 0.05)
    return {"weight": round(min(1.25, weight + reinforcement), 4), "role": role}


def sync_discriminator_masks(payload: dict[str, object]) -> dict[str, int]:
    """Migrate legacy token lists into persistent, weighted lexical masks."""
    discriminators = payload.get("discriminators", [])
    if not isinstance(discriminators, list):
        return {"discriminators": 0, "words": 0}
    word_count = 0
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        label = parent_theme(str(item.get("label", ""))) if _rule_kind(item.get("kind", "crime")) == "crime" else normalize_modus_label(item.get("label", ""))
        tokens = item.get("tokens", [])
        if not isinstance(tokens, list):
            continue
        existing = item.get("mask", {})
        if not isinstance(existing, dict):
            existing = {}
        try:
            confirmations = int(item.get("confirmations", 0) or 0)
        except (TypeError, ValueError):
            confirmations = 0
        mask: dict[str, dict[str, object]] = {}
        for raw_token in tokens:
            token = canonical_label(str(raw_token))
            if len(token) < 4 or token in BLOCKED_SENSOR_TERMS:
                continue
            previous = existing.get(token)
            default = _default_mask_entry(label, token, confirmations)
            if isinstance(previous, dict):
                try:
                    weight = float(previous.get("weight", default["weight"]) or default["weight"])
                except (TypeError, ValueError):
                    weight = float(default["weight"])
                role = str(previous.get("role", default["role"]) or default["role"])
                mask[token] = {"weight": round(max(0.05, min(1.25, weight)), 4), "role": role}
            elif isinstance(previous, (int, float)):
                mask[token] = {"weight": round(max(0.05, min(1.25, float(previous))), 4), "role": str(default["role"])}
            else:
                mask[token] = default
        if mask:
            item["mask"] = mask
            word_count += len(mask)
    payload["mask_model"] = {
        "kind": "weighted_lexical_mask",
        "roles": {"core": "sinal substantivo", "support": "reforco", "context": "contexto de baixo peso"},
        "score": "cobertura_ponderada_da_mascara",
    }
    return {"discriminators": len([item for item in discriminators if isinstance(item, dict)]), "words": word_count}


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
    if not label:
        return []
    parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
    # The LLM explanation identifies the class, while the document brings the
    # contextual vocabulary that must activate the WNN on a future case.
    evidence = " ".join(
        str(value)
        for value in (
            getattr(review, "evidence_text", ""),
            doc.get("body_text", ""),
            parsed.get("corpo", ""),
            doc.get("context", ""),
        )
        if str(value or "").strip()
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
        anchors = [
            *label_terms,
            *[token for token in evidence_tokens if token in hint_terms],
            *[token for token in evidence_tokens if token in label_terms],
        ]
        for token in anchors:
            if token not in selected:
                selected.append(token)
            if len(selected) >= 2:
                break
        # Add document-specific context so a learned marker is not merely a
        # duplicate of the canonical class name.
        for token in evidence_tokens:
            if token not in selected and len(token) >= 4:
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

    return output


def _discriminator_priority(item: dict[str, object]) -> tuple[int, int, float, str]:
    source = str(item.get("source", "") or "")
    curated = 1 if "curated" in source else 0
    confirmations = int(item.get("confirmations", 0) or 0)
    variant_count = int(item.get("variant_count", 1) or 1)
    weight = float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT)
    return (curated, variant_count, confirmations, weight, source)


def _is_direct_agent3_discriminator(item: dict[str, object]) -> bool:
    """True when a distinct discriminator, not only a variant, came from Agent 3."""
    return "agent3_learned_discriminator" in str(item.get("source", "") or "")


def _adaptive_label_capacity(items: list[dict[str, object]], max_per_label: int) -> int:
    """Grow a class only as its learned patterns are confirmed and reused."""
    if max_per_label <= INITIAL_DISCRIMINATOR_BASELINE:
        return max_per_label
    learned_evidence = 0
    for item in items:
        if _is_direct_agent3_discriminator(item):
            learned_evidence += 1
        for variant in item.get("marker_variants", []):
            if isinstance(variant, dict) and "agent3_learned_discriminator" in str(variant.get("source", "") or ""):
                learned_evidence += max(1, int(variant.get("confirmations", 1) or 1))
    # Five additional slots per verified/reused online pattern lets frequent
    # crimes expand quickly, while unseen or rare classes stay at the compact
    # foundation size.
    return min(max_per_label, INITIAL_DISCRIMINATOR_BASELINE + learned_evidence * 5)


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
        capacity = _adaptive_label_capacity(items, max_per_label)
        learned = [item for item in items if _is_direct_agent3_discriminator(item)]
        initial = [item for item in items if not _is_direct_agent3_discriminator(item)]
        # Variants enrich their existing owner without using a new slot.  For
        # distinct online patterns, give learned rules priority; the adaptive
        # capacity grows with their confirmations up to the global ceiling.
        learned_limit = capacity
        kept_learned = learned[:learned_limit]
        limited.extend([*kept_learned, *initial[: capacity - len(kept_learned)]])
    return limited


def _prune_marker_variants_per_label(
    discriminators: list[dict[str, object]],
    max_per_label: int,
) -> int:
    """Keep the most confirmed learned variants within each adaptive class budget."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for item in discriminators:
        key = (_rule_kind(item.get("kind", "crime")), canonical_label(str(item.get("label", ""))))
        grouped.setdefault(key, []).append(item)

    removed = 0
    for items in grouped.values():
        capacity = _adaptive_label_capacity(items, max_per_label)
        variant_slots = max(0, capacity - len(items))
        candidates: list[tuple[dict[str, object], dict[str, object]]] = []
        for owner in items:
            variants = owner.get("marker_variants", [])
            if not isinstance(variants, list):
                continue
            for variant in variants:
                if isinstance(variant, dict):
                    candidates.append((owner, variant))
        ranked = sorted(
            candidates,
            key=lambda pair: (
                1 if "agent3_learned_discriminator" in str(pair[1].get("source", "") or "") else 0,
                int(pair[1].get("confirmations", 0) or 0),
                str(pair[1].get("variant_signature", "")),
            ),
            reverse=True,
        )
        selected_ids = {id(variant) for _, variant in ranked[:variant_slots]}
        for owner in items:
            variants = owner.get("marker_variants", [])
            if not isinstance(variants, list):
                continue
            kept = [variant for variant in variants if isinstance(variant, dict) and id(variant) in selected_ids]
            removed += len(variants) - len(kept)
            if kept:
                owner["marker_variants"] = kept
                owner["variant_count"] = len(kept) + 1
            else:
                owner.pop("marker_variants", None)
                owner["variant_count"] = 1
            _apply_strength_metadata(owner)
    return removed


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
    retained_ids = {id(item) for item in discriminators}
    # Report only rules that survived the per-class cap.  A proposal generated
    # by Agent 3 is not learning unless it remains in the feature bank.
    return [item for item in added if id(item) in retained_ids]


def compact_feature_bank(
    feature_bank_path: Path | str,
    max_discriminators_per_label: int = 200,
) -> dict[str, int]:
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

    compacted = _cap_discriminators_per_label(compacted, max_per_label=max_discriminators_per_label)
    variants_removed = _prune_marker_variants_per_label(
        compacted,
        max_per_label=max_discriminators_per_label,
    )
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
        "variants_removed": variants_removed,
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
        # Crimes are extracted primarily from the body, where institutional
        # news normally states the legal fact beyond the generic headline.
        text = str(doc.get("x3_texto_noticia", "") or doc.get("body_text", "") or parsed.get("corpo", ""))
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
    for name, label in label_by_doc.items():
        doc = docs_by_name.get(name)
        if not doc:
            continue
        parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
        text = str(doc.get("x3_texto_noticia", "") or doc.get("body_text", "") or parsed.get("corpo", ""))
        _record_reverse_memory_observation(payload, label, text, source="foundation_sample_body")
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


def migrate_feature_bank_to_weighted_masks(feature_bank_path: Path | str) -> dict[str, object]:
    """Persist weighted lexical masks for a legacy WNN feature bank."""
    payload = load_feature_bank(feature_bank_path)
    if not payload:
        return {"migrated": False, "reason": "feature_bank_not_found"}
    summary = sync_discriminator_masks(payload)
    sync_feature_memory(payload)
    resolved = Path(feature_bank_path)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"migrated": True, "feature_bank": str(resolved), **summary}


def active_discriminators(text: str, feature_bank: dict[str, object]) -> list[dict[str, object]]:
    words = _token_set_from_text(text)
    active: list[dict[str, object]] = []
    for item in feature_bank.get("discriminators", []):
        if not isinstance(item, dict):
            continue
        if bool(item.get("quarantined", False)):
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


def mask_discriminators(text: str, feature_bank: dict[str, object]) -> list[dict[str, object]]:
    """Score every discriminator as a binary token mask against the input text."""
    words = _token_set_from_text(text)
    candidates: list[dict[str, object]] = []
    for item in feature_bank.get("discriminators", []):
        if not isinstance(item, dict):
            continue
        if bool(item.get("quarantined", False)):
            continue
        best: dict[str, object] | None = None
        for variant in _discriminator_variants(item):
            raw_tokens = variant.get("tokens", [])
            if not isinstance(raw_tokens, list):
                continue
            tokens = [canonical_label(str(token)) for token in raw_tokens if canonical_label(str(token))]
            if not tokens:
                continue
            matched_tokens = [token for token in tokens if _token_present_in_words(token, words)]
            if not matched_tokens:
                continue
            persisted_mask = item.get("mask", {}) if bool(variant.get("is_primary")) else {}
            if not isinstance(persisted_mask, dict):
                persisted_mask = {}
            label = (
                parent_theme(str(item.get("label", "")))
                if _rule_kind(item.get("kind", "crime")) == "crime"
                else normalize_modus_label(item.get("label", ""))
            )
            mask = {
                token: (
                    persisted_mask.get(token)
                    if isinstance(persisted_mask.get(token), dict)
                    else _default_mask_entry(label, token, int(item.get("confirmations", 0) or 0))
                )
                for token in tokens
            }
            weights = [float(mask[token].get("weight", MASK_SPECIFIC_TOKEN_WEIGHT) or MASK_SPECIFIC_TOKEN_WEIGHT) for token in tokens]
            matched_weight = sum(float(mask[token].get("weight", MASK_SPECIFIC_TOKEN_WEIGHT) or MASK_SPECIFIC_TOKEN_WEIGHT) for token in matched_tokens)
            coverage = matched_weight / sum(weights) if weights else 0.0
            candidate = {
                "id": str(item.get("id", "")),
                "kind": _rule_kind(item.get("kind", "crime")),
                "label": label,
                "subtheme": canonical_label(str(item.get("subtheme", ""))),
                "tokens": tokens,
                "matched_tokens": matched_tokens,
                "mask": mask,
                "mask_coverage": round(coverage, 6),
                "matched_variant_name": str(variant.get("name", "")),
                "variant_count": len(_discriminator_variants(item)),
                "weight": float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT),
                "strength": str(item.get("strength", "medium") or "medium"),
                "confirmations": int(item.get("confirmations", 0) or 0),
                "source": str(item.get("source", "")),
            }
            if best is None or float(candidate["mask_coverage"]) > float(best["mask_coverage"]):
                best = candidate
        if best is not None:
            candidates.append(best)
    return candidates


def apply_domain_guard(active: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Discard crime discriminators that lack a specific anchor for their own domain.

    The feature bank is intentionally left unchanged here: rejected rules remain
    visible for audit and can later be quarantined by bank compaction.
    """
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for item in active:
        if _rule_kind(item.get("kind", "crime")) != "crime":
            accepted.append(item)
            continue
        if canonical_label(str(item.get("label", ""))) not in THEME_MICRO_WORLD_ANCHORS:
            accepted.append(item)
            continue
        tokens = item.get("tokens", [])
        valid, reason = _marker_matches_micro_world(
            str(item.get("label", "")),
            tokens if isinstance(tokens, list) else [],
        )
        if valid:
            accepted.append(item)
            continue
        rejected.append(
            {
                "id": str(item.get("id", "")),
                "label": str(item.get("label", "")),
                "tokens": tokens if isinstance(tokens, list) else [],
                "source": str(item.get("source", "")),
                "reason": reason,
            }
        )
    return accepted, rejected


def downweight_generic_context_discriminators(active: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep cross-domain generic patterns as context, never as decisive evidence."""
    adjusted: list[dict[str, object]] = []
    for item in active:
        tokens = {
            canonical_label(str(token))
            for token in item.get("tokens", [])
            if canonical_label(str(token))
        }
        source = str(item.get("source", "") or "")
        if (
            _rule_kind(item.get("kind", "crime")) == "crime"
            and source.startswith("agent2_generalized")
            and tokens
            and tokens.issubset(GENERIC_CONTEXT_TOKENS)
        ):
            adjusted.append(
                {
                    **item,
                    "weight": round(float(item.get("weight", DEFAULT_FEATURE_WEIGHT) or DEFAULT_FEATURE_WEIGHT) * WEAK_SIGNAL_WEIGHT, 6),
                    "strength": "weak",
                    "generic_context_only": True,
                }
            )
            continue
        adjusted.append(item)
    return adjusted


WEAPON_DIRECT_OFFENCE_ANCHORS = {
    "porte",
    "posse",
    "comercio",
    "venda",
    "trafico",
    "fornecimento",
    "armamento",
    "municao",
    "municoes",
}


def suppress_incidental_weapon_context(active: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Avoid promoting a weapon mention over a complete protected-domain mask.

    Police reports often list an apprehended firearm while the investigated
    offence is child exploitation, environmental crime, or forced labour.  A
    weapon remains a valid primary class only when the text activates a direct
    weapon-offence anchor such as possession, trafficking, sale, or ammunition.
    """
    protected_complete = any(
        _active_label(item) in PROTECTED_DOMAIN_PRIORITY
        and float(item.get("mask_coverage", 0.0) or 0.0) >= 0.999
        for item in active
    )
    weapon_items = [item for item in active if _active_label(item) == "armas_municoes"]
    if not protected_complete or not weapon_items:
        return active, []
    weapon_tokens = _active_tokens(weapon_items)
    if weapon_tokens.intersection(WEAPON_DIRECT_OFFENCE_ANCHORS):
        return active, []
    retained = [item for item in active if _active_label(item) != "armas_municoes"]
    rejected = [
        {
            "id": str(item.get("id", "")),
            "label": "armas_municoes",
            "tokens": item.get("tokens", []),
            "source": str(item.get("source", "")),
            "reason": "incidental_weapon_context_with_protected_domain",
        }
        for item in weapon_items
    ]
    return retained, rejected


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


def _modus_axis_summary(active: list[dict[str, object]], limit: int = 6) -> tuple[list[str], float, int]:
    scores = _modus_scores(active)
    if not scores:
        return [], 0.0, 0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    total = sum(float(score) for _label, score in ranked)
    top = float(ranked[0][1]) if ranked else 0.0
    confidence = (top / total) if total > 0 else 0.0
    labels = [label for label, _score in ranked[:limit]]
    return labels, confidence, len(ranked)


def _with_evidence_source(active: list[dict[str, object]], source: str) -> list[dict[str, object]]:
    """Attach the document field that activated each discriminator for audit."""
    return [{**item, "evidence_source": source} for item in active]


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
    # Generalized patterns formed exclusively by cross-domain legal vocabulary
    # may help rank candidates, but cannot on their own create a crime decision.
    # A full mask match is still insufficient when every active item is context.
    decision_items = [item for item in active if _active_label(item) in {top_label, *secondary}]
    if decision_items and all(bool(item.get("generic_context_only", False)) for item in decision_items):
        return False
    if top_label == "crime_organizado":
        organization_anchors = set()
        for item in active:
            if _active_label(item) != "crime_organizado":
                continue
            matched = item.get("matched_tokens", item.get("tokens", []))
            if isinstance(matched, list):
                organization_anchors.update(matched)
        explicit_crime_organized = {"crime", "organizado"}.issubset(organization_anchors)
        if (
            len(organization_anchors.intersection(STRONG_ORGANIZED_CRIME_BRIDGE_TOKENS)) < 2
            and not explicit_crime_organized
        ):
            return False
        if not organization_anchors.intersection(ORGANIZED_CRIME_STRUCTURAL_TOKENS):
            return False
    counts = _label_evidence_counts(active)
    labels = [top_label, *secondary]
    total = {"strong": 0, "medium": 0, "weak": 0, "confirmed": 0}
    for label in labels:
        for key, value in counts.get(label, {}).items():
            total[key] = total.get(key, 0) + value
    if any(
        _active_label(item) in labels
        and float(item.get("mask_coverage", 0.0) or 0.0) >= 0.999
        for item in active
    ):
        return True
    if total["strong"] >= 1:
        return True
    if total["medium"] >= 2:
        return True
    if total["medium"] >= 1 and (cosine_supported or memory_supported or total["confirmed"] >= 1):
        return True
    return False


def _bleach_ambiguous_discriminators(
    active: list[dict[str, object]],
    scores_by_label: dict[str, float],
    required_margin: float,
) -> tuple[list[dict[str, object]], dict[str, float], float]:
    """Raise the evidence-coverage threshold to resolve a genuine near-tie.

    This is the WNN analogue of bleaching: weak discriminator activations are
    progressively ignored and the class scores are recomputed.  It never invents
    evidence; when no level produces a clear winner, the original ambiguous state
    is preserved for residual review by Agent 3.
    """
    ranked = sorted(scores_by_label.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or ranked[0][1] <= 0:
        return active, scores_by_label, 0.0
    baseline_margin = (ranked[0][1] - ranked[1][1]) / ranked[0][1]
    if baseline_margin >= required_margin:
        return active, scores_by_label, 0.0

    for coverage_threshold in BLEACHING_COVERAGE_STEPS:
        filtered = [
            item for item in active if float(item.get("mask_coverage", 0.0) or 0.0) >= coverage_threshold
        ]
        if len(filtered) == len(active) or not filtered:
            continue
        filtered_scores: dict[str, float] = {}
        for item in filtered:
            label = _active_label(item)
            score = float(item["weight"]) * float(item.get("mask_coverage", 0.0) or 0.0)
            filtered_scores[label] = max(filtered_scores.get(label, 0.0), score)
        filtered_ranked = sorted(filtered_scores.items(), key=lambda item: item[1], reverse=True)
        if not filtered_ranked or filtered_ranked[0][1] <= 0:
            continue
        runner_up = filtered_ranked[1][1] if len(filtered_ranked) > 1 else 0.0
        margin = (filtered_ranked[0][1] - runner_up) / filtered_ranked[0][1]
        if margin >= required_margin:
            return filtered, filtered_scores, coverage_threshold
    return active, scores_by_label, 0.0


def classify_with_wnn(
    text: str,
    feature_bank_path: Path | str,
    confidence_threshold: float = 0.50,
    margin_threshold: float = 0.12,
    min_active_discriminators: int = 2,
    cosine_candidates: list[dict[str, object]] | None = None,
    crime_text: str | None = None,
    modus_text: str | None = None,
    feature_bank_payload: dict[str, object] | None = None,
    sync_memory: bool = True,
    crime_tag_hints: list[str] | None = None,
    class_confidence_overrides: dict[str, float] | None = None,
) -> WNNClassification:
    """Classify the canonical crime from the document body.

    ``text`` preserves the former single-text API. When the optional field-specific
    inputs are provided, crime is first inferred from ``crime_text`` (the body).
    ``text`` and ``modus_text`` are retained for backwards-compatible callers;
    neither represents an additional decision path. ``crime_tag_hints`` is
    accepted but intentionally ignored. The retina is built only from
    ``crime_text`` (x3 / ``texto_noticia``); titles and tags never participate in
    a decision.
    """
    feature_bank = feature_bank_payload if feature_bank_payload is not None else load_feature_bank(feature_bank_path)
    if sync_memory:
        sync_feature_memory(feature_bank)
    crime_input = str(crime_text if crime_text is not None else text or "")
    memory_state = binary_memory_for_text(crime_input, feature_bank)
    crime_full_active = _with_evidence_source(active_discriminators(crime_input, feature_bank), "body")
    masked, guard_rejected = apply_domain_guard(mask_discriminators(crime_input, feature_bank))
    crime_active = [
        {**item, "evidence_source": "body"}
        for item in masked
        if _rule_kind(item.get("kind", "crime")) == "crime"
        and float(item.get("mask_coverage", 0.0) or 0.0) >= MASK_MIN_COVERAGE
    ]
    crime_active, incidental_weapon_rejected = suppress_incidental_weapon_context(crime_active)
    guard_rejected.extend(incidental_weapon_rejected)
    crime_active = downweight_generic_context_discriminators(crime_active)
    crime_evidence_source = "body"
    # The article evaluates a single axis: the canonical crime.  Modus
    # discriminators are deliberately ignored to prevent a second, unvalidated
    # classification task from affecting the incremental decision.
    active = crime_active
    modus_operandi: list[str] = []
    modus_confidence = 0.0
    modus_evidence_count = 0
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
            crime_confidence=0.0,
            crime_margin=0.0,
            crime_autonomous=False,
            modus_confidence=0.0,
            modus_evidence_count=0,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
            guard_rejected_discriminators=guard_rejected,
            crime_evidence_source=crime_evidence_source,
            modus_evidence_source="body",
        )

    if not crime_active:
        status = "abstain_insufficient_crime_discriminators"
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
            crime_confidence=0.0,
            crime_margin=0.0,
            crime_autonomous=False,
            modus_confidence=modus_confidence,
            modus_evidence_count=modus_evidence_count,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
            guard_rejected_discriminators=guard_rejected,
            crime_evidence_source=crime_evidence_source,
            modus_evidence_source="body",
        )
    active_ids = {str(item["id"]) for item in crime_full_active}

    scores_by_label: dict[str, float] = {}
    for item in crime_active:
        label = _active_label(item)
        mask_score = float(item["weight"]) * float(item.get("mask_coverage", 0.0) or 0.0)
        scores_by_label[label] = max(scores_by_label.get(label, 0.0), mask_score)

    crime_active, scores_by_label, bleaching_coverage_threshold = _bleach_ambiguous_discriminators(
        crime_active,
        scores_by_label,
        margin_threshold,
    )
    active = crime_active

    memory_scores_by_label: dict[str, float] = {}
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
            crime_confidence=0.0,
            crime_margin=0.0,
            crime_autonomous=False,
            modus_confidence=modus_confidence,
            modus_evidence_count=modus_evidence_count,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
            guard_rejected_discriminators=guard_rejected,
            crime_evidence_source=crime_evidence_source,
            modus_evidence_source="body",
        )

    top = scores[0]
    second_score = float(scores[1]["score"]) if len(scores) > 1 else 0.0
    total_score = sum(float(item["score"]) for item in scores)
    top_score = float(top["score"])
    top_label = canonical_label(str(top["label"]))
    operational_label, crimes, secondary, relation = _operational_decision(scores_by_label, crime_active)
    if operational_label and operational_label != top_label:
        top_label = operational_label
    theme_candidate = _multi_discriminator_candidate(scores_by_label, crime_active, relation)
    cosine_label, cosine_score = _cosine_top_candidate(cosine_candidates)
    ranked_labels = [label for label, _score in sorted(scores_by_label.items(), key=lambda item: item[1], reverse=True)]
    known_cooccurrence = _can_resolve_known_cooccurrence(
        scores_by_label,
        crime_active,
        top_label,
        secondary,
        relation,
    )
    cosine_tiebreak_used = bool(
        relation in {"tema_unico", "coocorrencia_sem_fusao"}
        and not known_cooccurrence
        and cosine_score >= COSINE_SUPPORT_MIN_SCORE
        and cosine_label
        and cosine_label in ranked_labels[:2]
        and cosine_label != top_label
    )
    if cosine_tiebreak_used:
        top_label = cosine_label
        crimes = [top_label]
        secondary = []
        relation = "cosine_tiebreak"

    if relation in {"cadeia_operacional", "crime_organizado_multidominio"} and top_label == "crime_organizado":
        operational_set = set(crimes)
        top_score = sum(float(scores_by_label.get(label, 0.0) or 0.0) for label in operational_set)
        second_score = max(
            [float(score) for label, score in scores_by_label.items() if label not in operational_set] or [0.0]
        )
    elif relation in {"dominio_preferencial", "dominio_preferencial_com_organizacao", "dominio_preferencial_contexto_organizacao_suprimida"}:
        preferred_set = set(crimes or [top_label])
        top_score = sum(float(scores_by_label.get(label, 0.0) or 0.0) for label in preferred_set)
        second_score = max(
            [
                float(score)
                for label, score in scores_by_label.items()
                if label not in preferred_set
                and not (relation == "dominio_preferencial_contexto_organizacao_suprimida" and label == "crime_organizado")
            ]
            or [0.0]
        )
    if cosine_tiebreak_used:
        top_score = float(scores_by_label.get(top_label, 0.0) or 0.0)
        second_score = max(
            [float(score) for label, score in scores_by_label.items() if label != top_label] or [0.0]
        )
    confidence = top_score / total_score if total_score else 0.0
    margin = (top_score - second_score) / top_score if top_score else 0.0
    crime_confidence = confidence
    crime_margin = margin
    cosine_supported = _cosine_supports_decision(top_label, secondary, cosine_candidates)
    memory_supported = any(memory_scores_by_label.get(label, 0.0) >= 0.25 for label in [top_label, *secondary])
    cosine_suspect, cosine_suspect_label, cosine_suspect_score = _cosine_suspicion(
        top_label,
        secondary,
        cosine_candidates,
    )
    # Classes with measured high precision and low recall may safely use a
    # lower, explicit acceptance threshold. Labels omitted from these maps keep
    # the global configuration; structural crime_organizado remains stricter.
    dynamic_confidence = (class_confidence_overrides or {}).get(top_label)
    class_confidence = float(dynamic_confidence) if dynamic_confidence is not None else CRIME_CONFIDENCE_THRESHOLDS.get(top_label)
    class_margin = CRIME_MARGIN_THRESHOLDS.get(top_label)
    if dynamic_confidence is not None:
        # Dynamic calibration is already bounded and is derived only from
        # completed prior batches; it may move above or below the global value.
        effective_confidence_threshold = float(dynamic_confidence)
        effective_margin_threshold = class_margin if class_margin is not None else margin_threshold
    elif top_label == "crime_organizado":
        effective_confidence_threshold = max(confidence_threshold, class_confidence or 0.0)
        effective_margin_threshold = max(margin_threshold, class_margin or 0.0)
    else:
        effective_confidence_threshold = min(confidence_threshold, class_confidence) if class_confidence is not None else confidence_threshold
        effective_margin_threshold = min(margin_threshold, class_margin) if class_margin is not None else margin_threshold
    if cosine_supported:
        effective_margin_threshold = min(effective_margin_threshold, COSINE_ASSISTED_MARGIN_THRESHOLD)
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
            crime_confidence=crime_confidence,
            crime_margin=crime_margin,
            crime_autonomous=False,
            modus_confidence=modus_confidence,
            modus_evidence_count=modus_evidence_count,
            theme_candidate=theme_candidate,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
            guard_rejected_discriminators=guard_rejected,
            crime_evidence_source=crime_evidence_source,
            modus_evidence_source="body",
        )

    if not _has_enough_marker_evidence(top_label, secondary, crime_active, cosine_supported, memory_supported):
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
            crime_confidence=crime_confidence,
            crime_margin=crime_margin,
            crime_autonomous=False,
            modus_confidence=modus_confidence,
            modus_evidence_count=modus_evidence_count,
            theme_candidate=theme_candidate,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
            guard_rejected_discriminators=guard_rejected,
            crime_evidence_source=crime_evidence_source,
            modus_evidence_source="body",
        )

    # A cosine candidate is generated from x3 and must agree with the WNN
    # leading theme. Once the marker-evidence guard above has passed, this
    # agreement is sufficient to route the document to that theme even when
    # the score confidence or margin alone would cause abstention. Genuine
    # cosine disagreement was already rejected by ``cosine_suspect``.
    if (
        (confidence < effective_confidence_threshold or margin < effective_margin_threshold)
        and not known_cooccurrence
        and not cosine_supported
    ):
        return WNNClassification(
            None,
            "abstain_ambiguous",
            confidence,
            margin,
            top_label,
            modus_operandi,
            active,
            scores,
            str(feature_bank_path),
            crime_confidence=crime_confidence,
            crime_margin=crime_margin,
            crime_autonomous=False,
            modus_confidence=modus_confidence,
            modus_evidence_count=modus_evidence_count,
            theme_candidate=theme_candidate,
            memory_binary=str(memory_state.get("binary", "")),
            memory_active_positions=list(memory_state.get("active_positions", [])),
            memory_version=int(memory_state.get("version", 0) or 0),
            memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
            guard_rejected_discriminators=guard_rejected,
            crime_evidence_source=crime_evidence_source,
            modus_evidence_source="body",
            bleaching_coverage_threshold=bleaching_coverage_threshold,
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
        "accepted_known_cooccurrence"
        if known_cooccurrence
        else "accepted_cosine_tiebreak"
        if cosine_tiebreak_used
        else "accepted_cosine_supported"
        if cosine_supported
        else "accepted_crime",
        confidence,
        margin,
        top_label,
        modus_operandi,
        active,
        scores,
        str(feature_bank_path),
        crime_confidence=crime_confidence,
        crime_margin=crime_margin,
        crime_autonomous=True,
        modus_confidence=modus_confidence,
        modus_evidence_count=modus_evidence_count,
        theme_candidate=theme_candidate,
        memory_binary=str(memory_state.get("binary", "")),
        memory_active_positions=list(memory_state.get("active_positions", [])),
        memory_version=int(memory_state.get("version", 0) or 0),
        memory_vocab_size=int(memory_state.get("vocab_size", 0) or 0),
        guard_rejected_discriminators=guard_rejected,
        crime_evidence_source=crime_evidence_source,
        modus_evidence_source="body",
        bleaching_coverage_threshold=bleaching_coverage_threshold,
    )
