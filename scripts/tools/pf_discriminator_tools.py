from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

try:
    from scripts.incremental.common import CLUSTER_ASSIGNMENTS_CSV, CLUSTER_SUMMARY_CSV, LOTS_DIR, THEMES_JSON, WNN_FEATURE_BANK_PATH
    from scripts.pf_wnn_classifier import (
        append_discriminators_from_learned_rules,
        binary_memory_for_text,
        compact_feature_bank,
        parent_theme,
        reconstruct_reverse_memory_prototype,
        sync_feature_memory,
    )
    from scripts.project_config import PROJECT_ROOT
except ModuleNotFoundError:
    from incremental.common import CLUSTER_ASSIGNMENTS_CSV, CLUSTER_SUMMARY_CSV, LOTS_DIR, THEMES_JSON, WNN_FEATURE_BANK_PATH
    from pf_wnn_classifier import (
        append_discriminators_from_learned_rules,
        binary_memory_for_text,
        compact_feature_bank,
        parent_theme,
        reconstruct_reverse_memory_prototype,
        sync_feature_memory,
    )
    from project_config import PROJECT_ROOT


BLOCKED_DISCRIMINATOR_TOKENS = {
    "acao",
    "apoio",
    "busca",
    "cumpre",
    "deflagra",
    "deflagrou",
    "destaque",
    "federal",
    "mandado",
    "mandados",
    "objetivo",
    "operacao",
    "policia",
    "prisao",
    "regional",
    "segunda",
    "terca",
    "quarta",
    "quinta",
    "sexta",
}


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).lower())
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents).strip()


def canonical_label(value: str) -> str:
    folded = fold_text(value)
    folded = re.sub(r"[^a-z0-9]+", "_", folded)
    return re.sub(r"_+", "_", folded).strip("_")


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_tokens(tokens: list[str]) -> list[str]:
    output: list[str] = []
    for token in tokens:
        cleaned = re.sub(r"[^a-z0-9_]+", "", fold_text(str(token))).strip("_")
        if len(cleaned) < 4 or cleaned in BLOCKED_DISCRIMINATOR_TOKENS:
            continue
        if cleaned not in output:
            output.append(cleaned)
    return output


def token_marker_hits(tokens: list[str], text: str) -> bool:
    folded = fold_text(text)
    return all(token in folded for token in normalize_tokens(tokens))


def read_json(path: Path) -> object:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class CanonicalThemeContextArgs(BaseModel):
    canonical_theme: str = Field(description="Tema canonico aprovado pelo Agente 1.")
    cluster_summary_csv: str = Field(default=str(CLUSTER_SUMMARY_CSV), description="Resumo dos clusters da amostra.")
    corpus_csv: str = Field(default=str(CLUSTER_ASSIGNMENTS_CSV), description="Atribuicoes de documentos aos clusters.")
    sample_size: int = Field(default=12, ge=1, le=50, description="Quantidade maxima de exemplos por tema.")


@tool(args_schema=CanonicalThemeContextArgs)
def carregar_contexto_tema_discriminador(
    canonical_theme: str,
    cluster_summary_csv: str = str(CLUSTER_SUMMARY_CSV),
    corpus_csv: str = str(CLUSTER_ASSIGNMENTS_CSV),
    sample_size: int = 12,
) -> str:
    """Carrega evidencias, clusters e corpos de noticias para o Agente 2 criar discriminadores WNN generalizaveis."""
    theme_id = parent_theme(canonical_theme)
    themes_payload = read_json(THEMES_JSON)
    themes = themes_payload.get("themes", []) if isinstance(themes_payload, dict) else []
    matched_themes = [
        item
        for item in themes
        if isinstance(item, dict)
        and item.get("decision") == "accept"
        and parent_theme(str(item.get("canonical_theme", ""))) == theme_id
    ]
    if not matched_themes:
        return json.dumps({"erro": f"tema canonico nao encontrado: {canonical_theme}"}, ensure_ascii=False)

    cluster_path = resolve_project_path(cluster_summary_csv)
    corpus_path = resolve_project_path(corpus_csv)
    cluster_rows: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    cluster_ids = sorted(
        {
            int(item)
            for theme in matched_themes
            for item in theme.get("included_cluster_ids", [])
            if str(item).lstrip("-").isdigit()
        }
    )

    if cluster_path.exists():
        cluster_df = pd.read_csv(cluster_path)
        if "cluster_id" in cluster_df.columns:
            cluster_rows = cluster_df.loc[cluster_df["cluster_id"].isin(cluster_ids)].fillna("").to_dict(orient="records")

    if corpus_path.exists():
        corpus_df = pd.read_csv(corpus_path)
        if "cluster_id" in corpus_df.columns:
            subset = corpus_df.loc[corpus_df["cluster_id"].isin(cluster_ids)].head(sample_size)
            columns = [column for column in ["arquivo", "body_text", "cluster_text", "cluster_domain_terms"] if column in subset.columns]
            examples = subset[columns].fillna("").to_dict(orient="records")

    return json.dumps(
        {
            "canonical_theme": theme_id,
            "child_themes": [canonical_label(str(item.get("canonical_theme", ""))) for item in matched_themes],
            "description": " | ".join(str(item.get("description", "")) for item in matched_themes if item.get("description")),
            "evidence_terms": [
                term
                for theme in matched_themes
                for term in theme.get("evidence_terms", [])
            ],
            "included_subthemes": [
                subtheme
                for theme in matched_themes
                for subtheme in theme.get("included_subthemes", [])
            ],
            "included_cluster_ids": cluster_ids,
            "clusters": cluster_rows,
            "examples": examples,
        },
        ensure_ascii=False,
    )


class DiscriminatorCandidateArgs(BaseModel):
    label: str = Field(description="Label canonica associada ao discriminador.")
    tokens_json: str = Field(description="Lista JSON de tokens/sinais substantivos, sem dependencia de ordem.")
    positive_texts_json: str = Field(default="[]", description="Lista JSON de textos positivos.")
    negative_texts_json: str = Field(default="[]", description="Lista JSON de textos negativos.")


@tool(args_schema=DiscriminatorCandidateArgs)
def validar_discriminador_wnn(
    label: str,
    tokens_json: str,
    positive_texts_json: str = "[]",
    negative_texts_json: str = "[]",
) -> str:
    """Valida um discriminador WNN baseado em tokens sem ordem fixa."""
    result: dict[str, object] = {
        "label": parent_theme(label),
        "subtheme": canonical_label(label) if parent_theme(label) != canonical_label(label) else "",
        "accepted": False,
        "tokens": [],
        "positive_hits": 0,
        "negative_hits": 0,
        "errors": [],
    }
    try:
        tokens = json.loads(tokens_json or "[]")
        positives = json.loads(positive_texts_json or "[]")
        negatives = json.loads(negative_texts_json or "[]")
    except json.JSONDecodeError as exc:
        result["errors"].append(f"json invalido: {exc}")
        return json.dumps(result, ensure_ascii=False)

    if not isinstance(tokens, list):
        result["errors"].append("tokens_json deve ser lista JSON")
        tokens = []
    if not isinstance(positives, list):
        positives = []
    if not isinstance(negatives, list):
        negatives = []

    normalized_tokens = normalize_tokens([str(item) for item in tokens])
    result["tokens"] = normalized_tokens
    if len(normalized_tokens) < 2:
        result["errors"].append("discriminador precisa de pelo menos dois tokens substantivos")
        return json.dumps(result, ensure_ascii=False)

    result["positive_hits"] = sum(1 for text in positives if token_marker_hits(normalized_tokens, str(text)))
    result["negative_hits"] = sum(1 for text in negatives if token_marker_hits(normalized_tokens, str(text)))
    if positives and int(result["positive_hits"]) == 0:
        result["errors"].append("nao bateu em positivos")
    if negatives and int(result["negative_hits"]) > max(2, int(0.15 * len(negatives))):
        result["errors"].append("muitos acertos em negativos")

    result["accepted"] = not result["errors"]
    return json.dumps(result, ensure_ascii=False)


class IncorporarDiscriminadorArgs(BaseModel):
    label: str = Field(description="Label canonica associada ao discriminador.")
    tokens_json: str = Field(description="Lista JSON de tokens/sinais substantivos.")
    name: str = Field(default="", description="Nome curto do discriminador dentro do tema canonico.")
    source: str = Field(default="agent2_discriminator_tool", description="Origem do discriminador.")
    rationale: str = Field(default="", description="Justificativa curta do discriminador.")


@tool(args_schema=IncorporarDiscriminadorArgs)
def incorporar_discriminador_wnn(
    label: str,
    tokens_json: str,
    name: str = "",
    source: str = "agent2_discriminator_tool",
    rationale: str = "",
) -> str:
    """Incorpora um discriminador WNN ao banco ativo sem criar regra deterministica."""
    try:
        tokens = json.loads(tokens_json or "[]")
    except json.JSONDecodeError as exc:
        return json.dumps({"accepted": False, "erro": f"tokens_json invalido: {exc}"}, ensure_ascii=False)
    if not isinstance(tokens, list):
        return json.dumps({"accepted": False, "erro": "tokens_json deve ser lista JSON"}, ensure_ascii=False)

    normalized_tokens = normalize_tokens([str(item) for item in tokens])
    if len(normalized_tokens) < 2:
        return json.dumps({"accepted": False, "erro": "tokens insuficientes", "tokens": normalized_tokens}, ensure_ascii=False)

    rule = {
        "label": canonical_label(label),
        "name": canonical_label(name) if name else "_".join(normalized_tokens),
        "source": source,
        "rationale": rationale,
        "tokens": normalized_tokens,
    }
    added = append_discriminators_from_learned_rules([rule], WNN_FEATURE_BANK_PATH)
    return json.dumps(
        {
            "accepted": bool(added),
            "feature_bank": str(WNN_FEATURE_BANK_PATH),
            "added": added,
            "tokens": normalized_tokens,
        },
        ensure_ascii=False,
    )
@tool
def carregar_banco_discriminadores(_: str = "") -> str:
    """Carrega resumo do banco ativo de discriminadores WNN."""
    payload = read_json(WNN_FEATURE_BANK_PATH)
    if not isinstance(payload, dict):
        payload = {}
    discriminators = payload.get("discriminators", [])
    labels: dict[str, int] = {}
    if isinstance(discriminators, list):
        for item in discriminators:
            if isinstance(item, dict):
                label = parent_theme(str(item.get("label", "")))
                labels[label] = labels.get(label, 0) + 1
    subthemes: dict[str, int] = {}
    if isinstance(discriminators, list):
        for item in discriminators:
            if isinstance(item, dict) and item.get("subtheme"):
                subtheme = canonical_label(str(item.get("subtheme", "")))
                subthemes[subtheme] = subthemes.get(subtheme, 0) + 1
    return json.dumps(
        {
            "feature_bank": str(WNN_FEATURE_BANK_PATH),
            "discriminator_count": len(discriminators) if isinstance(discriminators, list) else 0,
            "memory_vocab": payload.get("memory_vocab", {}),
            "labels": labels,
            "subthemes": subthemes,
            "sample": discriminators[:30] if isinstance(discriminators, list) else [],
        },
        ensure_ascii=False,
    )


@tool
def carregar_memoria_wnn(_: str = "") -> str:
    """Carrega a matriz/vocabulario binario da memoria WNN com posicoes estaveis."""
    payload = read_json(WNN_FEATURE_BANK_PATH)
    if not isinstance(payload, dict):
        payload = {}
    sync_feature_memory(payload)
    memory = payload.get("memory_vocab", {})
    tokens = memory.get("tokens", []) if isinstance(memory, dict) else []
    return json.dumps(
        {
            "feature_bank": str(WNN_FEATURE_BANK_PATH),
            "version": memory.get("version", 0) if isinstance(memory, dict) else 0,
            "size": len(tokens) if isinstance(tokens, list) else 0,
            "padding_policy": memory.get("padding_policy", "") if isinstance(memory, dict) else "",
            "bloom_filter": {
                key: memory.get("bloom_filter", {}).get(key)
                for key in ["algorithm", "bit_size", "hash_count", "item_count", "vocab_version"]
            }
            if isinstance(memory, dict) and isinstance(memory.get("bloom_filter"), dict)
            else {},
            "positions_sample": [
                {"position": index, "token": token}
                for index, token in enumerate(tokens[:120] if isinstance(tokens, list) else [])
            ],
        },
        ensure_ascii=False,
    )


class ProjetarTextoMemoriaArgs(BaseModel):
    texto: str = Field(description="Corpo da noticia ou trecho ja pre-processado para projetar na memoria binaria.")


@tool(args_schema=ProjetarTextoMemoriaArgs)
def projetar_texto_na_memoria_wnn(texto: str) -> str:
    """Transforma um texto em vetor binario 1/0 usando a memoria WNN atual."""
    payload = read_json(WNN_FEATURE_BANK_PATH)
    if not isinstance(payload, dict):
        payload = {}
    sync_feature_memory(payload)
    state = binary_memory_for_text(texto, payload)
    return json.dumps(
        {
            "feature_bank": str(WNN_FEATURE_BANK_PATH),
            "memory_version": state["version"],
            "vocab_size": state["vocab_size"],
            "active_count": state["active_count"],
            "active_positions": state["active_positions"][:120],
            "active_tokens": state["active_tokens"][:120],
            "bloom_enabled": state.get("bloom_enabled", False),
            "bloom_positive_queries": state.get("bloom_positive_queries", 0),
            "binary_preview": str(state["binary"])[:240],
            "truncated": len(str(state["binary"])) > 240,
        },
        ensure_ascii=False,
    )


class ReconstruirPrototipoDRASiWArgs(BaseModel):
    label: str = Field(description="Label canonica a reconstruir a partir das posicoes binarias mais ativadas.")
    limite: int = Field(default=12, ge=1, le=50, description="Quantidade maxima de tokens representativos.")


@tool(args_schema=ReconstruirPrototipoDRASiWArgs)
def reconstruir_prototipo_drasiw(label: str, limite: int = 12) -> str:
    """Reconstrói o protótipo textual de um tema a partir da memória reversa DRASiW."""
    payload = read_json(WNN_FEATURE_BANK_PATH)
    if not isinstance(payload, dict):
        return json.dumps({"available": False, "reason": "feature_bank_not_found"}, ensure_ascii=False)
    sync_feature_memory(payload)
    return json.dumps(
        reconstruct_reverse_memory_prototype(payload, label, limit=limite),
        ensure_ascii=False,
    )


@tool
def diagnosticar_banco_discriminadores(_: str = "") -> str:
    """Resume riscos do banco WNN: marcadores de 1 token, excesso aprendido e temas inflados."""
    payload = read_json(WNN_FEATURE_BANK_PATH)
    if not isinstance(payload, dict):
        payload = {}
    discriminators = payload.get("discriminators", [])
    if not isinstance(discriminators, list):
        discriminators = []
    by_label: dict[str, int] = {}
    learned_by_label: dict[str, int] = {}
    strength_by_label: dict[str, dict[str, int]] = {}
    single_token: list[dict[str, object]] = []
    for item in discriminators:
        if not isinstance(item, dict):
            continue
        label = canonical_label(str(item.get("label", "")))
        by_label[label] = by_label.get(label, 0) + 1
        strength = str(item.get("strength", "medium") or "medium")
        strength_by_label.setdefault(label, {"strong": 0, "medium": 0, "weak": 0})
        strength_by_label[label][strength if strength in {"strong", "medium", "weak"} else "medium"] += 1
        if str(item.get("source", "")).startswith("agent3_learned"):
            learned_by_label[label] = learned_by_label.get(label, 0) + 1
        tokens = item.get("tokens", [])
        if isinstance(tokens, list) and len(tokens) < 2:
            single_token.append(
                {
                    "id": item.get("id", ""),
                    "label": label,
                    "tokens": tokens,
                    "source": item.get("source", ""),
                }
            )
    return json.dumps(
        {
            "feature_bank": str(WNN_FEATURE_BANK_PATH),
            "total": len(discriminators),
            "labels": dict(sorted(by_label.items(), key=lambda item: -item[1])),
            "learned_by_label": dict(sorted(learned_by_label.items(), key=lambda item: -item[1])),
            "strength_by_label": strength_by_label,
            "single_token_count": len(single_token),
            "single_token_sample": single_token[:40],
        },
        ensure_ascii=False,
    )


@tool
def sanitizar_banco_discriminadores(_: str = "") -> str:
    """Compacta e sanitiza o banco WNN, movendo sinais de 1 token para weak_signals e limitando aprendizado por tema."""
    result = compact_feature_bank(WNN_FEATURE_BANK_PATH)
    return json.dumps({"feature_bank": str(WNN_FEATURE_BANK_PATH), **result}, ensure_ascii=False)


class AuditSampleArgs(BaseModel):
    labels_json: str = Field(default='["crime_organizado","crimes_contra_criancas","crimes_ambientais","trafico_drogas"]')
    per_label: int = Field(default=20, ge=1, le=100)


@tool(args_schema=AuditSampleArgs)
def gerar_amostra_auditoria_wnn(
    labels_json: str = '["crime_organizado","crimes_contra_criancas","crimes_ambientais","trafico_drogas"]',
    per_label: int = 20,
) -> str:
    """Gera CSV de amostra para auditoria manual das classificacoes aceitas pela WNN."""
    try:
        labels = [canonical_label(str(item)) for item in json.loads(labels_json)]
    except json.JSONDecodeError:
        labels = ["crime_organizado", "crimes_contra_criancas", "crimes_ambientais", "trafico_drogas"]
    frames: list[pd.DataFrame] = []
    for path in sorted(LOTS_DIR.glob("lote_*_classificacoes.csv")):
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        return json.dumps({"created": False, "reason": "sem lotes"}, ensure_ascii=False)
    data = pd.concat(frames, ignore_index=True)
    if "wnn_accepted" not in data.columns or "wnn_top_label" not in data.columns:
        return json.dumps({"created": False, "reason": "colunas WNN ausentes"}, ensure_ascii=False)
    output_rows: list[pd.DataFrame] = []
    accepted = data[data["wnn_accepted"].astype(str).str.lower().isin(["true", "1"])]
    for label in labels:
        subset = accepted[accepted["wnn_top_label"].astype(str).map(canonical_label).eq(label)]
        if not subset.empty:
            output_rows.append(subset.head(per_label))
    if not output_rows:
        return json.dumps({"created": False, "reason": "sem amostra para labels", "labels": labels}, ensure_ascii=False)
    audit = pd.concat(output_rows, ignore_index=True)
    columns = [
        column
        for column in [
            "arquivo",
            "titulo",
            "wnn_top_label",
            "wnn_confidence",
            "wnn_margin",
            "wnn_relacao_operacional",
            "wnn_marcadores_secundarios",
            "wnn_active_discriminators_count",
            "wnn_scores",
            "cosine_top_label",
            "cosine_top_score",
        ]
        if column in audit.columns
    ]
    audit = audit[columns]
    output_path = LOTS_DIR.parent / "amostra_auditoria_wnn.csv"
    audit.to_csv(output_path, index=False, encoding="utf-8-sig")
    return json.dumps({"created": True, "output": str(output_path), "rows": len(audit), "labels": labels}, ensure_ascii=False)


tools = [
    carregar_contexto_tema_discriminador,
    validar_discriminador_wnn,
    incorporar_discriminador_wnn,
    carregar_banco_discriminadores,
    carregar_memoria_wnn,
    projetar_texto_na_memoria_wnn,
    reconstruir_prototipo_drasiw,
    diagnosticar_banco_discriminadores,
    sanitizar_banco_discriminadores,
    gerar_amostra_auditoria_wnn,
]
tools_json = [convert_to_openai_function(item) for item in tools]
tools_run = {item.name: item for item in tools}


if __name__ == "__main__":
    print(json.dumps({"tools": [item.name for item in tools]}, ensure_ascii=False, indent=2))
