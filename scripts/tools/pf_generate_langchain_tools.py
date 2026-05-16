from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

try:
    from project_config import PROJECT_ROOT
except ModuleNotFoundError:
    from scripts.project_config import PROJECT_ROOT


DEFAULT_OUTPUT = PROJECT_ROOT / "scripts" / "tools" / "pf_incremental_langchain_tools.py"

TOOLS_SOURCE = '''from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.tools import tool
from pydantic import BaseModel, Field

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

try:
    from pf_regex_classifier import (
        append_learned_rule,
        clean_learned_rules_file,
        known_crime_labels,
        known_modus_labels,
    )
    from project_config import ANALYSIS_DIR, PROJECT_ROOT
    from scripts.incremental.common import ACTIVE_REGEX_BANK_PATH, AGENT2_REGEX_BANK_PATH, THEMES_JSON
except ModuleNotFoundError:
    from scripts.pf_regex_classifier import (
        append_learned_rule,
        clean_learned_rules_file,
        known_crime_labels,
        known_modus_labels,
    )
    from scripts.project_config import ANALYSIS_DIR, PROJECT_ROOT
    from scripts.incremental.common import ACTIVE_REGEX_BANK_PATH, AGENT2_REGEX_BANK_PATH, THEMES_JSON


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class ClusterSummaryArgs(BaseModel):
    cluster_id: int = Field(description="Identificador numerico do cluster.")
    cluster_summary_csv: str = Field(
        default=str(ANALYSIS_DIR / "incremental" / "resumo_clusters_amostra.csv"),
        description="CSV com resumo de clusters da amostra inicial.",
    )
    corpus_csv: str = Field(
        default=str(ANALYSIS_DIR / "incremental" / "cluster_assignments_amostra.csv"),
        description="CSV com atribuicoes de cluster da amostra inicial.",
    )
    sample_size: int = Field(default=8, ge=1, le=30, description="Quantidade maxima de titulos/amostras.")


@tool(args_schema=ClusterSummaryArgs)
def carregar_resumo_cluster(cluster_id: int, cluster_summary_csv: str, corpus_csv: str, sample_size: int = 8) -> str:
    """Carrega termos, labels e amostras de um cluster para apoiar geracao de topico canonico e regex."""
    cluster_path = resolve_project_path(cluster_summary_csv)
    corpus_path = resolve_project_path(corpus_csv)
    if not cluster_path.exists():
        return json.dumps({"erro": f"cluster_summary_csv nao encontrado: {cluster_path}"}, ensure_ascii=False)

    cluster_df = pd.read_csv(cluster_path)
    row_df = cluster_df.loc[cluster_df["cluster_id"].astype(int) == int(cluster_id)]
    if row_df.empty:
        return json.dumps({"erro": f"cluster_id nao encontrado: {cluster_id}"}, ensure_ascii=False)

    row = row_df.iloc[0].to_dict()
    samples: list[dict[str, object]] = []
    if corpus_path.exists():
        corpus_df = pd.read_csv(corpus_path)
        if "cluster_id" in corpus_df.columns:
            subset = corpus_df.loc[corpus_df["cluster_id"].astype("Int64") == int(cluster_id)].head(sample_size)
            columns = [column for column in ["titulo", "subtitulo", "tags", "crime_labels", "modus_labels"] if column in subset.columns]
            samples = subset[columns].fillna("").to_dict(orient="records")

    return json.dumps({"cluster": row, "amostras": samples}, ensure_ascii=False)


class RegexCandidateArgs(BaseModel):
    kind: str = Field(description="Tipo da regra: crime ou modus.")
    label: str = Field(description="Label canonica da regra.")
    pattern: str = Field(description="Expressao regular candidata.")
    positive_texts_json: str = Field(
        default="[]",
        description="Lista JSON de textos positivos onde a regex deveria bater.",
    )
    negative_texts_json: str = Field(
        default="[]",
        description="Lista JSON de textos negativos onde a regex nao deveria bater.",
    )


@tool(args_schema=RegexCandidateArgs)
def validar_regex_candidata(
    kind: str,
    label: str,
    pattern: str,
    positive_texts_json: str = "[]",
    negative_texts_json: str = "[]",
) -> str:
    """Valida uma regex candidata com compilacao e amostras positivas/negativas."""
    result: dict[str, object] = {
        "kind": kind,
        "label": label,
        "pattern": pattern,
        "accepted": False,
        "errors": [],
        "positive_hits": 0,
        "negative_hits": 0,
    }
    if kind not in {"crime", "modus"}:
        result["errors"].append("kind deve ser crime ou modus")
    if not pattern.strip():
        result["errors"].append("pattern vazio")

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        result["errors"].append(f"regex invalida: {exc}")
        return json.dumps(result, ensure_ascii=False)

    try:
        positives = json.loads(positive_texts_json or "[]")
        negatives = json.loads(negative_texts_json or "[]")
    except json.JSONDecodeError as exc:
        result["errors"].append(f"amostras JSON invalidas: {exc}")
        return json.dumps(result, ensure_ascii=False)

    if not isinstance(positives, list):
        positives = []
    if not isinstance(negatives, list):
        negatives = []

    result["positive_hits"] = sum(1 for text in positives if compiled.search(str(text)))
    result["negative_hits"] = sum(1 for text in negatives if compiled.search(str(text)))
    if positives and int(result["positive_hits"]) == 0:
        result["errors"].append("regex nao bateu em nenhuma amostra positiva")
    if int(result["negative_hits"]) > 0:
        result["errors"].append("regex bateu em amostra negativa")

    result["accepted"] = not result["errors"]
    return json.dumps(result, ensure_ascii=False)


class IncorporarRegexArgs(BaseModel):
    kind: str = Field(description="Tipo da regra: crime ou modus.")
    label: str = Field(description="Label canonica da regra.")
    pattern: str = Field(description="Expressao regular aprovada.")
    title: str = Field(default="", description="Titulo, cluster ou evidencia de origem da regra.")
    source: str = Field(default="agent3_llm_residual", description="Origem da regra: agent2_initial_regex ou agent3_llm_residual.")


@tool(args_schema=IncorporarRegexArgs)
def incorporar_regex_aprendida(kind: str, label: str, pattern: str, title: str = "", source: str = "agent3_llm_residual") -> str:
    """Incorpora uma regex aprovada ao banco ativo consumido nos lotes."""
    accepted = append_learned_rule(kind=kind, label=label, pattern=pattern, title=title, path=ACTIVE_REGEX_BANK_PATH, source=source)
    stats = clean_learned_rules_file(ACTIVE_REGEX_BANK_PATH)
    return json.dumps({"accepted": accepted, "clean_stats": stats}, ensure_ascii=False)


@tool
def carregar_temas_canonicos(_: str = "") -> str:
    """Carrega os temas canonicos aprovados pelo Agente 1."""
    if not THEMES_JSON.exists():
        return json.dumps({"erro": f"temas nao encontrados: {THEMES_JSON}"}, ensure_ascii=False)
    payload = json.loads(THEMES_JSON.read_text(encoding="utf-8"))
    themes = [
        theme
        for theme in payload.get("themes", [])
        if isinstance(theme, dict) and theme.get("decision") == "accept"
    ]
    return json.dumps({"source": str(THEMES_JSON), "themes": themes}, ensure_ascii=False)


@tool
def carregar_banco_regex(_: str = "") -> str:
    """Carrega o banco inicial do Agente 2 e o banco ativo de regex."""
    def read(path: Path) -> object:
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    return json.dumps(
        {
            "agent2_regex_bank_path": str(AGENT2_REGEX_BANK_PATH),
            "agent2_regex_bank": read(AGENT2_REGEX_BANK_PATH),
            "active_regex_bank_path": str(ACTIVE_REGEX_BANK_PATH),
            "active_regex_bank": read(ACTIVE_REGEX_BANK_PATH),
        },
        ensure_ascii=False,
    )


@tool
def listar_labels_conhecidas(_: str = "") -> str:
    """Lista labels de crime e modus atualmente conhecidas pelo classificador regex."""
    payload = {
        "crime_labels": sorted(known_crime_labels()),
        "modus_labels": sorted(known_modus_labels()),
    }
    return json.dumps(payload, ensure_ascii=False)


tools = [
    carregar_resumo_cluster,
    carregar_temas_canonicos,
    carregar_banco_regex,
    validar_regex_candidata,
    incorporar_regex_aprendida,
    listar_labels_conhecidas,
]
tools_json = [convert_to_openai_function(item) for item in tools]
tools_run = {item.name: item for item in tools}


if __name__ == "__main__":
    print(json.dumps({"tools": [item.name for item in tools]}, ensure_ascii=False, indent=2))
'''


def generate_tools_file(output_path: Path = DEFAULT_OUTPUT, overwrite: bool = True) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Arquivo ja existe: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(TOOLS_SOURCE, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera as tools LangChain do ciclo incremental da PF.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Arquivo Python de saida.")
    parser.add_argument("--no-overwrite", action="store_true", help="Nao sobrescreve arquivo existente.")
    args = parser.parse_args()

    output = generate_tools_file(Path(args.output), overwrite=not args.no_overwrite)
    print(f"[langchain-tools] tools geradas em: {output}")


if __name__ == "__main__":
    main()
