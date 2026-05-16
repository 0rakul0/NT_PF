from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

try:
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain_ollama import ChatOllama
except ModuleNotFoundError as exc:  # pragma: no cover - depende do ambiente local.
    raise SystemExit(
        "Dependencias LangChain nao encontradas. Instale langchain e langchain-ollama antes de executar este script."
    ) from exc

try:
    from scripts.schemas.pf_incremental_agent_schemas import InitialRegexAgentResponse, LearningAgentResponse, TopicAgentResponse
    from scripts.tools.pf_incremental_langchain_tools import tools
except ModuleNotFoundError:
    from pf_incremental_agent_schemas import InitialRegexAgentResponse, LearningAgentResponse, TopicAgentResponse
    from pf_incremental_langchain_tools import tools


DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"

TOPIC_AGENT_SYSTEM = """
Voce e o Agente 1 do treinamento incremental da base de noticias da PF.
Sua funcao e analisar clusters semanticos da amostra inicial e bifurcar temas canonicos amplos.
Use as tools para carregar contexto do cluster.
Nao gere regex. Agrupe clusters relacionados em temas canonicos amplos.
A resposta final deve obedecer exatamente ao schema TopicAgentResponse.
Nao existe revisao humana: temas incertos devem ser descartados ou colocados em quarentena automatica.
"""

INITIAL_REGEX_AGENT_SYSTEM = """
Voce e o Agente 2 do treinamento incremental da base de noticias da PF.
Sua funcao e receber um tema canonico aprovado pelo Agente 1 e gerar regex iniciais para esse tema.
Use as tools para validar regex candidatas.
A label da regex deve ser exatamente o tema canonico recebido do Agente 1.
Nao altere a taxonomia. Nao classifique lotes. Nao envie nada para revisao humana.
Regex incerta deve ser rejeitada ou colocada em quarentena automatica.
A resposta final deve obedecer exatamente ao schema InitialRegexAgentResponse.
"""

LEARNING_AGENT_SYSTEM = """
Voce e o Agente 3 do treinamento incremental da base de noticias da PF.
Sua funcao e revisar residuos classificados pela LLM e incorporar regex somente quando forem validas.
Use carregar_temas_canonicos para escolher uma label canonica aprovada pelo Agente 1.
Use carregar_banco_regex para evitar duplicidade e entender o banco ativo.
Use validar_regex_candidata antes de incorporar.
Use incorporar_regex_aprendida com source=agent3_llm_residual para regras aprovadas.
Se houver risco de falso positivo, rejeite ou coloque em quarentena automatica.
A resposta final deve obedecer exatamente ao schema LearningAgentResponse.
Nunca incorpore uma regex sem antes validar a compilacao e exemplos positivos/negativos quando estiverem disponiveis.
"""


def build_llm(model: str, base_url: str) -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url)


def build_agent(
    system_message: str,
    response_schema: type,
    model: str,
    base_url: str,
    verbose: bool = False,
):
    llm = build_llm(model=model, base_url=base_url)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,
        response_format=ToolStrategy(response_schema),
        debug=verbose,
    )


def extract_structured_response(response: dict[str, Any]) -> dict[str, Any]:
    structured = response.get("structured_response")
    if hasattr(structured, "model_dump"):
        return structured.model_dump()
    if isinstance(structured, dict):
        return structured
    return {"raw_response": str(structured or response)}


def run_topic_agent(cluster_id: int, model: str, base_url: str, verbose: bool = False) -> dict[str, Any]:
    agent = build_agent(
        TOPIC_AGENT_SYSTEM,
        TopicAgentResponse,
        model=model,
        base_url=base_url,
        verbose=verbose,
    )
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analise o cluster "
                        f"{cluster_id}. Carregue o resumo do cluster, consulte labels conhecidas e proponha "
                        "somente o topico canonico agregado. Nao gere regex."
                    ),
                }
            ]
        }
    )
    return extract_structured_response(response)


def run_initial_regex_agent(payload: str, model: str, base_url: str, verbose: bool = False) -> dict[str, Any]:
    agent = build_agent(
        INITIAL_REGEX_AGENT_SYSTEM,
        InitialRegexAgentResponse,
        model=model,
        base_url=base_url,
        verbose=verbose,
    )
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Gere regex iniciais para o tema canonico abaixo. Valide automaticamente "
                        "as candidatas e separe aceitas, rejeitadas e quarentenadas.\\n\\n"
                        f"{payload}"
                    ),
                }
            ]
        }
    )
    return extract_structured_response(response)


def run_learning_agent(payload: str, model: str, base_url: str, verbose: bool = False) -> dict[str, Any]:
    agent = build_agent(
        LEARNING_AGENT_SYSTEM,
        LearningAgentResponse,
        model=model,
        base_url=base_url,
        verbose=verbose,
    )
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Revise o aprendizado abaixo. Valide regex candidatas e incorpore apenas as aprovadas.\\n\\n"
                        "A label final deve ser um tema canonico do Agente 1.\\n\\n"
                        f"{payload}"
                    ),
                }
            ]
        }
    )
    return extract_structured_response(response)


def collect_learnings_from_jsonl(jsonl_path: Path, limit: int | None = None) -> dict[str, Any]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL nao encontrado: {jsonl_path}")

    learnings: list[dict[str, Any]] = []
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        learned_rules = record.get("regex_rules_aprendidas", [])
        if not isinstance(learned_rules, list) or not learned_rules:
            continue

        metadata = record.get("metadata_extraido", {})
        inference = record.get("inferencia_llm", {})
        regex_meta = record.get("regex_classificacao", {})
        title = ""
        if isinstance(metadata, dict):
            title = str(metadata.get("titulo", "")).strip()

        for learned_rule in learned_rules:
            if not isinstance(learned_rule, dict):
                continue
            learnings.append(
                {
                    "source_line": line_number,
                    "arquivo": record.get("arquivo", ""),
                    "titulo": title,
                    "fonte_classificacao": record.get("fonte_classificacao", ""),
                    "regex_classificacao": regex_meta if isinstance(regex_meta, dict) else {},
                    "inferencia_llm": inference if isinstance(inference, dict) else {},
                    "learned_rule": learned_rule,
                }
            )
            if limit is not None and len(learnings) >= limit:
                return {"source": str(jsonl_path), "learned_regex_rules": learnings}

    return {"source": str(jsonl_path), "learned_regex_rules": learnings}


def run_learning_agent_from_jsonl(
    jsonl_path: Path,
    model: str,
    base_url: str,
    limit: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    payload = collect_learnings_from_jsonl(jsonl_path, limit=limit)
    if not payload["learned_regex_rules"]:
        return {
            "batch_id": "",
            "decisions": [],
            "incorporated_count": 0,
            "rejected_count": 0,
            "quarantined_count": 0,
            "learned_labels": [],
            "residual_risks": [f"Nenhum aprendizado encontrado em {jsonl_path}"],
            "next_automatic_tests": [],
        }
    return run_learning_agent(
        json.dumps(payload, ensure_ascii=False),
        model=model,
        base_url=base_url,
        verbose=verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa agentes LangChain do ciclo incremental.")
    parser.add_argument("agent", choices=["themes", "topic", "initial-regex", "learning", "learning-jsonl"], help="Agente a executar.")
    parser.add_argument("--cluster-id", type=int, help="Cluster usado pelo agente topic.")
    parser.add_argument("--payload", default="", help="JSON/texto de aprendizado usado pelo agente learning.")
    parser.add_argument("--jsonl", default="", help="JSONL com regex_rules_aprendidas para o agent=learning-jsonl.")
    parser.add_argument("--limit", type=int, help="Limite de aprendizados lidos do JSONL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Ollama.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL do Ollama.")
    parser.add_argument("--verbose", action="store_true", help="Mostra passos do AgentExecutor.")
    args = parser.parse_args()

    if args.agent in {"themes", "topic"}:
        if args.cluster_id is None:
            raise SystemExit("--cluster-id e obrigatorio para agent=themes/topic no scaffold atual")
        result = run_topic_agent(args.cluster_id, model=args.model, base_url=args.base_url, verbose=args.verbose)
    elif args.agent == "initial-regex":
        if not args.payload:
            raise SystemExit("--payload e obrigatorio para agent=initial-regex")
        result = run_initial_regex_agent(args.payload, model=args.model, base_url=args.base_url, verbose=args.verbose)
    elif args.agent == "learning":
        if not args.payload:
            raise SystemExit("--payload e obrigatorio para agent=learning")
        result = run_learning_agent(args.payload, model=args.model, base_url=args.base_url, verbose=args.verbose)
    else:
        if not args.jsonl:
            raise SystemExit("--jsonl e obrigatorio para agent=learning-jsonl")
        result = run_learning_agent_from_jsonl(
            Path(args.jsonl),
            model=args.model,
            base_url=args.base_url,
            limit=args.limit,
            verbose=args.verbose,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
