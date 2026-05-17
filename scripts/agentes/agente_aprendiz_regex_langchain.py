from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from scripts.tools.pf_regex_learning_tools import tools


DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """
Voce e o Agente Aprendiz de Regex.
Sua unica funcao e chamar a tool aprender_regex_do_residual.
Voce nao classifica noticias e nao cria temas canonicos.
Receba uma noticia residual ja classificada pelo Agente 3 e gere/incorpore regex somente se a tool validar.
Retorne somente o JSON produzido pela tool.
"""


def build_agent(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, verbose: bool = False):
    return create_agent(
        model=ChatOllama(model=model, base_url=base_url),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        debug=verbose,
    )


def run(doc_json: str, review_json: str, negatives_json: str = "[]", model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, verbose: bool = False) -> dict[str, Any]:
    agent = build_agent(model=model, base_url=base_url, verbose=verbose)
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use a tool aprender_regex_do_residual com os argumentos abaixo.\n\n"
                        f"doc_json:\n{doc_json}\n\n"
                        f"review_json:\n{review_json}\n\n"
                        f"negatives_json:\n{negatives_json}"
                    ),
                }
            ]
        }
    )
    messages = response.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip().startswith("{"):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                continue
    return {"raw_response": str(response)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o Agente Aprendiz de Regex.")
    parser.add_argument("--doc-json", required=True)
    parser.add_argument("--review-json", required=True)
    parser.add_argument("--negatives-json", default="[]")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.doc_json, args.review_json, args.negatives_json, model=args.model, base_url=args.base_url, verbose=args.verbose),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
