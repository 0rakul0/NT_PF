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

from scripts.tools.pf_theme_tree_tools import tools


DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """
Voce e o Agente Organizador da Arvore de Temas.
Sua unica funcao e chamar a tool organizar_arvore_temas.
Nao execute a funcao do Agente 1 de fundacao e nao altere clusters iniciais.
A tool carrega os temas canonicos atuais, os candidatos do Agente 3, contagens,
evidencias, regex aprendidas e similaridade por cosseno, e grava os artefatos auditaveis.
Retorne somente o JSON produzido pela tool.
"""


def build_agent(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, verbose: bool = False):
    return create_agent(
        model=ChatOllama(model=model, base_url=base_url),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        debug=verbose,
    )


def run(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, verbose: bool = False) -> dict[str, Any]:
    agent = build_agent(model=model, base_url=base_url, verbose=verbose)
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Organize a arvore global de temas chamando a tool dedicada.",
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
    parser = argparse.ArgumentParser(description="Executa o Agente Organizador da Arvore de Temas.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(model=args.model, base_url=args.base_url, verbose=args.verbose), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
