from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from scripts.agentes.agente_organizador_arvore import run as run_theme_tree_organizer
from scripts.incremental.common import RunConfig


@tool
def organizar_arvore_temas(_: str = "") -> str:
    """Organiza a arvore global de temas com temas canonicos, candidatos, contagens, evidencias, regex e cosseno."""
    result = run_theme_tree_organizer(RunConfig(reset=False))
    return json.dumps(result, ensure_ascii=False, default=str)


tools = [organizar_arvore_temas]
tools_json = [convert_to_openai_function(item) for item in tools]
tools_run = {item.name: item for item in tools}


if __name__ == "__main__":
    print(json.dumps({"tools": [item.name for item in tools]}, ensure_ascii=False, indent=2))
