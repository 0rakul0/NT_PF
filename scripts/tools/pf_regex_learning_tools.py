from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from scripts.agentes.agente_aprendiz_regex import generate_regex_from_review
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse


@tool
def aprender_regex_do_residual(doc_json: str, review_json: str, negatives_json: str = "[]") -> str:
    """Gera, valida e incorpora regex a partir de um residual ja classificado pelo Agente 3."""
    doc = json.loads(doc_json)
    review = ResidualReviewAgentResponse.model_validate(json.loads(review_json))
    negatives = json.loads(negatives_json or "[]")
    if not isinstance(negatives, list):
        negatives = []
    incorporated = generate_regex_from_review(doc, review, [str(item) for item in negatives])
    return json.dumps({"incorporated": incorporated, "incorporated_count": len(incorporated)}, ensure_ascii=False, default=str)


tools = [aprender_regex_do_residual]
tools_json = [convert_to_openai_function(item) for item in tools]
tools_run = {item.name: item for item in tools}


if __name__ == "__main__":
    print(json.dumps({"tools": [item.name for item in tools]}, ensure_ascii=False, indent=2))
