from __future__ import annotations

import os
from pathlib import Path

from scripts import pf_analise_qualitativa, pf_llm_metadata


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MARKDOWN_DIR = BASE_DIR / "data" / "noticias_markdown"
DEFAULT_INDEX_CSV = BASE_DIR / "data" / "pf_operacoes_index.csv"
DEFAULT_CONTENT_CSV = BASE_DIR / "data" / "pf_operacoes_conteudos.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "analise_qualitativa"
DEFAULT_JSONL = DEFAULT_OUTPUT_DIR / "metadados_llm_noticias.jsonl"


def ensure_local_inputs() -> None:
    missing: list[str] = []
    if not DEFAULT_MARKDOWN_DIR.exists():
        missing.append(str(DEFAULT_MARKDOWN_DIR))
    if not DEFAULT_INDEX_CSV.exists():
        missing.append(str(DEFAULT_INDEX_CSV))
    if not DEFAULT_CONTENT_CSV.exists():
        missing.append(str(DEFAULT_CONTENT_CSV))

    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Os arquivos locais necessarios para o modo sem argumentos nao foram encontrados:\n"
            f"{formatted}"
        )


def main() -> None:
    ensure_local_inputs()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skip_llm = os.getenv("PF_SKIP_LLM", "").strip().lower() in {"1", "true", "yes"}

    if not skip_llm:
        print("[run_local] etapa 1/2: extraindo inferencias da LLM com os caminhos padrao...")
        pf_llm_metadata.main()
    else:
        print("[run_local] etapa 1/2: LLM pulada por PF_SKIP_LLM=1. Vou usar o JSONL existente se ele estiver presente.")

    print("[run_local] etapa 2/2: gerando artefatos analiticos com os caminhos padrao...")
    pf_analise_qualitativa.main([])

    print(f"[run_local] concluido. Artefatos em: {DEFAULT_OUTPUT_DIR}")
    print("[run_local] para abrir o painel: .\\.venv\\Scripts\\python.exe -m streamlit run .\\streamlit_app.py")


if __name__ == "__main__":
    main()
