from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from ollama import Client

try:
    from pf_llm_models import NoticiaEnriquecida, NoticiaLLMInference, NoticiaMetadataExtraido
except ModuleNotFoundError:
    from scripts.pf_llm_models import NoticiaEnriquecida, NoticiaLLMInference, NoticiaMetadataExtraido


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_DIR = PROJECT_ROOT / "data" / "noticias_markdown"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analise_qualitativa"
OUTPUT_JSONL = OUTPUT_DIR / "metadados_llm_noticias.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "metadados_llm_noticias.csv"
MODEL_NAME = os.getenv("PF_LLM_MODEL", "gemma3n:e2b")
OLLAMA_HOST = os.getenv("PF_LLM_HOST", os.getenv("PF_LLM_BASE_URL", "http://localhost:11434"))
TEMPERATURE = 0
MARKDOWN_PATTERN = "*.md"
LIMIT_RAW = os.getenv("PF_LLM_LIMIT", "").strip()
LIMIT = int(LIMIT_RAW) if LIMIT_RAW.isdigit() else None
MARKDOWN_FILES = tuple(sorted(MARKDOWN_DIR.glob(MARKDOWN_PATTERN)))
DATAFRAME_COLUMNS = [
    "arquivo",
    "titulo",
    "subtitulo",
    "data_publicacao",
    "data_atualizacao",
    "tags",
    "dateline",
    "nome_operacao_encontrado",
    "identidade_canonica",
    "classificacao",
    "crimes_mais_presentes",
    "modus_operandi",
]


def normalize_ollama_host(host: str) -> str:
    cleaned = host.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3]
    return cleaned or "http://localhost:11434"


def fold_to_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_title_key(value: str) -> str:
    folded = fold_to_ascii(value).lower()
    folded = re.sub(r"\s+", " ", folded).strip()
    return folded


def parse_news_markdown(markdown_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in markdown_text.splitlines()]
    non_empty = [line for line in lines if line]

    titulo = ""
    if non_empty and non_empty[0].startswith("#"):
        titulo = non_empty[0].lstrip("#").strip()

    subtitulo = non_empty[1] if len(non_empty) > 1 else ""

    publicado_match = re.search(r"Publicado em (\d{2}/\d{2}/\d{4})(?: \d{2}h\d{2})?", markdown_text)
    atualizado_match = re.search(r"Atualizado em (\d{2}/\d{2}/\d{4})(?: \d{2}h\d{2})?", markdown_text)
    tags_match = re.search(r"Tags:\s*(.+)", markdown_text)
    dateline_match = re.search(r"\*\*(.+?)\*\*", markdown_text)

    tags: list[str] = []
    if tags_match:
        tags = [part.strip() for part in tags_match.group(1).split(",") if part.strip()]

    dateline = dateline_match.group(1).strip() if dateline_match else ""

    corpo = markdown_text.strip()
    if dateline_match:
        corpo = markdown_text[dateline_match.end() :].strip()
    elif tags_match:
        corpo = markdown_text[tags_match.end() :].strip()

    return {
        "titulo": titulo,
        "subtitulo": subtitulo,
        "data_publicacao": publicado_match.group(1) if publicado_match else "",
        "data_atualizacao": atualizado_match.group(1) if atualizado_match else "",
        "tags": tags,
        "dateline": dateline,
        "corpo": corpo,
    }


def extract_operation_name_direct(parsed_news: dict[str, Any]) -> str:
    search_space = " ".join(
        [
            str(parsed_news.get("titulo", "")),
            str(parsed_news.get("subtitulo", "")),
            str(parsed_news.get("corpo", ""))[:1200],
        ]
    )
    search_space_ascii = fold_to_ascii(search_space)
    match = re.search(
        r"Operacao\s+([A-Z][A-Za-z0-9-]*(?:\s+[A-Z0-9][A-Za-z0-9-]*){0,5})",
        search_space_ascii,
    )
    if not match:
        return ""

    extracted = match.group(1).strip()
    if extracted.lower() == "pf":
        return ""
    return extracted


def build_extracted_metadata(parsed_news: dict[str, Any]) -> NoticiaMetadataExtraido:
    return NoticiaMetadataExtraido(
        titulo=str(parsed_news["titulo"]),
        subtitulo=str(parsed_news["subtitulo"]),
        data_publicacao=str(parsed_news["data_publicacao"]),
        data_atualizacao=str(parsed_news["data_atualizacao"]),
        tags=list(parsed_news["tags"]),
        dateline=str(parsed_news["dateline"]),
        nome_operacao_encontrado=extract_operation_name_direct(parsed_news),
    )


def build_prompt(news_body: str) -> str:
    schema_json = json.dumps(NoticiaLLMInference.model_json_schema(), ensure_ascii=False, indent=2)
    return f"""
Leia apenas o corpo da noticia abaixo e devolva somente um JSON valido.

Os metadados estruturais ja foram extraidos em outra etapa.
Nao repita titulo, data, tags, dateline nem nome de operacao.

Sua tarefa e inferir somente:
- identidade_canonica
- classificacao
- crimes_mais_presentes
- modus_operandi

Use exatamente este schema:
{schema_json}

Regras:
- identidade_canonica deve ser lowercase com underscores.
- todos os labels devem usar apenas ascii simples, sem acentos, sem cedilha e sem caracteres especiais.
- classificacao deve ser um de: "Por crime", "Com operacao nomeada", "Outras".
- crimes_mais_presentes deve trazer labels canonicos e curtos.
- modus_operandi deve trazer labels canonicos e curtos.
- quando o crime principal estiver claro, prefira uma identidade iniciando com "crime_".
- nao escreva explicacoes fora do JSON.

Corpo da noticia:
{news_body}
""".strip()


def build_client() -> Client:
    return Client(host=normalize_ollama_host(OLLAMA_HOST))


def run_model(client: Client, news_body: str) -> NoticiaLLMInference:
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Voce le noticias da Policia Federal e devolve apenas JSON estruturado "
                    "com identidade canonica, classificacao, crimes e modus operandi."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(news_body),
            },
        ],
        format=NoticiaLLMInference.model_json_schema(),
        options={"temperature": TEMPERATURE},
    )
    return NoticiaLLMInference.model_validate_json(response.message.content)


def build_jsonl_record(file_name: str, registro: NoticiaEnriquecida) -> dict[str, Any]:
    return {
        "arquivo": file_name,
        "metadata_extraido": registro.metadata_extraido.model_dump(),
        "inferencia_llm": registro.inferencia_llm.model_dump(),
    }


def build_dataframe_row(file_name: str, registro: NoticiaEnriquecida) -> dict[str, Any]:
    metadata = registro.metadata_extraido
    inferencia = registro.inferencia_llm
    return {
        "arquivo": file_name,
        "titulo": metadata.titulo,
        "subtitulo": metadata.subtitulo,
        "data_publicacao": metadata.data_publicacao,
        "data_atualizacao": metadata.data_atualizacao,
        "tags": metadata.tags,
        "dateline": metadata.dateline,
        "nome_operacao_encontrado": metadata.nome_operacao_encontrado,
        "identidade_canonica": inferencia.identidade_canonica,
        "classificacao": inferencia.classificacao,
        "crimes_mais_presentes": inferencia.crimes_mais_presentes,
        "modus_operandi": inferencia.modus_operandi,
    }


def build_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=DATAFRAME_COLUMNS)
    return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)


def build_dataframe_row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata_extraido = record.get("metadata_extraido", {})
    inferencia_llm = record.get("inferencia_llm", {})
    if not isinstance(metadata_extraido, dict):
        metadata_extraido = {}
    if not isinstance(inferencia_llm, dict):
        inferencia_llm = {}

    return {
        "arquivo": str(record.get("arquivo", "")).strip(),
        "titulo": str(metadata_extraido.get("titulo", "")).strip(),
        "subtitulo": str(metadata_extraido.get("subtitulo", "")).strip(),
        "data_publicacao": str(metadata_extraido.get("data_publicacao", "")).strip(),
        "data_atualizacao": str(metadata_extraido.get("data_atualizacao", "")).strip(),
        "tags": metadata_extraido.get("tags", []) if isinstance(metadata_extraido.get("tags", []), list) else [],
        "dateline": str(metadata_extraido.get("dateline", "")).strip(),
        "nome_operacao_encontrado": str(metadata_extraido.get("nome_operacao_encontrado", "")).strip(),
        "identidade_canonica": str(inferencia_llm.get("identidade_canonica", "")).strip(),
        "classificacao": str(inferencia_llm.get("classificacao", "")).strip(),
        "crimes_mais_presentes": inferencia_llm.get("crimes_mais_presentes", []) if isinstance(inferencia_llm.get("crimes_mais_presentes", []), list) else [],
        "modus_operandi": inferencia_llm.get("modus_operandi", []) if isinstance(inferencia_llm.get("modus_operandi", []), list) else [],
    }


def load_existing_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if not OUTPUT_JSONL.exists():
        return [], [], set()

    indexed_by_title: dict[str, dict[str, Any]] = {}
    for line in OUTPUT_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        metadata_extraido = record.get("metadata_extraido", {})
        if not isinstance(metadata_extraido, dict):
            continue

        title = str(metadata_extraido.get("titulo", "")).strip()
        if not title:
            continue

        indexed_by_title[normalize_title_key(title)] = record

    jsonl_records = list(indexed_by_title.values())
    dataframe_rows = [build_dataframe_row_from_record(record) for record in jsonl_records]
    processed_titles = {normalize_title_key(row["titulo"]) for row in dataframe_rows if row.get("titulo")}
    return jsonl_records, dataframe_rows, processed_titles


def save_outputs(dataframe: pd.DataFrame, jsonl_records: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for record in jsonl_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_ready = dataframe.copy()
    for column in ["tags", "crimes_mais_presentes", "modus_operandi"]:
        if column in csv_ready:
            csv_ready[column] = csv_ready[column].map(lambda values: json.dumps(values, ensure_ascii=False))
    csv_ready.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def main() -> pd.DataFrame:
    if not MARKDOWN_DIR.exists():
        raise FileNotFoundError(f"Diretorio de markdown nao encontrado: {MARKDOWN_DIR}")
    if not MARKDOWN_FILES:
        raise FileNotFoundError(f"Nenhum arquivo markdown encontrado em: {MARKDOWN_DIR}")

    markdown_files = list(MARKDOWN_FILES[:LIMIT] if LIMIT is not None else MARKDOWN_FILES)
    existing_jsonl_records, existing_dataframe_rows, processed_titles = load_existing_records()
    client = build_client()
    dataframe_rows: list[dict[str, Any]] = list(existing_dataframe_rows)
    jsonl_records: list[dict[str, Any]] = list(existing_jsonl_records)

    if processed_titles:
        print(f"Titulos ja processados encontrados: {len(processed_titles)}")

    for index, markdown_file in enumerate(markdown_files, start=1):
        markdown_text = markdown_file.read_text(encoding="utf-8")
        parsed_news = parse_news_markdown(markdown_text)
        title_key = normalize_title_key(str(parsed_news.get("titulo", "")))

        if title_key and title_key in processed_titles:
            print(f"[{index}/{len(markdown_files)}] pulando {markdown_file.name} porque o titulo ja foi processado.")
            continue

        metadata_extraido = build_extracted_metadata(parsed_news)
        inferencia_llm = run_model(client=client, news_body=str(parsed_news["corpo"]))
        registro = NoticiaEnriquecida(
            metadata_extraido=metadata_extraido,
            inferencia_llm=inferencia_llm,
        )

        dataframe_rows.append(build_dataframe_row(markdown_file.name, registro))
        jsonl_records.append(build_jsonl_record(markdown_file.name, registro))
        if title_key:
            processed_titles.add(title_key)

        print(f"[{index}/{len(markdown_files)}] {markdown_file.name}")
        print(registro.to_readable_block())
        print()

    dataframe = build_dataframe(dataframe_rows)
    save_outputs(dataframe, jsonl_records)

    print(f"Arquivos processados: {len(dataframe)}")
    print(f"JSONL salvo em: {OUTPUT_JSONL}")
    print(f"CSV salvo em: {OUTPUT_CSV}")
    return dataframe


if __name__ == "__main__":
    main()
