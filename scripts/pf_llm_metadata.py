from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from ollama import Client

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

try:
    from pf_llm_models import NoticiaEnriquecida, NoticiaLLMInference, NoticiaMetadataExtraido
    from project_config import (
        ANALYSIS_DIR,
        LLM_METADATA_CSV,
        LLM_METADATA_JSONL,
        LLMProviderSettings,
        LLMSettings,
        NEWS_MARKDOWN_DIR,
        PROJECT_ROOT,
        get_llm_settings,
    )
except ModuleNotFoundError:
    from scripts.pf_llm_models import NoticiaEnriquecida, NoticiaLLMInference, NoticiaMetadataExtraido
    from scripts.project_config import (
        ANALYSIS_DIR,
        LLM_METADATA_CSV,
        LLM_METADATA_JSONL,
        LLMProviderSettings,
        LLMSettings,
        NEWS_MARKDOWN_DIR,
        PROJECT_ROOT,
        get_llm_settings,
    )


DEFAULT_MARKDOWN_DIR = NEWS_MARKDOWN_DIR
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR
DEFAULT_OUTPUT_JSONL = LLM_METADATA_JSONL
DEFAULT_OUTPUT_CSV = LLM_METADATA_CSV
LLM_SETTINGS = get_llm_settings()
TEMPERATURE = LLM_SETTINGS.temperature
MAX_RETRIES = LLM_SETTINGS.max_retries
MARKDOWN_PATTERN = "*.md"
CLIENT_CACHE: dict[str, Any] = {}
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


@dataclass(frozen=True)
class RuntimeConfig:
    markdown_dir: Path
    output_dir: Path
    output_jsonl: Path
    output_csv: Path
    limit: int | None
    llm_settings: LLMSettings


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def resolve_project_path(path: Path | None, default: Path) -> Path:
    candidate = path or default
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def normalize_ollama_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3]
    return cleaned or "http://localhost:11434"


def normalize_groq_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/") or "https://api.groq.com/openai/v1"


def infer_provider_from_base_url(base_url: str) -> str | None:
    normalized = base_url.strip().lower()
    if not normalized:
        return None
    if "groq.com" in normalized:
        return "groq"
    if "11434" in normalized or "ollama" in normalized or "localhost" in normalized or "127.0.0.1" in normalized:
        return "ollama"
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--provider", choices=["auto", "ollama", "groq"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        ignored = " ".join(unknown)
        print(f"[llm] ignorando argumentos nao reconhecidos: {ignored}")
    return args


def build_provider_order(preferred_provider: str, has_groq_key: bool) -> tuple[str, ...]:
    if preferred_provider == "groq":
        return ("groq", "ollama")
    if preferred_provider == "ollama":
        return ("ollama", "groq")
    if has_groq_key:
        return ("groq", "ollama")
    return ("ollama", "groq")


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    llm_settings = LLM_SETTINGS
    preferred_provider = llm_settings.preferred_provider
    provider_hint = args.provider or preferred_provider

    inferred_provider = infer_provider_from_base_url(args.base_url or "")
    if args.provider is None and inferred_provider:
        provider_hint = inferred_provider

    groq_settings = llm_settings.groq
    ollama_settings = llm_settings.ollama

    if args.model:
        if provider_hint == "groq":
            groq_settings = replace(groq_settings, model_name=args.model)
        elif provider_hint == "ollama":
            ollama_settings = replace(ollama_settings, model_name=args.model)
        elif inferred_provider == "groq":
            groq_settings = replace(groq_settings, model_name=args.model)
        else:
            ollama_settings = replace(ollama_settings, model_name=args.model)

    if args.base_url:
        if provider_hint == "groq" or inferred_provider == "groq":
            groq_settings = replace(groq_settings, base_url=normalize_groq_base_url(args.base_url))
        else:
            ollama_settings = replace(ollama_settings, base_url=normalize_ollama_base_url(args.base_url))

    provider_order = build_provider_order(provider_hint, has_groq_key=bool(groq_settings.api_key))
    resolved_llm_settings = replace(
        llm_settings,
        preferred_provider=provider_hint,
        provider_order=provider_order,
        groq=groq_settings,
        ollama=ollama_settings,
    )

    output_jsonl = resolve_project_path(args.output_jsonl, DEFAULT_OUTPUT_JSONL)
    output_csv = resolve_project_path(
        args.output_csv,
        output_jsonl.with_suffix(".csv") if args.output_jsonl is not None else DEFAULT_OUTPUT_CSV,
    )
    output_dir = output_jsonl.parent
    limit = args.limit
    if limit is None:
        limit_raw = os.getenv("PF_LLM_LIMIT", "").strip()
        limit = int(limit_raw) if limit_raw.isdigit() else None

    return RuntimeConfig(
        markdown_dir=resolve_project_path(args.input_dir, DEFAULT_MARKDOWN_DIR),
        output_dir=output_dir,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
        limit=limit,
        llm_settings=resolved_llm_settings,
    )


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


def build_messages(news_body: str) -> list[dict[str, str]]:
    return [
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
    ]


def provider_settings_by_name(llm_settings: LLMSettings, provider: str) -> LLMProviderSettings:
    if provider == "groq":
        return llm_settings.groq
    return llm_settings.ollama


def provider_candidates(llm_settings: LLMSettings) -> list[LLMProviderSettings]:
    return [provider_settings_by_name(llm_settings, provider) for provider in llm_settings.provider_order]


def build_client(llm_settings: LLMSettings, provider: str) -> Any:
    provider_settings = provider_settings_by_name(llm_settings, provider)

    if provider == "groq":
        if OpenAI is None:
            raise ModuleNotFoundError(
                "A biblioteca openai nao esta instalada. Rode o setup atualizado ou instale openai para usar PF_LLM_PROVIDER=groq."
            )
        if not provider_settings.api_key:
            raise EnvironmentError(
                "GROQ_API_KEY nao encontrada. Defina GROQ_API_KEY ou PF_LLM_API_KEY quando PF_LLM_PROVIDER=groq."
            )
        return OpenAI(
            api_key=provider_settings.api_key,
            base_url=provider_settings.base_url,
        )

    return Client(host=provider_settings.base_url)


def get_client(llm_settings: LLMSettings, provider: str) -> Any:
    provider_settings = provider_settings_by_name(llm_settings, provider)
    cache_key = (provider, provider_settings.base_url, provider_settings.model_name)
    if cache_key not in CLIENT_CACHE:
        CLIENT_CACHE[cache_key] = build_client(llm_settings, provider)
    return CLIENT_CACHE[cache_key]


def extract_json_payload(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return match.group(0).strip()
    return cleaned


def truncate_text(value: str, limit: int = 240) -> str:
    single_line = re.sub(r"\s+", " ", value).strip()
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + "..."


def normalize_object_key(value: str) -> str:
    normalized = fold_to_ascii(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def normalize_classificacao(value: object) -> str:
    cleaned = normalize_object_key(str(value or ""))
    mapping = {
        "por_crime": "Por crime",
        "crime": "Por crime",
        "com_operacao_nomeada": "Com operacao nomeada",
        "operacao_nomeada": "Com operacao nomeada",
        "outras": "Outras",
        "outra": "Outras",
    }
    return mapping.get(cleaned, str(value or "").strip())


def coerce_list_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [str(item).strip() for item in value.values() if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]

    parts = [part.strip() for part in re.split(r"[,;\n|]+", text) if part.strip()]
    return parts or [text]


def unwrap_payload(payload: Any) -> Any:
    current = payload
    for _ in range(3):
        if not isinstance(current, dict):
            return current

        normalized_items = {normalize_object_key(str(key)): value for key, value in current.items()}
        for wrapper_key in ("resultado", "response", "output", "data", "json", "answer"):
            wrapped = normalized_items.get(wrapper_key)
            if isinstance(wrapped, dict):
                current = wrapped
                break
        else:
            return current
    return current


def coerce_inference_payload(payload: Any) -> dict[str, Any]:
    unwrapped = unwrap_payload(payload)
    if not isinstance(unwrapped, dict):
        raise TypeError("Resposta da LLM nao retornou um objeto JSON.")

    normalized_payload = {normalize_object_key(str(key)): value for key, value in unwrapped.items()}

    identidade = ""
    for key in ("identidade_canonica", "identidade", "identidade_canonica_do_caso", "identity"):
        if key in normalized_payload:
            identidade = str(normalized_payload[key] or "").strip()
            break

    classificacao = ""
    for key in ("classificacao", "classification", "categoria", "tipo"):
        if key in normalized_payload:
            classificacao = normalize_classificacao(normalized_payload[key])
            break

    crimes: object = []
    for key in ("crimes_mais_presentes", "crimes_presentes", "crimes", "crime"):
        if key in normalized_payload:
            crimes = normalized_payload[key]
            break

    modus: object = []
    for key in ("modus_operandi", "modus", "modos_operandi", "modos_de_atuacao", "modos_de_operacao"):
        if key in normalized_payload:
            modus = normalized_payload[key]
            break

    return {
        "identidade_canonica": identidade,
        "classificacao": classificacao,
        "crimes_mais_presentes": coerce_list_value(crimes),
        "modus_operandi": coerce_list_value(modus),
    }


def parse_inference_response(raw_text: str, provider: str, model_name: str) -> NoticiaLLMInference:
    cleaned = raw_text.strip()
    if not cleaned:
        raise RuntimeError(f"Provider {provider} retornou resposta vazia com o modelo {model_name}.")

    payload_text = extract_json_payload(cleaned)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Provider {provider} retornou texto sem JSON valido com o modelo {model_name}. "
            f"Trecho: {truncate_text(cleaned)}"
        ) from exc

    try:
        return NoticiaLLMInference.model_validate(coerce_inference_payload(payload))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"JSON do provider {provider} nao aderiu ao schema com o modelo {model_name}. "
            f"Trecho: {truncate_text(payload_text)} | erro: {exc}"
        ) from exc


def extract_ollama_usage(response: Any) -> TokenUsage:
    prompt_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
    completion_tokens = int(getattr(response, "eval_count", 0) or 0)
    total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def extract_openai_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def format_token_usage(token_usage: TokenUsage) -> str:
    return (
        f"prompt={token_usage.prompt_tokens} | "
        f"completion={token_usage.completion_tokens} | "
        f"total={token_usage.total_tokens}"
    )


def run_model_with_ollama(client: Client, news_body: str, model_name: str) -> tuple[NoticiaLLMInference, TokenUsage]:
    response = client.chat(
        model=model_name,
        messages=build_messages(news_body),
        format=NoticiaLLMInference.model_json_schema(),
        options={"temperature": TEMPERATURE},
    )
    return (
        parse_inference_response(
            raw_text=response.message.content or "",
            provider="ollama",
            model_name=model_name,
        ),
        extract_ollama_usage(response),
    )


def run_model_with_groq(client: OpenAI, news_body: str, model_name: str) -> tuple[NoticiaLLMInference, TokenUsage]:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=build_messages(news_body),
            response_format={"type": "json_object"},
            temperature=max(TEMPERATURE, 1e-8),
        )
        content = response.choices[0].message.content or ""
        return (
            parse_inference_response(
                raw_text=content,
                provider="groq",
                model_name=model_name,
            ),
            extract_openai_usage(response),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Falha unica no provider groq com o modelo {model_name}. "
            "Vou acionar o fallback local no proximo provider. "
            f"Erro: {exc}"
        ) from exc


def run_model_with_provider(
    llm_settings: LLMSettings,
    provider: str,
    client: Any,
    news_body: str,
) -> tuple[NoticiaLLMInference, TokenUsage]:
    provider_settings = provider_settings_by_name(llm_settings, provider)
    if provider == "groq":
        return run_model_with_groq(client=client, news_body=news_body, model_name=provider_settings.model_name)
    return run_model_with_ollama(client=client, news_body=news_body, model_name=provider_settings.model_name)


def run_model(llm_settings: LLMSettings, news_body: str) -> tuple[NoticiaLLMInference, str, str, TokenUsage]:
    errors: list[str] = []

    for provider_settings in provider_candidates(llm_settings):
        provider = provider_settings.provider
        try:
            client = get_client(llm_settings, provider)
            inference, token_usage = run_model_with_provider(
                llm_settings=llm_settings,
                provider=provider,
                client=client,
                news_body=news_body,
            )
            return inference, provider, provider_settings.model_name, token_usage
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider} ({provider_settings.model_name}): {exc}")
            print(f"[llm] falha em {provider} com modelo {provider_settings.model_name}: {exc}")

    raise RuntimeError(
        "Todos os providers LLM falharam. Detalhes: " + " | ".join(errors)
    )


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


def load_existing_records(output_jsonl: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if not output_jsonl.exists():
        return [], [], set()

    indexed_by_title: dict[str, dict[str, Any]] = {}
    for line in output_jsonl.read_text(encoding="utf-8").splitlines():
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


def save_outputs(output_dir: Path, output_jsonl: Path, output_csv: Path, dataframe: pd.DataFrame, jsonl_records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in jsonl_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_ready = dataframe.copy()
    for column in ["tags", "crimes_mais_presentes", "modus_operandi"]:
        if column in csv_ready:
            csv_ready[column] = csv_ready[column].map(lambda values: json.dumps(values, ensure_ascii=False))
    csv_ready.to_csv(output_csv, index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> pd.DataFrame:
    runtime_config = resolve_runtime_config(parse_args(argv))
    markdown_dir = runtime_config.markdown_dir
    markdown_files_all = tuple(sorted(markdown_dir.glob(MARKDOWN_PATTERN)))
    if not markdown_dir.exists():
        raise FileNotFoundError(f"Diretorio de markdown nao encontrado: {markdown_dir}")
    if not markdown_files_all:
        raise FileNotFoundError(f"Nenhum arquivo markdown encontrado em: {markdown_dir}")

    markdown_files = list(markdown_files_all[: runtime_config.limit] if runtime_config.limit is not None else markdown_files_all)
    existing_jsonl_records, existing_dataframe_rows, processed_titles = load_existing_records(runtime_config.output_jsonl)
    dataframe_rows: list[dict[str, Any]] = list(existing_dataframe_rows)
    jsonl_records: list[dict[str, Any]] = list(existing_jsonl_records)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    ordered_providers = " -> ".join(
        f"{settings.provider} ({settings.model_name})" for settings in provider_candidates(runtime_config.llm_settings)
    )
    print(f"Preferencia LLM: {runtime_config.llm_settings.preferred_provider}")
    print(f"Ordem de tentativa: {ordered_providers}")

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
        inferencia_llm, provider_used, model_used, token_usage = run_model(
            llm_settings=runtime_config.llm_settings,
            news_body=str(parsed_news["corpo"]),
        )
        registro = NoticiaEnriquecida(
            metadata_extraido=metadata_extraido,
            inferencia_llm=inferencia_llm,
        )
        total_prompt_tokens += token_usage.prompt_tokens
        total_completion_tokens += token_usage.completion_tokens
        total_tokens += token_usage.total_tokens

        dataframe_rows.append(build_dataframe_row(markdown_file.name, registro))
        jsonl_records.append(build_jsonl_record(markdown_file.name, registro))
        if title_key:
            processed_titles.add(title_key)

        print(
            f"[{index}/{len(markdown_files)}] {markdown_file.name} | "
            f"provider={provider_used} | modelo={model_used} | "
            f"tokens={format_token_usage(token_usage)}"
        )
        print(registro.to_readable_block())
        print()

    dataframe = build_dataframe(dataframe_rows)
    save_outputs(
        output_dir=runtime_config.output_dir,
        output_jsonl=runtime_config.output_jsonl,
        output_csv=runtime_config.output_csv,
        dataframe=dataframe,
        jsonl_records=jsonl_records,
    )

    print(f"Arquivos processados: {len(dataframe)}")
    print(f"JSONL salvo em: {runtime_config.output_jsonl}")
    print(f"CSV salvo em: {runtime_config.output_csv}")
    print(
        "[llm] consumo acumulado de tokens: "
        f"prompt={total_prompt_tokens} | completion={total_completion_tokens} | total={total_tokens}"
    )
    return dataframe


if __name__ == "__main__":
    main()
