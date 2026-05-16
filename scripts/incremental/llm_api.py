from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from ollama import Client
from openai import OpenAI
from pydantic import BaseModel

from scripts.incremental.common import RunConfig, append_event
from scripts.project_config import get_llm_settings


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def extract_json_object(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Resposta sem JSON: {content[:300]}")
    return json.loads(match.group(0))


def invoke_ollama_json(prompt: str, schema: type[SchemaT], config: RunConfig, timeout_seconds: int | None = None, model_name: str | None = None) -> SchemaT:
    timeout = timeout_seconds if timeout_seconds is not None else config.llm_timeout_seconds
    selected_model = model_name or config.model
    client = Client(host=config.base_url, timeout=timeout)
    response = client.chat(
        model=selected_model,
        messages=[
            {"role": "system", "content": "Responda somente um objeto JSON valido que siga exatamente o schema solicitado."},
            {"role": "user", "content": prompt},
        ],
        format=schema.model_json_schema(),
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 1024},
        keep_alive="10m",
    )
    content = response.message.content or ""
    return schema.model_validate(extract_json_object(content))


def invoke_groq_json(prompt: str, schema: type[SchemaT]) -> SchemaT:
    settings = get_llm_settings().groq
    if not settings.api_key:
        raise RuntimeError("GROQ_API_KEY/PF_GROQ_API_KEY nao configurada no ambiente.")

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    messages = [
        {"role": "system", "content": "Responda somente JSON valido, sem markdown."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=700,
        )
    except Exception:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            temperature=0,
            max_tokens=700,
        )
    content = response.choices[0].message.content or ""
    return schema.model_validate(extract_json_object(content))


def invoke_openai_json(prompt: str, schema: type[SchemaT]) -> SchemaT:
    settings = get_llm_settings().openai
    if not settings.api_key:
        raise RuntimeError("OPENAI_API_KEY/PF_OPENAI_API_KEY nao configurada no ambiente.")

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    messages = [
        {"role": "system", "content": "Responda somente JSON valido, sem markdown."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
            max_tokens=900,
        )
    except Exception:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            temperature=0,
            max_tokens=900,
        )
    content = response.choices[0].message.content or ""
    return schema.model_validate(extract_json_object(content))


def is_groq_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "rate_limit" in text or "rate limit" in text or "429" in text


def invoke_json_with_fallback(prompt: str, schema: type[SchemaT], config: RunConfig, stage: str) -> tuple[SchemaT, str, str]:
    settings = get_llm_settings()
    errors: list[dict[str, str]] = []
    ollama_was_tried = False
    for provider in settings.provider_order:
        try:
            if provider == "ollama":
                ollama_was_tried = True
                ollama_errors: list[dict[str, str]] = []
                for model_name in (config.model, *config.local_fallback_models):
                    try:
                        return invoke_ollama_json(prompt, schema, config, model_name=model_name), "ollama", model_name
                    except Exception as model_exc:
                        ollama_errors.append({"model": model_name, "error": str(model_exc)})
                        append_event({"stage": stage, "provider": "ollama", "model": model_name, "status": "error", "error": str(model_exc)})
                raise RuntimeError(f"Nenhum modelo local respondeu: {ollama_errors}")
            if provider == "groq":
                return invoke_groq_json(prompt, schema), "groq", settings.groq.model_name
            if provider == "openai":
                return invoke_openai_json(prompt, schema), "openai", settings.openai.model_name
        except Exception as exc:
            errors.append({"provider": provider, "error": str(exc)})
            append_event({"stage": stage, "provider": provider, "status": "error", "error": str(exc)})
            if provider == "groq" and is_groq_rate_limit(exc):
                retry_timeout = max(config.llm_timeout_seconds, 300)
                try:
                    result = None
                    retry_model = ""
                    retry_errors: list[dict[str, str]] = []
                    for model_name in (config.model, *config.local_fallback_models):
                        try:
                            result = invoke_ollama_json(prompt, schema, config, timeout_seconds=retry_timeout, model_name=model_name)
                            retry_model = model_name
                            break
                        except Exception as retry_model_exc:
                            retry_errors.append({"model": model_name, "error": str(retry_model_exc)})
                    if result is None:
                        raise RuntimeError(f"Nenhum modelo local respondeu no retry: {retry_errors}")
                    append_event(
                        {
                            "stage": stage,
                            "provider": "ollama",
                            "status": "retry_after_groq_limit_ok",
                            "timeout_seconds": retry_timeout,
                            "model": retry_model,
                        }
                    )
                    return result, "ollama", retry_model
                except Exception as retry_exc:
                    retry_provider = "ollama_retry_after_groq_limit" if ollama_was_tried else "ollama_after_groq_limit"
                    errors.append({"provider": retry_provider, "error": str(retry_exc)})
                    append_event({"stage": stage, "provider": retry_provider, "status": "error", "error": str(retry_exc)})
    raise RuntimeError(f"Nenhum provider respondeu para {stage}: {errors}")
