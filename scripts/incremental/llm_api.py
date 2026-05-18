from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from ollama import Client
from openai import OpenAI
from pydantic import BaseModel

from scripts.incremental.common import RunConfig, append_event
from scripts.project_config import get_llm_settings


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


ZERO_TOKEN_USAGE = TokenUsage()


def _usage_from_openai_response(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)


def _usage_from_ollama_response(response: Any) -> TokenUsage:
    prompt_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
    completion_tokens = int(getattr(response, "eval_count", 0) or 0)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def extract_json_object(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Resposta sem JSON: {content[:300]}")
    return json.loads(match.group(0))


def invoke_ollama_json(prompt: str, schema: type[SchemaT], config: RunConfig, timeout_seconds: int | None = None, model_name: str | None = None) -> tuple[SchemaT, TokenUsage]:
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
    return schema.model_validate(extract_json_object(content)), _usage_from_ollama_response(response)


def invoke_groq_json(prompt: str, schema: type[SchemaT]) -> tuple[SchemaT, TokenUsage]:
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
    return schema.model_validate(extract_json_object(content)), _usage_from_openai_response(response)


def invoke_openai_json(prompt: str, schema: type[SchemaT]) -> tuple[SchemaT, TokenUsage]:
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
    return schema.model_validate(extract_json_object(content)), _usage_from_openai_response(response)


def is_groq_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "rate_limit" in text or "rate limit" in text or "429" in text


def invoke_json_with_fallback(prompt: str, schema: type[SchemaT], config: RunConfig, stage: str) -> tuple[SchemaT, str, str, TokenUsage]:
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
                        result, token_usage = invoke_ollama_json(prompt, schema, config, model_name=model_name)
                        return result, "ollama", model_name, token_usage
                    except Exception as model_exc:
                        ollama_errors.append({"model": model_name, "error": str(model_exc)})
                        append_event({"stage": stage, "provider": "ollama", "model": model_name, "status": "error", "error": str(model_exc)})
                raise RuntimeError(f"Nenhum modelo local respondeu: {ollama_errors}")
            if provider == "groq":
                result, token_usage = invoke_groq_json(prompt, schema)
                return result, "groq", settings.groq.model_name, token_usage
            if provider == "openai":
                result, token_usage = invoke_openai_json(prompt, schema)
                return result, "openai", settings.openai.model_name, token_usage
        except Exception as exc:
            errors.append({"provider": provider, "error": str(exc)})
            append_event({"stage": stage, "provider": provider, "status": "error", "error": str(exc)})
            if provider == "groq" and is_groq_rate_limit(exc):
                retry_timeout = max(config.llm_timeout_seconds, 300)
                try:
                    result = None
                    token_usage = ZERO_TOKEN_USAGE
                    retry_model = ""
                    retry_errors: list[dict[str, str]] = []
                    for model_name in (config.model, *config.local_fallback_models):
                        try:
                            result, token_usage = invoke_ollama_json(prompt, schema, config, timeout_seconds=retry_timeout, model_name=model_name)
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
                    return result, "ollama", retry_model, token_usage
                except Exception as retry_exc:
                    retry_provider = "ollama_retry_after_groq_limit" if ollama_was_tried else "ollama_after_groq_limit"
                    errors.append({"provider": retry_provider, "error": str(retry_exc)})
                    append_event({"stage": stage, "provider": retry_provider, "status": "error", "error": str(retry_exc)})
    raise RuntimeError(f"Nenhum provider respondeu para {stage}: {errors}")
