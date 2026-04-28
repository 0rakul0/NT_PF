from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    from pf_llm_metadata import LLM_SETTINGS, ZERO_TOKEN_USAGE, format_token_usage, run_model
    from pf_regex_classifier import DEFAULT_CONFIDENCE_THRESHOLD, classify_news_body
except ModuleNotFoundError:
    from scripts.pf_llm_metadata import LLM_SETTINGS, ZERO_TOKEN_USAGE, format_token_usage, run_model
    from scripts.pf_regex_classifier import DEFAULT_CONFIDENCE_THRESHOLD, classify_news_body


def read_body(text: str = "", input_file: Path | str | None = None) -> str:
    if input_file:
        return Path(input_file).read_text(encoding="utf-8")
    if text:
        return text
    env_input_file = os.getenv("PF_CLASSIFY_BODY_INPUT_FILE", "").strip()
    if env_input_file:
        return Path(env_input_file).read_text(encoding="utf-8")
    env_text = os.getenv("PF_CLASSIFY_BODY_TEXT", "")
    if env_text:
        return env_text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Informe text/input_file por chamada Python, variavel de ambiente ou stdin.")


def main(
    text: str = "",
    input_file: Path | str | None = None,
    tags: list[str] | None = None,
    threshold: float | None = None,
    llm_fallback: bool | None = None,
) -> None:
    body = read_body(text=text, input_file=input_file)
    if threshold is None:
        threshold_raw = os.getenv("PF_REGEX_CONFIDENCE_THRESHOLD", "").strip().replace(",", ".")
        try:
            threshold = float(threshold_raw) if threshold_raw else DEFAULT_CONFIDENCE_THRESHOLD
        except ValueError:
            threshold = DEFAULT_CONFIDENCE_THRESHOLD
    if llm_fallback is None:
        llm_fallback = os.getenv("PF_CLASSIFY_BODY_LLM_FALLBACK", "").strip().lower() in {"1", "true", "yes"}

    env_tags = [part.strip() for part in os.getenv("PF_CLASSIFY_BODY_TAGS", "").split(",") if part.strip()]
    regex_result = classify_news_body(body, tags=tags or env_tags, confidence_threshold=threshold)
    source = "regex"
    provider = "regex"
    model = "pf_regex_classifier"
    token_usage = ZERO_TOKEN_USAGE
    inference = regex_result.inference

    if inference is None and llm_fallback:
        source = "llm"
        inference, provider, model, token_usage = run_model(LLM_SETTINGS, body)

    payload = {
        "source": source if inference is not None else "unclassified",
        "provider": provider if inference is not None else "",
        "model": model if inference is not None else "",
        "tokens": format_token_usage(token_usage),
        "regex": regex_result.to_dict(),
        "inference": inference.model_dump() if inference else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
