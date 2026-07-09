from __future__ import annotations

import re
from typing import Any


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


def build_llm_context(parsed_news: dict[str, Any]) -> str:
    tags = parsed_news.get("tags", [])
    tags_text = ", ".join(str(tag).strip() for tag in tags if str(tag).strip()) if isinstance(tags, list) else ""
    fields = [
        ("titulo", parsed_news.get("titulo", "")),
        ("subtitulo", parsed_news.get("subtitulo", "")),
        ("tags", tags_text),
        ("corpo", parsed_news.get("corpo", "")),
    ]
    return "\n\n".join(
        f"{label}:\n{str(value).strip()}" for label, value in fields if str(value or "").strip()
    )
