from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def fold_to_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_slug(value: str) -> str:
    text = fold_to_ascii(value).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


class NoticiaMetadataExtraido(BaseModel):
    """Metadados objetivos lidos diretamente do markdown, sem inferencia da LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    titulo: str = ""
    subtitulo: str = ""
    data_publicacao: str = Field(default="", pattern=r"^$|^\d{2}/\d{2}/\d{4}$")
    data_atualizacao: str = Field(default="", pattern=r"^$|^\d{2}/\d{2}/\d{4}$")
    tags: list[str] = Field(default_factory=list)
    dateline: str = ""
    nome_operacao_encontrado: str = ""

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            cleaned = item.strip()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized[:10]


class NoticiaLLMInference(BaseModel):
    """Campos que dependem de leitura semantica do conteudo da noticia."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "identidade_canonica": "crime_abuso_sexual_infantil",
                "classificacao": "Por crime",
                "crimes_mais_presentes": ["abuso_sexual_infantil"],
                "modus_operandi": ["atuacao_online", "busca_apreensao"],
            }
        },
    )

    identidade_canonica: str = Field(
        description="Identificador canonico em lowercase com underscores, como crime_abuso_sexual_infantil."
    )
    classificacao: Literal["Por crime", "Com operacao nomeada", "Outras"] = Field(
        description="Classificacao canonica do caso com base no conteudo da noticia."
    )
    crimes_mais_presentes: list[str] = Field(
        default_factory=list,
        description="Lista de crimes canonicos em lowercase com underscores."
    )
    modus_operandi: list[str] = Field(
        default_factory=list,
        description="Lista de modos de atuacao canonicos em lowercase com underscores."
    )

    @field_validator("identidade_canonica", mode="before")
    @classmethod
    def normalize_identity(cls, value: object) -> str:
        if value is None:
            return ""
        return normalize_slug(str(value))

    @field_validator("crimes_mais_presentes", "modus_operandi", mode="before")
    @classmethod
    def ensure_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @field_validator("crimes_mais_presentes", "modus_operandi")
    @classmethod
    def normalize_list_values(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            cleaned = normalize_slug(item)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized[:10]

    @model_validator(mode="after")
    def align_identity(self) -> "NoticiaLLMInference":
        if self.classificacao == "Por crime" and self.crimes_mais_presentes:
            first_crime = self.crimes_mais_presentes[0]
            if not self.identidade_canonica.startswith(("crime_", "crimes_")):
                self.identidade_canonica = f"crime_{first_crime}"

        if not self.identidade_canonica:
            raise ValueError("identidade_canonica nao pode ficar vazia.")
        if not self.crimes_mais_presentes and self.classificacao == "Por crime":
            raise ValueError("crimes_mais_presentes deve ser preenchido quando a classificacao for Por crime.")
        return self


class NoticiaEnriquecida(BaseModel):
    """Registro persistido com metadados extraidos e inferencia semantica separada."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    metadata_extraido: NoticiaMetadataExtraido
    inferencia_llm: NoticiaLLMInference

    def to_readable_block(self) -> str:
        extraido = self.metadata_extraido
        inferencia = self.inferencia_llm
        operation_name = extraido.nome_operacao_encontrado or "Sem nome de operacao explicito"
        crimes = ", ".join(inferencia.crimes_mais_presentes) if inferencia.crimes_mais_presentes else "sem_crime_explicitado"
        modus = ", ".join(inferencia.modus_operandi) if inferencia.modus_operandi else "sem_modus_explicitado"
        tags = ", ".join(extraido.tags) if extraido.tags else "sem_tags"
        return "\n".join(
            [
                f"Titulo: {extraido.titulo}",
                f"Data: {extraido.data_publicacao or 'sem_data'}",
                f"Dateline: {extraido.dateline or 'sem_dateline'}",
                f"Tags diretas: [{tags}]",
                f"Operacao direta: {operation_name}",
                f"Identidade canonica: {inferencia.identidade_canonica}",
                f"Classificacao: {inferencia.classificacao}",
                f"Crimes mais presentes: {crimes}",
                f"Modus operandi: {modus}",
            ]
        )
