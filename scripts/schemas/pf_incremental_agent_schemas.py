from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


TopicType = Literal["crime", "modus", "setor", "operacao", "tema_institucional", "misto", "ruido"]
ThemeDecision = Literal["accept", "merge", "split", "discard", "quarantine"]
ThemeTreeDecision = Literal["merge_into_existing", "promote_to_canonical", "keep_as_leaf", "discard", "quarantine"]


class CanonicalTheme(BaseModel):
    """Tema canonico criado a partir de um ou mais clusters exploratorios."""

    canonical_topic: str = Field(description="Topico canonico sugerido.")
    topic_type: TopicType = Field(description="Tipo substantivo do topico.")
    description: str = Field(description="Descricao curta e auditavel do topico.")
    included_cluster_ids: list[int] = Field(default_factory=list, description="Clusters exploratorios incluidos.")
    included_subthemes: list[str] = Field(default_factory=list, description="Subtemas englobados pelo tema canonico.")
    exclusion_rules: list[str] = Field(default_factory=list, description="Casos que nao devem entrar no tema.")
    evidence_terms: list[str] = Field(default_factory=list, description="Termos ou evidencias que sustentam o tema.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianca do agente na proposta.")
    automation_decision: ThemeDecision = Field(description="Decisao automatica sobre o tema.")


class ThemeBifurcationAgentResponse(BaseModel):
    """Resposta padronizada do Agente 1: bifurcador de temas."""

    sample_id: str = Field(default="", description="Identificador da amostra inicial.")
    themes: list[CanonicalTheme] = Field(default_factory=list, description="Temas canonicos propostos.")
    discarded_cluster_ids: list[int] = Field(default_factory=list, description="Clusters descartados automaticamente.")
    quarantined_cluster_ids: list[int] = Field(default_factory=list, description="Clusters em quarentena automatica.")
    global_risks: list[str] = Field(default_factory=list, description="Riscos metodologicos automaticos.")


class ResidualReviewAgentResponse(BaseModel):
    """Resposta padronizada do Agente 3 para revisar residuais do pipeline WNN."""

    decision: Literal["classificar", "quarentena", "novo_tema_candidato"] = Field(description="Decisao automatica sobre o residual.")
    canonical_label: str = Field(description="Uma label canonica do Agente 1, uma nova label candidata, ou vazio em quarentena.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confianca da revisao.")
    evidence_text: str = Field(default="", description="Trecho curto do texto que sustenta a label escolhida.")
    rationale: str = Field(default="", description="Justificativa curta da classificacao.")
    resumo_curto: str = Field(default="", description="Resumo factual curto do residual.")
    tema_principal: str = Field(default="", description="Tema dominante quando houver multiplos marcadores.")
    marcadores_secundarios: list[str] = Field(default_factory=list, description="Marcadores presentes, mas secundarios.")
    modus_operandi: list[str] = Field(default_factory=list, description="Modos de atuacao observados no caso.")
    relacao_operacional: str = Field(default="", description="cadeia_operacional, crime_organizado_multidominio, coocorrencia_sem_fusao ou tema_unico.")

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        decision = str(payload.get("decision", "") or "").strip().lower()
        canonical_label = str(payload.get("canonical_label", "") or "").strip()
        if decision and decision not in {"classificar", "classificado", "classificada", "classificacao", "classificaÃ§Ã£o", "aprovada", "aprovado", "novo_tema_candidato", "novo tema candidato", "novo_tema", "tema_novo", "nova_label", "novo_label", "quarentena", "quarentenar", "rejeitar", "incerto", "sem_label", "sem label"}:
            payload["decision"] = "classificar"
            if not canonical_label:
                payload["canonical_label"] = decision
        return payload

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, value: object) -> str:
        text = str(value or "").strip().lower()
        if text in {"classificar", "classificado", "classificada", "classificacao", "classificação", "aprovada", "aprovado"}:
            return "classificar"
        if text in {"novo_tema_candidato", "novo tema candidato", "novo_tema", "tema_novo", "nova_label", "novo_label"}:
            return "novo_tema_candidato"
        if text in {"quarentena", "quarentenar", "rejeitar", "incerto", "sem_label", "sem label"}:
            return "quarentena"
        return text

    @field_validator("canonical_label", mode="before")
    @classmethod
    def normalize_canonical_label(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("tema_principal", "relacao_operacional", mode="before")
    @classmethod
    def normalize_optional_slug(cls, value: object) -> str:
        if value is None:
            return ""
        from scripts.pf_llm_models import normalize_slug

        return normalize_slug(str(value))

    @field_validator("marcadores_secundarios", "modus_operandi", mode="before")
    @classmethod
    def ensure_secondary_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @field_validator("marcadores_secundarios", "modus_operandi")
    @classmethod
    def normalize_secondary_values(cls, value: list[str]) -> list[str]:
        from scripts.pf_llm_models import normalize_slug

        output: list[str] = []
        for item in value:
            cleaned = normalize_slug(item)
            if cleaned and cleaned not in output:
                output.append(cleaned)
        return output[:10]


# Aliases de compatibilidade com o scaffold inicial.
TopicAgentResponse = ThemeBifurcationAgentResponse


class OperationalCanonicalTheme(BaseModel):
    """Tema canonico usado pelo fluxo incremental executavel."""

    canonical_theme: str = Field(description="Nome canonico do tema em lowercase_com_underscores.")
    description: str = Field(description="Descricao curta do tema.")
    included_cluster_ids: list[int] = Field(default_factory=list)
    included_subthemes: list[str] = Field(default_factory=list)
    evidence_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    decision: ThemeDecision = Field(description="accept, merge, split, discard ou quarantine")


class OperationalThemeBifurcationResponse(BaseModel):
    """Resposta operacional do Agente 1 consumida pelo pipeline incremental."""

    themes: list[OperationalCanonicalTheme] = Field(default_factory=list)
    quarantined_cluster_ids: list[int] = Field(default_factory=list)
    discarded_cluster_ids: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ThemeCandidateDecision(BaseModel):
    """Decisao do Agente Organizador da Arvore sobre uma folha candidata."""

    candidate_label: str = Field(description="Label candidata observada no residual.")
    decision: ThemeTreeDecision = Field(description="Decisao global para a candidata.")
    parent_theme: str = Field(default="", description="Tema canonico pai quando a candidata for absorvida ou mantida como folha.")
    promoted_theme: str = Field(default="", description="Novo tema canonico quando houver promocao.")
    merged_candidate_labels: list[str] = Field(default_factory=list, description="Labels candidatas semanticamente equivalentes que devem ser fundidas.")
    evidence_terms: list[str] = Field(default_factory=list, description="Evidencias que sustentam a decisao.")
    rationale: str = Field(default="", description="Justificativa curta da decisao.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confianca da decisao.")


class ThemeTreeRefinementResponse(BaseModel):
    """Resposta do Agente Organizador da Arvore com visao global."""

    decisions: list[ThemeCandidateDecision] = Field(default_factory=list)
    promoted_canonical_themes: list[str] = Field(default_factory=list)
    merged_into_existing_count: int = Field(default=0, ge=0)
    promoted_count: int = Field(default=0, ge=0)
    kept_as_leaf_count: int = Field(default=0, ge=0)
    discarded_count: int = Field(default=0, ge=0)
    quarantined_count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)
