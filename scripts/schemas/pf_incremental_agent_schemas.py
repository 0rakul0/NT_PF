from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


RuleKind = Literal["crime", "modus"]
RuleDecision = Literal["incorporar", "rejeitar", "quarentena"]
TopicType = Literal["crime", "modus", "setor", "operacao", "tema_institucional", "misto", "ruido"]
ThemeDecision = Literal["accept", "merge", "split", "discard", "quarantine"]


class RegexCandidate(BaseModel):
    """Regex proposta por um agente para classificar casos futuros."""

    kind: RuleKind = Field(description="Tipo da regra candidata.")
    label: str = Field(description="Label canonica associada a regex.")
    pattern: str = Field(description="Expressao regular candidata.")
    rationale: str = Field(description="Justificativa textual para a regex.")
    expected_precision_risk: str = Field(
        default="medio",
        description="Risco estimado de falso positivo: baixo, medio ou alto.",
    )
    positive_examples: list[str] = Field(default_factory=list, description="Exemplos positivos usados.")
    negative_examples: list[str] = Field(default_factory=list, description="Exemplos negativos ou contraexemplos.")


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


class InitialRegexAgentResponse(BaseModel):
    """Resposta padronizada do Agente 2: gerador de regex iniciais."""

    theme_id: str = Field(default="", description="Identificador do tema canonico.")
    canonical_topic: str = Field(description="Tema canonico associado.")
    regex_candidates: list[RegexCandidate] = Field(default_factory=list, description="Regex candidatas avaliadas.")
    accepted_initial_rules: list[RegexCandidate] = Field(default_factory=list, description="Regex aprovadas.")
    rejected_candidates: list[RegexCandidate] = Field(default_factory=list, description="Regex rejeitadas.")
    quarantined_candidates: list[RegexCandidate] = Field(default_factory=list, description="Regex em quarentena automatica.")
    coverage_estimate: float = Field(ge=0.0, le=1.0, description="Estimativa de cobertura automatica.")
    precision_risk: str = Field(description="Risco estimado de falso positivo.")


class RegexIncorporationDecision(BaseModel):
    """Decisao individual do Agente 3 sobre uma regex candidata."""

    decision: RuleDecision = Field(description="Decisao tomada para a regex.")
    kind: RuleKind = Field(description="Tipo da regra.")
    label: str = Field(description="Label canonica avaliada.")
    pattern: str = Field(description="Regex avaliada.")
    validation_summary: str = Field(description="Resumo da validacao aplicada.")
    justification: str = Field(description="Justificativa da decisao.")


class LearningAgentResponse(BaseModel):
    """Resposta padronizada do Agente 3."""

    batch_id: str = Field(default="", description="Identificador do lote, quando disponivel.")
    decisions: list[RegexIncorporationDecision] = Field(default_factory=list, description="Decisoes por regex.")
    incorporated_count: int = Field(ge=0, description="Quantidade de regex incorporadas.")
    rejected_count: int = Field(ge=0, description="Quantidade de regex rejeitadas.")
    quarantined_count: int = Field(ge=0, description="Quantidade de regex em quarentena automatica.")
    learned_labels: list[str] = Field(default_factory=list, description="Labels afetadas pelo aprendizado.")
    residual_risks: list[str] = Field(default_factory=list, description="Riscos restantes apos a decisao.")
    next_automatic_tests: list[str] = Field(default_factory=list, description="Testes automaticos recomendados.")


class ResidualReviewAgentResponse(BaseModel):
    """Resposta padronizada do Agente 3 para revisar residuais do regex."""

    decision: Literal["classificar", "quarentena", "novo_tema_candidato"] = Field(description="Decisao automatica sobre o residual.")
    canonical_label: str = Field(description="Uma label canonica do Agente 1, uma nova label candidata, ou vazio em quarentena.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confianca da revisao.")
    evidence_text: str = Field(default="", description="Trecho curto do texto que sustenta a label escolhida.")
    rationale: str = Field(default="", description="Justificativa curta da classificacao.")
    resumo_curto: str = Field(default="", description="Resumo factual curto do residual.")

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


class RegexRuleProposal(BaseModel):
    """Regex proposta pelo Agente 2."""

    kind: RuleKind = Field(description="crime ou modus")
    label: str
    pattern: str
    rationale: str
    risk: str = Field(description="baixo, medio ou alto")


class InitialRegexResponse(BaseModel):
    """Resposta operacional do Agente 2 para um tema canonico."""

    canonical_theme: str
    accepted_rules: list[RegexRuleProposal] = Field(default_factory=list)
    rejected_rules: list[RegexRuleProposal] = Field(default_factory=list)
    quarantined_rules: list[RegexRuleProposal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class InitialRegexBatchResponse(BaseModel):
    """Resposta operacional do Agente 2 para todos os temas canonicos."""

    themes: list[InitialRegexResponse] = Field(default_factory=list)
