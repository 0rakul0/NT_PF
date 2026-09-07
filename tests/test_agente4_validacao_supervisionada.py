from scripts.agentes.agente4_validacao_supervisionada import validate_residual_review
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse


def _review(label: str) -> ResidualReviewAgentResponse:
    return ResidualReviewAgentResponse(
        decision="classificar",
        canonical_label=label,
        confidence=0.8,
        evidence_text="evidencia",
        rationale="justificativa",
        resumo_curto="resumo",
    )


def test_agent4_confirms_matching_label() -> None:
    result = validate_residual_review(_review("trafico_drogas"), ["trafico_drogas"])

    assert result.learning_authorized is True
    assert result.status == "confirmado_por_x2"
    assert result.raw_label == "trafico_drogas"
    assert result.learning_review.canonical_label == "trafico_drogas"


def test_agent4_corrects_single_target_before_learning() -> None:
    result = validate_residual_review(_review("lavagem_dinheiro"), ["trafico_drogas"])

    assert result.learning_authorized is True
    assert result.status == "corrigido_por_x2_unica"
    assert result.raw_label == "lavagem_dinheiro"
    assert result.learning_review.canonical_label == "trafico_drogas"


def test_agent4_blocks_ambiguous_multilabel_divergence() -> None:
    result = validate_residual_review(_review("lavagem_dinheiro"), ["trafico_drogas", "crime_organizado"])

    assert result.learning_authorized is False
    assert result.status == "divergencia_x2_multilabel_sem_evidencia_para_bleaching"
    assert result.learning_review.canonical_label == "lavagem_dinheiro"


def test_agent4_bleaches_multilabel_divergence_when_wnn_evidence_has_margin() -> None:
    result = validate_residual_review(
        _review("lavagem_dinheiro"),
        ["trafico_drogas", "crime_organizado"],
        [
            {"label": "trafico_drogas", "score": 1.4},
            {"label": "crime_organizado", "score": 1.0},
        ],
    )

    assert result.learning_authorized is True
    assert result.status == "desambiguado_por_bleaching_x2_multilabel"
    assert result.learning_review.canonical_label == "trafico_drogas"
    assert result.learning_review.marcadores_secundarios == ["crime_organizado"]
    assert round(result.bleaching_margin, 4) == round(2 / 7, 4)
