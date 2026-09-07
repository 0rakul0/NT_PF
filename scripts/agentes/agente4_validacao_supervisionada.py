"""Agente 4: validação supervisionada pós-decisão por tags de referência.

O agente recebe a decisão já produzida pelo Agente 3 e os rótulos canônicos
derivados de x2.  Ele nunca recebe x2 como entrada da retina ou da LLM; seu
papel é somente auditar a saída e decidir se ela pode alimentar a memória WNN.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse


BLEACHING_MIN_SCORE = 1.0
BLEACHING_MIN_MARGIN = 0.20


@dataclass(frozen=True)
class SupervisedValidationResult:
    """Raw and supervised forms of an Agent 3 residual decision."""

    raw_label: str
    learning_review: ResidualReviewAgentResponse
    learning_authorized: bool
    status: str
    target_labels: tuple[str, ...]
    bleaching_scores: tuple[tuple[str, float], ...] = ()
    bleaching_margin: float = 0.0


def _target_scores(targets: tuple[str, ...], wnn_scores: list[dict[str, object]] | None) -> list[tuple[str, float]]:
    scores_by_label: dict[str, float] = {}
    for item in wnn_scores or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "")
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if label:
            scores_by_label[label] = max(scores_by_label.get(label, 0.0), score)
    return sorted(((label, scores_by_label.get(label, 0.0)) for label in targets), key=lambda item: item[1], reverse=True)


def validate_residual_review(
    review: ResidualReviewAgentResponse,
    target_labels: list[str],
    wnn_scores: list[dict[str, object]] | None = None,
) -> SupervisedValidationResult:
    """Validate an Agent 3 classification before learning discriminators.

    A matching label is confirmed. A divergent label is corrected directly when
    x2 has exactly one mapped crime target. With multiple targets, bleaching
    ranks only the x2 candidates by their existing WNN textual evidence. It
    authorizes learning only if the best candidate has a strong score and a
    clear margin over the next candidate; otherwise the ambiguity is retained.
    """
    raw_label = str(review.canonical_label or "")
    targets = tuple(dict.fromkeys(str(label) for label in target_labels if str(label)))
    if review.decision != "classificar":
        return SupervisedValidationResult(raw_label, review, False, "nao_classificado_pelo_agente3", targets)
    if not targets:
        return SupervisedValidationResult(raw_label, review, False, "sem_tag_criminal_mapeada", targets)
    if raw_label in targets:
        return SupervisedValidationResult(raw_label, review, True, "confirmado_por_x2", targets)
    if len(targets) != 1:
        ranked_scores = _target_scores(targets, wnn_scores)
        winner, winner_score = ranked_scores[0]
        second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
        margin = (winner_score - second_score) / winner_score if winner_score else 0.0
        if winner_score < BLEACHING_MIN_SCORE or margin < BLEACHING_MIN_MARGIN:
            return SupervisedValidationResult(
                raw_label,
                review,
                False,
                "divergencia_x2_multilabel_sem_evidencia_para_bleaching",
                targets,
                tuple(ranked_scores),
                margin,
            )
        bleached_review = review.model_copy(
            update={
                "canonical_label": winner,
                "tema_principal": winner,
                "marcadores_secundarios": [label for label in targets if label != winner],
                "rationale": (
                    f"{review.rationale} [Rótulo de aprendizado desambiguado por bleaching x2: {winner}; "
                    f"score={winner_score:.3f}; margem={margin:.3f}.]"
                ).strip(),
            }
        )
        return SupervisedValidationResult(
            raw_label,
            bleached_review,
            True,
            "desambiguado_por_bleaching_x2_multilabel",
            targets,
            tuple(ranked_scores),
            margin,
        )

    supervised_label = targets[0]
    supervised_review = review.model_copy(
        update={
            "canonical_label": supervised_label,
            "tema_principal": supervised_label,
            "marcadores_secundarios": [label for label in review.marcadores_secundarios if label != supervised_label],
            "rationale": f"{review.rationale} [Rótulo de aprendizado supervisionado por x2: {supervised_label}.]".strip(),
        }
    )
    return SupervisedValidationResult(raw_label, supervised_review, True, "corrigido_por_x2_unica", targets)
