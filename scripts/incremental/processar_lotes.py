from __future__ import annotations

import time
from typing import Any

import pandas as pd

from scripts.agentes.agente_aprendiz_regex import generate_regex_from_review, review_to_inference
from scripts.agentes.agente3_residual import append_new_theme_candidate, review_residual
from scripts.incremental.common import (
    ACTIVE_REGEX_BANK_PATH,
    LOTS_DIR,
    METRICS_CSV,
    RARE_NEWS_LABEL,
    RESERVE_CSV,
    RUN_DIR,
    THEMES_JSON,
    RunConfig,
    append_event,
    docs_by_manifest,
    read_json,
    write_json,
)
from scripts.incremental.noticias_raras import append_rare_news_observation
from scripts.incremental.similaridade_cosseno import top_k_similar_themes
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse

try:
    from scripts.pf_regex_classifier import classify_news_body
except ModuleNotFoundError:
    from pf_regex_classifier import classify_news_body


def canonical_labels() -> list[str]:
    payload = read_json(THEMES_JSON)
    labels = [str(theme["canonical_theme"]) for theme in payload.get("themes", []) if theme.get("decision") == "accept"]
    if RARE_NEWS_LABEL not in labels:
        labels.append(RARE_NEWS_LABEL)
    return labels


def classify_with_regex(doc: dict[str, Any], threshold: float) -> dict[str, Any]:
    result = classify_news_body(doc["context"], tags=doc["parsed"].get("tags", []), confidence_threshold=threshold, learned_rules_file=ACTIVE_REGEX_BANK_PATH)
    accepted = result.inference is not None and result.confidence >= threshold
    return {
        "arquivo": doc["arquivo"],
        "titulo": doc["titulo"],
        "regex_accepted": accepted,
        "regex_confidence": round(result.confidence, 4),
        "regex_source": result.source,
        "regex_bank": str(ACTIVE_REGEX_BANK_PATH),
        "matched_rules_count": len(result.matched_rules),
        "inference": result.inference.model_dump() if result.inference else {},
        "classification_source": "agent2_regex" if accepted else "residual_pending_agent3",
        "agent3_reviewed": False,
        "agent3_decision": "",
        "agent3_canonical_label": "",
        "agent3_confidence": 0.0,
        "agent3_evidence_text": "",
        "agent3_rationale": "",
        "agent2_incremental_regex_count": 0,
        "agent2_incremental_regex": [],
        "cosine_top_label": "",
        "cosine_top_score": 0.0,
        "cosine_top_k": [],
    }


def run(config: RunConfig) -> dict[str, object]:
    reserve = docs_by_manifest(RESERVE_CSV)
    allowed_labels = canonical_labels()

    metrics = []
    completed_iterations: set[int] = set()
    if config.resume_batches and METRICS_CSV.exists():
        existing_metrics = pd.read_csv(METRICS_CSV)
        metrics = existing_metrics.to_dict(orient="records")
        completed_iterations = {int(row["iteration"]) for row in metrics if pd.notna(row.get("iteration"))}
    cumulative_docs = int(sum(int(row.get("docs", 0) or 0) for row in metrics))
    cumulative_regex = int(sum(int(row.get("regex_accepted", 0) or 0) for row in metrics))
    cumulative_llm = int(sum(int(row.get("llm_processed", 0) or 0) for row in metrics))
    batches = [reserve[index : index + config.batch_size] for index in range(0, len(reserve), config.batch_size)]
    if config.max_batches is not None:
        batches = batches[: config.max_batches]
    for iteration, batch in enumerate(batches, start=1):
        if iteration in completed_iterations:
            continue
        started = time.perf_counter()
        rows = []
        llm_processed = 0
        learned_rules = 0
        token_total = 0
        prompt_tokens_total = 0
        completion_tokens_total = 0
        agent3_attempted = 0
        agent3_classified = 0
        agent3_quarantined = 0
        agent3_new_theme_candidates = 0
        agent3_rare_news = 0
        agent3_errors = 0
        residual_limit = config.max_residual_llm_per_batch
        negatives = [doc["context"] for doc in batch[:40]]
        regex_accepted = 0
        regex_residual = 0

        for doc in batch:
            row = classify_with_regex(doc, config.regex_threshold)
            rows.append(row)
            if row["regex_accepted"]:
                regex_accepted += 1
                continue

            regex_residual += 1
            if residual_limit is not None and agent3_attempted >= residual_limit:
                continue

            agent3_attempted += 1
            cosine_candidates = top_k_similar_themes(doc["context"], top_k=5)
            row.update(
                {
                    "cosine_top_label": cosine_candidates[0]["label"] if cosine_candidates else "",
                    "cosine_top_score": cosine_candidates[0]["score"] if cosine_candidates else 0.0,
                    "cosine_top_k": cosine_candidates,
                }
            )
            try:
                review, provider, model_name, token_usage = review_residual(doc, allowed_labels, config, cosine_candidates)
            except Exception as exc:
                agent3_errors += 1
                row.update(
                    {
                        "classification_source": "agent3_review_error",
                        "agent3_reviewed": True,
                        "agent3_decision": "error",
                    "agent3_rationale": str(exc),
                }
            )
                append_event({"stage": "llm_residual", "iteration": iteration, "arquivo": doc["arquivo"], "status": "error", "error": str(exc)})
                continue
            llm_processed += 1
            prompt_tokens_total += token_usage.prompt_tokens
            completion_tokens_total += token_usage.completion_tokens
            token_total += token_usage.total_tokens
            if review.decision == "classificar":
                agent3_classified += 1
                if review.canonical_label == RARE_NEWS_LABEL:
                    agent3_rare_news += 1
                    rare_observation = append_rare_news_observation(
                        doc,
                        iteration,
                        review.evidence_text,
                        review.rationale,
                        review.confidence,
                    )
                    if rare_observation.get("promoted_label"):
                        review = ResidualReviewAgentResponse(
                            decision="novo_tema_candidato",
                            canonical_label=str(rare_observation["promoted_label"]),
                            confidence=review.confidence,
                            evidence_text=review.evidence_text,
                            rationale=(
                                "Noticia rara recorrente promovida automaticamente para candidato de tema: "
                                f"{rare_observation['promoted_label']}."
                            ),
                            resumo_curto=review.resumo_curto,
                        )
                        agent3_classified -= 1
                        agent3_rare_news -= 1
                        agent3_new_theme_candidates += 1
                        append_new_theme_candidate(doc, review, iteration)
            elif review.decision == "novo_tema_candidato":
                agent3_new_theme_candidates += 1
                append_new_theme_candidate(doc, review, iteration)
            else:
                agent3_quarantined += 1
            incorporated = (
                generate_regex_from_review(doc, review, negatives)
                if review.decision in {"classificar", "novo_tema_candidato"} and review.canonical_label != RARE_NEWS_LABEL
                else []
            )
            learned_rules += len(incorporated)
            row.update(
                {
                    "classification_source": "agent3_review",
                    "agent3_reviewed": True,
                    "agent3_decision": review.decision,
                    "agent3_canonical_label": review.canonical_label,
                    "agent3_confidence": round(review.confidence, 4),
                    "agent3_evidence_text": review.evidence_text,
                    "agent3_rationale": review.rationale,
                    "agent2_incremental_regex_count": len(incorporated),
                    "agent2_incremental_regex": incorporated,
                    "inference": review_to_inference(review).model_dump() if review.decision == "classificar" else {},
                }
            )
            append_event(
                {
                    "stage": "llm_residual",
                    "iteration": iteration,
                    "arquivo": doc["arquivo"],
                    "provider": provider,
                    "model": model_name,
                    "tokens": {
                        "prompt_tokens": token_usage.prompt_tokens,
                        "completion_tokens": token_usage.completion_tokens,
                        "total_tokens": token_usage.total_tokens,
                    },
                    "canonical_labels_available": allowed_labels,
                    "cosine_candidates": cosine_candidates,
                    "agent3_review": review.model_dump(),
                    "agent2_incremental_regex": incorporated,
                }
            )

        batch_df = pd.DataFrame(rows)
        batch_output = LOTS_DIR / f"lote_{iteration:04d}_classificacoes.csv"
        batch_output.parent.mkdir(parents=True, exist_ok=True)
        batch_df.to_csv(batch_output, index=False, encoding="utf-8-sig")
        docs = len(batch)
        cumulative_docs += docs
        cumulative_regex += regex_accepted
        cumulative_llm += llm_processed
        metrics.append(
            {
                "iteration": iteration,
                "batch_id": f"lote_{iteration:04d}",
                "docs": docs,
                "regex_accepted": regex_accepted,
                "regex_residual": regex_residual,
                "llm_processed": llm_processed,
                "agent3_attempted": agent3_attempted,
                "agent3_reviewed": llm_processed,
                "agent3_classified": agent3_classified,
                "agent3_quarantined": agent3_quarantined,
                "agent3_new_theme_candidates": agent3_new_theme_candidates,
                "agent3_rare_news": agent3_rare_news,
                "agent3_errors": agent3_errors,
                "learned_rules": learned_rules,
                "tokens_total": token_total,
                "prompt_tokens_total": prompt_tokens_total,
                "completion_tokens_total": completion_tokens_total,
                "avg_tokens_per_llm": round(token_total / llm_processed, 4) if llm_processed else 0,
                "regex_rate": round(regex_accepted / docs, 6) if docs else 0,
                "cumulative_docs": cumulative_docs,
                "cumulative_regex_accepted": cumulative_regex,
                "cumulative_llm_processed": cumulative_llm,
                "cumulative_regex_rate": round(cumulative_regex / cumulative_docs, 6) if cumulative_docs else 0,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "regex_bank": str(ACTIVE_REGEX_BANK_PATH),
                "output": str(batch_output),
            }
        )
        pd.DataFrame(metrics).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    result = {"stage": "processar_lotes", "metrics_csv": str(METRICS_CSV), "batches": len(metrics)}
    write_json(RUN_DIR / "processar_lotes_result.json", result)
    append_event(result)
    return result


def main() -> None:
    print(write_json(RUN_DIR / "processar_lotes_result.json", run(RunConfig())))


if __name__ == "__main__":
    main()
