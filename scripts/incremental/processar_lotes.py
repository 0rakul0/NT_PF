from __future__ import annotations

import time
from typing import Any

import pandas as pd

from scripts.agentes.agente3_residual import append_new_theme_candidate, review_residual
from scripts.agentes.agente4_validacao_supervisionada import validate_residual_review
from scripts.agentes.agente1_temas import KNOWN_CANONICAL_LABELS
from scripts.incremental.common import (
    DYNAMIC_THRESHOLDS_JSON,
    LOTS_DIR,
    METRICS_CSV,
    RARE_NEWS_LABEL,
    RESERVE_CSV,
    RUN_DIR,
    THEMES_JSON,
    RunConfig,
    WNN_FEATURE_BANK_PATH,
    append_event,
    docs_by_manifest,
    read_json,
    write_json,
)
from scripts.incremental.calibracao_limiares import calibrate_after_batch, load_active_thresholds
from scripts.incremental.noticias_raras import append_rare_news_observation
from scripts.incremental.preprocessamento_linguistico import preprocess_body_text
from scripts.incremental.similaridade_cosseno import top_k_similar_themes
from scripts.schemas.pf_incremental_agent_schemas import ResidualReviewAgentResponse
from scripts.incremental.dashboard_comparacao import run as update_dashboard
from scripts.avaliar_crimes_por_tags import crime_labels_from_tags
from scripts.pf_llm_models import NoticiaLLMInference

try:
    from scripts.pf_wnn_classifier import (
        CRIME_CONFIDENCE_THRESHOLDS,
        append_discriminators_from_learned_rules,
        classify_with_wnn,
        compact_feature_bank,
        record_reverse_memory_observation,
        suggest_discriminator_rules_from_review,
    )
except ModuleNotFoundError:
    from pf_wnn_classifier import CRIME_CONFIDENCE_THRESHOLDS, append_discriminators_from_learned_rules, classify_with_wnn, compact_feature_bank, record_reverse_memory_observation, suggest_discriminator_rules_from_review

try:
    from scripts.agentes.agente_organizador_arvore import run as run_theme_tree_organizer
except ModuleNotFoundError:
    from agentes.agente_organizador_arvore import run as run_theme_tree_organizer


def classification_text(doc: dict[str, Any]) -> str:
    """Build the retinal input exclusively from x3 (``texto_noticia``)."""
    parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
    semantic = str(doc.get("semantic_features", "") or "").strip()
    if semantic:
        return semantic
    body = str(doc.get("x3_texto_noticia", "") or doc.get("body_text", "") or parsed.get("corpo", "")).strip()
    return preprocess_body_text(body).semantic_features


def review_to_inference(review: ResidualReviewAgentResponse) -> NoticiaLLMInference:
    return NoticiaLLMInference(
        identidade_canonica=review.canonical_label,
        classificacao="Por crime",
        crimes_mais_presentes=[review.canonical_label, *review.marcadores_secundarios],
        tema_principal=review.tema_principal or review.canonical_label,
        marcadores_secundarios=review.marcadores_secundarios,
        relacao_operacional=review.relacao_operacional or "tema_unico",
        modus_operandi=review.modus_operandi,
        resumo_curto=review.resumo_curto,
        resumo_estruturado={},
        evidencia_textual=review.evidence_text,
        atores_mencionados=[],
        setor_afetado="",
        precisa_reprocessamento=review.decision != "classificar",
    )


def canonical_labels() -> list[str]:
    payload = read_json(THEMES_JSON)
    labels = [str(theme["canonical_theme"]) for theme in payload.get("themes", []) if theme.get("decision") == "accept"]
    for label in KNOWN_CANONICAL_LABELS:
        if label not in labels:
            labels.append(label)
    if RARE_NEWS_LABEL not in labels:
        labels.append(RARE_NEWS_LABEL)
    return labels


def initialize_classification_row(doc: dict[str, Any]) -> dict[str, Any]:
    parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
    return {
        "arquivo": doc["arquivo"],
        "titulo": doc["titulo"],
        "data_publicacao": str(parsed.get("data_publicacao", "") or ""),
        "inference": {},
        "classification_source": "residual_pending_wnn",
        "wnn_attempted": False,
        "wnn_accepted": False,
        "wnn_status": "",
        "wnn_confidence": 0.0,
        "wnn_margin": 0.0,
        "wnn_crime_confidence": 0.0,
        "wnn_crime_margin": 0.0,
        "wnn_crime_autonomous": False,
        "wnn_crime_evidence_source": "",
        "x2_tags_referencia": list(doc.get("x2_tags", doc.get("tags", [])) or []),
        "x2_target_labels": [],
        "wnn_top_label": "",
        "wnn_modus_operandi": [],
        "wnn_modus_confidence": 0.0,
        "wnn_modus_evidence_count": 0,
        "wnn_modus_evidence_source": "",
        "wnn_relacao_operacional": "",
        "wnn_marcadores_secundarios": [],
        "wnn_theme_candidate": {},
        "wnn_active_discriminators_count": 0,
        "wnn_active_discriminators": [],
        "wnn_scores": [],
        "wnn_memory_version": 0,
        "wnn_memory_vocab_size": 0,
        "wnn_memory_active_count": 0,
        "wnn_memory_active_positions": [],
        "wnn_memory_binary": "",
        "agent3_reviewed": False,
        "agent3_decision": "",
        "agent3_raw_canonical_label": "",
        "agent3_canonical_label": "",
        "agent3_learning_authorized": False,
        "agent3_learning_status": "",
        "agent4_bleaching_scores": [],
        "agent4_bleaching_margin": 0.0,
        "agent3_tema_principal": "",
        "agent3_marcadores_secundarios": [],
        "agent3_relacao_operacional": "",
        "agent3_modus_operandi": [],
        "agent3_confidence": 0.0,
        "agent3_evidence_text": "",
        "agent3_rationale": "",
        "agent2_incremental_wnn_count": 0,
        "agent2_incremental_wnn": [],
        "cosine_top_label": "",
        "cosine_top_score": 0.0,
        "cosine_top_k": [],
        "semantic_features": str(doc.get("semantic_features", "")),
        "semantic_original_tokens": int(doc.get("linguistic_metrics", {}).get("original_tokens", 0) or 0),
        "semantic_kept_tokens": int(doc.get("linguistic_metrics", {}).get("kept_tokens", 0) or 0),
        "semantic_reduction_ratio": float(doc.get("linguistic_metrics", {}).get("reduction_ratio", 0.0) or 0.0),
        "semantic_backend": str(doc.get("linguistic_metrics", {}).get("backend", "")),
    }


def export_temporal_crime_counts() -> tuple[str | None, str | None]:
    """Export occurrence and crime-type discovery timelines by publication month."""
    batch_paths = sorted(LOTS_DIR.glob("lote_*_classificacoes.csv"))
    if not batch_paths:
        return None, None
    frames = [pd.read_csv(path) for path in batch_paths]
    rows = pd.concat(frames, ignore_index=True)
    if rows.empty or "data_publicacao" not in rows:
        return None, None

    wnn_labels = rows.get("wnn_top_label", pd.Series("", index=rows.index)).fillna("").astype(str)
    residual_labels = rows.get("agent3_canonical_label", pd.Series("", index=rows.index)).fillna("").astype(str)
    accepted = rows.get("wnn_accepted", pd.Series(False, index=rows.index)).fillna(False).astype(str).str.lower().eq("true")
    rows["crime_canonico"] = residual_labels
    rows.loc[accepted & wnn_labels.ne(""), "crime_canonico"] = wnn_labels[accepted & wnn_labels.ne("")]
    rows = rows.loc[rows["crime_canonico"].ne("")].copy()
    rows["periodo"] = pd.to_datetime(rows["data_publicacao"], dayfirst=True, errors="coerce").dt.to_period("M").astype(str)
    rows = rows.loc[rows["periodo"].ne("NaT")]
    if rows.empty:
        return None, None
    summary = (
        rows.groupby(["periodo", "crime_canonico"], as_index=False)
        .size()
        .rename(columns={"size": "ocorrencias"})
        .sort_values(["periodo", "crime_canonico"])
    )
    output = RUN_DIR / "ocorrencias_crime_por_mes.csv"
    summary.to_csv(output, index=False, encoding="utf-8-sig")

    # A type is considered "discovered" in the first month in which its final
    # canonical label occurs in the chronological reserve.  This separates the
    # crime diversity observed in a month from the cumulative taxonomy seen so far.
    first_seen = rows.groupby("crime_canonico")["periodo"].min()
    monthly = (
        rows.groupby("periodo")["crime_canonico"]
        .agg(lambda labels: sorted(set(labels)))
        .reset_index(name="tipos_no_mes")
        .sort_values("periodo")
        .reset_index(drop=True)
    )
    monthly["tipos_distintos_no_mes"] = monthly["tipos_no_mes"].map(len)
    monthly["novos_tipos_no_mes"] = monthly.apply(
        lambda row: sum(first_seen[label] == row["periodo"] for label in row["tipos_no_mes"]),
        axis=1,
    )
    monthly["tipos_distintos_acumulados"] = monthly["novos_tipos_no_mes"].cumsum()
    monthly["tipos_crime_no_mes"] = monthly["tipos_no_mes"].map(" | ".join)
    diversity = monthly.drop(columns=["tipos_no_mes"])
    diversity_output = RUN_DIR / "linha_tempo_tipos_crime.csv"
    diversity.to_csv(diversity_output, index=False, encoding="utf-8-sig")
    return str(output), str(diversity_output)


def _progress_message(
    iteration: int,
    total_batches: int,
    doc_index: int,
    batch_size: int,
    wnn_accepted: int,
    residual: int,
    llm_processed: int,
    provider: str = "",
    model_name: str = "",
) -> str:
    base = f"[classificacao] lote {iteration}/{total_batches}: {doc_index}/{batch_size} noticias"
    body = f"wnn={wnn_accepted}, residuos={residual}, llm={llm_processed}"
    if provider or model_name:
        body += f", ultimo_provider={provider}/{model_name}"
    return f"{base}, {body}"


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
    cumulative_llm = int(sum(int(row.get("llm_processed", 0) or 0) for row in metrics))
    batches = [reserve[index : index + config.batch_size] for index in range(0, len(reserve), config.batch_size)]
    if config.max_batches is not None:
        batches = batches[: config.max_batches]
    total_batches = len(batches)
    total_docs = sum(len(batch) for batch in batches)
    print(
        f"[classificacao] iniciando {total_batches} lotes, {total_docs} noticias na reserva incremental",
        flush=True,
    )
    for iteration, batch in enumerate(batches, start=1):
        if iteration in completed_iterations:
            print(f"[classificacao] lote {iteration}/{total_batches} ja processado, pulando", flush=True)
            continue
        started = time.perf_counter()
        class_confidence_overrides = (
            load_active_thresholds(DYNAMIC_THRESHOLDS_JSON) if config.dynamic_class_thresholds else {}
        )
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
        wnn_multi_discriminator_candidates = 0
        agent3_rare_news = 0
        rare_promoted_candidates = 0
        agent3_errors = 0
        agent4_validated = 0
        agent4_confirmed = 0
        agent4_corrected = 0
        agent4_bleached = 0
        agent4_blocked = 0
        residual_limit = config.max_residual_llm_per_batch
        residual_docs = 0
        wnn_attempted = 0
        wnn_accepted = 0
        wnn_abstained = 0
        print(
            f"[classificacao] lote {iteration}/{total_batches} iniciado ({len(batch)} noticias)",
            flush=True,
        )

        for doc_index, doc in enumerate(batch, start=1):
            row = initialize_classification_row(doc)
            rows.append(row)
            residual_docs += 1
            crime_text = classification_text(doc)
            wnn_score_rows: list[dict[str, object]] = []
            parsed = doc.get("parsed", {}) if isinstance(doc.get("parsed"), dict) else {}
            raw_tags = doc.get("x2_tags", doc.get("tags") or parsed.get("tags") or [])
            target_labels = crime_labels_from_tags(raw_tags if isinstance(raw_tags, list) else [])
            cosine_candidates = top_k_similar_themes(crime_text, top_k=5, preprocessed=True)
            row.update(
                {
                    "x2_target_labels": target_labels,
                    "cosine_top_label": cosine_candidates[0]["label"] if cosine_candidates else "",
                    "cosine_top_score": cosine_candidates[0]["score"] if cosine_candidates else 0.0,
                    "cosine_top_k": cosine_candidates,
                }
            )
            if config.wnn_enabled:
                wnn_attempted += 1
                wnn_result = classify_with_wnn(
                    crime_text,
                    WNN_FEATURE_BANK_PATH,
                    confidence_threshold=config.wnn_confidence_threshold,
                    margin_threshold=config.wnn_margin_threshold,
                    min_active_discriminators=config.wnn_min_active_discriminators,
                    cosine_candidates=cosine_candidates,
                    crime_text=crime_text,
                    class_confidence_overrides=class_confidence_overrides,
                )
                wnn_dict = wnn_result.to_dict()
                wnn_score_rows = [item for item in wnn_dict.get("scores", []) if isinstance(item, dict)]
                row.update(
                    {
                        "wnn_attempted": True,
                        "wnn_accepted": wnn_result.accepted,
                        "wnn_status": wnn_result.status,
                        "wnn_confidence": round(wnn_result.confidence, 4),
                        "wnn_margin": round(wnn_result.margin, 4),
                        "wnn_crime_confidence": round(wnn_result.crime_confidence, 4),
                        "wnn_crime_margin": round(wnn_result.crime_margin, 4),
                        "wnn_crime_autonomous": wnn_result.crime_autonomous,
                        "wnn_crime_evidence_source": wnn_dict.get("crime_evidence_source", ""),
                        "wnn_top_label": wnn_result.top_label,
                        "wnn_modus_operandi": wnn_result.modus_operandi,
                        "wnn_modus_confidence": round(wnn_result.modus_confidence, 4),
                        "wnn_modus_evidence_count": int(wnn_result.modus_evidence_count),
                        "wnn_modus_evidence_source": wnn_dict.get("modus_evidence_source", ""),
                        "wnn_relacao_operacional": (
                            wnn_result.inference.relacao_operacional if wnn_result.inference else ""
                        ),
                        "wnn_marcadores_secundarios": (
                            wnn_result.inference.marcadores_secundarios if wnn_result.inference else []
                        ),
                        "wnn_theme_candidate": wnn_result.theme_candidate or {},
                        "wnn_active_discriminators_count": len(wnn_result.active_discriminators),
                        "wnn_active_discriminators": wnn_dict["active_discriminators"],
                        "wnn_scores": wnn_dict["scores"],
                        "wnn_memory_version": wnn_dict.get("memory_version", 0),
                        "wnn_memory_vocab_size": wnn_dict.get("memory_vocab_size", 0),
                        "wnn_memory_active_count": wnn_dict.get("memory_active_count", 0),
                        "wnn_memory_active_positions": wnn_dict.get("memory_active_positions", []),
                        "wnn_memory_binary": wnn_dict.get("memory_binary", ""),
                    }
                )
                append_event(
                    {
                        "stage": "wnn_classification",
                        "iteration": iteration,
                        "arquivo": doc["arquivo"],
                        "status": wnn_result.status,
                        "accepted": wnn_result.accepted,
                        "confidence": round(wnn_result.confidence, 4),
                        "margin": round(wnn_result.margin, 4),
                        "crime_confidence": round(wnn_result.crime_confidence, 4),
                        "crime_margin": round(wnn_result.crime_margin, 4),
                        "crime_autonomous": wnn_result.crime_autonomous,
                        "crime_evidence_source": wnn_dict.get("crime_evidence_source", ""),
                        "top_label": wnn_result.top_label,
                        "modus_operandi": wnn_result.modus_operandi,
                        "modus_confidence": round(wnn_result.modus_confidence, 4),
                        "modus_evidence_count": int(wnn_result.modus_evidence_count),
                        "modus_evidence_source": wnn_dict.get("modus_evidence_source", ""),
                        "relacao_operacional": wnn_result.inference.relacao_operacional if wnn_result.inference else "",
                        "marcadores_secundarios": wnn_result.inference.marcadores_secundarios if wnn_result.inference else [],
                        "theme_candidate": wnn_result.theme_candidate,
                        "active_discriminators_count": len(wnn_result.active_discriminators),
                        "active_discriminators": wnn_dict["active_discriminators"][:30],
                        "scores": wnn_dict["scores"][:5],
                        "memory_version": wnn_dict.get("memory_version", 0),
                        "memory_vocab_size": wnn_dict.get("memory_vocab_size", 0),
                        "memory_active_count": wnn_dict.get("memory_active_count", 0),
                        "memory_active_positions": wnn_dict.get("memory_active_positions", [])[:120],
                        "memory_binary": wnn_dict.get("memory_binary", ""),
                    }
                )
                if wnn_result.theme_candidate:
                    wnn_multi_discriminator_candidates += 1
                    candidate = wnn_result.theme_candidate
                    append_new_theme_candidate(
                        doc,
                        ResidualReviewAgentResponse(
                            decision="novo_tema_candidato",
                            canonical_label=str(candidate.get("candidate_label", "")),
                            confidence=round(max(0.1, wnn_result.confidence), 4),
                            evidence_text=", ".join(str(item) for item in candidate.get("marker_terms", [])[:12]),
                            rationale=str(candidate.get("rationale", "")),
                            resumo_curto="Composicao multi-discriminador registrada pela WNN para revisao da arvore.",
                            tema_principal=wnn_result.top_label,
                            marcadores_secundarios=[str(item) for item in candidate.get("labels", [])],
                            modus_operandi=wnn_result.modus_operandi,
                            relacao_operacional=str(candidate.get("relation", "coocorrencia_sem_fusao")),
                        ),
                        iteration,
                    )
                if wnn_result.accepted:
                    wnn_accepted += 1
                    row.update(
                        {
                            "classification_source": "agent2_wnn",
                            "inference": wnn_result.inference.model_dump() if wnn_result.inference else {},
                        }
                    )
                    if doc_index % 50 == 0 or doc_index == len(batch):
                        print(
                            _progress_message(
                                iteration,
                                total_batches,
                                doc_index,
                                len(batch),
                                wnn_accepted,
                                residual_docs - wnn_accepted,
                                llm_processed,
                            ),
                            flush=True,
                        )
                    continue
                wnn_abstained += 1

            if residual_limit is not None and agent3_attempted >= residual_limit:
                if doc_index % 50 == 0 or doc_index == len(batch):
                    print(
                        _progress_message(
                            iteration,
                            total_batches,
                            doc_index,
                            len(batch),
                            wnn_accepted,
                            residual_docs - wnn_accepted,
                            llm_processed,
                        ),
                        flush=True,
                    )
                continue

            agent3_attempted += 1
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
                print(
                    "[classificacao] "
                    f"lote {iteration}/{total_batches}: erro na revisao LLM de {doc['arquivo']}: {exc}",
                    flush=True,
                )
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
                            modus_operandi=review.modus_operandi,
                        )
                        agent3_classified -= 1
                        agent3_rare_news -= 1
                        agent3_new_theme_candidates += 1
                        rare_promoted_candidates += 1
                        append_new_theme_candidate(doc, review, iteration)
                        append_event(
                            {
                                "stage": "rare_news_cycle",
                                "iteration": iteration,
                                "arquivo": doc["arquivo"],
                                "status": "promoted_to_theme_candidate",
                                "candidate_label": review.canonical_label,
                                "signature_count": rare_observation.get("signature_count", 0),
                            }
                        )
            elif review.decision == "novo_tema_candidato":
                agent3_new_theme_candidates += 1
                append_new_theme_candidate(doc, review, iteration)
            else:
                agent3_quarantined += 1

            agent4_result = validate_residual_review(
                review,
                target_labels,
                wnn_score_rows,
            )
            learning_review = agent4_result.learning_review
            learning_authorized = agent4_result.learning_authorized
            learning_status = agent4_result.status
            raw_agent3_label = agent4_result.raw_label
            agent4_validated += 1
            if learning_status == "confirmado_por_x2":
                agent4_confirmed += 1
            elif learning_status == "corrigido_por_x2_unica":
                agent4_corrected += 1
            elif learning_status == "desambiguado_por_bleaching_x2_multilabel":
                agent4_bleached += 1
            elif not learning_authorized:
                agent4_blocked += 1
            incorporated = []
            if (
                learning_authorized
                and learning_review.decision == "classificar"
                and learning_review.canonical_label != RARE_NEWS_LABEL
            ):
                incorporated = suggest_discriminator_rules_from_review(doc, learning_review)
            incorporated_wnn = (
                append_discriminators_from_learned_rules(
                    incorporated,
                    WNN_FEATURE_BANK_PATH,
                    max_discriminators_per_label=config.wnn_max_discriminators_per_theme,
                )
                if incorporated
                else []
            )
            learned_rules += len(incorporated_wnn)
            reverse_memory_update = {"recorded": False, "reason": "review_not_classified"}
            if (
                learning_authorized
                and learning_review.decision == "classificar"
                and learning_review.canonical_label != RARE_NEWS_LABEL
            ):
                reverse_memory_update = record_reverse_memory_observation(
                    WNN_FEATURE_BANK_PATH,
                    learning_review.canonical_label,
                    str(doc.get("body_text", "") or doc.get("context", "")),
                    source="verified_residual",
                )
            row.update(
                {
                    "classification_source": "agent3_review",
                    "agent3_reviewed": True,
                    "agent3_decision": review.decision,
                    "agent3_raw_canonical_label": raw_agent3_label,
                    "agent3_canonical_label": learning_review.canonical_label,
                    "agent3_learning_authorized": learning_authorized,
                    "agent3_learning_status": learning_status,
                    "agent4_bleaching_scores": [
                        {"label": label, "score": round(score, 4)}
                        for label, score in agent4_result.bleaching_scores
                    ],
                    "agent4_bleaching_margin": round(agent4_result.bleaching_margin, 4),
                    "agent3_tema_principal": learning_review.tema_principal or learning_review.canonical_label,
                    "agent3_marcadores_secundarios": learning_review.marcadores_secundarios,
                    "agent3_modus_operandi": review.modus_operandi,
                    "agent3_relacao_operacional": review.relacao_operacional,
                    "agent3_confidence": round(review.confidence, 4),
                    "agent3_evidence_text": review.evidence_text,
                    "agent3_rationale": review.rationale,
                    "agent2_incremental_wnn_count": len(incorporated_wnn),
                    "agent2_incremental_wnn": incorporated_wnn,
                    "drasiw_reverse_memory": reverse_memory_update,
                    "inference": review_to_inference(learning_review).model_dump() if review.decision == "classificar" else {},
                }
            )
            append_event(
                {
                    "stage": "agente4_validacao_supervisionada",
                    "iteration": iteration,
                    "arquivo": doc["arquivo"],
                    "x2_target_labels": list(agent4_result.target_labels),
                    "raw_canonical_label": raw_agent3_label,
                    "learning_label": learning_review.canonical_label,
                    "learning_authorized": learning_authorized,
                    "status": learning_status,
                    "bleaching_scores": [
                        {"label": label, "score": round(score, 4)}
                        for label, score in agent4_result.bleaching_scores
                    ],
                    "bleaching_margin": round(agent4_result.bleaching_margin, 4),
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
                    "tema_principal": review.tema_principal or review.canonical_label,
                    "marcadores_secundarios": review.marcadores_secundarios,
                    "modus_operandi": review.modus_operandi,
                    "relacao_operacional": review.relacao_operacional,
                    "agent3_review": review.model_dump(),
                    "agent3_supervision": {
                        "x2_target_labels": target_labels,
                        "raw_canonical_label": raw_agent3_label,
                        "learning_label": learning_review.canonical_label,
                        "learning_authorized": learning_authorized,
                        "status": learning_status,
                    },
                    "agent2_incremental_wnn": incorporated_wnn,
                }
            )
            if doc_index % 50 == 0 or doc_index == len(batch) or llm_processed % 10 == 0:
                print(
                    _progress_message(
                        iteration,
                        total_batches,
                        doc_index,
                        len(batch),
                        wnn_accepted,
                        residual_docs - wnn_accepted,
                        llm_processed,
                        provider,
                        model_name,
                    ),
                    flush=True,
                )

        batch_df = pd.DataFrame(rows)
        batch_output = LOTS_DIR / f"lote_{iteration:04d}_classificacoes.csv"
        batch_output.parent.mkdir(parents=True, exist_ok=True)
        batch_df.to_csv(batch_output, index=False, encoding="utf-8-sig")
        calibration = calibrate_after_batch(
            rows,
            DYNAMIC_THRESHOLDS_JSON,
            iteration,
            default_threshold=config.wnn_confidence_threshold,
            enabled=config.dynamic_class_thresholds,
            min_references=config.dynamic_threshold_min_references,
            step=config.dynamic_threshold_step,
            base_thresholds=CRIME_CONFIDENCE_THRESHOLDS,
        )
        docs = len(batch)
        memory_sizes = pd.to_numeric(batch_df.get("wnn_memory_vocab_size", pd.Series(dtype=float)), errors="coerce").fillna(0)
        wnn_memory_vocab_size = int(memory_sizes.max()) if not batch_df.empty else 0
        wnn_memory_active_avg = (
            float(pd.to_numeric(batch_df.get("wnn_memory_active_count", pd.Series(dtype=float)), errors="coerce").fillna(0).mean())
            if not batch_df.empty
            else 0.0
        )
        semantic_original_tokens = sum(
            int(doc.get("linguistic_metrics", {}).get("original_tokens", 0) or 0)
            for doc in batch
        )
        semantic_kept_tokens = sum(
            int(doc.get("linguistic_metrics", {}).get("kept_tokens", 0) or 0)
            for doc in batch
        )
        semantic_reduction_ratio = (
            1.0 - (semantic_kept_tokens / semantic_original_tokens)
            if semantic_original_tokens
            else 0.0
        )
        cumulative_docs += docs
        cumulative_llm += llm_processed
        cumulative_wnn = int(sum(int(row.get("wnn_accepted", 0) or 0) for row in metrics)) + wnn_accepted
        metrics.append(
            {
                "iteration": iteration,
                "batch_id": f"lote_{iteration:04d}",
                "docs": docs,
                "wnn_attempted": wnn_attempted,
                "wnn_accepted": wnn_accepted,
                "wnn_abstained": wnn_abstained,
                "post_wnn_residual": residual_docs - wnn_accepted,
                "llm_processed": llm_processed,
                "agent3_attempted": agent3_attempted,
                "agent3_reviewed": llm_processed,
                "agent3_classified": agent3_classified,
                "agent3_quarantined": agent3_quarantined,
                "agent3_new_theme_candidates": agent3_new_theme_candidates,
                "agent4_validated": agent4_validated,
                "agent4_confirmed": agent4_confirmed,
                "agent4_corrected": agent4_corrected,
                "agent4_bleached": agent4_bleached,
                "agent4_blocked": agent4_blocked,
                "wnn_multi_discriminator_candidates": wnn_multi_discriminator_candidates,
                    "agent3_rare_news": agent3_rare_news,
                    "rare_promoted_candidates": rare_promoted_candidates,
                    "agent3_errors": agent3_errors,
                "learned_rules": learned_rules,
                "tokens_total": token_total,
                "prompt_tokens_total": prompt_tokens_total,
                "completion_tokens_total": completion_tokens_total,
                "avg_tokens_per_llm": round(token_total / llm_processed, 4) if llm_processed else 0,
                "semantic_original_tokens": semantic_original_tokens,
                "semantic_kept_tokens": semantic_kept_tokens,
                "semantic_reduction_ratio": round(semantic_reduction_ratio, 6),
                "wnn_memory_vocab_size": wnn_memory_vocab_size,
                "wnn_memory_active_avg": round(wnn_memory_active_avg, 4),
                "wnn_rate": round(wnn_accepted / docs, 6) if docs else 0,
                "dynamic_threshold_changes": sum(
                    1
                    for change in calibration.get("changes", [])
                    if change.get("previous_threshold") != change.get("new_threshold")
                ),
                "dynamic_thresholds_path": str(DYNAMIC_THRESHOLDS_JSON),
                "cumulative_docs": cumulative_docs,
                "cumulative_wnn_accepted": cumulative_wnn,
                "cumulative_llm_processed": cumulative_llm,
                "cumulative_wnn_rate": round(cumulative_wnn / cumulative_docs, 6) if cumulative_docs else 0,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "wnn_feature_bank": str(WNN_FEATURE_BANK_PATH),
                "output": str(batch_output),
            }
        )
        pd.DataFrame(metrics).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
        if (
            config.wnn_compaction_interval_batches > 0
            and iteration % config.wnn_compaction_interval_batches == 0
        ):
            try:
                compact_result = compact_feature_bank(
                    WNN_FEATURE_BANK_PATH,
                    max_discriminators_per_label=config.wnn_max_discriminators_per_theme,
                )
                append_event(
                    {
                        "stage": "wnn_compaction",
                        "iteration": iteration,
                        "status": "ok",
                        "result": compact_result,
                    }
                )
            except Exception as exc:
                append_event(
                    {
                        "stage": "wnn_compaction",
                        "iteration": iteration,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        if config.theme_tree_review_interval_batches > 0 and iteration % config.theme_tree_review_interval_batches == 0:
            try:
                tree_result = run_theme_tree_organizer(config)
                append_event(
                    {
                        "stage": "agente_organizador_arvore_incremental",
                        "iteration": iteration,
                        "status": "ok",
                        "result": tree_result,
                    }
                )
            except Exception as exc:
                append_event(
                    {
                        "stage": "agente_organizador_arvore_incremental",
                        "iteration": iteration,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        if (
            config.dashboard_update_interval_batches > 0
            and iteration % config.dashboard_update_interval_batches == 0
        ):
            try:
                update_dashboard()
            except Exception as exc:
                append_event({"stage": "dashboard_comparacao", "status": "error", "error": str(exc)})
        conclusion = (
            f"[classificacao] lote {iteration}/{total_batches} concluido: docs={docs}, wnn={wnn_accepted}, "
            f"residuos_pos_wnn={residual_docs - wnn_accepted}, llm={llm_processed}, "
            f"regras_aprendidas={learned_rules}, candidatos_compostos={wnn_multi_discriminator_candidates}, "
            f"taxa_wnn={wnn_accepted / docs:.2%}, tempo={metrics[-1]['elapsed_seconds']}s"
        )
        print(conclusion, flush=True)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    temporal_counts_csv, temporal_diversity_csv = export_temporal_crime_counts()
    result = {
        "stage": "processar_lotes",
        "metrics_csv": str(METRICS_CSV),
        "temporal_crime_counts_csv": temporal_counts_csv,
        "temporal_crime_diversity_csv": temporal_diversity_csv,
        "batches": len(metrics),
    }
    write_json(RUN_DIR / "processar_lotes_result.json", result)
    append_event(result)
    return result


def main() -> None:
    print(write_json(RUN_DIR / "processar_lotes_result.json", run(RunConfig())))


if __name__ == "__main__":
    main()
