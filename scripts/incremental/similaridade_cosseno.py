from __future__ import annotations

import pickle
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from scripts.agentes.agente1_temas import THEME_RULES
from scripts.incremental.common import (
    CLUSTER_ASSIGNMENTS_CSV,
    COSINE_PROFILE_JSON,
    COSINE_PROFILE_PKL,
    RUN_DIR,
    THEMES_JSON,
    RunConfig,
    append_event,
    read_json,
    write_json,
)


PREFERRED_TERMS_BY_LABEL = {label: list(terms) for label, terms in THEME_RULES}


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def term_pattern(term: str) -> str:
    tokens = re.findall(r"[a-z0-9]{3,}", fold_text(term))
    if not tokens:
        return ""
    return r"\s+".join(rf"\b{re.escape(token)}\w*\b" for token in tokens)


def doc_matches_label(row: pd.Series, label: str) -> bool:
    terms = PREFERRED_TERMS_BY_LABEL.get(label, [])
    if not terms:
        return True
    text = fold_text(" ".join([str(row.get("titulo", "")), str(row.get("tags", "")), str(row.get("context", ""))]))
    return any(re.search(pattern, text) for pattern in (term_pattern(term) for term in terms) if pattern)


def fit_theme_profiles() -> dict[str, Any]:
    themes_payload = read_json(THEMES_JSON)
    cluster_rows = pd.read_csv(CLUSTER_ASSIGNMENTS_CSV)
    theme_seed_texts = []
    for theme in themes_payload.get("themes", []):
        if theme.get("decision") == "accept":
            label = str(theme["canonical_theme"])
            seed_terms = PREFERRED_TERMS_BY_LABEL.get(label, [])
            theme_seed_texts.append(" ".join([label.replace("_", " "), *seed_terms, *seed_terms]))
    texts = cluster_rows["context"].fillna("").astype(str).tolist() + theme_seed_texts
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=30000,
        token_pattern=r"(?u)\b[a-zA-Z0-9_]{3,}\b",
    )
    matrix = vectorizer.fit_transform(texts)
    docs_matrix = matrix[: len(cluster_rows)]
    labels: list[str] = []
    centroids = []
    metadata: list[dict[str, Any]] = []
    terms = np.array(vectorizer.get_feature_names_out())

    for theme in themes_payload.get("themes", []):
        if theme.get("decision") != "accept":
            continue
        cluster_ids = [int(item) for item in theme.get("included_cluster_ids", [])]
        subset = cluster_rows.loc[cluster_rows["cluster_id"].isin(cluster_ids)]
        matched_indices = [int(index) for index, row in subset.iterrows() if doc_matches_label(row, str(theme["canonical_theme"]))]
        indices = matched_indices or subset.index.tolist()
        seed_text = " ".join(
            [
                str(theme["canonical_theme"]).replace("_", " "),
                *PREFERRED_TERMS_BY_LABEL.get(str(theme["canonical_theme"]), []),
                *PREFERRED_TERMS_BY_LABEL.get(str(theme["canonical_theme"]), []),
            ]
        )
        seed_vector = vectorizer.transform([seed_text])
        if indices:
            centroid = (docs_matrix[indices].mean(axis=0) + seed_vector) / 2
        else:
            centroid = seed_vector
        centroid = normalize(np.asarray(centroid), norm="l2")
        dense = np.asarray(centroid).ravel()
        top_indices = dense.argsort()[::-1][:12]
        top_terms = [str(terms[index]) for index in top_indices if dense[index] > 0]
        labels.append(str(theme["canonical_theme"]))
        centroids.append(dense)
        metadata.append(
            {
                "label": str(theme["canonical_theme"]),
                "cluster_ids": cluster_ids,
                "documents": len(indices),
                "matched_documents": len(matched_indices),
                "top_terms": top_terms,
                "evidence_terms": [*PREFERRED_TERMS_BY_LABEL.get(str(theme["canonical_theme"]), []), *theme.get("evidence_terms", [])][:12],
            }
        )

    centroid_matrix = np.vstack(centroids) if centroids else np.empty((0, matrix.shape[1]))
    return {
        "vectorizer": vectorizer,
        "labels": labels,
        "centroids": centroid_matrix,
        "metadata": metadata,
    }


def save_profiles(profile: dict[str, Any]) -> None:
    COSINE_PROFILE_PKL.parent.mkdir(parents=True, exist_ok=True)
    with COSINE_PROFILE_PKL.open("wb") as handle:
        pickle.dump(profile, handle)
    write_json(
        COSINE_PROFILE_JSON,
        {
            "labels": profile["labels"],
            "profiles": profile["metadata"],
        },
    )


def load_profiles() -> dict[str, Any] | None:
    if not COSINE_PROFILE_PKL.exists():
        return None
    with COSINE_PROFILE_PKL.open("rb") as handle:
        return pickle.load(handle)


def top_k_similar_themes(text: str, top_k: int = 5) -> list[dict[str, Any]]:
    profile = load_profiles()
    if not profile or len(profile.get("labels", [])) == 0:
        return []
    vector = profile["vectorizer"].transform([text])
    scores = cosine_similarity(vector, profile["centroids"])[0]
    order = scores.argsort()[::-1][:top_k]
    metadata_by_label = {item["label"]: item for item in profile.get("metadata", [])}
    candidates: list[dict[str, Any]] = []
    for index in order:
        label = profile["labels"][int(index)]
        meta = metadata_by_label.get(label, {})
        candidates.append(
            {
                "label": label,
                "score": round(float(scores[int(index)]), 6),
                "cluster_ids": meta.get("cluster_ids", []),
                "top_terms": meta.get("top_terms", [])[:8],
                "evidence_terms": meta.get("evidence_terms", [])[:8],
            }
        )
    return candidates


def run(config: RunConfig) -> dict[str, object]:
    profile = fit_theme_profiles()
    save_profiles(profile)
    result = {
        "stage": "similaridade_cosseno",
        "profile_pkl": str(COSINE_PROFILE_PKL),
        "profile_json": str(COSINE_PROFILE_JSON),
        "themes_profiled": len(profile["labels"]),
    }
    write_json(RUN_DIR / "similaridade_cosseno_result.json", result)
    append_event(result)
    return result


def main() -> None:
    print(write_json(RUN_DIR / "similaridade_cosseno_result.json", run(RunConfig())))


if __name__ == "__main__":
    main()
