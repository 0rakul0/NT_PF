from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from scripts.incremental.common import (
    CLUSTER_ASSIGNMENTS_CSV,
    CLUSTER_SUMMARY_CSV,
    RUN_DIR,
    SAMPLE_CSV,
    RunConfig,
    append_event,
    docs_by_manifest,
    write_json,
)


def build_semantic_clusters(sample: list[dict[str, Any]], seed: int) -> tuple[pd.DataFrame, TfidfVectorizer, Any]:
    texts = [item["context"] for item in sample]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=20000,
        token_pattern=r"(?u)\b[a-zA-Z0-9_]{3,}\b",
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(80, max(2, tfidf.shape[1] - 1), max(2, len(sample) - 1))
    embeddings = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(tfidf)
    embeddings = Normalizer(copy=False).fit_transform(embeddings)
    min_cluster_size = max(8, min(60, int(np.sqrt(len(sample))) * 2))
    try:
        labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(4, min_cluster_size // 2)).fit_predict(embeddings)
        algorithm = "hdbscan"
        if not any(int(label) != -1 for label in labels):
            raise ValueError("hdbscan retornou apenas ruido")
    except Exception:
        k = max(6, min(24, int(np.sqrt(len(sample)))))
        labels = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
        algorithm = "minibatch_kmeans_fallback"
    rows = [
        {
            "arquivo": item["arquivo"],
            "titulo": item["titulo"],
            "tags": " | ".join(str(tag) for tag in item.get("tags", [])),
            "cluster_id": int(label),
            "context": item["context"],
        }
        for item, label in zip(sample, labels, strict=False)
    ]
    cluster_rows = pd.DataFrame(rows)
    cluster_rows.attrs["algorithm"] = algorithm
    return cluster_rows, vectorizer, tfidf


def top_terms_for_cluster(tfidf, vectorizer: TfidfVectorizer, cluster_indices: list[int], top_n: int = 12) -> list[str]:
    if not cluster_indices:
        return []
    terms = np.array(vectorizer.get_feature_names_out())
    centroid = np.asarray(tfidf[cluster_indices].mean(axis=0)).ravel()
    ordered = centroid.argsort()[::-1]
    selected: list[str] = []
    for index in ordered:
        term = str(terms[index])
        if len(term) < 4 or term in selected:
            continue
        selected.append(term)
        if len(selected) >= top_n:
            break
    return selected


def summarize_clusters(cluster_rows: pd.DataFrame, tfidf, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    summaries = []
    for cluster_id in sorted(cluster_rows["cluster_id"].unique()):
        subset = cluster_rows.loc[cluster_rows["cluster_id"] == cluster_id]
        terms = top_terms_for_cluster(tfidf, vectorizer, subset.index.tolist())
        titles = subset["titulo"].head(8).tolist()
        tags = []
        for raw_tags in subset.get("tags", pd.Series(dtype=str)).fillna("").tolist():
            for tag in str(raw_tags).split(" | "):
                tag = tag.strip()
                if tag and tag not in tags:
                    tags.append(tag)
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(subset)),
                "is_noise": bool(int(cluster_id) == -1),
                "top_terms": " | ".join(terms),
                "sample_titles": " | ".join(str(title) for title in titles),
                "sample_tags": " | ".join(tags[:20]),
            }
        )
    return pd.DataFrame(summaries).sort_values(["is_noise", "size"], ascending=[True, False]).reset_index(drop=True)


def run(config: RunConfig) -> dict[str, object]:
    sample = docs_by_manifest(SAMPLE_CSV)
    cluster_rows, vectorizer, tfidf = build_semantic_clusters(sample, config.seed)
    cluster_rows.to_csv(CLUSTER_ASSIGNMENTS_CSV, index=False, encoding="utf-8-sig")
    cluster_summary = summarize_clusters(cluster_rows, tfidf, vectorizer)
    cluster_summary.to_csv(CLUSTER_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    result = {
        "stage": "clusterizacao_inicial",
        "algorithm": cluster_rows.attrs.get("algorithm", ""),
        "sample_docs": len(sample),
        "clusters_total": int(cluster_summary.loc[cluster_summary["cluster_id"] != -1, "cluster_id"].nunique()),
        "noise_clusters": int((cluster_summary["cluster_id"] == -1).sum()),
        "cluster_assignments_csv": str(CLUSTER_ASSIGNMENTS_CSV),
        "cluster_summary_csv": str(CLUSTER_SUMMARY_CSV),
    }
    write_json(RUN_DIR / "clusterizacao_result.json", result)
    append_event(result)
    return result


def main() -> None:
    print(write_json(RUN_DIR / "clusterizacao_result.json", run(RunConfig())))


if __name__ == "__main__":
    main()
