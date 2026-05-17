from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
from scripts.incremental.texto_dominio import build_domain_cluster_text


def domain_term_set(value: object) -> set[str]:
    return {term.strip() for term in str(value or "").split(" | ") if term.strip()}


FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("crimes_contra_criancas", ("abuso sexual", "pornografia infantil", "material pornografico", "crime contra crianca")),
    ("trafico_drogas", ("trafico", "droga")),
    ("crime_organizado", ("crime organizado", "faccao criminosa", "organizacao criminosa", "associacao criminosa")),
    ("contrabando_descaminho", ("contrabando", "descaminho", "cigarro contrabandeado")),
    ("corrupcao_recursos_publicos", ("licitacao", "recurso publico", "desvio de recurso publico", "corrupcao")),
    ("crimes_ambientais", ("crime ambiental", "garimpo ilegal", "ouro ilegal", "mineracao ilegal", "extracao ilegal", "madeira ilegal", "desmatamento")),
    ("falsificacao_documental", ("documento falso", "falsidade ideologica", "falsificacao")),
    ("crimes_previdenciarios", ("crime previdenciario", "beneficio previdenciario", "fraude previdenciaria", "fraude contra o inss")),
    ("lavagem_dinheiro", ("lavagem de dinheiro",)),
    ("moeda_falsa", ("moeda falsa", "cedula falsa")),
    ("armas_municoes", ("arma de fogo", "arma ilegal", "armamento", "municao", "posse ilegal")),
    ("radiodifusao_clandestina", ("radiodifusao clandestina", "radio clandestina")),
    ("crimes_eleitorais", ("crime eleitoral", "corrupcao eleitoral")),
    ("crimes_migratorios", ("migracao ilegal",)),
    ("saude_publica", ("medicamento ilegal", "anabolizante", "adulteracao")),
    ("roubo_assalto", ("roubo", "assalto")),
    ("receptacao", ("receptacao",)),
)


def dominant_family(terms: set[str]) -> str:
    scores: dict[str, int] = {}
    for family, needles in FAMILY_RULES:
        for term in terms:
            folded = term.lower()
            if any(needle in folded for needle in needles):
                scores[family] = scores.get(family, 0) + 1
    if not scores:
        return "indefinido"
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def consolidate_by_cosine(cluster_rows: pd.DataFrame, tfidf, threshold: float = 0.45) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    cluster_ids = sorted(int(cluster_id) for cluster_id in cluster_rows["cluster_id"].unique() if int(cluster_id) != -1)
    if len(cluster_ids) < 2:
        cluster_rows["raw_cluster_id"] = cluster_rows["cluster_id"]
        return cluster_rows, []

    centroids = []
    domain_terms: dict[int, set[str]] = {}
    for cluster_id in cluster_ids:
        subset = cluster_rows.loc[cluster_rows["cluster_id"] == cluster_id]
        centroids.append(np.asarray(tfidf[subset.index.tolist()].mean(axis=0)).ravel())
        domain_terms[cluster_id] = set().union(*(domain_term_set(value) for value in subset["cluster_domain_terms"].fillna("").tolist()))
    families = {cluster_id: dominant_family(terms) for cluster_id, terms in domain_terms.items()}

    similarities = cosine_similarity(np.vstack(centroids))
    parent = {cluster_id: cluster_id for cluster_id in cluster_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for left_index, left_id in enumerate(cluster_ids):
        for right_index in range(left_index + 1, len(cluster_ids)):
            right_id = cluster_ids[right_index]
            score = float(similarities[left_index, right_index])
            if score < threshold:
                continue
            if families[left_id] == "indefinido" or families[left_id] != families[right_id]:
                continue
            union(left_id, right_id)

    components: dict[int, list[int]] = {}
    for cluster_id in cluster_ids:
        components.setdefault(find(cluster_id), []).append(cluster_id)

    consolidated = cluster_rows.copy()
    consolidated["raw_cluster_id"] = consolidated["cluster_id"]
    raw_to_group = {raw: root for root, members in components.items() for raw in members}
    consolidated["cluster_id"] = consolidated["cluster_id"].map(lambda value: raw_to_group.get(int(value), int(value)))
    merge_records = [
        {
            "cosine_group_id": root,
            "merged_raw_cluster_ids": members,
            "dominant_family": families.get(root, "indefinido"),
            "size": int(consolidated.loc[consolidated["cluster_id"] == root].shape[0]),
        }
        for root, members in sorted(components.items())
        if len(members) > 1
    ]
    return consolidated, merge_records


def build_semantic_clusters(sample: list[dict[str, Any]], seed: int) -> tuple[pd.DataFrame, TfidfVectorizer, Any, list[dict[str, object]]]:
    prepared = [build_domain_cluster_text(item) for item in sample]
    texts = [cluster_text for cluster_text, _terms in prepared]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.75,
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
        non_noise_clusters = {int(label) for label in labels if int(label) != -1}
        noise_ratio = float(np.mean(np.asarray(labels) == -1))
        if len(non_noise_clusters) < 8 or noise_ratio > 0.35:
            raise ValueError(f"hdbscan instavel para texto criminal filtrado: clusters={len(non_noise_clusters)}, noise_ratio={noise_ratio:.3f}")
    except Exception:
        k = max(12, min(40, int(np.sqrt(len(sample)))))
        labels = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
        algorithm = "minibatch_kmeans_fallback"
    rows = [
        {
            "arquivo": item["arquivo"],
            "titulo": item["titulo"],
            "tags": " | ".join(str(tag) for tag in item.get("tags", [])),
            "cluster_id": int(label),
            "cluster_text": cluster_text,
            "cluster_domain_terms": " | ".join(domain_terms),
            "context": item["context"],
        }
        for item, label, (cluster_text, domain_terms) in zip(sample, labels, prepared, strict=False)
    ]
    cluster_rows = pd.DataFrame(rows)
    cluster_rows, cosine_merges = consolidate_by_cosine(cluster_rows, tfidf)
    cluster_rows.attrs["algorithm"] = algorithm
    cluster_rows.attrs["cosine_merges"] = cosine_merges
    return cluster_rows, vectorizer, tfidf, cosine_merges


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
        selected.append(term.replace("_", " "))
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
                "raw_cluster_ids": " | ".join(str(item) for item in sorted(int(value) for value in subset.get("raw_cluster_id", subset["cluster_id"]).unique())),
                "size": int(len(subset)),
                "is_noise": bool(int(cluster_id) == -1),
                "top_terms": " | ".join(terms),
                "sample_titles": " | ".join(str(title) for title in titles),
                "sample_tags": " | ".join(tags[:20]),
                "domain_terms": " | ".join(
                    dict.fromkeys(
                        term.strip()
                        for raw_terms in subset.get("cluster_domain_terms", pd.Series(dtype=str)).fillna("").tolist()
                        for term in str(raw_terms).split(" | ")
                        if term.strip()
                    )
                ),
            }
        )
    return pd.DataFrame(summaries).sort_values(["is_noise", "size"], ascending=[True, False]).reset_index(drop=True)


def run(config: RunConfig) -> dict[str, object]:
    sample = docs_by_manifest(SAMPLE_CSV)
    cluster_rows, vectorizer, tfidf, cosine_merges = build_semantic_clusters(sample, config.seed)
    cluster_rows.to_csv(CLUSTER_ASSIGNMENTS_CSV, index=False, encoding="utf-8-sig")
    cluster_summary = summarize_clusters(cluster_rows, tfidf, vectorizer)
    cluster_summary.to_csv(CLUSTER_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    result = {
        "stage": "clusterizacao_inicial",
        "algorithm": cluster_rows.attrs.get("algorithm", ""),
        "sample_docs": len(sample),
        "clusters_total": int(cluster_summary.loc[cluster_summary["cluster_id"] != -1, "cluster_id"].nunique()),
        "raw_clusters_total": int(cluster_rows["raw_cluster_id"].nunique()) if "raw_cluster_id" in cluster_rows.columns else int(cluster_summary.loc[cluster_summary["cluster_id"] != -1, "cluster_id"].nunique()),
        "noise_clusters": int((cluster_summary["cluster_id"] == -1).sum()),
        "cosine_merge_groups": len(cosine_merges),
        "cosine_merges": cosine_merges,
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
