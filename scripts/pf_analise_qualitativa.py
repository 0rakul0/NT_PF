from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer

try:
    from project_config import (
        ANALYSIS_DIR as DEFAULT_OUTPUT_DIR,
        BRAZIL_STATES,
        CONTENT_CSV as DEFAULT_CONTENT_CSV,
        INDEX_CSV as DEFAULT_INDEX_CSV,
        LLM_METADATA_JSONL as DEFAULT_LLM_METADATA_JSONL,
    )
except ModuleNotFoundError:
    from scripts.project_config import (
        ANALYSIS_DIR as DEFAULT_OUTPUT_DIR,
        BRAZIL_STATES,
        CONTENT_CSV as DEFAULT_CONTENT_CSV,
        INDEX_CSV as DEFAULT_INDEX_CSV,
        LLM_METADATA_JSONL as DEFAULT_LLM_METADATA_JSONL,
    )


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()

PORTUGUESE_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate",
    "com", "como", "contra", "da", "das", "de", "dela", "dele", "deles", "demais", "depois",
    "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa",
    "essas", "esse", "esses", "esta", "estao", "estar", "estas", "estava", "este", "estes",
    "foi", "foram", "ha", "isso", "isto", "ja", "la", "lhe", "lhes", "mais", "mas", "mesmo",
    "na", "nas", "nao", "nem", "no", "nos", "nossa", "nosso", "nossos", "num", "numa", "o",
    "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que",
    "quem", "se", "sem", "ser", "seu", "seus", "sido", "sob", "sobre", "sua", "suas", "tal",
    "tambem", "te", "tem", "tendo", "ter", "teve", "tipo", "toda", "todas", "todo", "todos",
    "um", "uma", "umas", "uns", "vos", "Ã ", "Ã s", "Ã©",
    "policia", "federal", "pf", "operacao", "operacoes", "noticia", "noticias"
}

CRIME_PATTERNS = {
    "lavagem_dinheiro": [
        "lavagem de dinheiro", "lavagem de capitais", "ocultacao de bens",
        "ocultaÃ§Ã£o de bens", "evasao de divisas", "evasÃ£o de divisas"
    ],
    "trafico_drogas": [
        "trafico de drogas", "trÃ¡fico de drogas", "entorpecentes", "maconha",
        "cocaina", "cocaÃ­na", "crack"
    ],
    "trafico_armas": [
        "trafico de armas", "trÃ¡fico de armas", "armas de fogo", "municoes", "muniÃ§Ãµes"
    ],
    "crimes_ambientais": [
        "crime ambiental", "crimes ambientais", "garimpo ilegal", "desmatamento",
        "pesca ilegal", "extraÃ§Ã£o ilegal", "extracao ilegal", "madeira ilegal", "ibama", "icmbio"
    ],
    "crime_eleitoral": [
        "crime eleitoral", "crimes eleitorais", "sufragio", "sufrÃ¡gio", "compra de votos"
    ],
    "corrupcao_desvio": [
        "corrupcao", "corrupÃ§Ã£o", "desvio de recursos", "desvio de verbas",
        "licitacao", "licitaÃ§Ã£o", "fraude em licitacao", "fraude em licitaÃ§Ã£o",
        "peculato", "prefeitura"
    ],
    "abuso_sexual_infantil": [
        "abuso sexual infantojuvenil", "abuso sexual de criancas", "abuso sexual de crianÃ§as",
        "pornografia infantil", "exploracao sexual infantil", "exploraÃ§Ã£o sexual infantil",
        "pedofilia"
    ],
    "contrabando_descaminho": [
        "contrabando", "descaminho", "cigarros falsificados", "cigarro falsificado",
        "mercadoria estrangeira", "importacao irregular", "importaÃ§Ã£o irregular"
    ],
    "fraude_bancaria_cibernetica": [
        "fraude bancaria", "fraude bancÃ¡ria", "fraudes bancarias", "fraudes bancÃ¡rias",
        "golpe", "crime cibernetico", "crime cibernÃ©tico", "phishing", "internet banking"
    ],
    "trabalho_escravo_trafico_pessoas": [
        "trabalho escravo", "trabalho analogo ao de escravo", "trabalho anÃ¡logo ao de escravo",
        "trafico de pessoas", "trÃ¡fico de pessoas", "condicoes degradantes", "condiÃ§Ãµes degradantes"
    ],
    "organizacao_criminosa": [
        "organizacao criminosa", "organizaÃ§Ã£o criminosa", "facÃ§Ã£o criminosa", "associacao criminosa",
        "associaÃ§Ã£o criminosa", "milicia", "milÃ­cia"
    ],
    "falsidade_documental": [
        "falsidade ideologica", "falsidade ideolÃ³gica", "documento falso", "passaporte falso",
        "moeda falsa", "certidao falsa", "certidÃ£o falsa"
    ],
}

MODUS_PATTERNS = {
    "busca_apreensao": ["busca e apreensao", "busca e apreensÃ£o", "mandado de busca", "mandados de busca"],
    "prisao": ["prisao preventiva", "prisÃ£o preventiva", "prisao temporaria", "prisÃ£o temporÃ¡ria", "mandado de prisÃ£o", "flagrante"],
    "bloqueio_bens": ["bloqueio de bens", "sequestro de bens", "indisponibilidade de bens", "bloqueio judicial"],
    "fiscalizacao": ["fiscalizacao", "fiscalizaÃ§Ã£o", "inspecao", "inspeÃ§Ã£o", "vistoria"],
    "atuacao_online": ["internet", "rede social", "whatsapp", "telegram", "site", "online"],
    "cooperacao_interagencias": ["ficco", "ibama", "receita federal", "funai", "prf", "cgU", "controladoria-geral da uniao", "gaeco", "forca integrada", "forÃ§a integrada"],
    "fronteira_transnacional": ["fronteira", "paraguai", "bolivia", "bolÃ­via", "argentina", "internacional", "transnacional"],
    "resgate_vitimas": ["resgate", "vitimas", "vÃ­timas", "trabalhadores resgatados", "pessoas resgatadas"],
    "desarticulacao_rede": ["desarticular", "desarticulaÃ§Ã£o", "esquema criminoso", "quadrilha", "rede criminosa"],
    "combate_financeiro": ["movimentacao financeira", "movimentaÃ§Ã£o financeira", "contas bancarias", "contas bancÃ¡rias", "sigilo bancario", "sigilo bancÃ¡rio"],
}

STATE_LOOKUP = {item["uf"]: item for item in BRAZIL_STATES}
STATE_NAME_TO_UF = {item["state"].lower(): item["uf"] for item in BRAZIL_STATES}
STATE_TAG_TO_UF = {
    unicodedata.normalize("NFKD", item["state"]).encode("ascii", "ignore").decode("ascii").lower(): item["uf"]
    for item in BRAZIL_STATES
}
STATE_TAG_TO_UF.update({item["uf"].lower(): item["uf"] for item in BRAZIL_STATES})
GEO_PHRASES = sorted(
    {item["state"].lower() for item in BRAZIL_STATES}
    | {item["capital"].lower() for item in BRAZIL_STATES}
    | {"brasil", "paraguai", "argentina", "bolivia", "uruguai", "amazonia"},
    key=len,
    reverse=True,
)
UF_PATTERN = "|".join(sorted([item["uf"].lower() for item in BRAZIL_STATES]))
DATALINE_REGEX = re.compile(r"\b[\w' -]{2,60}/([A-Z]{2})\b")
TITLE_STATE_PATTERNS = {
    item["uf"]: re.compile(rf"\b{re.escape(fold_text(item['state']))}\b")
    for item in BRAZIL_STATES
}
TITLE_STATE_PATTERNS["SP"] = re.compile(r"\bsao paulo\b(?!\s+de\b)")
TITLE_STATE_PATTERNS["PA"] = re.compile(r"\bestado do para\b|\bestado do parÃ¡\b")
# Override the broad geography patterns above with tighter, location-oriented rules.
DATALINE_REGEX = re.compile(r"\b[\w' -]{2,60}/([A-Z]{2})\b")
TITLE_STATE_PATTERNS = {
    item["uf"]: re.compile(rf"\b{re.escape(fold_text(item['state']))}\b")
    for item in BRAZIL_STATES
}
TITLE_STATE_PATTERNS["SP"] = re.compile(r"\bsao paulo\b(?!\s+de\b)")
TITLE_STATE_PATTERNS["PA"] = re.compile(r"\bestado do para\b")


@dataclass
class AnalysisArtifacts:
    corpus: pd.DataFrame
    neighbors: pd.DataFrame
    recurring_pairs: pd.DataFrame
    cluster_summary: pd.DataFrame
    temporal_summary: pd.DataFrame
    canonical_cluster_summary: pd.DataFrame
    crime_summary: pd.DataFrame
    modus_summary: pd.DataFrame
    semantic_series: pd.DataFrame


@dataclass(frozen=True)
class AnalysisConfig:
    index_csv: Path = DEFAULT_INDEX_CSV
    content_csv: Path = DEFAULT_CONTENT_CSV
    output_dir: Path = DEFAULT_OUTPUT_DIR
    llm_metadata_jsonl: Path = DEFAULT_LLM_METADATA_JSONL
    neighbors: int = 6
    series_threshold: float = 0.70
    random_state: int = 42


def build_analysis_config(
    index_csv: Path | str | None = None,
    content_csv: Path | str | None = None,
    output_dir: Path | str | None = None,
    llm_metadata_jsonl: Path | str | None = None,
    neighbors: int | None = None,
    series_threshold: float | None = None,
    random_state: int | None = None,
) -> AnalysisConfig:
    env_neighbors = os.getenv("PF_ANALYSIS_NEIGHBORS", "").strip()
    env_series_threshold = os.getenv("PF_ANALYSIS_SERIES_THRESHOLD", "").strip().replace(",", ".")
    env_random_state = os.getenv("PF_ANALYSIS_RANDOM_STATE", "").strip()
    return AnalysisConfig(
        index_csv=Path(index_csv or os.getenv("PF_INDEX_CSV", "") or DEFAULT_INDEX_CSV),
        content_csv=Path(content_csv or os.getenv("PF_CONTENT_CSV", "") or DEFAULT_CONTENT_CSV),
        output_dir=Path(output_dir or os.getenv("PF_ANALYSIS_OUTPUT_DIR", "") or DEFAULT_OUTPUT_DIR),
        llm_metadata_jsonl=Path(llm_metadata_jsonl or os.getenv("PF_LLM_METADATA_JSONL", "") or DEFAULT_LLM_METADATA_JSONL),
        neighbors=neighbors if neighbors is not None else int(env_neighbors) if env_neighbors.isdigit() else 6,
        series_threshold=(
            series_threshold
            if series_threshold is not None
            else float(env_series_threshold)
            if env_series_threshold
            else 0.70
        ),
        random_state=random_state if random_state is not None else int(env_random_state) if env_random_state.isdigit() else 42,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_markdown_text(path_value: str) -> str:
    path = Path(path_value)
    return path.read_text(encoding="utf-8")


def load_llm_metadata(jsonl_path: Path) -> pd.DataFrame:
    if not jsonl_path.exists():
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = payload.get("metadata", {})
            metadata_extraido = payload.get("metadata_extraido", {})
            inferencia_llm = payload.get("inferencia_llm", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if not isinstance(metadata_extraido, dict):
                metadata_extraido = {}
            if not isinstance(inferencia_llm, dict):
                inferencia_llm = {}

            if metadata:
                llm_identidade = metadata.get("identidade")
                llm_data = metadata.get("data")
                llm_tipo = metadata.get("tipo")
                llm_tags = metadata.get("tags", [])
                llm_rotulo_resumido = metadata.get("rotulo_resumido")
                llm_nomes_operacao_encontrados = metadata.get("nomes_operacao_encontrados", [])
                llm_crimes_mais_presentes = metadata.get("crimes_mais_presentes", [])
                llm_exemplos_iniciais = metadata.get("exemplos_iniciais", [])
                llm_modus_operandi = []
            else:
                llm_identidade = inferencia_llm.get("identidade_canonica")
                llm_data = metadata_extraido.get("data_publicacao")
                llm_tipo = inferencia_llm.get("classificacao")
                llm_tags = metadata_extraido.get("tags", [])
                llm_rotulo_resumido = inferencia_llm.get("identidade_canonica")
                operation_name = str(metadata_extraido.get("nome_operacao_encontrado", "")).strip()
                llm_nomes_operacao_encontrados = [operation_name] if operation_name else []
                llm_crimes_mais_presentes = inferencia_llm.get("crimes_mais_presentes", [])
                llm_exemplos_iniciais = [metadata_extraido.get("titulo")] if metadata_extraido.get("titulo") else []
                llm_modus_operandi = inferencia_llm.get("modus_operandi", [])

            rows.append(
                {
                    "arquivo_markdown": str(payload.get("arquivo", "")).strip(),
                    "llm_identidade": llm_identidade,
                    "llm_data": llm_data,
                    "llm_tipo": llm_tipo,
                    "llm_tags": llm_tags,
                    "llm_rotulo_resumido": llm_rotulo_resumido,
                    "llm_nomes_operacao_encontrados": llm_nomes_operacao_encontrados,
                    "llm_crimes_mais_presentes": llm_crimes_mais_presentes,
                    "llm_modus_operandi": llm_modus_operandi,
                    "llm_exemplos_iniciais": llm_exemplos_iniciais,
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["arquivo_markdown"], keep="last").reset_index(drop=True)


def strip_markdown(text: str) -> str:
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"[*_>#-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "sem_rotulo"


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def semantic_preprocess(text: str) -> str:
    original = re.sub(r"\b[\w' -]{2,60}/[A-Z]{2}\b", " ", text)
    folded = fold_text(original)
    for phrase in GEO_PHRASES:
        folded = re.sub(rf"\b{re.escape(phrase)}\b", " ", folded)
    folded = re.sub(rf"\b(?:{UF_PATTERN})\b", " ", folded)
    folded = re.sub(r"\d+", " ", folded)
    folded = re.sub(r"\s+", " ", folded)
    return folded.strip()


def parse_datetime(value: str | float | None) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    return pd.to_datetime(text, format="%d/%m/%Y %Hh%M", errors="coerce")


def extract_state_mentions(
    title: str,
    subtitle: str,
    tags: str,
    markdown_text: str,
) -> list[str]:
    found_ufs: set[str] = set()
    headline_text = fold_text(f"{title} {subtitle}")
    lead_text = markdown_text[:1200]

    for raw_tag in [part.strip() for part in str(tags).split("|") if part.strip()]:
        tag_key = fold_text(raw_tag)
        uf = STATE_TAG_TO_UF.get(tag_key)
        if uf:
            found_ufs.add(uf)

    for uf, pattern in TITLE_STATE_PATTERNS.items():
        if pattern.search(headline_text):
            found_ufs.add(uf)

    for match in DATALINE_REGEX.findall(lead_text):
        uf = match.upper()
        if uf in STATE_LOOKUP:
            found_ufs.add(uf)

    return sorted(found_ufs)


def ufs_to_states(ufs: list[str]) -> list[str]:
    return [STATE_LOOKUP[uf]["state"] for uf in ufs if uf in STATE_LOOKUP]


def extract_operation_name(title: str) -> str | None:
    folded = fold_text(title)
    match = re.search(r"operacao\s+([a-z0-9\s-]{2,60})", folded)
    if not match:
        return None

    stop_tokens = {"em", "no", "na", "nos", "nas", "contra", "para", "com", "do", "da", "de", "e"}
    raw_tokens = match.group(1).split()
    if not raw_tokens or raw_tokens[0] in stop_tokens:
        return None

    tokens = []
    for token in raw_tokens:
        if token in stop_tokens and tokens:
            break
        if token in {"fase", "fases"} and tokens:
            break
        token = token.strip("-")
        if not token:
            continue
        tokens.append(token)
        if len(tokens) >= 3:
            break

    if tokens:
        return " ".join(token.capitalize() for token in tokens)
    return None


def keyword_hits(text: str, patterns: dict[str, list[str]]) -> list[str]:
    lowered = fold_text(text)
    hits = []
    for label, keywords in patterns.items():
        if any(fold_text(keyword) in lowered for keyword in keywords):
            hits.append(label)
    return hits


def ensure_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    return [text] if text else []


def dedupe_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def clean_optional_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def merge_canonical_crimes(
    heuristic_labels: list[str],
    llm_labels: list[str],
    llm_identity: str | float | None,
) -> list[str]:
    merged = dedupe_list([*ensure_list(heuristic_labels), *ensure_list(llm_labels)])
    identity_text = clean_optional_text(llm_identity)
    if identity_text.startswith("crime_"):
        merged = dedupe_list([identity_text.removeprefix("crime_"), *merged])
    return merged


def merge_canonical_modus(
    heuristic_labels: list[str],
    llm_labels: list[str],
) -> list[str]:
    return dedupe_list([*ensure_list(heuristic_labels), *ensure_list(llm_labels)])


def normalize_cluster_token(value: str) -> str:
    folded = fold_text(value).strip()
    if not folded:
        return ""
    return normalize_label(folded).strip("_")


def build_cluster_context(
    titulo: str,
    subtitulo: str,
    texto_sem_geo: str,
    llm_identidade: str | float | None,
    llm_rotulo_resumido: str | float | None,
    llm_crimes_mais_presentes: object,
    llm_tags: object,
    nome_operacao_llm: str | float | None,
) -> str:
    tokens: list[str] = []
    identidade = normalize_cluster_token(clean_optional_text(llm_identidade))
    rotulo = normalize_cluster_token(clean_optional_text(llm_rotulo_resumido))
    operacao = normalize_cluster_token(clean_optional_text(nome_operacao_llm))
    crimes = [normalize_cluster_token(item) for item in ensure_list(llm_crimes_mais_presentes)]
    tags = [normalize_cluster_token(item) for item in ensure_list(llm_tags)]

    if identidade:
        tokens.extend([identidade] * 8)
    if rotulo and rotulo != identidade:
        tokens.extend([rotulo] * 6)
    for crime in crimes[:3]:
        if crime:
            tokens.extend([crime] * 6)
    for tag in tags[:5]:
        if tag:
            tokens.extend([tag] * 3)
    if operacao:
        tokens.extend([operacao] * 2)

    return " ".join(
        part
        for part in [
            (f"{titulo} " * 3).strip(),
            (f"{subtitulo} " * 2).strip(),
            " ".join(tokens).strip(),
            str(texto_sem_geo or "").strip(),
        ]
        if part
    ).strip()


def infer_canonical_cluster(row: pd.Series) -> tuple[str, str, str]:
    llm_identity = clean_optional_text(row.get("llm_identidade"))
    llm_label = clean_optional_text(row.get("llm_rotulo_resumido"))
    llm_type = clean_optional_text(row.get("llm_tipo"))
    operation_name = clean_optional_text(row.get("nome_operacao_llm")) or clean_optional_text(row.get("nome_operacao"))
    crime_labels = ensure_list(row.get("crime_labels"))

    if llm_identity:
        canonical_type = llm_type or ("Por crime" if llm_identity.startswith("crime_") else "Outras")
        return llm_identity, llm_identity, canonical_type

    if llm_label:
        canonical_type = llm_type or ("Por crime" if llm_label.startswith("crime_") else "Outras")
        return llm_label, llm_label, canonical_type

    if crime_labels:
        canonical_label = f"crime_{crime_labels[0]}"
        return canonical_label, canonical_label, "Por crime"

    if operation_name:
        canonical_label = f"operacao_{normalize_label(operation_name)}"
        return canonical_label, canonical_label, "Com operacao nomeada"

    cluster_id = int(row.get("cluster_id", -1))
    canonical_label = f"cluster_exploratorio_{cluster_id}"
    return canonical_label, canonical_label, "Outras"


def build_canonical_cluster_display_label(
    canonical_cluster_id: int,
    canonical_cluster_label: str,
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    size: int,
) -> str:
    first_year = int(first_date.year) if pd.notna(first_date) else 0
    last_year = int(last_date.year) if pd.notna(last_date) else 0
    return f"{canonical_cluster_label} | {first_year}-{last_year} | n={size} | id={canonical_cluster_id}"


def canonical_cluster_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []

    grouped = sorted(df.groupby("cluster_canonico_key"), key=lambda item: len(item[1]), reverse=True)
    for canonical_cluster_id, (cluster_key, subset) in enumerate(grouped, start=1):
        subset = subset.sort_values("data_publicacao_dt")
        canonical_label = str(subset["cluster_canonico_label"].iloc[0])
        canonical_type = str(subset["cluster_canonico_tipo"].iloc[0])
        first_date = subset["data_publicacao_dt"].min()
        last_date = subset["data_publicacao_dt"].max()
        cluster_modes = subset["cluster_id"].mode().tolist()[:3]
        operation_names = [name for name in subset["nome_operacao_llm"].dropna().tolist() if str(name).strip()]
        if not operation_names:
            operation_names = [name for name in subset["nome_operacao"].dropna().tolist() if str(name).strip()]
        crime_labels = Counter(label for labels in subset["crime_labels"] for label in labels)
        llm_tags = Counter(tag for labels in subset["llm_tags"] if isinstance(labels, list) for tag in labels)
        sample_titles = subset["titulo"].dropna().head(3).tolist()
        display_label = build_canonical_cluster_display_label(
            canonical_cluster_id=canonical_cluster_id,
            canonical_cluster_label=canonical_label,
            first_date=first_date,
            last_date=last_date,
            size=len(subset),
        )

        rows.append(
            {
                "cluster_canonico_id": canonical_cluster_id,
                "cluster_canonico_key": cluster_key,
                "cluster_canonico_label": canonical_label,
                "cluster_canonico_tipo": canonical_type,
                "cluster_canonico_display_label": display_label,
                "size": len(subset),
                "first_date": first_date,
                "last_date": last_date,
                "span_days": (last_date - first_date).days if pd.notna(first_date) and pd.notna(last_date) else np.nan,
                "active_years": int(subset["ano"].nunique()),
                "cluster_modes": " | ".join(map(str, cluster_modes)),
                "operation_names": " | ".join(pd.Series(operation_names).value_counts().head(5).index.tolist()),
                "crime_modes": " | ".join([label for label, _ in crime_labels.most_common(5)]),
                "llm_tags": " | ".join([tag for tag, _ in llm_tags.most_common(5)]),
                "sample_titles": " || ".join(sample_titles),
            }
        )

        for _, item in subset.iterrows():
            assignments.append(
                {
                    "link": item["link"],
                    "cluster_canonico_id": canonical_cluster_id,
                    "cluster_canonico_key": cluster_key,
                    "cluster_canonico_label": canonical_label,
                    "cluster_canonico_tipo": canonical_type,
                }
            )

    return pd.DataFrame(assignments), pd.DataFrame(rows)


def load_corpus(index_csv: Path, content_csv: Path, llm_metadata_jsonl: Path) -> pd.DataFrame:
    index_df = pd.read_csv(index_csv)
    content_df = pd.read_csv(content_csv)
    content_df = content_df.copy()
    content_df["status"] = content_df["status"].fillna("")
    content_df["markdown_path"] = content_df["markdown_path"].fillna("")
    content_df = content_df.loc[content_df["status"].eq("ok") & content_df["markdown_path"].ne("")].copy()
    content_df["markdown_exists"] = content_df["markdown_path"].map(lambda value: Path(value).exists())
    missing_markdown = int((~content_df["markdown_exists"]).sum())
    if missing_markdown:
        print(f"[analysis] ignorando {missing_markdown} registros com markdown ausente no manifesto.")
    content_df = content_df.loc[content_df["markdown_exists"]].drop(columns=["markdown_exists"])

    df = index_df.merge(content_df, on="link", how="inner", validate="one_to_one")
    if df.empty:
        raise ValueError("Nenhum artigo com status 'ok' e markdown disponivel foi encontrado para analise.")

    df["arquivo_markdown"] = df["markdown_path"].map(lambda value: Path(str(value)).name)
    llm_df = load_llm_metadata(llm_metadata_jsonl)
    if not llm_df.empty:
        df = df.merge(llm_df, on="arquivo_markdown", how="left")
    else:
        df["llm_identidade"] = np.nan
        df["llm_data"] = np.nan
        df["llm_tipo"] = np.nan
        df["llm_tags"] = [[] for _ in range(len(df))]
        df["llm_rotulo_resumido"] = np.nan
        df["llm_nomes_operacao_encontrados"] = [[] for _ in range(len(df))]
        df["llm_crimes_mais_presentes"] = [[] for _ in range(len(df))]
        df["llm_modus_operandi"] = [[] for _ in range(len(df))]
        df["llm_exemplos_iniciais"] = [[] for _ in range(len(df))]

    df["markdown_text"] = df["markdown_path"].map(read_markdown_text)
    df["texto_limpo"] = df["markdown_text"].map(strip_markdown)
    df["texto_busca"] = (
        df["titulo"].fillna("") + " "
        + df["subtitulo"].fillna("") + " "
        + df["tags"].fillna("").str.replace("|", " ", regex=False) + " "
        + df["texto_limpo"].fillna("")
    ).str.strip()
    df["texto_busca_normalizado"] = df["texto_busca"].map(fold_text)
    df["texto_sem_geo"] = df["texto_busca"].map(semantic_preprocess)
    df["texto_ponderado"] = (
        (df["titulo"].fillna("") + " ") * 3
        + (df["subtitulo"].fillna("") + " ") * 2
        + df["texto_sem_geo"].fillna("")
    ).str.strip()
    df["data_publicacao_dt"] = df["publicado_em_extraido"].map(parse_datetime)
    df["ano"] = df["data_publicacao_dt"].dt.year
    df["mes"] = df["data_publicacao_dt"].dt.month
    df["ano_mes"] = df["data_publicacao_dt"].dt.to_period("M").astype(str)
    df["nome_operacao"] = df["titulo"].fillna("").map(extract_operation_name)
    df["nome_operacao_llm"] = df["llm_nomes_operacao_encontrados"].map(
        lambda items: items[0] if isinstance(items, list) and items else None
    )
    df["llm_tags"] = df["llm_tags"].map(ensure_list)
    df["llm_crimes_mais_presentes"] = df["llm_crimes_mais_presentes"].map(ensure_list)
    df["llm_modus_operandi"] = df["llm_modus_operandi"].map(ensure_list)
    df["llm_nomes_operacao_encontrados"] = df["llm_nomes_operacao_encontrados"].map(ensure_list)
    df["llm_exemplos_iniciais"] = df["llm_exemplos_iniciais"].map(ensure_list)
    df["crime_labels_heuristic"] = df["texto_busca"].map(lambda x: keyword_hits(x, CRIME_PATTERNS))
    df["crime_labels"] = df.apply(
        lambda row: merge_canonical_crimes(
            heuristic_labels=row["crime_labels_heuristic"],
            llm_labels=row["llm_crimes_mais_presentes"],
            llm_identity=row["llm_identidade"],
        ),
        axis=1,
    )
    df["modus_labels_heuristic"] = df["texto_busca"].map(lambda x: keyword_hits(x, MODUS_PATTERNS))
    df["modus_labels"] = df.apply(
        lambda row: merge_canonical_modus(
            heuristic_labels=row["modus_labels_heuristic"],
            llm_labels=row["llm_modus_operandi"],
        ),
        axis=1,
    )
    df["texto_cluster_hibrido"] = df.apply(
        lambda row: build_cluster_context(
            titulo=row["titulo"],
            subtitulo=row["subtitulo"],
            texto_sem_geo=row["texto_sem_geo"],
            llm_identidade=row["llm_identidade"],
            llm_rotulo_resumido=row["llm_rotulo_resumido"],
            llm_crimes_mais_presentes=row["llm_crimes_mais_presentes"],
            llm_tags=row["llm_tags"],
            nome_operacao_llm=row["nome_operacao_llm"],
        ),
        axis=1,
    )
    df["ufs_mencionadas"] = df.apply(
        lambda row: extract_state_mentions(
            title=row["titulo"],
            subtitle=row["subtitulo"],
            tags=row["tags"],
            markdown_text=row["markdown_text"],
        ),
        axis=1,
    )
    df["estados_mencionados"] = df["ufs_mencionadas"].map(ufs_to_states)
    return df


def build_semantic_space(texts: pd.Series, random_state: int) -> tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=False,
        preprocessor=semantic_preprocess,
        stop_words=sorted(PORTUGUESE_STOPWORDS),
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.80,
        max_features=40000,
        token_pattern=r"(?u)\b[a-z0-9_-]{2,}\b",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    n_components = min(160, max(2, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(tfidf_matrix)
    reduced = Normalizer(copy=False).fit_transform(reduced)
    return vectorizer, tfidf_matrix, reduced


def choose_cluster_count(embeddings: np.ndarray, random_state: int) -> int:
    candidate_ks = [12, 16, 20, 24, 28, 32]
    sample_size = min(2000, len(embeddings))
    if sample_size < 200:
        return 12

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(embeddings), size=sample_size, replace=False)
    sample = embeddings[sample_idx]
    best_k = candidate_ks[0]
    best_score = -1.0

    for k in candidate_ks:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=1024,
            n_init=10,
        )
        labels = model.fit_predict(sample)
        score = silhouette_score(sample, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def assign_clusters(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    n_clusters = choose_cluster_count(embeddings, random_state=random_state)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=20,
    )
    return model.fit_predict(embeddings)


def top_terms_by_cluster(
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    labels: np.ndarray,
    top_n: int = 10,
) -> dict[int, list[str]]:
    terms = np.array(vectorizer.get_feature_names_out())
    results: dict[int, list[str]] = {}
    geo_terms = {normalize_label(fold_text(item["state"])) for item in BRAZIL_STATES}
    geo_terms |= {normalize_label(fold_text(item["capital"])) for item in BRAZIL_STATES}
    geo_terms |= {item["uf"].lower() for item in BRAZIL_STATES}

    for cluster_id in sorted(set(labels)):
        idx = np.where(labels == cluster_id)[0]
        centroid = np.asarray(tfidf_matrix[idx].mean(axis=0)).ravel()
        ordered = centroid.argsort()[::-1]
        chosen: list[str] = []
        for term_index in ordered:
            term = str(terms[term_index])
            term_key = normalize_label(term)
            if term_key in geo_terms:
                continue
            if any(token in geo_terms for token in term.split()):
                continue
            chosen.append(term)
            if len(chosen) >= top_n:
                break
        results[int(cluster_id)] = chosen
    return results


def nearest_neighbors(df: pd.DataFrame, embeddings: np.ndarray, n_neighbors: int) -> pd.DataFrame:
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    rows: list[dict[str, object]] = []
    for source_idx in range(len(df)):
        for rank in range(1, n_neighbors):
            target_idx = int(indices[source_idx, rank])
            similarity = float(1 - distances[source_idx, rank])
            source_date = df.iloc[source_idx]["data_publicacao_dt"]
            target_date = df.iloc[target_idx]["data_publicacao_dt"]
            rows.append(
                {
                    "source_link": df.iloc[source_idx]["link"],
                    "source_titulo": df.iloc[source_idx]["titulo"],
                    "source_data": source_date,
                    "target_link": df.iloc[target_idx]["link"],
                    "target_titulo": df.iloc[target_idx]["titulo"],
                    "target_data": target_date,
                    "rank": rank,
                    "cosine_similarity": round(similarity, 6),
                    "source_cluster_id": int(df.iloc[source_idx]["cluster_id"]),
                    "target_cluster_id": int(df.iloc[target_idx]["cluster_id"]),
                    "source_operation_name": df.iloc[source_idx]["nome_operacao"],
                    "target_operation_name": df.iloc[target_idx]["nome_operacao"],
                    "same_cluster": bool(df.iloc[source_idx]["cluster_id"] == df.iloc[target_idx]["cluster_id"]),
                    "gap_days": abs((target_date - source_date).days) if pd.notna(source_date) and pd.notna(target_date) else np.nan,
                }
            )
    return pd.DataFrame(rows)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def semantic_series(df: pd.DataFrame, recurring_pairs_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    link_to_idx = {link: idx for idx, link in enumerate(df["link"].tolist())}
    uf = UnionFind(len(df))

    for _, row in recurring_pairs_df.iterrows():
        uf.union(link_to_idx[row["source_link"]], link_to_idx[row["target_link"]])

    component_map: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(df)):
        component_map[uf.find(idx)].append(idx)

    assignments = []
    summaries = []
    component_id = 0
    for members in sorted(component_map.values(), key=len, reverse=True):
        if len(members) < 2:
            continue
        component_id += 1
        subset = df.iloc[members].sort_values("data_publicacao_dt")
        title_examples = subset["titulo"].head(3).tolist()
        operation_names = [name for name in subset["nome_operacao_llm"].dropna().tolist() if name]
        if not operation_names:
            operation_names = [name for name in subset["nome_operacao"].dropna().tolist() if name]
        crime_labels = Counter(label for labels in subset["crime_labels"] for label in labels)
        llm_identities = subset["llm_identidade"].fillna("").tolist() if "llm_identidade" in subset else []
        llm_labels = subset["llm_rotulo_resumido"].fillna("").tolist() if "llm_rotulo_resumido" in subset else []
        cluster_modes = subset["cluster_id"].mode().tolist()[:3]
        series_label = build_series_label(
            operation_names,
            crime_labels,
            title_examples,
            cluster_modes,
            llm_identities=llm_identities,
            llm_labels=llm_labels,
        )
        series_group_key = build_series_group_key(
            operation_names,
            crime_labels,
            series_label,
            llm_identities=llm_identities,
        )
        llm_identity = most_common_non_empty(llm_identities)
        llm_type = most_common_non_empty(subset["llm_tipo"].fillna("").tolist()) if "llm_tipo" in subset else None
        llm_tags = Counter(tag for labels in subset["llm_tags"] if isinstance(labels, list) for tag in labels)

        for member_idx in members:
            assignments.append(
                {
                    "link": df.iloc[member_idx]["link"],
                    "semantic_series_id": component_id,
                    "titulo": df.iloc[member_idx]["titulo"],
                    "data_publicacao_dt": df.iloc[member_idx]["data_publicacao_dt"],
                }
            )

        first_date = subset["data_publicacao_dt"].min()
        last_date = subset["data_publicacao_dt"].max()
        series_display_label = build_series_display_label(
            semantic_series_id=component_id,
            series_label=series_label,
            first_date=first_date,
            last_date=last_date,
            cluster_modes=cluster_modes,
            size=len(members),
        )
        summaries.append(
            {
                "semantic_series_id": component_id,
                "size": len(members),
                "first_date": first_date,
                "last_date": last_date,
                "span_days": (last_date - first_date).days if pd.notna(first_date) and pd.notna(last_date) else np.nan,
                "active_years": int(subset["ano"].nunique()),
                "cluster_modes": " | ".join(map(str, cluster_modes)),
                "operation_names": " | ".join(pd.Series(operation_names).value_counts().head(5).index.tolist()),
                "series_group_key": series_group_key,
                "series_label": series_label,
                "series_display_label": series_display_label,
                "crime_modes": " | ".join([label for label, _ in crime_labels.most_common(5)]),
                "llm_identity": llm_identity,
                "llm_type": llm_type,
                "llm_tags": " | ".join([tag for tag, _ in llm_tags.most_common(5)]),
                "sample_titles": " || ".join(title_examples),
            }
        )

    return pd.DataFrame(assignments), pd.DataFrame(summaries)


def consolidate_semantic_series(
    df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments_df.empty or summaries_df.empty:
        return assignments_df, summaries_df

    series_meta = summaries_df.copy()

    members = assignments_df[["link", "semantic_series_id"]].merge(
        series_meta[["semantic_series_id", "series_label", "series_group_key"]],
        on="semantic_series_id",
        how="left",
    ).merge(
        df[
            [
                "link",
                "titulo",
                "data_publicacao_dt",
                "ano",
                "cluster_id",
                "nome_operacao",
                "nome_operacao_llm",
                "crime_labels",
                "llm_identidade",
                "llm_rotulo_resumido",
                "llm_tipo",
                "llm_tags",
                "llm_crimes_mais_presentes",
            ]
        ],
        on="link",
        how="left",
    )

    groups = []
    for _, subset in members.groupby(["series_group_key"], dropna=False):
        if subset.empty:
            continue

        subset = subset.sort_values("data_publicacao_dt")
        title_examples = subset["titulo"].dropna().head(3).tolist()
        operation_names = [name for name in subset["nome_operacao_llm"].dropna().tolist() if name]
        if not operation_names:
            operation_names = [name for name in subset["nome_operacao"].dropna().tolist() if name]
        crime_labels = Counter(label for labels in subset["crime_labels"] for label in labels)
        llm_identities = subset["llm_identidade"].fillna("").tolist()
        llm_labels = subset["llm_rotulo_resumido"].fillna("").tolist()
        cluster_modes = subset["cluster_id"].mode().tolist()[:3]
        first_date = subset["data_publicacao_dt"].min()
        last_date = subset["data_publicacao_dt"].max()
        size = int(subset["link"].nunique())
        series_label = build_series_label(
            operation_names,
            crime_labels,
            title_examples,
            cluster_modes,
            llm_identities=llm_identities,
            llm_labels=llm_labels,
        )
        series_group_key = str(subset["series_group_key"].iloc[0])
        llm_identity = most_common_non_empty(llm_identities)
        llm_type = most_common_non_empty(subset["llm_tipo"].fillna("").tolist())
        llm_tags = Counter(tag for labels in subset["llm_tags"] if isinstance(labels, list) for tag in labels)
        groups.append(
            {
                "series_group_key": series_group_key,
                "series_label": series_label,
                "first_date": first_date,
                "last_date": last_date,
                "size": size,
                "active_years": int(subset["ano"].nunique()),
                "cluster_modes": " | ".join(map(str, cluster_modes)),
                "operation_names": " | ".join(pd.Series(operation_names).value_counts().head(5).index.tolist()),
                "crime_modes": " | ".join([label for label, _ in crime_labels.most_common(5)]),
                "llm_identity": llm_identity,
                "llm_type": llm_type,
                "llm_tags": " | ".join([tag for tag, _ in llm_tags.most_common(5)]),
                "sample_titles": " || ".join(title_examples),
                "links": subset["link"].drop_duplicates().tolist(),
            }
        )

    groups_df = pd.DataFrame(groups).sort_values(
        ["size", "first_date", "series_label"], ascending=[False, True, True]
    ).reset_index(drop=True)

    consolidated_summaries = []
    consolidated_assignments = []
    for new_id, row in enumerate(groups_df.itertuples(index=False), start=1):
        cluster_modes = [int(part) for part in str(row.cluster_modes).split(" | ") if str(part).strip()]
        series_display_label = build_series_display_label(
            semantic_series_id=new_id,
            series_label=row.series_label,
            first_date=row.first_date,
            last_date=row.last_date,
            cluster_modes=cluster_modes,
            size=int(row.size),
        )
        consolidated_summaries.append(
            {
                "semantic_series_id": new_id,
                "size": int(row.size),
                "first_date": row.first_date,
                "last_date": row.last_date,
                "span_days": (row.last_date - row.first_date).days if pd.notna(row.first_date) and pd.notna(row.last_date) else np.nan,
                "active_years": int(row.active_years),
                "cluster_modes": row.cluster_modes,
                "operation_names": row.operation_names,
                "series_group_key": row.series_group_key,
                "series_label": row.series_label,
                "series_display_label": series_display_label,
                "crime_modes": row.crime_modes,
                "llm_identity": row.llm_identity,
                "llm_type": row.llm_type,
                "llm_tags": row.llm_tags,
                "sample_titles": row.sample_titles,
            }
        )
        for link in row.links:
            consolidated_assignments.append(
                {
                    "link": link,
                    "semantic_series_id": new_id,
                }
            )

    return pd.DataFrame(consolidated_assignments), pd.DataFrame(consolidated_summaries)


def build_unique_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = df.copy()
    pairs["pair_key"] = pairs.apply(
        lambda row: "||".join(sorted([str(row["source_link"]), str(row["target_link"])])),
        axis=1,
    )
    pairs = pairs.drop_duplicates(subset=["pair_key"], keep="first").drop(columns=["pair_key"]).reset_index(drop=True)
    return pairs


def most_common_non_empty(values: list[str]) -> str | None:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def build_series_label(
    operation_names: list[str],
    crime_labels: Counter,
    title_examples: list[str],
    cluster_modes: list[int],
    llm_identities: list[str] | None = None,
    llm_labels: list[str] | None = None,
) -> str:
    llm_identity = most_common_non_empty(llm_identities or [])
    if llm_identity:
        return llm_identity
    llm_label = most_common_non_empty(llm_labels or [])
    if llm_label:
        return llm_label
    if operation_names:
        return " | ".join(pd.Series(operation_names).value_counts().head(3).index.tolist())
    if crime_labels:
        top_crimes = [label for label, _ in crime_labels.most_common(2)]
        return f"serie_por_crime_{' | '.join(top_crimes)}"
    if title_examples:
        return title_examples[0][:90]
    if cluster_modes:
        return f"serie_cluster_{cluster_modes[0]}"
    return "serie_sem_nome"


def build_series_group_key(
    operation_names: list[str],
    crime_labels: Counter,
    series_label: str,
    llm_identities: list[str] | None = None,
) -> str:
    llm_identity = most_common_non_empty(llm_identities or [])
    if llm_identity:
        return f"llm::{normalize_label(llm_identity)}"
    if operation_names:
        top_operations = pd.Series(operation_names).value_counts().head(5).index.tolist()
        return f"operation::{normalize_label(' | '.join(top_operations))}"
    if crime_labels:
        top_crime_modes = [label for label, _ in crime_labels.most_common(5)]
        return f"crime::{normalize_label(' | '.join(top_crime_modes))}"
    return f"label::{normalize_label(series_label)}"


def build_series_display_label(
    semantic_series_id: int,
    series_label: str,
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    cluster_modes: list[int],
    size: int,
) -> str:
    first_year = int(first_date.year) if pd.notna(first_date) else 0
    last_year = int(last_date.year) if pd.notna(last_date) else 0
    cluster_hint = f"c{cluster_modes[0]}" if cluster_modes else "c?"
    return f"{series_label} | {first_year}-{last_year} | {cluster_hint} | n={size} | id={semantic_series_id}"


def cluster_summary(df: pd.DataFrame, top_terms: dict[int, list[str]]) -> pd.DataFrame:
    rows = []
    geo_tag_keys = set(STATE_TAG_TO_UF.keys())
    geo_tag_keys |= {fold_text(item["capital"]) for item in BRAZIL_STATES}
    for cluster_id, subset in df.groupby("cluster_id"):
        tags = Counter()
        crimes = Counter()
        modus = Counter()
        states = Counter()
        for value in subset["tags"].fillna(""):
            for tag in [part.strip() for part in value.split("|") if part.strip()]:
                tag_key = fold_text(tag)
                if tag_key in geo_tag_keys:
                    continue
                tags[tag] += 1
        for labels in subset["crime_labels"]:
            crimes.update(labels)
        for labels in subset["modus_labels"]:
            modus.update(labels)
        for ufs in subset["ufs_mencionadas"]:
            states.update(ufs)

        first_date = subset["data_publicacao_dt"].min()
        last_date = subset["data_publicacao_dt"].max()
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(subset),
                "first_date": first_date,
                "last_date": last_date,
                "active_years": int(subset["ano"].nunique()),
                "top_terms": " | ".join(top_terms.get(int(cluster_id), [])),
                "top_tags": " | ".join([tag for tag, _ in tags.most_common(8)]),
                "top_states": " | ".join([STATE_LOOKUP[uf]["state"] for uf, _ in states.most_common(6)]),
                "top_crimes": " | ".join([tag for tag, _ in crimes.most_common(6)]),
                "top_modus": " | ".join([tag for tag, _ in modus.most_common(6)]),
                "sample_titles": " || ".join(subset.sort_values("data_publicacao_dt")["titulo"].head(3).tolist()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    summary["cluster_label"] = summary.apply(
        lambda row: f"cluster_{row['cluster_id']}_{normalize_label(' '.join(str(row['top_terms']).split(' | ')[:4]))}",
        axis=1,
    )
    return summary


def temporal_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, subset in df.groupby("cluster_id"):
        monthly_counts = subset.groupby("ano_mes").size().sort_index()
        month_of_year = subset.groupby("mes").size()
        sorted_dates = subset["data_publicacao_dt"].dropna().sort_values()
        gaps = sorted_dates.diff().dt.days.dropna()
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "noticias": len(subset),
                "active_years": int(subset["ano"].nunique()),
                "active_months": int(monthly_counts.shape[0]),
                "peak_year_month": monthly_counts.idxmax() if not monthly_counts.empty else None,
                "peak_year_month_count": int(monthly_counts.max()) if not monthly_counts.empty else 0,
                "peak_calendar_month": int(month_of_year.idxmax()) if not month_of_year.empty else np.nan,
                "peak_calendar_month_count": int(month_of_year.max()) if not month_of_year.empty else 0,
                "peak_calendar_month_share": round(float(month_of_year.max() / len(subset)), 4) if not month_of_year.empty else np.nan,
                "median_gap_days": round(float(gaps.median()), 2) if not gaps.empty else np.nan,
                "mean_gap_days": round(float(gaps.mean()), 2) if not gaps.empty else np.nan,
                "repeats_across_years": bool(subset["ano"].nunique() >= 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["repeats_across_years", "noticias"], ascending=[False, False]).reset_index(drop=True)


def canonical_temporal_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for canonical_cluster_id, subset in df.groupby("cluster_canonico_id"):
        subset = subset.sort_values("data_publicacao_dt")
        monthly_counts = subset.groupby("ano_mes").size().sort_index()
        month_of_year = subset.groupby("mes").size()
        sorted_dates = subset["data_publicacao_dt"].dropna().sort_values()
        gaps = sorted_dates.diff().dt.days.dropna()
        rows.append(
            {
                "cluster_canonico_id": int(canonical_cluster_id),
                "cluster_canonico_label": str(subset["cluster_canonico_label"].iloc[0]),
                "cluster_canonico_tipo": str(subset["cluster_canonico_tipo"].iloc[0]),
                "noticias": len(subset),
                "active_years": int(subset["ano"].nunique()),
                "active_months": int(monthly_counts.shape[0]),
                "peak_year_month": monthly_counts.idxmax() if not monthly_counts.empty else None,
                "peak_year_month_count": int(monthly_counts.max()) if not monthly_counts.empty else 0,
                "peak_calendar_month": int(month_of_year.idxmax()) if not month_of_year.empty else np.nan,
                "peak_calendar_month_count": int(month_of_year.max()) if not month_of_year.empty else 0,
                "peak_calendar_month_share": round(float(month_of_year.max() / len(subset)), 4) if not month_of_year.empty else np.nan,
                "median_gap_days": round(float(gaps.median()), 2) if not gaps.empty else np.nan,
                "mean_gap_days": round(float(gaps.mean()), 2) if not gaps.empty else np.nan,
                "repeats_across_years": bool(subset["ano"].nunique() >= 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["repeats_across_years", "noticias"], ascending=[False, False]).reset_index(drop=True)


def summarize_labels(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        for label in row[column]:
            records.append({"ano": row["ano"], f"{prefix}_label": label})
    summary = pd.DataFrame(records)
    if summary.empty:
        return pd.DataFrame(columns=["ano", f"{prefix}_label", "noticias"])
    return (
        summary.groupby(["ano", f"{prefix}_label"])
        .size()
        .reset_index(name="noticias")
        .sort_values(["ano", "noticias"], ascending=[True, False])
        .reset_index(drop=True)
    )


def summarize_states(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exploded = (
        df[["ano", "cluster_id", "ufs_mencionadas"]]
        .explode("ufs_mencionadas")
        .dropna(subset=["ufs_mencionadas"])
        .rename(columns={"ufs_mencionadas": "uf"})
    )
    if exploded.empty:
        empty = pd.DataFrame(columns=["ano", "uf", "state", "lat", "lon", "noticias"])
        return empty, empty

    exploded["state"] = exploded["uf"].map(lambda uf: STATE_LOOKUP[uf]["state"])
    exploded["lat"] = exploded["uf"].map(lambda uf: STATE_LOOKUP[uf]["lat"])
    exploded["lon"] = exploded["uf"].map(lambda uf: STATE_LOOKUP[uf]["lon"])

    by_year = (
        exploded.groupby(["ano", "uf", "state", "lat", "lon"])
        .size()
        .reset_index(name="noticias")
        .sort_values(["ano", "noticias"], ascending=[True, False])
        .reset_index(drop=True)
    )
    by_cluster = (
        exploded.groupby(["cluster_id", "uf", "state", "lat", "lon"])
        .size()
        .reset_index(name="noticias")
        .sort_values(["cluster_id", "noticias"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return by_year, by_cluster


def top_examples_by_label(df: pd.DataFrame, label_column: str, label_name: str, top_n: int = 3) -> list[str]:
    subset = df[df[label_column].map(lambda labels: label_name in labels)]
    if subset.empty:
        return []
    return subset.sort_values("data_publicacao_dt")["titulo"].head(top_n).tolist()


def write_report(
    output_file: Path,
    df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    canonical_cluster_df: pd.DataFrame,
    canonical_temporal_df: pd.DataFrame,
    crime_df: pd.DataFrame,
    modus_df: pd.DataFrame,
) -> None:
    top_pairs = (
        neighbors_df.loc[neighbors_df["cosine_similarity"] >= 0.80]
        .sort_values("cosine_similarity", ascending=False)
        .pipe(build_unique_pairs)
        .head(10)
    )
    recurring_clusters = temporal_df[temporal_df["repeats_across_years"]].head(10)
    recurring_canonical_clusters = canonical_temporal_df[canonical_temporal_df["repeats_across_years"]].head(10)
    biggest_clusters = cluster_df.head(10)
    biggest_canonical_clusters = canonical_cluster_df.head(10)
    top_crimes = crime_df.groupby("crime_label")["noticias"].sum().sort_values(ascending=False).head(10)
    top_modus = modus_df.groupby("modus_label")["noticias"].sum().sort_values(ascending=False).head(10)

    lines = [
        "# Analise Qualitativa das Noticias de Operacoes da PF",
        "",
        f"- Base analisada: {len(df)} noticias",
        f"- Periodo coberto: {df['data_publicacao_dt'].min().date()} a {df['data_publicacao_dt'].max().date()}",
        f"- Clusters semanticos: {df['cluster_id'].nunique()}",
        f"- Clusters canonicos: {len(canonical_cluster_df)}",
        "",
        "## O que este pipeline responde",
        "",
        "- Quais noticias sao semanticamente parecidas entre si no espaço exploratorio",
        "- Quais identidades canonicas se repetem ao longo do tempo",
        "- Quais crimes e modos de atuacao aparecem com mais frequencia",
        "- Quais clusters canonicos concentram recorrencia temporal mais clara",
        "- Onde as noticias se concentram por estado, sem misturar geografia com o vocabulario dos clusters",
        "",
        "## Maiores clusters canonicos",
        "",
    ]

    for _, row in biggest_canonical_clusters.iterrows():
        lines.append(
            f"- {row['cluster_canonico_label']}: {int(row['size'])} noticias | tipo: {row['cluster_canonico_tipo']} | crimes: {row['crime_modes']} | operacoes: {row['operation_names'] or 'sem_operacao'}"
        )

    lines.extend([
        "",
        "## Maiores clusters",
        "",
    ])

    for _, row in biggest_clusters.iterrows():
        lines.append(
            f"- Cluster {int(row['cluster_id'])}: {int(row['size'])} noticias | termos: {row['top_terms']} | crimes: {row['top_crimes']} | estados: {row['top_states']}"
        )

    lines.extend(["", "## Pares com alta similaridade", ""])
    for _, row in top_pairs.iterrows():
        lines.append(
            f"- {row['source_titulo']} <-> {row['target_titulo']} | similaridade={row['cosine_similarity']:.3f} | gap_dias={row['gap_days']}"
        )

    lines.extend(["", "## Recorrencia temporal por cluster", ""])
    for _, row in recurring_clusters.iterrows():
        lines.append(
            f"- Cluster {int(row['cluster_id'])}: {int(row['noticias'])} noticias em {int(row['active_years'])} anos | pico={row['peak_year_month']} | mes_calendario_pico={int(row['peak_calendar_month']) if pd.notna(row['peak_calendar_month']) else 'NA'}"
        )

    lines.extend(["", "## Recorrencia temporal por cluster canonico", ""])
    for _, row in recurring_canonical_clusters.iterrows():
        lines.append(
            f"- {row['cluster_canonico_label']}: {int(row['noticias'])} noticias em {int(row['active_years'])} anos | pico={row['peak_year_month']} | tipo={row['cluster_canonico_tipo']}"
        )

    lines.extend(["", "## Crimes mais presentes", ""])
    for label, count in top_crimes.items():
        examples = top_examples_by_label(df, "crime_labels", label)
        lines.append(f"- {label}: {int(count)} noticias | exemplos: {' || '.join(examples)}")

    lines.extend(["", "## Modus operandi mais presentes", ""])
    for label, count in top_modus.items():
        examples = top_examples_by_label(df, "modus_labels", label)
        lines.append(f"- {label}: {int(count)} noticias | exemplos: {' || '.join(examples)}")

    lines.extend([
        "",
        "## Sugestoes de exploracao adicional",
        "",
        "- Comparar a evolucao anual dos clusters canonicos de corrupcao, lavagem de dinheiro e crime ambiental",
        "- Medir quando uma identidade canonica se espalha por varios clusters exploratorios para detectar variacao narrativa",
        "- Isolar noticias com alta similaridade e grande distancia temporal para detectar repeticao de modus operandi",
        "- Criar uma rede de coocorrencia entre crimes canonicos e tags para identificar combinacoes frequentes",
        "- Separar operacoes com nome proprio recorrente para ver continuidade institucional da atuacao",
    ])

    output_file.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(config: AnalysisConfig) -> AnalysisArtifacts:
    output_dir = config.output_dir
    ensure_dir(output_dir)

    df = load_corpus(config.index_csv, config.content_csv, llm_metadata_jsonl=config.llm_metadata_jsonl)
    vectorizer, tfidf_matrix, embeddings = build_semantic_space(df["texto_cluster_hibrido"], random_state=config.random_state)
    df["cluster_id"] = assign_clusters(embeddings, random_state=config.random_state)

    top_terms = top_terms_by_cluster(tfidf_matrix, vectorizer, df["cluster_id"].to_numpy())
    cluster_df = cluster_summary(df, top_terms=top_terms)
    label_map = dict(zip(cluster_df["cluster_id"], cluster_df["cluster_label"]))
    df["cluster_label"] = df["cluster_id"].map(label_map)
    canonical_cluster_values = df.apply(infer_canonical_cluster, axis=1, result_type="expand")
    canonical_cluster_values.columns = ["cluster_canonico_key", "cluster_canonico_label", "cluster_canonico_tipo"]
    df[["cluster_canonico_key", "cluster_canonico_label", "cluster_canonico_tipo"]] = canonical_cluster_values
    canonical_assignments, canonical_cluster_df = canonical_cluster_summary(df)
    df = df.merge(
        canonical_assignments[
            ["link", "cluster_canonico_id", "cluster_canonico_key", "cluster_canonico_label", "cluster_canonico_tipo"]
        ],
        on=["link", "cluster_canonico_key", "cluster_canonico_label", "cluster_canonico_tipo"],
        how="left",
    )
    canonical_label_map = dict(
        zip(canonical_cluster_df["cluster_canonico_id"], canonical_cluster_df["cluster_canonico_display_label"])
    )
    df["cluster_canonico_display_label"] = df["cluster_canonico_id"].map(canonical_label_map)

    neighbors_df = nearest_neighbors(df, embeddings, n_neighbors=config.neighbors)
    recurring_pairs = neighbors_df[
        (neighbors_df["gap_days"].fillna(0) >= 30)
        & (neighbors_df["same_cluster"])
        & (
            (neighbors_df["cosine_similarity"] >= max(config.series_threshold, 0.90))
            | (
                neighbors_df["source_operation_name"].fillna("").ne("")
                & (neighbors_df["source_operation_name"] == neighbors_df["target_operation_name"])
                & (neighbors_df["cosine_similarity"] >= config.series_threshold)
            )
        )
    ].sort_values(["cosine_similarity", "gap_days"], ascending=[False, False]).reset_index(drop=True)
    recurring_pairs = build_unique_pairs(recurring_pairs)

    series_assignments, series_summary = semantic_series(df, recurring_pairs)
    series_assignments, series_summary = consolidate_semantic_series(df, series_assignments, series_summary)
    if not series_assignments.empty:
        df = df.merge(series_assignments[["link", "semantic_series_id"]], on="link", how="left")
    else:
        df["semantic_series_id"] = np.nan

    temporal_df = temporal_recurrence(df)
    canonical_temporal_df = canonical_temporal_recurrence(df)
    crime_df = summarize_labels(df, column="crime_labels", prefix="crime")
    modus_df = summarize_labels(df, column="modus_labels", prefix="modus")
    states_year_df, states_cluster_df = summarize_states(df)

    corpus_output = output_dir / "corpus_enriquecido.csv"
    neighbors_output = output_dir / "vizinhos_semelhantes.csv"
    recurring_pairs_output = output_dir / "pares_recorrentes.csv"
    cluster_output = output_dir / "resumo_clusters.csv"
    temporal_output = output_dir / "recorrencia_temporal.csv"
    crime_output = output_dir / "crimes_por_ano.csv"
    modus_output = output_dir / "modus_operandi_por_ano.csv"
    series_output = output_dir / "series_semanticas.csv"
    cluster_year_output = output_dir / "clusters_por_ano.csv"
    canonical_cluster_output = output_dir / "clusters_canonicos.csv"
    canonical_cluster_year_output = output_dir / "clusters_canonicos_por_ano.csv"
    canonical_temporal_output = output_dir / "recorrencia_temporal_clusters_canonicos.csv"
    states_year_output = output_dir / "estados_por_ano.csv"
    states_cluster_output = output_dir / "estados_por_cluster.csv"
    report_output = output_dir / "analise_qualitativa.md"

    df[
        [
            "link", "titulo", "subtitulo", "data_publicacao_dt", "ano", "mes", "ano_mes", "tags",
            "nome_operacao", "cluster_id", "cluster_label", "semantic_series_id", "crime_labels", "modus_labels",
            "ufs_mencionadas", "estados_mencionados", "texto_busca_normalizado", "markdown_path",
            "crime_labels_heuristic", "texto_cluster_hibrido",
            "modus_labels_heuristic",
            "cluster_canonico_id", "cluster_canonico_key", "cluster_canonico_label", "cluster_canonico_tipo",
            "cluster_canonico_display_label",
            "llm_identidade", "llm_tipo", "llm_tags", "llm_rotulo_resumido", "llm_nomes_operacao_encontrados",
            "llm_crimes_mais_presentes", "llm_modus_operandi", "llm_exemplos_iniciais",
        ]
    ].to_csv(corpus_output, index=False, encoding="utf-8-sig")
    neighbors_df.to_csv(neighbors_output, index=False, encoding="utf-8-sig")
    recurring_pairs.to_csv(recurring_pairs_output, index=False, encoding="utf-8-sig")
    cluster_df.to_csv(cluster_output, index=False, encoding="utf-8-sig")
    temporal_df.to_csv(temporal_output, index=False, encoding="utf-8-sig")
    crime_df.to_csv(crime_output, index=False, encoding="utf-8-sig")
    modus_df.to_csv(modus_output, index=False, encoding="utf-8-sig")
    series_summary.to_csv(series_output, index=False, encoding="utf-8-sig")
    canonical_cluster_df.to_csv(canonical_cluster_output, index=False, encoding="utf-8-sig")
    canonical_temporal_df.to_csv(canonical_temporal_output, index=False, encoding="utf-8-sig")
    states_year_df.to_csv(states_year_output, index=False, encoding="utf-8-sig")
    states_cluster_df.to_csv(states_cluster_output, index=False, encoding="utf-8-sig")

    (
        df.groupby(["ano", "cluster_id", "cluster_label"])
        .size()
        .reset_index(name="noticias")
        .sort_values(["ano", "noticias"], ascending=[True, False])
        .to_csv(cluster_year_output, index=False, encoding="utf-8-sig")
    )
    (
        df.groupby(["ano", "cluster_canonico_id", "cluster_canonico_label", "cluster_canonico_tipo"])
        .size()
        .reset_index(name="noticias")
        .sort_values(["ano", "noticias"], ascending=[True, False])
        .to_csv(canonical_cluster_year_output, index=False, encoding="utf-8-sig")
    )

    write_report(
        output_file=report_output,
        df=df,
        neighbors_df=neighbors_df,
        cluster_df=cluster_df,
        temporal_df=temporal_df,
        canonical_cluster_df=canonical_cluster_df,
        canonical_temporal_df=canonical_temporal_df,
        crime_df=crime_df,
        modus_df=modus_df,
    )

    return AnalysisArtifacts(
        corpus=df,
        neighbors=neighbors_df,
        recurring_pairs=recurring_pairs,
        cluster_summary=cluster_df,
        temporal_summary=temporal_df,
        canonical_cluster_summary=canonical_cluster_df,
        crime_summary=crime_df,
        modus_summary=modus_df,
        semantic_series=series_summary,
    )


def main(
    index_csv: Path | str | None = None,
    content_csv: Path | str | None = None,
    output_dir: Path | str | None = None,
    llm_metadata_jsonl: Path | str | None = None,
    neighbors: int | None = None,
    series_threshold: float | None = None,
    random_state: int | None = None,
) -> None:
    config = build_analysis_config(
        index_csv=index_csv,
        content_csv=content_csv,
        output_dir=output_dir,
        llm_metadata_jsonl=llm_metadata_jsonl,
        neighbors=neighbors,
        series_threshold=series_threshold,
        random_state=random_state,
    )
    artifacts = run_analysis(config)
    print(f"[analysis] noticias: {len(artifacts.corpus)}")
    print(f"[analysis] clusters: {artifacts.corpus['cluster_id'].nunique()}")
    print(f"[analysis] clusters_canonicos: {len(artifacts.canonical_cluster_summary)}")
    print(f"[analysis] series_semanticas_legacy: {len(artifacts.semantic_series)}")
    print(f"[analysis] saida: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
