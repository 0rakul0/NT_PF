from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analise_qualitativa"
NEWS_MARKDOWN_DIR = DATA_DIR / "noticias_markdown"
REFERENCE_DIR = DATA_DIR / "reference"

INDEX_CSV = DATA_DIR / "pf_operacoes_index.csv"
CONTENT_CSV = DATA_DIR / "pf_operacoes_conteudos.csv"
LLM_METADATA_JSONL = ANALYSIS_DIR / "metadados_llm_noticias.jsonl"
LLM_METADATA_CSV = ANALYSIS_DIR / "metadados_llm_noticias.csv"
BRAZIL_STATES_GEOJSON = REFERENCE_DIR / "brazil_states.geojson"


def _strip_env_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned


def load_project_env() -> None:
    for env_name in (".env.local", ".env"):
        env_path = PROJECT_ROOT / env_name
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            env_key = key.strip()
            if not env_key or env_key in os.environ:
                continue
            os.environ[env_key] = _strip_env_quotes(value)


load_project_env()

BASE_LIST_URL = "https://www.gov.br/pf/pt-br/assuntos/noticias/noticias-operacoes"
DEFAULT_SCRAPE_STEP = 60
DEFAULT_HTTP_TIMEOUT_SECONDS = 30
DEFAULT_REQUEST_SLEEP_SECONDS = 0.2

BRAZIL_STATES = [
    {"uf": "AC", "state": "Acre", "capital": "Rio Branco", "lat": -9.97499, "lon": -67.8243},
    {"uf": "AL", "state": "Alagoas", "capital": "Maceio", "lat": -9.64985, "lon": -35.70895},
    {"uf": "AP", "state": "Amapa", "capital": "Macapa", "lat": 0.03493, "lon": -51.0694},
    {"uf": "AM", "state": "Amazonas", "capital": "Manaus", "lat": -3.11903, "lon": -60.0217},
    {"uf": "BA", "state": "Bahia", "capital": "Salvador", "lat": -12.9718, "lon": -38.5011},
    {"uf": "CE", "state": "Ceara", "capital": "Fortaleza", "lat": -3.73186, "lon": -38.5267},
    {"uf": "DF", "state": "Distrito Federal", "capital": "Brasilia", "lat": -15.7797, "lon": -47.9297},
    {"uf": "ES", "state": "Espirito Santo", "capital": "Vitoria", "lat": -20.3155, "lon": -40.3128},
    {"uf": "GO", "state": "Goias", "capital": "Goiania", "lat": -16.6864, "lon": -49.2643},
    {"uf": "MA", "state": "Maranhao", "capital": "Sao Luis", "lat": -2.53874, "lon": -44.2825},
    {"uf": "MT", "state": "Mato Grosso", "capital": "Cuiaba", "lat": -15.6014, "lon": -56.0974},
    {"uf": "MS", "state": "Mato Grosso do Sul", "capital": "Campo Grande", "lat": -20.4697, "lon": -54.6201},
    {"uf": "MG", "state": "Minas Gerais", "capital": "Belo Horizonte", "lat": -19.9167, "lon": -43.9345},
    {"uf": "PA", "state": "Para", "capital": "Belem", "lat": -1.45583, "lon": -48.5044},
    {"uf": "PB", "state": "Paraiba", "capital": "Joao Pessoa", "lat": -7.11532, "lon": -34.861},
    {"uf": "PR", "state": "Parana", "capital": "Curitiba", "lat": -25.4284, "lon": -49.2733},
    {"uf": "PE", "state": "Pernambuco", "capital": "Recife", "lat": -8.05389, "lon": -34.8811},
    {"uf": "PI", "state": "Piaui", "capital": "Teresina", "lat": -5.08917, "lon": -42.8019},
    {"uf": "RJ", "state": "Rio de Janeiro", "capital": "Rio de Janeiro", "lat": -22.9068, "lon": -43.1729},
    {"uf": "RN", "state": "Rio Grande do Norte", "capital": "Natal", "lat": -5.79448, "lon": -35.211},
    {"uf": "RS", "state": "Rio Grande do Sul", "capital": "Porto Alegre", "lat": -30.0346, "lon": -51.2177},
    {"uf": "RO", "state": "Rondonia", "capital": "Porto Velho", "lat": -8.76077, "lon": -63.8999},
    {"uf": "RR", "state": "Roraima", "capital": "Boa Vista", "lat": 2.82384, "lon": -60.6753},
    {"uf": "SC", "state": "Santa Catarina", "capital": "Florianopolis", "lat": -27.5949, "lon": -48.5482},
    {"uf": "SP", "state": "Sao Paulo", "capital": "Sao Paulo", "lat": -23.5505, "lon": -46.6333},
    {"uf": "SE", "state": "Sergipe", "capital": "Aracaju", "lat": -10.9472, "lon": -37.0731},
    {"uf": "TO", "state": "Tocantins", "capital": "Palmas", "lat": -10.184, "lon": -48.3336},
]

STATE_POINTS = {
    item["uf"]: {
        "state": item["state"],
        "capital": item["capital"],
        "lat": item["lat"],
        "lon": item["lon"],
    }
    for item in BRAZIL_STATES
}

STREAMLIT_REQUIRED_DATA_FILES = [
    ANALYSIS_DIR / "corpus_enriquecido.csv",
    ANALYSIS_DIR / "resumo_clusters.csv",
    ANALYSIS_DIR / "recorrencia_temporal.csv",
    ANALYSIS_DIR / "clusters_canonicos.csv",
    ANALYSIS_DIR / "clusters_canonicos_por_ano.csv",
    ANALYSIS_DIR / "recorrencia_temporal_clusters_canonicos.csv",
    ANALYSIS_DIR / "crimes_por_ano.csv",
    ANALYSIS_DIR / "modus_operandi_por_ano.csv",
    ANALYSIS_DIR / "series_semanticas.csv",
    ANALYSIS_DIR / "pares_recorrentes.csv",
    ANALYSIS_DIR / "estados_por_ano.csv",
    ANALYSIS_DIR / "estados_por_cluster.csv",
    ANALYSIS_DIR / "analise_qualitativa.md",
    BRAZIL_STATES_GEOJSON,
]


@dataclass(frozen=True)
class LLMProviderSettings:
    provider: str
    model_name: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class LLMSettings:
    preferred_provider: str
    provider_order: tuple[str, ...]
    ollama: LLMProviderSettings
    groq: LLMProviderSettings
    temperature: float
    max_retries: int


def _default_model_for_provider(provider: str) -> str:
    if provider == "groq":
        return "llama-3.3-70b-versatile"
    return "gemma3n:e2b"


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower() or "auto"


def _normalize_ollama_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3]
    return cleaned or "http://localhost:11434"


def _normalize_groq_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    return cleaned or "https://api.groq.com/openai/v1"


def get_llm_settings() -> LLMSettings:
    provider = "ollama"

    raw_model_name = os.getenv("PF_LLM_MODEL", "").strip()
    raw_base_url = os.getenv("PF_LLM_BASE_URL", "").strip()
    ollama_host = os.getenv("PF_LLM_HOST", "").strip()
    api_key = ""

    groq_model = os.getenv("PF_GROQ_MODEL", "").strip()
    ollama_model = os.getenv("PF_OLLAMA_MODEL", "").strip()
    groq_base_url = os.getenv("PF_GROQ_BASE_URL", "").strip()
    ollama_base_url = os.getenv("PF_OLLAMA_BASE_URL", "").strip()

    if raw_model_name:
        ollama_model = raw_model_name

    if raw_base_url and "groq.com" not in raw_base_url.lower():
        ollama_base_url = raw_base_url

    groq_settings = LLMProviderSettings(
        provider="groq",
        model_name=groq_model or _default_model_for_provider("groq"),
        base_url=_normalize_groq_base_url(groq_base_url),
        api_key=api_key,
    )
    ollama_settings = LLMProviderSettings(
        provider="ollama",
        model_name=ollama_model or _default_model_for_provider("ollama"),
        base_url=_normalize_ollama_base_url(ollama_host or ollama_base_url or raw_base_url),
        api_key="",
    )

    provider_order = ("ollama",)

    temperature_raw = os.getenv("PF_LLM_TEMPERATURE", "0").strip()
    max_retries_raw = os.getenv("PF_LLM_MAX_RETRIES", "3").strip()
    temperature = float(temperature_raw) if temperature_raw else 0.0
    max_retries = int(max_retries_raw) if max_retries_raw.isdigit() else 3

    return LLMSettings(
        preferred_provider=provider,
        provider_order=provider_order,
        ollama=ollama_settings,
        groq=groq_settings,
        temperature=temperature,
        max_retries=max_retries,
    )
