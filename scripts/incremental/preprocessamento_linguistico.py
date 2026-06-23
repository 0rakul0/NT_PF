from __future__ import annotations

import pickle
import re
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import nltk
    from nltk.corpus import mac_morpho, stopwords
    from nltk.tag import DefaultTagger, UnigramTagger
except ImportError:  # pragma: no cover - fallback exercised only without optional dependency
    nltk = None
    mac_morpho = None
    stopwords = None
    DefaultTagger = None
    UnigramTagger = None

from scripts.project_config import ANALYSIS_DIR


TAGGER_CACHE_PATH = ANALYSIS_DIR / "preprocessing_cache" / "mac_morpho_unigram.pkl"
ALLOWED_POS_PREFIXES = ("N", "V", "ADJ", "PCP")
MIN_TOKEN_LENGTH = 3
NEGATION_TOKENS = {"nao", "sem"}

FALLBACK_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "aquela",
    "aquele",
    "aqueles",
    "as",
    "ate",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "ela",
    "ele",
    "eles",
    "em",
    "entre",
    "era",
    "essa",
    "esse",
    "esta",
    "este",
    "foi",
    "foram",
    "ha",
    "isso",
    "ja",
    "mais",
    "mas",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "que",
    "se",
    "ser",
    "sua",
    "suas",
    "seu",
    "seus",
    "tambem",
    "um",
    "uma",
}

GENERIC_OPERATIONAL_TOKENS = {
    "acao",
    "apoio",
    "combater",
    "combate",
    "conjunto",
    "cumprir",
    "cumpre",
    "deflagrar",
    "deflagra",
    "durante",
    "federal",
    "hoje",
    "integrada",
    "investigar",
    "investigacao",
    "mandado",
    "mandados",
    "operacao",
    "policia",
    "realizar",
    "realizou",
    "sexta",
    "suspeito",
    "suspeitos",
}
GENERIC_OPERATIONAL_STEMS = (
    "apoi",
    "combat",
    "conjunt",
    "cumpr",
    "deflagr",
    "inform",
    "integr",
    "investig",
    "mandad",
    "operac",
    "polici",
    "realiz",
    "suspeit",
)
CONTEXT_NOISE_TOKENS = {
    "abril",
    "agosto",
    "comunicacao",
    "dezembro",
    "domingo",
    "fevereiro",
    "sexta",
    "janeiro",
    "julho",
    "junho",
    "maio",
    "manha",
    "marco",
    "novembro",
    "outubro",
    "quarta",
    "quinta",
    "segunda",
    "social",
    "sabado",
    "setembro",
    "tarde",
    "terca",
}

DOMAIN_PHRASES = (
    "abuso sexual infantil",
    "abuso sexual infantojuvenil",
    "associacao criminosa",
    "condicao analoga escravidao",
    "condicoes analogas escravidao",
    "crime ambiental",
    "crime organizado",
    "desmatamento ilegal",
    "desvio recursos publicos",
    "estupro vulneravel",
    "exploracao sexual infantil",
    "extracao ilegal madeira",
    "faccao criminosa",
    "fraude licitacao",
    "garimpo ilegal",
    "lavagem dinheiro",
    "madeira ilegal",
    "mineracao ilegal",
    "moeda falsa",
    "organizacao criminosa",
    "pornografia infantil",
    "radio clandestina",
    "radiodifusao clandestina",
    "trabalho analogo escravidao",
    "trabalho escravo",
    "trafico animais silvestres",
    "trafico drogas",
)

DOMAIN_TOKEN_HINTS = {
    token
    for phrase in DOMAIN_PHRASES
    for token in phrase.split()
}
DOMAIN_TOKEN_HINTS.update(
    {
        "anatel",
        "bpc",
        "ibama",
        "icmbio",
        "inss",
        "interpol",
        "loas",
    }
)


@dataclass(frozen=True)
class LinguisticPreprocessingResult:
    semantic_features: str
    tokens: list[str]
    phrases: list[str]
    original_tokens: int
    kept_tokens: int
    reduction_ratio: float
    backend: str
    elapsed_seconds: float

    def metrics(self) -> dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "kept_tokens": self.kept_tokens,
            "reduction_ratio": round(self.reduction_ratio, 6),
            "backend": self.backend,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_-]{2,}", fold_text(value))


def _download_resource(resource: str) -> bool:
    if nltk is None:
        return False
    try:
        return bool(nltk.download(resource, quiet=True))
    except Exception:
        return False


@lru_cache(maxsize=1)
def portuguese_stopwords() -> set[str]:
    if stopwords is None:
        return set(FALLBACK_STOPWORDS)
    try:
        words = stopwords.words("portuguese")
    except LookupError:
        if not _download_resource("stopwords"):
            return set(FALLBACK_STOPWORDS)
        try:
            words = stopwords.words("portuguese")
        except LookupError:
            return set(FALLBACK_STOPWORDS)
    result = {fold_text(word) for word in words}.union(FALLBACK_STOPWORDS)
    result.discard("nao")
    result.discard("sem")
    return result


def _normalized_training_sentences() -> list[list[tuple[str, str]]]:
    if mac_morpho is None:
        return []
    try:
        tagged = mac_morpho.tagged_sents()
    except LookupError:
        if not _download_resource("mac_morpho"):
            return []
        try:
            tagged = mac_morpho.tagged_sents()
        except LookupError:
            return []
    normalized: list[list[tuple[str, str]]] = []
    for sentence in tagged:
        clean_sentence = [
            (fold_text(word), str(tag))
            for word, tag in sentence
            if tokenize(word)
        ]
        if clean_sentence:
            normalized.append(clean_sentence)
    return normalized


@lru_cache(maxsize=1)
def portuguese_pos_tagger() -> Any | None:
    if UnigramTagger is None or DefaultTagger is None:
        return None
    if TAGGER_CACHE_PATH.exists():
        try:
            with TAGGER_CACHE_PATH.open("rb") as handle:
                return pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError):
            pass
    training = _normalized_training_sentences()
    if not training:
        return None
    tagger = UnigramTagger(training, backoff=DefaultTagger("UNK"))
    TAGGER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TAGGER_CACHE_PATH.open("wb") as handle:
            pickle.dump(tagger, handle)
    except OSError:
        pass
    return tagger


def _domain_phrases(text: str) -> list[str]:
    folded = fold_text(text)
    phrases: list[str] = []
    for phrase in DOMAIN_PHRASES:
        pattern = r"\b" + r"\W+".join(re.escape(token) + r"\w*" for token in phrase.split()) + r"\b"
        if re.search(pattern, folded) and phrase not in phrases:
            phrases.append(phrase.replace(" ", "_"))
    return phrases


def _fallback_content_tokens(tokens: list[str]) -> list[str]:
    stop = portuguese_stopwords()
    return [
        token
        for token in tokens
        if token not in stop
        and not _is_generic_operational_token(token)
        and (len(token) >= 4 or token in DOMAIN_TOKEN_HINTS or token in NEGATION_TOKENS)
    ]


def _is_generic_operational_token(token: str) -> bool:
    return (
        token in GENERIC_OPERATIONAL_TOKENS
        or token in CONTEXT_NOISE_TOKENS
        or any(token.startswith(stem) for stem in GENERIC_OPERATIONAL_STEMS)
    )


def preprocess_body_text(text: str) -> LinguisticPreprocessingResult:
    started = time.perf_counter()
    raw_tokens = tokenize(text)
    stop = portuguese_stopwords()
    candidates = [
        token
        for token in raw_tokens
        if token not in stop
        and not _is_generic_operational_token(token)
        and not token.isdigit()
    ]
    tagger = portuguese_pos_tagger()
    if tagger is None:
        kept = _fallback_content_tokens(candidates)
        backend = "nltk_stopwords_fallback"
    else:
        tagged = tagger.tag(candidates)
        kept = [
            token
            for token, tag in tagged
            if str(tag).upper().startswith(ALLOWED_POS_PREFIXES)
            or token in DOMAIN_TOKEN_HINTS
            or token in NEGATION_TOKENS
        ]
        backend = "nltk_mac_morpho_unigram"

    deduped_sequence: list[str] = []
    for token in kept:
        if token and (not deduped_sequence or token != deduped_sequence[-1]):
            deduped_sequence.append(token)
    phrases = _domain_phrases(text)
    semantic_features = " ".join([*phrases, *deduped_sequence]).strip()
    original_count = len(raw_tokens)
    kept_count = len(deduped_sequence)
    reduction = 1.0 - (kept_count / original_count) if original_count else 0.0
    return LinguisticPreprocessingResult(
        semantic_features=semantic_features,
        tokens=deduped_sequence,
        phrases=phrases,
        original_tokens=original_count,
        kept_tokens=kept_count,
        reduction_ratio=max(0.0, min(1.0, reduction)),
        backend=backend,
        elapsed_seconds=time.perf_counter() - started,
    )
