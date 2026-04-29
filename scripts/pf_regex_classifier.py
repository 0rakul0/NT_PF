from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from pf_regex_rules import (
        BAD_LEARNED_PATTERN_TERMS,
        CRIME_RULE_DEFS,
        LABEL_ALIASES,
        MODUS_RULE_DEFS,
        TERM_STOPWORDS,
    )
    from pf_llm_models import NoticiaLLMInference, normalize_slug
except ModuleNotFoundError:
    from scripts.pf_regex_rules import (
        BAD_LEARNED_PATTERN_TERMS,
        CRIME_RULE_DEFS,
        LABEL_ALIASES,
        MODUS_RULE_DEFS,
        TERM_STOPWORDS,
    )
    from scripts.pf_llm_models import NoticiaLLMInference, normalize_slug


DEFAULT_CONFIDENCE_THRESHOLD = 0.85
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEARNED_RULES_PATH = PROJECT_ROOT / "data" / "analise_qualitativa" / "regex_classifier_rules.json"
LEARNED_RULE_WEIGHT = 1.4


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def canonical_label(value: str) -> str:
    label = normalize_slug(value)
    return LABEL_ALIASES.get(label, label)


@dataclass(frozen=True)
class RegexRule:
    label: str
    patterns: tuple[str, ...]
    weight: float = 1.0
    compiled: tuple[re.Pattern[str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compiled",
            tuple(re.compile(pattern, re.IGNORECASE) for pattern in self.patterns),
        )

    def hits(self, text: str) -> list[str]:
        matches: list[str] = []
        for pattern in self.compiled:
            for match in pattern.finditer(text):
                value = re.sub(r"\s+", " ", match.group(0)).strip()
                if value and value not in matches:
                    matches.append(value)
        return matches


LEARNED_RULES_CACHE: dict[Path, tuple[RegexRule, ...]] = {}
GENERIC_CRIME_LABELS = {"organizacao_criminosa"}
NON_CRIME_LABELS = {
    "atuacao_online",
    "bloqueio_bens",
    "busca_apreensao",
    "busca_e_apreensao",
    "comercializacao",
    "combate_financeiro",
    "cooperacao_interagencias",
    "criacao_empresa_fantasma",
    "cumprimento_mandado",
    "denuncia",
    "desarticulacao_rede",
    "exames_periciais",
    "fiscalizacao",
    "fronteira_transnacional",
    "inclusao_falsos_entrevistadores",
    "prisao",
    "processos_punitivos",
    "resgate_vitimas",
    "suspensao_atividades",
}
IGNORED_TAGS = {
    "destaque",
    "operacao_pf",
    "pf",
    "policia_federal",
}
TAG_RULE_DEFS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    (
        "crimes_ambientais",
        (
            r"\bouros?\b",
            r"\bgarimpos?\b",
            r"\bgarimpos? ilegais?\b",
            r"\bminerios?\b",
            r"\bmineracao\b",
            r"\bmadeira\b",
            r"\bmadeira ilegal\b",
            r"\bcomercio ilegal de madeira\b",
            r"\bexploracao ilegal de madeira\b",
            r"\bmeio ambiente\b",
            r"\bcrimes? ambientais?\b",
            r"\bpesca\b",
            r"\bpesca ilegal\b",
            r"\bgrilagem de terras?\b",
            r"\bagrotoxicos?\b",
        ),
        2.4,
    ),
    (
        "crime_eleitoral",
        (
            r"\bcrime eleitoral\b",
            r"\bcrimes eleitorais\b",
            r"\bfraude eleitoral\b",
            r"\bcorrupcao eleitoral\b",
            r"\beleicoes?\b",
            r"\beleitoral\b",
            r"\bcampanha eleitoral\b",
        ),
        2.2,
    ),
    (
        "fraude_previdenciaria",
        (
            r"\binss\b",
            r"\bprevidencia\b",
            r"\bprevidenciario\b",
            r"\bprevidenciaria\b",
            r"\bfraudes previdenciarias\b",
            r"\bauxilio emergencial\b",
            r"\bbeneficio emergencial\b",
            r"\bseguro-desemprego\b",
            r"\bbolsa familia\b",
        ),
        2.2,
    ),
    (
        "exploracao_pessoas",
        (
            r"\btrabalho escravo\b",
            r"\btrabalho analogo a escravidao\b",
            r"\btrabalho analogo ao de escravo\b",
            r"\btrafico internacional de pessoas\b",
            r"\btrafico de pessoas\b",
            r"\btrafico internacional de orgaos humanos\b",
        ),
        2.2,
    ),
    (
        "crimes_contra_crianca",
        (
            r"\bpornografia infantil\b",
            r"\babuso sexual\b",
            r"\babuso infantojuvenil\b",
            r"\bexploracao sexual infantojuvenil\b",
            r"\bviolencia sexual infantil\b",
            r"\bcrimes contra menores\b",
            r"\bcriancas e adolescentes\b",
        ),
        2.2,
    ),
    (
        "radiodifusao_irregular",
        (
            r"\bradio clandestina\b",
            r"\bradios clandestinas\b",
            r"\bradiodifusao\b",
        ),
        2.0,
    ),
    (
        "desenvolvimento_clandestino_atividade_telecomunicacao",
        (
            r"\btelecomunicacoes?\b",
            r"\batividade clandestina de telecomunicacao\b",
            r"\btelecomunicacao clandestina\b",
        ),
        2.0,
    ),
)
BAD_LEARNED_LABEL_TERMS = {
    "comercializacao_medicamentos_nao_autorizados": {"seguro", "seguros", "seguradora"},
}
SENSITIVE_LABEL_EVIDENCE = {
    "crimes_contra_crianca": (
        r"\babuso sexual\b",
        r"\bexploracao sexual\b",
        r"\bpornografia\b",
        r"\binfantojuvenil\b",
        r"\bpedofilia\b",
        r"\bestupro de vulneravel\b",
    ),
}
SPECIFIC_CRIME_PRIORITY = {
    "crimes_contra_crianca": 91,
    "corrupcao_desvio": 95,
    "fraude_previdenciaria": 92,
    "pornografia_infantil": 91,
    "abuso_sexual_infantil": 90,
    "trafico_drogas": 88,
    "crimes_ambientais": 87,
    "lavagem_dinheiro": 86,
    "contrabando_descaminho": 82,
    "crimes_armas": 81,
    "posse_irregular_arma_fogo": 81,
    "trafico_armas": 80,
    "falsificacoes": 76,
    "falsidade_documental": 76,
    "crimes_ciberneticos_financeiros": 74,
    "fraude_bancaria_cibernetica": 74,
    "crime_eleitoral": 73,
    "crimes_sistema_financeiro": 72,
    "exploracao_pessoas": 70,
    "roubo_furto": 68,
}


@dataclass(frozen=True)
class RegexClassification:
    inference: NoticiaLLMInference | None
    confidence: float
    source: str
    matched_rules: list[dict[str, object]]
    operation_name: str = ""

    @property
    def accepted(self) -> bool:
        return self.inference is not None and self.confidence >= DEFAULT_CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.inference is not None,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "operation_name": self.operation_name,
            "matched_rules": self.matched_rules,
            "inference": self.inference.model_dump() if self.inference else None,
        }


def build_regex_rules(rule_defs: tuple[tuple[str, tuple[str, ...], float], ...]) -> tuple[RegexRule, ...]:
    return tuple(RegexRule(label=label, patterns=patterns, weight=weight) for label, patterns, weight in rule_defs)


CRIME_RULES = build_regex_rules(CRIME_RULE_DEFS)
MODUS_RULES = build_regex_rules(MODUS_RULE_DEFS)
TAG_RULES = build_regex_rules(TAG_RULE_DEFS)


OPERATION_RE = re.compile(
    r"\boperacao\s+(?!contra\b|para\b|em\b|de\b|do\b|da\b|no\b|na\b|com\b)"
    r"([a-z0-9][a-z0-9-]*(?:\s+[a-z0-9][a-z0-9-]*){0,5})\b",
    re.IGNORECASE,
)


def dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = canonical_label(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def prioritize_crimes(crimes: list[str], matches: list[dict[str, object]]) -> list[str]:
    crimes = [crime for crime in crimes if crime not in NON_CRIME_LABELS]
    matched_hits = {
        canonical_label(str(item["label"])): " ".join(str(hit) for hit in item.get("hits", []))
        for item in matches
    }
    match_scores = {canonical_label(str(item["label"])): float(item.get("score", 0.0) or 0.0) for item in matches}
    if "pornografia_infantil" in crimes and re.search(r"\bpornografia\b", matched_hits.get("pornografia_infantil", "")):
        crimes = ["pornografia_infantil", *[crime for crime in crimes if crime != "pornografia_infantil"]]
    specific = [crime for crime in crimes if crime not in GENERIC_CRIME_LABELS]
    if specific and crimes and crimes[0] in GENERIC_CRIME_LABELS:
        top_specific = max(
              specific,
              key=lambda crime: (
                  match_scores.get(crime, 0.0),
                  SPECIFIC_CRIME_PRIORITY.get(crime, 0),
              ),
          )
        crimes = [top_specific, *[crime for crime in crimes if crime != top_specific]]

    crimes = sorted(
        crimes,
          key=lambda crime: (
              0 if crime in GENERIC_CRIME_LABELS and len(crimes) > 1 else 1,
              match_scores.get(crime, 0.0),
              SPECIFIC_CRIME_PRIORITY.get(crime, 0),
          ),
        reverse=True,
    )
    return crimes


def crime_identity(label: str) -> str:
    if label.startswith("crimes_"):
        return label
    return label if label.startswith("crime_") else f"crime_{label}"


def canonical_identity_label(identity: str) -> str:
    normalized = normalize_slug(identity)
    if normalized.startswith("crime_"):
        return canonical_label(normalized.removeprefix("crime_"))
    return canonical_label(normalized)


def has_sensitive_evidence(label: str, evidence_text: str) -> bool:
    patterns = SENSITIVE_LABEL_EVIDENCE.get(canonical_label(label))
    if not patterns:
        return True
    normalized = fold_text(evidence_text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def inference_needs_regex_rescue(inference: NoticiaLLMInference, evidence_text: str) -> bool:
    identity_label = canonical_identity_label(inference.identidade_canonica)
    crimes = [canonical_label(crime) for crime in inference.crimes_mais_presentes]
    if (
        not crimes
        or inference.classificacao == "Outras"
        or identity_label in {"crime_desconhecido", "desconhecido", "contato_institucional"}
    ):
        return True

    return any(not has_sensitive_evidence(crime, evidence_text) for crime in crimes)


def only_crime_matches(matches: list[dict[str, object]]) -> list[dict[str, object]]:
    return [item for item in matches if canonical_label(str(item.get("label", ""))) not in NON_CRIME_LABELS]


def score_rules(text: str, rules: tuple[RegexRule, ...]) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    for rule in rules:
        hits = rule.hits(text)
        if hits:
            scored.append(
                {
                    "label": canonical_label(rule.label),
                    "hits": hits[:5],
                    "hit_count": len(hits),
                    "score": round(rule.weight * len(hits), 3),
                }
            )
    return sorted(scored, key=lambda item: float(item["score"]), reverse=True)


def merge_match_scores(
    body_matches: list[dict[str, object]],
    tag_matches: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for source, matches in (("body", body_matches), ("tag", tag_matches)):
        for item in matches:
            label = canonical_label(str(item["label"]))
            bucket = merged.setdefault(label, {"label": label, "hits": [], "hit_count": 0, "score": 0.0, "sources": []})
            if source not in bucket["sources"]:
                bucket["sources"].append(source)
            bucket["hit_count"] = int(bucket["hit_count"]) + int(item["hit_count"])
            multiplier = 0.55 if source == "tag" else 1.0
            bucket["score"] = float(bucket["score"]) + float(item["score"]) * multiplier
            hits = bucket["hits"]
            if isinstance(hits, list):
                for hit in item["hits"]:
                    tagged_hit = f"tag:{hit}" if source == "tag" else str(hit)
                    if tagged_hit not in hits:
                        hits.append(tagged_hit)
                del hits[5:]

    for item in merged.values():
        item["score"] = round(float(item["score"]), 3)
    return sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)


def normalize_tags(tags: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    for tag in tags or []:
        folded = fold_text(str(tag)).strip()
        slug = normalize_slug(folded)
        if not folded or slug in IGNORED_TAGS:
            continue
        normalized.append(folded)
    return normalized


def learned_rules_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.getenv("PF_REGEX_LEARNED_RULES", "").strip()
    return Path(env_path) if env_path else DEFAULT_LEARNED_RULES_PATH


def load_learned_rules(path: Path | str | None = None, kind: str | None = None) -> tuple[RegexRule, ...]:
    resolved_path = learned_rules_path(path).resolve()
    if resolved_path in LEARNED_RULES_CACHE:
        rules = LEARNED_RULES_CACHE[resolved_path]
    else:
        rules_list: list[RegexRule] = []
        if resolved_path.exists():
            for item in compact_learned_rules(read_learned_rules(resolved_path)):
                label = canonical_label(str(item.get("label", "")))
                pattern = str(item.get("pattern", "")).strip()
                try:
                    weight = float(item.get("weight", LEARNED_RULE_WEIGHT) or LEARNED_RULE_WEIGHT)
                    rules_list.append(RegexRule(label=label, patterns=(pattern,), weight=weight))
                except (re.error, TypeError, ValueError):
                    continue
        rules = tuple(rules_list)
        LEARNED_RULES_CACHE[resolved_path] = rules

    if kind is None:
        return rules

    payload = compact_learned_rules(read_learned_rules(resolved_path)) if resolved_path.exists() else []
    labels_for_kind = {
        canonical_label(str(item.get("label", "")))
        for item in payload
        if isinstance(item, dict) and str(item.get("kind", "")).strip() == kind
    }
    return tuple(rule for rule in rules if rule.label in labels_for_kind)


def clear_learned_rules_cache(path: Path | str | None = None) -> None:
    resolved_path = learned_rules_path(path).resolve()
    LEARNED_RULES_CACHE.pop(resolved_path, None)


def known_crime_labels(path: Path | str | None = None, include_learned: bool = True) -> list[str]:
    labels = {canonical_label(rule.label) for rule in CRIME_RULES}
    if include_learned:
        labels.update(canonical_label(rule.label) for rule in load_learned_rules(path=path, kind="crime"))
    return sorted(label for label in labels if label not in NON_CRIME_LABELS)


def known_modus_labels(path: Path | str | None = None) -> list[str]:
    labels = {canonical_label(rule.label) for rule in MODUS_RULES}
    labels.update(canonical_label(rule.label) for rule in load_learned_rules(path=path, kind="modus"))
    return sorted(labels)


def extract_operation_name(news_body: str) -> str:
    text = fold_text(news_body)
    match = OPERATION_RE.search(text[:1800])
    if not match:
        return ""
    raw_name = re.sub(r"\s+", " ", match.group(1)).strip()
    parts = []
    stopwords = {"para", "contra", "em", "de", "do", "da", "no", "na", "com", "por"}
    for token in raw_name.split():
        if normalize_slug(token) in stopwords:
            break
        parts.append(token)
    name = " ".join(parts).strip()
    first_token = normalize_slug(name.split()[0] if name.split() else "")
    blocked = {"pf", "policia", "federal", "contra", "para", "combate", "repressao", "apurar", "investigar"}
    if normalize_slug(name) in blocked or first_token in blocked:
        return ""
    return name


def estimate_confidence(
    crime_matches: list[dict[str, object]],
    modus_matches: list[dict[str, object]],
    operation_name: str,
    tag_crime_labels: set[str] | None = None,
    tag_modus_labels: set[str] | None = None,
) -> float:
    tag_crime_labels = tag_crime_labels or set()
    tag_modus_labels = tag_modus_labels or set()
    if crime_matches:
        top_score = float(crime_matches[0]["score"])
        total_hits = sum(int(item["hit_count"]) for item in crime_matches)
        confidence = 0.61 + min(0.27, top_score * 0.15) + min(0.12, total_hits * 0.045)
        if modus_matches:
            confidence += 0.05
        if operation_name:
            confidence += 0.04
        if tag_crime_labels:
            confidence += 0.08
        if tag_modus_labels:
            confidence += 0.02
        return min(confidence, 0.95)

    if operation_name:
        confidence = 0.66 + min(0.08, len(modus_matches) * 0.02)
        return min(confidence, 0.78)

    return 0.0


def classify_news_body(
    news_body: str,
    tags: Iterable[str] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    learned_rules_file: Path | str | None = None,
) -> RegexClassification:
    normalized = fold_text(news_body)
    normalized_tags = " ".join(normalize_tags(tags))
    crime_rules = (*CRIME_RULES, *load_learned_rules(path=learned_rules_file, kind="crime"))
    modus_rules = (*MODUS_RULES, *load_learned_rules(path=learned_rules_file, kind="modus"))
    body_crime_matches = score_rules(normalized, crime_rules)
    body_modus_matches = score_rules(normalized, modus_rules)
    tag_crime_matches = score_rules(normalized_tags, (*TAG_RULES, *crime_rules)) if normalized_tags else []
    tag_modus_matches = score_rules(normalized_tags, modus_rules) if normalized_tags else []
    crime_matches = only_crime_matches(merge_match_scores(body_crime_matches, tag_crime_matches))
    modus_matches = merge_match_scores(body_modus_matches, tag_modus_matches)
    tag_crime_labels = {canonical_label(str(item["label"])) for item in tag_crime_matches}
    tag_modus_labels = {canonical_label(str(item["label"])) for item in tag_modus_matches}
    operation_name = extract_operation_name(news_body)
    confidence = estimate_confidence(
        crime_matches,
        modus_matches,
        operation_name,
        tag_crime_labels=tag_crime_labels,
        tag_modus_labels=tag_modus_labels,
    )

    if confidence < confidence_threshold:
        return RegexClassification(
            inference=None,
            confidence=confidence,
            source="regex_below_threshold",
            matched_rules=[*crime_matches[:3], *modus_matches[:3]],
            operation_name=operation_name,
        )

    if crime_matches:
        crimes = prioritize_crimes(dedupe(str(item["label"]) for item in crime_matches[:5]), crime_matches)
        identity = crime_identity(crimes[0])
        classification = "Por crime"
    elif operation_name:
        crimes = []
        identity = f"operacao_{normalize_slug(operation_name)}"
        classification = "Com operacao nomeada"
    else:
        return RegexClassification(
            inference=None,
            confidence=0.0,
            source="regex_no_match",
            matched_rules=[],
        )

    inference = NoticiaLLMInference(
        identidade_canonica=identity,
        classificacao=classification,
        crimes_mais_presentes=crimes,
        modus_operandi=dedupe(str(item["label"]) for item in modus_matches[:6]),
    )
    return RegexClassification(
        inference=inference,
        confidence=confidence,
        source="regex",
        matched_rules=[*crime_matches[:5], *modus_matches[:6]],
        operation_name=operation_name,
    )


def candidate_terms(label: str, text: str, limit: int = 4) -> list[str]:
    label_terms = [term for term in canonical_label(label).split("_") if len(term) >= 4 and term not in TERM_STOPWORDS]
    text_terms = re.findall(r"\b[a-z0-9]{4,}\b", text)
    selected: list[str] = []
    for term in label_terms:
        if re.search(rf"\b{re.escape(term)}\w*\b", text) and term not in selected:
            selected.append(term)
            continue
        if len(term) >= 8:
            stem = term[:8]
            if re.search(rf"\b{re.escape(stem)}\w*\b", text) and stem not in selected:
                selected.append(stem)

    frequencies: dict[str, int] = {}
    for term in text_terms:
        if not term.isdigit() and term not in TERM_STOPWORDS and term not in BAD_LEARNED_PATTERN_TERMS:
            frequencies[term] = frequencies.get(term, 0) + 1
    for term, _ in sorted(frequencies.items(), key=lambda item: (-item[1], text.find(item[0]))):
        if term not in selected:
            selected.append(term)
        if len(selected) >= limit:
            break
    return selected[:limit]


def label_evidence_terms(label: str, text: str) -> list[str]:
    terms = []
    for term in canonical_label(label).split("_"):
        if len(term) < 4 or term in TERM_STOPWORDS or term in BAD_LEARNED_PATTERN_TERMS:
            continue
        if re.search(rf"\b{re.escape(term)}\w*\b", text):
            terms.append(term)
            continue
        if len(term) >= 8:
            stem = term[:8]
            if re.search(rf"\b{re.escape(stem)}\w*\b", text):
                terms.append(stem)
    return dedupe(terms)


def has_enough_label_evidence(label: str, text: str) -> bool:
    label_terms = [
        term
        for term in canonical_label(label).split("_")
        if len(term) >= 4 and term not in TERM_STOPWORDS and term not in BAD_LEARNED_PATTERN_TERMS
    ]
    min_label_evidence = 2 if len(label_terms) >= 2 else 1
    return len(label_evidence_terms(label, text)) >= min_label_evidence


def evidence_window(news_body: str, labels: list[str]) -> str:
    normalized = fold_text(news_body)
    sentences = [part.strip() for part in re.split(r"[.!?;\n]+", normalized) if part.strip()]
    label_terms = [
        term
        for label in labels
        for term in canonical_label(label).split("_")
        if len(term) >= 4 and term not in TERM_STOPWORDS
    ]
    for sentence in sentences:
        if any(re.search(rf"\b{re.escape(term)}\w*\b", sentence) for term in label_terms):
            return sentence[:600]
    return normalized[:600]


def build_learned_pattern(label: str, news_body: str) -> str:
    evidence = evidence_window(news_body, [label])
    if not has_enough_label_evidence(label, evidence):
        return ""
    terms = candidate_terms(label, evidence)
    if len(terms) < 2:
        return ""
    ordered_terms = sorted(terms, key=lambda term: evidence.find(term) if term in evidence else len(evidence))
    return r".{0,240}".join(rf"\b{re.escape(term)}\w*\b" for term in ordered_terms[:4])


def read_learned_rules(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def normalize_flat_learned_rule_record(record: dict[str, object]) -> dict[str, object] | None:
    kind = str(record.get("kind", "")).strip()
    label = canonical_label(str(record.get("label", record.get("classificador", ""))))
    pattern = str(record.get("pattern", record.get("pattern_do_classificador", ""))).strip()
    if kind not in {"crime", "modus"} or not label or not pattern:
        return None
    if kind == "crime" and label in NON_CRIME_LABELS:
        return None
    if kind == "crime":
        static_crime_labels = {canonical_label(rule.label) for rule in CRIME_RULES}
        if label not in static_crime_labels:
            return None
    pattern_terms = re.findall(r"\\b([a-z0-9]{3,})", pattern.lower())
    if kind == "crime" and BAD_LEARNED_LABEL_TERMS.get(label, set()).intersection(pattern_terms):
        return None
    alpha_terms = [
        term
        for term in pattern_terms
        if not term.isdigit() and term not in TERM_STOPWORDS and term not in BAD_LEARNED_PATTERN_TERMS
    ]
    label_terms = [
        term
        for term in label.split("_")
        if len(term) >= 4 and term not in TERM_STOPWORDS and term not in BAD_LEARNED_PATTERN_TERMS
    ]
    label_evidence_count = sum(
        1
        for term in label_terms
        if term in alpha_terms or (len(term) >= 8 and any(pattern_term.startswith(term[:8]) for pattern_term in alpha_terms))
    )
    min_label_evidence = 2 if len(label_terms) >= 2 else 1
    numeric_terms = [term for term in pattern_terms if term.isdigit()]
    if len(alpha_terms) < 2 or numeric_terms or label_evidence_count < min_label_evidence:
        return None
    if any(term in BAD_LEARNED_PATTERN_TERMS for term in pattern_terms):
        return None
    try:
        re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None

    examples_raw = record.get("examples", [])
    examples = examples_raw if isinstance(examples_raw, list) else []
    clean_examples = []
    for example in examples:
        text = str(example).strip()
        if text and text not in clean_examples:
            clean_examples.append(text)

    try:
        uses = int(record.get("uses", 0) or 0)
    except (TypeError, ValueError):
        uses = 0

    try:
        weight = float(record.get("weight", LEARNED_RULE_WEIGHT) or LEARNED_RULE_WEIGHT)
    except (TypeError, ValueError):
        weight = LEARNED_RULE_WEIGHT

    return {
        "kind": kind,
        "label": label,
        "pattern": pattern,
        "weight": weight,
        "source": str(record.get("source", "llm_feedback")).strip() or "llm_feedback",
        "uses": max(uses, 1),
        "examples": clean_examples[:10],
    }


def flatten_learned_rules(records: list[dict[str, object]]) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        patterns = record.get("patterns_do_classificador", record.get("patterns"))
        if isinstance(patterns, list):
            base_examples = record.get("examples", [])
            for item in patterns:
                if isinstance(item, dict):
                    candidate = {
                        "kind": record.get("kind"),
                        "classificador": record.get("classificador", record.get("label")),
                        "pattern_do_classificador": item.get("pattern_do_classificador", item.get("pattern")),
                        "weight": item.get("weight", record.get("weight", LEARNED_RULE_WEIGHT)),
                        "source": item.get("source", record.get("source", "llm_feedback")),
                        "uses": item.get("uses", 1),
                        "examples": item.get("examples", base_examples),
                    }
                else:
                    candidate = {
                        "kind": record.get("kind"),
                        "classificador": record.get("classificador", record.get("label")),
                        "pattern_do_classificador": item,
                        "weight": record.get("weight", LEARNED_RULE_WEIGHT),
                        "source": record.get("source", "llm_feedback"),
                        "uses": record.get("uses", 1),
                        "examples": base_examples,
                    }
                normalized = normalize_flat_learned_rule_record(candidate)
                if normalized is not None:
                    flattened.append(normalized)
            continue

        normalized = normalize_flat_learned_rule_record(record)
        if normalized is not None:
            flattened.append(normalized)
    return flattened


def compact_learned_rules(records: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    for normalized in flatten_learned_rules(records):
        key = (
            str(normalized["kind"]),
            str(normalized["label"]),
            str(normalized["pattern"]),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = normalized
            continue

        existing["uses"] = int(existing.get("uses", 0) or 0) + int(normalized.get("uses", 0) or 0)
        examples = existing.setdefault("examples", [])
        if isinstance(examples, list):
            for example in normalized.get("examples", []):
                if isinstance(example, str) and example and example not in examples:
                    examples.append(example)
            del examples[10:]

    return sorted(
        merged.values(),
        key=lambda item: (str(item["kind"]), str(item["label"]), str(item["pattern"])),
    )


def group_learned_rules(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for record in compact_learned_rules(records):
        key = (str(record["kind"]), str(record["label"]))
        group = grouped.setdefault(
            key,
            {
                "kind": record["kind"],
                "classificador": record["label"],
                "source": "llm_feedback",
                "uses": 0,
                "patterns_do_classificador": [],
                "examples": [],
            },
        )
        group["uses"] = int(group.get("uses", 0) or 0) + int(record.get("uses", 0) or 0)

        group_examples = group.setdefault("examples", [])
        if isinstance(group_examples, list):
            for example in record.get("examples", []):
                if isinstance(example, str) and example and example not in group_examples:
                    group_examples.append(example)
            del group_examples[10:]

        patterns = group.setdefault("patterns_do_classificador", [])
        if isinstance(patterns, list):
            patterns.append(
                {
                    "pattern_do_classificador": record["pattern"],
                    "weight": record.get("weight", LEARNED_RULE_WEIGHT),
                    "uses": record.get("uses", 1),
                    "examples": record.get("examples", []),
                }
            )

    for group in grouped.values():
        patterns = group.get("patterns_do_classificador", [])
        if isinstance(patterns, list):
            patterns.sort(key=lambda item: str(item.get("pattern_do_classificador", "")) if isinstance(item, dict) else str(item))
    return sorted(grouped.values(), key=lambda item: (str(item["kind"]), str(item["classificador"])))


def clean_learned_rules_file(path: Path | str | None = None) -> dict[str, int | str]:
    resolved_path = learned_rules_path(path)
    before_records = read_learned_rules(resolved_path)
    after_records = group_learned_rules(before_records)
    before_flat_count = len(flatten_learned_rules(before_records))
    after_flat_count = len(flatten_learned_rules(after_records))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(after_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    clear_learned_rules_cache(resolved_path)
    return {
        "path": str(resolved_path),
        "before": len(before_records),
        "after": len(after_records),
        "before_patterns": before_flat_count,
        "after_patterns": after_flat_count,
        "removed": len(before_records) - len(after_records),
        "removed_patterns": before_flat_count - after_flat_count,
    }


def append_learned_rule(
    kind: str,
    label: str,
    pattern: str,
    title: str = "",
    path: Path | str | None = None,
) -> bool:
    normalized_label = canonical_label(label)
    if not normalized_label or not pattern:
        return False

    resolved_path = learned_rules_path(path)
    records = compact_learned_rules(read_learned_rules(resolved_path))
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("kind") == kind and record.get("label") == normalized_label and record.get("pattern") == pattern:
            examples = record.setdefault("examples", [])
            if isinstance(examples, list) and title and title not in examples:
                examples.append(title)
            record["uses"] = int(record.get("uses", 0) or 0) + 1
            break
    else:
        records.append(
            {
                "kind": kind,
                "label": normalized_label,
                "pattern": pattern,
                "weight": LEARNED_RULE_WEIGHT,
                "source": "llm_feedback",
                "uses": 1,
                "examples": [title] if title else [],
            }
        )

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(group_learned_rules(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    clear_learned_rules_cache(resolved_path)
    return True


def improve_regex_from_llm(
    news_body: str,
    inference: NoticiaLLMInference,
    title: str = "",
    learned_rules_file: Path | str | None = None,
) -> list[dict[str, str]]:
    learned: list[dict[str, str]] = []
    for label in inference.crimes_mais_presentes:
        pattern = build_learned_pattern(label, news_body)
        if append_learned_rule("crime", label, pattern, title=title, path=learned_rules_file):
            learned.append({"kind": "crime", "label": canonical_label(label), "pattern": pattern})

    for label in inference.modus_operandi:
        pattern = build_learned_pattern(label, news_body)
        if append_learned_rule("modus", label, pattern, title=title, path=learned_rules_file):
            learned.append({"kind": "modus", "label": canonical_label(label), "pattern": pattern})
    return learned


def build_incremental_regex_report(jsonl_path: Path, limit: int = 30) -> dict[str, object]:
    misses: dict[str, dict[str, object]] = {}
    if not jsonl_path.exists():
        return {"source": str(jsonl_path), "labels": []}

    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        regex_meta = record.get("regex_classificacao", {})
        if isinstance(regex_meta, dict) and regex_meta.get("accepted"):
            continue

        inference = record.get("inferencia_llm", {})
        metadata = record.get("metadata_extraido", {})
        if not isinstance(inference, dict) or not isinstance(metadata, dict):
            continue

        labels = inference.get("crimes_mais_presentes", [])
        if not isinstance(labels, list):
            labels = []
        if not labels:
            labels = [inference.get("identidade_canonica", "")]

        title = str(metadata.get("titulo", "")).strip()
        for label in dedupe(str(item) for item in labels):
            bucket = misses.setdefault(label, {"label": label, "count": 0, "examples": []})
            bucket["count"] = int(bucket["count"]) + 1
            examples = bucket["examples"]
            if isinstance(examples, list) and title and len(examples) < 5:
                examples.append(title)

    labels_report = sorted(misses.values(), key=lambda item: int(item["count"]), reverse=True)[:limit]
    return {"source": str(jsonl_path), "labels": labels_report}


def read_input(text: str = "", input_file: Path | str | None = None) -> str:
    if input_file:
        return Path(input_file).read_text(encoding="utf-8")
    if text:
        return text
    env_input_file = os.getenv("PF_REGEX_INPUT_FILE", "").strip()
    if env_input_file:
        return Path(env_input_file).read_text(encoding="utf-8")
    env_text = os.getenv("PF_REGEX_TEXT", "")
    if env_text:
        return env_text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def main(
    text: str = "",
    input_file: Path | str | None = None,
    tags: list[str] | None = None,
    threshold: float | None = None,
    incremental_report: Path | str | None = None,
    limit: int | None = None,
) -> None:
    report_path = incremental_report or os.getenv("PF_REGEX_INCREMENTAL_REPORT", "").strip()
    report_limit = limit
    if report_limit is None:
        report_limit_raw = os.getenv("PF_REGEX_REPORT_LIMIT", "").strip()
        report_limit = int(report_limit_raw) if report_limit_raw.isdigit() else 30
    if report_path:
        print(json.dumps(build_incremental_regex_report(Path(report_path), limit=report_limit), ensure_ascii=False, indent=2))
        return

    confidence_threshold = threshold
    if confidence_threshold is None:
        threshold_raw = os.getenv("PF_REGEX_CONFIDENCE_THRESHOLD", "").strip().replace(",", ".")
        try:
            confidence_threshold = float(threshold_raw) if threshold_raw else DEFAULT_CONFIDENCE_THRESHOLD
        except ValueError:
            confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

    body = read_input(text=text, input_file=input_file)
    env_tags = [part.strip() for part in os.getenv("PF_REGEX_TAGS", "").split(",") if part.strip()]
    result = classify_news_body(body, tags=tags or env_tags, confidence_threshold=confidence_threshold)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
