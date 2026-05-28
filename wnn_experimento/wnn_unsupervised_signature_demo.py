"""
Demo nao supervisionada: WNN como memoria de assinaturas.

Este script usa as mesmas 300 noticias e o mesmo banco de discriminadores regex,
mas nao usa labels para treinar. Todas as 210 noticias do treino entram na
memoria como assinaturas binarias. No teste, cada noticia procura os vizinhos
mais parecidos na memoria.

Saida possivel:
    conhecido         -> ha vizinho parecido o suficiente na memoria
    novo              -> nao ha similaridade suficiente
    sem_sinal         -> nenhum discriminador foi ativado

Importante:
    Esta abordagem nao produz classes como "trafico_drogas" diretamente.
    Ela produz grupos de semelhanca/novidade. As sugestoes de tema impressas
    sao inferidas pelos discriminadores mais ativos, nao por labels de treino.
"""

from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR

# Configuracao fixa do experimento nao supervisionado.
CSV_PATH = ROOT / "data_exemplo_noticias" / "manifest.csv"
REGEX_BANK_PATH = SCRIPT_DIR / "wnn_regex_bank_pf.json"
DOCUMENT_LIMIT = 0
STATUS_COLUMN = "status"
STATUS_OK = "ok"
PATH_COLUMN = "markdown_path"
TITLE_COLUMN = "titulo_extraido"
SUBTITLE_COLUMN = "subtitulo_extraido"
TAGS_COLUMN = "tags_extraidas"
TRAIN_RATIO = 0.70
SPLIT_SEED = 20260528
SIMILARITY_MIN = 0.35
NEIGHBORS_TO_SHOW = 3
EXAMPLES_TO_PRINT = 20


@dataclass(frozen=True)
class NewsDoc:
    """Documento usado na memoria de assinaturas."""

    doc_id: int
    title: str
    path: Path
    text: str


@dataclass(frozen=True)
class Discriminator:
    """Feature regex binaria usada para construir assinaturas."""

    name: str
    label: str
    pattern: str


@dataclass(frozen=True)
class MemoryItem:
    """Item memorizado pela WNN nao supervisionada."""

    doc: NewsDoc
    features: frozenset[str]


@dataclass(frozen=True)
class Neighbor:
    """Vizinho recuperado da memoria."""

    doc: NewsDoc
    similarity: float
    shared_features: int


@dataclass(frozen=True)
class UnsupervisedResult:
    """Resultado da busca por semelhanca."""

    status: str
    active_features: frozenset[str]
    suggested_theme: str
    neighbors: list[Neighbor]


def fix_mojibake(text: str) -> str:
    """Corrige casos comuns de texto UTF-8 lido como Latin-1."""
    if "Ã" not in text and "Â" not in text:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def fold_text(text: str) -> str:
    """Normaliza texto para regex: minusculo, ASCII e sem acentos."""
    text = fix_mojibake(text or "")
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii").lower()


def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve caminho absoluto ou relativo."""
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_regex_bank(path: Path) -> dict[str, list[str]]:
    """Carrega JSON no formato {"label": ["regex", ...]}."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("O banco regex deve ser um objeto JSON.")

    bank: dict[str, list[str]] = {}
    for raw_label, raw_patterns in payload.items():
        label = str(raw_label).strip()
        if not label or not isinstance(raw_patterns, list):
            continue
        patterns = [str(item).strip() for item in raw_patterns if str(item).strip()]
        for pattern in patterns:
            re.compile(pattern, flags=re.IGNORECASE)
        if patterns:
            bank[label] = patterns
    return bank


def build_discriminators(regex_bank: dict[str, list[str]]) -> list[Discriminator]:
    """Transforma regex em discriminadores nomeados."""
    discriminators: list[Discriminator] = []
    for label, patterns in regex_bank.items():
        for index, pattern in enumerate(patterns, start=1):
            discriminators.append(
                Discriminator(
                    name=f"{label}::d{index:02d}",
                    label=label,
                    pattern=pattern,
                )
            )
    return discriminators


def load_news(csv_path: Path, base_dir: Path) -> list[NewsDoc]:
    """Le o manifest e monta textos consolidados."""
    docs: list[NewsDoc] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            if STATUS_COLUMN and STATUS_OK and row.get(STATUS_COLUMN) != STATUS_OK:
                continue
            rel_path = row.get(PATH_COLUMN, "")
            if not rel_path:
                continue
            path = resolve_path(base_dir, rel_path)
            if not path.exists():
                continue

            body = path.read_text(encoding="utf-8", errors="replace")
            first_line = body.splitlines()[0].lstrip("# ") if body.splitlines() else ""
            title = row.get(TITLE_COLUMN) or first_line
            subtitle = row.get(SUBTITLE_COLUMN, "")
            tags = row.get(TAGS_COLUMN, "")
            docs.append(
                NewsDoc(
                    doc_id=index,
                    title=fix_mojibake(title),
                    path=path,
                    text="\n".join([title, subtitle, tags, body]),
                )
            )
            if DOCUMENT_LIMIT and len(docs) >= DOCUMENT_LIMIT:
                break
    return docs


def active_features(text: str, discriminators: list[Discriminator]) -> frozenset[str]:
    """Gera assinatura binaria do documento."""
    normalized = fold_text(text)
    return frozenset(
        item.name
        for item in discriminators
        if re.search(item.pattern, normalized, flags=re.IGNORECASE)
    )


def split_docs(docs: list[NewsDoc]) -> tuple[list[NewsDoc], list[NewsDoc]]:
    """Divide todas as noticias em 70/30 antes de qualquer rotulagem."""
    rng = random.Random(SPLIT_SEED)
    shuffled = list(docs)
    rng.shuffle(shuffled)
    train_count = round(len(shuffled) * TRAIN_RATIO)
    return shuffled[:train_count], shuffled[train_count:]


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Similaridade de Jaccard entre duas assinaturas."""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def suggest_theme(features: frozenset[str]) -> str:
    """Infere tema dominante apenas pelos discriminadores ativos."""
    counts: Counter[str] = Counter(feature.split("::", 1)[0] for feature in features)
    if not counts:
        return "-"
    label, hits = counts.most_common(1)[0]
    return f"{label} ({hits})"


class SignatureMemory:
    """Memoria nao supervisionada baseada em assinaturas de features."""

    def __init__(self) -> None:
        self.items: list[MemoryItem] = []

    def train(self, docs: list[NewsDoc], discriminators: list[Discriminator]) -> None:
        """Memoriza todas as noticias do treino que ativam ao menos uma feature."""
        self.items = [
            MemoryItem(doc=doc, features=features)
            for doc in docs
            if (features := active_features(doc.text, discriminators))
        ]

    def query(self, doc: NewsDoc, discriminators: list[Discriminator]) -> UnsupervisedResult:
        """Busca vizinhos parecidos na memoria e decide conhecido/novo/sem_sinal."""
        features = active_features(doc.text, discriminators)
        if not features:
            return UnsupervisedResult("sem_sinal", features, "-", [])

        neighbors = [
            Neighbor(
                doc=item.doc,
                similarity=jaccard(features, item.features),
                shared_features=len(features & item.features),
            )
            for item in self.items
        ]
        neighbors = sorted(
            neighbors,
            key=lambda item: (item.similarity, item.shared_features),
            reverse=True,
        )[:NEIGHBORS_TO_SHOW]

        best_similarity = neighbors[0].similarity if neighbors else 0.0
        status = "conhecido" if best_similarity >= SIMILARITY_MIN else "novo"
        return UnsupervisedResult(status, features, suggest_theme(features), neighbors)


def shorten(value: object, width: int) -> str:
    """Encurta texto para tabela."""
    text = str(value)
    return text if len(text) <= width else text[: max(0, width - 3)] + "..."


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Imprime tabela ASCII simples."""
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    divider = "+".join("-" * (width + 2) for width in widths)

    def line(values: list[str]) -> str:
        return "|".join(f" {values[index].ljust(widths[index])} " for index in range(len(values)))

    print(divider)
    print(line(headers))
    print(divider)
    for row in rows:
        print(line(row))
    print(divider)


def main() -> None:
    """Executa o experimento nao supervisionado completo."""
    regex_bank = load_regex_bank(REGEX_BANK_PATH)
    discriminators = build_discriminators(regex_bank)
    docs = load_news(CSV_PATH, ROOT)
    train_docs, test_docs = split_docs(docs)

    memory = SignatureMemory()
    memory.train(train_docs, discriminators)

    results = [(doc, memory.query(doc, discriminators)) for doc in test_docs]
    status_counts = Counter(result.status for _doc, result in results)
    with_signal = sum(1 for _doc, result in results if result.active_features)

    print(f"CSV: {CSV_PATH}")
    print(f"Banco regex: {REGEX_BANK_PATH}")
    print(f"Noticias lidas: {len(docs)}")
    print(f"Noticias no treino: {len(train_docs)}")
    print(f"Noticias no teste: {len(test_docs)}")
    print(f"Discriminadores regex: {len(discriminators)}")
    print(f"Assinaturas memorizadas: {len(memory.items)}")
    print(f"Noticias de teste com algum sinal: {with_signal}")
    print("Status no teste:", dict(status_counts))
    print(f"Limiar de similaridade: {SIMILARITY_MIN}")

    rows: list[list[str]] = []
    for index, (doc, result) in enumerate(results[:EXAMPLES_TO_PRINT], start=1):
        best = result.neighbors[0] if result.neighbors else None
        rows.append(
            [
                str(index),
                shorten(doc.title, 48),
                result.status,
                result.suggested_theme,
                str(len(result.active_features)),
                f"{best.similarity:.3f}" if best else "0.000",
                str(best.shared_features) if best else "0",
                shorten(best.doc.title, 42) if best else "-",
            ]
        )

    print("\n=== Busca nao supervisionada por assinatura ===")
    print_table(
        [
            "#",
            "titulo_teste",
            "status",
            "tema_sugerido",
            "features",
            "sim_top",
            "features_comuns",
            "vizinho_mais_proximo",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
