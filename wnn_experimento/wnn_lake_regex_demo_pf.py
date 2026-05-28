"""
Motor demonstrativo de WNN com discriminadores regex externos.

Este arquivo e um laboratorio independente da pipeline principal. A ideia e
mostrar como uma rede neural sem peso (WNN) pode classificar documentos a partir
de discriminadores regex, preservando auditabilidade e priorizando qualidade.

Contribuicao metodologica:
    A WNN nao substitui a regex; ela usa discriminadores regex como sinais
    binarios e aprende combinacoes recorrentes, aceitando classificacao apenas
    quando ha confianca e margem suficientes.

O desenho e intencionalmente conservador:

1. Entrada de dados
   O script le um CSV de catalogo e os arquivos de texto/Markdown apontados por
   uma coluna de caminho. O padrao publicavel usa uma amostra pequena em
   data_exemplo_noticias/manifest.csv, com cerca de 300 noticias copiadas para
   data_exemplo_noticias/noticias_markdown, dentro desta propria pasta de
   experimento. Para outro repositorio, basta
   informar outro CSV e outra coluna de caminho.

2. Banco de discriminadores
   As regex nao ficam no motor. Elas ficam em um arquivo JSON externo no formato:

       {
         "tema_a": ["regex 1", "regex 2"],
         "tema_b": ["regex 3", "regex 4"]
       }

   Cada regex vira uma feature binaria auditavel. Se a regex bate no texto, a
   feature fica ativa. Isso permite trocar o dominio sem reescrever a WNN.

3. Rotulagem fraca para treino
   Para esta demo, as proprias regex produzem um rotulo de referencia quando um
   tema tem hits suficientes e margem sobre o segundo tema. Isso nao substitui
   verdade humana. Serve apenas para demonstrar como a memoria associativa pode
   aprender padroes recorrentes a partir de uma fonte programatica.

4. Memoria WNN
   A WNN nao usa pesos nem gradiente. Ela memoriza features, pares e trios de
   features observados por tema. Na classificacao, cada tema recebe votos quando
   o documento ativa combinacoes ja vistas. Pares e trios valem mais porque
   representam evidencia combinada, nao apenas um termo isolado.

5. Divisao 70/30
   Sem argumentos, o script divide os documentos rotulados em 70% para treino
   e 30% para teste, preservando a proporcao por tema sempre que possivel.

6. Qualidade acima de cobertura
   O objetivo nao e classificar tudo; e aceitar somente decisoes com boa
   separacao. O restante fica como nao classificado pela WNN. Este experimento
   nao aciona LLM residual.

7. Como ler os resultados
   Acuracia e precisao sao calculadas contra o rotulo regex de referencia. Em
   um estudo de qualidade real, a etapa seguinte seria validar uma amostra das
   decisoes aceitas por revisao humana, registrando erros, features ativadas,
   votos e margem.

Execucao padrao no repositorio atual:

    python wnn_experimento/wnn_lake_regex_demo_pf.py

Sem argumentos, o script ja usa o CSV local, o banco PF externo, divisao 70/30
e limiares conservadores. Para reaproveitar o motor em outro repositorio ou
com outro banco de regex, edite a secao de configuracao fixa no topo do arquivo.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR

# Configuracao fixa do experimento.
#
# O script nao recebe argumentos de linha de comando. Para usar outro conjunto
# de dados ou outro banco de discriminadores, edite os valores abaixo.
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
MIN_HITS_FOR_WEAK_LABEL = 2
MIN_MARGIN_FOR_WEAK_LABEL = 1
CONFIDENCE_MIN = 0.45
MARGIN_MIN = 2
TOP_VOTES_MIN = 1
EXAMPLES_TO_PRINT = 10


@dataclass(frozen=True)
class NewsDoc:
    """
    Documento textual usado pelo motor.

    Atributos:
        title: titulo exibido nos relatorios.
        path: caminho do arquivo Markdown/texto carregado.
        text: texto consolidado usado na regex e na WNN. Inclui titulo,
            subtitulo, tags e corpo para aumentar o contexto dos discriminadores.
    """

    title: str
    path: Path
    text: str


@dataclass(frozen=True)
class Discriminator:
    """
    Feature regex auditavel.

    Atributos:
        name: identificador unico da feature no formato "label::dNN".
        label: classe/tema canonico associado ao discriminador.
        pattern: expressao regular usada para ativar a feature no texto.
    """

    name: str
    label: str
    pattern: str


@dataclass
class WNNResult:
    """
    Resultado bruto da classificacao WNN.

    Atributos:
        label: classe aceita pela WNN, ou None quando houve abstencao.
        votes: pontuacao final por classe.
        confidence: votos da classe vencedora divididos pelo total de votos.
        margin: diferenca entre a classe mais votada e a segunda colocada.
        active: lista de features regex ativadas pelo documento.
    """

    label: str | None
    votes: dict[str, int]
    confidence: float
    margin: int
    active: list[str]


def fix_mojibake(text: str) -> str:
    """
    Tenta corrigir textos UTF-8 lidos como Latin-1 em dumps antigos.

    Entrada:
        text: texto possivelmente com mojibake, como "operaÃ§Ã£o".

    Saida:
        Texto corrigido quando a conversao for segura; caso contrario, retorna
        o valor original.
    """
    if "Ã" not in text and "Â" not in text:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def fold_text(text: str) -> str:
    """
    Normaliza texto para comparacao regex sem depender de acentos.

    Entrada:
        text: texto livre.

    Saida:
        Texto em minusculas, sem acentos e com caracteres ASCII.
    """
    text = fix_mojibake(text or "")
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii").lower()


def resolve_path(base_dir: Path, value: str) -> Path:
    """
    Resolve caminhos absolutos ou relativos.

    Entrada:
        base_dir: diretorio base usado quando `value` e relativo.
        value: caminho vindo das constantes de configuracao ou do CSV.

    Saida:
        Path absoluto ou relativo resolvido contra `base_dir`.
    """

    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_regex_bank(path: Path) -> dict[str, list[str]]:
    """
    Carrega banco externo de regex e valida o formato minimo.

    Entrada:
        path: JSON no formato {"label": ["regex_1", "regex_2"]}.

    Saida:
        Dicionario normalizado de label para lista de regex.

    Erros:
        Levanta ValueError quando o JSON nao e um objeto ou quando nenhum
        discriminador valido foi encontrado. Regex invalidas falham no compile.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("O banco regex deve ser um objeto JSON: {label: [patterns]}.")

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

    if not bank:
        raise ValueError("Nenhum discriminador regex valido foi encontrado.")
    return bank


def build_discriminators(regex_bank: dict[str, list[str]]) -> list[Discriminator]:
    """
    Transforma cada regex em uma feature binaria nomeada e auditavel.

    Entrada:
        regex_bank: dicionario {"label": ["pattern", ...]}.

    Saida:
        Lista de Discriminator. Cada regex recebe nome estavel no formato
        "label::dNN", usado em relatorios e na memoria WNN.
    """
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


def load_news(
    csv_path: Path,
    base_dir: Path,
    limit: int,
    status_column: str,
    status_ok: str,
    path_column: str,
    title_column: str,
    subtitle_column: str,
    tags_column: str,
) -> list[NewsDoc]:
    """
    Le documentos a partir de um CSV simples de catalogo.

    Entradas:
        csv_path: CSV com uma linha por documento.
        base_dir: raiz para resolver caminhos relativos.
        limit: maximo de documentos lidos; 0 significa sem limite.
        status_column: coluna usada para filtrar documentos validos.
        status_ok: valor aceito em `status_column`.
        path_column: coluna que aponta para o arquivo texto/Markdown.
        title_column: coluna com titulo exibivel.
        subtitle_column: coluna com subtitulo ou resumo.
        tags_column: coluna com tags/metadados textuais.

    Saida:
        Lista de NewsDoc com texto consolidado. Linhas sem arquivo ou com
        caminho inexistente sao ignoradas.
    """
    docs: list[NewsDoc] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if status_column and status_ok and row.get(status_column) != status_ok:
                continue

            rel_path = row.get(path_column, "")
            if not rel_path:
                continue

            path = resolve_path(base_dir, rel_path)
            if not path.exists():
                continue

            body = path.read_text(encoding="utf-8", errors="replace")
            first_line = body.splitlines()[0].lstrip("# ") if body.splitlines() else ""
            title = row.get(title_column) or first_line
            subtitle = row.get(subtitle_column, "")
            tags = row.get(tags_column, "")
            docs.append(
                NewsDoc(
                    title=fix_mojibake(title),
                    path=path,
                    text="\n".join([title, subtitle, tags, body]),
                )
            )

            if limit and len(docs) >= limit:
                break
    return docs


def active_features(text: str, discriminators: list[Discriminator]) -> set[str]:
    """
    Retorna quais discriminadores regex foram ativados no documento.

    Entrada:
        text: texto do documento.
        discriminators: lista de features regex possiveis.

    Saida:
        Conjunto com nomes das features ativadas, como "trafico_drogas::d02".
    """
    normalized = fold_text(text)
    active: set[str] = set()
    for discriminator in discriminators:
        if re.search(discriminator.pattern, normalized, flags=re.IGNORECASE):
            active.add(discriminator.name)
    return active


def regex_scores(text: str, discriminators: list[Discriminator]) -> Counter[str]:
    """
    Soma hits por tema usando apenas as features ativas.

    Entrada:
        text: texto do documento.
        discriminators: lista de features regex.

    Saida:
        Counter em que a chave e a label e o valor e a quantidade de
        discriminadores ativados para essa label.
    """
    active = active_features(text, discriminators)
    scores: Counter[str] = Counter()
    for name in active:
        label = name.split("::", 1)[0]
        scores[label] += 1
    return scores


def weak_label(
    text: str,
    discriminators: list[Discriminator],
    min_hits: int,
    min_margin: int,
) -> str | None:
    """
    Cria um rotulo de treino apenas quando a regex tem evidencia clara.

    Entradas:
        text: texto do documento.
        discriminators: discriminadores regex externos.
        min_hits: minimo de features ativadas para o tema vencedor.
        min_margin: diferenca minima entre o primeiro e o segundo tema.

    Saida:
        Label aceita para treino/teste, ou None quando o documento nao tem
        sinal suficiente. Esse None nao e erro: e abstencao na rotulagem fraca.
    """
    scores = regex_scores(text, discriminators)
    if not scores:
        return None

    ranking = scores.most_common(2)
    label, hits = ranking[0]
    second_hits = ranking[1][1] if len(ranking) > 1 else 0
    return label if hits >= min_hits and hits - second_hits >= min_margin else None


class AssociativeWNN:
    """
    Memoria associativa sem pesos.

    Cada label guarda features, pares e trios de features vistas no treino.
    Na inferencia, o documento vota nas labels cujas memorias reconhecem essas
    combinacoes. A classe pode se abster quando votos, confianca ou margem
    forem insuficientes.
    """

    def __init__(self, discriminators: list[Discriminator]) -> None:
        """
        Inicializa a memoria WNN vazia.

        Entrada:
            discriminators: universo de features regex que podem ser ativadas.

        Saida:
            Nenhuma. A instancia fica pronta para receber treino em `train`.
        """

        self.discriminators = discriminators
        self.single_memory: dict[str, set[str]] = defaultdict(set)
        self.pair_memory: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.triple_memory: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self.labels: set[str] = set()

    def train(self, rows: list[tuple[NewsDoc, str]]) -> None:
        """
        Memoriza padroes observados nos exemplos rotulados.

        Entrada:
            rows: lista de pares (documento, label). A label normalmente vem da
                rotulagem fraca por regex neste experimento.

        Comportamento:
            Para cada documento, a WNN guarda apenas features pertencentes a
            propria label do exemplo. Isso reduz vazamento de coocorrencias de
            outros temas. A memoria armazena features isoladas, pares e trios.

        Saida:
            Nenhuma. O estado interno `single_memory`, `pair_memory`,
            `triple_memory` e `labels` e atualizado.
        """

        for doc, label in rows:
            active = sorted(
                feature
                for feature in active_features(doc.text, self.discriminators)
                if feature.startswith(f"{label}::")
            )
            if not active:
                continue

            self.labels.add(label)
            self.single_memory[label].update(active)
            self.pair_memory[label].update(itertools.combinations(active, 2))
            self.triple_memory[label].update(itertools.combinations(active, 3))

    def classify(
        self,
        text: str,
        confidence_min: float = 0.45,
        margin_min: int = 2,
        top_votes_min: int = 1,
    ) -> WNNResult:
        """
        Classifica um texto ou se abstem quando a decisao e fraca.

        Entradas:
            text: texto a classificar.
            confidence_min: confianca minima para aceitar a classe vencedora.
            margin_min: margem minima entre primeiro e segundo colocados.
            top_votes_min: minimo absoluto de votos da classe vencedora.

        Saida:
            WNNResult com:
                label: classe aceita, ou None quando a WNN nao classifica;
                votes: pontuacao por classe;
                confidence: proporcao de votos do vencedor;
                margin: diferenca entre primeiro e segundo;
                active: features ativadas no texto.
        """

        active = sorted(active_features(text, self.discriminators))
        pairs = set(itertools.combinations(active, 2))
        triples = set(itertools.combinations(active, 3))
        votes: dict[str, int] = {}

        for label in sorted(self.labels):
            single_votes = sum(1 for feature in active if feature in self.single_memory[label])
            pair_votes = sum(1 for pair in pairs if pair in self.pair_memory[label])
            triple_votes = sum(1 for triple in triples if triple in self.triple_memory[label])
            votes[label] = single_votes + (2 * pair_votes) + (3 * triple_votes)

        ranking = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        if not ranking or ranking[0][1] == 0:
            return WNNResult(None, votes, 0.0, 0, active)

        top_label, top_votes = ranking[0]
        second_votes = ranking[1][1] if len(ranking) > 1 else 0
        total_votes = sum(votes.values()) or 1
        confidence = top_votes / total_votes
        margin = top_votes - second_votes
        accepted = (
            top_votes >= top_votes_min
            and confidence >= confidence_min
            and margin >= margin_min
        )
        return WNNResult(top_label if accepted else None, votes, confidence, margin, active)


def split_docs_train_test_ratio(
    docs: list[NewsDoc],
    train_ratio: float,
    seed: int,
) -> tuple[list[NewsDoc], list[NewsDoc]]:
    """
    Divide todas as noticias em treino e teste antes da rotulagem fraca.

    Entradas:
        docs: lista completa de documentos carregados do manifest.
        train_ratio: proporcao de noticias usada no treino.
        seed: semente para embaralhamento reprodutivel.

    Saida:
        Tupla (train_docs, test_docs). Com 300 noticias e train_ratio=0.70,
        a saida esperada e 210 documentos de treino e 90 de teste.
    """

    rng = random.Random(seed)
    shuffled = list(docs)
    rng.shuffle(shuffled)
    train_count = round(len(shuffled) * train_ratio)
    return shuffled[:train_count], shuffled[train_count:]


def build_labeled_rows(
    docs: list[NewsDoc],
    discriminators: list[Discriminator],
    min_hits: int,
    min_margin: int,
) -> list[tuple[NewsDoc, str]]:
    """
    Aplica a rotulagem fraca por regex a uma lista de documentos.

    Entradas:
        docs: documentos de treino ou teste.
        discriminators: discriminadores regex externos.
        min_hits: minimo de features ativadas para aceitar uma referencia.
        min_margin: margem minima sobre o segundo tema.

    Saida:
        Lista de pares (documento, label). Documentos sem referencia clara sao
        omitidos da lista, mas continuam existindo no split original.
    """

    labeled: list[tuple[NewsDoc, str]] = []
    for doc in docs:
        label = weak_label(
            doc.text,
            discriminators=discriminators,
            min_hits=min_hits,
            min_margin=min_margin,
        )
        if label:
            labeled.append((doc, label))
    return labeled


def evaluate(
    wnn: AssociativeWNN,
    rows: list[tuple[NewsDoc, str]],
    confidence_min: float,
    margin_min: int,
    top_votes_min: int,
) -> dict[str, Any]:
    """
    Avalia a WNN contra os rotulos de referencia gerados pelas regex.

    Entradas:
        wnn: instancia treinada.
        rows: exemplos de teste no formato (documento, label_referencia).
        confidence_min: limiar de confianca usado na classificacao.
        margin_min: limiar de margem usado na classificacao.
        top_votes_min: votos minimos usados na classificacao.

    Saida:
        Dicionario com metricas globais e por tema:
            total: quantidade de exemplos avaliados;
            accepted: quantidade de decisoes aceitas pela WNN;
            abstained: quantidade de documentos nao classificados;
            correct: decisoes aceitas iguais ao rotulo de referencia;
            coverage: accepted / total;
            accuracy_on_all: correct / total;
            accuracy_on_accepted: correct / accepted;
            per_label: metricas por label;
            wrong: primeiros erros aceitos;
            abstained_examples: primeiros casos nao classificados.
    """

    total = len(rows)
    accepted = 0
    correct = 0
    abstained = 0
    wrong: list[tuple[NewsDoc, str, WNNResult]] = []
    abstained_examples: list[tuple[NewsDoc, str, WNNResult]] = []
    per_label_total: Counter[str] = Counter()
    per_label_correct: Counter[str] = Counter()
    per_label_accepted: Counter[str] = Counter()

    for doc, expected in rows:
        result = wnn.classify(
            doc.text,
            confidence_min=confidence_min,
            margin_min=margin_min,
            top_votes_min=top_votes_min,
        )
        per_label_total[expected] += 1

        if result.label is None:
            abstained += 1
            if len(abstained_examples) < 5:
                abstained_examples.append((doc, expected, result))
            continue

        accepted += 1
        per_label_accepted[expected] += 1
        if result.label == expected:
            correct += 1
            per_label_correct[expected] += 1
        elif len(wrong) < 5:
            wrong.append((doc, expected, result))

    per_label = {}
    for label in sorted(per_label_total):
        label_total = per_label_total[label]
        label_accepted = per_label_accepted[label]
        per_label[label] = {
            "testes": label_total,
            "aceitos": label_accepted,
            "corretos": per_label_correct[label],
            "cobertura": round(label_accepted / label_total, 3) if label_total else 0.0,
            "precisao_aceitos": round(per_label_correct[label] / label_accepted, 3)
            if label_accepted
            else 0.0,
        }

    return {
        "total": total,
        "accepted": accepted,
        "abstained": abstained,
        "correct": correct,
        "coverage": accepted / total if total else 0.0,
        "accuracy_on_all": correct / total if total else 0.0,
        "accuracy_on_accepted": correct / accepted if accepted else 0.0,
        "per_label": per_label,
        "wrong": wrong,
        "abstained_examples": abstained_examples,
    }


def print_examples(
    title: str,
    examples: list[tuple[NewsDoc, str]],
    wnn: AssociativeWNN,
    thresholds: dict[str, float | int],
    root: Path,
    limit: int,
) -> None:
    """
    Imprime uma tabela visual de exemplos classificados pela WNN.

    Entradas:
        title: titulo da secao impressa.
        examples: exemplos no formato (documento, label_referencia).
        wnn: instancia treinada.
        thresholds: dicionario com confidence_min, margin_min e top_votes_min.
        root: raiz usada para caminhos relativos, mantida para extensoes.
        limit: maximo de linhas exibidas.

    Saida:
        Nenhuma. Escreve no stdout uma tabela com titulo, referencia,
        decisao_wnn, top_score, confianca, margem e status.
    """

    print(f"\n=== {title} ===")
    table_rows: list[list[str]] = []
    for index, (doc, expected) in enumerate(examples[:limit], start=1):
        result = wnn.classify(
            doc.text,
            confidence_min=float(thresholds["confidence_min"]),
            margin_min=int(thresholds["margin_min"]),
            top_votes_min=int(thresholds["top_votes_min"]),
        )
        top_votes = dict(sorted(result.votes.items(), key=lambda item: item[1], reverse=True)[:5])
        top_label, top_score = next(iter(top_votes.items()), ("-", 0))
        decision = result.label or "nao_classificado"
        status = "classificado" if result.label else "abstencao"
        table_rows.append(
            [
                str(index),
                shorten(doc.title, 54),
                shorten(expected, 24),
                shorten(decision, 24),
                f"{top_label}:{top_score}",
                f"{result.confidence:.3f}",
                str(result.margin),
                status,
            ]
        )

    print_table(
        ["#", "titulo", "referencia", "decisao_wnn", "top_score", "conf", "margem", "status"],
        table_rows,
    )


def shorten(value: object, width: int) -> str:
    """
    Encurta texto para manter a tabela legivel.

    Entrada:
        value: valor a converter para string.
        width: largura maxima.

    Saida:
        String original ou truncada com "...".
    """

    text = str(value)
    return text if len(text) <= width else text[: max(0, width - 3)] + "..."


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """
    Imprime tabela ASCII simples.

    Entradas:
        headers: nomes das colunas.
        rows: linhas ja convertidas para string.

    Saida:
        Nenhuma. A tabela e enviada para stdout.
    """

    if not rows:
        print("(sem exemplos)")
        return

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
    """
    Executa o experimento completo.

    Fluxo:
        1. le as constantes de configuracao do topo do arquivo;
        2. carrega banco regex externo;
        3. transforma regex em discriminadores;
        4. le documentos do manifest;
        5. divide todas as noticias em 70/30;
        6. cria rotulos fracos dentro do treino e do teste;
        7. treina a memoria WNN com os rotulados do treino;
        8. avalia nos rotulados do teste;
        9. imprime metricas, tabela visual e abstencoes.
    """

    root = ROOT.resolve()
    csv_path = resolve_path(root, str(CSV_PATH)).resolve()
    regex_bank_path = resolve_path(root, str(REGEX_BANK_PATH)).resolve()

    regex_bank = load_regex_bank(regex_bank_path)
    discriminators = build_discriminators(regex_bank)
    docs = load_news(
        csv_path=csv_path,
        base_dir=root,
        limit=DOCUMENT_LIMIT,
        status_column=STATUS_COLUMN,
        status_ok=STATUS_OK,
        path_column=PATH_COLUMN,
        title_column=TITLE_COLUMN,
        subtitle_column=SUBTITLE_COLUMN,
        tags_column=TAGS_COLUMN,
    )

    train_docs, test_docs = split_docs_train_test_ratio(
        docs,
        train_ratio=TRAIN_RATIO,
        seed=SPLIT_SEED,
    )
    train_rows = build_labeled_rows(
        train_docs,
        discriminators=discriminators,
        min_hits=MIN_HITS_FOR_WEAK_LABEL,
        min_margin=MIN_MARGIN_FOR_WEAK_LABEL,
    )
    test_rows = build_labeled_rows(
        test_docs,
        discriminators=discriminators,
        min_hits=MIN_HITS_FOR_WEAK_LABEL,
        min_margin=MIN_MARGIN_FOR_WEAK_LABEL,
    )

    all_labeled_count = len(train_rows) + len(test_rows)
    train_counts = Counter(label for _doc, label in train_rows)
    per_label = Counter(label for _doc, label in [*train_rows, *test_rows])

    wnn = AssociativeWNN(discriminators)
    wnn.train(train_rows)

    thresholds: dict[str, float | int] = {
        "confidence_min": CONFIDENCE_MIN,
        "margin_min": MARGIN_MIN,
        "top_votes_min": TOP_VOTES_MIN,
    }

    metrics = evaluate(
        wnn,
        test_rows,
        confidence_min=float(thresholds["confidence_min"]),
        margin_min=int(thresholds["margin_min"]),
        top_votes_min=int(thresholds["top_votes_min"]),
    )

    print(f"CSV: {csv_path}")
    print(f"Banco regex: {regex_bank_path}")
    print(f"Documentos lidos: {len(docs)}")
    print(f"Temas no banco regex: {len(regex_bank)}")
    print(f"Discriminadores regex: {len(discriminators)}")
    print(f"Noticias no treino: {len(train_docs)}")
    print(f"Noticias no teste: {len(test_docs)}")
    print(f"Documentos rotulados por regex fraca: {all_labeled_count}")
    print(f"Rotulados no treino: {len(train_rows)}")
    print(f"Rotulados no teste: {len(test_rows)}")
    print("Rotulados de treino por tema:", dict(sorted(train_counts.items())))
    print("Top temas encontrados:", dict(per_label.most_common(8)))

    print("\n=== Avaliacao WNN ===")
    print(f"Exemplos de treino: {len(train_docs)}")
    print(f"Exemplos de teste: {len(test_docs)}")
    print(f"Exemplos rotulados usados no treino WNN: {len(train_rows)}")
    print(f"Exemplos rotulados usados na avaliacao: {metrics['total']}")
    print(f"Divisao treino/teste: {TRAIN_RATIO:.0%}/{1 - TRAIN_RATIO:.0%}")
    print("Limiar usado:", thresholds)
    print(f"Decisoes aceitas pela WNN: {metrics['accepted']}")
    print(f"Nao classificados por baixa confianca: {metrics['abstained']}")
    print(f"Cobertura WNN: {metrics['coverage']:.3f}")
    print(f"Acuracia sobre todos os testes: {metrics['accuracy_on_all']:.3f}")
    print(f"Precisao entre decisoes aceitas: {metrics['accuracy_on_accepted']:.3f}")

    print("\nPor tema:")
    for label, values in metrics["per_label"].items():
        print(f"- {label}: {values}")

    print_examples(
        "Classificacao WNN em documentos fora do treino",
        test_rows,
        wnn,
        thresholds,
        root,
        EXAMPLES_TO_PRINT,
    )

    if metrics["wrong"]:
        print("\n=== Primeiros erros aceitos pela WNN ===")
        for doc, expected, result in metrics["wrong"]:
            top_votes = dict(sorted(result.votes.items(), key=lambda item: item[1], reverse=True)[:5])
            print("\nTITULO:", doc.title)
            print("ESPERADO_REGEX:", expected)
            print("DECISAO_WNN:", result.label)
            print("TOP_VOTOS_WNN:", top_votes)

    if metrics["abstained_examples"]:
        print("\n=== Primeiros nao classificados por baixa confianca ===")
        for doc, expected, result in metrics["abstained_examples"]:
            top_votes = dict(sorted(result.votes.items(), key=lambda item: item[1], reverse=True)[:5])
            print("\nTITULO:", doc.title)
            print("ESPERADO_REGEX:", expected)
            print("TOP_VOTOS_WNN:", top_votes)


if __name__ == "__main__":
    main()
