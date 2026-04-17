from __future__ import annotations

import argparse
import hashlib
import html
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter


BASE_LIST_URL = "https://www.gov.br/pf/pt-br/assuntos/noticias/noticias-operacoes"
DEFAULT_INDEX_CSV = Path("data/pf_operacoes_index.csv")
DEFAULT_CONTENT_CSV = Path("data/pf_operacoes_conteudos.csv")
DEFAULT_MARKDOWN_DIR = Path("data/noticias_markdown")
DEFAULT_STEP = 60
DEFAULT_TIMEOUT = 30
DEFAULT_SLEEP = 0.2
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)

PUBLISHED_RE = re.compile(
    r"publicado\s+(?P<date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<time>\d{2}h\d{2})\s+(?P<kind>\S+)",
    re.IGNORECASE,
)
ARTICLE_DATES_RE = re.compile(
    r"Publicado em\s+(?P<published>\d{2}/\d{2}/\d{4}\s+\d{2}h\d{2})"
    r"(?:\s+Atualizado em\s+(?P<updated>\d{2}/\d{2}/\d{4}\s+\d{2}h\d{2}))?",
    re.IGNORECASE,
)


@dataclass
class ArticleCore:
    section: str | None
    category_label: str | None
    title: str
    subtitle: str | None
    published_at: str | None
    updated_at: str | None
    tags: list[str]
    content_html: str


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_soup(session: requests.Session, url: str, timeout: int) -> BeautifulSoup:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def parse_listing_date(raw_text: str | None) -> tuple[str | None, str | None, str | None]:
    text = clean_text(raw_text)
    if not text:
        return None, None, None

    match = PUBLISHED_RE.search(text)
    if not match:
        return text, None, None

    published_at = f"{match.group('date')} {match.group('time')}"
    return published_at, match.group("date"), match.group("time")


def parse_listing_item(article: Tag, offset: int, step: int, item_index: int) -> dict[str, object]:
    subtitle_node = article.select_one(".subtitle")
    headline = article.select_one("h2.tileHeadline a")
    summary = article.select_one("p.tileBody")
    byline = article.select_one(".documentByLine")

    subtitle_label = clean_text(subtitle_node.get_text(" ", strip=True) if subtitle_node else None)
    published_at, published_date, published_time = parse_listing_date(
        byline.get_text(" ", strip=True) if byline else None
    )

    content_type = None
    byline_text = clean_text(byline.get_text(" ", strip=True) if byline else None)
    if byline_text:
        parts = byline_text.split()
        if parts:
            content_type = parts[-1]

    title = clean_text(headline.get_text(" ", strip=True) if headline else None)
    url = headline["href"] if headline and headline.has_attr("href") else None
    tags = [
        clean_text(tag.get_text(" ", strip=True))
        for tag in article.select(".keywords a")
    ]
    tags = [tag for tag in tags if tag]

    return {
        "offset": offset,
        "page_number": (offset // step) + 1,
        "item_index": item_index,
        "categoria": subtitle_label,
        "titulo": title,
        "subtitulo": clean_text(summary.get_text(" ", strip=True) if summary else None),
        "data_publicacao": published_date,
        "hora_publicacao": published_time,
        "publicado_em": published_at,
        "tipo_conteudo": content_type,
        "tags": " | ".join(tags),
        "total_tags": len(tags),
        "link": url,
    }


def collect_listing(
    session: requests.Session,
    start_offset: int,
    end_offset: int | None,
    step: int,
    timeout: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    offset = start_offset

    while True:
        if end_offset is not None and offset > end_offset:
            break

        url = f"{BASE_LIST_URL}?b_start:int={offset}"
        soup = fetch_soup(session, url, timeout=timeout)
        articles = soup.select("article.tileItem")
        print(f"[collect] offset={offset} items={len(articles)}")

        if not articles:
            break

        for item_index, article in enumerate(articles, start=1):
            rows.append(parse_listing_item(article, offset=offset, step=step, item_index=item_index))

        offset += step
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).sort_values(["offset", "item_index"]).reset_index(drop=True)
    return df


def parse_article_core(soup: BeautifulSoup) -> ArticleCore:
    content = soup.select_one("#content article")
    if content is None:
        raise ValueError("Nao foi possivel localizar '#content article' na pagina.")

    content_core = content.select_one("#content-core, .content-core")
    if content_core is None:
        raise ValueError("Nao foi possivel localizar '#content-core' na pagina.")

    content_core = BeautifulSoup(str(content_core), "lxml").select_one("#content-core, .content-core")
    assert content_core is not None

    for selector in ["script", "style", "noscript", ".documentActions", ".visualClear"]:
        for node in content_core.select(selector):
            node.decompose()

    dates_container = content.select_one("#viewlet-above-content-body, .viewlet-above-content-body")
    dates_text = clean_text(dates_container.get_text(" ", strip=True) if dates_container else None)
    published_at = None
    updated_at = None
    if dates_text:
        dates_match = ARTICLE_DATES_RE.search(dates_text)
        if dates_match:
            published_at = dates_match.group("published")
            updated_at = dates_match.group("updated")

    tags = [
        clean_text(tag.get_text(" ", strip=True))
        for tag in soup.select("#content > div.column a")
    ]
    tags = [tag for tag in tags if tag]

    title_node = content.select_one("h1.documentFirstHeading")
    if title_node is None:
        raise ValueError("Nao foi possivel localizar o titulo principal da noticia.")

    section = content.select_one("p.section")
    category_label = content.select_one("p.nitfSubtitle")
    subtitle = content.select_one(".documentDescription")

    return ArticleCore(
        section=clean_text(section.get_text(" ", strip=True) if section else None),
        category_label=clean_text(category_label.get_text(" ", strip=True) if category_label else None),
        title=clean_text(title_node.get_text(" ", strip=True)) or "",
        subtitle=clean_text(subtitle.get_text(" ", strip=True) if subtitle else None),
        published_at=published_at,
        updated_at=updated_at,
        tags=tags,
        content_html=str(content_core),
    )


def build_docling_html(article: ArticleCore) -> str:
    parts: list[str] = ["<html><body><article>"]

    if article.section:
        parts.append(f"<p>{html.escape(article.section)}</p>")
    if article.category_label:
        parts.append(f"<p>{html.escape(article.category_label)}</p>")

    parts.append(f"<h1>{html.escape(article.title)}</h1>")

    if article.subtitle:
        parts.append(f"<p>{html.escape(article.subtitle)}</p>")

    if article.published_at:
        parts.append(f"<p>Publicado em {html.escape(article.published_at)}</p>")
    if article.updated_at:
        parts.append(f"<p>Atualizado em {html.escape(article.updated_at)}</p>")
    if article.tags:
        parts.append(f"<p>Tags: {html.escape(', '.join(article.tags))}</p>")

    parts.append(article.content_html)
    parts.append("</article></body></html>")
    return "".join(parts)


def slug_from_url(url: str) -> str:
    slug = Path(urlparse(url).path).name
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", slug).strip("-").lower()
    if slug:
        return slug
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    return f"noticia-{digest}"


def markdown_filename(url: str) -> str:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug_from_url(url)}-{digest}.md"


def export_markdown(converter: DocumentConverter, article: ArticleCore, output_file: Path) -> str:
    html_content = build_docling_html(article)
    result = converter.convert_string(
        html_content,
        format=InputFormat.HTML,
        name=output_file.name.replace(".md", ".html"),
    )
    markdown = result.document.export_to_markdown()
    output_file.write_text(markdown, encoding="utf-8")
    return markdown


def load_index_csv(index_csv: Path) -> pd.DataFrame:
    if not index_csv.exists():
        raise FileNotFoundError(f"Arquivo de indice nao encontrado: {index_csv}")
    df = pd.read_csv(index_csv)
    if "link" not in df.columns:
        raise ValueError("O CSV de indice precisa ter a coluna 'link'.")
    return df


def load_existing_manifest(content_csv: Path) -> pd.DataFrame:
    if content_csv.exists():
        return pd.read_csv(content_csv)
    return pd.DataFrame(
        columns=[
            "link",
            "markdown_path",
            "status",
            "titulo_extraido",
            "subtitulo_extraido",
            "publicado_em_extraido",
            "atualizado_em_extraido",
            "tags_extraidas",
            "erro",
        ]
    )


def normalize_link(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def prepare_index_df(df_index: pd.DataFrame) -> pd.DataFrame:
    prepared = df_index.copy()
    prepared["link"] = prepared["link"].map(normalize_link)
    return prepared.dropna(subset=["link"]).drop_duplicates(subset=["link"], keep="first").reset_index(drop=True)


def prepare_manifest_df(manifest: pd.DataFrame) -> pd.DataFrame:
    prepared = manifest.copy()
    prepared["link"] = prepared["link"].map(normalize_link)
    return prepared.dropna(subset=["link"]).drop_duplicates(subset=["link"], keep="last").reset_index(drop=True)


def build_manifest_lookup(manifest: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        record["link"]: record
        for record in manifest.to_dict(orient="records")
        if isinstance(record.get("link"), str)
    }


def resolve_markdown_path(markdown_path: object, markdown_dir: Path, link: str) -> Path:
    if isinstance(markdown_path, str) and markdown_path.strip():
        return Path(markdown_path.strip())
    return markdown_dir / markdown_filename(link)


def article_already_exists(
    link: str,
    manifest_lookup: dict[str, dict[str, object]],
    markdown_dir: Path,
) -> bool:
    existing = manifest_lookup.get(link)
    if not existing:
        return False

    if existing.get("status") != "ok":
        return False

    markdown_path = resolve_markdown_path(existing.get("markdown_path"), markdown_dir, link)
    return markdown_path.exists()


def count_missing_articles(
    df_index: pd.DataFrame,
    manifest_lookup: dict[str, dict[str, object]],
    markdown_dir: Path,
) -> int:
    missing = 0
    for link in df_index["link"].tolist():
        if not article_already_exists(link, manifest_lookup, markdown_dir):
            missing += 1
    return missing


def extract_articles(
    session: requests.Session,
    index_csv: Path,
    content_csv: Path,
    markdown_dir: Path,
    timeout: int,
    sleep_seconds: float,
    limit: int | None,
    only_missing: bool,
) -> pd.DataFrame:
    df_index = prepare_index_df(load_index_csv(index_csv))
    manifest = prepare_manifest_df(load_existing_manifest(content_csv))
    manifest_lookup = build_manifest_lookup(manifest)

    converter = DocumentConverter()
    markdown_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = manifest.to_dict(orient="records")
    processed = 0

    for _, row in df_index.iterrows():
        link = row["link"]
        if only_missing and article_already_exists(link, manifest_lookup, markdown_dir):
            print(f"[extract] pulando existente: {link}")
            continue
        if limit is not None and processed >= limit:
            break

        output_file = markdown_dir / markdown_filename(link)
        print(f"[extract] {link}")

        try:
            soup = fetch_soup(session, link, timeout=timeout)
            article = parse_article_core(soup)
            export_markdown(converter, article, output_file)
            rows.append(
                {
                    "link": link,
                    "markdown_path": str(output_file),
                    "status": "ok",
                    "titulo_extraido": article.title,
                    "subtitulo_extraido": article.subtitle,
                    "publicado_em_extraido": article.published_at,
                    "atualizado_em_extraido": article.updated_at,
                    "tags_extraidas": " | ".join(article.tags),
                    "erro": None,
                }
            )
            manifest_lookup[link] = rows[-1]
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "link": link,
                    "markdown_path": str(output_file),
                    "status": "erro",
                    "titulo_extraido": None,
                    "subtitulo_extraido": None,
                    "publicado_em_extraido": None,
                    "atualizado_em_extraido": None,
                    "tags_extraidas": None,
                    "erro": str(exc),
                }
            )
            manifest_lookup[link] = rows[-1]

        processed += 1
        updated_manifest = pd.DataFrame(rows).drop_duplicates(subset=["link"], keep="last").reset_index(drop=True)
        ensure_parent(content_csv)
        updated_manifest.to_csv(content_csv, index=False, encoding="utf-8-sig")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return pd.DataFrame(rows).drop_duplicates(subset=["link"], keep="last").reset_index(drop=True)


def command_collect(args: argparse.Namespace) -> None:
    session = build_session()
    df = collect_listing(
        session=session,
        start_offset=args.start_offset,
        end_offset=args.end_offset,
        step=args.step,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )
    ensure_parent(args.output_csv)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[collect] noticias salvas: {len(df)}")
    print(f"[collect] csv: {args.output_csv.resolve()}")


def command_extract(args: argparse.Namespace) -> None:
    manifest = extract_articles(
        session=build_session(),
        index_csv=args.index_csv,
        content_csv=args.output_csv,
        markdown_dir=args.markdown_dir,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        limit=args.limit,
        only_missing=args.only_missing,
    )
    print(f"[extract] registros no manifesto: {len(manifest)}")
    print(f"[extract] manifesto: {args.output_csv.resolve()}")
    print(f"[extract] markdown dir: {args.markdown_dir.resolve()}")


def command_sync(args: argparse.Namespace) -> None:
    session = build_session()
    df_index = collect_listing(
        session=session,
        start_offset=args.start_offset,
        end_offset=args.end_offset,
        step=args.step,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
    )
    ensure_parent(args.index_csv)
    df_index.to_csv(args.index_csv, index=False, encoding="utf-8-sig")
    print(f"[sync] indice atualizado com {len(df_index)} noticias em {args.index_csv.resolve()}")

    prepared_index = prepare_index_df(df_index)
    manifest = prepare_manifest_df(load_existing_manifest(args.content_csv))
    manifest_lookup = build_manifest_lookup(manifest)
    missing_count = count_missing_articles(prepared_index, manifest_lookup, args.markdown_dir)
    print(f"[sync] noticias pendentes de extracao: {missing_count}")

    updated_manifest = extract_articles(
        session=session,
        index_csv=args.index_csv,
        content_csv=args.content_csv,
        markdown_dir=args.markdown_dir,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        limit=args.limit,
        only_missing=True,
    )
    print(f"[sync] registros no manifesto: {len(updated_manifest)}")
    print(f"[sync] manifesto: {args.content_csv.resolve()}")
    print(f"[sync] markdown dir: {args.markdown_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coleta noticias de operacoes da PF e extrai o conteudo principal com docling."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Coleta o indice das noticias e gera um CSV.")
    collect_parser.add_argument("--start-offset", type=int, default=0)
    collect_parser.add_argument("--end-offset", type=int, default=None)
    collect_parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    collect_parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    collect_parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP)
    collect_parser.add_argument("--output-csv", type=Path, default=DEFAULT_INDEX_CSV)
    collect_parser.set_defaults(func=command_collect)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Abre cada link do CSV de indice, extrai o conteudo principal e salva em markdown.",
    )
    extract_parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    extract_parser.add_argument("--output-csv", type=Path, default=DEFAULT_CONTENT_CSV)
    extract_parser.add_argument("--markdown-dir", type=Path, default=DEFAULT_MARKDOWN_DIR)
    extract_parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    extract_parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP)
    extract_parser.add_argument("--limit", type=int, default=None)
    extract_parser.add_argument("--only-missing", action="store_true")
    extract_parser.set_defaults(func=command_extract)

    sync_parser = subparsers.add_parser(
        "sync",
        help="Atualiza o indice e extrai automaticamente apenas as noticias faltantes.",
    )
    sync_parser.add_argument("--start-offset", type=int, default=0)
    sync_parser.add_argument("--end-offset", type=int, default=None)
    sync_parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    sync_parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    sync_parser.add_argument("--content-csv", type=Path, default=DEFAULT_CONTENT_CSV)
    sync_parser.add_argument("--markdown-dir", type=Path, default=DEFAULT_MARKDOWN_DIR)
    sync_parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    sync_parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP)
    sync_parser.add_argument("--limit", type=int, default=None)
    sync_parser.set_defaults(func=command_sync)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
