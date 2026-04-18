from __future__ import annotations

import hashlib
import html
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

try:
    from project_config import (
        BASE_LIST_URL,
        CONTENT_CSV as DEFAULT_CONTENT_CSV,
        DEFAULT_HTTP_TIMEOUT_SECONDS as DEFAULT_TIMEOUT,
        DEFAULT_REQUEST_SLEEP_SECONDS as DEFAULT_SLEEP,
        DEFAULT_SCRAPE_STEP as DEFAULT_STEP,
        INDEX_CSV as DEFAULT_INDEX_CSV,
        NEWS_MARKDOWN_DIR as DEFAULT_MARKDOWN_DIR,
        PROJECT_ROOT,
    )
except ModuleNotFoundError:
    from scripts.project_config import (
        BASE_LIST_URL,
        CONTENT_CSV as DEFAULT_CONTENT_CSV,
        DEFAULT_HTTP_TIMEOUT_SECONDS as DEFAULT_TIMEOUT,
        DEFAULT_REQUEST_SLEEP_SECONDS as DEFAULT_SLEEP,
        DEFAULT_SCRAPE_STEP as DEFAULT_STEP,
        INDEX_CSV as DEFAULT_INDEX_CSV,
        NEWS_MARKDOWN_DIR as DEFAULT_MARKDOWN_DIR,
        PROJECT_ROOT,
    )

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
MARKDOWN_FULL_PUBLISHED_RE = re.compile(
    r"^Publicado em\s+(?P<published>\d{2}/\d{2}/\d{4}(?:\s+\d{2}h\d{2})?)$",
    re.IGNORECASE,
)
MARKDOWN_UPDATED_RE = re.compile(
    r"^Atualizado em\s+(?P<updated>\d{2}/\d{2}/\d{4}(?:\s+\d{2}h\d{2})?)$",
    re.IGNORECASE,
)
MARKDOWN_TAGS_RE = re.compile(r"^Tags:\s*(?P<tags>.+)$", re.IGNORECASE)
ARTICLE_DATES_RE = re.compile(
    r"Publicado em\s+(?P<published>\d{2}/\d{2}/\d{4}\s+\d{2}h\d{2})"
    r"(?:\s+Atualizado em\s+(?P<updated>\d{2}/\d{2}/\d{4}\s+\d{2}h\d{2}))?",
    re.IGNORECASE,
)
MARKDOWN_PUBLISHED_RE = re.compile(r"^Publicado em\s+(?P<date>\d{2}/\d{2}/\d{4})(?:\s+\d{2}h\d{2})?", re.IGNORECASE)
PAGE_OFFSET_RE = re.compile(r"b_start:int=(?P<offset>\d+)")


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


@dataclass
class ExistingNewsInventory:
    records: list["LocalMarkdownRecord"]
    known_links: set[str]
    known_paths: set[Path]
    known_title_date_keys: set[str]
    known_title_keys: set[str]


@dataclass(frozen=True)
class LocalMarkdownRecord:
    markdown_path: Path
    title: str
    subtitle: str
    published_at: str
    updated_at: str
    tags: list[str]


@dataclass
class InventoryComparisonState:
    remaining_title_date_keys: set[str]
    remaining_title_keys: set[str]


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def fold_to_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_news_key(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    normalized = fold_to_ascii(str(value)).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def build_title_date_key(title: object, published_date: object) -> str:
    title_key = normalize_news_key(title)
    date_key = normalize_news_key(published_date)
    if not title_key or not date_key:
        return ""
    return f"{title_key}|{date_key}"


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


def collect_listing_page(
    session: requests.Session,
    offset: int,
    step: int,
    timeout: int,
) -> tuple[pd.DataFrame, BeautifulSoup]:
    url = f"{BASE_LIST_URL}?b_start:int={offset}"
    soup = fetch_soup(session, url, timeout=timeout)
    articles = soup.select("article.tileItem")
    print(f"[collect] offset={offset} items={len(articles)}")

    rows = [parse_listing_item(article, offset=offset, step=step, item_index=item_index) for item_index, article in enumerate(articles, start=1)]
    return pd.DataFrame(rows), soup


def infer_last_offset(first_page_soup: BeautifulSoup) -> int:
    offsets = [
        int(match.group("offset"))
        for anchor in first_page_soup.select('a[href*="b_start:int="]')
        if anchor.has_attr("href")
        for match in [PAGE_OFFSET_RE.search(anchor["href"])]
        if match
    ]
    return max(offsets, default=0)


def infer_site_count(
    session: requests.Session,
    step: int,
    timeout: int,
) -> tuple[int, pd.DataFrame]:
    first_page_df, first_page_soup = collect_listing_page(session=session, offset=0, step=step, timeout=timeout)
    if first_page_df.empty:
        return 0, first_page_df

    last_offset = infer_last_offset(first_page_soup)
    if last_offset == 0:
        return len(first_page_df), first_page_df

    last_page_df, _ = collect_listing_page(session=session, offset=last_offset, step=step, timeout=timeout)
    site_count = last_offset + len(last_page_df)
    return site_count, first_page_df


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


def extract_markdown_inventory_entry(markdown_file: Path) -> tuple[str, str]:
    title = ""
    published_date = ""

    with markdown_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if not title and line.startswith("#"):
                title = line.lstrip("#").strip()
                continue
            if not published_date:
                published_match = MARKDOWN_PUBLISHED_RE.match(line)
                if published_match:
                    published_date = published_match.group("date")
            if title and published_date:
                break

    return title, published_date


def parse_local_markdown_record(markdown_file: Path) -> LocalMarkdownRecord:
    non_empty_lines = [
        line.strip()
        for line in markdown_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]

    title = ""
    subtitle = ""
    published_at = ""
    updated_at = ""
    tags: list[str] = []

    for line in non_empty_lines:
        if not title and line.startswith("#"):
            title = line.lstrip("#").strip()
            continue

        if not published_at:
            published_match = MARKDOWN_FULL_PUBLISHED_RE.match(line)
            if published_match:
                published_at = published_match.group("published")
                continue

        if not updated_at:
            updated_match = MARKDOWN_UPDATED_RE.match(line)
            if updated_match:
                updated_at = updated_match.group("updated")
                continue

        if not tags:
            tags_match = MARKDOWN_TAGS_RE.match(line)
            if tags_match:
                tags = [part.strip() for part in tags_match.group("tags").split(",") if part.strip()]
                continue

        if title and not subtitle and not line.startswith(("Publicado em", "Atualizado em", "Tags:")):
            subtitle = line

    return LocalMarkdownRecord(
        markdown_path=markdown_file.resolve(),
        title=title,
        subtitle=subtitle,
        published_at=published_at,
        updated_at=updated_at,
        tags=tags,
    )


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
        candidate = Path(markdown_path.strip())
        if candidate.is_absolute():
            return candidate
        return (PROJECT_ROOT / candidate).resolve()
    return markdown_dir / markdown_filename(link)


def serialize_markdown_path(markdown_path: Path) -> str:
    resolved = markdown_path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def build_existing_news_inventory(
    markdown_dir: Path,
    manifest_lookup: dict[str, dict[str, object]],
) -> ExistingNewsInventory:
    records: list[LocalMarkdownRecord] = []
    known_links: set[str] = set()
    known_paths: set[Path] = set()
    known_title_date_keys: set[str] = set()
    known_title_keys: set[str] = set()

    for markdown_file in sorted(markdown_dir.glob("*.md")):
        record = parse_local_markdown_record(markdown_file)
        records.append(record)
        known_paths.add(record.markdown_path)

        published_date = record.published_at.split()[0] if record.published_at else ""
        title_key = normalize_news_key(record.title)
        title_date_key = build_title_date_key(record.title, published_date)
        if title_key:
            known_title_keys.add(title_key)
        if title_date_key:
            known_title_date_keys.add(title_date_key)

    for link, record in manifest_lookup.items():
        if record.get("status") != "ok":
            continue

        markdown_path = resolve_markdown_path(record.get("markdown_path"), markdown_dir, link).resolve()
        if markdown_path.exists():
            known_links.add(link)
            known_paths.add(markdown_path)

    return ExistingNewsInventory(
        records=records,
        known_links=known_links,
        known_paths=known_paths,
        known_title_date_keys=known_title_date_keys,
        known_title_keys=known_title_keys,
    )


def register_article_in_inventory(
    inventory: ExistingNewsInventory,
    markdown_path: Path,
    link: str | None,
    title: object,
    published_date: object,
) -> None:
    inventory.known_paths.add(markdown_path.resolve())
    if link:
        inventory.known_links.add(link)

    title_key = normalize_news_key(title)
    title_date_key = build_title_date_key(title, published_date)
    if title_key:
        inventory.known_title_keys.add(title_key)
    if title_date_key:
        inventory.known_title_date_keys.add(title_date_key)


def build_inventory_comparison_state(inventory: ExistingNewsInventory) -> InventoryComparisonState:
    return InventoryComparisonState(
        remaining_title_date_keys=set(inventory.known_title_date_keys),
        remaining_title_keys=set(inventory.known_title_keys),
    )


def row_exists_in_inventory(
    row: pd.Series | dict[str, object],
    inventory: ExistingNewsInventory,
    comparison_state: InventoryComparisonState | None = None,
) -> bool:
    title = row["titulo"] if isinstance(row, pd.Series) else row.get("titulo")
    published_date = row["data_publicacao"] if isinstance(row, pd.Series) else row.get("data_publicacao")

    title_date_key = build_title_date_key(title, published_date)
    if title_date_key:
        if comparison_state and title_date_key in comparison_state.remaining_title_date_keys:
            comparison_state.remaining_title_date_keys.discard(title_date_key)
            return True
        if title_date_key in inventory.known_title_date_keys:
            return True
        return False

    title_key = normalize_news_key(title)
    if not title_key:
        return False

    if comparison_state and title_key in comparison_state.remaining_title_keys:
        comparison_state.remaining_title_keys.discard(title_key)
        return True
    return title_key in inventory.known_title_keys


def article_already_exists(
    row: pd.Series | dict[str, object],
    manifest_lookup: dict[str, dict[str, object]],
    inventory: ExistingNewsInventory,
    markdown_dir: Path,
) -> bool:
    link = normalize_link(row["link"] if isinstance(row, pd.Series) else row.get("link"))
    if not link:
        return False

    expected_path = (markdown_dir / markdown_filename(link)).resolve()
    if expected_path in inventory.known_paths or expected_path.exists():
        return True

    existing = manifest_lookup.get(link)
    if existing and existing.get("status") == "ok":
        markdown_path = resolve_markdown_path(existing.get("markdown_path"), markdown_dir, link).resolve()
        if markdown_path.exists():
            return True

    return row_exists_in_inventory(row=row, inventory=inventory)


INDEX_COLUMNS = [
    "offset",
    "page_number",
    "item_index",
    "categoria",
    "titulo",
    "subtitulo",
    "data_publicacao",
    "hora_publicacao",
    "publicado_em",
    "tipo_conteudo",
    "tags",
    "total_tags",
    "link",
]


def split_published_at(published_at: str) -> tuple[str | None, str | None]:
    cleaned = clean_text(published_at)
    if not cleaned:
        return None, None

    parts = cleaned.split()
    if len(parts) >= 2 and re.fullmatch(r"\d{2}/\d{2}/\d{4}", parts[0]) and re.fullmatch(r"\d{2}h\d{2}", parts[1]):
        return parts[0], parts[1]
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", parts[0]):
        return parts[0], None
    return cleaned, None


def ensure_index_columns(df_index: pd.DataFrame) -> pd.DataFrame:
    normalized = df_index.copy()
    for column in INDEX_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized.loc[:, INDEX_COLUMNS]


def load_or_create_index(index_csv: Path) -> pd.DataFrame:
    if index_csv.exists():
        return ensure_index_columns(pd.read_csv(index_csv))
    return pd.DataFrame(columns=INDEX_COLUMNS)


def build_index_keys(df_index: pd.DataFrame) -> tuple[set[str], set[str]]:
    title_date_keys: set[str] = set()
    title_keys: set[str] = set()

    for _, row in df_index.iterrows():
        title = row.get("titulo")
        published_date = row.get("data_publicacao")
        title_key = normalize_news_key(title)
        title_date_key = build_title_date_key(title, published_date)
        if title_key:
            title_keys.add(title_key)
        if title_date_key:
            title_date_keys.add(title_date_key)

    return title_date_keys, title_keys


def row_exists_in_index(
    title: object,
    published_date: object,
    known_title_date_keys: set[str],
    known_title_keys: set[str],
) -> bool:
    title_date_key = build_title_date_key(title, published_date)
    if title_date_key and title_date_key in known_title_date_keys:
        return True

    if normalize_news_key(published_date):
        return False

    title_key = normalize_news_key(title)
    return bool(title_key and title_key in known_title_keys)


def build_manifest_path_lookup(
    manifest: pd.DataFrame,
    markdown_dir: Path,
) -> dict[Path, dict[str, object]]:
    lookup: dict[Path, dict[str, object]] = {}
    for record in manifest.to_dict(orient="records"):
        link = normalize_link(record.get("link"))
        if not link:
            continue
        markdown_path = resolve_markdown_path(record.get("markdown_path"), markdown_dir, link).resolve()
        lookup[markdown_path] = record
    return lookup


def build_index_row_from_local_record(
    record: LocalMarkdownRecord,
    manifest_record: dict[str, object] | None,
) -> dict[str, object]:
    published_date, published_time = split_published_at(record.published_at)
    return {
        "offset": pd.NA,
        "page_number": pd.NA,
        "item_index": pd.NA,
        "categoria": pd.NA,
        "titulo": record.title or pd.NA,
        "subtitulo": record.subtitle or pd.NA,
        "data_publicacao": published_date or pd.NA,
        "hora_publicacao": published_time or pd.NA,
        "publicado_em": record.published_at or pd.NA,
        "tipo_conteudo": "Markdown local",
        "tags": " | ".join(record.tags) if record.tags else pd.NA,
        "total_tags": len(record.tags),
        "link": normalize_link(manifest_record.get("link")) if manifest_record else pd.NA,
    }


def append_missing_local_rows_to_index(
    df_index: pd.DataFrame,
    inventory: ExistingNewsInventory,
    manifest: pd.DataFrame,
    markdown_dir: Path,
) -> tuple[pd.DataFrame, int]:
    known_title_date_keys, known_title_keys = build_index_keys(df_index)
    manifest_by_path = build_manifest_path_lookup(manifest, markdown_dir)
    rows_to_append: list[dict[str, object]] = []

    for record in inventory.records:
        published_date = record.published_at.split()[0] if record.published_at else ""
        if row_exists_in_index(record.title, published_date, known_title_date_keys, known_title_keys):
            continue

        rows_to_append.append(
            build_index_row_from_local_record(
                record=record,
                manifest_record=manifest_by_path.get(record.markdown_path),
            )
        )

        title_key = normalize_news_key(record.title)
        title_date_key = build_title_date_key(record.title, published_date)
        if title_key:
            known_title_keys.add(title_key)
        if title_date_key:
            known_title_date_keys.add(title_date_key)

    if not rows_to_append:
        return df_index, 0

    appended = pd.concat([df_index, pd.DataFrame(rows_to_append)], ignore_index=True)
    return ensure_index_columns(appended), len(rows_to_append)


def collect_site_rows_missing_locally(
    df_site: pd.DataFrame,
    manifest_lookup: dict[str, dict[str, object]],
    inventory: ExistingNewsInventory,
    markdown_dir: Path,
) -> pd.DataFrame:
    missing_rows: list[dict[str, object]] = []
    for _, row in df_site.iterrows():
        if article_already_exists(row, manifest_lookup, inventory, markdown_dir):
            continue
        missing_rows.append(row.to_dict())
    return pd.DataFrame(missing_rows, columns=df_site.columns)


def append_site_rows_to_index(df_index: pd.DataFrame, df_site_rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df_site_rows.empty:
        return df_index, 0

    known_title_date_keys, known_title_keys = build_index_keys(df_index)
    rows_to_append: list[dict[str, object]] = []

    for _, row in df_site_rows.iterrows():
        if row_exists_in_index(row.get("titulo"), row.get("data_publicacao"), known_title_date_keys, known_title_keys):
            continue

        rows_to_append.append({column: row.get(column, pd.NA) for column in INDEX_COLUMNS})

        title_key = normalize_news_key(row.get("titulo"))
        title_date_key = build_title_date_key(row.get("titulo"), row.get("data_publicacao"))
        if title_key:
            known_title_keys.add(title_key)
        if title_date_key:
            known_title_date_keys.add(title_date_key)

    if not rows_to_append:
        return df_index, 0

    appended = pd.concat([df_index, pd.DataFrame(rows_to_append)], ignore_index=True)
    return ensure_index_columns(appended), len(rows_to_append)


def collect_recent_site_rows(
    session: requests.Session,
    pages_to_scan: int,
    step: int,
    timeout: int,
    sleep_seconds: float,
    first_page_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if pages_to_scan <= 0:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    frames: list[pd.DataFrame] = []
    for page_index in range(pages_to_scan):
        offset = page_index * step
        if page_index == 0 and first_page_df is not None:
            page_df = first_page_df.copy()
            print(f"[collect] reutilizando primeira pagina em memoria com {len(page_df)} itens")
        else:
            page_df, _ = collect_listing_page(session=session, offset=offset, step=step, timeout=timeout)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if page_df.empty:
            break
        frames.append(page_df)

    if not frames:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    merged = pd.concat(frames, ignore_index=True)
    return ensure_index_columns(prepare_index_df(merged))


def find_missing_site_rows_progressively(
    session: requests.Session,
    missing_count_estimate: int,
    manifest_lookup: dict[str, dict[str, object]],
    inventory: ExistingNewsInventory,
    markdown_dir: Path,
    step: int,
    timeout: int,
    sleep_seconds: float,
    first_page_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if missing_count_estimate <= 0:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    target_count = max(1, missing_count_estimate)
    comparison_state = build_inventory_comparison_state(inventory)
    page_index = 0
    found_rows: list[dict[str, object]] = []
    found_links: set[str] = set()

    while len(found_rows) < target_count:
        offset = page_index * step
        if page_index == 0 and first_page_df is not None:
            page_df = first_page_df.copy()
            print(f"[collect] reutilizando primeira pagina em memoria com {len(page_df)} itens")
        else:
            page_df, _ = collect_listing_page(session=session, offset=offset, step=step, timeout=timeout)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if page_df.empty:
            break

        new_rows_on_page = 0
        for _, row in page_df.iterrows():
            link = normalize_link(row.get("link"))
            if not link or link in found_links:
                continue

            exists = article_already_exists(
                row=row,
                manifest_lookup=manifest_lookup,
                inventory=inventory,
                markdown_dir=markdown_dir,
            )
            if not exists:
                found_rows.append({column: row.get(column, pd.NA) for column in INDEX_COLUMNS})
                found_links.add(link)
                new_rows_on_page += 1
                continue

            row_exists_in_inventory(
                row=row,
                inventory=inventory,
                comparison_state=comparison_state,
            )

        print(
            f"[sync] pagina {(page_index + 1)} verificada | "
            f"novos_titulos_encontrados={new_rows_on_page} | acumulado={len(found_rows)} | "
            f"titulos_pendentes={len(comparison_state.remaining_title_date_keys)}"
        )

        if len(page_df) < step:
            break

        page_index += 1

    if not found_rows:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    return pd.DataFrame(found_rows, columns=INDEX_COLUMNS)


def save_manifest(content_csv: Path, manifest: pd.DataFrame) -> None:
    ensure_parent(content_csv)
    manifest.to_csv(content_csv, index=False, encoding="utf-8-sig")


def download_new_site_articles(
    session: requests.Session,
    df_site_rows: pd.DataFrame,
    manifest: pd.DataFrame,
    content_csv: Path,
    markdown_dir: Path,
    timeout: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    if df_site_rows.empty:
        return manifest

    converter = DocumentConverter()
    markdown_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = manifest.to_dict(orient="records")

    for _, row in df_site_rows.iterrows():
        link = normalize_link(row.get("link"))
        if not link:
            continue

        output_file = markdown_dir / markdown_filename(link)
        print(f"[sync] baixando: {link}")

        try:
            soup = fetch_soup(session, link, timeout=timeout)
            article = parse_article_core(soup)
            export_markdown(converter, article, output_file)
            rows.append(
                {
                    "link": link,
                    "markdown_path": serialize_markdown_path(output_file),
                    "status": "ok",
                    "titulo_extraido": article.title,
                    "subtitulo_extraido": article.subtitle,
                    "publicado_em_extraido": article.published_at,
                    "atualizado_em_extraido": article.updated_at,
                    "tags_extraidas": " | ".join(article.tags),
                    "erro": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "link": link,
                    "markdown_path": serialize_markdown_path(output_file),
                    "status": "erro",
                    "titulo_extraido": None,
                    "subtitulo_extraido": None,
                    "publicado_em_extraido": None,
                    "atualizado_em_extraido": None,
                    "tags_extraidas": None,
                    "erro": str(exc),
                }
            )

        manifest = pd.DataFrame(rows).drop_duplicates(subset=["link"], keep="last").reset_index(drop=True)
        save_manifest(content_csv, manifest)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return manifest


def main() -> None:
    if sys.argv[1:]:
        ignored = " ".join(sys.argv[1:])
        print(f"[sync] ignorando argumentos extras e usando o fluxo padrao: {ignored}")

    index_csv = DEFAULT_INDEX_CSV
    content_csv = DEFAULT_CONTENT_CSV
    markdown_dir = DEFAULT_MARKDOWN_DIR

    session = build_session()
    site_count, first_page_df = infer_site_count(
        session=session,
        step=DEFAULT_STEP,
        timeout=DEFAULT_TIMEOUT,
    )
    first_page_df = ensure_index_columns(first_page_df)

    df_index = load_or_create_index(index_csv)
    manifest = prepare_manifest_df(load_existing_manifest(content_csv))
    manifest_lookup = build_manifest_lookup(manifest)
    inventory = build_existing_news_inventory(markdown_dir, manifest_lookup)

    local_count = len(inventory.records)
    print(f"[sync] noticias no site: {site_count}")
    print(f"[sync] markdowns locais: {local_count}")

    site_rows_to_download = pd.DataFrame(columns=INDEX_COLUMNS)
    site_rows_appended = 0
    if site_count > local_count:
        missing_count_estimate = site_count - local_count
        pages_to_scan = max(1, math.ceil(missing_count_estimate / DEFAULT_STEP))
        print(f"[sync] diferenca estimada: {missing_count_estimate}")
        print(f"[sync] paginas iniciais a verificar: {pages_to_scan}")

        recent_site_rows = collect_recent_site_rows(
            session=session,
            pages_to_scan=pages_to_scan,
            step=DEFAULT_STEP,
            timeout=DEFAULT_TIMEOUT,
            sleep_seconds=DEFAULT_SLEEP,
            first_page_df=first_page_df,
        )
        site_rows_to_download = collect_site_rows_missing_locally(
            recent_site_rows,
            manifest_lookup,
            inventory,
            markdown_dir,
        )
        if site_rows_to_download.empty:
            print("[sync] nenhum titulo novo encontrado na janela inicial. Vou continuar a busca nas paginas seguintes.")
            site_rows_to_download = find_missing_site_rows_progressively(
                session=session,
                missing_count_estimate=missing_count_estimate,
                manifest_lookup=manifest_lookup,
                inventory=inventory,
                markdown_dir=markdown_dir,
                step=DEFAULT_STEP,
                timeout=DEFAULT_TIMEOUT,
                sleep_seconds=DEFAULT_SLEEP,
                first_page_df=first_page_df,
            )
        df_index, site_rows_appended = append_site_rows_to_index(df_index, site_rows_to_download)
        print(f"[sync] novas noticias identificadas no site: {len(site_rows_to_download)}")
        if not site_rows_to_download.empty:
            manifest = download_new_site_articles(
                session=session,
                df_site_rows=site_rows_to_download,
                manifest=manifest,
                content_csv=content_csv,
                markdown_dir=markdown_dir,
                timeout=DEFAULT_TIMEOUT,
                sleep_seconds=DEFAULT_SLEEP,
            )
            manifest_lookup = build_manifest_lookup(manifest)
            inventory = build_existing_news_inventory(markdown_dir, manifest_lookup)
    else:
        print("[sync] nenhuma coleta nova sera feita porque a contagem do site nao supera a contagem local.")

    df_index, local_rows_appended = append_missing_local_rows_to_index(df_index, inventory, manifest, markdown_dir)
    df_index = ensure_index_columns(df_index)

    ensure_parent(index_csv)
    df_index.to_csv(index_csv, index=False, encoding="utf-8-sig")
    save_manifest(content_csv, manifest)

    print(f"[sync] linhas novas anexadas ao indice a partir do site: {site_rows_appended}")
    print(f"[sync] linhas novas anexadas ao indice a partir dos markdowns locais: {local_rows_appended}")
    print(f"[sync] indice salvo em: {index_csv.resolve()}")
    print(f"[sync] manifesto salvo em: {content_csv.resolve()}")
    print(f"[sync] markdown dir: {markdown_dir.resolve()}")


if __name__ == "__main__":
    main()
