from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.text.run import Run


IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
}


def iter_block_items(parent: DocumentType | _Cell) -> Iterable[Paragraph | Table]:
    parent_elm = parent.element.body if isinstance(parent, DocumentType) else parent._tc
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def escape_markdown_table_cell(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return text.replace("|", r"\|")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = fix_mojibake(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fix_mojibake(text: str) -> str:
    if not any(marker in text for marker in ("Ã", "Â", "â€", "â€“", "â€”")):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text
    original_markers = sum(text.count(marker) for marker in ("Ã", "Â", "â€", "â€“", "â€”"))
    repaired_markers = sum(repaired.count(marker) for marker in ("Ã", "Â", "â€", "â€“", "â€”"))
    return repaired if repaired_markers < original_markers else text


def markdown_emphasis(text: str, run: Run) -> str:
    if not text:
        return ""
    if run.bold:
        text = f"**{text}**"
    if run.italic:
        text = f"*{text}*"
    return text


def save_run_images(run_elm, doc: DocumentType, media_dir: Path, image_counter: list[int]) -> list[str]:
    image_lines: list[str] = []
    for blip in run_elm.xpath(".//a:blip"):
        rel_id = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
        if not rel_id or rel_id not in doc.part.related_parts:
            continue
        part = doc.part.related_parts[rel_id]
        content_type = getattr(part, "content_type", "")
        ext = IMAGE_EXTENSIONS.get(content_type, Path(getattr(part, "partname", "")).suffix or ".bin")
        image_counter[0] += 1
        filename = f"image-{image_counter[0]:03d}{ext}"
        out_path = media_dir / filename
        out_path.write_bytes(part.blob)
        image_lines.append(f"![Imagem extraida](media/{filename})")
    return image_lines


def paragraph_text_and_images(paragraph: Paragraph, doc: DocumentType, media_dir: Path, image_counter: list[int]) -> tuple[str, list[str]]:
    pieces: list[str] = []
    image_lines: list[str] = []

    for child in paragraph._p.iterchildren():
        if child.tag == qn("w:r"):
            run = Run(child, paragraph)
            pieces.append(markdown_emphasis(run.text, run))
            image_lines.extend(save_run_images(child, doc, media_dir, image_counter))
        elif child.tag == qn("w:hyperlink"):
            rel_id = child.get(qn("r:id"))
            href = None
            if rel_id and rel_id in paragraph.part.rels:
                href = paragraph.part.rels[rel_id].target_ref
            link_text = clean_text("".join(node.text or "" for node in child.xpath(".//w:t")))
            if link_text and href:
                pieces.append(f"[{link_text}]({href})")
            elif link_text:
                pieces.append(link_text)

    return clean_text("".join(pieces)), image_lines


def paragraph_prefix(paragraph: Paragraph) -> str:
    style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
    ppr = paragraph._p.pPr
    num_pr = ppr.numPr if ppr is not None else None
    ilvl = 0
    if num_pr is not None and num_pr.ilvl is not None and num_pr.ilvl.val is not None:
        ilvl = int(num_pr.ilvl.val)
    indent = "  " * ilvl

    if "heading" in style or "titulo" in style or "título" in style:
        match = re.search(r"(\d+)", style)
        level = min(int(match.group(1)), 6) if match else 1
        return "#" * level + " "

    if num_pr is not None or "list" in style or "lista" in style:
        if "number" in style or "numero" in style or "número" in style:
            return f"{indent}1. "
        return f"{indent}- "

    return ""


def table_to_markdown(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            cell_text = "<br>".join(clean_text(p.text) for p in cell.paragraphs if clean_text(p.text))
            cells.append(escape_markdown_table_cell(cell_text))
        rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator, *body])


def convert_docx_to_markdown(input_docx: Path, output_md: Path, media_dir: Path) -> None:
    media_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(str(input_docx))
    image_counter = [0]
    lines: list[str] = [
        f"<!-- Fonte: {input_docx.name} -->",
        "<!-- Markdown gerado automaticamente por scripts/docx_to_markdown.py. -->",
        "",
    ]

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text, image_lines = paragraph_text_and_images(block, doc, media_dir, image_counter)
            prefix = paragraph_prefix(block)
            if text:
                lines.append(prefix + text)
                lines.append("")
            for image_line in image_lines:
                lines.append(image_line)
                lines.append("")
        elif isinstance(block, Table):
            table_md = table_to_markdown(block)
            if table_md:
                lines.append(table_md)
                lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a DOCX file to auditable Markdown.")
    parser.add_argument("input_docx", type=Path)
    parser.add_argument("output_md", type=Path)
    parser.add_argument("--media-dir", type=Path, required=True)
    args = parser.parse_args()

    convert_docx_to_markdown(args.input_docx, args.output_md, args.media_dir)


if __name__ == "__main__":
    main()
