from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    from project_config import CONTENT_CSV as DEFAULT_CONTENT_CSV, NEWS_MARKDOWN_DIR as DEFAULT_MARKDOWN_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from scripts.project_config import (
        CONTENT_CSV as DEFAULT_CONTENT_CSV,
        NEWS_MARKDOWN_DIR as DEFAULT_MARKDOWN_DIR,
        PROJECT_ROOT,
    )


COPY_PREFIX = "copy_of_"
HASH_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}\.md$", re.IGNORECASE)
NUMBERED_VARIANT_RE = re.compile(r"-\d+-[0-9a-f]{8}\.md$", re.IGNORECASE)


def resolve_markdown_path(markdown_path: object, markdown_dir: Path) -> Path | None:
    if not isinstance(markdown_path, str):
        return None

    cleaned = markdown_path.strip()
    if not cleaned:
        return None

    candidate = Path(cleaned)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def serialize_markdown_path(markdown_path: Path) -> str:
    resolved = markdown_path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_manifest(content_csv: Path) -> pd.DataFrame:
    if not content_csv.exists():
        raise FileNotFoundError(f"Manifesto nao encontrado: {content_csv}")

    manifest = pd.read_csv(content_csv)
    if "markdown_path" not in manifest.columns:
        raise ValueError("O manifesto precisa ter a coluna 'markdown_path'.")
    return manifest


def build_file_hash_groups(markdown_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for markdown_file in sorted(markdown_dir.glob("*.md")):
        digest = hashlib.md5(markdown_file.read_bytes()).hexdigest()
        groups[digest].append(markdown_file.resolve())
    return groups


def filename_score(path: Path, manifest_references: int) -> tuple[int, int, int, int, str]:
    name = path.name.lower()
    is_copy_name = 1 if name.startswith(COPY_PREFIX) else 0
    is_numbered_variant = 1 if NUMBERED_VARIANT_RE.search(name) else 0
    has_hash_suffix = 0 if HASH_SUFFIX_RE.search(name) else 1

    # Prefer files already referenced by more manifest rows, then the less suspicious names.
    return (
        -manifest_references,
        is_copy_name,
        is_numbered_variant,
        has_hash_suffix,
        len(name),
        name,
    )


def choose_canonical_path(paths: list[Path], manifest_reference_counts: Counter[Path]) -> Path:
    return min(paths, key=lambda path: filename_score(path, manifest_reference_counts[path]))


def main(
    content_csv: Path | str | None = None,
    markdown_dir: Path | str | None = None,
    dry_run: bool | None = None,
) -> None:
    content_csv_path = Path(content_csv or os.getenv("PF_CONTENT_CSV", "") or DEFAULT_CONTENT_CSV)
    markdown_dir_path = Path(markdown_dir or os.getenv("PF_MARKDOWN_DIR", "") or DEFAULT_MARKDOWN_DIR)
    if dry_run is None:
        dry_run = os.getenv("PF_DEDUPE_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}

    content_csv = content_csv_path.resolve() if not content_csv_path.is_absolute() else content_csv_path
    markdown_dir = markdown_dir_path.resolve() if not markdown_dir_path.is_absolute() else markdown_dir_path

    manifest = load_manifest(content_csv)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    resolved_paths = manifest["markdown_path"].map(lambda value: resolve_markdown_path(value, markdown_dir))
    manifest_reference_counts: Counter[Path] = Counter(path for path in resolved_paths if path is not None)

    hash_groups = build_file_hash_groups(markdown_dir)
    duplicate_groups = {digest: paths for digest, paths in hash_groups.items() if len(paths) > 1}

    print(f"[dedupe] grupos de conteudo duplicado: {len(duplicate_groups)}")
    if not duplicate_groups:
        print("[dedupe] nenhum arquivo duplicado por conteudo foi encontrado.")
        return

    manifest_updates = 0
    deleted_files: list[Path] = []

    for digest, paths in sorted(duplicate_groups.items(), key=lambda item: [path.name for path in item[1]]):
        canonical_path = choose_canonical_path(paths, manifest_reference_counts)
        duplicate_paths = [path for path in paths if path != canonical_path]

        print(f"[dedupe] hash={digest} canonico={canonical_path.name}")
        for duplicate_path in duplicate_paths:
            duplicate_mask = resolved_paths == duplicate_path
            duplicate_rows = int(duplicate_mask.sum())
            if duplicate_rows:
                manifest.loc[duplicate_mask, "markdown_path"] = serialize_markdown_path(canonical_path)
                resolved_paths.loc[duplicate_mask] = canonical_path
                manifest_reference_counts[canonical_path] += duplicate_rows
                manifest_reference_counts[duplicate_path] -= duplicate_rows
                manifest_updates += duplicate_rows

            print(
                f"[dedupe] removendo redundante: {duplicate_path.name} | "
                f"linhas_manifesto_atualizadas={duplicate_rows}"
            )

            if not dry_run and duplicate_path.exists():
                duplicate_path.unlink()
            deleted_files.append(duplicate_path)

    if dry_run:
        print(f"[dedupe] dry-run: {len(deleted_files)} arquivos seriam removidos.")
        print(f"[dedupe] dry-run: {manifest_updates} linhas do manifesto seriam atualizadas.")
        return

    manifest.to_csv(content_csv, index=False, encoding="utf-8-sig")
    print(f"[dedupe] arquivos removidos: {len(deleted_files)}")
    print(f"[dedupe] linhas do manifesto atualizadas: {manifest_updates}")
    print(f"[dedupe] manifesto salvo em: {content_csv}")


if __name__ == "__main__":
    main()
