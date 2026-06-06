from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from rf_research.data.catalog import DatasetSpec


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - dataset publishers provide MD5 checksums
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "real-rf-dg/0.1 (academic reproducibility)"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        if offset and response.status_code != 206:
            offset = 0
            partial.unlink(missing_ok=True)
        response.raise_for_status()
        mode = "ab" if offset else "wb"
        total = expected_size or int(response.headers.get("content-length", 0)) + offset
        with (
            partial.open(mode) as handle,
            tqdm(
                total=total,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    partial.replace(destination)


def _safe_extract(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (output_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive path in {archive}: {member.filename}")
        bundle.extractall(output_dir)


def fetch_dataset(spec: DatasetSpec, root: Path, extract: bool = True) -> dict:
    archive_dir = root / "archives" / spec.key
    extract_dir = root / "extracted" / spec.key
    required = sum(item.size for item in spec.archives)
    free = shutil.disk_usage(root.parent if root.parent.exists() else Path.cwd()).free
    if free < required * 1.2:
        raise OSError(
            f"Insufficient free disk for {spec.key}: need at least "
            f"{required * 1.2 / 2**30:.1f} GiB"
        )

    archive_records = []
    for item in spec.archives:
        archive = archive_dir / item.filename
        if not archive.exists() or archive.stat().st_size != item.size:
            _download(item.url, archive, item.size)
        actual_md5 = md5sum(archive)
        if actual_md5 != item.md5:
            archive.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {archive.name}: {actual_md5} != {item.md5}"
            )
        if extract:
            marker = extract_dir / f".{item.filename}.extracted"
            if not marker.exists():
                _safe_extract(archive, extract_dir)
                marker.write_text(actual_md5 + "\n", encoding="ascii")
        archive_records.append(
            {
                "filename": item.filename,
                "url": item.url,
                "size": item.size,
                "md5": actual_md5,
            }
        )

    metadata = {
        "key": spec.key,
        "title": spec.title,
        "institution": spec.institution,
        "doi": spec.doi,
        "license": spec.license,
        "citation": spec.citation,
        "archives": archive_records,
    }
    metadata_path = root / "extracted" / spec.key / "dataset_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata

