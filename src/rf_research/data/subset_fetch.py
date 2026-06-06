from __future__ import annotations

import binascii
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from rf_research.data.catalog import DatasetSpec
from rf_research.data.manifest import _select_fraction
from rf_research.data.remote_zip import RemoteZip, RemoteZipEntry


def _safe_destination(root: Path, member: str) -> Path:
    member_path = PurePosixPath(member)
    if (
        member_path.is_absolute()
        or ".." in member_path.parts
        or any("\\" in part or ":" in part for part in member_path.parts)
    ):
        raise ValueError(f"Unsafe remote ZIP member: {member}")
    root_path = os.path.abspath(root)
    destination = os.path.abspath(os.path.join(root_path, *member_path.parts))
    if os.path.commonpath([root_path, destination]) != root_path:
        raise ValueError(f"Remote ZIP member escapes output directory: {member}")
    return Path(destination)


def _wideft_rows(entries: list[RemoteZipEntry]) -> list[dict]:
    rows = []
    for entry in entries:
        if not entry.name.lower().endswith(".sc16"):
            continue
        parts = PurePosixPath(entry.name).parts
        if len(parts) < 5:
            continue
        protocol, manufacturer, device = parts[-4], parts[-3], parts[-2]
        rows.append(
            {
                "source": "wideft",
                "record_id": entry.name,
                "label_name": f"{manufacturer}/{device}",
                "domain": protocol,
                "_entry": entry,
            }
        )
    return rows


def _wlan_rows(
    archives: list[tuple[str, RemoteZip, str]],
) -> list[dict]:
    rows = []
    for archive_name, remote, domain in archives:
        for entry in remote.entries:
            if not entry.name.lower().endswith(".npz"):
                continue
            label = PurePosixPath(entry.name).name.split("_", 1)[0]
            rows.append(
                {
                    "source": "wlan",
                    "record_id": f"{archive_name}:{entry.name}",
                    "label_name": label,
                    "domain": domain,
                    "_entry": entry,
                    "_archive": archive_name,
                }
            )
    return rows


def _record_without_internal(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _existing_result(
    destination: Path,
    expected_size: int,
    expected_crc32: int | None = None,
) -> dict[str, str | int] | None:
    if not destination.exists() or destination.stat().st_size != expected_size:
        return None
    sha256 = hashlib.sha256()
    crc = 0
    with destination.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            sha256.update(chunk)
            crc = binascii.crc32(chunk, crc)
    crc &= 0xFFFFFFFF
    if expected_crc32 is not None and crc != expected_crc32:
        destination.unlink()
        return None
    return {
        "downloaded_compressed_bytes": 0,
        "saved_uncompressed_bytes": expected_size,
        "sha256": sha256.hexdigest(),
        "crc32": f"{crc:08x}",
        "complete_entry": int(expected_crc32 is not None),
        "reused_existing": 1,
    }


def _fetch_complete_entries(
    selected: list[dict],
    remotes: dict[str, RemoteZip],
    output_dir: Path,
    download_workers: int,
) -> list[dict]:
    def fetch_one(row: dict) -> dict:
        entry = row["_entry"]
        archive_name = row.get("_archive", next(iter(remotes)))
        destination = _safe_destination(output_dir, entry.name)
        result = _existing_result(
            destination, entry.uncompressed_size, entry.crc32
        )
        if result is None:
            result = remotes[archive_name].extract(entry, destination)
            result["reused_existing"] = 0
        result.setdefault("entry", entry.name)
        result.setdefault("remote_compressed_size", entry.compressed_size)
        result.setdefault("remote_uncompressed_size", entry.uncompressed_size)
        return {
            **_record_without_internal(row),
            "archive": archive_name,
            "local_path": str(destination.resolve()),
            **result,
        }

    workers = max(1, download_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(fetch_one, selected),
                total=len(selected),
                desc="Selective real-signal extraction",
                unit="file",
            )
        )


def _fetch_pla_prefixes(
    remote: RemoteZip,
    output_dir: Path,
    fraction: float,
    window_size: int,
) -> tuple[list[dict], int]:
    entries = [
        entry for entry in remote.entries if entry.name.lower().endswith(".bin")
    ]
    records = []
    sample_bytes = 8
    alignment = window_size * sample_bytes
    for entry in tqdm(entries, desc="Streaming PLA prefixes", unit="device"):
        requested = int(math.floor(entry.uncompressed_size * fraction))
        requested = max(alignment, requested - requested % alignment)
        requested = min(requested, entry.uncompressed_size)
        destination = _safe_destination(output_dir, entry.name)
        complete = requested == entry.uncompressed_size
        result = _existing_result(
            destination,
            requested,
            entry.crc32 if complete else None,
        )
        if result is None:
            estimated_compressed = math.ceil(
                entry.compressed_size
                * (requested / entry.uncompressed_size)
                * 1.02
            )
            result = remote.extract_prefix(
                entry,
                destination,
                uncompressed_bytes=requested,
                chunk_size=256 * 1024,
                first_chunk_size=max(256 * 1024, estimated_compressed),
                verify_complete=complete,
            )
            result["reused_existing"] = 0
        result.setdefault("entry", entry.name)
        result.setdefault("remote_compressed_size", entry.compressed_size)
        result.setdefault("remote_uncompressed_size", entry.uncompressed_size)
        label = PurePosixPath(entry.name).parent.name
        records.append(
            {
                "source": "pla",
                "record_id": entry.name,
                "label_name": label,
                "domain": "controlled",
                "archive": "PLA_dataset.zip",
                "local_path": str(destination.resolve()),
                **result,
            }
        )
    return records, len(entries)


def fetch_dataset_subset(
    spec: DatasetSpec,
    root: Path,
    fraction: float,
    seed: int,
    window_size: int,
    download_workers: int = 8,
) -> dict:
    if not 0 < fraction <= 1:
        raise ValueError("subset_fraction must be in (0, 1]")
    output_dir = root / "extracted" / spec.key
    output_dir.mkdir(parents=True, exist_ok=True)

    remotes = {
        archive.filename: RemoteZip(archive.url) for archive in spec.archives
    }
    total_records: int
    source_entries: list[RemoteZipEntry]
    selection_unit: str
    if spec.key == "wideft":
        rows = _wideft_rows(remotes["WIDEFT.zip"].entries)
        selected = _select_fraction(rows, fraction, seed)
        records = _fetch_complete_entries(
            selected, remotes, output_dir, download_workers
        )
        total_records = len(rows)
        source_entries = [row["_entry"] for row in rows]
        selection_unit = "complete_capture_records"
    elif spec.key == "wlan":
        archive_domains = [
            (
                archive.filename,
                remotes[archive.filename],
                "office_room"
                if "office" in archive.filename
                else "anechoic_chamber",
            )
            for archive in spec.archives
        ]
        rows = _wlan_rows(archive_domains)
        selected = _select_fraction(rows, fraction, seed)
        records = _fetch_complete_entries(
            selected, remotes, output_dir, download_workers
        )
        total_records = len(rows)
        source_entries = [row["_entry"] for row in rows]
        selection_unit = "complete_capture_records"
    elif spec.key == "pla":
        records, total_records = _fetch_pla_prefixes(
            remotes["PLA_dataset.zip"], output_dir, fraction, window_size
        )
        source_entries = [
            entry
            for entry in remotes["PLA_dataset.zip"].entries
            if entry.name.lower().endswith(".bin")
        ]
        selection_unit = "aligned_uncompressed_bytes_per_device"
    else:
        raise KeyError(f"Subset acquisition is not implemented for {spec.key}")

    downloaded = sum(int(record["downloaded_compressed_bytes"]) for record in records)
    selected_compressed = sum(
        int(record["remote_compressed_size"]) for record in records
    )
    saved = sum(int(record["saved_uncompressed_bytes"]) for record in records)
    remote_compressed = sum(entry.compressed_size for entry in source_entries)
    remote_uncompressed = sum(entry.uncompressed_size for entry in source_entries)
    metadata = {
        "mode": "subset",
        "source": spec.key,
        "title": spec.title,
        "institution": spec.institution,
        "doi": spec.doi,
        "license": spec.license,
        "citation": spec.citation,
        "seed": seed,
        "subset_fraction": fraction,
        "selection_unit": selection_unit,
        "remote_record_count": total_records,
        "selected_record_count": len(records),
        "effective_record_fraction": len(records) / total_records,
        "remote_compressed_signal_bytes": remote_compressed,
        "remote_uncompressed_signal_bytes": remote_uncompressed,
        "downloaded_compressed_bytes": downloaded,
        "selected_remote_compressed_bytes": selected_compressed,
        "saved_uncompressed_bytes": saved,
        "effective_uncompressed_fraction": saved / remote_uncompressed,
        "archives": [
            {
                "filename": archive.filename,
                "url": archive.url,
                "publisher_md5": archive.md5,
                "publisher_size": archive.size,
            }
            for archive in spec.archives
        ],
        "records": records,
    }
    metadata_path = output_dir / "subset_acquisition.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
