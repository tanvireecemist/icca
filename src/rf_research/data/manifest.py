from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from rf_research.data.catalog import DATASETS

MANIFEST_COLUMNS = [
    "source",
    "path",
    "record_id",
    "label_name",
    "label",
    "domain",
    "offset",
    "n_samples",
    "dtype",
    "scale",
    "split",
]


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _wideft_records(root: Path, window_size: int) -> Iterable[dict]:
    base = root / "extracted" / "wideft"
    for path in sorted(base.rglob("*.sc16")):
        relative = path.relative_to(base)
        parts = relative.parts
        if len(parts) < 5:
            continue
        protocol, manufacturer, device = parts[-4], parts[-3], parts[-2]
        yield {
            "source": "wideft",
            "path": str(path.resolve()),
            "record_id": relative.as_posix(),
            "label_name": f"{manufacturer}/{device}",
            "domain": protocol,
            "offset": 0,
            "n_samples": window_size,
            "dtype": "sc16",
            "scale": 32768.0,
        }


def _pla_records(root: Path, window_size: int, stride: int) -> Iterable[dict]:
    base = root / "extracted" / "pla"
    for path in sorted(base.rglob("*.bin")):
        relative = path.relative_to(base)
        label_name = path.parent.name
        total_samples = path.stat().st_size // 8
        if total_samples < window_size:
            continue
        for offset in range(0, total_samples - window_size + 1, stride):
            yield {
                "source": "pla",
                "path": str(path.resolve()),
                "record_id": f"{relative.as_posix()}@{offset}",
                "label_name": label_name,
                "domain": "controlled",
                "offset": offset,
                "n_samples": window_size,
                "dtype": "complex64",
                "scale": 1.0,
            }


def _wlan_records(root: Path, window_size: int) -> Iterable[dict]:
    base = root / "extracted" / "wlan"
    for path in sorted(base.rglob("*.npz")):
        relative = path.relative_to(base)
        name = path.name
        label_name = name.split("_", 1)[0]
        domain = "office_room" if "office_room" in name else "anechoic_chamber"
        yield {
            "source": "wlan",
            "path": str(path.resolve()),
            "record_id": relative.as_posix(),
            "label_name": label_name,
            "domain": domain,
            "offset": 0,
            "n_samples": window_size,
            "dtype": "npz",
            "scale": 1.0,
        }


def discover_records(
    root: Path,
    sources: list[str],
    window_size: int,
    stride: int,
) -> list[dict]:
    builders = {
        "wideft": _wideft_records,
        "pla": _pla_records,
        "wlan": _wlan_records,
    }
    records: list[dict] = []
    for source in sources:
        if source == "pla":
            rows = builders[source](root, window_size, stride)
        else:
            rows = builders[source](root, window_size)
        records.extend(rows)
    return records


def _select_fraction(records: list[dict], fraction: float, seed: int) -> list[dict]:
    if not 0 < fraction <= 1:
        raise ValueError("subset_fraction must be in (0, 1]")
    source_groups: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        source_groups[row["source"]][
            (row["label_name"], row["domain"])
        ].append(row)

    selected: list[dict] = []
    for source, strata in source_groups.items():
        total = sum(len(group) for group in strata.values())
        target = max(len(strata), int(math.floor(total * fraction + 0.5)))
        allocation: dict[tuple[str, str], int] = {}
        remainder_order: list[tuple[float, tuple[str, str]]] = []
        for key, group in strata.items():
            exact = len(group) * fraction
            allocation[key] = max(1, int(math.floor(exact)))
            remainder_order.append((exact - math.floor(exact), key))
        remaining = target - sum(allocation.values())
        if remaining < 0:
            for _, key in sorted(
                remainder_order,
                key=lambda item: (
                    item[0],
                    _stable_hash(f"{source}|{item[1]}", seed),
                ),
            ):
                while remaining < 0 and allocation[key] > 1:
                    allocation[key] -= 1
                    remaining += 1
                if remaining == 0:
                    break
        for _, key in sorted(
            remainder_order,
            key=lambda item: (
                -item[0],
                _stable_hash(f"{source}|{item[1]}", seed),
            ),
        ):
            if remaining <= 0:
                break
            allocation[key] += 1
            remaining -= 1
        if remaining != 0:
            raise RuntimeError(f"Unable to allocate subset quota for {source}")
        for key, group in strata.items():
            ordered = sorted(
                group, key=lambda row: _stable_hash(row["record_id"], seed)
            )
            selected.extend(ordered[: allocation[key]])
    return selected


def _allocation(count: int, train_fraction: float, val_fraction: float) -> tuple[int, int]:
    if count < 3:
        return max(1, count - 1), 0
    train = max(1, round(count * train_fraction))
    val = max(1, round(count * val_fraction))
    if train + val >= count:
        train = max(1, count - 2)
        val = 1
    return train, val


def _assign_splits(
    rows: list[dict],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["label_name"])].append(row)

    for (source, _), group in groups.items():
        if source == "wlan":
            train_domain = [row for row in group if row["domain"] == "anechoic_chamber"]
            test_domain = [row for row in group if row["domain"] == "office_room"]
            ordered = sorted(
                train_domain, key=lambda row: _stable_hash(row["record_id"], seed + 1)
            )
            train_count, _ = _allocation(len(ordered), 0.85, 0.15)
            for index, row in enumerate(ordered):
                row["split"] = "train" if index < train_count else "val"
            for row in test_domain:
                row["split"] = "test"
            continue

        if source == "pla":
            ordered = sorted(group, key=lambda row: (row["path"], row["offset"]))
        else:
            ordered = sorted(
                group, key=lambda row: _stable_hash(row["record_id"], seed + 2)
            )
        train_count, val_count = _allocation(
            len(ordered), train_fraction, val_fraction
        )
        for index, row in enumerate(ordered):
            if index < train_count:
                row["split"] = "train"
            elif index < train_count + val_count:
                row["split"] = "val"
            else:
                row["split"] = "test"


def build_manifest(config: dict) -> tuple[Path, Path]:
    data = config["data"]
    project = config["project"]
    root = Path(data["root"])
    sources = list(data["sources"])
    for source in sources:
        if source not in DATASETS:
            raise KeyError(f"Unknown source: {source}")

    records = discover_records(
        root=root,
        sources=sources,
        window_size=int(data["window_size"]),
        stride=int(data["window_stride"]),
    )

    acquisition_mode = data.get("acquisition_mode", "full")
    acquisition = {}
    if acquisition_mode == "subset":
        missing_provenance = [
            source
            for source in sources
            if not (root / "extracted" / source / "subset_acquisition.json").exists()
        ]
        if missing_provenance:
            raise FileNotFoundError(
                "Subset acquisition provenance is missing for: "
                + ", ".join(missing_provenance)
                + ". Run `rfbench fetch` with acquisition_mode=subset."
            )
        allowed_paths: set[str] = set()
        for source in sources:
            provenance = root / "extracted" / source / "subset_acquisition.json"
            source_metadata = json.loads(provenance.read_text(encoding="utf-8"))
            acquisition[source] = source_metadata
            allowed_paths.update(
                str(Path(record["local_path"]).resolve())
                for record in source_metadata["records"]
            )
        records = [
            row for row in records if str(Path(row["path"]).resolve()) in allowed_paths
        ]
        selected = records
    elif acquisition_mode == "full":
        selected = _select_fraction(
            records,
            fraction=float(data["subset_fraction"]),
            seed=int(project["seed"]),
        )
    else:
        raise ValueError(f"Unknown acquisition_mode: {acquisition_mode}")

    missing = [
        source
        for source in sources
        if not any(row["source"] == source for row in selected)
    ]
    if missing:
        raise FileNotFoundError(
            "No selected records found for: "
            + ", ".join(missing)
            + ". Run `rfbench fetch` first."
        )
    _assign_splits(
        selected,
        seed=int(project["seed"]),
        train_fraction=float(data["train_fraction"]),
        val_fraction=float(data["val_fraction"]),
    )

    label_maps: dict[str, dict[str, int]] = {}
    for source in sources:
        names = sorted({row["label_name"] for row in selected if row["source"] == source})
        label_maps[source] = {name: index for index, name in enumerate(names)}
    for row in selected:
        row["label"] = label_maps[row["source"]][row["label_name"]]

    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{project['run_name']}_{int(float(data['subset_fraction']) * 100):02d}pct"
    manifest_path = manifest_dir / f"{stem}.csv"
    metadata_path = manifest_dir / f"{stem}.json"

    frame = pd.DataFrame(selected)[MANIFEST_COLUMNS]
    frame.sort_values(["source", "label", "split", "record_id"], inplace=True)
    frame.to_csv(manifest_path, index=False)

    discovered_counts = pd.DataFrame(records).groupby("source").size().to_dict()
    if acquisition_mode == "full":
        for source in sources:
            provenance = root / "extracted" / source / "subset_acquisition.json"
            if provenance.exists():
                acquisition[source] = json.loads(
                    provenance.read_text(encoding="utf-8")
                )
    selected_counts = frame.groupby(["source", "split"]).size().to_dict()
    metadata = {
        "seed": int(project["seed"]),
        "acquisition_mode": acquisition_mode,
        "subset_fraction": float(data["subset_fraction"]),
        "selection_rule": (
            "exact round-half-up source quota with largest-remainder "
            "source/label/domain stratification after SHA-256 ordering"
        ),
        "window_size": int(data["window_size"]),
        "window_stride": int(data["window_stride"]),
        "manifest": str(manifest_path.resolve()),
        "discovered_records": discovered_counts,
        "selected_records": {
            f"{source}/{split}": int(count)
            for (source, split), count in selected_counts.items()
        },
        "label_maps": label_maps,
        "subset_acquisition": acquisition,
        "datasets": {
            source: {
                "title": DATASETS[source].title,
                "doi": DATASETS[source].doi,
                "license": DATASETS[source].license,
                "archives": [
                    {"url": item.url, "md5": item.md5, "size": item.size}
                    for item in DATASETS[source].archives
                ],
            }
            for source in sources
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return manifest_path, metadata_path
