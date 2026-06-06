from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchiveSpec:
    filename: str
    url: str
    md5: str
    size: int


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    institution: str
    doi: str
    license: str
    archives: tuple[ArchiveSpec, ...]
    citation: str


DATASETS: dict[str, DatasetSpec] = {
    "wideft": DatasetSpec(
        key="wideft",
        title="WIDEFT: A Corpus of Radio Frequency Signals for Wireless Device "
        "Fingerprint Research",
        institution="New Mexico State University",
        doi="10.5281/zenodo.4110980",
        license="CC-BY-4.0",
        archives=(
            ArchiveSpec(
                filename="WIDEFT.zip",
                url="https://zenodo.org/api/records/4116383/files/WIDEFT.zip/content",
                md5="985702f80e97d5631bca519f56a0eec1",
                size=2_053_215_491,
            ),
        ),
        citation=(
            "A. B. Siddik et al., WIDEFT: A Corpus of Radio Frequency Signals "
            "for Wireless Device Fingerprint Research, HST 2021."
        ),
    ),
    "pla": DatasetSpec(
        key="pla",
        title="INRIA I/Q Signal Dataset for RF Fingerprinting and Physical Layer "
        "Authentication",
        institution="Inria Lille and University of Cambridge",
        doi="10.5281/zenodo.18268648",
        license="CC-BY-4.0",
        archives=(
            ArchiveSpec(
                filename="PLA_dataset.zip",
                url=(
                    "https://zenodo.org/api/records/18268648/files/"
                    "PLA_dataset.zip/content"
                ),
                md5="aff583bee6f4efccd08fe78c731bf03d",
                size=710_685_409,
            ),
        ),
        citation=(
            "I. Alla et al., Robust Device Authentication in Multi-Node "
            "Networks: ML-Assisted Hybrid PLA Exploiting Hardware Impairments, "
            "ACSAC 2024."
        ),
    ),
    "wlan": DatasetSpec(
        key="wlan",
        title="RF Fingerprinting WLAN Dataset",
        institution="Wroclaw University of Science and Technology",
        doi="10.5281/zenodo.18515187",
        license="CC-BY-4.0",
        archives=(
            ArchiveSpec(
                filename="anechoic_chamber.zip",
                url=(
                    "https://zenodo.org/api/records/18515187/files/"
                    "anechoic_chamber.zip/content"
                ),
                md5="9af7491dc891d89969832f0efdee89de",
                size=137_021_472,
            ),
            ArchiveSpec(
                filename="office_room.zip",
                url=(
                    "https://zenodo.org/api/records/18515187/files/"
                    "office_room.zip/content"
                ),
                md5="8cb50121448016a6c7a1293051b26e1b",
                size=107_795_533,
            ),
        ),
        citation=(
            "M. Stojke, RF Fingerprinting WLAN Dataset, Zenodo, 2026, "
            "doi:10.5281/zenodo.18515187."
        ),
    ),
}


def selected_specs(keys: list[str]) -> list[DatasetSpec]:
    unknown = sorted(set(keys) - DATASETS.keys())
    if unknown:
        raise KeyError(f"Unknown dataset source(s): {', '.join(unknown)}")
    return [DATASETS[key] for key in keys]

