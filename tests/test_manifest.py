from __future__ import annotations

from collections import Counter

from rf_research.data.manifest import _assign_splits, _select_fraction


def test_exact_ten_percent_per_stratum() -> None:
    records = []
    for label in ("a", "b"):
        for domain in ("wifi", "bluetooth"):
            for index in range(100):
                records.append(
                    {
                        "source": "wideft",
                        "label_name": label,
                        "domain": domain,
                        "record_id": f"{label}/{domain}/{index}",
                    }
                )
    selected = _select_fraction(records, 0.10, seed=2026)
    counts = Counter(
        (row["source"], row["label_name"], row["domain"]) for row in selected
    )
    assert set(counts.values()) == {10}
    assert len(selected) == 40


def test_variable_strata_preserve_exact_source_quota() -> None:
    records = []
    for label, count in (("a", 23), ("b", 18)):
        for index in range(count):
            records.append(
                {
                    "source": "wlan",
                    "label_name": label,
                    "domain": "office",
                    "record_id": f"{label}/{index}",
                }
            )
    selected = _select_fraction(records, 0.10, seed=2026)
    assert len(selected) == 4


def test_wlan_office_is_test_only() -> None:
    rows = []
    for domain in ("anechoic_chamber", "office_room"):
        for index in range(10):
            rows.append(
                {
                    "source": "wlan",
                    "label_name": "001",
                    "domain": domain,
                    "record_id": f"{domain}/{index}",
                }
            )
    _assign_splits(rows, seed=2026, train_fraction=0.7, val_fraction=0.15)
    assert {row["split"] for row in rows if row["domain"] == "office_room"} == {
        "test"
    }
    assert {row["split"] for row in rows if row["domain"] == "anechoic_chamber"} == {
        "train",
        "val",
    }
