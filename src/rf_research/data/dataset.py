from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from rf_research.data.readers import load_iq


class RFManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, source_map: dict[str, int]) -> None:
        self.frame = frame.reset_index(drop=True)
        self.source_map = source_map

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        iq = load_iq(
            path=row["path"],
            dtype=row["dtype"],
            offset=int(row["offset"]),
            n_samples=int(row["n_samples"]),
            scale=float(row["scale"]),
        )
        return {
            "iq": torch.from_numpy(iq),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "source": torch.tensor(
                self.source_map[row["source"]], dtype=torch.long
            ),
            "record_index": torch.tensor(index, dtype=torch.long),
        }

    def record(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "record_id": row["record_id"],
            "source": row["source"],
            "label_name": row["label_name"],
            "domain": row["domain"],
            "split": row["split"],
        }


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class RFDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: dict,
        manifest_path: str | Path | None = None,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.data_config = config["data"]
        self.batch_size = int(batch_size or config["trainer"]["batch_size"])
        self.manifest_path = Path(manifest_path) if manifest_path else self._default_manifest()
        self.metadata_path = self.manifest_path.with_suffix(".json")
        self.train_dataset: RFManifestDataset | None = None
        self.val_dataset: RFManifestDataset | None = None
        self.test_dataset: RFManifestDataset | None = None
        self.source_map: dict[str, int] = {}
        self.label_maps: dict[str, dict[str, int]] = {}

    def _default_manifest(self) -> Path:
        fraction = int(float(self.data_config["subset_fraction"]) * 100)
        name = self.config["project"]["run_name"]
        return Path(self.data_config["root"]) / "manifests" / f"{name}_{fraction:02d}pct.csv"

    def setup(self, stage: str | None = None) -> None:
        frame = pd.read_csv(
            self.manifest_path,
            dtype={
                "source": "string",
                "path": "string",
                "record_id": "string",
                "label_name": "string",
                "domain": "string",
                "dtype": "string",
                "split": "string",
            },
        )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.label_maps = metadata["label_maps"]
        self.source_map = {
            source: index for index, source in enumerate(sorted(self.label_maps))
        }
        if stage in (None, "fit"):
            self.train_dataset = RFManifestDataset(
                frame[frame.split == "train"], self.source_map
            )
            self.val_dataset = RFManifestDataset(
                frame[frame.split == "val"], self.source_map
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = RFManifestDataset(
                frame[frame.split == "test"], self.source_map
            )

    @property
    def num_classes(self) -> dict[str, int]:
        return {source: len(labels) for source, labels in self.label_maps.items()}

    def _loader(
        self,
        dataset: RFManifestDataset,
        shuffle: bool,
        balanced: bool = False,
    ) -> DataLoader:
        workers = int(self.data_config["num_workers"])
        sampler = None
        if balanced:
            keys = dataset.frame.source.astype(str) + ":" + dataset.frame.label.astype(str)
            counts = keys.value_counts()
            weights = keys.map(lambda key: 1.0 / counts[key]).to_numpy()
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights.copy(), dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": workers,
            "pin_memory": bool(self.data_config["pin_memory"]),
            "drop_last": balanced,
            "worker_init_fn": _seed_worker,
        }
        if workers:
            kwargs["persistent_workers"] = bool(self.data_config["persistent_workers"])
            kwargs["prefetch_factor"] = int(self.data_config["prefetch_factor"])
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader")
        return self._loader(self.train_dataset, shuffle=False, balanced=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test_dataloader")
        return self._loader(self.test_dataset, shuffle=False)

    def recommended_workers(self) -> int:
        return min(16, max(1, (os.cpu_count() or 2) - 2))
