from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix


def _scalar(value: Any) -> float | int | str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else str(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class JsonlMetricsCallback(L.Callback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.path = output_dir / "metrics.jsonl"

    def _write(self, trainer: L.Trainer, stage: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": stage,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            **{
                key: _scalar(value)
                for key, value in trainer.callback_metrics.items()
            },
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._write(trainer, "train")

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if not trainer.sanity_checking:
            self._write(trainer, "validation")


class GPUTelemetryCallback(L.Callback):
    HEADER = [
        "timestamp",
        "utilization_gpu_pct",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
        "temperature_c",
    ]

    def __init__(self, output_dir: Path, interval_seconds: int = 5) -> None:
        self.path = output_dir / "gpu_telemetry.csv"
        self.interval_seconds = max(1, interval_seconds)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _poll(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.HEADER)
            while not self.stop_event.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,utilization.gpu,memory.used,"
                            "memory.total,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    for line in result.stdout.strip().splitlines():
                        writer.writerow([item.strip() for item in line.split(",")])
                    handle.flush()
                except (OSError, subprocess.SubprocessError):
                    return
                self.stop_event.wait(self.interval_seconds)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval_seconds + 2)


def _expected_calibration_error(frame: pd.DataFrame, bins: int = 15) -> float:
    confidence = frame.confidence.to_numpy()
    correct = (frame.label == frame.prediction).to_numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


class TestArtifactCallback(L.Callback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        predictions = getattr(pl_module, "test_predictions", [])
        dataset = trainer.datamodule.test_dataset
        if not predictions or dataset is None:
            return
        rows = []
        for prediction in predictions:
            metadata = dataset.record(int(prediction["record_index"]))
            rows.append({**metadata, **prediction})
        frame = pd.DataFrame(rows).sort_values(["source", "record_id"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.output_dir / "predictions.csv", index=False)

        report_rows: list[dict[str, Any]] = []
        summary: dict[str, Any] = {"sources": {}}
        label_maps = trainer.datamodule.label_maps
        for source, source_frame in frame.groupby("source"):
            index_to_name = {
                index: name for name, index in label_maps[source].items()
            }
            labels = sorted(index_to_name)
            names = [index_to_name[index] for index in labels]
            report = classification_report(
                source_frame.label,
                source_frame.prediction,
                labels=labels,
                target_names=names,
                output_dict=True,
                zero_division=0,
            )
            for label_name, metrics in report.items():
                if isinstance(metrics, dict):
                    report_rows.append(
                        {"source": source, "class": label_name, **metrics}
                    )
            matrix = confusion_matrix(
                source_frame.label, source_frame.prediction, labels=labels
            )
            pd.DataFrame(matrix, index=names, columns=names).to_csv(
                self.output_dir / f"confusion_matrix_{source}.csv"
            )
            summary["sources"][source] = {
                "accuracy": float((source_frame.label == source_frame.prediction).mean()),
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "weighted_f1": float(report["weighted avg"]["f1-score"]),
                "ece_15_bin": _expected_calibration_error(source_frame),
                "samples": int(len(source_frame)),
            }
        pd.DataFrame(report_rows).to_csv(
            self.output_dir / "classification_report.csv", index=False
        )
        summary["callback_metrics"] = {
            key: _scalar(value) for key, value in trainer.callback_metrics.items()
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

