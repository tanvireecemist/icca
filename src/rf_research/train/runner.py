from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from rf_research.config import run_dir, save_config
from rf_research.data.dataset import RFDataModule
from rf_research.train.callbacks import (
    GPUTelemetryCallback,
    JsonlMetricsCallback,
    TestArtifactCallback,
)
from rf_research.train.module import RFFingerprintModule


def _reject_embedded_secrets(config: dict[str, Any]) -> None:
    serialized = json.dumps(config)
    if "wandb_v1_" in serialized:
        raise ValueError(
            "A W&B API key was embedded in configuration. Use WANDB_API_KEY as "
            "an environment variable or Lightning Studio secret."
        )


def _hardware_settings(config: dict[str, Any]) -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = bool(config["trainer"]["benchmark"])


def _logger(config: dict[str, Any], output_dir: Path, phase: str):
    loggers: list[Any] = [
        CSVLogger(save_dir=str(output_dir), name="csv", version=phase)
    ]
    if config["logger"].get("wandb", False):
        if not os.getenv("WANDB_API_KEY"):
            raise RuntimeError(
                "logger.wandb is enabled but WANDB_API_KEY is not set as a secret."
            )
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError as error:
            raise RuntimeError("Install the `wandb` extra to enable W&B.") from error
        loggers.append(
            WandbLogger(
                project=os.getenv("WANDB_PROJECT", config["project"]["name"]),
                entity=os.getenv("WANDB_ENTITY") or None,
                name=config["project"]["run_name"],
                save_dir=str(output_dir),
                log_model=False,
            )
        )
    return loggers


def _trainer(
    config: dict[str, Any],
    output_dir: Path,
    include_checkpointing: bool = True,
) -> L.Trainer:
    trainer_config = config["trainer"]
    callbacks: list[L.Callback] = [
        JsonlMetricsCallback(output_dir),
        GPUTelemetryCallback(
            output_dir, int(config["logger"]["gpu_telemetry_seconds"])
        ),
        LearningRateMonitor(logging_interval="step"),
        TestArtifactCallback(output_dir),
    ]
    if include_checkpointing:
        callbacks.extend(
            [
                ModelCheckpoint(
                    dirpath=output_dir / "checkpoints",
                    filename="best-{epoch:02d}",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    auto_insert_metric_name=False,
                ),
                EarlyStopping(
                    monitor="val/loss",
                    mode="min",
                    patience=int(trainer_config["early_stopping_patience"]),
                ),
            ]
        )
    optional_limits = {
        key: trainer_config[key]
        for key in (
            "limit_train_batches",
            "limit_val_batches",
            "limit_test_batches",
        )
        if key in trainer_config
    }
    return L.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        precision=trainer_config["precision"],
        max_epochs=int(trainer_config["max_epochs"]),
        accumulate_grad_batches=int(trainer_config["accumulate_grad_batches"]),
        gradient_clip_val=float(trainer_config["gradient_clip_val"]),
        benchmark=bool(trainer_config["benchmark"]),
        deterministic=bool(trainer_config["deterministic"]),
        log_every_n_steps=int(trainer_config["log_every_n_steps"]),
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        logger=_logger(
            config, output_dir, "training" if include_checkpointing else "testing"
        ),
        enable_progress_bar=True,
        **optional_limits,
    )


def _build_model(config: dict[str, Any], data: RFDataModule) -> RFFingerprintModule:
    model = RFFingerprintModule(config, data.num_classes, data.source_map)
    if config["trainer"].get("compile", False):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
        model.encoder = torch.compile(
            model.encoder, mode=config["trainer"].get("compile_mode", "default")
        )
    return model


def train(config: dict[str, Any]) -> tuple[Path, Path]:
    _reject_embedded_secrets(config)
    _hardware_settings(config)
    L.seed_everything(int(config["project"]["seed"]), workers=True)
    output_dir = run_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    data = RFDataModule(config)
    data.setup("fit")
    model = _build_model(config, data)
    trainer = _trainer(config, output_dir)
    trainer.fit(model, datamodule=data)
    csv_metrics = Path(trainer.loggers[0].log_dir) / "metrics.csv"
    if csv_metrics.exists():
        shutil.copy2(csv_metrics, output_dir / "metrics.csv")
    checkpoint = Path(trainer.checkpoint_callback.best_model_path)
    if not checkpoint.exists():
        checkpoint = output_dir / "checkpoints" / "last.ckpt"
    return checkpoint, output_dir


def test(config: dict[str, Any], checkpoint: str | Path | None = None) -> Path:
    _reject_embedded_secrets(config)
    _hardware_settings(config)
    L.seed_everything(int(config["project"]["seed"]), workers=True)
    output_dir = run_dir(config)
    data = RFDataModule(config)
    data.setup("test")
    model = _build_model(config, data)
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if checkpoint_path is None:
        candidates = sorted((output_dir / "checkpoints").glob("best-*.ckpt"))
        checkpoint_path = candidates[-1] if candidates else output_dir / "checkpoints" / "last.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    trainer = _trainer(config, output_dir, include_checkpointing=False)
    trainer.test(model, datamodule=data, ckpt_path=str(checkpoint_path))
    csv_metrics = Path(trainer.loggers[0].log_dir) / "metrics.csv"
    if csv_metrics.exists():
        shutil.copy2(csv_metrics, output_dir / "test_metrics.csv")
    return output_dir / "summary.json"
