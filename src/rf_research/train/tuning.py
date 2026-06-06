from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from rf_research.config import save_config
from rf_research.data.dataset import RFDataModule
from rf_research.train.module import RFFingerprintModule


def _round_batch(value: int) -> int:
    return max(8, int(value // 8) * 8)


def _single_benchmark(
    config: dict[str, Any],
    batch_size: int,
    warmup_steps: int,
    benchmark_steps: int,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    gc.collect()
    datamodule = RFDataModule(config, batch_size=batch_size)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    if len(loader) == 0:
        return {"batch_size": batch_size, "status": "too_large_for_dataset"}
    model = RFFingerprintModule(
        config, datamodule.num_classes, datamodule.source_map
    ).cuda()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["trainer"]["learning_rate"]),
        weight_decay=float(config["trainer"]["weight_decay"]),
        fused=True,
    )
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []
    try:
        for step in range(warmup_steps + benchmark_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            iq = batch["iq"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            sources = batch["source"].cuda(non_blocking=True)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embedding = model(iq)
                loss = embedding.new_zeros(())
                for source_index, source_name in model.index_to_source.items():
                    mask = sources == source_index
                    count = int(mask.sum())
                    if count:
                        logits = model.heads[source_name](embedding[mask])
                        loss = loss + F.cross_entropy(logits, labels[mask]) * (
                            count / labels.shape[0]
                        )
                loss = loss + float(config["model"]["domain_weight"]) * F.cross_entropy(
                    model.domain_head(embedding), sources
                )
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step >= warmup_steps:
                durations.append(time.perf_counter() - start)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return {
            "batch_size": batch_size,
            "status": "ok",
            "samples_per_second": batch_size / (sum(durations) / len(durations)),
            "peak_allocated_gib": peak_allocated / 2**30,
            "peak_reserved_gib": peak_reserved / 2**30,
            "memory_fraction": peak_reserved / total_memory,
        }
    except torch.OutOfMemoryError:
        return {"batch_size": batch_size, "status": "oom"}
    finally:
        del model, optimizer, datamodule, loader
        torch.cuda.empty_cache()
        gc.collect()


def tune_batch_size(config: dict[str, Any]) -> tuple[Path, Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("Batch tuning requires an NVIDIA CUDA GPU.")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name.upper():
        print(f"Warning: tuning requested for an L4, detected {device_name!r}.")

    tune = config["tuning"]
    data = RFDataModule(config)
    data.setup("fit")
    train_size = len(data.train_dataset)
    del data
    start = int(tune["start_batch_size"])
    maximum = min(int(tune["max_batch_size"]), train_size)
    candidates: list[int] = []
    value = start
    while value <= maximum:
        candidates.append(value)
        value *= 2
    if candidates[-1] != maximum and maximum > candidates[-1]:
        candidates.append(_round_batch(maximum))

    results: list[dict[str, Any]] = []
    for candidate in sorted(set(candidates)):
        result = _single_benchmark(
            config,
            batch_size=candidate,
            warmup_steps=int(tune["warmup_steps"]),
            benchmark_steps=int(tune["benchmark_steps"]),
        )
        results.append(result)
        if result["status"] == "oom":
            break

    target_memory = float(tune["memory_fraction"])
    fitting = [
        row
        for row in results
        if row["status"] == "ok" and row["memory_fraction"] <= target_memory
    ]
    if not fitting:
        fitting = [row for row in results if row["status"] == "ok"]
    if not fitting:
        raise RuntimeError("No tested batch size fit on the GPU.")
    max_fit = max(row["batch_size"] for row in fitting)
    safe_max = _round_batch(max_fit * float(tune["safety_factor"]))
    safe_candidates = [row for row in fitting if row["batch_size"] <= safe_max]
    if not safe_candidates:
        safe_candidates = [min(fitting, key=lambda row: row["batch_size"])]
    best = max(safe_candidates, key=lambda row: row["samples_per_second"])

    tuned = json.loads(json.dumps(config))
    tuned.pop("_config_path", None)
    tuned["trainer"]["batch_size"] = int(best["batch_size"])
    tuning_dir = Path(config["project"]["output_dir"]) / "tuning"
    config_stem = Path(config.get("_config_path", "research_l4.yaml")).stem
    tuned_path = tuning_dir / f"{config_stem}.tuned.yaml"
    report_path = tuning_dir / f"{config_stem}.batch_benchmark.json"
    save_config(tuned, tuned_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "device": device_name,
                "selected_batch_size": best["batch_size"],
                "target_memory_fraction": target_memory,
                "safety_factor": float(tune["safety_factor"]),
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuned_path, report_path
