from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rf_research.config import load_config
from rf_research.data.catalog import selected_specs
from rf_research.data.fetch import fetch_dataset
from rf_research.data.manifest import build_manifest
from rf_research.data.subset_fetch import fetch_dataset_subset
from rf_research.train.runner import test as test_run
from rf_research.train.runner import train as train_run
from rf_research.train.tuning import tune_batch_size

app = typer.Typer(
    no_args_is_help=True,
    help="Reproducible multi-source real-RF fingerprinting experiments.",
)


@app.command()
def fetch(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    no_extract: Annotated[
        bool, typer.Option(help="Download and verify without extracting.")
    ] = False,
) -> None:
    """Download real datasets, verify checksums, and extract safely."""
    settings = load_config(config)
    root = Path(settings["data"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    mode = settings["data"].get("acquisition_mode", "full")
    for spec in selected_specs(settings["data"]["sources"]):
        typer.echo(f"Fetching {spec.key}: {spec.title}")
        if mode == "subset":
            if no_extract:
                raise typer.BadParameter(
                    "--no-extract is incompatible with subset acquisition"
                )
            metadata = fetch_dataset_subset(
                spec,
                root,
                fraction=float(settings["data"]["subset_fraction"]),
                seed=int(settings["project"]["seed"]),
                window_size=int(settings["data"]["window_size"]),
                download_workers=int(
                    settings["data"].get("download_workers", 8)
                ),
            )
            fraction_key = (
                "effective_uncompressed_fraction"
                if metadata["source"] == "pla"
                else "effective_record_fraction"
            )
            typer.echo(
                f"Selected {metadata['selected_record_count']} of "
                f"{metadata['remote_record_count']} remote records; "
                f"effective fraction {metadata[fraction_key]:.4%}; "
                f"downloaded "
                f"{metadata['downloaded_compressed_bytes'] / 2**20:.1f} MiB"
            )
        elif mode == "full":
            fetch_dataset(spec, root, extract=not no_extract)
        else:
            raise typer.BadParameter(f"Unknown acquisition_mode: {mode}")


@app.command()
def manifest(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Create the deterministic, source-stratified 10% experiment manifest."""
    settings = load_config(config)
    manifest_path, metadata_path = build_manifest(settings)
    typer.echo(f"Manifest: {manifest_path}")
    typer.echo(f"Metadata: {metadata_path}")


@app.command("tune-batch")
def tune_batch(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Benchmark real training steps and select a high-throughput L4 batch size."""
    settings = load_config(config)
    tuned, report = tune_batch_size(settings)
    typer.echo(f"Tuned config: {tuned}")
    typer.echo(f"Benchmark report: {report}")


@app.command()
def train(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
) -> None:
    """Train and checkpoint the multi-source model."""
    checkpoint, output = train_run(load_config(config))
    typer.echo(f"Best checkpoint: {checkpoint}")
    typer.echo(f"Artifacts: {output}")


@app.command()
def test(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    checkpoint: Annotated[
        Path | None, typer.Option(exists=True, readable=True)
    ] = None,
) -> None:
    """Evaluate a checkpoint and export predictions and standard metrics."""
    summary = test_run(load_config(config), checkpoint)
    typer.echo(f"Summary: {summary}")


@app.command()
def run(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    skip_fetch: Annotated[bool, typer.Option()] = False,
    skip_tuning: Annotated[bool, typer.Option()] = False,
) -> None:
    """Execute fetch, manifest, L4 tuning, training, and testing."""
    settings = load_config(config)
    if not skip_fetch:
        root = Path(settings["data"]["root"])
        root.mkdir(parents=True, exist_ok=True)
        mode = settings["data"].get("acquisition_mode", "full")
        for spec in selected_specs(settings["data"]["sources"]):
            if mode == "subset":
                fetch_dataset_subset(
                    spec,
                    root,
                    fraction=float(settings["data"]["subset_fraction"]),
                    seed=int(settings["project"]["seed"]),
                    window_size=int(settings["data"]["window_size"]),
                    download_workers=int(
                        settings["data"].get("download_workers", 8)
                    ),
                )
            elif mode == "full":
                fetch_dataset(spec, root)
            else:
                raise typer.BadParameter(f"Unknown acquisition_mode: {mode}")
    build_manifest(settings)
    active_config = config
    if not skip_tuning:
        active_config, _ = tune_batch_size(settings)
    active = load_config(active_config)
    checkpoint, _ = train_run(active)
    summary = test_run(active, checkpoint)
    typer.echo(f"Completed run: {summary}")


if __name__ == "__main__":
    app()
