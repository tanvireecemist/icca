from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean = deepcopy(config)
    clean.pop("_config_path", None)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False)
    return output


def run_dir(config: dict[str, Any]) -> Path:
    project = config["project"]
    return Path(project["output_dir"]) / project["run_name"]

