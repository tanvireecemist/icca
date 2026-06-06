from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from lightning_sdk import Studio

EXCLUDED = {
    ".git",
    ".pytest_cache",
    ".pytest_cache_local",
    ".pytest_tmp",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "lightning_logs",
    "outputs",
    "wandb",
}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in EXCLUDED
        or name.endswith((".ckpt", ".pyc", ".pyo"))
        or name == ".env"
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the research repository to an existing Lightning Studio."
    )
    parser.add_argument("--owner", default=os.getenv("LIGHTNING_OWNER"))
    parser.add_argument(
        "--owner-type",
        choices=("user", "org"),
        default=os.getenv("LIGHTNING_OWNER_TYPE", "user"),
    )
    parser.add_argument("--teamspace", default=os.getenv("LIGHTNING_TEAMSPACE"))
    parser.add_argument("--studio", default=os.getenv("LIGHTNING_STUDIO"))
    parser.add_argument("--remote-path", default="real-rf-dg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [
        name
        for name, value in (
            ("owner", args.owner),
            ("teamspace", args.teamspace),
            ("studio", args.studio),
        )
        if not value
    ]
    if missing:
        raise SystemExit("Missing Lightning setting(s): " + ", ".join(missing))

    repository = Path(__file__).resolve().parents[1]
    studio_kwargs = {"name": args.studio, "teamspace": args.teamspace}
    studio_kwargs[args.owner_type] = args.owner
    studio = Studio(**studio_kwargs, create_ok=False)

    with tempfile.TemporaryDirectory(prefix="real-rf-dg-sync-") as temporary:
        staged = Path(temporary) / "real-rf-dg"
        shutil.copytree(repository, staged, ignore=_ignore)
        studio.upload_folder(
            folder_path=str(staged),
            remote_path=args.remote_path,
            progress_bar=True,
        )
    print(
        f"Uploaded repository to Studio {args.owner}/{args.teamspace}/"
        f"{args.studio}:{args.remote_path}"
    )


if __name__ == "__main__":
    main()
