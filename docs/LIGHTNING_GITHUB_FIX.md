# GitHub and Lightning AI Clone Fix

Use this when Lightning AI cannot clone or pull the repository cleanly.

## Root Causes

1. Do not run `git clone <https://github.com/tanvireecemist/icca.git>`.
   In Bash, angle brackets are shell redirection syntax. Use the URL directly.
2. The repository must not track downloaded datasets, outputs, checkpoints,
   caches, or secrets. These are generated again inside Lightning Studio.

## Correct Lightning Clone/Update Commands

Run inside the Lightning AI Studio terminal after selecting an NVIDIA L4:

```bash
cd /teamspace/studios/this_studio

if [ ! -d icca/.git ]; then
  git clone --depth 1 https://github.com/tanvireecemist/icca.git icca
fi

cd icca
git pull --ff-only
```

Then run:

```bash
nvidia-smi
bash scripts/setup_lightning_l4.sh
source .venv/bin/activate

rfbench fetch --config configs/research_l4.yaml
rfbench manifest --config configs/research_l4.yaml
rfbench tune-batch --config configs/research_l4.yaml
rfbench train --config outputs/tuning/research_l4.tuned.yaml
rfbench test --config outputs/tuning/research_l4.tuned.yaml
```

Or after setup:

```bash
bash scripts/run_lightning_l4.sh configs/research_l4.yaml
```

## Clean A Repo That Already Tracks Data

Run locally from the repository root, then push:

```bash
git rm -r --cached --ignore-unmatch data outputs checkpoints wandb lightning_logs
git rm -r --cached --ignore-unmatch .pytest_cache .pytest_cache_local .pytest_tmp .ruff_cache

git add .gitignore README.md pyproject.toml scripts/sync_lightning.py docs/LIGHTNING_GITHUB_FIX.md
git commit -m "Clean repository for Lightning AI runs"
git push origin main
```

After this, use `git clone --depth 1` on Lightning so the Studio only downloads
the current clean tree.

## If The Repo Is New And Still Huge

If the repository was created only for this project and has no collaborators,
the cleanest fix is to rewrite the GitHub branch to remove data from history:

```bash
git checkout --orphan clean-main
git rm -r --cached .
git add .
git commit -m "Clean code-only ICCA repository"
git branch -M main
git push --force origin main
```

Only use the force-push path when you are sure overwriting the remote history is
acceptable. It is the fastest way to make future Lightning clones small.

## W&B Key

Do not put a W&B key in GitHub, YAML, Markdown, shell history, or source code.
The current configs use `wandb: false`, so metrics and predictions are exported
as ordinary local files.
