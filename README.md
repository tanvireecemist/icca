# Real-RF-DG

Publication-oriented code for **domain-generalized radio-frequency device
fingerprinting using only real captured signals**. The default experiment uses
an auditable 10% subset of each source:

- WIDEFT: 138 physical devices, Bluetooth/Wi-Fi/other signal bursts.
- INRIA PLA: raw 2.4 GHz I/Q captures from 12 physical transmitters.
- RF Fingerprinting WLAN 2026: three transmitters in an anechoic chamber and
  an office room.

The common model is a dual-domain temporal-frequency transformer. A shared
encoder learns raw-I/Q and spectral features, source-specific heads avoid
mixing incompatible device identities, and a gradient-reversal source
classifier discourages dataset shortcuts.

## ICCA 2026 constraints

ICCA requires a double-blind ACM-format paper of 6-8 pages. The final
submission deadline shown on the conference site is **June 6, 2026**. The
conference dates are October 15-16, 2026.

## Lightning AI L4 quick start

Run these commands inside a Lightning AI Studio configured with one NVIDIA L4:

```bash
# Do not wrap the URL in angle brackets. In Bash, <...> means redirection.
if [ ! -d icca/.git ]; then
  git clone --depth 1 https://github.com/tanvireecemist/icca.git icca
fi
cd icca
git pull --ff-only

bash scripts/setup_lightning_l4.sh
source .venv/bin/activate

rfbench fetch --config configs/research_l4.yaml
rfbench manifest --config configs/research_l4.yaml
rfbench tune-batch --config configs/research_l4.yaml
rfbench train --config outputs/tuning/research_l4.tuned.yaml
rfbench test --config outputs/tuning/research_l4.tuned.yaml
```

If `git pull --ff-only` reports local changes inside the Studio, preserve any
needed outputs first and then run from a clean clone. The repository should not
track `data/`, `outputs/`, checkpoints, caches, or secrets; datasets are fetched
directly inside the Studio. See `docs/LIGHTNING_GITHUB_FIX.md` if GitHub clone
or pull is slow, hangs, or fails.

To upload this repository from Windows into an existing Studio, set the
`LIGHTNING_*` variables shown in `.env.example`, authenticate the Lightning
SDK, and run:

```powershell
.\scripts\sync_lightning.ps1
```

The sync staging step excludes `.venv`, `data`, `outputs`, checkpoints, and
secrets. The datasets are downloaded directly inside the Studio.

`tune-batch` measures actual training steps on the real 10% subset. It searches
for the largest stable batch, benchmarks nearby candidates, and writes a tuned
configuration. This is more defensible than claiming one hard-coded batch size
fully utilizes every L4 host.

The L4 configuration uses `acquisition_mode: subset`. It reads each publisher
ZIP directory with HTTP Range requests, decides the deterministic 10% subset
first, and downloads only selected signal entries. For PLA, where each device
is one large compressed entry, it stream-decompresses and saves only the first
10% of each physical-device capture.

## Reproducibility

```bash
uv python install 3.11
uv venv --python 3.11
uv pip install -e ".[dev,cloud]"
pytest
ruff check .
```

Data archives, extracted data, generated manifests, checkpoints, and results
are ignored by Git. Every generated manifest records source URL, source
checksum, seed, selection fraction, label, split, byte offset, and sample
length.

## Results

Each run writes standard files under `outputs/<run_name>/`:

- `metrics.csv`: epoch metrics from Lightning.
- `metrics.jsonl`: structured callback metrics.
- `summary.json`: final metrics and run metadata.
- `predictions.csv`: sample-level predictions and confidence.
- `classification_report.csv`: per-source precision, recall, and F1.
- `confusion_matrix_<source>.csv`: confusion matrices.
- `gpu_telemetry.csv`: utilization, memory, power, and temperature.

W&B is optional. Set `logger.wandb: true` and provide `WANDB_API_KEY` through a
Lightning Studio secret. Never place the key in YAML, source code, or shell
history.

## Paper assets

`paper/methodology_diagrams.tex` is an Overleaf-ready standalone TikZ document.
Select **pdfLaTeX** in Overleaf. The generated diagrams cover dataset curation,
the network, training, and evaluation.

## Important research caveat

This repository provides a rigorous experiment, not a guarantee of paper
acceptance or a fabricated result. Report only metrics produced by completed
runs, preserve failed runs, and disclose that each dataset was restricted to a
deterministic 10% subset.
