# ICCA RealRF-DG Research Checkpoint

Snapshot date: **June 6, 2026**  
Workspace: `C:\Users\Dominic O'Brian\Documents\ICCA`  
Status: implementation and local validation are substantially complete; the
strict all-source 10% local fetch, manifest, CPU smoke run, static checks, and
paper recompilation are complete; the Lightning AI L4 execution is not yet
authenticated or completed.

## Continuation Update: June 6, 2026 17:15 +06

The strict all-source local validation path has now advanced past the original
checkpoint.

Completed after this handoff:

- Confirmed no existing `rf_research.cli fetch` process was running before
  resuming.
- Resumed and completed the strict subset fetch for all configured sources in
  `configs/validation_all_sources.yaml`.
- Wrote all three provenance files:
  - `data/subset_all/extracted/wideft/subset_acquisition.json`
  - `data/subset_all/extracted/pla/subset_acquisition.json`
  - `data/subset_all/extracted/wlan/subset_acquisition.json`
- Verified local extracted file counts:
  - WIDEFT: `1470` `.sc16` files
  - PLA: `12` `.bin` prefixes
  - WLAN: `41` `.npz` files
  - stale `.part` files: `0`
- Built the combined manifest:
  - CSV: `data/subset_all/manifests/validation_all_sources_10pct.csv`
  - JSON: `data/subset_all/manifests/validation_all_sources_10pct.json`
- Audited the manifest:
  - total rows: `11759`
  - source rows: PLA `10248`, WIDEFT `1470`, WLAN `41`
  - split rows: train `8213`, validation `1790`, test `1756`
  - source/split rows:
    - PLA train `7173`, validation `1533`, test `1542`
    - WIDEFT train `1029`, validation `247`, test `194`
    - WLAN train `11`, validation `10`, test `20`
  - paths outside `data/subset_all/extracted`: `0`
  - missing paths: `0`
- Ran the all-source CPU smoke training:
  - checkpoint: `outputs/validation_all_sources/checkpoints/best-00.ckpt`
  - artifacts root: `outputs/validation_all_sources`
- Ran the all-source CPU smoke test:
  - summary: `outputs/validation_all_sources/summary.json`
  - predictions: `outputs/validation_all_sources/predictions.csv`
  - reports and confusion matrix files were exported.
- Re-ran static validation:
  - `ruff`: all checks passed
  - `pytest`: `8 passed`
  - pytest emitted only a `.pytest_cache` write warning caused by local cache
    permissions.
- Recompiled paper artifacts:
  - `paper/methodology_diagrams.pdf`
  - `paper/icca_paper_skeleton.pdf`
  - MiKTeX was missing `upquote.sty`; installing the single MiKTeX package
    `upquote` fixed the compile. No source-code workaround was added.

Acquisition summary after completion:

```text
WIDEFT:
  selected records: 1470 / 14700
  effective record fraction: 10.0000%
  effective uncompressed fraction: 9.878637929096006%
  final successful invocation reused records: 1470
  downloaded_compressed_bytes in final provenance: 0
  selected_remote_compressed_bytes: 202,852,881
  saved_uncompressed_bytes: 536,930,104

PLA:
  selected records: 12 / 12 physical device prefixes
  effective uncompressed fraction: 9.99424%
  downloaded_compressed_bytes: 73,496,024
  selected_remote_compressed_bytes: 710,678,375
  saved_uncompressed_bytes: 167,903,232

WLAN:
  selected records: 41 / 411
  effective record fraction: 9.975669099756691%
  downloaded_compressed_bytes: 23,812,213
  selected_remote_compressed_bytes: 23,812,213
  saved_uncompressed_bytes: 71,408,880
```

Important nuance: WIDEFT had been partially downloaded before the final
successful fetch invocation. Because the provenance file is written only after
the source completes, the final WIDEFT `downloaded_compressed_bytes` value is
`0` and all records are marked as reused. Do not interpret this as "no network
was ever used for WIDEFT"; use `selected_remote_compressed_bytes` as the
auditable selected compressed payload size unless a future cumulative transfer
ledger is implemented.

Local smoke-test metrics remain pipeline validation only. The latest
`outputs/validation_all_sources/summary.json` covers two limited CPU test
batches and must not be reported as scientific results.

Still incomplete:

- Lightning AI authentication and Studio selection.
- L4 batch tuning inside the intended Lightning Studio.
- L4 training/testing with measured GPU telemetry.
- At least three seeds.
- Required ablations.
- Bootstrap confidence intervals/statistical analysis.
- Final empirical Results section and final 6-8 page double-blind paper.

## Read This First

This file is the handoff for another Codex account. Continue from the existing
workspace. Do not recreate the project or delete existing data.

At the time of this checkpoint, this command was still running in the
background:

```powershell
.\.venv\Scripts\python.exe -m rf_research.cli fetch --config configs\validation_all_sources.yaml
```

Observed process IDs at the checkpoint were `19028` (venv launcher) and `2744`
(Python child). Process IDs are not stable. Check by command line instead:

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -like '*rf_research.cli fetch*validation_all_sources*'
  } |
  Select-Object ProcessId, ParentProcessId, CommandLine
```

Do not launch a duplicate fetch while this process exists. Completed files are
CRC-verified and safely reusable after interruption. Temporary `.part` files
represent only active or interrupted entries and are replaced atomically.

## User Goal

Produce a publishable ICCA research codebase for communication/RF signal
learning with deep learning that:

- uses only real, publisher-hosted datasets from multiple sources;
- acquires and trains/tests on 10% of each source without synthetic data;
- runs on a Lightning AI NVIDIA L4 GPU;
- tunes batch size and related settings to use the L4 effectively;
- exports metrics and predictions to ordinary files without requiring W&B;
- includes Overleaf-compatible LaTeX methodology diagrams and a paper skeleton;
- is reproducible, provenance-driven, and robust enough to run without manual
  data surgery.

The proposed study is:

**RealRF-DG: Source-Aware Dual-Domain Learning for Data-Efficient Radio
Fingerprinting Across Real Capture Environments**

Research question:

> Can a shared temporal-frequency encoder trained on only 10% of several real
> RF corpora learn device-discriminative features without relying on
> dataset-specific shortcuts?

## Critical Security Note

The user pasted a W&B API key into the original conversation. It was
intentionally **not saved, echoed, used, or committed** anywhere in this
workspace.

The user must revoke that exposed key in W&B and create a replacement only if
W&B is later enabled. Prefer environment variables or Lightning secrets. Never
put a replacement key in source code, Markdown, YAML, shell history, or Git.

The current configs use:

```yaml
logger:
  wandb: false
```

File-based metrics are already implemented, so W&B is optional.

## ICCA 2026 Requirements Verified on June 6, 2026

Official site: <https://icca.aiub.edu/>

- Conference: 4th International Conference on Computing Advancements
  (ICCA 2026)
- Theme: Age of Computing and Augmented Life
- Venue: AIUB, Dhaka, Bangladesh
- Conference dates: October 15-16, 2026
- Final paper deadline: **June 6, 2026**
- Notification: August 5, 2026
- Camera-ready: August 20, 2026
- Registration deadline: September 5, 2026
- Review: double blind; author names and affiliations must be omitted
- Paper length: 6-8 pages
- Publication: accepted and presented papers are intended for submission to
  the ACM Digital Library, subject to policy and quality checks
- Relevant tracks include Deep Learning & Neural Networks, Wireless & Mobile
  Communication Systems, 5G/6G Networks & Intelligent Communications,
  Network Security & Secure Communication, Secure Digital Identity &
  Authentication, and Green Computing & Energy-Efficient AI Systems

Author guidelines:
<https://icca.aiub.edu/authorsguideline.php>

Because the final deadline is the snapshot date, a complete empirical paper is
not yet honestly ready: the L4 experiments, three-seed runs, ablations,
statistics, and final writing remain incomplete.

## Environment

Local platform: Windows PowerShell  
Local Python: 3.11  
Virtual environment: `.venv`

Verified package versions:

```text
torch=2.12.0+cpu
lightning=2.6.5
lightning_sdk=2026.06.05
cuda_available=False
```

The local machine is CPU-only. The production config is for a Lightning AI
Studio with an NVIDIA L4.

Install/recreate locally:

```powershell
uv sync --all-extras
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Last verified before this handoff:

```text
ruff: All checks passed
pytest: 8 passed
```

This directory is currently **not a Git repository**. `git status` reports:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Initialize Git only if desired, and never commit `data/`, `outputs/`, `.env`,
credentials, or API keys.

## Repository Map

Important files:

```text
configs/research_l4.yaml
configs/smoke_cpu.yaml
configs/validation_all_sources.yaml
docs/DATASETS.md
docs/RESEARCH_DESIGN.md
paper/acmart.cls
paper/ACM-Reference-Format.bst
paper/icca_paper_skeleton.tex
paper/icca_paper_skeleton.pdf
paper/methodology_diagrams.tex
paper/methodology_diagrams.pdf
paper/references.bib
scripts/setup_lightning_l4.sh
scripts/run_lightning_l4.sh
scripts/sync_lightning.py
scripts/sync_lightning.ps1
src/rf_research/cli.py
src/rf_research/config.py
src/rf_research/data/catalog.py
src/rf_research/data/dataset.py
src/rf_research/data/fetch.py
src/rf_research/data/manifest.py
src/rf_research/data/readers.py
src/rf_research/data/remote_zip.py
src/rf_research/data/subset_fetch.py
src/rf_research/model/layers.py
src/rf_research/model/losses.py
src/rf_research/model/network.py
src/rf_research/train/callbacks.py
src/rf_research/train/module.py
src/rf_research/train/runner.py
src/rf_research/train/tuning.py
tests/test_manifest.py
tests/test_model.py
tests/test_remote_zip.py
```

CLI:

```powershell
.\.venv\Scripts\python.exe -m rf_research.cli --help
```

Commands:

- `fetch`: acquire and verify real data
- `manifest`: create deterministic source-stratified experiment manifests
- `tune-batch`: benchmark candidate batches and choose an L4-safe size
- `train`: train and checkpoint
- `test`: evaluate and export metrics/predictions
- `run`: execute the full pipeline

## Datasets and Exact Provenance

No synthetic modulation corpus is used.

### WIDEFT

- Title: WIDEFT: A Corpus of Radio Frequency Signals for Wireless Device
  Fingerprint Research
- Institution: New Mexico State University
- DOI: <https://doi.org/10.5281/zenodo.4110980>
- License: CC BY 4.0
- Publisher archive size: `2,053,215,491` bytes
- Publisher MD5: `985702f80e97d5631bca519f56a0eec1`
- URL:
  <https://zenodo.org/api/records/4116383/files/WIDEFT.zip/content>
- Remote ZIP index observed: 14,921 total entries and approximately 14,702
  signal records
- Format: interleaved signed 16-bit I/Q `.sc16`
- 10% method: deterministic seeded selection per device/protocol stratum
- Expected selected record count: 1,470
- Acquisition: read remote ZIP central directory through HTTP Range; download
  only selected complete entries; verify ZIP CRC32 and record SHA-256

### INRIA PLA

- Title: INRIA I/Q Signal Dataset for RF Fingerprinting and Physical Layer
  Authentication
- Institutions: Inria Lille and University of Cambridge
- DOI: <https://doi.org/10.5281/zenodo.18268648>
- License: CC BY 4.0
- Publisher archive size: `710,685,409` bytes
- Publisher MD5: `aff583bee6f4efccd08fe78c731bf03d`
- URL:
  <https://zenodo.org/api/records/18268648/files/PLA_dataset.zip/content>
- 12 physical device `.bin` captures
- Format: complex64 I/Q
- 10% method: stream and save the first window-aligned 10% of every physical
  device capture
- The provenance correctly reports
  `selection_unit=aligned_uncompressed_bytes_per_device`; it does not
  misleadingly claim that 12 selected prefixes are 10% of 12 files
- Split control: physical offsets are kept contiguous before
  train/validation/test assignment to reduce leakage

An earlier pre-optimization proof saved `167,903,232` uncompressed bytes but
transferred `100,663,296` compressed bytes because the initial range chunk was
8 MiB per device. The code was then changed to estimate the first compressed
range from the requested uncompressed ratio plus 2%, followed by 256 KiB
chunks. Re-measure this ratio in the all-source validation run.

### RF Fingerprinting WLAN Dataset

- Institution: Wroclaw University of Science and Technology
- DOI: <https://doi.org/10.5281/zenodo.18515187>
- License: CC BY 4.0
- Anechoic archive:
  - size: `137,021,472` bytes
  - MD5: `9af7491dc891d89969832f0efdee89de`
  - URL:
    <https://zenodo.org/api/records/18515187/files/anechoic_chamber.zip/content>
- Office archive:
  - size: `107,795,533` bytes
  - MD5: `8cb50121448016a6c7a1293051b26e1b`
  - URL:
    <https://zenodo.org/api/records/18515187/files/office_room.zip/content>
- Format: real `.npz` captures
- Remote corpus: 411 captures
- Exact selected count: 41
- 10% method: deterministic selection per transmitter/environment stratum
- Domain-generalization protocol: train/validate on anechoic captures and test
  only on office captures

Strict WLAN acquisition was already proven:

```text
selected: 41 / 411 records
compressed bytes downloaded: 23,812,213
publisher archive bytes: 244,817,005
network fraction: approximately 9.73%
uncompressed bytes saved: 71,408,880
```

A second fetch downloaded zero bytes and reused all 41 CRC-verified files.

## Selective Remote ZIP Implementation

`src/rf_research/data/remote_zip.py`:

- reads a non-ZIP64 central directory through HTTP Range requests;
- parses local headers and supports ZIP stored and DEFLATE entries;
- extracts complete entries with CRC32 verification;
- streams prefixes and stops decompression at the requested byte count;
- writes to `.part`, then atomically renames on success;
- has retry handling for HTTP 429/500/502/503/504;
- honors `Retry-After`;
- uses bounded exponential backoff.

`src/rf_research/data/subset_fetch.py`:

- performs deterministic seeded 10% selection before downloads;
- uses exact record quotas for WIDEFT and WLAN;
- uses aligned byte-prefix quotas for PLA;
- validates safe ZIP destination paths lexically on Windows;
- reuses existing files only after size and CRC/hash checks;
- records source DOI, license, publisher checksum, URLs, selection unit,
  effective fractions, transfer bytes, saved bytes, per-record hashes, and
  local paths in `subset_acquisition.json`.

Important bugs already found and fixed through live fetching:

1. Concurrent `Path.resolve()` calls produced an intermittent Windows
   containment failure while sibling directories were being created. Replaced
   with lexical `abspath/commonpath` containment and explicit rejection of
   backslashes, drive syntax, absolute paths, and `..`.
2. Eight download workers triggered Zenodo HTTP 429 responses. Added retry and
   `Retry-After` support and reduced production/validation download workers to
   two.

GPU batch size tuning is independent from publisher download concurrency.

## Current Live Fetch State

Target root:

```text
data/subset_all
```

Config:

```text
configs/validation_all_sources.yaml
```

At approximately 16:12 Asia/Dhaka on June 6, 2026:

```text
WIDEFT completed .sc16 files: 1,170 / 1,470 target
WIDEFT completed bytes: 358,742,784
In-flight .part files: normally 2 while two workers are active
PLA: not yet started in this combined run
WLAN: not yet started in this combined run
subset_acquisition.json: not yet written for WIDEFT because metadata is
  committed after the source finishes
```

At final handoff verification around 16:14, the process was still active and
the completed WIDEFT count had advanced to `1,213` files and `385,486,720`
bytes.

The fetch continued to advance after command-tool timeouts. A timeout did not
kill the child process. The user later interrupted only the waiting command,
not necessarily the fetch.

Check current state:

```powershell
$files = Get-ChildItem data\subset_all\extracted\wideft -Recurse `
  -Filter *.sc16 -File -ErrorAction SilentlyContinue
[pscustomobject]@{
  Completed = $files.Count
  Bytes = ($files | Measure-Object Length -Sum).Sum
  Latest = ($files | Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty LastWriteTime)
}

Get-ChildItem data\subset_all\extracted -Recurse `
  -Filter subset_acquisition.json -File -ErrorAction SilentlyContinue
```

If no fetch process exists and the source is incomplete, resume safely:

```powershell
.\.venv\Scripts\python.exe -m rf_research.cli fetch `
  --config configs\validation_all_sources.yaml
```

Do not delete completed files. The resumed command validates and reuses them.

If stale `.part` files remain after no fetch process is running, they are safe
to remove, but this is optional because extraction deletes/replaces its own
matching `.part` before retrying.

## Manifest Rules

`src/rf_research/data/manifest.py` is the source of truth for experimental
selection and splits.

In `acquisition_mode: subset`:

- a source must have `subset_acquisition.json`;
- only exact `local_path` values listed in provenance are admitted;
- the manifest does not apply another 10% reduction;
- old full-dataset files elsewhere in the data root cannot leak into the run.

In `acquisition_mode: full`, the manifest performs local deterministic 10%
selection.

Selection uses seeded SHA-256 ordering with largest-remainder allocation over
source/label/domain strata, retaining at least one item per stratum where
possible.

WLAN split behavior:

- anechoic captures -> train/validation
- office captures -> test

PLA is split using physical offset ordering. WIDEFT uses source/device-aware
deterministic partitions.

After the all-source fetch completes:

```powershell
.\.venv\Scripts\python.exe -m rf_research.cli manifest `
  --config configs\validation_all_sources.yaml
```

Then inspect row counts by source/split and ensure every path belongs to
`data/subset_all/extracted`.

## Model and Training Design

The network is a source-aware dual-domain RF fingerprint model:

- raw temporal I/Q branch;
- frequency branch using FFT magnitude and phase-step features;
- fused transformer representation;
- source-specific device classification heads;
- supervised contrastive loss;
- gradient-reversal source/domain confusion;
- shared embedding output.

Research contributions currently stated:

1. Checksum-verified, manifest-driven benchmark using only real captures.
2. Dual-domain encoder combining temporal and frequency representations.
3. Source-specific heads plus adversarial source confusion.
4. Cross-environment WLAN evaluation.
5. Reproducible L4 batch/throughput tuning and ordinary metric exports.

Required final-paper ablations:

- time branch only;
- frequency branch only;
- no source-adversarial loss;
- no supervised contrastive loss;
- shared global classifier instead of source-specific heads;
- fixed batch size versus tuned L4 batch size.

Required statistical protocol:

- at least three seeds;
- report mean and standard deviation;
- paired bootstrap confidence intervals over test predictions;
- do not choose the best seed for headline results.

## L4 Production Configuration

Config:

```text
configs/research_l4.yaml
```

Key settings:

```yaml
data:
  acquisition_mode: subset
  download_workers: 2
  subset_fraction: 0.10
  window_size: 4096
  window_stride: 4096
  num_workers: 12
  prefetch_factor: 4
  pin_memory: true
  persistent_workers: true

trainer:
  accelerator: gpu
  devices: 1
  precision: bf16-mixed
  max_epochs: 80
  batch_size: 512
  benchmark: true
  deterministic: false
  compile: true
  compile_mode: reduce-overhead

tuning:
  start_batch_size: 128
  max_batch_size: 4096
  memory_fraction: 0.90
  safety_factor: 0.85
  warmup_steps: 8
  benchmark_steps: 30
```

The batch tuner benchmarks real training steps, checks peak GPU memory and
throughput, and writes a tuned YAML under `outputs/tuning/`. Do not simply
force batch 4096; use the measured safe result. The target is high throughput
within approximately 90% memory utilization, then apply the safety factor.

`scripts/setup_lightning_l4.sh` verifies:

- CUDA is available;
- GPU name includes `L4`;
- VRAM is at least approximately 22 GiB.

Studio run:

```bash
bash scripts/setup_lightning_l4.sh
bash scripts/run_lightning_l4.sh configs/research_l4.yaml
```

## Lightning AI Status and Blocker

Lightning AI Studio has not been authenticated from this workspace.

Missing values:

```text
LIGHTNING_OWNER
LIGHTNING_OWNER_TYPE=user|org
LIGHTNING_TEAMSPACE
LIGHTNING_STUDIO
```

Sync command after those values are available:

```powershell
$env:LIGHTNING_OWNER='...'
$env:LIGHTNING_OWNER_TYPE='user'
$env:LIGHTNING_TEAMSPACE='...'
$env:LIGHTNING_STUDIO='...'
.\.venv\Scripts\python.exe scripts\sync_lightning.py
```

The sync script intentionally excludes:

- `.git`
- `.venv`
- caches and `__pycache__`
- `data`
- `outputs`
- `wandb`
- `.env`
- checkpoints and bytecode

The data should be fetched directly inside Lightning Studio using the strict
subset acquisition pipeline, rather than uploading local datasets.

Browser observations:

- The in-app Browser could open the public ICCA site.
- The in-app Browser showed Lightning as unavailable in the current region.
- Chrome automation was requested, but the selected Chrome `Default` profile
  did not have the Codex extension/backend available.
- Local diagnostics found the Codex Chrome extension enabled in Chrome
  profiles 6 and 10, not in the selected Default profile.
- The native host configuration appeared correct.

A new account should either:

1. use a Chrome profile with the Codex extension and an authenticated
   Lightning session; or
2. provide the Lightning owner/teamspace/studio identifiers and authenticate
   via the Lightning SDK.

Starting an L4 Studio incurs cloud cost. Confirm the intended Studio and
account before starting or switching machines.

## Metrics and Existing Validation Evidence

The runner exports:

- `metrics.csv`
- `metrics.jsonl`
- `test_metrics.csv`
- `summary.json`
- `classification_report.csv`
- per-source confusion matrices
- `predictions.csv`
- `resolved_config.yaml`
- checkpoints
- GPU telemetry where CUDA/NVML is available

Existing successful CPU smoke runs:

```text
outputs/smoke_real_wlan
outputs/smoke_selective_wlan
```

Both trained and tested end to end on real WLAN data with two train, two
validation, and two test batches. The selective smoke test used only 16 test
samples because of the configured batch limit; its metrics are pipeline
validation, not research findings:

```text
accuracy: 0.25
macro F1: 0.0842105263
weighted F1: 0.1052631579
ECE (15 bins): 0.0368980989
samples: 16
test loss: 1.9083908796
```

Do not report these smoke metrics in the paper as final results.

After the combined fetch and manifest complete, run the all-source CPU
integration smoke:

```powershell
.\.venv\Scripts\python.exe -m rf_research.cli train `
  --config configs\validation_all_sources.yaml

.\.venv\Scripts\python.exe -m rf_research.cli test `
  --config configs\validation_all_sources.yaml
```

This config has one epoch and limits each stage to two batches.

## Paper and Overleaf

Paper files:

```text
paper/icca_paper_skeleton.tex
paper/icca_paper_skeleton.pdf
paper/methodology_diagrams.tex
paper/methodology_diagrams.pdf
paper/references.bib
paper/acmart.cls
paper/ACM-Reference-Format.bst
```

The official ICCA ACM class/template files are included.

Overleaf compiler: **pdfLaTeX**

Both TeX files were compiled successfully with pdfLaTeX and visually
inspected. The methodology document contains publication-ready TikZ diagrams
for the real-data acquisition pipeline and model/training methodology.

The paper skeleton is intentionally incomplete. It must not be filled with
invented L4 results. Keep it double blind for submission and fit the final
manuscript into 6-8 pages.

Compile locally from `paper/`:

```powershell
pdflatex -interaction=nonstopmode methodology_diagrams.tex
pdflatex -interaction=nonstopmode icca_paper_skeleton.tex
bibtex icca_paper_skeleton
pdflatex -interaction=nonstopmode icca_paper_skeleton.tex
pdflatex -interaction=nonstopmode icca_paper_skeleton.tex
```

## Immediate Next Actions

1. Check whether the all-source fetch process is still running.
2. If running, let it finish. Do not duplicate it.
3. If stopped, resume the same fetch command; completed files will be reused.
4. Inspect all three `subset_acquisition.json` files and calculate:
   selected records, exact effective fraction, compressed bytes transferred,
   uncompressed bytes saved, and reused-record counts.
5. Run the combined manifest and audit source/split/label/domain counts.
6. Run the all-source CPU train/test smoke.
7. Run `ruff` and `pytest` again.
8. Recompile both LaTeX documents.
9. Authenticate Lightning AI and identify the intended owner, teamspace, and
   Studio.
10. Sync only the repository, fetch data inside the Studio, run L4 batch
    tuning, train, and test.
11. Run at least three seeds and all required ablations.
12. Add statistical analysis and honest measured results to the paper.

## Recommended Verification Commands

```powershell
# Static checks
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q

# Fetch or resume
.\.venv\Scripts\python.exe -m rf_research.cli fetch `
  --config configs\validation_all_sources.yaml

# Manifest
.\.venv\Scripts\python.exe -m rf_research.cli manifest `
  --config configs\validation_all_sources.yaml

# Small all-source integration run
.\.venv\Scripts\python.exe -m rf_research.cli train `
  --config configs\validation_all_sources.yaml
.\.venv\Scripts\python.exe -m rf_research.cli test `
  --config configs\validation_all_sources.yaml

# Production workflow inside an L4 Studio
bash scripts/setup_lightning_l4.sh
bash scripts/run_lightning_l4.sh configs/research_l4.yaml
```

## Integrity Constraints for the Next Agent

- Do not generate or substitute synthetic RF data.
- Do not silently download/store complete archives for a claimed strict 10%
  experiment.
- Do not apply a second 10% reduction after selective acquisition.
- Do not mix old full-dataset files into the strict subset manifest.
- Do not report smoke-test metrics as scientific results.
- Do not invent L4 throughput, utilization, memory, accuracy, or ablation
  values.
- Do not store credentials.
- Do not delete existing completed subset files.
- Preserve publisher DOI, license, checksums, hashes, and acquisition
  provenance in all final artifacts.
- Keep the submission double blind until the camera-ready stage.
