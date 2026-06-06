param([string]$RemotePath = "real-rf-dg")

$ErrorActionPreference = "Stop"

foreach ($name in "LIGHTNING_OWNER", "LIGHTNING_TEAMSPACE", "LIGHTNING_STUDIO") {
    if (-not (Get-Item "Env:$name" -ErrorAction SilentlyContinue).Value) {
        throw "Set $name before syncing. See .env.example."
    }
}

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Create .venv and install the cloud extra first."
}

& $python (Join-Path $PSScriptRoot "sync_lightning.py") --remote-path $RemotePath
