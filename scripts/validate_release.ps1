$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
python (Join-Path $PSScriptRoot "validate_release.py") $repoRoot

