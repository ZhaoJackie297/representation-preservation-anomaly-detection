param(
    [string]$Root
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Root) {
    $Root = Join-Path $repoRoot "references\external"
}

$repositories = @(
    @{
        Name = "PAI"
        Url = "https://github.com/pantheon5100/PAI.git"
        Commit = "29cb3bbf8e40b1610be9a3e91f4500c88fd930d3"
    },
    @{
        Name = "Anomaly-Transformer"
        Url = "https://github.com/thuml/Anomaly-Transformer.git"
        Commit = "b0ee470c8012bff857fb600462aec6209c4a18d9"
    },
    @{
        Name = "OmniAnomaly"
        Url = "https://github.com/NetManAIOps/OmniAnomaly.git"
        Commit = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"
    },
    @{
        Name = "t-nnclr"
        Url = "https://github.com/imics-lab/t-nnclr.git"
        Commit = "9c6439991202986a2af108798d5092ea34f0e9ed"
    },
    @{
        Name = "TS-TCC"
        Url = "https://github.com/emadeldeen24/TS-TCC.git"
        Commit = "15bdfd6eced6aeb6c638839828effe1b4b2ea391"
    }
)

New-Item -ItemType Directory -Path $Root -Force | Out-Null
$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path

foreach ($repository in $repositories) {
    $destination = Join-Path $resolvedRoot $repository.Name
    if (-not (Test-Path -LiteralPath $destination)) {
        git clone $repository.Url $destination
    }
    git -C $destination fetch --all --tags
    git -C $destination checkout $repository.Commit
    Write-Host "Ready: $destination @ $($repository.Commit)"
}
