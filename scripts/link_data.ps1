param(
    [Parameter(Mandatory = $true)]
    [string]$ProcessedR42,

    [Parameter(Mandatory = $true)]
    [string]$ProcessedR52,

    [Parameter(Mandatory = $true)]
    [string]$ProcessedR62,

    [Parameter(Mandatory = $true)]
    [string]$OmniAnomaly,

    [string]$Experiments
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $repoRoot "runtime\cert"

function New-DataJunction {
    param(
        [string]$Name,
        [string]$Target
    )

    if (-not (Test-Path -LiteralPath $Target -PathType Container)) {
        throw "Target directory does not exist: $Target"
    }

    $destination = Join-Path $runtimeRoot $Name
    if (Test-Path -LiteralPath $destination) {
        throw "Destination already exists: $destination"
    }

    New-Item -ItemType Junction -Path $destination -Target $Target | Out-Null
    Write-Host "Linked $destination -> $Target"
}

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
New-DataJunction -Name "processed_r42" -Target $ProcessedR42
New-DataJunction -Name "processed_r52" -Target $ProcessedR52
New-DataJunction -Name "processed_r62" -Target $ProcessedR62
New-DataJunction -Name "OmniAnomaly" -Target $OmniAnomaly

if ($Experiments) {
    New-DataJunction -Name "experiments" -Target $Experiments
}

