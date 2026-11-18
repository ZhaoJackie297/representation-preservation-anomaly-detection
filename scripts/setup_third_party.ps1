param(
    [switch]$SkipSoftCLT,
    [switch]$SkipTS2VecCheck,
    [string]$SoftCLTSource,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$references = Join-Path $repoRoot "references"
$softcltRepo = Join-Path $references "softclt_repo"
$ts2vecRepo = Join-Path $references "ts2vec_official"
$repoRootFull = [System.IO.Path]::GetFullPath($repoRoot)
$softcltRepoFull = [System.IO.Path]::GetFullPath($softcltRepo)
if (-not $softcltRepoFull.StartsWith($repoRootFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "SoftCLT destination is outside the repository: $softcltRepoFull"
}

if (-not $SkipTS2VecCheck) {
    if (-not (Test-Path (Join-Path $ts2vecRepo "ts2vec.py"))) {
        throw "TS2Vec snapshot is missing: $ts2vecRepo"
    }
    Write-Host "TS2Vec snapshot found: $ts2vecRepo"
}

if (-not $SkipSoftCLT) {
    if (Test-Path $softcltRepo) {
        $isGitRepo = Test-Path (Join-Path $softcltRepo ".git")
        if (-not $isGitRepo -and -not $Force) {
            throw "SoftCLT destination exists but is not a Git repository. Rerun with -Force to replace it: $softcltRepo"
        }
        if ($Force) {
            if ($SoftCLTSource) {
                $sourceFull = [System.IO.Path]::GetFullPath($SoftCLTSource)
                if ($sourceFull.TrimEnd('\') -eq $softcltRepoFull.TrimEnd('\')) {
                    throw "SoftCLTSource and destination must differ."
                }
            }
            Remove-Item -LiteralPath $softcltRepo -Recurse -Force
        }
    }

    if (-not (Test-Path $softcltRepo)) {
        if ($SoftCLTSource) {
            if (-not (Test-Path (Join-Path $SoftCLTSource "softclt_ts2vec"))) {
                throw "SoftCLTSource is not a SoftCLT checkout: $SoftCLTSource"
            }
            robocopy $SoftCLTSource $softcltRepo /E /XD .git __pycache__ /NFL /NDL /NJH /NJS | Out-Null
            if ($LASTEXITCODE -ge 8) {
                throw "Failed to copy SoftCLT from $SoftCLTSource"
            }
            Copy-Item (Join-Path $SoftCLTSource "README.md") $softcltRepo -Force
        }
        else {
            git clone https://github.com/seunghan96/softclt.git $softcltRepo
        }
    }

    if (Test-Path (Join-Path $softcltRepo ".git")) {
        git -C $softcltRepo fetch --all --tags
        git -C $softcltRepo checkout 14c638979b129075d7a1111e9f529b9a275ea394
    }
    elseif (-not (Test-Path (Join-Path $softcltRepo "softclt_ts2vec\soft_ts2vec.py"))) {
        throw "SoftCLT checkout is incomplete: $softcltRepo"
    }
    Write-Host "SoftCLT checkout ready: $softcltRepo"
}

$env:TS2VEC_REPO = (Resolve-Path $ts2vecRepo).Path
Write-Host "TS2VEC_REPO=$env:TS2VEC_REPO"

