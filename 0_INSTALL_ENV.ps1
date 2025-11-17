param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvPath = Join-Path $root ".venv_lm"
$venvPy   = Join-Path $venvPath "Scripts\python.exe"

Write-Host "=== Installing YOLO env in: $venvPath ==="

if (!(Test-Path $venvPath)) {
    Write-Host "Creating virtual environment .venv_lm ..."

    $pyCandidates = @()

    # 1) Python из основного модуля 2_Landmarking_v1.0 (если есть)
    $mainPy = "D:\GM\tools\2_Landmarking_v1.0\.venv_lm\Scripts\python.exe"
    if (Test-Path $mainPy) {
        $pyCandidates += $mainPy
    }

    # 2) py из PATH
    $cmdPy = Get-Command py -ErrorAction SilentlyContinue
    if ($cmdPy) {
        $pyCandidates += "py"
    }

    # 3) python из PATH
    $cmdPython = Get-Command python -ErrorAction SilentlyContinue
    if ($cmdPython) {
        $pyCandidates += "python"
    }

    if ($pyCandidates.Count -eq 0) {
        throw "No Python found: neither main env nor 'py' nor 'python'."
    }

    $pyCmd = $pyCandidates[0]
    Write-Host "Using Python: $pyCmd"

    if ($pyCmd -eq "py") {
        & py -3 -m venv ".venv_lm"
    } else {
        & $pyCmd -m venv ".venv_lm"
    }
}

if (!(Test-Path $venvPy)) {
    throw "Virtual env python not found: $venvPy"
}

Write-Host "Upgrading pip/setuptools/wheel ..."
& $venvPy -m pip install --upgrade pip setuptools wheel

# Базовые зависимости для аннотатора + YOLO pose
$pkgs = @(
    "numpy",
    "pillow",
    "opencv-python",
    "pyyaml",
    "pandas",
    "matplotlib",
    "scipy",
    "ultralytics"
)

Write-Host "Installing packages:"
$pkgs | ForEach-Object { Write-Host "  - $_" }

& $venvPy -m pip install $pkgs

Write-Host "=== YOLO env installation finished successfully. ==="
