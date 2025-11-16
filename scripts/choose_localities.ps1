param(
  [string]$Title = "Choose LOCALITIES base (folder with <locality>\\png)",
  [switch]$Silent
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms | Out-Null

# tool = ...\tools\2_Landmarking_v1.0
$tool = Split-Path -Parent $PSScriptRoot
# rootTools = ...\tools
$rootTools = Split-Path -Parent $tool
# grandRoot = ...\GM
$grandRoot = Split-Path -Parent $rootTools

$cfg  = Join-Path $tool "cfg"
$logs = Join-Path $tool "logs"
$last = Join-Path $cfg  "last_base.txt"
New-Item -ItemType Directory -Force -Path $cfg,$logs | Out-Null

# стартовая папка для диалога
$init = ""
if (Test-Path $last) {
    try { $init = (Get-Content $last -Raw).Trim() } catch { $init = "" }
}
if (-not $init) {
    $def = Join-Path $grandRoot "photos"
    if (Test-Path $def) { $init = $def } else { $init = $grandRoot }
}

$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = $Title
$dlg.SelectedPath = $init
$dlg.ShowNewFolderButton = $false
$res = $dlg.ShowDialog()

# Если нажали Cancel — тихо выходим, НИКАКИХ вопросов, exitcode=0
if ($res -ne [System.Windows.Forms.DialogResult]::OK) {
    if (-not $Silent) { Write-Host "[INFO] Localities base selection cancelled by user." }
    $log = Join-Path $logs "choose_localities_last.log"
    @(
      "TS=$(Get-Date)",
      "CANCELLED=1"
    ) | Out-File -FilePath $log -Encoding utf8
    exit 0
}

$chosen = $dlg.SelectedPath

# Пишем выбранную базу локальностей в cfg\last_base.txt
[System.IO.File]::WriteAllText($last, $chosen, (New-Object System.Text.UTF8Encoding($false)))

# Логируем факт (без всяких удалений/джанкшнов)
$log = Join-Path $logs "choose_localities_last.log"
@(
  "TS=$(Get-Date)",
  "CHOSEN=$chosen"
) | Out-File -FilePath $log -Encoding utf8

if (-not $Silent) {
    Write-Host "[OK] Localities base:" $chosen
}
exit 0
