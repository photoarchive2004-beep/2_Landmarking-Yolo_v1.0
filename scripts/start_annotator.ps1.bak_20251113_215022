param(
  [string]$BasePath,
  [int]$Select = 0
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $here "..\..")).Path   # D:\GM
$cfg  = Join-Path $here "..\cfg"
$logs = Join-Path $here "..\logs"
New-Item -ItemType Directory -Force -Path $cfg,$logs | Out-Null
$lastBaseFile = Join-Path $cfg "last_base.txt"

# Диалог выбора папки
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$init = $BasePath
if (-not $init -and (Test-Path $lastBaseFile)) { $init = Get-Content $lastBaseFile -Raw }
if (-not $init -or -not (Test-Path $init)) { $init = Join-Path $root "photos" }

$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = "Choose localities root (folder with <locality>\png)"
$dlg.SelectedPath = $init
$dlg.ShowNewFolderButton = $false
$res = $dlg.ShowDialog()
if ($res -ne [System.Windows.Forms.DialogResult]::OK) {
  "[INFO] Canceled by user." | Out-File (Join-Path $logs "ps_launcher_last.log") -Encoding utf8
  exit 0
}
$base = $dlg.SelectedPath
Set-Content -Encoding UTF8 -NoNewline -Path $lastBaseFile -Value $base

# Собираем локальности (подпапки, где есть png/*.png)
$locs = Get-ChildItem -LiteralPath $base -Directory | ForEach-Object {
  $pngDir = Join-Path $_.FullName "png"
  if (Test-Path $pngDir) {
    $png = Get-ChildItem -LiteralPath $pngDir -File -Filter *.png -ErrorAction SilentlyContinue
    if ($png.Count -gt 0) {
      $csv = Get-ChildItem -LiteralPath $pngDir -File -Filter *.csv -ErrorAction SilentlyContinue
      $first = $png | Sort-Object Name | Select-Object -First 1
      $scaleOk = Test-Path (Join-Path $pngDir ("{0}.scale.csv" -f [System.IO.Path]::GetFileNameWithoutExtension($first.Name)))
      [PSCustomObject]@{
        Name    = $_.Name
        PngDir  = $pngDir
        Nimg    = $png.Count
        Nann    = $csv.Count
        Pct     = if ($png.Count) { [int](100*$csv.Count/$png.Count) } else { 0 }
        Scale   = if ($scaleOk) { "" } else { "Set Scale!" }
      }
    }
  }
} | Sort-Object Name

if (-not $locs -or $locs.Count -eq 0) {
  "[ERR] No localities with \png found in '{0}'" -f $base | Out-File (Join-Path $logs "ps_launcher_last.log") -Encoding utf8
  exit 2
}

# Меню (если не задан Select), иначе авто-выбор
$sel = $Select
if ($sel -lt 1 -or $sel -gt $locs.Count) {
  Write-Host ""
  Write-Host "== GM Landmarking: Points Annotator v1.0 ==" -ForegroundColor Cyan
  Write-Host ("[BASE] {0}" -f $base)
  Write-Host ""
  $i = 0
  foreach ($l in $locs) {
    $i++
    $scaleTxt = if ($l.Scale) { "  Set Scale!" } else { "" }
    Write-Host (" {0}) {1} [{2}/{3}] {4}%{5}" -f $i, $l.Name, $l.Nimg, $l.Nann, $l.Pct, $scaleTxt)
  }
  Write-Host ""
  $inp = Read-Host "Enter number (Q to quit)"
  if ($inp -match '^[Qq]$') { exit 0 }
  if (-not [int]::TryParse($inp, [ref]$sel)) { Write-Host "[ERR] Invalid selection."; exit 3 }
  if ($sel -lt 1 -or $sel -gt $locs.Count) { Write-Host "[ERR] Invalid selection."; exit 3 }
}

$chosen = $locs[$sel-1]
$pngDir = $chosen.PngDir
"[RUN] locality={0}; images={1}" -f $chosen.Name, $pngDir | Out-File (Join-Path $logs "ps_launcher_last.log") -Encoding utf8

# Python exe
$py = Join-Path $here "..\.venv_lm\Scripts\python.exe"
$pyArgs = @()
if (-not (Test-Path $py)) {
  if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py"; $pyArgs = @("-3") }
  else { $py = "python" }
}

# Запуск GUI
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $py
$psi.Arguments = ($pyArgs + @((Join-Path $here "..\annot_gui_custom.py"), "--root", $root, "--images", $pngDir)) -join " "
$psi.WorkingDirectory = (Join-Path $here "..")
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$proc = [System.Diagnostics.Process]::Start($psi)
$so = $proc.StandardOutput.ReadToEnd()
$se = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()
$exit = $proc.ExitCode

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
Set-Content  -Path (Join-Path $logs ("launcher_stdout_{0}.txt" -f $ts)) -Value $so -Encoding utf8
Set-Content  -Path (Join-Path $logs ("launcher_stderr_{0}.txt" -f $ts)) -Value $se -Encoding utf8
Add-Content  -Path (Join-Path $logs "annotator_run_last.txt") -Value ("RUN_TS={0}`r`nBASE={1}`r`nLOC={2}`r`nPNG={3}`r`nEXIT={4}`r`n" -f (Get-Date), $base, $chosen.Name, $pngDir, $exit) -Encoding utf8
exit $exit