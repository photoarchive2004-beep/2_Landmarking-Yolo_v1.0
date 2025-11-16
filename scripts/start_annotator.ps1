param(
  [string]$BasePath,
  [int]$Select = 0
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path          # ...\tools\2_Landmarking_v1.0\scripts
$root = (Resolve-Path (Join-Path $here "..\..")).Path            # D:\GM
$tool = (Resolve-Path (Join-Path $here "..")).Path               # ...\tools\2_Landmarking_v1.0
$cfg  = Join-Path $tool "cfg"
$logs = Join-Path $tool "logs"
New-Item -ItemType Directory -Force -Path $cfg,$logs | Out-Null

# 0) LM_number.txt must exist and be >1
$lmFile = Join-Path $tool "LM_number.txt"
if (!(Test-Path $lmFile)) { Set-Content -Path $lmFile -Value "16" -Encoding ASCII }
try {
  $N = [int](Get-Content $lmFile -Raw)
  if ($N -le 1) { throw "LM_number.txt must be >1" }
} catch {
  Set-Content -Path $lmFile -Value "16" -Encoding ASCII
}

# 1) РџР°РјСЏС‚СЊ РїСѓС‚Рё
$lastBaseFile = Join-Path $cfg "last_base.txt"
$init = $BasePath
if (-not $init -and (Test-Path $lastBaseFile)) { $init = Get-Content $lastBaseFile -Raw }
if (-not $init -or -not (Test-Path $init)) { $init = Join-Path $root "photos" }

# 2) Р”РёР°Р»РѕРі РІС‹Р±РѕСЂР° РїР°РїРєРё
Add-Type -AssemblyName System.Windows.Forms | Out-Null
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

# 3) Р›РѕРєР°Р»СЊРЅРѕСЃС‚Рё
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

# 4) РњРµРЅСЋ (РµСЃР»Рё РЅРµ Р·Р°РґР°РЅ Select)
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

# 5) Python exe
$py = Join-Path $tool ".venv_lm311\Scripts\python.exe"
if (!(Test-Path $py)) { $py = Join-Path $tool ".venv_lm311\Scripts\python.exe"
if (!(Test-Path $py)) { $py = Join-Path $tool ".venv_lm\Scripts\python.exe" }
if (!(Test-Path $py)) { if (Get-Command py -ErrorAction SilentlyContinue) { $py="py"; $pyArgs=@("-3") } else { $py="python" } } else { $py="python" } }
  else { $py = "python" }
}

# 6) Р—Р°РїСѓСЃРє GUI С‡РµСЂРµР· PowerShell &, С‡С‚РѕР±С‹ Р°СЂРіСѓРјРµРЅС‚С‹ СЃ РїСЂРѕР±РµР»Р°РјРё РїРµСЂРµРґР°Р»РёСЃСЊ РїСЂР°РІРёР»СЊРЅРѕ
$script = Join-Path $tool "annot_gui_custom.py"
$stdout = Join-Path $logs ("launcher_stdout_{0}.txt" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$stderr = Join-Path $logs ("launcher_stderr_{0}.txt" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

# Р·Р°С…РІР°С‚РёРј РІС‹РІРѕРґ РІ С„Р°Р№Р»С‹
$sw = [Diagnostics.Stopwatch]::StartNew()
& $py @pyArgs $script "--root" $root "--images" $pngDir 1> $stdout 2> $stderr
$exit = $LASTEXITCODE
$sw.Stop()

Add-Content -Path (Join-Path $logs "annotator_run_last.txt") -Value ("RUN_TS={0}`r`nBASE={1}`r`nLOC={2}`r`nPNG={3}`r`nEXIT={4}`r`nDUR_S={5:n3}`r`n" -f (Get-Date), $base, $chosen.Name, $pngDir, $exit, $sw.Elapsed.TotalSeconds) -Encoding utf8
exit $exit

