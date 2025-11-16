param(
  [string]$Title = "Choose localities root (folder with <locality>\png)",
  [string]$InitialPath = "",
  [string]$OutFile = ""
)
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = $Title
if ($InitialPath -and (Test-Path $InitialPath)) { $dlg.SelectedPath = $InitialPath }
$dlg.ShowNewFolderButton = $false
$res = $dlg.ShowDialog()
if ($res -eq [System.Windows.Forms.DialogResult]::OK) {
  if ($OutFile) {
    $enc = New-Object System.Text.UTF8Encoding($false) # no BOM
    [System.IO.File]::WriteAllText($OutFile, $dlg.SelectedPath, $enc)
  } else {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dlg.SelectedPath
  }
  exit 0
} else {
  exit 1
}