param()

$ErrorActionPreference = "Stop"

$root = Get-Location
$logsDir = Join-Path $root "logs"
if (!(Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logsDir "sync_repo_yolo_$stamp.log"
Start-Transcript -Path $logFile -Force

try {
    Write-Host "Root: $root"

    # 1. Ensure git repo
    if (!(Test-Path ".git")) {
        git init
        git branch -M main
        Write-Host "Initialized new git repository with main branch."
    } else {
        Write-Host ".git folder already exists, using existing repo."
    }

    # 2. Configure origin
    $remoteUrl = "https://github.com/photoarchive2004-beep/2_Landmarking-Yolo_v1.0.git"
    $origin    = git remote get-url origin 2>$null
    if (-not $origin) {
        git remote add origin $remoteUrl
        Write-Host "Added origin: $remoteUrl"
    } elseif ($origin -ne $remoteUrl) {
        git remote set-url origin $remoteUrl
        Write-Host "Updated origin URL to: $remoteUrl"
    } else {
        Write-Host "Origin already set correctly."
    }

    # 3. Minimal .gitignore (не трогаем, если уже есть)
    $gitignorePath = Join-Path $root ".gitignore"
    if (!(Test-Path $gitignorePath)) {
@"
# Python virtual environments
.venv/
.venv_*/
env/
venv/

# Byte-compiled / cache
__pycache__/
*.py[cod]

# IDE / editors
.vscode/
.idea/

# OS junk
.DS_Store
Thumbs.db
"@ | Set-Content -Encoding UTF8 $gitignorePath
        Write-Host "Created basic .gitignore."
    } else {
        Write-Host ".gitignore already exists, not touching it."
    }

    # 4. Capture working tree (без .git)
    $treeFile = Join-Path $logsDir "tree_$stamp.txt"
    Get-ChildItem -Recurse -Force |
        Where-Object { $_.FullName -notlike "*\.git\*" } |
        Sort-Object FullName |
        ForEach-Object {
            $rel = Resolve-Path $_.FullName -Relative
            "{0}`t{1}`t{2:yyyy-MM-dd HH:mm:ss}" -f $rel, $_.Length, $_.LastWriteTime
        } | Set-Content -Encoding UTF8 $treeFile
    Write-Host "Tree saved to: $treeFile"

    # 5. git add / commit
    git add -A
    $changes = git status --porcelain
    if ($changes) {
        git commit -m "Sync local YOLO repo $stamp"
        Write-Host "Committed local changes."
    } else {
        Write-Host "Nothing to commit, working tree clean."
    }

    # 6. git push with output capture
    Write-Host "=== git push origin main ==="
    $pushLog = Join-Path $logsDir "git_push_$stamp.log"
    git push -u origin main 2>&1 | Tee-Object -FilePath $pushLog
    $pushExitCode = $LASTEXITCODE
    if ($pushExitCode -ne 0) {
        Write-Host "git push FAILED with exit code $pushExitCode" -ForegroundColor Red
    } else {
        Write-Host "git push succeeded."
        Write-Host "=== git ls-remote --heads origin ==="
        git ls-remote --heads origin | Tee-Object -FilePath (Join-Path $logsDir "git_lsremote_$stamp.log")
    }

    Write-Host "Sync attempt completed. Log: $logFile"
}
finally {
    Stop-Transcript
}
