param()

$ErrorActionPreference = "Stop"

$root = Get-Location
Write-Host "Repo root: $root"

# --- Файлы-хвосты от старого HRNet/MMPose стека ---
$hrnetJunk = @(
    "scripts\_check_hrnet_quick.py",
    "scripts\_check_mmpose_hrnet_import.py",
    "scripts\_check_mmpose_hrnet_import_final.py",
    "scripts\_check_mmpose_hrnet_import_final2.py",
    "scripts\_check_mmpose_install.py",
    "scripts\_check_mmpose_only.py",
    "scripts\_patch_hrnet_heatmap_shapes.py",
    "scripts\_patch_hrnet_heatmaps_and_pck.py",
    "scripts\_patch_hrnet_import_raw.py",
    "scripts\_patch_mmcv_ext_loader_stub.py",
    "scripts\_patch_mmpose_heads_init_for_gm.py",
    "scripts\_patch_mmpose_models_init_for_gm.py",
    "scripts\_patch_xtcocotools_mask_stub.py",
    "scripts\_show_mmpose_stack_versions.py",
    "scripts\_show_stack_after_full_fix.py",
    "scripts\_show_stack_after_numpy_xtc_fix.py",
    "scripts\_show_stack_before_full_fix.py",
    "scripts\_show_stack_before_numpy_xtc_fix.py",
    "scripts\_show_xtcocotools_stack.py",
    "scripts\train_hrnet_backup_before_debug.py"
)

Write-Host "=== Removing old HRNet/MMPose helper scripts ==="
foreach ($rel in $hrnetJunk) {
    $full = Join-Path $root $rel
    if (Test-Path $full) {
        Write-Host "  Removing $rel"
        Remove-Item $full -Force
    } else {
        Write-Host "  Skipping (not found): $rel"
    }
}

# 3. Чистим хвосты логов именно от этих скриптов
$logsDir = Join-Path $root "logs"
if (Test-Path $logsDir) {
    Write-Host "=== Cleaning HRNet-diagnostic logs in logs/ ==="
    Get-ChildItem $logsDir -File |
        Where-Object {
            $_.Name -like "code_D_GM_tools_2_Landmarking-Yolo_v1.0_scripts__check_*" -or
            $_.Name -like "code_D_GM_tools_2_Landmarking-Yolo_v1.0_scripts__patch_*" -or
            $_.Name -like "code_D_GM_tools_2_Landmarking-Yolo_v1.0_scripts__show_*" -or
            $_.Name -like "cleanup_hrnet_*"
        } |
        ForEach-Object {
            Write-Host "  Removing log:" $_.Name
            Remove-Item $_.FullName -Force
        }
} else {
    Write-Host "logs/ folder not found, skip log cleanup."
}

Write-Host "=== Local cleanup finished ==="

# 4. Синхронизация с GitHub через штатный скрипт
if (Test-Path ".\sync_repo_yolo_v3.ps1") {
    Write-Host ""
    Write-Host "=== Running sync_repo_yolo_v3.ps1 (git add/commit/push + ls-remote) ==="
    powershell -NoProfile -ExecutionPolicy Bypass -File ".\sync_repo_yolo_v3.ps1"
} else {
    Write-Host "[WARN] sync_repo_yolo_v3.ps1 not found, skipping auto-push."
}
