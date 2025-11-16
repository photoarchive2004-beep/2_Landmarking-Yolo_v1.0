@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem ---- Paths ----
set "TOOL_DIR=%~dp0"
set "LOG_DIR=%TOOL_DIR%logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

rem ---- Localities base picker (Windows dialog) ----
powershell -NoProfile -ExecutionPolicy Bypass -File "%TOOL_DIR%scripts\choose_localities.ps1" -Silent
if errorlevel 1 (
  echo [WARN] Localities base picker failed or was cancelled.>>"%LOG_DIR%trainer_last.log"
  echo [WARN] Localities base picker failed or was cancelled.
  pause
  exit /b 1
)

rem ---- Resolve PHOTOS_DIR from cfg\last_base.txt, if present ----
set "CFG_DIR=%TOOL_DIR%cfg"
set "LAST_BASE=%CFG_DIR%\last_base.txt"
set "PHOTOS_DIR="

if exist "%LAST_BASE%" (
  set /p PHOTOS_DIR=<"%LAST_BASE%"
)

if not defined PHOTOS_DIR (
  echo [ERR] Localities base path not found in cfg\last_base.txt.>>"%LOG_DIR%trainer_last.log"
  echo [ERR] Localities base path not found in cfg\last_base.txt.
  pause
  exit /b 1
)

rem ---- Python resolver ----
set "PY=%TOOL_DIR%.venv_lm\Scripts\python.exe"
if not exist "%PY%" (
  where py >nul 2>&1 && (set "PY=py -3")
)

if /I "%PY%"=="py -3" (
  py -3 -c "import sys" >nul 2>&1 || set "PY="
)

if not defined PY (
  where python >nul 2>&1 && (set "PY=python")
)

if not defined PY (
  echo [ERR] Landmarking environment not found.>>"%LOG_DIR%trainer_last.log"
  echo [ERR] Landmarking environment not found.
  echo Run 0_INSTALL_ENV.ps1.
  pause
  exit /b 1
)

title == GM Landmarking: HRNet Trainer v1.0 ==
echo == GM Landmarking: HRNet Trainer v1.0 ==

rem ---- Initialization: structure + localities status ----
"%PY%" "%TOOL_DIR%scripts\init_structure.py" 1>>"%LOG_DIR%init_trainer_last.log" 2>&1
"%PY%" "%TOOL_DIR%scripts\rebuild_localities_status.py" 1>>"%LOG_DIR%status_trainer_last.log" 2>&1

rem ---- Launch trainer menu (Python) ----
"%PY%" "%TOOL_DIR%scripts\trainer_menu.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo [ERR] Trainer menu exited with code %RC%.>>"%LOG_DIR%trainer_last.log"
  echo [ERR] Trainer menu exited with code %RC%.
  pause
)

endlocal
exit /b 0
