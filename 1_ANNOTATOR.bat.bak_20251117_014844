@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem ---- Paths ----
set "HERE=%~dp0"
for %%I in ("%HERE%\..\..") do set "ROOT=%%~fI"
set "TOOL_DIR=%ROOT%\tools\2_Landmarking_v1.0"
set "PHOTOS_DIR=%ROOT%\photos"
set "LOG_DIR=%TOOL_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

rem ---- Localities base picker (Windows dialog) ----
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\choose_localities.ps1" -Silent
if errorlevel 1 (
    echo [WARN] Localities base picker failed or was cancelled.>>"%LOG_DIR%\annotator_last.log"
    echo [WARN] Localities base picker failed or was cancelled. Using existing "%PHOTOS_DIR%".
)

rem ---- Resolve PHOTOS_DIR from cfg\last_base.txt, if present ----
set "CFG_DIR=%TOOL_DIR%\cfg"
set "LAST_BASE=%CFG_DIR%\last_base.txt"
if exist "%LAST_BASE%" (
    set /p PHOTOS_DIR=<"%LAST_BASE%"
)
if not defined PHOTOS_DIR (
    set "PHOTOS_DIR=%ROOT%\photos"
)

rem ---- Python resolver ----
set "PY=%TOOL_DIR%\.venv_lm\Scripts\python.exe"
if not exist "%PY%" (
    where.exe py >nul 2>&1 && (set "PY=py -3")
)
if /I "%PY%"=="py -3" (
    py -3 -c "import sys" >nul 2>&1 || set "PY="
)
if not defined PY (
    where.exe python >nul 2>&1 && (set "PY=python")
)
if not defined PY (
    echo [ERR] Landmarking environment not found.>>"%LOG_DIR%\annotator_last.log"
    echo [ERR] Landmarking environment not found.
    echo Run 0_INSTALL_ENV.ps1.
    pause
    exit /b 1
)

title == GM Landmarking: Points Annotator v1.0 ==
echo == GM Landmarking: Points Annotator v1.0 ==

if not exist "%PHOTOS_DIR%" (
    echo [ERR] Photos dir not found: %PHOTOS_DIR%
    pause
    exit /b 1
)

rem ---- Initialization: structure + localities status ----
"%PY%" "%TOOL_DIR%\scripts\init_structure.py" 1>>"%LOG_DIR%\init_annotator_last.log" 2>&1
"%PY%" "%TOOL_DIR%\scripts\rebuild_localities_status.py" 1>>"%LOG_DIR%\status_annotator_last.log" 2>&1

rem ---- Menu ----
echo.

rem If called from trainer in REVIEW_AUTO mode, force given locality
if /I "%GM_MODE%"=="REVIEW_AUTO" (
    if defined GM_LOCALITY (
        set "SEL_LOC=%GM_LOCALITY%"
    )
)

if not defined SEL_LOC (
    %PY% "%TOOL_DIR%\scripts\menu_list.py" --print --root "%ROOT%"
    set /p CH=
    if /I "!CH!"=="Q" goto :EOF

    set "TMP_SEL=%TEMP%\gm_sel_%RANDOM%.txt"
    %PY% "%TOOL_DIR%\scripts\menu_list.py" --pick !CH! --root "%ROOT%" 1> "!TMP_SEL!" 2> "%LOG_DIR%\menu_pick_last.err"

    if exist "!TMP_SEL!" (
        set /p SEL_LOC=<"!TMP_SEL!"
        del /q "!TMP_SEL!" 2>nul
    )

    if not defined SEL_LOC (
        echo [ERR] Invalid selection.
        echo See "%LOG_DIR%\menu_pick_last.err"
        pause
        exit /b 2
    )
)

set "PNG_DIR=%PHOTOS_DIR%\!SEL_LOC!\png"
if not exist "!PNG_DIR!" (
    echo [ERR] Locality path not found: !PNG_DIR!
    pause
    exit /b 2
)

rem ---- Find first PNG robustly (sorted)
set "FIRST_PNG="
for /f "usebackq delims=" %%F in (`dir /b /a-d "!PNG_DIR!\*.png" ^| sort`) do (
    set "FIRST_PNG=%%F"
    goto _fp_done
)
:_fp_done
if not defined FIRST_PNG (
    echo [ERR] No PNG files in !PNG_DIR!
    pause
    exit /b 3
)

rem ---- Auto Scale Wizard (no questions)
if not exist "!PNG_DIR!\!FIRST_PNG!.scale.csv" (
    echo [INFO] No SCALE for "!FIRST_PNG!" -> starting Scale Wizard...
    set "GUI_LOG=%LOG_DIR%\gui_scale_last.log"
    %PY% "%TOOL_DIR%\annot_gui_custom.py" --root "%ROOT%" --images "!PNG_DIR!" --start-from "!FIRST_PNG!" --scale-wizard 1> "!GUI_LOG!" 2>&1
    if not exist "!PNG_DIR!\!FIRST_PNG!.scale.csv" (
        echo [ERR] Scale file still missing after wizard.
        echo See "!GUI_LOG!" below:
        type "!GUI_LOG!" | more
        pause
    )
)

rem ---- Launch custom GUI (always)
set "GUI_LOG=%LOG_DIR%\gui_run_last.log"
%PY% "%TOOL_DIR%\scripts\annotator_wrapper.py" --root "%ROOT%" --images "!PNG_DIR!" 1> "!GUI_LOG!" 2>&1

set "RC=%ERRORLEVEL%"
if not "!RC!"=="0" (
    echo [ERR] GUI exited with code !RC!. See "!GUI_LOG!" below:
    type "!GUI_LOG!" | more
    pause
)

exit /b 0


