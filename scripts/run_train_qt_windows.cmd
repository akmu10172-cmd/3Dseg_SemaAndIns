@echo off
setlocal

REM Run PySide6 launcher from Windows by delegating to WSL.
REM Usage:
REM   scripts\run_train_qt_windows.cmd
REM   scripts\run_train_qt_windows.cmd /home/ysy/miniconda3/envs/opengaussian/bin/python

set PY_BIN=%~1
if "%PY_BIN%"=="" set PY_BIN=/home/ysy/miniconda3/envs/opengaussian/bin/python

set PROJ_WSL=/mnt/d/3Dseg_SemaAndIns
set RUN_SH=%PROJ_WSL%/scripts/run_train_qt.sh

echo [INFO] Using WSL python: %PY_BIN%
echo [INFO] Running: %RUN_SH%

wsl.exe -e bash -lc "cd %PROJ_WSL% && bash scripts/run_train_qt.sh %PY_BIN%"
set CODE=%ERRORLEVEL%
if not "%CODE%"=="0" (
  echo [ERROR] run_train_qt failed with code %CODE%
  exit /b %CODE%
)

echo [DONE] train_qt exited normally.
exit /b 0

