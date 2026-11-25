@echo off
REM AquaSentinel desktop client - no browser, no server, no websocket.
cd /d "%~dp0"
set NUMBA_DISABLE_JIT=1
set KMP_DUPLICATE_LIB_OK=TRUE
set "PYTHON_EXE=python"
if exist "C:\Users\hp\OneDrive\Desktop\lab\tf_env\Scripts\python.exe" (
    set "PYTHON_EXE=C:\Users\hp\OneDrive\Desktop\lab\tf_env\Scripts\python.exe"
) else if exist "..\lab\tf_env\Scripts\python.exe" (
    set "PYTHON_EXE=..\lab\tf_env\Scripts\python.exe"
) else if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
)
"%PYTHON_EXE%" -u -m desktop %*
