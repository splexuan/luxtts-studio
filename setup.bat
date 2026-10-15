@echo off
rem ======================================================================
rem  LuxTTS Studio - one-time environment check
rem
rem  WRITING RULES FOR THIS FILE (learned the hard way, do not undo):
rem    1. CRLF line endings. LF-only makes double-click exit silently.
rem    2. Pure ASCII only. cmd reads .bat with the OEM code page (936 here);
rem       any non-ASCII byte sequence can be read at the wrong offset and
rem       produce bogus "'xxx' is not recognized as an internal command".
rem    3. Never switch the console code page in this file.
rem    Write it with: p.write_bytes(text.replace("\n","\r\n").encode("gbk"))
rem
rem  What this does NOT do: download the 7.7 G portable runtime, the model
rem  checkpoints, or the Whisper weights. Those are manual steps, listed by
rem  this script with the exact folder to drop them into. A setup script
rem  that silently fetches 11 GB is worse than one that tells you what is
rem  missing.
rem ======================================================================
setlocal
cd /d "%~dp0"

set "PYTHONPATH="
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=%~dp0runtime\python.exe"

echo ==============================================================
echo  LuxTTS Studio - environment check
echo ==============================================================
echo.

rem ---- 1. portable runtime ------------------------------------------
if exist "%PY%" goto have_runtime
echo [MISSING] runtime\python.exe
echo   This is the bundled portable Python 3.12 + all dependencies (~7.7 G).
echo   Copy the whole "runtime" folder from a working machine into:
echo     %~dp0
echo.
pause
exit /b 1
:have_runtime
echo [ok] runtime\python.exe
"%PY%" -c "import sys;print('       Python', sys.version.split()[0])" || goto bad_python

rem ---- 2. source tree -----------------------------------------------
if exist "%~dp0web\server.py" goto have_web
echo [MISSING] web\server.py  -- the repository seems incomplete.
pause
exit /b 1
:have_web
echo [ok] web\server.py

rem ---- 3. upstream code ----------------------------------------------
if exist "%~dp0upstream\zipvoice\__init__.py" goto have_upstream
echo [MISSING] upstream\zipvoice\
echo   Copy the "upstream" folder (it is the LuxTTS / ZipVoice source).
goto ask_missing
:have_upstream
echo [ok] upstream\zipvoice\

rem ---- 4. model checkpoints -----------------------------------------
if exist "%~dp0checkpoints\config.json" goto have_ckpt
echo [MISSING] checkpoints\config.json
echo   Model weights (model.pt, *.onnx, tokens.txt) plus the vocoder.
echo   Roughly 1.1 G. Copy the whole "checkpoints" folder into:
echo     %~dp0
goto ask_missing
:have_ckpt
echo [ok] checkpoints\config.json

rem ---- 5. whisper (needed for transcription when building a voice) ---
if exist "%~dp0whisper-base" goto have_whisper
echo [WARN] whisper-base\ not found.
echo   Without it the UI cannot transcribe a reference recording, so you will
echo   have to type the reference text by hand when creating a voice.
echo   Copy the "whisper-base" folder into:
echo     %~dp0
goto ask_missing
:have_whisper
echo [ok] whisper-base\

echo.
echo [MISSING] nothing else is required. You can start with run_web.bat.
goto done

rem ---- hardware / dependency report ----------------------------------
:bad_python
echo.
echo [ERROR] runtime\python.exe failed to start. The portable runtime is
echo   probably incomplete or was copied while in use.
pause
exit /b 1

:ask_missing
echo.
echo --------------------------------------------------------------
echo  See the "setting up on another machine" section in README.md
echo  for where each piece comes from.
echo --------------------------------------------------------------
pause
exit /b 1

:done
echo.
echo --------------------------------------------------------------
echo  All good. Start the studio with:
echo    run_web.bat
echo --------------------------------------------------------------
echo.
endlocal
exit /b 0