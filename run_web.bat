@echo off
rem ======================================================================
rem  LuxTTS Studio
rem
rem  WRITING RULES FOR THIS FILE (learned the hard way, do not undo):
rem    1. CRLF line endings. LF-only makes double-click exit silently.
rem    2. Pure ASCII only. cmd reads .bat with the OEM code page (936 here);
rem       any non-ASCII byte sequence can be read at the wrong offset and
rem       produce bogus "'xxx' is not recognized as an internal command".
rem    3. Never switch the console code page in this file.
rem    Write it with: p.write_bytes(text.replace("\n","\r\n").encode("gbk"))
rem ======================================================================
setlocal
cd /d "%~dp0"

set "PYTHONPATH="
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=%~dp0runtime\python.exe"

if not exist "%PY%" (
  echo [ERROR] Missing runtime\python.exe
  echo   That is the bundled portable Python runtime, about 7.7 G.
  echo   See the "setting up on another machine" section in README.md.
  pause
  exit /b 1
)
if not exist "%~dp0web\server.py" (
  echo [ERROR] Missing web\server.py
  pause
  exit /b 1
)

rem The browser is opened by the server itself (--open). Do NOT try to read the
rem port out of config.py with "for /f" here: when the command string starts
rem with a double quote cmd refuses to run it, and any inner single quote
rem (sys.path.insert(0,'.')) truncates the command early. Both failures are
rem silent. The port's single source of truth stays app/config.py PORT --
rem server.py prints the same URL on startup.
echo Starting LuxTTS Studio ... it will open a browser tab by itself.
"%PY%" -s -W ignore::RuntimeWarning:pydub.utils "%~dp0web\server.py" --open
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [ERROR] Service exited with code %RC%. Scroll up for the reason.
  pause
)
endlocal
exit /b %RC%
