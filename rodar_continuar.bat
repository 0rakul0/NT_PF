@echo off
setlocal

cd /d "%~dp0"
set PF_RESUME_RUN=true
set PF_RESET_RUN=false
call rodar_sistema.bat
exit /b %ERRORLEVEL%
