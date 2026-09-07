@echo off
setlocal

cd /d "%~dp0"
set PF_RESUME_RUN=false
set PF_RESET_RUN=true
call rodar_sistema.bat --reset
exit /b %ERRORLEVEL%
