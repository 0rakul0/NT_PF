@echo off
setlocal

cd /d "%~dp0"

if "%PF_DASH_HOST%"=="" set PF_DASH_HOST=127.0.0.1
if "%PF_DASH_PORT%"=="" set PF_DASH_PORT=8050

echo ============================================================
echo Rodando dashboard Dash NT_PF
echo ============================================================
echo.
echo URL: http://%PF_DASH_HOST%:%PF_DASH_PORT%
echo.

where uv >nul 2>nul
if errorlevel 1 (
    echo Erro: uv nao encontrado no PATH.
    echo Instale o uv antes de executar este projeto.
    echo.
    pause
    exit /b 1
)

echo Preparando ambiente Python...
uv sync --group extraction --group agents
if errorlevel 1 (
    echo Erro ao preparar ambiente.
    echo.
    pause
    exit /b 1
)
echo.

start "" "http://%PF_DASH_HOST%:%PF_DASH_PORT%"
.\.venv\Scripts\python.exe -B -m scripts.incremental.dashboard_dash

set EXIT_CODE=%ERRORLEVEL%
echo.
echo Dashboard encerrado. Codigo: %EXIT_CODE%
echo.
pause
exit /b %EXIT_CODE%
