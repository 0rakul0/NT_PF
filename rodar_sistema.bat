@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo Rodando sistema NT_PF
echo ============================================================
echo.
echo Fluxo padrao:
echo - sincronizar/gerar base
echo - limpar artefatos anteriores
echo - amostra inicial 30%% com estratificacao temporal
echo - massa incremental 70%%
echo - lotes de 10 noticias
echo - gerar metricas, graficos e README automatico
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

.\.venv\Scripts\python.exe -B .\rodar_sistema.py
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
    echo Execucao concluida com sucesso.
    echo Resultados em: data\analise_qualitativa\incremental
) else (
    echo Execucao terminou com erro. Codigo: %EXIT_CODE%
)
echo.
pause
exit /b %EXIT_CODE%
