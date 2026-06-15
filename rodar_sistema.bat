@echo off
setlocal

cd /d "%~dp0"

if "%PF_SKIP_SYNC%"=="" set PF_SKIP_SYNC=true
if "%PF_PRESERVE_PREVIOUS_RUN%"=="" set PF_PRESERVE_PREVIOUS_RUN=false

echo ============================================================
echo Rodando sistema NT_PF
echo ============================================================
echo.
echo Fluxo padrao:
echo - sincronizar/gerar base: PF_SKIP_SYNC=%PF_SKIP_SYNC%
echo - preservar execucao anterior: PF_PRESERVE_PREVIOUS_RUN=%PF_PRESERVE_PREVIOUS_RUN%
echo - limpar artefatos anteriores
echo - amostra inicial 10%% com estratificacao temporal
echo - massa incremental 90%%
echo - lotes de 500 noticias
echo - Agente 2 gera discriminadores para a WNN
echo - regex deterministica desativada por padrao
echo - agente separado revisa a arvore ao final de cada lote
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
    echo Dashboard em: data\analise_qualitativa\dashboard_comparacao.html
) else (
    echo Execucao terminou com erro. Codigo: %EXIT_CODE%
)
echo.
pause
exit /b %EXIT_CODE%
