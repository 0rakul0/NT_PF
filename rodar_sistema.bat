@echo off
setlocal

cd /d "%~dp0"

if "%PF_SKIP_SYNC%"=="" set PF_SKIP_SYNC=false
if "%PF_PRESERVE_PREVIOUS_RUN%"=="" set PF_PRESERVE_PREVIOUS_RUN=true
if "%PF_LLM_PROVIDER%"=="" set PF_LLM_PROVIDER=openai
if "%PF_SAMPLE_FRACTION%"=="" set PF_SAMPLE_FRACTION=0.10
if "%PF_BATCH_SIZE%"=="" set PF_BATCH_SIZE=500
if "%PF_WNN_MAX_DISCRIMINATORS_PER_THEME%"=="" set PF_WNN_MAX_DISCRIMINATORS_PER_THEME=200
if "%PF_WNN_COMPACTION_INTERVAL_BATCHES%"=="" set PF_WNN_COMPACTION_INTERVAL_BATCHES=1
set PF_WNN_CONFIDENCE_THRESHOLD=0.50

echo ============================================================
echo Rodando sistema NT_PF
echo ============================================================
echo.
echo Fluxo padrao:
echo - sincronizar/gerar base: PF_SKIP_SYNC=%PF_SKIP_SYNC%
echo - preservar execucao anterior: PF_PRESERVE_PREVIOUS_RUN=%PF_PRESERVE_PREVIOUS_RUN%
echo - limpar artefatos anteriores
echo - provedor LLM: PF_LLM_PROVIDER=%PF_LLM_PROVIDER%
echo - fundacao: PF_SAMPLE_FRACTION=%PF_SAMPLE_FRACTION% dos registros mais antigos
echo - massa incremental: 90%% dos registros posteriores, em ordem cronologica
echo - tamanho do lote: PF_BATCH_SIZE=%PF_BATCH_SIZE% noticias
echo - memoria WNN: 35 marcadores iniciais por classe, expansivel ate %PF_WNN_MAX_DISCRIMINATORS_PER_THEME%
echo - metamorfose da memoria: ao fim de cada lote
echo - limiar de confianca WNN: PF_WNN_CONFIDENCE_THRESHOLD=%PF_WNN_CONFIDENCE_THRESHOLD%
echo - Agente 2 gera discriminadores para a WNN
echo - classificacao deterministica baseada em memoria WNN, com bleaching para desempate
echo - casos residuais seguem para o Agente 3
echo - exportar ocorrencias por crime e mes para analise temporal
echo - revisao da arvore tematica ao final da execucao
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
