@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo Gerar base de dados de noticias da PF
echo ============================================================
echo.
echo Este arquivo baixa/sincroniza as noticias publicas da PF e gera:
echo - data\pf_operacoes_index.csv
echo - data\pf_operacoes_conteudos.csv
echo - data\noticias_markdown\
echo.

if not exist ".\.venv\Scripts\python.exe" (
    echo Ambiente virtual nao encontrado em .venv.
    echo.
    where uv >nul 2>nul
    if errorlevel 1 (
        echo Erro: uv nao encontrado no PATH.
        echo Instale o uv ou crie o ambiente .venv antes de executar.
        echo.
        pause
        exit /b 1
    )

    echo Preparando ambiente Python para extracao...
    uv sync --group extraction
    if errorlevel 1 (
        echo Erro ao preparar ambiente.
        echo.
        pause
        exit /b 1
    )
    echo.
)

echo Iniciando geracao/sincronizacao da base...
echo.

".\.venv\Scripts\python.exe" -B ".\scripts\pf_operacoes_pipeline.py"
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
    echo Base gerada/sincronizada com sucesso.
    echo.
    echo Arquivos esperados:
    echo - data\pf_operacoes_index.csv
    echo - data\pf_operacoes_conteudos.csv
    echo - data\noticias_markdown\
) else (
    echo Geracao da base terminou com erro. Codigo: %EXIT_CODE%
)

echo.
pause
exit /b %EXIT_CODE%
