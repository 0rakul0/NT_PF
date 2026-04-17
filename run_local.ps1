$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Nao encontrei $python. Crie a venv e instale as dependencias antes de rodar."
}

& $python .\run_local.py
