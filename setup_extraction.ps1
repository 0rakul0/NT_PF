$ErrorActionPreference = "Stop"

$pythonCandidates = @(
    "py -3.13",
    "py -3.12",
    "python"
)

$pythonCmd = $null

foreach ($candidate in $pythonCandidates) {
    try {
        Invoke-Expression "$candidate --version" | Out-Null
        $pythonCmd = $candidate
        break
    }
    catch {
    }
}

if (-not $pythonCmd) {
    throw "Python nao encontrado. Instale Python 3.12.x ou 3.13.x antes de continuar."
}

Write-Host "[setup-extraction] usando interpretador: $pythonCmd"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[setup-extraction] criando ambiente virtual..."
    Invoke-Expression "$pythonCmd -m venv .venv"
}
else {
    Write-Host "[setup-extraction] ambiente virtual ja existe."
}

Write-Host "[setup-extraction] atualizando ferramentas base..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

Write-Host "[setup-extraction] fixando pydantic e pydantic_core compativeis..."
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --only-binary=:all: pydantic==2.12.5 pydantic_core==2.41.5

Write-Host "[setup-extraction] instalando base de analise e dashboard..."
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r .\requirements-runtime.txt

Write-Host "[setup-extraction] instalando pilha de extracao..."
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r .\requirements-extraction.txt

Write-Host "[setup-extraction] validando consistencia do ambiente..."
.\.venv\Scripts\python.exe -m pip check

Write-Host "[setup-extraction] ambiente pronto para coleta, extracao, run_local.py e streamlit_app.py."
