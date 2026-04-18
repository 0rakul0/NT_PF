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

Write-Host "[setup] usando interpretador: $pythonCmd"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "[setup] criando ambiente virtual..."
    Invoke-Expression "$pythonCmd -m venv .venv"
}
else {
    Write-Host "[setup] ambiente virtual ja existe."
}

Write-Host "[setup] atualizando ferramentas base de instalacao..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

Write-Host "[setup] fixando pydantic e pydantic_core compativeis..."
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --only-binary=:all: pydantic==2.12.5 pydantic_core==2.41.5

if (Test-Path ".\requirements-lock.txt") {
    Write-Host "[setup] instalando dependencias travadas de requirements-lock.txt..."
    .\.venv\Scripts\python.exe -m pip install --no-cache-dir -r .\requirements-lock.txt
}
else {
    Write-Host "[setup] instalando dependencias de requirements.txt..."
    .\.venv\Scripts\python.exe -m pip install --no-cache-dir -r .\requirements.txt
}

Write-Host "[setup] validando consistencia do ambiente..."
.\.venv\Scripts\python.exe -m pip check

Write-Host "[setup] ambiente pronto."
