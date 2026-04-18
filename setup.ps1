$ErrorActionPreference = "Stop"

param(
    [ValidateSet("local", "runtime", "extraction")]
    [string]$Profile = "local"
)

$pythonCandidates = @(
    @{ Command = "py -3.13"; Request = "3.13" },
    @{ Command = "py -3.12"; Request = "3.12" },
    @{ Command = "python"; Request = "python" }
)

$pythonCmd = $null
$pythonRequest = $null

foreach ($candidate in $pythonCandidates) {
    try {
        Invoke-Expression "$($candidate.Command) --version" | Out-Null
        $pythonCmd = $candidate.Command
        $pythonRequest = $candidate.Request
        break
    }
    catch {
    }
}

if (-not $pythonCmd) {
    throw "Python nao encontrado. Instale Python 3.12.x ou 3.13.x antes de continuar."
}

$venvPython = ".\.venv\Scripts\python.exe"
$hasUv = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)
$hasPyproject = Test-Path ".\pyproject.toml"
$hasUvLock = Test-Path ".\uv.lock"

Write-Host "[setup] perfil: $Profile"
Write-Host "[setup] usando interpretador: $pythonCmd"

if ($hasUv -and $hasPyproject) {
    Write-Host "[setup] usando uv com pyproject.toml..."
    $uvArgs = @("sync", "--python", $pythonRequest)

    if ($hasUvLock) {
        $uvArgs += "--frozen"
    }

    if ($Profile -in @("local", "extraction")) {
        $uvArgs += @("--group", "extraction")
    }

    & uv @uvArgs

    switch ($Profile) {
        "runtime" {
            Write-Host "[setup] ambiente pronto com uv para run_local.py, streamlit_app.py e provedores LLM."
        }
        "extraction" {
            Write-Host "[setup] ambiente pronto com uv para coleta, extracao, run_local.py e streamlit_app.py."
        }
        default {
            Write-Host "[setup] ambiente local validado pronto com uv."
        }
    }
    exit 0
}

if (-not (Test-Path $venvPython)) {
    Write-Host "[setup] criando ambiente virtual..."
    Invoke-Expression "$pythonCmd -m venv .venv"
}
else {
    Write-Host "[setup] ambiente virtual ja existe."
}

Write-Host "[setup] atualizando ferramentas base..."
& $venvPython -m pip install --upgrade pip setuptools wheel

$requirementsFile = switch ($Profile) {
    "runtime" { ".\requirements-runtime.txt" }
    "extraction" { ".\requirements.txt" }
    default {
        if (Test-Path ".\requirements-lock.txt") {
            ".\requirements-lock.txt"
        }
        else {
            ".\requirements.txt"
        }
    }
}

Write-Host "[setup] instalando dependencias de $requirementsFile..."
& $venvPython -m pip install --no-cache-dir -r $requirementsFile

Write-Host "[setup] validando consistencia do ambiente..."
& $venvPython -m pip check

switch ($Profile) {
    "runtime" {
        Write-Host "[setup] ambiente pronto para run_local.py, streamlit_app.py e provedores LLM remotos."
    }
    "extraction" {
        Write-Host "[setup] ambiente pronto para coleta, extracao, run_local.py e streamlit_app.py."
    }
    default {
        Write-Host "[setup] ambiente pronto com o perfil local validado."
    }
}
