param([switch]$Offline)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Project environment is missing. Follow RUNNING.md to create .venv first.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Offline) { $env:FRAMEFIND_OFFLINE = '1' }
    & $taskPython -m streamlit run app.py
} finally {
    Pop-Location
}

