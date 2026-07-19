# Launches the seismic analysis app in your browser.
# Usage:  .\run_app.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$streamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $streamlit)) {
    Write-Error "No virtual environment found. Create one first:`n  uv venv --python 3.14 .venv`n  uv pip install -r requirements.txt"
}

& $streamlit run (Join-Path $PSScriptRoot "app.py")
