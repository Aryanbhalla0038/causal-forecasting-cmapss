# Run the full Causal Forecasting pipeline end-to-end.
#
# Usage (from the repo root):
#   .\run_all.ps1                # experiments + report + tests + dashboard
#   .\run_all.ps1 -SkipDashboard # everything except launching streamlit
#   .\run_all.ps1 -SkipTests     # skip pytest
#   .\run_all.ps1 -Clean         # delete prior artifacts before running
#
# If PowerShell blocks the script the first time, allow it once with:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

param(
    [string]$Python = 'C:\Users\HP\AppData\Local\Python\bin\python.exe',
    [switch]$SkipDashboard,
    [switch]$SkipTests,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

function Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

if (-not (Test-Path $Python)) {
    $fallback = (Get-Command python -ErrorAction SilentlyContinue)
    if ($null -ne $fallback) {
        $Python = $fallback.Source
        Write-Host "Configured Python not found; falling back to $Python" -ForegroundColor Yellow
    } else {
        throw "Python interpreter not found. Pass -Python <path> or install Python 3.11+."
    }
}

if ($Clean) {
    Step "Cleaning prior artifacts"
    Remove-Item data\processed\cmapss_*_multiunit -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item data\processed\cmapss_all_datasets_summary.csv -Force -ErrorAction SilentlyContinue
    Remove-Item reports\figures\*multiunit* -Force -ErrorAction SilentlyContinue
    Remove-Item reports\final_report.md -Force -ErrorAction SilentlyContinue
    Remove-Item reports\final_metrics.csv -Force -ErrorAction SilentlyContinue
}

Step "Installing project (editable)"
& $Python -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Step "Running FD001-FD004 experiments"
& $Python scripts/run_all_cmapss_experiments.py
if ($LASTEXITCODE -ne 0) { throw "experiments failed" }

Step "Generating final report"
& $Python scripts/generate_final_report.py
if ($LASTEXITCODE -ne 0) { throw "report generation failed" }

if (-not $SkipTests) {
    Step "Running tests"
    & $Python -m pytest tests -q
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}

Write-Host ""
Write-Host "All artifacts ready:" -ForegroundColor Green
Write-Host "  data/processed/cmapss_all_datasets_summary.csv"
Write-Host "  reports/final_report.md"
Write-Host "  reports/final_metrics.csv"
Write-Host "  reports/figures/*.html"

if (-not $SkipDashboard) {
    Step "Launching Streamlit dashboard at http://localhost:8501"
    & $Python -m streamlit run apps/streamlit_dashboard.py
}
