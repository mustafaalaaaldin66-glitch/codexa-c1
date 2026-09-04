$ErrorActionPreference = "Stop"
$env:HF_HOME = "D:\hf_cache"
$env:PIP_CACHE_DIR = "D:\pip_cache"

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "     CODEXA C1 V9 AUTOMATED MASTER PIPELINE       " -ForegroundColor Cyan
Write-Host "==================================================`n" -ForegroundColor Cyan

# 1. Check Diagnostics
Write-Host "[1/2] Executing System Diagnostics..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe doctor.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Diagnostics failed. Pipeline stopped." -ForegroundColor Red
    exit
}

# 2. Run Data Pipeline Automatically
Write-Host "`n[2/2] Launching Step 9 Data Pipeline..." -ForegroundColor Green
$pipelineFile = Get-ChildItem .\data_pipeline\*.py | Select-Object -First 1

if ($pipelineFile) {
    Write-Host "Found pipeline script: $($pipelineFile.Name)" -ForegroundColor Gray
    .\.venv\Scripts\python.exe $pipelineFile.FullName
} else {
    Write-Host "[WARNING] No script found directly in data_pipeline folder. Directory contents:" -ForegroundColor Red
    Get-ChildItem .\data_pipeline\
}
