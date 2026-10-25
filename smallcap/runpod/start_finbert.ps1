# R1: runs smallcap\runpod\finbert.py (FinBERT on a RunPod pod, then the rest of the PC scoring) in a process that
# survives this session. Start it after stopping the PC scoring chain.
#   powershell -ExecutionPolicy Bypass -File smallcap\runpod\start.ps1
# Progress: Get-Content data\cache\logs\runpod_finbert.log -Tail 5
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$logs = Join-Path $repo "data\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$p = Start-Process -FilePath (Join-Path $repo ".venv\Scripts\python.exe") `
    -ArgumentList @("-W", "ignore", "smallcap\runpod\finbert.py") -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $logs "runpod_finbert.out") -RedirectStandardError (Join-Path $logs "runpod_finbert.err") `
    -WindowStyle Hidden -PassThru
Write-Output "started pid $($p.Id)  (log: data\cache\logs\runpod_finbert.log)"
