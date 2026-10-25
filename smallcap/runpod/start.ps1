# R1: runs smallcap\runpod\orchestrate.py (ChronoGPT on a RunPod pod for the replication sample) in a process that
# survives this session. It waits for the release fetch, so it can be started at any time.
#   powershell -ExecutionPolicy Bypass -File smallcap\runpod\start.ps1
# Progress: Get-Content data\cache\logs\runpod_rep.log -Tail 5
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$logs = Join-Path $repo "data\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$p = Start-Process -FilePath (Join-Path $repo ".venv\Scripts\python.exe") `
    -ArgumentList @("-W", "ignore", "smallcap\runpod\orchestrate.py") -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $logs "runpod_rep.out") -RedirectStandardError (Join-Path $logs "runpod_rep.err") `
    -WindowStyle Hidden -PassThru
Write-Output "started pid $($p.Id)  (log: data\cache\logs\runpod_rep.log)"
