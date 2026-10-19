# Runs the no-outcome scoring steps of the fix experiments in a detached process (survives closing the editor).
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$logs = Join-Path $repo "data\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$p = Start-Process -FilePath (Join-Path $repo ".venv\Scripts\python.exe") `
    -ArgumentList @("experiments/F_fixes/score.py", "stats", "dictionary", "finbert", "chrono_raw", "chrono") `
    -WorkingDirectory $repo -RedirectStandardOutput (Join-Path $logs "fixes_score.log") `
    -RedirectStandardError (Join-Path $logs "fixes_score.err") -WindowStyle Hidden -PassThru
Write-Output "started pid $($p.Id)"
