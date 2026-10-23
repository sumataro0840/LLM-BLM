# Runs S1 once, in a process that survives this session, after the outcome-free scoring has finished:
# waits until both data\cache\logs\smallcap_score.log and smallcap_llm.log end with "all done" (checked every 5 minutes).
#   powershell -ExecutionPolicy Bypass -File experiments\S1_smallcap_drift\start_after_scoring.ps1
# Progress: Get-Content data\cache\logs\s1.log -Tail 5
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$logs = Join-Path $repo "data\cache\logs"
$wait = @"
`$score = Join-Path '$logs' 'smallcap_score.log'
`$llm = Join-Path '$logs' 'smallcap_llm.log'
while (-not ((Select-String -Path `$score -Pattern '^all done' -Quiet) -and (Select-String -Path `$llm -Pattern '^all done' -Quiet))) {
    Start-Sleep -Seconds 300
}
Set-Location '$repo'
& '$repo\.venv\Scripts\python.exe' -W ignore experiments/S1_smallcap_drift/run.py > '$logs\s1.log' 2> '$logs\s1.err'
"@
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $wait) `
    -WindowStyle Hidden -PassThru
Write-Output "waiting in pid $($p.Id); S1 starts when the scoring has finished (log: data\cache\logs\s1.log)"
