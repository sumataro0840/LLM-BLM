# R1: runs S1 (run.py, blm_paths.py), S2, S3, S4 and R1 once on the replication sample, in a process that survives
# this session, after the data chain (smallcap\start_replica.ps1) and the gpt-4o-mini scoring have finished:
# waits until data\cache\logs\rep_chain.log and rep_llm.log contain "all done" (checked every 5 minutes).
#   powershell -ExecutionPolicy Bypass -File experiments\R1_replication\start_analysis.ps1
# Progress: Get-Content data\cache\logs\r1_chain.log   (the chain stops at the first step that fails)
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$logs = Join-Path $repo "data\cache\logs"
$wait = @"
`$chain = Join-Path '$logs' 'rep_chain.log'
`$llm = Join-Path '$logs' 'rep_llm.log'
while (-not ((Test-Path `$chain) -and (Select-String -Path `$chain -Pattern 'all done' -Quiet) -and (Test-Path `$llm) -and (Select-String -Path `$llm -Pattern 'all done' -Quiet))) {
    if ((Test-Path `$chain) -and (Select-String -Path `$chain -Pattern 'failed' -Quiet)) { 'data chain failed' | Out-File '$logs\r1_chain.log' -Append; exit 1 }
    Start-Sleep -Seconds 300
}
`$env:SMALLCAP_DB = '$repo\data\smallcap_rep.sqlite'
`$env:SMALLCAP_CACHE = '$repo\data\cache\smallcap_rep'
`$env:SMALLCAP_RESULTS = 'results_rep'
`$env:PYTHONIOENCODING = 'utf-8'
Set-Location '$repo'
`$py = '$repo\.venv\Scripts\python.exe'
foreach (`$step in @('S1_smallcap_drift/run.py', 'S1_smallcap_drift/blm_paths.py', 'S2_smallcap_risk/run.py', 'S3_smallcap_risk_blm/run.py', 'S4_te_calibration/run.py', 'R1_replication/run.py')) {
    `$name = 'r1_' + (`$step -replace '[/.]', '_')
    & `$py -W ignore ('experiments/' + `$step) > ('$logs\' + `$name + '.log') 2> ('$logs\' + `$name + '.err')
    if (`$LASTEXITCODE -ne 0) { 'failed: ' + `$step | Out-File '$logs\r1_chain.log' -Append; exit 1 }
    'done: ' + `$step + ' ' + (Get-Date -Format s) | Out-File '$logs\r1_chain.log' -Append
}
'all done ' + (Get-Date -Format s) | Out-File '$logs\r1_chain.log' -Append
"@
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $wait) `
    -WindowStyle Hidden -PassThru
Write-Output "waiting in pid $($p.Id); R1 starts when the data and the scores are ready (log: data\cache\logs\r1_chain.log)"
