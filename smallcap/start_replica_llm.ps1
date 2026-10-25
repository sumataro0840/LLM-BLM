# Replication sample (R1): gpt-4o-mini scoring (paid API, about 7 USD; approved by the user on 2026-10-25) of the new
# sample's releases filed from 2023-11-01, in a process that survives this session. Waits until the release fetch of
# smallcap\start_replica.ps1 has finished (rep_chain.log contains "done: rep_build_releases"), then runs alongside the
# GPU scoring.
#   powershell -ExecutionPolicy Bypass -File smallcap\start_replica_llm.ps1
# Progress: Get-Content data\cache\logs\rep_llm.log -Tail 3
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "data\cache\logs"
$job = @"
`$chain = Join-Path '$logs' 'rep_chain.log'
while (-not ((Test-Path `$chain) -and (Select-String -Path `$chain -Pattern 'done: rep_build_releases' -Quiet))) {
    if ((Test-Path `$chain) -and (Select-String -Path `$chain -Pattern 'failed' -Quiet)) { exit 1 }
    Start-Sleep -Seconds 300
}
`$env:SMALLCAP_DB = '$repo\data\smallcap_rep.sqlite'
`$env:SMALLCAP_CACHE = '$repo\data\cache\smallcap_rep'
`$env:PYTHONIOENCODING = 'utf-8'
Set-Location '$repo'
& '$repo\.venv\Scripts\python.exe' -W ignore -m smallcap.score llm > '$logs\rep_llm.log' 2> '$logs\rep_llm.err'
"@
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $job) `
    -WindowStyle Hidden -PassThru
Write-Output "waiting in pid $($p.Id); the gpt-4o-mini scoring starts after the release fetch (log: data\cache\logs\rep_llm.log)"
