# G3 sample (やること.md G3): fetch the releases, fundamentals and surprises of the 400 companies of data/smallcap_g3.sqlite
# and score them (dictionary, FinBERT on this PC; ChronoGPT on a RunPod pod started alongside by
# smallcap\runpod\start.ps1 with SMALLCAP_TAG=g3), in processes that survive this session. No stock reaction is computed.
#   powershell -ExecutionPolicy Bypass -File smallcap\start_g3.ps1
# Progress: Get-Content data\cache\logs\g3_chain.log ; Get-Content data\cache\logs\runpod_g3.log -Tail 5
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "data\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$chain = @"
`$env:SMALLCAP_DB = '$repo\data\smallcap_g3.sqlite'
`$env:SMALLCAP_CACHE = '$repo\data\cache\smallcap_g3'
`$env:PYTHONIOENCODING = 'utf-8'
Set-Location '$repo'
`$py = '$repo\.venv\Scripts\python.exe'
foreach (`$step in @(@('build','releases'), @('build','fundamentals'), @('build','surprise'), @('score','dictionary'), @('score','finbert'))) {
    `$name = 'g3_' + `$step[0] + '_' + `$step[1]
    & `$py -W ignore -m ('smallcap.' + `$step[0]) `$step[1] > ('$logs\' + `$name + '.log') 2> ('$logs\' + `$name + '.err')
    if (`$LASTEXITCODE -ne 0) { 'failed: ' + `$name | Out-File '$logs\g3_chain.log' -Append; exit 1 }
    'done: ' + `$name + ' ' + (Get-Date -Format s) | Out-File '$logs\g3_chain.log' -Append
}
# ChronoGPT: wait for the pod; its vectors land in data\cache\smallcap_g3, so this step finds them done (or computes the rest)
while (-not ((Test-Path '$logs\runpod_g3.log') -and (Select-String -Path '$logs\runpod_g3.log' -Pattern 'all done|failed' -Quiet))) { Start-Sleep -Seconds 120 }
& `$py -W ignore -m smallcap.score chrono_raw chrono_clean > '$logs\g3_score_chrono.log' 2> '$logs\g3_score_chrono.err'
if (`$LASTEXITCODE -ne 0) { 'failed: g3_score_chrono' | Out-File '$logs\g3_chain.log' -Append; exit 1 }
'done: g3_score_chrono ' + (Get-Date -Format s) | Out-File '$logs\g3_chain.log' -Append
'all done ' + (Get-Date -Format s) | Out-File '$logs\g3_chain.log' -Append
"@
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $chain) `
    -WindowStyle Hidden -PassThru
Write-Output "started pid $($p.Id)  (log: data\cache\logs\g3_chain.log)"
