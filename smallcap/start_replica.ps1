# Replication sample (R1): fetch the releases, fundamentals and surprises of the new 1,000 companies and score them
# (dictionary, FinBERT, ChronoGPT raw and cleaned), in a process that survives this session. Uses its own database
# and score cache through SMALLCAP_DB and SMALLCAP_CACHE; the original data are not touched.
#   powershell -ExecutionPolicy Bypass -File smallcap\start_replica.ps1
# Progress: Get-Content data\cache\logs\rep_*.log -Tail 3   (the chain stops at the first step that fails)
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "data\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$chain = @"
`$env:SMALLCAP_DB = '$repo\data\smallcap_rep.sqlite'
`$env:SMALLCAP_CACHE = '$repo\data\cache\smallcap_rep'
`$env:PYTHONIOENCODING = 'utf-8'
Set-Location '$repo'
`$py = '$repo\.venv\Scripts\python.exe'
foreach (`$step in @(@('build','releases'), @('build','fundamentals'), @('build','surprise'), @('score','all'))) {
    `$name = 'rep_' + `$step[0] + '_' + `$step[1]
    & `$py -W ignore -m ('smallcap.' + `$step[0]) `$step[1] > ('$logs\' + `$name + '.log') 2> ('$logs\' + `$name + '.err')
    if (`$LASTEXITCODE -ne 0) { 'failed: ' + `$name | Out-File '$logs\rep_chain.log' -Append; exit 1 }
    'done: ' + `$name + ' ' + (Get-Date -Format s) | Out-File '$logs\rep_chain.log' -Append
}
'all done ' + (Get-Date -Format s) | Out-File '$logs\rep_chain.log' -Append
"@
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $chain) `
    -WindowStyle Hidden -PassThru
Write-Output "started pid $($p.Id)  (log: data\cache\logs\rep_chain.log)"
