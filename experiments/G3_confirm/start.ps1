# G3: runs S1's run.py, S2's run.py and G3's run.py once on the G3 sample (data\smallcap_g3.sqlite), in order, in a
# process that survives this session. Start it only after preregistration.md and the code are committed.
#   powershell -ExecutionPolicy Bypass -File experiments\G3_confirm\start.ps1
# Progress: Get-Content data\cache\logs\g3_analysis.log   (the chain stops at the first step that fails)
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$logs = Join-Path $repo "data\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$chain = @"
`$env:SMALLCAP_DB = '$repo\data\smallcap_g3.sqlite'
`$env:SMALLCAP_CACHE = '$repo\data\cache\smallcap_g3'
`$env:SMALLCAP_RESULTS = 'results_g3'
`$env:PYTHONIOENCODING = 'utf-8'
Set-Location '$repo'
`$py = '$repo\.venv\Scripts\python.exe'
foreach (`$step in @('S1_smallcap_drift/run.py', 'S2_smallcap_risk/run.py', 'G3_confirm/run.py')) {
    `$name = 'g3_' + (`$step -replace '[/.]', '_')
    & `$py -W ignore ('experiments/' + `$step) > ('$logs\' + `$name + '.log') 2> ('$logs\' + `$name + '.err')
    if (`$LASTEXITCODE -ne 0) { 'failed: ' + `$step | Out-File '$logs\g3_analysis.log' -Append; exit 1 }
    'done: ' + `$step + ' ' + (Get-Date -Format s) | Out-File '$logs\g3_analysis.log' -Append
}
'all done ' + (Get-Date -Format s) | Out-File '$logs\g3_analysis.log' -Append
"@
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $chain) `
    -WindowStyle Hidden -PassThru
Write-Output "started pid $($p.Id)  (log: data\cache\logs\g3_analysis.log)"
