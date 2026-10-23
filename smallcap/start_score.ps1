# 小型株の決算発表文の点数付け（株価の反応は使わない）を、このセッションや VS Code を閉じても止まらない別のプロセスで動かす。
#   powershell -ExecutionPolicy Bypass -File smallcap\start_score.ps1
# 進み具合:  Get-Content data\cache\logs\smallcap_score.log -Tail 3
# 途中で止まっても、もう一度実行すれば続きから。GPU を使うほかの作業は同時に動かさない。
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "data\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$p = Start-Process -FilePath (Join-Path $repo ".venv\Scripts\python.exe") `
    -ArgumentList @("-m", "smallcap.score", "all") -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $logs "smallcap_score.log") -RedirectStandardError (Join-Path $logs "smallcap_score.err") `
    -WindowStyle Hidden -PassThru
Write-Output "started pid $($p.Id)  (log: data\cache\logs\smallcap_score.log)"
