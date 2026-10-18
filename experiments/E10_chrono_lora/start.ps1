# E10 を、このセッションや VS Code を閉じても止まらない別のプロセスで 1 回だけ動かす。
#   powershell -ExecutionPolicy Bypass -File experiments\E10_chrono_lora\start.ps1
# 進み具合:  Get-Content data\cache\logs\e10.log -Tail 3
# 学習 約 4.9 時間 + 予測 約 43 分。GPU を使うほかの作業 (ChronoGPT、FinBERT) は同時に動かさない。
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$logs = Join-Path $repo "data\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
if (Test-Path (Join-Path $repo "experiments\E10_chrono_lora\results\summary.json")) {
    Write-Output "results/summary.json があります。E10 は実行済みです (事前登録どおり 1 回だけ)。"
    exit 1
}
$p = Start-Process -FilePath (Join-Path $repo ".venv\Scripts\python.exe") `
    -ArgumentList @("experiments/E10_chrono_lora/run.py") -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $logs "e10.log") -RedirectStandardError (Join-Path $logs "e10.err") `
    -WindowStyle Hidden -PassThru
Write-Output "started pid $($p.Id)  (log: data\cache\logs\e10.log)"
