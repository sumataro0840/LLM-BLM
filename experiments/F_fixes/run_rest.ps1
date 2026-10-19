# Runs the remaining steps of the fix experiments in order, each only if the previous one succeeded:
# fix 5 scoring -> develop.py -> bert.py -> gpt.py -> final.py.  Waits for the process -WaitFor (the first scoring run).
param([int]$WaitFor = 0)
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$py = Join-Path $repo ".venv\Scripts\python.exe"
$logs = Join-Path $repo "data\cache\logs"
Set-Location $repo
if ($WaitFor -gt 0) { Wait-Process -Id $WaitFor -ErrorAction SilentlyContinue }
$steps = @(
    @("score5", @("experiments/F_fixes/score.py", "compress", "finbert5", "chrono5")),
    @("develop", @("experiments/F_fixes/develop.py")),
    @("bert", @("experiments/F_fixes/bert.py")),
    @("gpt", @("experiments/F_fixes/gpt.py")),
    @("final", @("experiments/F_fixes/final.py"))
)
foreach ($s in $steps) {
    $name = $s[0]
    "$(Get-Date -Format s) start $name" | Out-File -Append (Join-Path $logs "fixes_rest.log")
    $p = Start-Process -FilePath $py -ArgumentList $s[1] -WorkingDirectory $repo -Wait -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "fixes_$name.log") -RedirectStandardError (Join-Path $logs "fixes_$name.err")
    "$(Get-Date -Format s) end $name exit $($p.ExitCode)" | Out-File -Append (Join-Path $logs "fixes_rest.log")
    if ($p.ExitCode -ne 0) { exit $p.ExitCode }
}
