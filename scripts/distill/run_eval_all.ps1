# Waits for training to finish, then: generate base, generate sft, score. Log: outputs/distill/eval_stdout.log
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$py = Join-Path $repo ".venv-distill\Scripts\python.exe"
$d = Join-Path $repo "outputs\distill"
$log = Join-Path $d "eval_stdout.log"
function Log($m) { "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m | Tee-Object -FilePath $log -Append }

Log "runner start; waiting for training to exit"
$pids = @()
if (Test-Path "$d\train.pid") { $pids = [regex]::Matches((Get-Content "$d\train.pid" -Raw), '(?:launcher|worker)=(\d+)|^\s*(\d+)') | ForEach-Object { if ($_.Groups[1].Success) { [int]$_.Groups[1].Value } else { [int]$_.Groups[2].Value } } }
while ($true) {
    $alive = $pids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }
    if (-not $alive) { break }
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 20   # let GPU memory be released
Log "training processes exited"

$adapter = Join-Path $d "adapter_final"
if (-not (Test-Path $adapter)) {
    $ck = Get-ChildItem "$d\ckpt" -Directory -ErrorAction SilentlyContinue | Sort-Object { [int](($_.Name) -replace '\D', '0') } | Select-Object -Last 1
    if ($ck) { $adapter = $ck.FullName; Log "WARNING: adapter_final missing; using latest checkpoint $adapter" }
    else { Log "ERROR: no adapter found; only base will be evaluated"; $adapter = $null }
}

Log "generate base"
& $py scripts/distill/eval.py generate --adapter none --tag base --max-new-tokens 5120 2>&1 | Tee-Object -FilePath $log -Append
if ($adapter) {
    Log "generate sft with $adapter"
    & $py scripts/distill/eval.py generate --adapter $adapter --tag sft --max-new-tokens 5120 2>&1 | Tee-Object -FilePath $log -Append
    Log "score"
    & $py scripts/distill/eval.py score --tags base sft --skip-missing 2>&1 | Tee-Object -FilePath $log -Append
} else {
    & $py scripts/distill/eval.py score --tags base --skip-missing 2>&1 | Tee-Object -FilePath $log -Append
}
Log "done"
