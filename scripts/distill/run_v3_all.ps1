# v3: events input + failure-token loss weight (5) + unmasked review_required_segment. Log: outputs/distill_v3/runner.log
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$py = Join-Path $repo ".venv-distill\Scripts\python.exe"
$env:PYTHONUNBUFFERED = "1"; $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"; $env:PYTHONIOENCODING = "utf-8"
$d = Join-Path $repo "outputs\distill_v3"
New-Item -ItemType Directory -Force $d | Out-Null
$log = Join-Path $d "runner.log"
function Log($m) { "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m | Tee-Object -FilePath $log -Append }
function PidOf($f) { if (Test-Path $f) { return [int](([regex]::Match((Get-Content $f -Raw), '\d+')).Value) } else { return 0 } }

$pids = @((PidOf "outputs\distill_unmasked\runner.pid"), (PidOf "outputs\distill\probe\probe.pid")) | Where-Object { $_ }
Log "runner start; waiting for PIDs $($pids -join ',') to exit"
while ($true) {
    $alive = $pids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }
    if (-not $alive) { break }
    Start-Sleep -Seconds 60
}
Log "all prior jobs exited; sleeping 30s"
Start-Sleep -Seconds 30

Log "train v3 (events, fail-token-weight 5, unmask review_required_segment)"
$p = Start-Process -FilePath $py -ArgumentList @("scripts/distill/train.py","--epochs","2","--stride","3","--max_hours","5","--max_len","10000","--out-dir","outputs/distill_v3","--unmask-reasons","review_required_segment","--fail-token-weight","5","--events") `
  -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $d "train_stdout.log") -RedirectStandardError (Join-Path $d "train_stderr.log")
$p.Id | Out-File -Encoding ascii (Join-Path $d "train.pid")
Log "training PID $($p.Id)"
$null = $p.WaitForExit([int](5.2 * 3600 * 1000))
if (-not $p.HasExited) { Log "WARNING: hard cap hit, killing training"; Stop-Process -Id $p.Id -Force }
Start-Sleep -Seconds 20
Log "training exited"

$adapter = Join-Path $d "adapter_final"
if (-not (Test-Path $adapter)) {
    $ck = Get-ChildItem "$d\ckpt" -Directory -ErrorAction SilentlyContinue | Sort-Object { [int](($_.Name) -replace '\D', '0') } | Select-Object -Last 1
    if ($ck) { $adapter = $ck.FullName; Log "WARNING: adapter_final missing; using $adapter" }
    else { Log "ERROR: no adapter; aborting"; exit 1 }
}
Log "generate sft_v3 with $adapter (events input)"
& $py scripts/distill/eval.py generate --adapter $adapter --tag sft_v3 --events --max-new-tokens 5120 2>&1 | Tee-Object -FilePath $log -Append
Log "score_failures"
& $py scripts/distill/score_failures.py --tags base sft sft_unmasked sft_v3 2>&1 | Tee-Object -FilePath $log -Append
Log "score"
& $py scripts/distill/eval.py score --tags base sft sft_unmasked sft_v3 --skip-missing --scores-dir (Join-Path $d "scores") 2>&1 | Tee-Object -FilePath $log -Append
Log "done"
