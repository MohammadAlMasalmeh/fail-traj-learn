# Ablation: waits for eval runner, trains with review_required_segment UNMASKED, generates, scores 3-way. Log: outputs/distill_unmasked/runner.log
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$py = Join-Path $repo ".venv-distill\Scripts\python.exe"
$env:PYTHONUNBUFFERED = "1"; $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$d = Join-Path $repo "outputs\distill_unmasked"
New-Item -ItemType Directory -Force $d | Out-Null
$log = Join-Path $d "runner.log"
function Log($m) { "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m | Tee-Object -FilePath $log -Append }

Log "runner start; waiting for eval runner (outputs/distill/eval_runner.pid) to exit"
$ep = $null
if (Test-Path "outputs\distill\eval_runner.pid") { $ep = [int](([regex]::Match((Get-Content "outputs\distill\eval_runner.pid" -Raw), '\d+')).Value) }
while ($ep -and (Get-Process -Id $ep -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
Log "eval runner exited; sleeping 30s"
Start-Sleep -Seconds 30

Log "train (unmasked review_required_segment)"
$p = Start-Process -FilePath $py -ArgumentList @("scripts/distill/train.py","--epochs","2","--stride","3","--max_hours","5","--out-dir","outputs/distill_unmasked","--unmask-reasons","review_required_segment") `
  -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $d "train_stdout.log") -RedirectStandardError (Join-Path $d "train_stderr.log")
$p.Id | Out-File -Encoding ascii (Join-Path $d "train.pid")
Log "training PID $($p.Id)"
$null = $p.WaitForExit(5.2 * 3600 * 1000)
if (-not $p.HasExited) { Log "WARNING: hard cap hit, killing training"; Stop-Process -Id $p.Id -Force }
Start-Sleep -Seconds 20
Log "training exited"

$adapter = Join-Path $d "adapter_final"
if (-not (Test-Path $adapter)) {
    $ck = Get-ChildItem "$d\ckpt" -Directory -ErrorAction SilentlyContinue | Sort-Object { [int](($_.Name) -replace '\D', '0') } | Select-Object -Last 1
    if ($ck) { $adapter = $ck.FullName; Log "WARNING: adapter_final missing; using $adapter" }
    else { Log "ERROR: no adapter; aborting"; exit 1 }
}
Log "generate sft_unmasked with $adapter"
& $py scripts/distill/eval.py generate --adapter $adapter --tag sft_unmasked --max-new-tokens 5120 2>&1 | Tee-Object -FilePath $log -Append
Log "score"
& $py scripts/distill/eval.py score --tags base sft sft_unmasked --skip-missing --scores-dir (Join-Path $d "scores") 2>&1 | Tee-Object -FilePath $log -Append
Log "done"
