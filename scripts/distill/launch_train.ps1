# Launch detached training; PID(s) written to outputs/distill/train.pid
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$py = Join-Path $repo ".venv-distill\Scripts\python.exe"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$out = Join-Path $repo "outputs\distill"
$p = Start-Process -FilePath $py -ArgumentList @("scripts/distill/train.py","--epochs","2","--stride","3","--max_hours","5") `
  -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $out "train_stdout.log") -RedirectStandardError (Join-Path $out "train_stderr.log")
$p.Id | Out-File -Encoding ascii (Join-Path $out "train.pid")
Write-Output "launched PID $($p.Id)"
