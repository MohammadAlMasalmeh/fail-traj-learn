$ErrorActionPreference = "Continue"
Set-Location "C:\Users\Mohammad\Desktop\FRI\fail-traj-learn"
$log = "outputs\distill\probe\probe.log"
function L($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }
New-Item -ItemType Directory -Force outputs\distill\probe | Out-Null
$pids = @(Get-Content outputs\distill_unmasked\runner.pid | ForEach-Object { "$_".Trim() } | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ })
L "waiting for PIDs: $($pids -join ',')"
while ($true) {
  $alive = @($pids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
  if ($alive.Count -eq 0) { break }
  Start-Sleep -Seconds 20
}
L "runner exited; sleeping 30s"
Start-Sleep -Seconds 30
$py = ".venv-distill\Scripts\python.exe"; $ad = "outputs/distill_unmasked/adapter_final"
$env:PYTHONUNBUFFERED = "1"
foreach ($step in @(@("own","--adapter",$ad), @("teacher","--adapter",$ad), @("score"))) {
  $extra = if ($step.Count -gt 1) { $step[1..($step.Count-1)] } else { @() }
  L "running probe $($step -join ' ')"
  & $py scripts/distill/probe_failure.py $step[0] --tag sft_unmasked $extra *>> $log
  L "exit $LASTEXITCODE"
}
L "done"
