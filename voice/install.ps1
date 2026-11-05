# Registers the voice listener to start hidden at logon (current user, no admin needed).
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
$action = New-ScheduledTaskAction -Execute $py -Argument "-m voice.listener" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName "ClaudeVoice" -Action $action -Trigger $trigger -Settings $set -Force | Out-Null
Write-Host "Registered ClaudeVoice. Start now:  Start-ScheduledTask ClaudeVoice   Stop:  Stop-ScheduledTask ClaudeVoice"
