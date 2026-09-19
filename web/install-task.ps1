# 부팅할 때 서버를 자동으로 띄운다 (Windows 작업 스케줄러).
#
#   .\install-task.ps1              서버만
#   .\install-task.ps1 -Tunnel      터널까지 (외부 접속)
#   .\install-task.ps1 -Remove      등록 해제
#
# 관리자 권한이 필요 없다. 로그인할 때 숨겨진 창으로 실행된다.

param([switch]$Tunnel, [switch]$Remove)

$ErrorActionPreference = "Stop"
$name = "QuantDashboard"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "  등록을 해제했습니다." -ForegroundColor Yellow
    return
}

$here = $PSScriptRoot
$args = if ($Tunnel) { "-Tunnel" } else { "" }

# run.ps1 을 숨긴 창으로 띄운다. -ExecutionPolicy Bypass 가 있어야 정책과 무관하게 돈다
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$here\run.ps1`" $args" `
    -WorkingDirectory $here

$trigger = New-ScheduledTaskTrigger -AtLogOn
# 노트북이 배터리로 돌 때도 멈추지 않게, 네트워크가 늦게 붙는 경우를 대비해 30초 지연
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit 0
$trigger.Delay = "PT30S"

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Settings $settings -Description "퀀트 대시보드 서버" -Force | Out-Null

Write-Host ""
Write-Host "  등록했습니다. 다음 로그인부터 자동으로 실행됩니다." -ForegroundColor Green
if ($Tunnel) {
    Write-Host "  ⚠️ 터널 주소는 실행할 때마다 바뀝니다. 고정하려면 README 의 '고정 주소' 참고." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  지금 바로 시작:  Start-ScheduledTask -TaskName $name"
Write-Host "  중지:            Stop-ScheduledTask -TaskName $name"
Write-Host "  해제:            .\install-task.ps1 -Remove"
