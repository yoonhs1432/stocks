# 로그인할 때 서버를 자동으로 띄운다 (Windows 작업 스케줄러).
#
#   .\install-task.ps1              서버만
#   .\install-task.ps1 -Tunnel      Cloudflare 임시 주소까지 (실행할 때마다 주소가 바뀐다)
#   .\install-task.ps1 -Funnel      Tailscale 고정 주소까지 (주소가 안 바뀐다 — 권장)
#   .\install-task.ps1 -Remove      등록 해제
#   .\install-task.ps1 -Funnel -AtStartup   **로그인하지 않아도** 부팅하자마자 (관리자 창에서)
#   .\install-task.ps1 -Funnel -RemoteControl   서버 + 폰에서 PC 조종(rc.ps1)까지 — 권장
#
# 관리자 권한이 필요 없다. 로그인할 때 숨겨진 창으로 실행된다.
#
# -RemoteControl — 재부팅 뒤 PowerShell 을 열어 rc.ps1 을 칠 필요가 없다. 작업 `QuantRemoteControl` 이
#   `rc.ps1 -Auto` 를 최소화된 창으로 띄우고, 끊기면 다시 붙인다(기록은 data\rc.log).
#   등록하기 전에 **한 번은 손으로 .\rc.ps1** 을 띄워 'Enable Remote Control?' 에 y 를 눌러 둔다.
#
# -AtStartup — 정전·수동 재부팅 뒤 PC 가 **로그인 화면에 멈춰 있어도** 서버가 뜬다.
#   암호를 저장하지 않는 S4U 방식이라 관리자 권한으로 한 번 등록해야 한다.
#   (Tailscale 도 로그인 없이 붙어 있으려면 `tailscale set --unattended=true` — 2026-10-17 켜 둠)
#
# ⚠️ 트리거에 **이 사용자**를 꼭 적는다. `-AtLogOn` 만 쓰면 "아무 사용자나 로그인할 때" 가
#    되어 관리자 권한을 요구하고, 일반 창에서는 '액세스가 거부되었습니다' 로 실패한다.

param([switch]$Tunnel, [switch]$Funnel, [switch]$Remove, [switch]$AtStartup, [switch]$RemoteControl)

$ErrorActionPreference = "Stop"
$name = "QuantDashboard"
$rcName = "QuantRemoteControl"
$user = "$env:USERDOMAIN\$env:USERNAME"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $rcName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "  등록을 해제했습니다." -ForegroundColor Yellow
    return
}

$here = $PSScriptRoot
$args = if ($Funnel) { "-Funnel" } elseif ($Tunnel) { "-Tunnel" } else { "" }

# run.ps1 을 숨긴 창으로 띄운다. -ExecutionPolicy Bypass 가 있어야 정책과 무관하게 돈다
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$here\run.ps1`" $args" `
    -WorkingDirectory $here

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
# 네트워크가 늦게 붙는 경우를 대비해 30초 지연
$trigger.Delay = "PT30S"
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$triggers = @($trigger)

if ($AtStartup) {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
             ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) {
        Write-Host ""
        Write-Host "  -AtStartup 은 관리자 권한이 필요합니다." -ForegroundColor Yellow
        Write-Host "  시작 메뉴에서 PowerShell 을 '관리자 권한으로 실행' 한 뒤 다시 실행하세요:"
        Write-Host "    cd `"$here`"; .\install-task.ps1 -Funnel -AtStartup"
        return
    }
    $boot = New-ScheduledTaskTrigger -AtStartup
    $boot.Delay = "PT45S"          # 네트워크·Tailscale 이 붙을 시간
    $triggers = @($boot, $trigger)
    # S4U — 로그인하지 않아도 이 사용자로 돈다. 암호를 저장하지 않는다.
    # 로그인 트리거도 남겨 두지만, 이미 돌고 있으면 MultipleInstances IgnoreNew 로 하나만 뜬다.
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
}
# 배터리여도 멈추지 않고, 시간 제한 없이, run.ps1 자체가 죽으면 1분 뒤 다시(최대 999번)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit 0 `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $name -Action $action -Trigger $triggers -Principal $principal `
    -Settings $settings -Description "퀀트 대시보드 서버" -Force | Out-Null

if ($RemoteControl) {
    # 트리거·권한·재시작 설정은 서버와 같다. 창만 최소화로 — 작업 표시줄에서 상태를 볼 수 있게
    $rcAction = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$here\rc.ps1`" -Auto" `
        -WorkingDirectory (Split-Path $here -Parent)
    Register-ScheduledTask -TaskName $rcName -Action $rcAction -Trigger $triggers -Principal $principal `
        -Settings $settings -Description "폰에서 이 PC 조종 (Claude Code Remote Control)" -Force | Out-Null
}

Write-Host ""
if ($AtStartup) {
    Write-Host "  등록했습니다. 다음 부팅부터 로그인하지 않아도 실행됩니다." -ForegroundColor Green
} else {
    Write-Host "  등록했습니다. 다음 로그인부터 자동으로 실행됩니다." -ForegroundColor Green
}
if ($Funnel) {
    Write-Host "  고정 주소(ts.net)로 뜹니다. 미리 한 번 'tailscale funnel 8000' 을 허용해 두세요." -ForegroundColor Cyan
} elseif ($Tunnel) {
    Write-Host "  ⚠️ 터널 주소는 실행할 때마다 바뀝니다. 고정하려면 README 의 '고정 주소' 참고." -ForegroundColor Yellow
}
if ($RemoteControl) {
    Write-Host "  Remote Control(quant-pc)도 같이 뜹니다. 폰의 Claude 앱 세션 목록에서 quant-pc 를 고르세요." -ForegroundColor Cyan
    Write-Host "  처음이라면 먼저 .\rc.ps1 을 한 번 손으로 띄워 'Enable Remote Control?' 에 y 를 눌러 두세요." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  지금 바로 시작:  Start-ScheduledTask -TaskName $name"
if ($RemoteControl) { Write-Host "                  Start-ScheduledTask -TaskName $rcName" }
Write-Host "  중지:            Stop-ScheduledTask -TaskName $name"
Write-Host "  해제:            .\install-task.ps1 -Remove"
