# PC 를 폰에서 직접 조종하기 (Claude Code Remote Control).
#
#   .\rc.ps1                  이름 기본값 quant-pc
#   .\rc.ps1 -Name "집PC"     세션 이름 지정
#   .\rc.ps1 -Auto            안내 없이 띄우고, 끊기면 다시 붙인다 (작업 스케줄러용)
#
# 재부팅해도 저절로 붙게 하려면:  .\install-task.ps1 -Funnel -RemoteControl
#
# 이 창을 열어 둔 동안, 폰(claude.ai/code 또는 Claude 앱)에서 **이 PC 의 Claude Code** 를
# 그대로 쓸 수 있다. 실행은 전부 이 PC 에서 일어난다 — 파일도, 명령도, 서버도 이 PC 것이다.
# 클라우드 세션과 달리 저장소를 따로 받아 오지 않으므로 web/data 같은 이 PC 안의 것도 본다.
#
# 언제 쓰나 — 서버가 아예 안 뜨거나, 파이썬·터널이 깨졌거나, 로그를 봐야 할 때.
# 평소 "고친 코드 받기" 는 설정 탭의 업데이트 버튼이면 충분하다.

param([string]$Name = "quant-pc", [switch]$Auto)

$ErrorActionPreference = "Stop"

# 저장소 루트에서 띄운다 (web 뿐 아니라 전체를 볼 수 있게)
Set-Location (Split-Path $PSScriptRoot -Parent)

# 방금 설치했다면 이 창의 PATH 에 아직 반영되지 않았다
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

$claude = (Get-Command claude -ErrorAction SilentlyContinue).Source
if (-not $claude) {
    $claude = @("$env:USERPROFILE\.local\bin\claude.exe",
                "$env:LOCALAPPDATA\Microsoft\WinGet\Links\claude.exe") |
              Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $claude) {
    if ($Auto) { throw "Claude Code 를 찾을 수 없습니다 (rc.ps1 -Auto)" }
    Write-Host ""
    Write-Host "  Claude Code 가 없습니다. 아래를 실행하고 PowerShell 창을 새로 여세요." -ForegroundColor Yellow
    Write-Host "    irm https://claude.ai/install.ps1 | iex"
    Write-Host "  설치 후 한 번은 'claude' 를 실행해 로그인해 두세요 (Pro/Max 계정 필요)."
    return
}

# 자동 모드 — 로그인·부팅 때 작업 스케줄러가 띄운다. 끊기거나 죽으면 다시 붙인다.
# 같은 이름(quant-pc)으로 다시 뜨므로 폰에서는 세션 목록에서 같은 이름을 고르면 된다.
if ($Auto) {
    $log = Join-Path $PSScriptRoot "data\rc.log"
    New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
    function Note([string]$msg) {
        $line = "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $msg
        Add-Content -Path $log -Value $line -Encoding UTF8
        Write-Host "  $msg"
    }
    $wait = 10
    while ($true) {
        $started = Get-Date
        Note "Remote Control 시작 ($Name)"
        & $claude remote-control --name $Name
        $code = $LASTEXITCODE
        # 한참 잘 붙어 있다 끊긴 거면 바로, 띄우자마자 끊기기를 반복하면(네트워크 없음 등) 점점 천천히
        if (((Get-Date) - $started).TotalMinutes -gt 5) { $wait = 10 }
        Note "Remote Control 이 종료 코드 $code 로 끝났습니다 · ${wait}초 뒤 다시 붙입니다"
        Start-Sleep -Seconds $wait
        $wait = [Math]::Min($wait * 2, 300)
    }
}

Write-Host ""
Write-Host "  폰에서 이 PC 를 조종할 수 있게 띄웁니다." -ForegroundColor Cyan
Write-Host "  · 처음이면 'Enable Remote Control? (y/n)' 에 y 를 누르세요."
Write-Host "  · 주소가 뜨면 폰에서 그 주소를 열거나, 스페이스바를 눌러 QR 을 찍으세요."
Write-Host "  · 나중에는 claude.ai/code 세션 목록에서 '$Name' 으로 찾으면 됩니다."
Write-Host "  · 끝낼 때는 이 창에서 Ctrl+C."
Write-Host ""

& $claude remote-control --name $Name
