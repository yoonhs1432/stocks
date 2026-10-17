# 서버 감시 루프 — run.ps1 이 띄운다. 직접 실행할 일은 없다.
#
# 설정 탭의 "업데이트 받기" 를 누르면 서버가 새 코드를 받은 뒤 종료 코드 10 으로 스스로
# 끝난다. 그때 여기서 곧바로 다시 띄운다. **터널(cloudflared·tailscale)은 따로 떠 있어
# 건드리지 않으므로 주소가 바뀌지 않는다** — 폰에서 새로고침만 하면 새 코드가 뜬다.
#
# 서버가 **죽어도** 다시 띄운다. 예전에는 종료 코드 3 이 아니면 루프를 빠져나가서,
# 한 번 죽으면 사람이 PC 앞에 가서 다시 켜기 전까지 폰이 계속 끊겨 있었다.
# 연달아 죽으면(포트를 다른 프로세스가 잡고 있는 등) 간격을 10초 → 최대 60초로 늘린다.
# 멈추고 싶으면 run.ps1 창을 닫거나 작업 스케줄러에서 QuantDashboard 를 중지한다.

param([string]$Py = "py")

Set-Location $PSScriptRoot
$env:QUANT_SUPERVISED = "1"      # 서버는 이 값을 보고 "재시작해 줄 사람이 있다" 고 판단한다

# 언제 왜 다시 떴는지 남긴다 — 숨긴 창으로 돌면 화면으로는 알 길이 없다
$log = Join-Path $PSScriptRoot "data\serve-loop.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Note([string]$msg) {
    $line = "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $msg
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Host "  $msg"
}

$wait = 10
while ($true) {
    $started = Get-Date
    Note "서버 시작"
    & $Py server.py
    $code = $LASTEXITCODE
    # 10 = 새 코드를 받았다(server.py RESTART_CODE). 3 은 uvicorn 이 포트를 못 열었을 때라 여기 넣지 않는다
    if ($code -eq 10) {
        Note "새 코드를 받았습니다. 서버를 다시 띄웁니다"
        $wait = 10
        Start-Sleep -Seconds 1
        continue
    }
    # 한참 잘 돌다 죽은 거면 바로, 띄우자마자 죽기를 반복하면 점점 천천히
    if (((Get-Date) - $started).TotalMinutes -gt 5) { $wait = 10 }
    Note "서버가 종료 코드 $code 로 멈췄습니다 · ${wait}초 뒤 다시 띄웁니다"
    Start-Sleep -Seconds $wait
    $wait = [Math]::Min($wait * 2, 60)
}
