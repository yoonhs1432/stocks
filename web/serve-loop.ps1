# 서버 감시 루프 — run.ps1 이 띄운다. 직접 실행할 일은 없다.
#
# 설정 탭의 "업데이트 받기" 를 누르면 서버가 새 코드를 받은 뒤 종료 코드 3 으로 스스로
# 끝난다. 그때 여기서 곧바로 다시 띄운다. **터널(cloudflared·tailscale)은 따로 떠 있어
# 건드리지 않으므로 주소가 바뀌지 않는다** — 폰에서 새로고침만 하면 새 코드가 뜬다.

param([string]$Py = "py")

Set-Location $PSScriptRoot
$env:QUANT_SUPERVISED = "1"      # 서버는 이 값을 보고 "재시작해 줄 사람이 있다" 고 판단한다

while ($true) {
    & $Py server.py
    if ($LASTEXITCODE -ne 3) { break }
    Write-Host ""
    Write-Host "  새 코드를 받았습니다. 서버를 다시 띄웁니다…" -ForegroundColor Cyan
    Start-Sleep -Seconds 1
}
