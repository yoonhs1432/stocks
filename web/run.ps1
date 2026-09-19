# 서버 실행 (+ 외부 접속용 Cloudflare 터널).
#
#   .\run.ps1              집·PC 에서만 (같은 와이파이)
#   .\run.ps1 -Tunnel      외부에서도 접속 (https 주소가 만들어진다)
#
# 창을 닫거나 Ctrl+C 를 누르면 둘 다 멈춘다.

param([switch]$Tunnel)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 파이썬 찾기 — py 런처 우선
$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" }
      elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
      else { throw "파이썬을 찾을 수 없습니다. winget install --id Python.Python.3.12 후 창을 새로 여세요." }

# 같은 네트워크에서 접속할 주소
$lan = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" } |
        Select-Object -First 1).IPAddress

$server = Start-Process $py -ArgumentList "server.py" -PassThru -NoNewWindow

try {
    Start-Sleep -Seconds 3
    Write-Host ""
    Write-Host "  이 PC:        http://localhost:8000"
    if ($lan) { Write-Host "  같은 와이파이: http://${lan}:8000" }

    if ($Tunnel) {
        if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
            Write-Host ""
            Write-Host "  cloudflared 가 없습니다. 설치 후 다시 실행하세요:" -ForegroundColor Yellow
            Write-Host "    winget install --id Cloudflare.cloudflared"
        } else {
            Write-Host ""
            Write-Host "  터널을 엽니다. 아래 trycloudflare.com 주소를 폰에서 열면 됩니다." -ForegroundColor Cyan
            Write-Host "  (주소는 껐다 켤 때마다 바뀝니다. 고정하려면 README 의 '고정 주소' 참고)"
            Write-Host ""
            cloudflared tunnel --url http://localhost:8000
        }
    }
    Wait-Process -Id $server.Id
}
finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
