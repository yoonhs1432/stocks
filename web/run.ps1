# 서버 실행 (+ 외부 접속).
#
#   .\run.ps1              집·PC 에서만 (같은 와이파이)
#   .\run.ps1 -Tunnel      Cloudflare 임시 주소로 외부 접속 (주소가 껐다 켤 때마다 바뀐다)
#   .\run.ps1 -Funnel      Tailscale 고정 주소로 외부 접속 (주소가 안 바뀐다. README 참고)
#
# 창을 닫거나 Ctrl+C 를 누르면 전부 멈춘다.

param([switch]$Tunnel, [switch]$Funnel)

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

# 방금 설치한 프로그램은 이 창의 PATH 에 아직 없다. 다시 읽어 둔다.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

function Find-Exe([string]$name, [string[]]$places) {
    $p = (Get-Command $name -ErrorAction SilentlyContinue).Source
    if ($p) { return $p }
    return $places | Where-Object { Test-Path $_ } | Select-Object -First 1
}

# 서버는 감시 루프가 띄운다 — 설정 탭에서 "업데이트 받기" 를 눌러 서버가 다시 떠도
# 터널은 그대로라 주소가 바뀌지 않는다.
$server = Start-Process powershell -PassThru -NoNewWindow -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "serve-loop.ps1"), "-Py", $py)

try {
    Start-Sleep -Seconds 3
    Write-Host ""
    Write-Host "  이 PC:        http://localhost:8000"
    if ($lan) { Write-Host "  같은 와이파이: http://${lan}:8000" }

    if ($Funnel) {
        $ts = Find-Exe "tailscale" @(
            "$env:ProgramFiles\Tailscale\tailscale.exe",
            "${env:ProgramFiles(x86)}\Tailscale\tailscale.exe")
        if (-not $ts) {
            Write-Host ""
            Write-Host "  tailscale 을 찾지 못했습니다." -ForegroundColor Yellow
            Write-Host "    winget install --id tailscale.tailscale"
            Write-Host "  설치 후 PowerShell 창을 새로 열고, README 의 '고정 주소' 를 한 번 따라 하세요."
        } else {
            Write-Host ""
            Write-Host "  고정 주소로 엽니다. 아래 ts.net 주소는 앞으로 바뀌지 않습니다." -ForegroundColor Cyan
            Write-Host ""
            & $ts funnel 8000
        }
    }
    elseif ($Tunnel) {
        $cf = Find-Exe "cloudflared" @(
            "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe",
            "$env:ProgramFiles\cloudflared\cloudflared.exe",
            "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe")
        if (-not $cf) {
            Write-Host ""
            Write-Host "  cloudflared 를 찾지 못했습니다." -ForegroundColor Yellow
            Write-Host "    winget install --id Cloudflare.cloudflared"
            Write-Host "  설치했는데도 이 메시지가 나오면 PowerShell 창을 새로 열어 주세요."
        } else {
            Write-Host ""
            Write-Host "  터널을 엽니다. 아래 trycloudflare.com 주소를 폰에서 열면 됩니다." -ForegroundColor Cyan
            Write-Host "  (주소는 껐다 켤 때마다 바뀝니다. 고정하려면 -Funnel 또는 README 참고)"
            Write-Host ""
            & $cf tunnel --url http://localhost:8000
        }
    }
    Wait-Process -Id $server.Id
}
finally {
    # 감시 루프와 그 아래 파이썬까지 같이 정리한다 (/T = 자식 프로세스 포함)
    if ($server -and -not $server.HasExited) {
        taskkill /PID $server.Id /T /F 2>$null | Out-Null
    }
}
