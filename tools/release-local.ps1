# release-local.ps1 - android.yml (self-hosted runner) 를 이 PC 에서 그대로 돌린다.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\release-local.ps1               # 빌드 -> android-latest 게시
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\release-local.ps1 -Notes "..."  # 릴리스 노트 지정
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\release-local.ps1 -NoPublish    # 빌드 검증만 (publish=false)
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\release-local.ps1 -Test         # draft 'local-release-test-stocks' 에 올림
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\release-local.ps1 -VerifyTag -SourceDir <체크아웃> -Notes "..."
#       # 검증: 실제와 같은 발행 코드(create / edit + upload --clobber)로 'local-release-verify-stocks-android-latest'
#       #       에 올린다 (prerelease, --latest=false). android-latest 는 건드리지 않는다. 끝나면 지운다:
#       #       gh release delete local-release-verify-stocks-android-latest -R yoonhs1432/stocks --yes --cleanup-tag
#   -SourceDir <폴더>  빌드할 소스 체크아웃(기본: 이 스크립트가 든 저장소). 예: git worktree add <폴더> <커밋>
#
# workflow 와 같은 것:
#   - android/ assembleDebug, android-toss/ assembleDebug versionJson (RELEASE_NOTES = 노트 또는 마지막 커밋 제목)
#   - 자산 이름 app-debug.apk, quant-toss-debug.apk, version.json
#   - 릴리스 android-latest, 제목 "Android 최신 빌드 (debug)", prerelease, 본문은 release-notes.md 와 같은 형식
#   - 서명: 저장소에 커밋된 android*/app/quant.keystore (gradle 의 signingConfigs "shared"). GitHub secret 불필요.
#   - 게시는 main 에서만 (workflow 의 github.ref == 'refs/heads/main' 조건). -Test 는 어느 브랜치든 된다.
# 기존 android-latest 가 있으면 softprops/action-gh-release 처럼 태그는 옮기지 않고 제목/본문/prerelease 를 고치고
# 같은 이름의 자산을 덮어쓴다 (--clobber).
#
# 도구: C:\Users\yoonh\android-build (JDK 17, gradle-8.9, sdk), Git for Windows, gh (로그인 상태).
# Windows PowerShell 5.1 호환.

[CmdletBinding()]
param(
    [switch]$Test,
    [switch]$NoPublish,
    [string]$Notes = "",
    [string]$SourceDir = "",
    [switch]$VerifyTag
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$Repo     = 'yoonhs1432/stocks'
$RealTag  = 'android-latest'
$TestTag  = 'local-release-test-stocks'
$Title    = 'Android 최신 빌드 (debug)'
$VerifyName = 'local-release-verify-stocks-android-latest'
$Root     = if ($SourceDir) { (Resolve-Path $SourceDir).Path } else { Split-Path -Parent $PSScriptRoot }
# 검증 모드: 발행 코드는 실제와 그대로 같고, 태그 이름과 --latest=false, 새로 만들 때의 target 만 다르다.
# (폰의 AppUpdate.kt 는 releases/download/android-latest/ 고정 URL 만 보므로 이 태그를 집어 가지 않는다.)
$PubTag   = $RealTag
$PubTarget = 'main'
$relFlags = @()
if ($VerifyTag) {
    if ($Test -or $NoPublish) { throw "-VerifyTag 는 -Test / -NoPublish 와 같이 쓸 수 없다." }
    $PubTag   = $VerifyName
    $relFlags = @('--latest=false')
}
function Assert-VerifyName($tag) {
    if (-not $VerifyTag) { return }
    if ($tag -ne $VerifyName -or $tag -eq $RealTag) { throw "GUARD: 검증 모드인데 실제 이름 '$tag' 를 쓰려 했다 - 중단" }
    if ($relFlags -notcontains '--latest=false') { throw "GUARD: 검증 모드 플래그 누락 - 중단" }
}
$Build    = 'C:\Users\yoonh\android-build'
$Jdk      = Join-Path $Build 'jdk-17.0.20.1+1'
$GradleBat = Join-Path $Build 'gradle-8.9\bin\gradle.bat'

function Step($m) { Write-Host ""; Write-Host ("=== {0} ({1})" -f $m, (Get-Date -Format 'HH:mm:ss')) }
function Check-Exit($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }
function Test-Release($tag) {
    # PS 5.1: EAP=Stop 에서 네이티브 stderr 를 돌리면 예외가 된다 - 잠시 Continue 로
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & gh release view $tag -R $Repo --json tagName 2>&1 | Out-Null; return ($LASTEXITCODE -eq 0) } finally { $ErrorActionPreference = $old }
}
function Write-Utf8($path, $text) { [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false))) }

Set-Location $Root

# --- PATH (workflow 의 PATH 단계와 같다) ---
$env:PATH = "C:\Program Files\Git\bin;C:\Program Files\Git\usr\bin;$Jdk\bin;$Build\gradle-8.9\bin;$env:PATH"
$env:JAVA_HOME = $Jdk
if (-not $env:ANDROID_HOME) { $env:ANDROID_HOME = Join-Path $Build 'sdk' }
if (-not $env:ANDROID_SDK_ROOT) { $env:ANDROID_SDK_ROOT = $env:ANDROID_HOME }

$publish = -not $NoPublish
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$sha = (git rev-parse HEAD).Trim()
if ($publish -and -not $Test) {
    # workflow: github.ref == 'refs/heads/main'. 로컬에선 main 브랜치이거나 HEAD 가 origin/main 과 같아야 한다.
    git fetch -q origin main
    Check-Exit 'git fetch'
    $originMain = (git rev-parse origin/main).Trim()
    if ($VerifyTag) {
        # 검증은 과거 커밋(기준 릴리스를 만든 커밋)을 다시 빌드한다 - 그 커밋이 main 에 있었는지만 본다
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & git merge-base --is-ancestor $sha $originMain; $anc = $LASTEXITCODE
        $ErrorActionPreference = $old
        if ($anc -ne 0) { throw "검증: $sha 가 origin/main 에 없다." }
        $PubTarget = $sha
    } elseif ($branch -ne 'main' -and $sha -ne $originMain) {
        throw "실제 게시는 main 에서만 한다 (지금 $branch @ $sha, origin/main @ $originMain). -Test 나 -NoPublish 를 쓰세요."
    }
    if (git status --porcelain --untracked-files=no) { throw "추적 파일에 커밋 안 된 변경이 있다 - 먼저 커밋하세요." }
}

Step 'Toolchain'
& java -version
Check-Exit 'java'
& $GradleBat --version | Select-String '^Gradle'
Write-Host "ANDROID_HOME=$env:ANDROID_HOME"

Step 'Build debug APK (android)'
Push-Location (Join-Path $Root 'android')
try { & $GradleBat assembleDebug --stacktrace; Check-Exit 'gradle android assembleDebug' } finally { Pop-Location }

Step 'Build debug APK (android-toss) + version.json'
$relNotes = $Notes
if (-not $relNotes) { $relNotes = (git log -1 --format=%s) -join "`n" }
$env:RELEASE_NOTES = $relNotes
Push-Location (Join-Path $Root 'android-toss')
try { & $GradleBat assembleDebug versionJson --stacktrace; Check-Exit 'gradle android-toss assembleDebug versionJson' } finally { Pop-Location }

Step 'Rename APKs'
$out = Join-Path $Root 'android-toss\build\local-release'
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item -Force (Join-Path $Root 'android\app\build\outputs\apk\debug\app-debug.apk') (Join-Path $out 'app-debug.apk')
Copy-Item -Force (Join-Path $Root 'android-toss\app\build\outputs\apk\debug\app-debug.apk') (Join-Path $out 'quant-toss-debug.apk')
Copy-Item -Force (Join-Path $Root 'android-toss\app\build\outputs\version.json') (Join-Path $out 'version.json')
$files = @('app-debug.apk', 'quant-toss-debug.apk', 'version.json') | ForEach-Object { Join-Path $out $_ }
Get-Item $files | Format-Table Name, Length -AutoSize | Out-Host
Get-Content -Raw -Encoding UTF8 (Join-Path $out 'version.json') | Out-Host

if (-not $publish) { Write-Host "빌드 검증만 (-NoPublish). 릴리스는 건드리지 않았다."; exit 0 }

Step 'Release notes'
$stamp = (git log -1 --format='%cd %h' --date=format:'%m-%d %H:%M') -join ''
$lines = @(
    '갤럭시에서 이 APK를 받아 설치하세요 (출처 불명 앱 설치 허용 필요).',
    '',
    '- `quant-toss-debug.apk` — 토스증권 연동판 (주력)',
    '- `app-debug.apk` — 기존 앱 (퀀트 대시보드)',
    '',
    "빌드: $stamp",
    ''
)
if ($Notes) { $lines += $Notes } else { $lines += '### 최근 변경'; $lines += (git log -15 --format='- %s') }
$notesFile = Join-Path $out 'release-notes.md'
Write-Utf8 $notesFile (($lines -join "`n") + "`n")
Get-Content -Raw -Encoding UTF8 $notesFile | Out-Host

if ($Test) {
    Step "TEST: draft $TestTag 에 게시 (android-latest 는 건드리지 않는다)"
    if (Test-Release $TestTag) {
        # draft 는 태그가 없어 --cleanup-tag 가 422 를 낼 수 있다 - 지워졌는지로 판단
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & gh release delete $TestTag -R $Repo --yes --cleanup-tag 2>&1 | Out-Host
        $ErrorActionPreference = $old
        if (Test-Release $TestTag) { throw "old test release $TestTag 를 못 지웠다" }
    }
    & gh release create $TestTag -R $Repo --draft --prerelease --target $sha --title "[TEST] $Title" --notes-file $notesFile $files
    Check-Exit 'gh release create (test)'
    Write-Host "draft 생성: $TestTag  (지우기: gh release delete $TestTag -R $Repo --yes --cleanup-tag)"
    exit 0
}

Step "Publish to rolling release $PubTag"
Assert-VerifyName $PubTag
if (Test-Release $PubTag) {
    # softprops: 기존 릴리스는 태그를 그대로 두고 제목/본문/prerelease 를 갱신, 같은 이름 자산은 교체
    & gh release edit $PubTag -R $Repo --title $Title --prerelease --notes-file $notesFile @relFlags
    Check-Exit 'gh release edit'
    & gh release upload $PubTag -R $Repo --clobber $files
    Check-Exit 'gh release upload'
} else {
    & gh release create $PubTag -R $Repo --prerelease --target $PubTarget --title $Title --notes-file $notesFile @relFlags $files
    Check-Exit 'gh release create'
}
Write-Host "게시 완료: https://github.com/$Repo/releases/tag/$PubTag"
