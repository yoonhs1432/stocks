#!/usr/bin/env bash
# 로컬 컴파일 확인 — 푸시 전에 반드시 돌린다. (PC 러너와 같은 도구를 쓴다)
#   bash improve/build.sh          → android-toss 디버그 APK
#   bash improve/build.sh compile  → 코틀린 컴파일만 (빠름)
set -uo pipefail
TC="/c/Users/yoonh/android-build"
export JAVA_HOME="C:\Users\yoonh\android-build\jdk-17.0.20.1+1"
export ANDROID_HOME="C:\Users\yoonh\android-build\sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$TC/jdk-17.0.20.1+1/bin:$PATH"
cd "$(dirname "$0")/../android-toss"
task="assembleDebug"
[ "${1:-}" = "compile" ] && task="compileDebugKotlin"
"$TC/gradle-8.9/bin/gradle" "$task" --console=plain -q 2>&1 | grep -v -E "^$|SDK XML version|warning:|^w: " | tail -40
echo "BUILD_EXIT=${PIPESTATUS[0]}"
