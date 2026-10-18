/**
 * 어느 빌드를 깔았는지 앱 안에서 보이게 하는 도장 — `09-24 ba72f9f` 식.
 *
 * 사이드로드라 스토어 버전 표시가 없어서, 화면에 옛 버전 증상이 나와도 **APK 가 옛것인지
 * 코드가 잘못된 건지** 구분할 방법이 없었다. git 이 없으면 "dev".
 */
val buildStamp: String = runCatching {
    ProcessBuilder("git", "log", "-1", "--format=%cd %h", "--date=format:%m-%d")
        .directory(rootDir).start().inputStream.bufferedReader().readText().trim()
}.getOrNull()?.takeIf { it.isNotBlank() } ?: "dev"

/**
 * 자동 업데이트가 '더 새 빌드인가' 를 가리는 번호 — 커밋 시각을 2024-01-01 부터의 **분**으로.
 * 같은 커밋은 어디서 빌드해도 같은 번호, 새 커밋은 항상 더 크다. git 이 없으면 1(설치 안 됨).
 */
val commitMinutes: Int = runCatching {
    val ct = ProcessBuilder("git", "log", "-1", "--format=%ct")
        .directory(rootDir).start().inputStream.bufferedReader().readText().trim().toLong()
    ((ct - 1_704_067_200L) / 60).toInt()
}.getOrNull() ?: 1

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.quant.dashboard"
    compileSdk = 34

    defaultConfig {
        // 기존 앱과 나란히 설치되도록 applicationId만 분리 (namespace는 동일하게 유지)
        applicationId = "com.quant.dashboard.toss"
        minSdk = 26
        targetSdk = 34
        versionCode = commitMinutes
        versionName = buildStamp
        buildConfigField("String", "BUILD_STAMP", "\"$buildStamp\"")
    }

    // 고정 서명키 — 매 빌드 동일 키로 서명해야 폰에서 '덮어쓰기 설치'가 됨
    // (개인 사이드로드용. 키가 매번 바뀌면 '패키지 충돌'로 삭제 후 재설치해야 함)
    signingConfigs {
        create("shared") {
            storeFile = file("quant.keystore")
            storePassword = "quant123"
            keyAlias = "quant"
            keyPassword = "quant123"
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("shared")
        }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("shared")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")

    val composeBom = platform("androidx.compose:compose-bom:2024.09.02")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")

    // 토스증권 API 자격증명 암호화 저장 (EncryptedSharedPreferences)
    implementation("androidx.security:security-crypto:1.0.0")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // 자동 업데이트 — 앱을 안 열어도 한 시간마다 새 빌드를 확인한다
    implementation("androidx.work:work-runtime-ktx:2.9.1")
}

/**
 * 릴리스에 같이 올리는 `version.json` — 폰이 이것만 받아 보고 새 APK 를 받을지 정한다.
 * 노트는 `-PreleaseNotes=...` 로 받는다(비면 빈 문자열).
 */
tasks.register("versionJson") {
    val out = layout.buildDirectory.file("outputs/version.json")
    val notes = (project.findProperty("releaseNotes") as String?) ?: ""
    outputs.file(out)
    doLast {
        out.get().asFile.writeText(groovy.json.JsonOutput.toJson(
            mapOf("versionCode" to commitMinutes, "stamp" to buildStamp, "notes" to notes)))
    }
}
