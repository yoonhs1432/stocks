package com.quant.dashboard.data

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKeys

/**
 * 증권사 Open API 자격증명 — 기기 내부 **암호화** 저장(EncryptedSharedPreferences).
 *
 * ⚠️ 절대 지켜야 할 것 (저장소·APK가 모두 공개이므로):
 *  1. 키를 코드·리소스·gradle에 넣지 말 것 — 사용자가 설정 탭에서 직접 입력한다.
 *  2. `Store.settings()`(filesDir/settings.json)에 두지 말 것 — 그 경로는 Gist로
 *     동기화되므로 원격 저장소에 자격증명이 올라간다.
 *  3. 암호화 저장에 실패하면 **평문으로 물러나지 않고** 연동을 비활성화한다.
 */
object BrokerCreds {
    private const val FILE = "broker_creds"
    private const val K_KEY = "app_key"
    private const val K_SECRET = "app_secret"
    private const val K_ACCOUNT = "account_no"

    private var prefs: SharedPreferences? = null

    /** MainActivity.onCreate에서 1회 호출. 실패 시 prefs=null → 연동 기능 비활성. */
    fun init(ctx: Context) {
        prefs = try {
            val alias = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
            EncryptedSharedPreferences.create(
                FILE, alias, ctx,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
            )
        } catch (e: Exception) {
            null
        }
    }

    /** 암호화 저장을 쓸 수 있는지 (false면 연동 UI를 막는다). */
    fun available(): Boolean = prefs != null

    fun appKey(): String = prefs?.getString(K_KEY, "").orEmpty()
    fun appSecret(): String = prefs?.getString(K_SECRET, "").orEmpty()
    fun accountNo(): String = prefs?.getString(K_ACCOUNT, "").orEmpty()

    /** 3개 값이 모두 있어야 연동된 것으로 본다. */
    fun isConfigured(): Boolean =
        appKey().isNotBlank() && appSecret().isNotBlank() && accountNo().isNotBlank()

    fun save(appKey: String, appSecret: String, accountNo: String) {
        prefs?.edit()
            ?.putString(K_KEY, appKey.trim())
            ?.putString(K_SECRET, appSecret.trim())
            ?.putString(K_ACCOUNT, accountNo.trim())
            ?.apply()
    }

    fun clear() {
        prefs?.edit()?.clear()?.apply()
    }

    /** 로그·화면 표시용 마스킹 (앞 4자만). 원문은 어디에도 출력하지 말 것. */
    fun maskedKey(): String {
        val k = appKey()
        return if (k.length <= 4) "•".repeat(k.length) else k.take(4) + "•".repeat(minOf(k.length - 4, 12))
    }
}
