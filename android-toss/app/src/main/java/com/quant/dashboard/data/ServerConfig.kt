package com.quant.dashboard.data

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKeys

/**
 * 집 PC 서버 접속 정보 — 주소와 접속 암호. **기기 내부 암호화 저장.**
 *
 * 예전에는 여기에 증권사 앱키·시크릿이 있었다. 이제 그건 PC 에만 있고, 폰에는
 * 서버 주소와 암호만 둔다. 폰을 잃어버려도 새어 나가는 게 훨씬 적다.
 *
 * 암호화 저장에 실패하면 **평문으로 물러나지 않고** 연결을 비활성화한다.
 */
object ServerConfig {
    private const val FILE = "server_config"
    private const val K_URL = "url"
    private const val K_TOKEN = "token"
    private const val K_ACCOUNT_NO = "account_no"

    private var prefs: SharedPreferences? = null

    /** MainActivity.onCreate 에서 1회. 실패하면 prefs=null → 연결 UI 를 막는다. */
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

    fun available(): Boolean = prefs != null

    /** `https://…ts.net` 처럼. 끝의 `/` 는 떼어 저장한다. */
    fun url(): String = prefs?.getString(K_URL, "").orEmpty()

    fun token(): String = prefs?.getString(K_TOKEN, "").orEmpty()

    fun accountNo(): String = prefs?.getString(K_ACCOUNT_NO, "").orEmpty()

    fun isSet(): Boolean = url().isNotBlank() && token().isNotBlank()

    fun save(url: String, token: String) {
        var u = url.trim().trimEnd('/')
        if (u.isNotBlank() && !u.startsWith("http")) u = "https://$u"
        prefs?.edit()?.putString(K_URL, u)?.putString(K_TOKEN, token.trim())?.apply()
    }

    fun saveAccountNo(no: String) {
        if (no.isBlank() || no == accountNo()) return
        prefs?.edit()?.putString(K_ACCOUNT_NO, no)?.apply()
    }

    fun clear() {
        prefs?.edit()?.clear()?.apply()
    }

    /** 화면 표시용 — 암호는 앞 3자만. 원문은 어디에도 찍지 말 것. */
    fun maskedToken(): String {
        val t = token()
        return if (t.length <= 3) "•".repeat(t.length) else t.take(3) + "•".repeat(minOf(t.length - 3, 12))
    }

    /** 계좌번호 마스킹 (뒤 4자리만). */
    fun maskedAccount(): String {
        val a = accountNo()
        return if (a.length <= 4) a else "•".repeat(a.length - 4) + a.takeLast(4)
    }
}
