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
    private const val K_URL2 = "url2"
    private const val K_LAST_OK = "last_ok"
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

    /**
     * 보조 주소 — 집 랜(`http://192.168.x.x:8000`) 같은 **두 번째 길**.
     *
     * 터널(`…ts.net`)은 PC 가 자거나 터널이 내려가면 **이름조차 안 풀린다**. 집 와이파이에
     * 있을 때는 랜 주소로 바로 갈 수 있으므로, 한쪽이 막히면 다른 쪽으로 넘어간다.
     */
    fun altUrl(): String = prefs?.getString(K_URL2, "").orEmpty()

    /** 마지막에 실제로 응답한 주소 — 다음부터 이쪽을 먼저 부른다(헛걸음 15초 절약). */
    private fun lastOk(): String = prefs?.getString(K_LAST_OK, "").orEmpty()

    /** 성공한 주소를 기억한다. [Server] 가 부른다. */
    fun noteOk(base: String) {
        if (base.isBlank() || base == lastOk()) return
        prefs?.edit()?.putString(K_LAST_OK, base)?.apply()
    }

    /** 시도할 주소들 — **최근에 됐던 것부터**. 중복·빈 값은 뺀다. */
    fun bases(): List<String> {
        val list = listOf(url(), altUrl()).map { it.trimEnd('/') }.filter { it.isNotBlank() }.distinct()
        val ok = lastOk()
        return if (ok.isNotBlank() && ok in list) listOf(ok) + list.filter { it != ok } else list
    }

    fun token(): String = prefs?.getString(K_TOKEN, "").orEmpty()

    fun accountNo(): String = prefs?.getString(K_ACCOUNT_NO, "").orEmpty()

    fun isSet(): Boolean = bases().isNotEmpty() && token().isNotBlank()

    fun save(url: String, token: String) {
        prefs?.edit()?.putString(K_URL, norm(url))?.putString(K_TOKEN, token.trim())?.apply()
    }

    /** 보조 주소만 따로 저장. 비우면 지운다. */
    fun saveAlt(url: String) {
        prefs?.edit()?.putString(K_URL2, norm(url))?.apply()
    }

    /**
     * 주소 다듬기. 사설 IP(`192.168.…`)에는 https 인증서가 없으므로 **http** 를 붙인다 —
     * https 를 붙이면 인증서 오류로 연결이 막힌다.
     */
    private fun norm(url: String): String {
        val u = url.trim().trimEnd('/')
        if (u.isBlank() || u.startsWith("http")) return u
        val lan = u.startsWith("192.168.") || u.startsWith("10.") || u.startsWith("127.") ||
            u.startsWith("localhost") || Regex("^172\\.(1[6-9]|2\\d|3[01])\\.").containsMatchIn(u)
        return (if (lan) "http://" else "https://") + u
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
