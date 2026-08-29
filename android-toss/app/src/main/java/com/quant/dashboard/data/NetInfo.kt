package com.quant.dashboard.data

import java.net.HttpURLConnection
import java.net.URL

/**
 * 현재 공인 IP 조회 — 토스 Open API 허용 IP 목록에 등록할 값을 앱에서 바로 확인하기 위한 것.
 *
 * 같은 공유기에 붙어 있으면 폰·PC가 같은 공인 IP를 쓴다. 반대로 WiFi 가 끊겨 LTE 로 넘어가면
 * IP 가 바뀌므로 403(access_denied)이 난다 — 그때 이 값을 다시 등록하면 된다.
 */
object NetInfo {
    private val ENDPOINTS = listOf(
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://ifconfig.me/ip",
    )

    /** 실패 시 null. IO 디스패처에서 호출. */
    fun publicIp(): String? {
        for (u in ENDPOINTS) {
            try {
                val conn = (URL(u).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    setRequestProperty("User-Agent", "curl/8")
                    connectTimeout = 5000
                    readTimeout = 5000
                }
                try {
                    if (conn.responseCode != 200) continue
                    val ip = conn.inputStream.bufferedReader().use { it.readText() }.trim()
                    // 아주 느슨한 형태 검증 (IPv4/IPv6 모두 허용)
                    if (ip.isNotBlank() && ip.length <= 45 && (ip.contains('.') || ip.contains(':'))) return ip
                } finally {
                    conn.disconnect()
                }
            } catch (e: Exception) {
                // 다음 엔드포인트 시도
            }
        }
        return null
    }
}
