package com.quant.dashboard.data

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * GitHub Gist 읽기 — 데스크톱(Streamlit)이 올린 매매기록/종목 데이터를 가져온다.
 * 파일명: quant_trade_history.json / quant_target_tickers.json / quant_settings.json
 */
object Gist {
    const val FILE_TRADES = "quant_trade_history.json"
    const val FILE_TICKERS = "quant_target_tickers.json"

    /** Gist 내 파일 content 반환. 실패 시 null. IO 디스패처에서 호출. */
    fun fetchFile(token: String, gistId: String, filename: String): String? {
        if (token.isBlank() || gistId.isBlank()) return null
        try {
            val url = URL("https://api.github.com/gists/$gistId")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                setRequestProperty("Authorization", "token $token")
                setRequestProperty("Accept", "application/vnd.github+json")
                connectTimeout = 8000
                readTimeout = 8000
            }
            if (conn.responseCode != 200) { conn.disconnect(); return null }
            val text = conn.inputStream.bufferedReader().use { it.readText() }
            val files = JSONObject(text).optJSONObject("files") ?: return null
            if (!files.has(filename)) return null
            return files.getJSONObject(filename).optString("content").ifBlank { null }
        } catch (e: Exception) {
            return null
        }
    }
}
