package com.quant.dashboard.data

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Yahoo Finance chart API 직접 호출 (키 불필요). 종가 시계열만 사용.
 * query1/query2 두 호스트 fallback. org.json + HttpURLConnection (외부 의존 0).
 */
object Yahoo {
    private val HOSTS = listOf("query1.finance.yahoo.com", "query2.finance.yahoo.com")

    /** (epochSec, close) 리스트. 실패 시 빈 리스트. */
    fun closes(symbol: String, range: String = "2y"): List<Pair<Long, Double>> {
        val yahooSym = toYahooSymbol(symbol)
        for (host in HOSTS) {
            try {
                val url = URL(
                    "https://$host/v8/finance/chart/" +
                        URLEncoder.encode(yahooSym, "UTF-8") +
                        "?range=$range&interval=1d"
                )
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    setRequestProperty("User-Agent", "Mozilla/5.0")
                    connectTimeout = 8000
                    readTimeout = 8000
                }
                val code = conn.responseCode
                if (code != 200) {
                    conn.disconnect()
                    continue
                }
                val text = conn.inputStream.bufferedReader().use { it.readText() }
                val result = JSONObject(text)
                    .getJSONObject("chart")
                    .getJSONArray("result")
                    .getJSONObject(0)
                val ts = result.getJSONArray("timestamp")
                val quote = result.getJSONObject("indicators")
                    .getJSONArray("quote").getJSONObject(0)
                val closeArr = quote.getJSONArray("close")
                val out = ArrayList<Pair<Long, Double>>(ts.length())
                for (i in 0 until ts.length()) {
                    if (closeArr.isNull(i)) continue
                    out.add(Pair(ts.getLong(i), closeArr.getDouble(i)))
                }
                if (out.isNotEmpty()) return out
            } catch (e: Exception) {
                // 다음 호스트 시도
            }
        }
        return emptyList()
    }

    /** 한국 6자리 코드는 Yahoo에서 .KS 접미사 필요 (KOSDAQ는 추후 .KQ 보정). */
    private fun toYahooSymbol(symbol: String): String {
        return if (symbol.length == 6 && symbol.all { it.isDigit() }) "$symbol.KS" else symbol
    }
}
