package com.quant.dashboard.data

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Yahoo Finance chart API 직접 호출 (키 불필요). 종가 시계열만 사용.
 * query1/query2 두 호스트 fallback. org.json + HttpURLConnection (외부 의존 0).
 */
/** 봉 1개 (일봉). */
data class Candle(val t: Long, val open: Double, val high: Double, val low: Double, val close: Double)

object Yahoo {
    private val HOSTS = listOf("query1.finance.yahoo.com", "query2.finance.yahoo.com")

    /** (epochSec, close) 리스트. 실패 시 빈 리스트. */
    fun closes(symbol: String, range: String = "2y", interval: String = "1d"): List<Pair<Long, Double>> =
        ohlc(symbol, range, interval).map { Pair(it.t, it.close) }

    /** OHLC 봉 리스트. 실패 시 빈 리스트. 한국 종목은 .KS→.KQ 순으로 시도. */
    fun ohlc(symbol: String, range: String = "2y", interval: String = "1d"): List<Candle> {
        for (sym in symbolCandidates(symbol)) {
            val out = fetch(sym, range, interval)
            if (out.isNotEmpty()) return out
        }
        return emptyList()
    }

    private fun fetch(yahooSym: String, range: String, interval: String): List<Candle> {
        for (host in HOSTS) {
            try {
                val url = URL(
                    "https://$host/v8/finance/chart/" +
                        URLEncoder.encode(yahooSym, "UTF-8") +
                        "?range=$range&interval=$interval"
                )
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    setRequestProperty("User-Agent", "Mozilla/5.0")
                    connectTimeout = 8000
                    readTimeout = 8000
                }
                if (conn.responseCode != 200) { conn.disconnect(); continue }
                val text = conn.inputStream.bufferedReader().use { it.readText() }
                val result = JSONObject(text)
                    .getJSONObject("chart").getJSONArray("result").getJSONObject(0)
                val ts = result.getJSONArray("timestamp")
                val quote = result.getJSONObject("indicators")
                    .getJSONArray("quote").getJSONObject(0)
                val o = quote.getJSONArray("open")
                val h = quote.getJSONArray("high")
                val l = quote.getJSONArray("low")
                val c = quote.getJSONArray("close")
                val out = ArrayList<Candle>(ts.length())
                for (i in 0 until ts.length()) {
                    if (c.isNull(i) || o.isNull(i) || h.isNull(i) || l.isNull(i)) continue
                    out.add(Candle(ts.getLong(i), o.getDouble(i), h.getDouble(i), l.getDouble(i), c.getDouble(i)))
                }
                if (out.isNotEmpty()) return out
            } catch (e: Exception) {
                // 다음 호스트 시도
            }
        }
        return emptyList()
    }

    /** 한국 6자리 코드는 KOSPI(.KS)/KOSDAQ(.KQ) 둘 다 시도. 그 외는 그대로. */
    private fun symbolCandidates(symbol: String): List<String> =
        if (symbol.length == 6 && symbol.all { it.isDigit() }) listOf("$symbol.KS", "$symbol.KQ")
        else listOf(symbol)
}
