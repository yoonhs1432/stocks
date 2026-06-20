package com.quant.dashboard.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.time.LocalDate

/** 매매 기록 1건. desktop trade_history.json과 동일 필드. */
data class Trade(
    val date: String,   // YYYY-MM-DD
    val type: String,   // buy | sell
    val qty: Int,
    val price: Double,
    val memo: String? = null,
)

/**
 * 폰 로컬 영속화 — filesDir에 JSON 저장 (외부 의존성 없음, org.json).
 * MainActivity.onCreate에서 init(applicationContext) 1회 호출.
 */
object Store {
    private var dir: File? = null

    fun init(ctx: Context) {
        dir = ctx.filesDir
    }

    private fun f(name: String) = File(dir, name)

    // ── 매매 기록 ──
    fun loadTrades(): LinkedHashMap<String, MutableList<Trade>> {
        val out = LinkedHashMap<String, MutableList<Trade>>()
        val file = f("trades.json")
        if (dir == null || !file.exists()) return out
        try {
            val obj = JSONObject(file.readText())
            for (key in obj.keys()) {
                val arr = obj.getJSONArray(key)
                val list = ArrayList<Trade>()
                for (i in 0 until arr.length()) {
                    val t = arr.optJSONObject(i) ?: continue
                    val date = t.optString("date", "").trim()
                    val type = t.optString("type", "").trim().lowercase()
                    if (date.isEmpty() || (type != "buy" && type != "sell")) continue
                    list.add(
                        Trade(
                            date = date,
                            type = type,
                            qty = t.optDouble("qty", 0.0).toInt(),
                            price = t.optDouble("price", 0.0),
                            memo = if (t.has("memo") && !t.isNull("memo")) t.optString("memo").ifBlank { null } else null,
                        )
                    )
                }
                out[key] = list
            }
        } catch (e: Exception) {
            // 손상 시 빈 맵
        }
        return out
    }

    fun saveTrades(map: Map<String, List<Trade>>) {
        val obj = JSONObject()
        for ((k, v) in map) {
            val arr = JSONArray()
            for (t in v) {
                val o = JSONObject()
                o.put("date", t.date); o.put("type", t.type)
                o.put("qty", t.qty); o.put("price", t.price)
                if (!t.memo.isNullOrBlank()) o.put("memo", t.memo)
                arr.put(o)
            }
            obj.put(k, arr)
        }
        try { f("trades.json").writeText(obj.toString(2)) } catch (e: Exception) {}
    }

    fun addTrade(ticker: String, t: Trade) {
        val m = loadTrades()
        m.getOrPut(ticker) { ArrayList() }.add(t)
        saveTrades(m)
    }

    fun deleteTrade(ticker: String, index: Int) {
        val m = loadTrades()
        m[ticker]?.let {
            if (index in it.indices) {
                it.removeAt(index)
                if (it.isEmpty()) m.remove(ticker)
                saveTrades(m)
            }
        }
    }

    /** 기존 매매 기록의 메모만 수정 (app.py 매매 메모 편집 미러). */
    fun updateTradeMemo(ticker: String, index: Int, memo: String?) {
        val m = loadTrades()
        m[ticker]?.let {
            if (index in it.indices) {
                it[index] = it[index].copy(memo = memo?.ifBlank { null })
                saveTrades(m)
            }
        }
    }

    // ── 종목 리스트 ──
    fun loadTickers(): MutableList<String> {
        val file = f("tickers.json")
        if (dir != null && file.exists()) {
            try {
                val arr = JSONObject(file.readText()).getJSONArray("tickers")
                val out = ArrayList<String>()
                for (i in 0 until arr.length()) out.add(arr.getString(i))
                if (out.isNotEmpty()) return out
            } catch (e: Exception) {}
        }
        return Tickers.DEFAULT.toMutableList()
    }

    fun saveTickers(list: List<String>) {
        val obj = JSONObject()
        obj.put("tickers", JSONArray(list))
        try { f("tickers.json").writeText(obj.toString(2)) } catch (e: Exception) {}
    }

    fun addTicker(t: String) {
        val u = t.trim().uppercase()
        if (u.isEmpty()) return
        val l = loadTickers()
        if (u !in l) { l.add(u); saveTickers(l) }
    }

    fun removeTicker(t: String) {
        val l = loadTickers()
        if (l.size > MIN_TICKERS) { l.remove(t); saveTickers(l) }
    }

    const val MIN_TICKERS = 3   // app.py MIN_TICKERS 미러

    // ── 종목 표시명 override (app.py 사용자 이름 override 미러) ──
    fun nameOverrides(): Map<String, String> {
        val o = settings().optJSONObject("names") ?: return emptyMap()
        val out = LinkedHashMap<String, String>()
        for (k in o.keys()) o.optString(k).takeIf { it.isNotBlank() }?.let { out[k] = it }
        return out
    }

    fun setNameOverride(ticker: String, name: String?) {
        val s = settings()
        val o = s.optJSONObject("names") ?: JSONObject()
        if (name.isNullOrBlank()) o.remove(ticker) else o.put(ticker, name.trim())
        saveSettings(s.put("names", o))
    }

    // ── 개별 종목 표식 (개별/ETF 필터용, app.py individual_tickers 미러) ──
    fun individualTickers(): Set<String> {
        val arr = settings().optJSONArray("individual") ?: return emptySet()
        val out = HashSet<String>()
        for (i in 0 until arr.length()) out.add(arr.getString(i))
        return out
    }

    /** 한국 6자리 종목은 기본적으로 개별로 간주 + 사용자 지정. */
    fun isIndividual(ticker: String): Boolean =
        ticker in individualTickers() || (ticker.length == 6 && ticker.all { it.isDigit() })

    fun setIndividual(ticker: String, individual: Boolean) {
        val cur = individualTickers().toMutableSet()
        if (individual) cur.add(ticker) else cur.remove(ticker)
        saveSettings(settings().put("individual", JSONArray(cur.toList())))
    }

    /** 봉 기준: "1d"(일봉) / "1wk"(주봉). app.py 봉 기준 미러. */
    fun candleInterval(): String = settings().optString("interval", "1d")
    fun setCandleInterval(v: String) { saveSettings(settings().put("interval", v)) }

    /** 차트 조회기간(개월): 1/2/4/12. app.py 차트 조회기간 미러. 기본 2개월. */
    fun chartMonths(): Int = settings().optInt("chart_months", 2)
    fun setChartMonths(v: Int) { saveSettings(settings().put("chart_months", v)) }

    // ── 기준일(As-of) 시뮬레이션 ──
    /** 설정된 기준일 (ISO yyyy-MM-dd). 미설정/공백이면 null. */
    fun asofDate(): String? = settings().optString("asof", "").ifBlank { null }

    fun setAsofDate(s: String?) { saveSettings(settings().put("asof", s?.trim() ?: "")) }

    private fun asofCutoffSec(): Long? {
        val d = asofDate() ?: return null
        return try { LocalDate.parse(d).toEpochDay() * 86400L + 86_399L } catch (e: Exception) { null }
    }

    /** 기준일 이후 데이터를 잘라낸 시계열. 기준일 미설정 시 원본 그대로. */
    fun sliceAsof(series: List<Pair<Long, Double>>): List<Pair<Long, Double>> {
        val cut = asofCutoffSec() ?: return series
        return series.filter { it.first <= cut }
    }

    fun sliceAsofCandles(series: List<Candle>): List<Candle> {
        val cut = asofCutoffSec() ?: return series
        return series.filter { it.t <= cut }
    }

    // ── 설정 (시드, 분석 기간) ──
    private fun settings(): JSONObject {
        val file = f("settings.json")
        if (dir != null && file.exists()) {
            try { return JSONObject(file.readText()) } catch (e: Exception) {}
        }
        return JSONObject()
    }

    private fun saveSettings(o: JSONObject) {
        try { f("settings.json").writeText(o.toString(2)) } catch (e: Exception) {}
    }

    fun seedUsd(): Double = settings().optDouble("seed", 20_000.0)
    fun setSeedUsd(v: Double) { saveSettings(settings().put("seed", v)) }

    /** Yahoo range 문자열: "6mo" / "1y" / "2y". 기본 2y. */
    fun lookbackRange(): String = settings().optString("range", "2y")
    fun setLookbackRange(r: String) { saveSettings(settings().put("range", r)) }

    // ── Gist 연동 ──
    fun gistToken(): String = settings().optString("gist_token", "")
    fun gistId(): String = settings().optString("gist_id", "")
    fun setGist(token: String, id: String) {
        saveSettings(settings().put("gist_token", token.trim()).put("gist_id", id.trim()))
    }

    /** 데스크톱 quant_trade_history.json 포맷 → 로컬 매매기록 덮어쓰기. 종목 수 반환.
     *  데스크톱은 누락 필드를 r.get('qty',0)처럼 관대하게 다루므로 여기서도 opt*로 안전 파싱. */
    fun saveTradesFromJson(text: String): Int {
        val obj = JSONObject(text)
        val map = LinkedHashMap<String, MutableList<Trade>>()
        for (key in obj.keys()) {
            val arr = obj.optJSONArray(key) ?: continue
            val list = ArrayList<Trade>()
            for (i in 0 until arr.length()) {
                val t = arr.optJSONObject(i) ?: continue
                val date = t.optString("date", "").trim()
                val type = t.optString("type", "").trim().lowercase()
                if (date.isEmpty() || (type != "buy" && type != "sell")) continue
                // qty: 정수/실수/문자열 무엇이든 관대하게
                val qty = when {
                    t.has("qty") && !t.isNull("qty") -> t.optDouble("qty", 0.0).toInt()
                    t.has("quantity") && !t.isNull("quantity") -> t.optDouble("quantity", 0.0).toInt()
                    t.has("shares") && !t.isNull("shares") -> t.optDouble("shares", 0.0).toInt()
                    else -> 0
                }
                val price = t.optDouble("price", 0.0)
                val memo = if (t.has("memo") && !t.isNull("memo")) t.optString("memo").ifBlank { null } else null
                list.add(Trade(date, type, qty, price, memo))
            }
            if (list.isNotEmpty()) map[key] = list
        }
        saveTrades(map)
        return map.size
    }

    /** 데스크톱 quant_target_tickers.json 포맷 → 로컬 종목 리스트 덮어쓰기. */
    fun saveTickersFromJson(text: String): Int {
        val obj = JSONObject(text)
        val arr = obj.optJSONArray("tickers") ?: return 0
        val out = ArrayList<String>()
        for (i in 0 until arr.length()) {
            val s = arr.optString(i, "").trim().uppercase()
            if (s.isNotEmpty()) out.add(s)
        }
        if (out.isNotEmpty()) { saveTickers(out); return out.size }
        return 0
    }
}
